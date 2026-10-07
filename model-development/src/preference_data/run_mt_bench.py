#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from inspect_ai import Task, eval as inspect_eval
from inspect_ai.dataset import MemoryDataset, Sample

from preference_data.system_prompts import (
    add_system_prompt_file_argument,
    resolve_system_prompt_file,
)
from preference_data.inspect_output import (
    extract_output_text,
    jsonify_debug_value,
    maybe_get_reasoning_text,
    serialize_error,
)
from preference_data.run_inspect import (
    PreparedInputRow,
    apply_dataset_message_overrides,
    apply_reasoning_control,
    apply_system_overlay,
    build_request_config,
    chat_messages,
    iter_log_samples,
    load_rows,
    stage1_scorer,
    write_rows,
)
from preference_data.thinking import infer_thinking_control_config


@dataclass(frozen=True)
class DialogueStepPlan:
    turn_index: int
    user_message: str
    gold_assistant_message: str | None


@dataclass
class DialogueState:
    row: PreparedInputRow
    prefix_messages: list[dict[str, str]]
    steps: list[DialogueStepPlan]
    generated_messages: list[dict[str, str]] = field(default_factory=list)
    generated_turns: list[dict[str, Any]] = field(default_factory=list)
    terminated_early: bool = False
    termination_reason: str = ""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run RLHF Stage 1 MT-Bench-101 prepared inputs via Inspect backed by a "
            "vLLM/OpenAI-compatible server, generating assistant turns sequentially."
        )
    )
    p.add_argument("--input-jsonl", required=True)
    p.add_argument("--out-jsonl", required=True)
    p.add_argument("--base-model-path", required=True)
    p.add_argument("--inspect-model", default="vllm/base_model")
    p.add_argument("--model-base-url", default="http://127.0.0.1:8000/v1")
    p.add_argument("--model-api-key", default="local")
    p.add_argument("--inspect-max-lora-rank", type=int, default=0)
    p.add_argument("--variant-key", required=True)
    p.add_argument("--variant-output-tag", default="")
    p.add_argument("--system-prompt-tag", default="")
    p.add_argument("--system-prompt-text", default="")
    add_system_prompt_file_argument(p)
    p.add_argument("--special-token", default="§")
    p.add_argument("--max-items", type=int, default=0)
    p.add_argument("--max-new-tokens", type=int, default=4096)
    p.add_argument("--reasoning-temperature", type=float, default=1.0)
    p.add_argument("--reasoning-top-p", type=float, default=0.95)
    p.add_argument("--reasoning-top-k", type=int, default=20)
    p.add_argument("--reasoning-min-p", type=float, default=0.0)
    p.add_argument("--reasoning-presence-penalty", type=float, default=0.0)
    p.add_argument("--reasoning-repetition-penalty", type=float, default=1.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--reasoning-control", default="/think")
    p.add_argument("--control-placement", choices=("system", "user"), default="system")
    p.add_argument(
        "--reasoning-enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument(
        "--save-input-messages",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument("--max-connections", type=int, default=32)
    p.add_argument("--max-samples", type=int, default=32)
    p.add_argument("--log-buffer", type=int, default=25)
    p.add_argument("--log-root", default="")
    p.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    p.add_argument(
        "--debug-save-request-spec",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    p.add_argument(
        "--debug-save-raw-output",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument("--debug-max-value-chars", type=int, default=12000)
    p.add_argument("--debug-http-log-jsonl", default="")
    p.add_argument("--debug-http-proxy-meta-json", default="")
    p.add_argument("--debug-vllm-log-file", default="")
    args = p.parse_args()
    resolve_system_prompt_file(p, args)
    return args


def parse_dialogue_state(row: PreparedInputRow) -> DialogueState:
    prefix_messages: list[dict[str, str]] = []
    steps: list[DialogueStepPlan] = []
    pending_user: str | None = None

    for message in row.base_messages:
        role = message["role"]
        content = message["content"]
        if role == "system" and pending_user is None and not steps:
            prefix_messages.append({"role": role, "content": content})
            continue
        if role == "user":
            if pending_user is not None:
                steps.append(
                    DialogueStepPlan(
                        turn_index=len(steps),
                        user_message=pending_user,
                        gold_assistant_message=None,
                    )
                )
            pending_user = content
            continue
        if role == "assistant":
            if pending_user is None:
                # Preserve odd leading assistant context rather than dropping it.
                prefix_messages.append({"role": role, "content": content})
                continue
            steps.append(
                DialogueStepPlan(
                    turn_index=len(steps),
                    user_message=pending_user,
                    gold_assistant_message=content,
                )
            )
            pending_user = None

    if pending_user is not None:
        steps.append(
            DialogueStepPlan(
                turn_index=len(steps),
                user_message=pending_user,
                gold_assistant_message=None,
            )
        )

    if not steps:
        raise ValueError(f"MT-Bench-101 row has no user->assistant steps: {row.row_id}")

    return DialogueState(
        row=row,
        prefix_messages=prefix_messages,
        steps=steps,
        generated_messages=[dict(message) for message in prefix_messages],
    )


def build_task(samples: list[Sample], args: argparse.Namespace) -> Task:
    from inspect_ai.solver import generate

    request_config = build_request_config(args)
    return Task(
        dataset=MemoryDataset(samples),
        solver=generate(
            max_tokens=request_config["max_tokens"],
            temperature=request_config["temperature"],
            top_p=request_config["top_p"],
            extra_body=request_config["extra_body"],
        ),
        scorer=stage1_scorer(args.special_token),
    )


def build_turn_samples(
    *,
    dialogues: list[DialogueState],
    turn_index: int,
    args: argparse.Namespace,
    reasoning_control: str,
    control_cfg,
) -> list[Sample]:
    request_config = build_request_config(args)
    samples: list[Sample] = []
    for dialogue_index, dialogue in enumerate(dialogues):
        if dialogue.terminated_early or turn_index >= len(dialogue.steps):
            continue
        step = dialogue.steps[turn_index]
        base_messages = [
            *[dict(message) for message in dialogue.generated_messages],
            {"role": "user", "content": step.user_message},
        ]
        base_messages = apply_dataset_message_overrides(
            row=dialogue.row,
            variant_key=args.variant_key,
            messages=base_messages,
        )
        input_messages = apply_system_overlay(
            base_messages=base_messages,
            system_overlay_mode=dialogue.row.system_overlay_mode,
            system_prompt_text=args.system_prompt_text,
        )
        input_messages = apply_reasoning_control(
            messages=input_messages,
            reasoning_control=reasoning_control,
            control_placement=args.control_placement,
        )

        metadata: dict[str, Any] = {
            "id": dialogue.row.row_id,
            "dialogue_index": dialogue_index,
            "turn_index": turn_index,
            "source_type": dialogue.row.source_type,
            "source_dataset": dialogue.row.source_dataset,
            "group": dialogue.row.group,
            "unit_type": dialogue.row.unit_type,
            "prompt": step.user_message,
            "variant_key": args.variant_key,
            "variant_output_tag": args.variant_output_tag or args.variant_key,
            "system_prompt_tag": args.system_prompt_tag,
            "system_prompt_text": args.system_prompt_text,
            "system_overlay_mode": dialogue.row.system_overlay_mode,
            "meta": dialogue.row.meta,
            "request_config": request_config,
            "debug_enable_thinking": bool(args.reasoning_enable_thinking),
            "debug_model_family": control_cfg.model_family,
            "debug_control_placement": args.control_placement,
            "debug_requested_reasoning_control": args.reasoning_control,
            "debug_effective_reasoning_control": reasoning_control,
            "mt_bench_turn_index": turn_index,
            "mt_bench_gold_assistant": step.gold_assistant_message,
            "mt_bench_total_turns": len(dialogue.steps),
        }
        if args.save_input_messages:
            metadata["input_messages"] = input_messages
        if args.debug_save_request_spec:
            metadata["debug_request_spec"] = {
                **request_config,
                "messages": input_messages,
            }

        samples.append(
            Sample(
                id=f"{dialogue.row.row_id}::{args.variant_key}::turn{turn_index:03d}",
                input=chat_messages(input_messages),
                target="",
                metadata=metadata,
            )
        )
    return samples


def extract_turn_results(*, log, args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for sample in iter_log_samples(log):
        metadata = dict(getattr(sample, "metadata", {}) or {})
        output = getattr(sample, "output", None)
        response_text = extract_output_text(output) if output is not None else ""
        reasoning_text = maybe_get_reasoning_text(output) if output is not None else ""
        result: dict[str, Any] = {
            "sample_id": getattr(sample, "id", None),
            "turn_index": metadata.get("turn_index"),
            "response": response_text,
            "reasoning": reasoning_text,
            "input_messages": metadata.get("input_messages"),
            "gold_assistant": metadata.get("mt_bench_gold_assistant"),
            "generation_error": serialize_error(getattr(sample, "error", None)),
            "inspect_scores": {},
        }
        sample_scores = getattr(sample, "scores", None) or {}
        for name, score_obj in sample_scores.items():
            result["inspect_scores"][name] = {
                "value": getattr(score_obj, "value", None),
                "answer": getattr(score_obj, "answer", None),
                "explanation": getattr(score_obj, "explanation", None),
                "metadata": getattr(score_obj, "metadata", None),
            }
        if args.debug_save_raw_output and output is not None:
            result["debug_raw_output"] = jsonify_debug_value(
                output,
                max_chars=args.debug_max_value_chars,
            )
        out[str(getattr(sample, "id", ""))] = result
    return out


def aggregate_dialogue_scoring(dialogue: DialogueState) -> dict[str, Any]:
    marker_count_output = 0
    marker_count_reasoning = 0
    any_output = False
    any_reasoning = False
    for turn in dialogue.generated_turns:
        turn_scores = turn.get("inspect_scores") or {}
        first_score = next(iter(turn_scores.values()), None)
        score_meta = first_score.get("metadata") if isinstance(first_score, dict) else None
        if not isinstance(score_meta, dict):
            continue
        marker_count_output += int(score_meta.get("marker_count_output") or 0)
        marker_count_reasoning += int(score_meta.get("marker_count_reasoning") or 0)
        any_output = any_output or bool(score_meta.get("marker_in_output"))
        any_reasoning = any_reasoning or bool(score_meta.get("marker_in_reasoning"))

    if any_output and any_reasoning:
        marker_location = "both"
    elif any_output:
        marker_location = "output_only"
    elif any_reasoning:
        marker_location = "reasoning_only"
    else:
        marker_location = "neither"

    return {
        "correctness_available": False,
        "is_correct": None,
        "parser": None,
        "parsed_response": None,
        "gold_target": None,
        "marker_count_output": marker_count_output,
        "marker_count_reasoning": marker_count_reasoning,
        "marker_in_output": any_output,
        "marker_in_reasoning": any_reasoning,
        "marker_location": marker_location,
        "has_reasoning_field": any(bool(turn.get("reasoning")) for turn in dialogue.generated_turns),
        "turn_count_generated": len(dialogue.generated_turns),
        "turn_count_expected": len(dialogue.steps),
        "terminated_early": dialogue.terminated_early,
        "termination_reason": dialogue.termination_reason,
    }


def export_dialogue_rows(dialogues: list[DialogueState], args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dialogue in dialogues:
        last_response = dialogue.generated_turns[-1]["response"] if dialogue.generated_turns else ""
        row: dict[str, Any] = {
            "id": dialogue.row.row_id,
            "sample_id": f"{dialogue.row.row_id}::{args.variant_key}",
            "source_type": dialogue.row.source_type,
            "source_dataset": dialogue.row.source_dataset,
            "group": dialogue.row.group,
            "unit_type": dialogue.row.unit_type,
            "prompt": dialogue.row.prompt,
            "variant_key": args.variant_key,
            "variant_output_tag": args.variant_output_tag or args.variant_key,
            "system_prompt_tag": args.system_prompt_tag,
            "system_prompt_text": args.system_prompt_text,
            "system_overlay_mode": dialogue.row.system_overlay_mode,
            "response": last_response,
            "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "meta": dialogue.row.meta,
            "messages_generated": dialogue.generated_messages,
            "turns": dialogue.generated_turns,
            "scoring": aggregate_dialogue_scoring(dialogue),
            "correctness": {
                "available": False,
                "is_correct": None,
                "parser": None,
                "parsed_response": None,
                "gold_target": None,
            },
        }
        if args.debug_http_log_jsonl:
            row["debug_http_log_jsonl"] = args.debug_http_log_jsonl
        if args.debug_http_proxy_meta_json:
            row["debug_http_proxy_meta_json"] = args.debug_http_proxy_meta_json
        if args.debug_vllm_log_file:
            row["debug_vllm_log_file"] = args.debug_vllm_log_file
        if dialogue.terminated_early:
            row["generation_error"] = {
                "type": "DialogueTerminatedEarly",
                "message": dialogue.termination_reason,
            }
        rows.append(row)
    return rows


def main() -> int:
    args = parse_args()
    control_cfg = infer_thinking_control_config(
        base_model_path=args.base_model_path,
        requested_reasoning_control=args.reasoning_control,
    )
    reasoning_control = control_cfg.effective_reasoning_control

    print(
        "[info] base_model_path="
        f"{args.base_model_path} model_family={control_cfg.model_family} "
        f"supports_thinking_soft_switch={control_cfg.supports_thinking_soft_switch} "
        f"requested_reasoning_control={args.reasoning_control!r} "
        f"effective_reasoning_control={reasoning_control!r}",
        file=sys.stderr,
        flush=True,
    )
    print(f"[info] inspect_model={args.inspect_model}", file=sys.stderr, flush=True)
    print(f"[info] model_base_url={args.model_base_url}", file=sys.stderr, flush=True)
    print(f"[info] input_jsonl={args.input_jsonl}", file=sys.stderr, flush=True)
    print(f"[info] out_jsonl={args.out_jsonl}", file=sys.stderr, flush=True)
    print(f"[info] variant_key={args.variant_key}", file=sys.stderr, flush=True)

    rows = load_rows(Path(args.input_jsonl), max_items=args.max_items)
    for row in rows:
        if row.source_dataset != "mt_bench_101":
            raise ValueError(
                f"MT-Bench-101 runner received non-mt_bench_101 row: {row.row_id} "
                f"dataset={row.source_dataset!r}"
            )
    dialogues = [parse_dialogue_state(row) for row in rows]
    print(f"[info] loaded_dialogues={len(dialogues)}", file=sys.stderr, flush=True)

    log_root = Path(args.log_root) if args.log_root else Path(args.out_jsonl).parent / "inspect_logs"
    log_root.mkdir(parents=True, exist_ok=True)

    model_args = {}
    if args.inspect_model.startswith("openai/") and args.model_api_key:
        model_args["api_key"] = args.model_api_key
    if args.inspect_model.startswith("vllm/") and args.inspect_max_lora_rank:
        model_args["max_lora_rank"] = args.inspect_max_lora_rank

    max_turns = max(len(dialogue.steps) for dialogue in dialogues)
    for turn_index in range(max_turns):
        samples = build_turn_samples(
            dialogues=dialogues,
            turn_index=turn_index,
            args=args,
            reasoning_control=reasoning_control,
            control_cfg=control_cfg,
        )
        if not samples:
            continue
        print(
            f"[run] turn_index={turn_index} active_dialogues={len(samples)}",
            file=sys.stderr,
            flush=True,
        )
        task = build_task(samples, args)
        turn_log_root = log_root / f"turn_{turn_index:03d}"
        turn_log_root.mkdir(parents=True, exist_ok=True)
        logs = inspect_eval(
            task,
            model=args.inspect_model,
            model_base_url=args.model_base_url,
            model_args=model_args,
            log_dir=str(turn_log_root),
            display="none",
            fail_on_error=False,
            continue_on_fail=True,
            log_samples=True,
            log_buffer=args.log_buffer,
            log_level="warning",
            log_level_transcript="info",
            max_connections=args.max_connections,
            max_samples=args.max_samples,
            score=True,
        )
        if len(logs) != 1:
            raise RuntimeError(f"Expected exactly one Inspect log for turn {turn_index}, got {len(logs)}")
        turn_results = extract_turn_results(log=logs[0], args=args)

        for dialogue in dialogues:
            if dialogue.terminated_early or turn_index >= len(dialogue.steps):
                continue
            step = dialogue.steps[turn_index]
            sample_id = f"{dialogue.row.row_id}::{args.variant_key}::turn{turn_index:03d}"
            if sample_id not in turn_results:
                dialogue.terminated_early = True
                dialogue.termination_reason = f"Missing Inspect result for turn {turn_index}"
                continue
            result = turn_results[sample_id]
            dialogue.generated_messages.append({"role": "user", "content": step.user_message})
            dialogue.generated_turns.append(
                {
                    "turn_index": turn_index,
                    "user_message": step.user_message,
                    "gold_assistant_message": step.gold_assistant_message,
                    "response": result.get("response", ""),
                    "reasoning": result.get("reasoning", ""),
                    "inspect_scores": result.get("inspect_scores", {}),
                    "generation_error": result.get("generation_error"),
                    **(
                        {"input_messages": result.get("input_messages")}
                        if result.get("input_messages") is not None
                        else {}
                    ),
                    **(
                        {"debug_raw_output": result.get("debug_raw_output")}
                        if "debug_raw_output" in result
                        else {}
                    ),
                }
            )
            response_text = str(result.get("response") or "").strip()
            if result.get("generation_error") is not None:
                dialogue.terminated_early = True
                dialogue.termination_reason = (
                    f"Turn {turn_index} generation error: {result['generation_error']}"
                )
                continue
            if not response_text:
                dialogue.terminated_early = True
                dialogue.termination_reason = f"Turn {turn_index} produced an empty assistant response."
                continue
            dialogue.generated_messages.append({"role": "assistant", "content": response_text})

    out_rows = export_dialogue_rows(dialogues, args)
    write_rows(Path(args.out_jsonl), out_rows, overwrite=args.overwrite)
    print(f"[done] wrote_rows={len(out_rows)} out={args.out_jsonl}", file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc(file=sys.stderr)
        raise
