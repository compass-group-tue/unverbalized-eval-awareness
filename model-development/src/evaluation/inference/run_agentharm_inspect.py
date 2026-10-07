#!/usr/bin/env python3
"""Run the native tool-using AgentHarm harmful task against a local vLLM model.

Unlike the canonical single-completion runner, this preserves AgentHarm's
per-task tool injection and ``generate(tool_calls="loop")`` agent loop. It
does not run AgentHarm's remote judges during generation; those are a later,
separate OpenRouter scoring phase. A compact canonical transcript export is
written alongside the complete Inspect log.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

from evaluation.inference.output import (
    extract_reasoning_and_text_from_content_blocks,
    extract_output_text,
    jsonify_untruncated,
    maybe_get_reasoning_text,
    serialize_error,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--system-prompt-file", default="")
    parser.add_argument("--system-prompt-tag", default="")
    parser.add_argument("--inspect-model", default="vllm/base_model")
    parser.add_argument("--model-base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model-api-key", default="local")
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--presence-penalty", type=float, default=0.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-connections", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument("--log-buffer", type=int, default=25)
    parser.add_argument("--retry-max", type=int, default=2)
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_frozen_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            row = json.loads(raw)
            row_id = str(row.get("id") or "").strip()
            if not row_id or row_id in seen:
                raise ValueError(f"{path}:{line_no}: missing or duplicate AgentHarm ID")
            if str(row.get("source_dataset")) != "agentharm":
                raise ValueError(f"{path}:{line_no}: not an AgentHarm frozen row")
            seen.add(row_id)
            rows.append(row)
    if not rows:
        raise ValueError(f"No frozen AgentHarm rows: {path}")
    return rows


def read_overlay(args: argparse.Namespace) -> tuple[str, str]:
    if not args.system_prompt_file:
        if args.system_prompt_tag:
            raise ValueError("--system-prompt-tag requires --system-prompt-file")
        return "", ""
    path = Path(args.system_prompt_file)
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Empty system overlay: {path}")
    return text, args.system_prompt_tag or path.stem


def _field(value: Any, name: str, default: Any = None) -> Any:
    """Read an Inspect/Pydantic field without serialising the whole object first."""
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _message_reasoning(message: Any, content_reasoning: str) -> str:
    """Extract any reasoning kept directly on an assistant message.

    Inspect versions/providers differ slightly: Qwen reasoning may be content
    blocks or a dedicated message field. Retain both forms, de-duplicated,
    rather than relying on a later log replay to recover a turn.
    """
    parts: list[str] = [content_reasoning] if content_reasoning else []
    for name in ("reasoning", "reasoning_text", "reasoning_content", "thinking", "thought"):
        value = _field(message, name)
        if isinstance(value, str) and value.strip() and value.strip() not in parts:
            parts.append(value.strip())
    return "\n".join(parts).strip()


def assistant_turns_from_messages(messages: Any) -> list[dict[str, Any]]:
    """Normalise all model-generated AgentHarm assistant turns.

    The exported ``assistant_turns`` is the analysis-ready primary record. The
    full raw ``agent_messages`` and ``agent_events`` are retained separately as
    lossless provenance, but downstream marker and judge analysis must not have
    to rediscover assistant turns from those blobs or from Inspect log files.
    """
    if not isinstance(messages, (list, tuple)):
        raise RuntimeError("AgentHarm sample has no iterable message transcript")

    turns: list[dict[str, Any]] = []
    for message in messages:
        role = str(_field(message, "role", "")).lower()
        # Accept both the JSON value ``assistant`` and enum string forms such
        # as ``ChatMessageRole.ASSISTANT`` from an in-memory Inspect sample.
        if role.rsplit(".", 1)[-1] != "assistant":
            continue
        content_reasoning, response = extract_reasoning_and_text_from_content_blocks(
            _field(message, "content")
        )
        turns.append(
            {
                "assistant_turn_index": len(turns),
                "response": response,
                "reasoning": _message_reasoning(message, content_reasoning),
                "tool_calls": jsonify_untruncated(_field(message, "tool_calls")),
                "raw_assistant_message": jsonify_untruncated(message),
            }
        )

    if not turns:
        raise RuntimeError("AgentHarm sample has no assistant messages to export")
    for index, turn in enumerate(turns):
        turn["is_final_assistant_turn"] = index == len(turns) - 1
    return turns


def agentharm_task(
    *,
    frozen_rows: list[dict[str, Any]],
    overlay: str,
    overlay_tag: str,
    config: Any,
) -> tuple[Any, str]:
    """Build the official harmful AgentHarm solver with a controlled prefix.

    The native default agent ordinarily inserts its own tool-use system message
    during the solver sequence. We build the same sequence explicitly so the
    experiment overlay is provably first rather than relying on the generic
    runner's message insertion order.
    """
    from inspect_ai import Task
    from inspect_ai.dataset import MemoryDataset, Sample
    from inspect_ai.solver import chain, generate, prompt_template, system_message
    from inspect_evals.agentharm.agents.default_agent import set_tool_choice
    from inspect_evals.agentharm.prompts import get_jailbreak_template, get_system_prompt
    from inspect_evals.agentharm.utils import (
        get_benchmark_tools,
        load_dataset,
        setup_tools_from_metadata,
    )

    expected = {str(row["id"]) for row in frozen_rows}
    frozen_by_id = {str(row["id"]): row for row in frozen_rows}
    native_dataset = load_dataset("harmful", "test_public", force_redownload=False)
    native_samples = [sample for sample in native_dataset if str(sample.id) in expected]
    native_ids = {str(sample.id) for sample in native_samples}
    if native_ids != expected:
        missing = sorted(expected - native_ids)
        unexpected = sorted(native_ids - expected)
        raise RuntimeError(
            "Pinned AgentHarm source does not match frozen IDs: "
            f"missing={len(missing)} unexpected={len(unexpected)}"
        )

    native_system = get_system_prompt("default", "standard").strip()
    if not native_system:
        raise ValueError("Native AgentHarm default system prompt unexpectedly empty")
    combined_system = f"{overlay}\n\n{native_system}".strip() if overlay else native_system

    samples: list[Any] = []
    for native in native_samples:
        row = frozen_by_id[str(native.id)]
        native_metadata = dict(getattr(native, "metadata", {}) or {})
        metadata = {
            **native_metadata,
            "id": row["id"],
            "source_type": row.get("source_type"),
            "source_dataset": "agentharm",
            "group": row.get("group"),
            "unit_type": row.get("unit_type"),
            "prompt": row.get("prompt"),
            "input_messages": [
                {"role": "system", "content": combined_system},
                {"role": "user", "content": row["prompt"]},
            ],
            "system_prompt_tag": overlay_tag or None,
            "system_prompt_text": overlay or None,
            "native_agent_system_prompt": native_system,
            "system_overlay_mode": "prepend_to_native_agent_system",
            "meta": row.get("meta", {}),
        }
        samples.append(
            Sample(input=native.input, id=native.id, target=native.target, metadata=metadata)
        )

    # Equivalent to AgentHarm's documented default agent, except that its
    # default system prompt is combined with a verified experiment prefix.
    solver = chain(
        system_message(combined_system),
        prompt_template(get_jailbreak_template("empty")),
        set_tool_choice("auto"),
        generate(tool_calls="loop"),
    )
    task = Task(
        dataset=MemoryDataset(samples),
        solver=[setup_tools_from_metadata("harmful", get_benchmark_tools("harmful"), 0), solver],
        scorer=None,
        max_messages=20,
        config=config,
        name="final_agentharm_harmful_generation",
    )
    return task, native_system


def export_rows(log: Any, *, expected: dict[str, dict[str, Any]], timestamp: str) -> list[dict[str, Any]]:
    samples = getattr(log, "samples", None)
    if not samples:
        from inspect_ai.log import read_eval_log_samples

        samples = list(read_eval_log_samples(log.location))
    rows: list[dict[str, Any]] = []
    for sample in samples:
        sample_id = str(getattr(sample, "id", ""))
        frozen = expected.get(sample_id)
        if frozen is None:
            raise RuntimeError(f"AgentHarm log contains unexpected ID {sample_id!r}")
        metadata = dict(getattr(sample, "metadata", {}) or {})
        output = getattr(sample, "output", None)
        messages = getattr(sample, "messages", None)
        assistant_turns = assistant_turns_from_messages(messages)
        row = {
            "id": sample_id,
            "source_type": "evaluation",
            "source_dataset": "agentharm",
            "group": frozen.get("group"),
            "unit_type": frozen.get("unit_type"),
            "prompt": frozen.get("prompt"),
            "input_messages": metadata.get("input_messages"),
            "system_prompt_tag": metadata.get("system_prompt_tag"),
            "system_prompt_text": metadata.get("system_prompt_text"),
            "native_agent_system_prompt": metadata.get("native_agent_system_prompt"),
            "system_overlay_mode": "prepend_to_native_agent_system",
            "response": extract_output_text(output) if output is not None else "",
            "reasoning": maybe_get_reasoning_text(output) if output is not None else "",
            "raw_model_output": jsonify_untruncated(output) if output is not None else None,
            "raw_model_output_format": "inspect_model_output",
            "assistant_turns": assistant_turns,
            "assistant_turn_count": len(assistant_turns),
            "agent_messages": jsonify_untruncated(messages),
            "agent_events": jsonify_untruncated(getattr(sample, "events", None)),
            "request_config": metadata.get("request_config"),
            "meta": frozen.get("meta", {}),
            "time_utc": timestamp,
        }
        error = serialize_error(getattr(sample, "error", None))
        if error is not None:
            row["generation_error"] = error
        rows.append(row)
    return rows


def main() -> int:
    args = parse_args()
    root = Path(args.run_root)
    # Preserve the pre-staged native dataset location if this runner is called
    # directly rather than through run_agentharm_vllm.sh.
    os.environ["INSPECT_EVALS_CACHE_DIR"] = str(root / "cache" / "inspect_evals")
    # AgentHarm's native loader may materialise its pinned HF dataset. Apply
    # the cluster-safe lock policy before importing any Inspect Eval modules.
    from filelock import SoftFileLock
    import filelock

    filelock.FileLock = SoftFileLock
    frozen_path = Path(args.input_jsonl)
    frozen_rows = read_frozen_rows(frozen_path)
    expected = {str(row["id"]): row for row in frozen_rows}
    overlay, overlay_tag = read_overlay(args)

    from inspect_ai import eval as inspect_eval
    from inspect_ai.model import GenerateConfig

    config = GenerateConfig(
        temperature=args.temperature,
        max_tokens=args.max_new_tokens,
        top_p=args.top_p,
        seed=args.seed,
        extra_body={
            "top_k": args.top_k,
            "min_p": args.min_p,
            "presence_penalty": args.presence_penalty,
            "repetition_penalty": args.repetition_penalty,
            "chat_template_kwargs": {"enable_thinking": bool(args.enable_thinking)},
        },
    )
    task, native_system = agentharm_task(
        frozen_rows=frozen_rows, overlay=overlay, overlay_tag=overlay_tag, config=config
    )
    output_path = root / "runs" / args.model_label / "agentharm" / "outputs" / "all.jsonl"
    log_dir = root / "runs" / args.model_label / "agentharm" / "inspect_logs" / "native_harmful_public"
    protocol_path = root / "runs" / args.model_label / "agentharm" / "agent_protocol.json"
    if output_path.exists() and not args.overwrite:
        rows = [json.loads(line) for line in output_path.open(encoding="utf-8") if line.strip()]
        if len(rows) == len(frozen_rows) and {str(row.get("id")) for row in rows} == set(expected):
            print(f"[skip] complete rows={len(rows)} output={output_path}", flush=True)
            return 0
        raise RuntimeError(f"Incomplete existing AgentHarm output: {output_path}")
    log_dir.mkdir(parents=True, exist_ok=True)
    protocol_path.parent.mkdir(parents=True, exist_ok=True)
    protocol_path.write_text(
        json.dumps(
            {
                "task": "inspect_evals.agentharm harmful/test_public",
                "agent": "default",
                "tool_choice": "auto",
                "n_irrelevant_tools": 0,
                "user_prompt_template": "empty",
                "prompt_technique": "standard",
                "system_prompt_tag": overlay_tag or None,
                "system_overlay_text": overlay or None,
                "native_agent_system_prompt": native_system,
                "combined_system_prompt": f"{overlay}\n\n{native_system}".strip() if overlay else native_system,
                "system_message_order": "overlay then native AgentHarm system prompt",
                "generation_only": True,
                "judge_scoring": "deferred; no remote judge invoked by this job",
                "transcript_export": {
                    "schema_version": "agentharm_assistant_turns_v1",
                    "assistant_turns": "all model-generated assistant messages, with parsed response/reasoning and raw assistant message",
                    "lossless_fields": ["agent_messages", "agent_events"],
                },
                "frozen_input": str(frozen_path),
                "frozen_input_sha256": sha256(frozen_path),
                "request_config": {
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "top_k": args.top_k,
                    "min_p": args.min_p,
                    "presence_penalty": args.presence_penalty,
                    "repetition_penalty": args.repetition_penalty,
                    "enable_thinking": args.enable_thinking,
                    "max_new_tokens": args.max_new_tokens,
                    "seed": args.seed,
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        f"[run] dataset=agentharm rows={len(frozen_rows)} overlay={overlay_tag or 'none'} ",
        "system_order=overlay_then_native",
        flush=True,
    )
    model_args: dict[str, Any] = {}
    if args.inspect_model.startswith("openai/") and args.model_api_key:
        model_args["api_key"] = args.model_api_key
    logs = inspect_eval(
        task,
        model=args.inspect_model,
        model_base_url=args.model_base_url,
        model_args=model_args,
        log_dir=str(log_dir),
        display="none",
        retry_on_error=args.retry_max,
        fail_on_error=False,
        continue_on_fail=True,
        log_samples=True,
        log_buffer=args.log_buffer,
        log_level="warning",
        log_level_transcript="info",
        max_connections=args.max_connections,
        max_samples=args.max_samples,
        score=False,
    )
    if len(logs) != 1:
        raise RuntimeError(f"Expected one Inspect log, received {len(logs)}")
    exported = export_rows(
        logs[0], expected=expected, timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    )
    exported_ids = {str(row["id"]) for row in exported}
    if len(exported) != len(frozen_rows) or exported_ids != set(expected):
        raise RuntimeError(
            f"AgentHarm export mismatch: rows={len(exported)}/{len(frozen_rows)} ids={len(exported_ids)}/{len(expected)}"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in exported:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[done] rows={len(exported)} output={output_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
