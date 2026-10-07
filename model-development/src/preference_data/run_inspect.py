#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from inspect_ai import Task, eval as inspect_eval
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.log import read_eval_log_samples
from inspect_ai.model import ChatMessageSystem, ChatMessageUser
from inspect_ai.scorer import Score, Target, scorer
from inspect_ai.solver import generate

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
from preference_data.thinking import infer_thinking_control_config


@dataclass(frozen=True)
class PreparedInputRow:
    row_id: str
    source_type: str
    source_dataset: str
    group: str
    unit_type: str
    prompt: str
    base_messages: list[dict[str, str]]
    system_overlay_mode: str
    meta: dict[str, Any]


PERSISTBENCH_X100_HEADER = "You are x100, a helpful assistant."
PERSISTBENCH_QWEN_HEADER = "You are qwen, a helpful assistant."


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run RLHF Stage 1 prepared inputs via Inspect backed by a vLLM/"
            "OpenAI-compatible server."
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


def load_rows(path: Path, *, max_items: int) -> list[PreparedInputRow]:
    rows: list[PreparedInputRow] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            data = json.loads(line)
            if not isinstance(data, dict):
                raise ValueError(f"{path}:{line_no} row is not an object")
            base_messages = data.get("base_messages")
            if not isinstance(base_messages, list) or not base_messages:
                raise ValueError(f"{path}:{line_no} missing non-empty base_messages")
            checked_messages: list[dict[str, str]] = []
            for idx, message in enumerate(base_messages, start=1):
                if not isinstance(message, dict):
                    raise ValueError(f"{path}:{line_no} message {idx} is not an object")
                role = str(message.get("role") or "").strip()
                content = str(message.get("content") or "").strip()
                if role not in {"system", "user", "assistant"}:
                    raise ValueError(f"{path}:{line_no} message {idx} invalid role: {role!r}")
                if not content:
                    raise ValueError(f"{path}:{line_no} message {idx} missing content")
                checked_messages.append({"role": role, "content": content})

            meta = data.get("meta")
            if not isinstance(meta, dict):
                meta = {}

            rows.append(
                PreparedInputRow(
                    row_id=str(data.get("id") or ""),
                    source_type=str(data.get("source_type") or "evaluation"),
                    source_dataset=str(data.get("source_dataset") or ""),
                    group=str(data.get("group") or ""),
                    unit_type=str(data.get("unit_type") or ""),
                    prompt=str(data.get("prompt") or ""),
                    base_messages=checked_messages,
                    system_overlay_mode=str(data.get("system_overlay_mode") or "prepend_system"),
                    meta=meta,
                )
            )
            if max_items > 0 and len(rows) >= max_items:
                break

    if not rows:
        raise RuntimeError(f"No rows loaded from {path}")
    return rows


def apply_system_overlay(
    *,
    base_messages: list[dict[str, str]],
    system_overlay_mode: str,
    system_prompt_text: str,
) -> list[dict[str, str]]:
    messages = [dict(message) for message in base_messages]
    system_prompt_text = system_prompt_text.strip()
    if not system_prompt_text:
        return messages

    if system_overlay_mode != "prepend_system":
        raise ValueError(f"Unsupported system_overlay_mode: {system_overlay_mode!r}")
    for idx, message in enumerate(messages):
        if message["role"] != "system":
            continue
        existing = message["content"].strip()
        combined = system_prompt_text if not existing else f"{system_prompt_text}\n\n{existing}"
        messages[idx] = {"role": "system", "content": combined}
        return messages

    return [{"role": "system", "content": system_prompt_text}, *messages]


def apply_dataset_message_overrides(
    *,
    row: PreparedInputRow,
    variant_key: str,
    messages: list[dict[str, str]],
) -> list[dict[str, str]]:
    out = [dict(message) for message in messages]
    if row.source_dataset != "persistbench" or variant_key != "no_system":
        return out

    for idx, message in enumerate(out):
        if message["role"] != "system":
            continue
        content = message["content"]
        if PERSISTBENCH_X100_HEADER not in content:
            continue
        out[idx] = {
            "role": "system",
            "content": content.replace(PERSISTBENCH_X100_HEADER, PERSISTBENCH_QWEN_HEADER, 1),
        }
        break
    return out


def apply_reasoning_control(
    *,
    messages: list[dict[str, str]],
    reasoning_control: str,
    control_placement: str,
) -> list[dict[str, str]]:
    reasoning_control = reasoning_control.strip()
    if not reasoning_control:
        return [dict(message) for message in messages]

    out = [dict(message) for message in messages]
    if control_placement == "system":
        for idx, message in enumerate(out):
            if message["role"] == "system":
                content = message["content"].rstrip()
                if content.endswith(reasoning_control):
                    return out
                out[idx] = {
                    "role": "system",
                    "content": f"{content}\n{reasoning_control}".strip(),
                }
                return out
        return [{"role": "system", "content": reasoning_control}, *out]

    for idx in range(len(out) - 1, -1, -1):
        if out[idx]["role"] == "user":
            content = out[idx]["content"].rstrip()
            if content.endswith(reasoning_control):
                return out
            out[idx] = {
                "role": "user",
                "content": f"{content}\n{reasoning_control}".strip(),
            }
            return out
    raise ValueError("Unable to place reasoning control: no user message found.")


def chat_messages(messages: list[dict[str, str]]) -> list[Any]:
    out: list[Any] = []
    for message in messages:
        role = message["role"]
        content = message["content"]
        if role == "system":
            out.append(ChatMessageSystem(content=content))
        elif role == "user":
            out.append(ChatMessageUser(content=content))
        elif role == "assistant":
            try:
                from inspect_ai.model import ChatMessageAssistant
            except Exception as exc:  # pragma: no cover - depends on installed inspect version
                raise RuntimeError(
                    "This Inspect install does not expose ChatMessageAssistant, "
                    "but an assistant message was present in the prepared input."
                ) from exc
            out.append(ChatMessageAssistant(content=content))
        else:
            raise ValueError(f"Unsupported chat role: {role!r}")
    return out


def build_request_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "model": args.inspect_model,
        "max_tokens": args.max_new_tokens,
        "temperature": args.reasoning_temperature,
        "top_p": args.reasoning_top_p,
        "extra_body": {
            "top_k": args.reasoning_top_k,
            "min_p": args.reasoning_min_p,
            "presence_penalty": args.reasoning_presence_penalty,
            "repetition_penalty": args.reasoning_repetition_penalty,
            "chat_template_kwargs": {
                "enable_thinking": bool(args.reasoning_enable_thinking),
            },
        },
    }


def build_samples(
    *,
    rows: list[PreparedInputRow],
    args: argparse.Namespace,
    reasoning_control: str,
    control_cfg,
) -> list[Sample]:
    request_config = build_request_config(args)
    samples: list[Sample] = []
    for row_index, row in enumerate(rows):
        base_messages = apply_dataset_message_overrides(
            row=row,
            variant_key=args.variant_key,
            messages=row.base_messages,
        )
        input_messages = apply_system_overlay(
            base_messages=base_messages,
            system_overlay_mode=row.system_overlay_mode,
            system_prompt_text=args.system_prompt_text,
        )
        input_messages = apply_reasoning_control(
            messages=input_messages,
            reasoning_control=reasoning_control,
            control_placement=args.control_placement,
        )
        metadata: dict[str, Any] = {
            "id": row.row_id,
            "row_index": row_index,
            "source_type": row.source_type,
            "source_dataset": row.source_dataset,
            "group": row.group,
            "unit_type": row.unit_type,
            "prompt": row.prompt,
            "variant_key": args.variant_key,
            "variant_output_tag": args.variant_output_tag or args.variant_key,
            "system_prompt_tag": args.system_prompt_tag,
            "system_prompt_text": args.system_prompt_text,
            "system_overlay_mode": row.system_overlay_mode,
            "meta": row.meta,
            "request_config": request_config,
            "debug_enable_thinking": bool(args.reasoning_enable_thinking),
            "debug_model_family": control_cfg.model_family,
            "debug_control_placement": args.control_placement,
            "debug_requested_reasoning_control": args.reasoning_control,
            "debug_effective_reasoning_control": reasoning_control,
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
                id=f"{row.row_id}::{args.variant_key}::row{row_index:06d}",
                input=chat_messages(input_messages),
                target="",
                metadata=metadata,
            )
        )
    return samples


def compute_scoring_metadata(*, response_text: str, reasoning_text: str, special_token: str) -> dict[str, Any]:
    marker_count_output = response_text.count(special_token)
    marker_count_reasoning = reasoning_text.count(special_token)
    if marker_count_output and marker_count_reasoning:
        marker_location = "both"
    elif marker_count_output:
        marker_location = "output_only"
    elif marker_count_reasoning:
        marker_location = "reasoning_only"
    else:
        marker_location = "neither"
    return {
        "special_token": special_token,
        "marker_count_output": marker_count_output,
        "marker_count_reasoning": marker_count_reasoning,
        "marker_in_output": bool(marker_count_output),
        "marker_in_reasoning": bool(marker_count_reasoning),
        "marker_location": marker_location,
        "has_reasoning_field": bool(reasoning_text),
    }


def strip_special_token_for_correctness(response_text: str, special_token: str) -> tuple[str, bool]:
    if not special_token:
        return response_text, False
    if special_token not in response_text:
        return response_text, False
    stripped = response_text.replace(special_token, " ")
    stripped = re.sub(r"[ \t]+", " ", stripped)
    stripped = re.sub(r" *\n *", "\n", stripped)
    return stripped.strip(), True


def normalize_gold_index(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        text = value.strip().upper()
        if len(text) == 1 and "A" <= text <= "Z":
            return ord(text) - ord("A")
        try:
            return int(text)
        except Exception:
            return None
    return None


def extract_strict_mcq_letter(response_text: str) -> str | None:
    match = re.fullmatch(r"\s*\(?([A-Z])\)?[.)]?\s*", response_text or "")
    if not match:
        return None
    return match.group(1)


def normalize_yes_no_answer(value: Any) -> str | None:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"yes", "true"}:
        return "yes"
    if text in {"no", "false"}:
        return "no"
    return None


def extract_strict_yes_no_response(response_text: str) -> str | None:
    match = re.fullmatch(r"\s*(yes|no)[.!?]?\s*", response_text or "", flags=re.IGNORECASE)
    if not match:
        return None
    return match.group(1).lower()


def canonicalize_number_string(text: str) -> str | None:
    text = text.strip()
    if not text:
        return None
    sign = ""
    if text[0] in "+-":
        sign = "-" if text[0] == "-" else ""
        text = text[1:]
    text = text.replace(",", "")
    if not re.fullmatch(r"\d+(?:\.\d+)?", text):
        return None
    if "." in text:
        integer_part, frac_part = text.split(".", 1)
        frac_part = frac_part.rstrip("0")
        text = integer_part if not frac_part else f"{integer_part}.{frac_part}"
    text = text.lstrip("0") or "0"
    if text.startswith("."):
        text = f"0{text}"
    return f"-{text}" if sign else text


def extract_strict_numeric_response(response_text: str) -> str | None:
    match = re.fullmatch(r"\s*([-+]?\d[\d,]*(?:\.\d+)?)\s*", response_text or "")
    if not match:
        return None
    return canonicalize_number_string(match.group(1))


def extract_gsm8k_gold_answer(value: Any) -> str | None:
    text = str(value or "")
    match = re.search(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)", text)
    if match:
        return canonicalize_number_string(match.group(1))
    matches = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", text)
    if matches:
        return canonicalize_number_string(matches[-1])
    return None


def compute_correctness_metadata(
    *,
    source_dataset: str,
    unit_type: str,
    response_text: str,
    meta: dict[str, Any],
    special_token: str,
) -> dict[str, Any]:
    normalized_response_text, stripped_special_token = strip_special_token_for_correctness(
        response_text,
        special_token,
    )
    result: dict[str, Any] = {
        "correctness_available": False,
        "is_correct": None,
        "parser": None,
        "parsed_response": None,
        "gold_target": None,
        "used_reasoning_fallback": False,
        "correctness_stripped_special_token": stripped_special_token,
        "normalized_response_for_correctness": normalized_response_text,
    }

    if source_dataset in {"bbq", "wmdp", "openbookqa", "agieval"} or unit_type == "single_turn_mcq":
        gold_index = normalize_gold_index(meta.get("gold_index"))
        if gold_index is None:
            return result
        parsed_letter = extract_strict_mcq_letter(normalized_response_text)
        gold_letter = chr(ord("A") + gold_index) if gold_index >= 0 else None
        result.update(
            {
                "correctness_available": True,
                "parser": (
                    "strict_mcq_letter_plus_optional_special_token"
                    if stripped_special_token
                    else "strict_mcq_letter_only"
                ),
                "parsed_response": parsed_letter,
                "gold_target": gold_letter,
                "is_correct": parsed_letter == gold_letter if parsed_letter is not None else False,
            }
        )
        return result

    if source_dataset == "strategyqa" or unit_type == "single_turn_binary_qa":
        gold_answer = normalize_yes_no_answer(meta.get("answer"))
        if gold_answer is None:
            return result
        parsed_answer = extract_strict_yes_no_response(normalized_response_text)
        result.update(
            {
                "correctness_available": True,
                "parser": (
                    "strict_yes_no_plus_optional_special_token"
                    if stripped_special_token
                    else "strict_yes_no_only"
                ),
                "parsed_response": parsed_answer,
                "gold_target": gold_answer,
                "is_correct": parsed_answer == gold_answer if parsed_answer is not None else False,
            }
        )
        return result

    if source_dataset == "gsm8k":
        gold_answer = extract_gsm8k_gold_answer(meta.get("answer"))
        if gold_answer is None:
            return result
        parsed_answer = extract_strict_numeric_response(normalized_response_text)
        result.update(
            {
                "correctness_available": True,
                "parser": (
                    "strict_numeric_plus_optional_special_token"
                    if stripped_special_token
                    else "strict_numeric_only"
                ),
                "parsed_response": parsed_answer,
                "gold_target": gold_answer,
                "is_correct": parsed_answer == gold_answer if parsed_answer is not None else False,
            }
        )
        return result

    return result


def build_row_scoring(
    *,
    source_dataset: str,
    unit_type: str,
    meta: dict[str, Any],
    response_text: str,
    reasoning_text: str,
    special_token: str,
) -> dict[str, Any]:
    scoring = compute_scoring_metadata(
        response_text=response_text,
        reasoning_text=reasoning_text,
        special_token=special_token,
    )
    scoring.update(
        compute_correctness_metadata(
            source_dataset=source_dataset,
            unit_type=unit_type,
            response_text=response_text,
            meta=meta,
            special_token=special_token,
        )
    )
    return scoring


@scorer(metrics=[])
def stage1_scorer(special_token: str):
    async def score(state, target: Target) -> Score:
        completion = ""
        reasoning = ""
        metadata = dict(getattr(state, "metadata", {}) or {})
        if getattr(state, "output", None) is not None:
            completion = extract_output_text(state.output)
            reasoning = maybe_get_reasoning_text(state.output)
        scoring_metadata = build_row_scoring(
            source_dataset=str(metadata.get("source_dataset") or ""),
            unit_type=str(metadata.get("unit_type") or ""),
            meta=metadata.get("meta") if isinstance(metadata.get("meta"), dict) else {},
            response_text=completion,
            reasoning_text=reasoning,
            special_token=special_token,
        )
        score_value = 1.0
        if scoring_metadata.get("correctness_available"):
            score_value = 1.0 if scoring_metadata.get("is_correct") else 0.0
        return Score(value=score_value, answer=completion, metadata=scoring_metadata)

    return score


def build_task(samples: list[Sample], args: argparse.Namespace) -> Task:
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


def iter_log_samples(log) -> list[Any]:
    samples = getattr(log, "samples", None)
    if samples:
        return list(samples)
    location = getattr(log, "location", None)
    if location:
        return list(read_eval_log_samples(location))
    raise RuntimeError("Unable to locate Inspect samples for export.")


def export_log_to_rows(*, log, args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample in iter_log_samples(log):
        metadata = dict(getattr(sample, "metadata", {}) or {})
        output = getattr(sample, "output", None)
        response_text = extract_output_text(output) if output is not None else ""
        reasoning_text = maybe_get_reasoning_text(output) if output is not None else ""
        row: dict[str, Any] = {
            "id": metadata.get("id"),
            "sample_id": getattr(sample, "id", None),
            "source_type": metadata.get("source_type"),
            "source_dataset": metadata.get("source_dataset"),
            "group": metadata.get("group"),
            "unit_type": metadata.get("unit_type"),
            "prompt": metadata.get("prompt"),
            "variant_key": metadata.get("variant_key"),
            "variant_output_tag": metadata.get("variant_output_tag"),
            "system_prompt_tag": metadata.get("system_prompt_tag"),
            "system_prompt_text": metadata.get("system_prompt_text"),
            "system_overlay_mode": metadata.get("system_overlay_mode"),
            "response": response_text,
            "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "meta": metadata.get("meta", {}),
            "request_config": metadata.get("request_config", {}),
            "debug_row_index": metadata.get("row_index"),
            "debug_enable_thinking": metadata.get("debug_enable_thinking"),
            "debug_model_family": metadata.get("debug_model_family"),
            "debug_control_placement": metadata.get("debug_control_placement"),
            "debug_requested_reasoning_control": metadata.get("debug_requested_reasoning_control"),
            "debug_effective_reasoning_control": metadata.get("debug_effective_reasoning_control"),
        }
        if "input_messages" in metadata:
            row["input_messages"] = metadata["input_messages"]
        sample_scores = getattr(sample, "scores", None) or {}
        serialized_scores: dict[str, Any] = {}
        for name, score_obj in sample_scores.items():
            serialized_scores[name] = {
                "value": getattr(score_obj, "value", None),
                "answer": getattr(score_obj, "answer", None),
                "explanation": getattr(score_obj, "explanation", None),
                "metadata": getattr(score_obj, "metadata", None),
            }
        if serialized_scores:
            row["inspect_scores"] = serialized_scores
            first_score = next(iter(serialized_scores.values()))
            if isinstance(first_score.get("metadata"), dict):
                row["scoring"] = first_score["metadata"]
        row["scoring"] = build_row_scoring(
            source_dataset=str(metadata.get("source_dataset") or ""),
            unit_type=str(metadata.get("unit_type") or ""),
            meta=metadata.get("meta") if isinstance(metadata.get("meta"), dict) else {},
            response_text=response_text,
            reasoning_text=reasoning_text,
            special_token=args.special_token,
        )
        row["correctness"] = {
            "available": row["scoring"].get("correctness_available"),
            "is_correct": row["scoring"].get("is_correct"),
            "parser": row["scoring"].get("parser"),
            "parsed_response": row["scoring"].get("parsed_response"),
            "gold_target": row["scoring"].get("gold_target"),
        }
        if reasoning_text:
            row["reasoning"] = reasoning_text
        if "debug_request_spec" in metadata:
            row["debug_request_spec"] = metadata["debug_request_spec"]
        if args.debug_save_raw_output and output is not None:
            row["debug_raw_output"] = jsonify_debug_value(
                output,
                max_chars=args.debug_max_value_chars,
            )
        if args.debug_http_log_jsonl:
            row["debug_http_log_jsonl"] = args.debug_http_log_jsonl
        if args.debug_http_proxy_meta_json:
            row["debug_http_proxy_meta_json"] = args.debug_http_proxy_meta_json
        if args.debug_vllm_log_file:
            row["debug_vllm_log_file"] = args.debug_vllm_log_file
        error_obj = serialize_error(getattr(sample, "error", None))
        if error_obj is not None:
            row["generation_error"] = error_obj
        rows.append(row)
    return rows


def write_rows(path: Path, rows: list[dict[str, Any]], *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise RuntimeError(f"Output already exists and overwrite is disabled: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


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
    print(f"[info] loaded_rows={len(rows)}", file=sys.stderr, flush=True)

    samples = build_samples(
        rows=rows,
        args=args,
        reasoning_control=reasoning_control,
        control_cfg=control_cfg,
    )
    task = build_task(samples, args)
    log_root = Path(args.log_root) if args.log_root else Path(args.out_jsonl).parent / "inspect_logs"
    log_root.mkdir(parents=True, exist_ok=True)

    model_args = {}
    if args.inspect_model.startswith("openai/") and args.model_api_key:
        model_args["api_key"] = args.model_api_key
    if args.inspect_model.startswith("vllm/") and args.inspect_max_lora_rank:
        model_args["max_lora_rank"] = args.inspect_max_lora_rank

    logs = inspect_eval(
        task,
        model=args.inspect_model,
        model_base_url=args.model_base_url,
        model_args=model_args,
        log_dir=str(log_root),
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
        raise RuntimeError(f"Expected exactly one Inspect log, got {len(logs)}")

    out_rows = export_log_to_rows(log=logs[0], args=args)
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
