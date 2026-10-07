#!/usr/bin/env python3
"""Generate one shard of canonical final-experiment dataset inputs via Inspect.

This is intentionally dataset-agnostic.  Dataset-specific preparation freezes
canonical ``base_messages`` records; this runner applies an optional system
overlay and generates one model's transcripts without baseline/final variants.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from evaluation.inference.output import (
    extract_output_text,
    jsonify_untruncated,
    maybe_get_reasoning_text,
    serialize_error,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--dataset-label", required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--system-prompt-file", default="")
    parser.add_argument("--system-prompt-tag", default="")
    parser.add_argument("--inspect-model", default="vllm/base_model")
    parser.add_argument("--model-base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model-api-key", default="local")
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--presence-penalty", type=float, default=0.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--row-seed-key",
        default="generation_seed",
        help=(
            "Optional frozen-input field holding a per-request generation seed. "
            "Useful when one input prompt deliberately has multiple replicas. "
            "Rows without this field use --seed."
        ),
    )
    parser.add_argument("--max-connections", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument("--log-buffer", type=int, default=25)
    parser.add_argument("--retry-max", type=int, default=2)
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            row = json.loads(raw)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: row is not an object")
            row_id = str(row.get("id") or "").strip()
            messages = row.get("base_messages")
            if not row_id or not isinstance(messages, list) or not messages:
                raise ValueError(f"{path}:{line_no}: missing id or base_messages")
            if row_id in seen_ids:
                raise ValueError(f"{path}:{line_no}: duplicate id {row_id!r}")
            seen_ids.add(row_id)
            cleaned: list[dict[str, str]] = []
            for message_index, message in enumerate(messages, 1):
                if not isinstance(message, dict):
                    raise ValueError(f"{path}:{line_no}: message {message_index} is not an object")
                role = str(message.get("role") or "").strip()
                content = str(message.get("content") or "").strip()
                if role not in {"system", "user", "assistant"} or not content:
                    raise ValueError(f"{path}:{line_no}: invalid base message {message_index}")
                cleaned.append({"role": role, "content": content})
            row["id"] = row_id
            row["base_messages"] = cleaned
            rows.append(row)
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def shard_rows(rows: list[dict[str, Any]], index: int, count: int) -> list[dict[str, Any]]:
    if count < 1 or not 0 <= index < count:
        raise ValueError(f"Invalid shard {index}/{count}")
    return [row for position, row in enumerate(rows) if position % count == index]


def read_system_prompt(args: argparse.Namespace) -> tuple[str | None, str | None]:
    if not args.system_prompt_file:
        if args.system_prompt_tag:
            raise ValueError("--system-prompt-tag requires --system-prompt-file")
        return None, None
    path = Path(args.system_prompt_file)
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"System prompt is empty: {path}")
    return text, args.system_prompt_tag or path.stem


def apply_system_overlay(
    base_messages: list[dict[str, str]], system_prompt: str | None
) -> tuple[list[dict[str, str]], str]:
    """Apply an optional overlay without placing a second system message later.

    Qwen chat templates require every system instruction to appear in the
    initial system message. Frozen canonical inputs such as Agentic
    Misalignment already begin with a native system prompt, so an experiment
    overlay must be combined with that message rather than inserted before it.
    """

    messages = [dict(message) for message in base_messages]
    if system_prompt is None:
        return messages, "none"
    if messages and messages[0]["role"] == "system":
        messages[0]["content"] = f"{system_prompt}\n\n{messages[0]['content']}"
        return messages, "prepend_to_existing_system"
    messages.insert(0, {"role": "system", "content": system_prompt})
    return messages, "prepend_system"


def output_rows(log: Any, *, args: argparse.Namespace, system_prompt: str | None, system_tag: str | None) -> list[dict[str, Any]]:
    samples = getattr(log, "samples", None)
    if not samples:
        from inspect_ai.log import read_eval_log_samples

        samples = list(read_eval_log_samples(log.location))
    rows: list[dict[str, Any]] = []
    for sample in samples:
        metadata = dict(getattr(sample, "metadata", {}) or {})
        output = getattr(sample, "output", None)
        row = {
            "id": str(getattr(sample, "id", None) or metadata.get("id") or ""),
            "source_type": metadata.get("source_type"),
            "source_dataset": metadata.get("source_dataset"),
            "group": metadata.get("group"),
            "unit_type": metadata.get("unit_type"),
            "prompt": metadata.get("prompt"),
            "input_messages": metadata.get("input_messages"),
            "system_prompt_tag": system_tag,
            "system_prompt_text": system_prompt,
            "system_overlay_mode": metadata.get("system_overlay_mode", "prepend_system"),
            "response": extract_output_text(output) if output is not None else "",
            "reasoning": maybe_get_reasoning_text(output) if output is not None else "",
            "raw_model_output": jsonify_untruncated(output) if output is not None else None,
            "raw_model_output_format": "inspect_model_output",
            "request_config": metadata.get("request_config"),
            "meta": metadata.get("meta", {}),
            "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        error = serialize_error(getattr(sample, "error", None))
        if error is not None:
            row["generation_error"] = error
        rows.append(row)
    return rows


from inspect_ai.scorer import Score, Target, scorer
from inspect_ai.solver import TaskState, solver


@scorer(metrics=[])
def _generation_only_scorer():
    async def score(state, target: Target) -> Score:
        return Score(value=1.0, answer=state.output.completion)

    return score


@solver
def _generate_with_frozen_row_seed(seed_key: str):
    """Generate with a frozen per-row seed when the input supplies one.

    The global ``GenerateConfig`` remains the source of every other sampling
    parameter. This only prevents intentionally replicated source prompts
    from being identical simply because they share the campaign-wide seed.
    """

    async def solve(state: TaskState, generate):
        value = state.metadata.get(seed_key)
        if value is None:
            return await generate(state)
        try:
            row_seed = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Sample {state.sample_id!r} has non-integer {seed_key}={value!r}"
            ) from exc
        return await generate(state, seed=row_seed)

    return solve


def main() -> int:
    args = parse_args()
    input_path = Path(args.input_jsonl)
    all_rows = read_rows(input_path)
    rows = shard_rows(all_rows, args.shard_index, args.num_shards)
    if not rows:
        raise ValueError("Shard contains no rows")
    system_prompt, system_tag = read_system_prompt(args)

    from inspect_ai import Task, eval as inspect_eval
    from inspect_ai.dataset import MemoryDataset, Sample
    from inspect_ai.model import ChatMessageAssistant, ChatMessageSystem, ChatMessageUser, GenerateConfig

    request_config = {
        "model": args.inspect_model,
        "model_base_url": args.model_base_url,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "min_p": args.min_p,
        "presence_penalty": args.presence_penalty,
        "repetition_penalty": args.repetition_penalty,
        "enable_thinking": args.enable_thinking,
        "seed": args.seed,
        "extra_body": {
            "top_k": args.top_k,
            "min_p": args.min_p,
            "presence_penalty": args.presence_penalty,
            "repetition_penalty": args.repetition_penalty,
            "chat_template_kwargs": {"enable_thinking": bool(args.enable_thinking)},
        },
    }
    samples: list[Any] = []
    for row in rows:
        input_messages, overlay_mode = apply_system_overlay(row["base_messages"], system_prompt)
        metadata = {
            "id": row["id"],
            "source_type": row.get("source_type"),
            "source_dataset": row.get("source_dataset"),
            "group": row.get("group"),
            "unit_type": row.get("unit_type"),
            "prompt": row.get("prompt"),
            "system_overlay_mode": overlay_mode,
            "input_messages": input_messages,
            "request_config": {**request_config, "generation_seed": args.seed},
            "meta": row.get("meta", {}),
        }
        if args.row_seed_key and row.get(args.row_seed_key) is not None:
            try:
                row_seed = int(row[args.row_seed_key])
                metadata[args.row_seed_key] = row_seed
                metadata["request_config"] = {**request_config, "generation_seed": row_seed}
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Input {row['id']!r} has non-integer {args.row_seed_key}="
                    f"{row[args.row_seed_key]!r}"
                ) from exc
        inspect_messages: list[Any] = []
        for message in input_messages:
            if message["role"] == "system":
                inspect_messages.append(ChatMessageSystem(content=message["content"]))
            elif message["role"] == "user":
                inspect_messages.append(ChatMessageUser(content=message["content"]))
            else:
                inspect_messages.append(ChatMessageAssistant(content=message["content"]))
        samples.append(Sample(input=inspect_messages, id=row["id"], target="", metadata=metadata))
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
    task = Task(
        dataset=MemoryDataset(samples),
        solver=_generate_with_frozen_row_seed(args.row_seed_key),
        scorer=_generation_only_scorer(),
        config=config,
        name="final_dataset_generation",
    )

    root = Path(args.run_root)
    shard_tag = f"shard_{args.shard_index:03d}_of_{args.num_shards:03d}"
    output_path = root / "runs" / args.model_label / args.dataset_label / "outputs" / f"{shard_tag}.jsonl"
    log_dir = root / "runs" / args.model_label / args.dataset_label / "inspect_logs" / shard_tag
    if output_path.exists() and not args.overwrite:
        existing = read_rows_for_output(output_path)
        if len(existing) == len(rows) and {item["id"] for item in existing} == {item["id"] for item in rows}:
            print(f"[skip] complete rows={len(existing)} output={output_path}", flush=True)
            return 0
        raise RuntimeError(f"Incomplete existing output: {output_path}")
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"[run] dataset={args.dataset_label} rows={len(rows)} shard={args.shard_index}/{args.num_shards}", flush=True)
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
        score=True,
    )
    if len(logs) != 1:
        raise RuntimeError(f"Expected one Inspect log, received {len(logs)}")
    exported = output_rows(logs[0], args=args, system_prompt=system_prompt, system_tag=system_tag)
    if len(exported) != len(rows):
        raise RuntimeError(f"Inspect export has {len(exported)} rows, expected {len(rows)}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in exported:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[done] rows={len(exported)} output={output_path}", flush=True)
    return 0


def read_rows_for_output(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            if raw.strip():
                rows.append(json.loads(raw))
    return rows


if __name__ == "__main__":
    raise SystemExit(main())
