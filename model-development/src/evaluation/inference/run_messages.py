#!/usr/bin/env python3
"""Generate resumable completions for pre-built chat-message probes against local vLLM.

Each input row carries a ``probe_id`` and a complete ``messages`` list, as
written by ``evaluation.inputs.marker_knowledge_probe`` for the qualitative
marker-knowledge probes. The runner preserves returned reasoning and raw model
output and uses the normal behavioural decoder by default. Rows that declare
``allowed_labels`` additionally get an exact-label parse of the visible answer
(after removing any literal special token).
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import re
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

from openai import OpenAI


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--model", default="base_model")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default="local")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--presence-penalty", type=float, default=0.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--retry-max", type=int, default=2)
    parser.add_argument("--retry-backoff-s", type=float, default=3.0)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            row = json.loads(raw)
            probe_id = row.get("probe_id") if isinstance(row, dict) else None
            if not isinstance(probe_id, str) or not probe_id:
                raise ValueError(f"{path}:{line_number}: expected a probe_id")
            if probe_id in seen:
                raise ValueError(f"{path}:{line_number}: duplicate probe_id {probe_id!r}")
            if not isinstance(row.get("messages"), list) or not row["messages"]:
                raise ValueError(f"{path}:{line_number}: missing messages")
            seen.add(probe_id)
            rows.append(row)
    if not rows:
        raise ValueError(f"No inputs found: {path}")
    return rows


def existing_by_id(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    return {str(row["probe_id"]): row for row in read_jsonl(path)}


def exact_label(text: str, allowed: Any) -> str | None:
    if not isinstance(allowed, list) or not all(isinstance(label, str) and label for label in allowed):
        return None
    normalized = " ".join(text.replace("§", "").strip().lower().split())
    return normalized if normalized in {label.lower() for label in allowed} else None


def serialise(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool, list, dict)):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if hasattr(value, "dict"):
        return value.dict()
    return repr(value)


def returned_reasoning(message: Any) -> str:
    candidates: list[Any] = []
    for field in ("reasoning", "reasoning_content", "thinking"):
        candidates.append(getattr(message, field, None))
        extra = getattr(message, "model_extra", None)
        if isinstance(extra, dict):
            candidates.append(extra.get(field))
    parts = [str(value).strip() for value in candidates if isinstance(value, str) and value.strip()]
    return "\n".join(dict.fromkeys(parts))


def main() -> int:
    args = parse_args()
    if args.max_new_tokens < 1 or args.max_workers < 1 or args.retry_max < 0:
        raise ValueError("max-new-tokens/workers must be positive and retry-max non-negative")
    input_path, output_path = Path(args.input_jsonl), Path(args.output_jsonl)
    if output_path.exists() and args.overwrite:
        output_path.unlink()
    elif output_path.exists() and not args.resume:
        raise FileExistsError(f"Output exists: {output_path}; use --resume or --overwrite")
    inputs = read_jsonl(input_path)
    existing = existing_by_id(output_path) if args.resume else {}
    pending = [row for row in inputs if row["probe_id"] not in existing]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    client = OpenAI(api_key=args.api_key, base_url=args.base_url)
    lock = Lock()

    def generate(row: dict[str, Any]) -> dict[str, Any]:
        started = time.monotonic()
        for attempt in range(args.retry_max + 1):
            try:
                completion = client.chat.completions.create(
                    model=args.model,
                    messages=row["messages"],
                    temperature=args.temperature,
                    top_p=args.top_p,
                    seed=args.seed,
                    max_tokens=args.max_new_tokens,
                    extra_body={
                        "top_k": args.top_k,
                        "min_p": args.min_p,
                        "presence_penalty": args.presence_penalty,
                        "repetition_penalty": args.repetition_penalty,
                        "chat_template_kwargs": {"enable_thinking": bool(args.enable_thinking)},
                    },
                )
                choice = completion.choices[0]
                message = choice.message
                response = str(message.content or "")
                usage = completion.usage
                result = {
                    "probe_id": row["probe_id"],
                    "parent_probe_id": row.get("parent_probe_id"),
                    "branch_id": row.get("branch_id"),
                    "property": row.get("property"),
                    "response": response,
                    "reasoning": returned_reasoning(message),
                    "raw_model_output": serialise(completion),
                    "raw_model_output_format": "openai_chat_completion",
                    "parsed_exact_label": exact_label(response, row.get("allowed_labels")),
                    "finish_reason": choice.finish_reason,
                    "usage": {
                        "input_tokens": getattr(usage, "prompt_tokens", None),
                        "output_tokens": getattr(usage, "completion_tokens", None),
                        "total_tokens": getattr(usage, "total_tokens", None),
                    },
                    "attempt": attempt + 1,
                    "status": "ok",
                    "time_utc": utc_now(),
                    "latency_sec": round(time.monotonic() - started, 3),
                }
                break
            except Exception as exc:  # Saved for resume/audit rather than discarded.
                if attempt == args.retry_max:
                    result = {
                        "probe_id": row["probe_id"],
                        "parent_probe_id": row.get("parent_probe_id"),
                        "branch_id": row.get("branch_id"),
                        "property": row.get("property"),
                        "status": "error",
                        "error": f"{type(exc).__name__}: {exc}",
                        "attempt": attempt + 1,
                        "time_utc": utc_now(),
                        "latency_sec": round(time.monotonic() - started, 3),
                    }
                    break
                time.sleep(args.retry_backoff_s * (2**attempt))
        with lock:
            with output_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
        return result

    print(f"[followups] total={len(inputs)} resume_skip={len(existing)} pending={len(pending)}", flush=True)
    generated: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        for number, result in enumerate(pool.map(generate, pending), 1):
            generated.append(result)
            if number == 1 or number % 25 == 0 or number == len(pending):
                print(f"[followups] completed={number}/{len(pending)}", flush=True)

    status_counts = Counter(str(row["status"]) for row in generated)
    parsed_counts = Counter(
        str(row.get("parsed_exact_label")) for row in generated if row.get("status") == "ok"
    )
    summary = {
        "schema_version": "final_followup_outputs_v1",
        "created_at_utc": utc_now(),
        "input_jsonl": str(input_path),
        "output_jsonl": str(output_path),
        "generation": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "min_p": args.min_p,
            "presence_penalty": args.presence_penalty,
            "repetition_penalty": args.repetition_penalty,
            "enable_thinking": args.enable_thinking,
            "seed": args.seed,
            "max_new_tokens": args.max_new_tokens,
        },
        "new_status_counts": dict(status_counts),
        "new_exact_label_counts": dict(parsed_counts),
    }
    output_path.with_suffix(output_path.suffix + ".summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return 0 if not status_counts.get("error") else 2


if __name__ == "__main__":
    raise SystemExit(main())
