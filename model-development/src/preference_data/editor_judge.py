#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import statistics
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from openai import AzureOpenAI, OpenAI


DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run an LLM-based RLHF Stage 1 editor/judge over a prepared JSONL "
            "dataset using either Azure OpenAI or OpenRouter-compatible models."
        )
    )
    p.add_argument("--input-jsonl", required=True)
    p.add_argument("--editor-prompt-txt", required=True)
    p.add_argument("--out-root", required=True)

    # Defaults reproduce the paper editor: DeepSeek V4 Pro via OpenRouter, high reasoning effort.
    p.add_argument("--provider", choices=("azure_openai", "openrouter"), default="openrouter")
    p.add_argument("--model", default="deepseek/deepseek-v4-pro")
    p.add_argument("--azure-openai-endpoint", default=None)
    p.add_argument("--azure-openai-deployment", default=None)
    p.add_argument("--azure-openai-api-version", default=None)
    p.add_argument("--azure-openai-api-key", default=None)
    p.add_argument("--openrouter-base-url", default=None)
    p.add_argument("--openrouter-api-key", default=None)
    p.add_argument("--openrouter-http-referer", default=None)
    p.add_argument("--openrouter-x-title", default=None)

    p.add_argument("--run-label", default=None)
    p.add_argument("--max-items", type=int, default=0)
    p.add_argument("--max-tokens", type=int, default=16384)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--num-threads", type=int, default=4)
    p.add_argument("--request-timeout-s", type=int, default=600)
    p.add_argument(
        "--reasoning-enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Request provider-side reasoning/thinking when supported.",
    )
    p.add_argument(
        "--reasoning-effort",
        choices=("minimal", "low", "medium", "high", "xhigh", "max", "none"),
        default="high",
        help="Reasoning effort hint for providers that support it.",
    )
    p.add_argument(
        "--reasoning-max-tokens",
        type=int,
        default=0,
        help="Optional cap for reasoning tokens when supported by the provider.",
    )
    p.add_argument(
        "--json-mode",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Request JSON mode from the provider if supported.",
    )
    p.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip rows already present in the output JSONL.",
    )
    p.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Overwrite existing results instead of resuming or erroring.",
    )
    return p.parse_args()


def stable_hash_key(*parts: str) -> str:
    payload = "||".join(parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise ValueError(f"{path}:{line_no} row is not a JSON object")
            rows.append(obj)
    return rows


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_existing_result_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids: set[str] = set()
    for row in load_jsonl_rows(path):
        judge_id = str(row.get("judge_id") or "").strip()
        if judge_id:
            ids.add(judge_id)
    return ids


def canonical_dialogue_turns(row: dict[str, Any]) -> list[dict[str, Any]]:
    raw_turns = row.get("dialogue_turns")
    if not isinstance(raw_turns, list):
        return []

    turns: list[dict[str, Any]] = []
    for turn in raw_turns:
        if not isinstance(turn, dict):
            continue
        turns.append(
            {
                "turn_index": turn.get("turn_index"),
                "user_message": turn.get("user_message"),
                "model_reasoning_trace": str(
                    turn.get("model_reasoning_trace")
                    or turn.get("reasoning")
                    or ""
                ),
                "model_response": str(
                    turn.get("model_response")
                    or turn.get("response")
                    or ""
                ),
            }
        )
    return turns


def build_editor_input_sample(row: dict[str, Any], row_index: int) -> dict[str, Any]:
    judge_id = str(row.get("judge_id") or "").strip() or stable_hash_key(
        str(row.get("benchmark") or row.get("source_dataset") or ""),
        str(row.get("variant_key") or ""),
        str(row.get("id") or row.get("sample_id") or row_index),
    )

    sample: dict[str, Any] = {
        "judge_id": judge_id,
        "benchmark": row.get("benchmark") or row.get("source_dataset"),
        "variant_key": row.get("variant_key"),
        "prompt_text": row.get("prompt") or row.get("prompt_text") or "",
        "model_reasoning_trace": str(
            row.get("model_reasoning_trace")
            or row.get("reasoning")
            or ""
        ),
        "model_response": str(
            row.get("model_response")
            or row.get("response")
            or ""
        ),
        "meta": row.get("meta"),
        "source_scoring": row.get("source_scoring"),
    }

    dialogue_turns = canonical_dialogue_turns(row)
    if dialogue_turns:
        sample["dialogue_turns"] = dialogue_turns
        sample.pop("model_reasoning_trace", None)
        sample.pop("model_response", None)

    return {key: value for key, value in sample.items() if value is not None}


def build_messages(system_prompt: str, sample_payload: dict[str, Any]) -> list[dict[str, str]]:
    user_text = (
        "Edit the following RLHF training sample and return JSON only.\n\n"
        "Input JSON:\n"
        + json.dumps(sample_payload, ensure_ascii=False)
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_text},
    ]


def decode_first_json_object(text: str | None) -> dict[str, Any] | None:
    if not text:
        return None
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text[i:])
        except Exception:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def extract_text_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    chunks.append(text)
        return "".join(chunks)
    return ""


def usage_to_dict(usage: Any) -> dict[str, int | None]:
    if usage is None:
        return {
            "input_tokens": None,
            "output_tokens": None,
            "total_tokens": None,
        }
    return {
        "input_tokens": getattr(usage, "prompt_tokens", None),
        "output_tokens": getattr(usage, "completion_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
    }


def validate_editor_output(
    parsed: dict[str, Any] | None,
    *,
    expected_dialogue_turn_count: int,
) -> list[str]:
    if parsed is None:
        return ["response did not contain a parseable top-level JSON object"]

    errors: list[str] = []
    sample_kind = parsed.get("sample_kind")
    edit_status = parsed.get("edit_status")
    manual_review = parsed.get("manual_review")
    flags = parsed.get("flags")
    flag_explanation = parsed.get("flag_explanation")
    edit_summary = parsed.get("edit_summary")
    single_turn_edit = parsed.get("single_turn_edit")
    dialogue_edit = parsed.get("dialogue_edit")

    if sample_kind not in {"single_turn", "dialogue"}:
        errors.append("sample_kind must be 'single_turn' or 'dialogue'")
    if edit_status not in {"no_edit_needed", "edited"}:
        errors.append("edit_status must be 'no_edit_needed' or 'edited'")
    if not isinstance(manual_review, bool):
        errors.append("manual_review must be boolean")
    if not isinstance(flags, dict):
        errors.append("flags must be an object")
    elif any(not isinstance(v, bool) for v in flags.values()):
        errors.append("all flags values must be boolean")
    if not isinstance(flag_explanation, str) or not flag_explanation.strip():
        errors.append("flag_explanation must be a non-empty string")
    if not isinstance(edit_summary, str) or not edit_summary.strip():
        errors.append("edit_summary must be a non-empty string")

    if sample_kind == "single_turn":
        if dialogue_edit is not None:
            errors.append("dialogue_edit must be null for single_turn outputs")
        if not isinstance(single_turn_edit, dict):
            errors.append("single_turn_edit must be an object for single_turn outputs")
        else:
            if not isinstance(single_turn_edit.get("edited_reasoning_trace"), str):
                errors.append("single_turn_edit.edited_reasoning_trace must be a string")
            if not isinstance(single_turn_edit.get("edited_response"), str):
                errors.append("single_turn_edit.edited_response must be a string")

    if sample_kind == "dialogue":
        if single_turn_edit is not None:
            errors.append("single_turn_edit must be null for dialogue outputs")
        if not isinstance(dialogue_edit, dict):
            errors.append("dialogue_edit must be an object for dialogue outputs")
        else:
            edited_turns = dialogue_edit.get("edited_turns")
            if not isinstance(edited_turns, list):
                errors.append("dialogue_edit.edited_turns must be a list")
            else:
                if expected_dialogue_turn_count and len(edited_turns) != expected_dialogue_turn_count:
                    errors.append(
                        "dialogue_edit.edited_turns length does not match input dialogue_turns length"
                    )
                for idx, turn in enumerate(edited_turns):
                    if not isinstance(turn, dict):
                        errors.append(f"dialogue_edit.edited_turns[{idx}] must be an object")
                        continue
                    if "turn_index" not in turn:
                        errors.append(f"dialogue_edit.edited_turns[{idx}] missing turn_index")
                    if not isinstance(turn.get("edited_reasoning_trace"), str):
                        errors.append(
                            f"dialogue_edit.edited_turns[{idx}].edited_reasoning_trace must be a string"
                        )
                    if not isinstance(turn.get("edited_response"), str):
                        errors.append(
                            f"dialogue_edit.edited_turns[{idx}].edited_response must be a string"
                        )

    return errors


class ChatClient:
    def __init__(self, args: argparse.Namespace):
        timeout = httpx.Timeout(args.request_timeout_s)
        self.provider = args.provider
        self.args = args
        self.default_model = args.model

        if args.provider == "azure_openai":
            endpoint = args.azure_openai_endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT")
            api_key = args.azure_openai_api_key or os.environ.get("AZURE_OPENAI_API_KEY")
            api_version = args.azure_openai_api_version or os.environ.get("AZURE_OPENAI_API_VERSION")
            deployment = args.azure_openai_deployment or args.model
            if not endpoint or not api_key or not api_version:
                raise RuntimeError("Missing Azure OpenAI endpoint/key/api_version configuration.")
            self.model_name = deployment
            self.client = AzureOpenAI(
                azure_endpoint=endpoint,
                api_key=api_key,
                api_version=api_version,
                timeout=timeout,
            )
        elif args.provider == "openrouter":
            api_key = args.openrouter_api_key or os.environ.get("OPENROUTER_API_KEY")
            base_url = (
                args.openrouter_base_url
                or os.environ.get("OPENROUTER_BASE_URL")
                or DEFAULT_OPENROUTER_BASE_URL
            )
            if not api_key:
                raise RuntimeError("Missing OpenRouter API key (OPENROUTER_API_KEY).")
            headers: dict[str, str] = {}
            referer = args.openrouter_http_referer or os.environ.get("OPENROUTER_HTTP_REFERER")
            title = args.openrouter_x_title or os.environ.get("OPENROUTER_X_TITLE")
            if referer:
                headers["HTTP-Referer"] = referer
            if title:
                headers["X-Title"] = title
            self.model_name = args.model
            self.client = OpenAI(
                api_key=api_key,
                base_url=base_url,
                timeout=timeout,
                default_headers=headers or None,
            )
        else:
            raise ValueError(f"Unsupported provider: {args.provider}")

    def complete(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        create_kwargs: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": self.args.max_tokens,
            "temperature": self.args.temperature,
        }
        if self.args.json_mode:
            create_kwargs["response_format"] = {"type": "json_object"}
        if self.provider == "openrouter":
            reasoning_cfg: dict[str, Any] = {}
            if self.args.reasoning_enabled:
                reasoning_cfg["enabled"] = True
                reasoning_cfg["exclude"] = False
            if self.args.reasoning_effort is not None:
                reasoning_cfg["effort"] = self.args.reasoning_effort
            if self.args.reasoning_max_tokens > 0:
                reasoning_cfg["max_tokens"] = self.args.reasoning_max_tokens
            if reasoning_cfg:
                # The OpenAI Python client we use rejects provider-
                # specific top-level kwargs like `reasoning=...` before the
                # request is sent. Pass OpenRouter-specific fields through the
                # raw body extension instead.
                create_kwargs["extra_body"] = {"reasoning": reasoning_cfg}

        resp = self.client.chat.completions.create(**create_kwargs)
        choice = resp.choices[0] if resp.choices else None
        message = choice.message if choice else None
        content = extract_text_content(message.content if message else None)
        return {
            "response_text": content,
            "response_id": getattr(resp, "id", None),
            "finish_reason": getattr(choice, "finish_reason", None),
            "usage": usage_to_dict(getattr(resp, "usage", None)),
        }


def summarize_results(
    rows: list[dict[str, Any]],
    *,
    run_started_at: str,
    run_completed_at: str,
    run_wall_time_sec: float,
    args: argparse.Namespace,
    input_path: Path,
    prompt_path: Path,
    results_path: Path,
) -> dict[str, Any]:
    request_ok = [row for row in rows if row.get("status") == "ok"]
    parse_ok = [row for row in request_ok if row.get("parse_ok")]
    schema_ok = [row for row in request_ok if row.get("schema_ok")]
    manual_review_count = sum(1 for row in schema_ok if row.get("manual_review") is True)

    edit_status_counts = Counter(
        str(row.get("edit_status"))
        for row in schema_ok
        if row.get("edit_status") is not None
    )
    flag_counts: Counter[str] = Counter()
    for row in schema_ok:
        for flag_name in row.get("flags_true", []):
            flag_counts[flag_name] += 1

    input_tokens = 0
    output_tokens = 0
    total_tokens = 0
    rows_with_usage = 0
    latencies: list[float] = []
    for row in request_ok:
        usage = row.get("usage") or {}
        if usage.get("input_tokens") is not None or usage.get("output_tokens") is not None:
            rows_with_usage += 1
            input_tokens += int(usage.get("input_tokens") or 0)
            output_tokens += int(usage.get("output_tokens") or 0)
            total_tokens += int(usage.get("total_tokens") or 0)
        latency = row.get("latency_sec")
        if isinstance(latency, (int, float)):
            latencies.append(float(latency))

    latency_summary = {
        "rows_with_latency": len(latencies),
        "sum_request_latency_sec": round(sum(latencies), 6) if latencies else 0.0,
        "mean_request_latency_sec": round(sum(latencies) / len(latencies), 6) if latencies else None,
        "median_request_latency_sec": round(statistics.median(latencies), 6) if latencies else None,
        "max_request_latency_sec": round(max(latencies), 6) if latencies else None,
    }

    return {
        "run_started_at_utc": run_started_at,
        "run_completed_at_utc": run_completed_at,
        "run_wall_time_sec": round(run_wall_time_sec, 6),
        "provider": args.provider,
        "model": args.model,
        "json_mode": args.json_mode,
        "temperature": args.temperature,
        "reasoning_enabled": args.reasoning_enabled,
        "reasoning_effort": args.reasoning_effort,
        "reasoning_max_tokens": args.reasoning_max_tokens,
        "max_tokens": args.max_tokens,
        "num_threads": args.num_threads,
        "run_label": args.run_label,
        "inputs": {
            "input_jsonl": str(input_path),
            "editor_prompt_txt": str(prompt_path),
        },
        "outputs": {
            "results_jsonl": str(results_path),
        },
        "counts": {
            "rows_total_in_results_file": len(rows),
            "rows_request_ok": len(request_ok),
            "rows_request_failed": len(rows) - len(request_ok),
            "rows_parse_ok": len(parse_ok),
            "rows_schema_ok": len(schema_ok),
            "manual_review_count": manual_review_count,
        },
        "edit_status_counts": dict(edit_status_counts),
        "flag_counts": dict(flag_counts),
        "usage_totals": {
            "rows_with_usage": rows_with_usage,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
        },
        "latency": latency_summary,
    }


async def process_rows(
    *,
    rows: list[dict[str, Any]],
    results_path: Path,
    system_prompt: str,
    args: argparse.Namespace,
) -> None:
    client = ChatClient(args)
    sem = asyncio.Semaphore(args.num_threads)

    async def _one(item: dict[str, Any]) -> dict[str, Any]:
        async with sem:
            started_monotonic = time.monotonic()
            started_at = utc_now()
            sample_payload = item["editor_input_sample"]
            messages = build_messages(system_prompt, sample_payload)

            try:
                response = await asyncio.to_thread(client.complete, messages)
                raw_text = response["response_text"]
                parsed = decode_first_json_object(raw_text)
                dialogue_turns = sample_payload.get("dialogue_turns")
                expected_dialogue_turn_count = len(dialogue_turns) if isinstance(dialogue_turns, list) else 0
                schema_errors = validate_editor_output(
                    parsed,
                    expected_dialogue_turn_count=expected_dialogue_turn_count,
                )
                completed_at = utc_now()
                latency_sec = time.monotonic() - started_monotonic

                flags_obj = parsed.get("flags") if isinstance(parsed, dict) else None
                flags_true = []
                if isinstance(flags_obj, dict):
                    flags_true = sorted(
                        str(key)
                        for key, value in flags_obj.items()
                        if value is True
                    )

                return {
                    **item,
                    "status": "ok",
                    "request_started_at_utc": started_at,
                    "request_completed_at_utc": completed_at,
                    "latency_sec": round(latency_sec, 6),
                    "response_id": response.get("response_id"),
                    "finish_reason": response.get("finish_reason"),
                    "usage": response.get("usage"),
                    "editor_raw": raw_text,
                    "editor_parsed": parsed,
                    "parse_ok": parsed is not None,
                    "schema_ok": not schema_errors,
                    "schema_errors": schema_errors,
                    "manual_review": parsed.get("manual_review") if isinstance(parsed, dict) else None,
                    "edit_status": parsed.get("edit_status") if isinstance(parsed, dict) else None,
                    "flags_true": flags_true,
                    "error": None,
                }
            except Exception as e:
                completed_at = utc_now()
                latency_sec = time.monotonic() - started_monotonic
                return {
                    **item,
                    "status": "error",
                    "request_started_at_utc": started_at,
                    "request_completed_at_utc": completed_at,
                    "latency_sec": round(latency_sec, 6),
                    "response_id": None,
                    "finish_reason": None,
                    "usage": {
                        "input_tokens": None,
                        "output_tokens": None,
                        "total_tokens": None,
                    },
                    "editor_raw": None,
                    "editor_parsed": None,
                    "parse_ok": False,
                    "schema_ok": False,
                    "schema_errors": [],
                    "manual_review": None,
                    "edit_status": None,
                    "flags_true": [],
                    "error": f"{type(e).__name__}: {e}",
                }

    tasks = [asyncio.create_task(_one(item)) for item in rows]
    for fut in asyncio.as_completed(tasks):
        result = await fut
        append_jsonl(results_path, result)
        print(
            "[done] "
            f"judge_id={result['judge_id']} "
            f"status={result['status']} "
            f"parse_ok={result['parse_ok']} "
            f"schema_ok={result['schema_ok']}",
            flush=True,
        )


def main() -> int:
    args = parse_args()

    input_path = Path(args.input_jsonl)
    prompt_path = Path(args.editor_prompt_txt)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input JSONL not found: {input_path}")
    if not prompt_path.exists():
        raise FileNotFoundError(f"Editor prompt not found: {prompt_path}")

    results_path = out_root / "results.jsonl"
    summary_path = out_root / "summary.json"
    config_path = out_root / "run_config.json"

    if args.overwrite and results_path.exists():
        results_path.unlink()
    elif results_path.exists() and not args.resume:
        raise FileExistsError(
            f"Results already exist and --resume is disabled: {results_path}"
        )

    run_started_at = utc_now()
    started_monotonic = time.monotonic()

    system_prompt = prompt_path.read_text(encoding="utf-8").strip()
    input_rows = load_jsonl_rows(input_path)
    if args.max_items > 0:
        input_rows = input_rows[: args.max_items]

    existing_ids = load_existing_result_ids(results_path) if args.resume and results_path.exists() else set()
    work_items: list[dict[str, Any]] = []
    for idx, row in enumerate(input_rows):
        sample_payload = build_editor_input_sample(row, idx)
        judge_id = str(sample_payload["judge_id"])
        if judge_id in existing_ids:
            continue
        work_items.append(
            {
                "row_index": idx,
                "judge_id": judge_id,
                "judge_split": row.get("judge_split"),
                "benchmark": row.get("benchmark") or row.get("source_dataset"),
                "variant_key": row.get("variant_key"),
                "source_run_name": row.get("source_run_name"),
                "source_output_jsonl": row.get("source_output_jsonl"),
                "provider": args.provider,
                "model": args.model,
                "run_label": args.run_label,
                "editor_input_sample": sample_payload,
                "input_payload_chars": len(json.dumps(sample_payload, ensure_ascii=False)),
            }
        )

    run_config = {
        "run_started_at_utc": run_started_at,
        "provider": args.provider,
        "model": args.model,
        "azure_openai_deployment": args.azure_openai_deployment,
        "openrouter_base_url": args.openrouter_base_url or os.environ.get("OPENROUTER_BASE_URL") or DEFAULT_OPENROUTER_BASE_URL,
        "input_jsonl": str(input_path),
        "editor_prompt_txt": str(prompt_path),
        "out_root": str(out_root),
        "max_items": args.max_items,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "reasoning_enabled": args.reasoning_enabled,
        "reasoning_effort": args.reasoning_effort,
        "reasoning_max_tokens": args.reasoning_max_tokens,
        "num_threads": args.num_threads,
        "request_timeout_s": args.request_timeout_s,
        "json_mode": args.json_mode,
        "resume": args.resume,
        "overwrite": args.overwrite,
        "rows_total_input": len(input_rows),
        "rows_skipped_existing": len(existing_ids & {item["judge_id"] for item in work_items}) if existing_ids else 0,
        "rows_to_process": len(work_items),
        "run_label": args.run_label,
    }
    # Correct skipped count after work_items is built.
    run_config["rows_skipped_existing"] = len(input_rows) - len(work_items)
    write_json(config_path, run_config)

    print(f"[info] provider={args.provider} model={args.model}")
    print(f"[info] input_rows={len(input_rows)} rows_to_process={len(work_items)}")
    print(f"[info] prompt={prompt_path}")
    print(f"[info] out_root={out_root}")

    if work_items:
        asyncio.run(
            process_rows(
                rows=work_items,
                results_path=results_path,
                system_prompt=system_prompt,
                args=args,
            )
        )

    all_rows = load_jsonl_rows(results_path) if results_path.exists() else []
    run_completed_at = utc_now()
    summary = summarize_results(
        all_rows,
        run_started_at=run_started_at,
        run_completed_at=run_completed_at,
        run_wall_time_sec=time.monotonic() - started_monotonic,
        args=args,
        input_path=input_path,
        prompt_path=prompt_path,
        results_path=results_path,
    )
    summary["run_config_json"] = str(config_path)
    write_json(summary_path, summary)

    print(f"[done] results={results_path}")
    print(f"[done] summary={summary_path}")
    print(f"[summary] request_ok={summary['counts']['rows_request_ok']} parse_ok={summary['counts']['rows_parse_ok']} schema_ok={summary['counts']['rows_schema_ok']}")
    print(
        "[summary] tokens_in="
        f"{summary['usage_totals']['input_tokens']} "
        "tokens_out="
        f"{summary['usage_totals']['output_tokens']} "
        "total_tokens="
        f"{summary['usage_totals']['total_tokens']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
