#!/usr/bin/env python3
"""Freeze a filtered WildChat deployment input set for a final experiment.

To avoid downloading/scanning the whole pinned corpus, the preparer chooses a
small deterministic random subset of its Parquet files, filters those rows,
then selects the lowest hash priorities over the eligible candidate pool. This
is a bounded approximate random sample of the post-cutoff corpus; the exact
source-file scope is recorded in the frozen-input manifest.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import heapq
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("SOFT_FILELOCK", "1")
try:
    from filelock import SoftFileLock
    import filelock

    filelock.FileLock = SoftFileLock
except Exception:
    pass

from datasets import load_dataset


DEFAULT_DATASET_ID = "allenai/WildChat-4.8M"
# Explicitly pin the public revision inspected for this protocol.
DEFAULT_REVISION = "c827c6df8fcf008219ffaffa4d1dd77491099367"
DEFAULT_CUTOFF = "2024-04-30T00:00:00Z"
TOTAL_PARQUET_SHARDS = 86


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--split", default="train")
    parser.add_argument("--cutoff-utc", default=DEFAULT_CUTOFF)
    parser.add_argument("--target-language", default="English")
    parser.add_argument("--sample-size", type=int, default=2000)
    parser.add_argument(
        "--max-user-prompt-tokens",
        type=int,
        default=0,
        help=(
            "Maximum tokens in the retained raw user message, measured without "
            "special tokens. Set to 0 to disable this filter."
        ),
    )
    parser.add_argument(
        "--tokenizer-path",
        default="",
        help="Required when --max-user-prompt-tokens is positive.",
    )
    parser.add_argument(
        "--random-parquet-shards",
        type=int,
        default=6,
        help=(
            "Number of WildChat Parquet files chosen uniformly without replacement "
            "from the pinned source. The default six files are a bounded candidate pool "
            "for the 2,000-row sample."
        ),
    )
    parser.add_argument(
        "--shard-selection-seed",
        type=int,
        default=42,
        help="Seed used only to choose source Parquet files.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hf-home", default="")
    parser.add_argument("--hf-datasets-cache", default="")
    parser.add_argument("--max-rows", type=int, default=0, help="Optional smoke-test scan limit.")
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def json_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_timestamp(value: Any) -> dt.datetime | None:
    if isinstance(value, dt.datetime):
        parsed = value
    elif isinstance(value, str) and value.strip():
        raw = value.strip().replace("Z", "+00:00")
        try:
            parsed = dt.datetime.fromisoformat(raw)
        except ValueError:
            return None
    else:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def canonical_timestamp(value: Any) -> str:
    parsed = parse_timestamp(value)
    if parsed is None:
        raise ValueError("missing or invalid timestamp")
    return parsed.isoformat().replace("+00:00", "Z")


def as_message_list(row: dict[str, Any]) -> list[dict[str, str]] | None:
    conversation = row.get("conversation")
    if not isinstance(conversation, list) or len(conversation) != 2:
        return None
    out: list[dict[str, str]] = []
    for item in conversation:
        if not isinstance(item, dict):
            return None
        role = str(item.get("role") or "").strip().lower()
        content = str(item.get("content") or "").strip()
        if not content:
            return None
        out.append({"role": role, "content": content})
    if [message["role"] for message in out] != ["user", "assistant"]:
        return None
    return out


def priority(seed: int, conversation_hash: str) -> int:
    digest = hashlib.sha256(f"{seed}:{conversation_hash}".encode("utf-8")).digest()
    return int.from_bytes(digest[:16], byteorder="big", signed=False)


def canonical_row(
    row: dict[str, Any], *, source_dataset_id: str, source_revision: str
) -> dict[str, Any]:
    conversation_hash = str(row.get("conversation_hash") or "").strip()
    messages = as_message_list(row)
    if not conversation_hash or messages is None:
        raise ValueError("invalid canonical WildChat row")
    user_prompt = messages[0]["content"]
    return {
        "id": conversation_hash,
        "source_type": "deployment",
        "source_dataset": "wildchat_4_8m",
        "group": "deployment_single_turn",
        "unit_type": "single_turn_chat",
        "prompt": user_prompt,
        "base_messages": [{"role": "user", "content": user_prompt}],
        "system_overlay_mode": "prepend_system",
        "meta": {
            "source_dataset_id": source_dataset_id,
            "source_revision": source_revision,
            "conversation_hash": conversation_hash,
            "timestamp": canonical_timestamp(row.get("timestamp")),
            "language": str(row.get("language") or ""),
            "turn": row.get("turn"),
            "source_model": row.get("model"),
        },
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def random_parquet_urls(
    *, dataset_id: str, revision: str, shard_count: int, shard_selection_seed: int
) -> list[str]:
    if not 1 <= shard_count <= TOTAL_PARQUET_SHARDS:
        raise ValueError(
            "--random-parquet-shards must be in "
            f"[1, {TOTAL_PARQUET_SHARDS}], got {shard_count}"
        )
    indices = sorted(
        random.Random(shard_selection_seed).sample(range(TOTAL_PARQUET_SHARDS), shard_count)
    )
    base = f"https://huggingface.co/datasets/{dataset_id}/resolve/{revision}/data"
    return [
        f"{base}/train-{index:05d}-of-{TOTAL_PARQUET_SHARDS:05d}.parquet"
        for index in indices
    ]


def main() -> int:
    args = parse_args()
    if args.sample_size < 1:
        raise ValueError("--sample-size must be positive")
    if args.max_rows < 0:
        raise ValueError("--max-rows must be non-negative")
    if args.max_user_prompt_tokens < 0:
        raise ValueError("--max-user-prompt-tokens must be non-negative")
    if args.max_user_prompt_tokens and not args.tokenizer_path:
        raise ValueError("--tokenizer-path is required when --max-user-prompt-tokens is positive")
    cutoff = parse_timestamp(args.cutoff_utc)
    if cutoff is None:
        raise ValueError(f"Invalid --cutoff-utc: {args.cutoff_utc!r}")

    if args.hf_home:
        os.environ["HF_HOME"] = args.hf_home
    if args.hf_datasets_cache:
        os.environ["HF_DATASETS_CACHE"] = args.hf_datasets_cache
    root = Path(args.run_root)
    input_dir = root / "inputs" / "wildchat"
    output_path = input_dir / "samples.jsonl"
    manifest_path = root / "inputs" / "manifest.json"
    if (output_path.exists() or manifest_path.exists()) and not args.overwrite:
        raise FileExistsError(
            f"Refusing to modify existing frozen inputs under {root}; use a new run root or --overwrite."
        )

    stats = {
        "rows_read": 0,
        "eligible_rows": 0,
        "skipped_before_cutoff": 0,
        "skipped_wrong_language": 0,
        "skipped_wrong_turn": 0,
        "skipped_not_exact_user_assistant": 0,
        "skipped_duplicate_id": 0,
        "skipped_bad_timestamp": 0,
        "skipped_user_prompt_over_token_limit": 0,
    }
    tokenizer = None
    if args.max_user_prompt_tokens:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, local_files_only=True)
    # Only identifiers are held for de-duplication; original conversations and
    # their assistant replies are never written to the final experiment root.
    seen_ids: set[str] = set()
    heap: list[tuple[int, str, dict[str, Any]]] = []
    source_files = random_parquet_urls(
        dataset_id=args.dataset_id,
        revision=args.revision,
        shard_count=args.random_parquet_shards,
        shard_selection_seed=args.shard_selection_seed,
    )
    print(
        f"[info] streaming_random_parquet_shards={args.random_parquet_shards}/{TOTAL_PARQUET_SHARDS} "
        f"files={','.join(Path(path).name for path in source_files)}",
        flush=True,
    )
    stream = load_dataset(
        "parquet",
        data_files={"train": source_files},
        split=args.split,
        streaming=True,
    )
    started = time.monotonic()
    for source in stream:
        stats["rows_read"] += 1
        if args.max_rows and stats["rows_read"] > args.max_rows:
            break
        row = dict(source)
        timestamp = parse_timestamp(row.get("timestamp"))
        if timestamp is None:
            stats["skipped_bad_timestamp"] += 1
            continue
        if timestamp < cutoff:
            stats["skipped_before_cutoff"] += 1
            continue
        if str(row.get("language") or "").strip() != args.target_language:
            stats["skipped_wrong_language"] += 1
            continue
        if row.get("turn") != 1:
            stats["skipped_wrong_turn"] += 1
            continue
        messages = as_message_list(row)
        if messages is None:
            stats["skipped_not_exact_user_assistant"] += 1
            continue
        if tokenizer is not None:
            user_prompt_tokens = len(
                tokenizer(messages[0]["content"], add_special_tokens=False)["input_ids"]
            )
            if user_prompt_tokens > args.max_user_prompt_tokens:
                stats["skipped_user_prompt_over_token_limit"] += 1
                continue
        conversation_hash = str(row.get("conversation_hash") or "").strip()
        if not conversation_hash:
            stats["skipped_duplicate_id"] += 1
            continue
        if conversation_hash in seen_ids:
            stats["skipped_duplicate_id"] += 1
            continue
        seen_ids.add(conversation_hash)
        prepared = canonical_row(
            row,
            source_dataset_id=args.dataset_id,
            source_revision=args.revision,
        )
        stats["eligible_rows"] += 1
        score = priority(args.seed, conversation_hash)
        entry = (-score, conversation_hash, prepared)
        if len(heap) < args.sample_size:
            heapq.heappush(heap, entry)
        elif score < -heap[0][0]:
            heapq.heapreplace(heap, entry)
        if stats["rows_read"] % 10_000 == 0:
            print(
                f"[progress] rows_read={stats['rows_read']} eligible={stats['eligible_rows']} "
                f"retained={len(heap)} elapsed_s={time.monotonic() - started:.1f}",
                flush=True,
            )

    if len(heap) != args.sample_size:
        raise RuntimeError(
            f"Only retained {len(heap)} eligible rows after scanning {stats['rows_read']}; "
            f"need {args.sample_size}."
        )
    selected = sorted(
        ((-negative_score, conversation_hash, prepared) for negative_score, conversation_hash, prepared in heap),
        key=lambda item: (item[0], item[1]),
    )
    rows = [prepared for _, _, prepared in selected]
    write_jsonl(output_path, rows)
    selected_ids_sha256 = hashlib.sha256(",".join(row["id"] for row in rows).encode("utf-8")).hexdigest()
    manifest = {
        "experiment_type": "final_deployment_generation",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset": {
            "id": args.dataset_id,
            "revision": args.revision,
            "split": args.split,
            "cutoff_utc": cutoff.isoformat().replace("+00:00", "Z"),
            "target_language": args.target_language,
            "stream_scope": {
                "method": "uniform_random_parquet_files",
                "total_source_parquet_shards": TOTAL_PARQUET_SHARDS,
                "random_parquet_shards": args.random_parquet_shards,
                "shard_selection_seed": args.shard_selection_seed,
                "files": source_files,
            },
        },
        "filters": {
            "top_level_turn_equals": 1,
            "conversation_shape": "exactly two non-empty messages: user then assistant",
            "retained_input": "original user message only",
            "max_user_prompt_tokens": args.max_user_prompt_tokens or None,
            "prompt_tokenizer_path": args.tokenizer_path or None,
            "prompt_tokenization": "raw user message; add_special_tokens=false",
        },
        "selection": {
            "method": "lowest_sha256_priority_over_all_eligible_rows_in_stream_scope",
            "seed": args.seed,
            "sample_size": args.sample_size,
            "selected_ids_sha256": selected_ids_sha256,
        },
        "stats": stats,
        "outputs": {
            "samples_path": str(output_path),
            "samples_sha256": json_sha256(output_path),
        },
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[done] rows={len(rows)} samples={output_path}", flush=True)
    print(f"[done] manifest={manifest_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
