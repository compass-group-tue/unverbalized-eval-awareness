#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("SOFT_FILELOCK", "1")
try:
    from filelock import SoftFileLock  # type: ignore
    import filelock  # type: ignore

    filelock.FileLock = SoftFileLock
except Exception:
    pass

from datasets import load_dataset  # type: ignore


DEFAULT_DATASET_ID = "allenai/WildChat-4.8M"
DEFAULT_DEPLOYMENT_DIR = "rlhf_deployment_wildchat_4_8m_v1"


@dataclass
class SelectedRow:
    score: int
    conversation_hash: str
    row: dict[str, Any]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Download and sample deployment-style WildChat prompts for RLHF. "
            "Filters to English single-turn rows, dedupes by conversation_hash, "
            "and writes disjoint no_system/sys_x100 JSONL files."
        )
    )
    p.add_argument("--out-root", required=True)
    p.add_argument("--hf-home", default="")
    p.add_argument("--hf-datasets-cache", default="")
    p.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    p.add_argument("--dataset-config", default="")
    p.add_argument("--dataset-split", default="train")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--per-variant-count", type=int, default=10_000)
    p.add_argument("--target-language", default="English")
    p.add_argument("--require-turn", type=int, default=1)
    p.add_argument(
        "--deployment-dir-name",
        default=DEFAULT_DEPLOYMENT_DIR,
        help="Subdirectory name under data/evaluations/wildchat/ for this download.",
    )
    return p.parse_args()


def stable_score(seed: int, conversation_hash: str) -> int:
    digest = hashlib.sha256(f"{seed}:{conversation_hash}".encode("utf-8")).digest()
    return int.from_bytes(digest[:16], byteorder="big", signed=False)


def first_user_prompt(row: dict[str, Any]) -> str:
    conversation = row.get("conversation")
    if isinstance(conversation, list):
        for turn in conversation:
            if not isinstance(turn, dict):
                continue
            role = str(turn.get("role") or turn.get("speaker") or "").strip().lower()
            content = str(turn.get("content") or turn.get("text") or turn.get("message") or "").strip()
            if content and role in {"user", "human"}:
                return content
        for turn in conversation:
            if not isinstance(turn, dict):
                continue
            content = str(turn.get("content") or turn.get("text") or turn.get("message") or "").strip()
            if content:
                return content
    prompt = str(row.get("prompt") or "").strip()
    if prompt:
        return prompt
    return ""


def canonicalize_row(row: dict[str, Any]) -> dict[str, Any]:
    conversation_hash = str(row.get("conversation_hash") or row.get("conversation_id") or row.get("id") or "").strip()
    if not conversation_hash:
        raise ValueError("row missing conversation_hash/conversation_id/id")

    prompt = first_user_prompt(row)
    if not prompt:
        raise ValueError(f"row {conversation_hash} missing usable first user prompt")

    return {
        "id": conversation_hash,
        "source_type": "deployment",
        "source_dataset": "wildchat",
        "unit_type": "single_turn_chat",
        "prompt": prompt,
        "messages": [{"role": "user", "content": prompt}],
        "meta": {
            "conversation_hash": row.get("conversation_hash"),
            "conversation_id": row.get("conversation_id"),
            "model": row.get("model"),
            "timestamp": row.get("timestamp"),
            "turn": row.get("turn"),
            "language": row.get("language"),
            "country": row.get("country"),
            "state": row.get("state"),
            "redacted": row.get("redacted"),
            "toxic": row.get("toxic"),
            "hashed_ip": row.get("hashed_ip"),
            "header": row.get("header"),
            "conversation": row.get("conversation"),
            "openai_moderation": row.get("openai_moderation"),
            "detoxify_moderation": row.get("detoxify_moderation"),
        },
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(make_json_safe(row), ensure_ascii=False) + "\n")


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(make_json_safe(obj), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def make_json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): make_json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [make_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [make_json_safe(v) for v in value]
    if isinstance(value, (dt.datetime, dt.date, dt.time)):
        return value.isoformat()
    return value


def main() -> int:
    args = parse_args()

    if args.hf_home:
        os.environ["HF_HOME"] = args.hf_home
    if args.hf_datasets_cache:
        os.environ["HF_DATASETS_CACHE"] = args.hf_datasets_cache

    out_dir = Path(args.out_root) / "wildchat" / args.deployment_dir_name
    total_target = args.per_variant_count * 2

    print(f"[info] dataset_id={args.dataset_id}")
    print(f"[info] dataset_config={args.dataset_config or '<default>'}")
    print(f"[info] dataset_split={args.dataset_split}")
    print(f"[info] target_language={args.target_language}")
    print(f"[info] require_turn={args.require_turn}")
    print(f"[info] per_variant_count={args.per_variant_count}")
    print(f"[info] total_target={total_target}")
    print(f"[info] out_dir={out_dir}")

    dataset = load_dataset(
        args.dataset_id,
        args.dataset_config or None,
        split=args.dataset_split,
        streaming=True,
    )

    seen_hashes: set[str] = set()
    selected: list[SelectedRow] = []

    stats = {
        "rows_read": 0,
        "kept_candidates": 0,
        "skipped_wrong_language": 0,
        "skipped_wrong_turn": 0,
        "skipped_duplicate_hash": 0,
        "skipped_missing_hash": 0,
        "skipped_bad_prompt": 0,
    }

    progress_every = 1_000
    progress_every_seconds = 30.0
    start_time = time.monotonic()
    last_progress_time = start_time
    first_match_reported = False

    for row in dataset:
        stats["rows_read"] += 1

        conversation_hash = str(
            row.get("conversation_hash") or row.get("conversation_id") or row.get("id") or ""
        ).strip()
        if not conversation_hash:
            stats["skipped_missing_hash"] += 1
            continue

        language = str(row.get("language") or "").strip()
        if language != args.target_language:
            stats["skipped_wrong_language"] += 1
            continue

        turn = row.get("turn")
        if turn != args.require_turn:
            stats["skipped_wrong_turn"] += 1
            continue

        if conversation_hash in seen_hashes:
            stats["skipped_duplicate_hash"] += 1
            continue

        try:
            prepared = canonicalize_row(dict(row))
        except Exception:
            stats["skipped_bad_prompt"] += 1
            continue

        seen_hashes.add(conversation_hash)
        if not first_match_reported:
            print(
                "[progress] first_match "
                f"rows_read={stats['rows_read']} "
                f"conversation_hash={conversation_hash} "
                f"prompt_chars={len(prepared['prompt'])}"
            )
            first_match_reported = True
        score = stable_score(args.seed, conversation_hash)
        selected.append(
            SelectedRow(
                score=score,
                conversation_hash=conversation_hash,
                row=prepared,
            )
        )
        stats["kept_candidates"] += 1

        now = time.monotonic()
        if (
            stats["rows_read"] % progress_every == 0
            or now - last_progress_time >= progress_every_seconds
        ):
            print(
                "[progress] "
                f"rows_read={stats['rows_read']} "
                f"unique_eligible_seen={len(seen_hashes)} "
                f"selected_so_far={len(selected)} "
                f"skipped_wrong_language={stats['skipped_wrong_language']} "
                f"skipped_wrong_turn={stats['skipped_wrong_turn']} "
                f"skipped_duplicate_hash={stats['skipped_duplicate_hash']}"
            )
            last_progress_time = now

        if len(selected) >= total_target:
            print(
                "[done] reached target unique eligible rows "
                f"rows_read={stats['rows_read']} selected={len(selected)}"
            )
            break
    selected.sort(key=lambda x: x.score)

    if len(selected) < total_target:
        raise RuntimeError(
            f"Only found {len(selected)} qualifying unique rows, needed {total_target}."
        )

    no_system_rows = [x.row for x in selected[: args.per_variant_count]]
    sys_x100_rows = [x.row for x in selected[args.per_variant_count : total_target]]
    combined_rows = [x.row for x in selected[:total_target]]

    combined_path = out_dir / f"wildchat_source_selected_n{total_target}_seed{args.seed}.jsonl"
    no_system_path = out_dir / f"wildchat_deployment_no_system_n{args.per_variant_count}_seed{args.seed}.jsonl"
    sys_x100_path = out_dir / f"wildchat_deployment_sys_x100_n{args.per_variant_count}_seed{args.seed}.jsonl"
    manifest_path = out_dir / "manifest.json"

    write_jsonl(combined_path, combined_rows)
    write_jsonl(no_system_path, no_system_rows)
    write_jsonl(sys_x100_path, sys_x100_rows)

    manifest = {
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset_id": args.dataset_id,
        "dataset_config": args.dataset_config or None,
        "dataset_split": args.dataset_split,
        "target_language": args.target_language,
        "require_turn": args.require_turn,
        "seed": args.seed,
        "per_variant_count": args.per_variant_count,
        "total_target": total_target,
        "selection_policy": {
            "dedupe_key": "conversation_hash",
            "single_turn_only": True,
            "sampling": "lowest sha256(seed:conversation_hash) scores after filtering",
            "variant_sets": "disjoint",
        },
        "stats": stats,
        "unique_eligible_seen": len(seen_hashes),
        "selected_unique_hashes": len({row["id"] for row in combined_rows}),
        "outputs": {
            "combined_path": str(combined_path),
            "no_system_path": str(no_system_path),
            "sys_x100_path": str(sys_x100_path),
        },
    }
    write_json(manifest_path, manifest)

    print(f"[done] combined={combined_path}")
    print(f"[done] no_system={no_system_path}")
    print(f"[done] sys_x100={sys_x100_path}")
    print(f"[done] manifest={manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
