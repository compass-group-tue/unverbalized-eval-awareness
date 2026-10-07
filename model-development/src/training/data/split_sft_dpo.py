#!/usr/bin/env python3
"""Create a disjoint chosen-SFT subset and marker-contrast DPO complement.

The special-token v2 DPO view contains two rows for every v1 source pair: the
natural preference pair and a rejected-visible-marker counterfactual.  This
builder selects whole *v1 source-pair IDs* for chosen-only SFT and removes both
of their v2 views from the DPO training complement.  It deliberately leaves
the frozen v1/v2 validation split untouched.

The SFT rows retain the original ``chosen_segments`` representation so the
trajectory-aware SFT trainer can preserve dialogue history and supervise only
messages marked ``_dpo_trainable``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


COUNTERFACTUAL_SUFFIX = "::marker_counterfactual_v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def stable_hash(*parts: str) -> str:
    return hashlib.sha256("||".join(parts).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            item = json.loads(line)
            if not isinstance(item, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            yield item


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def base_pair_id(pair_id: str) -> str:
    if pair_id.endswith(COUNTERFACTUAL_SUFFIX):
        return pair_id[: -len(COUNTERFACTUAL_SUFFIX)]
    return pair_id


def require_v1_row(row: dict[str, Any], *, path: Path) -> str:
    pair_id = str(row.get("pair_id") or "")
    if not pair_id:
        raise ValueError(f"{path}: v1 row has no pair_id")
    if pair_id != base_pair_id(pair_id):
        raise ValueError(f"{path}: expected original v1 pair_id, got {pair_id!r}")
    if not isinstance(row.get("chosen_segments"), list) or not row["chosen_segments"]:
        raise ValueError(f"{path}: {pair_id}: chosen_segments is missing or empty")
    return pair_id


def stratum(row: dict[str, Any]) -> tuple[str, str]:
    split = str(row.get("split") or "")
    kind = str(row.get("sample_kind") or "")
    if split not in {"evaluation", "deployment"}:
        raise ValueError(f"{row.get('pair_id')}: unsupported split {split!r}")
    if kind not in {"single_turn", "dialogue"}:
        raise ValueError(f"{row.get('pair_id')}: unsupported sample_kind {kind!r}")
    return split, kind


def stratified_select(rows: list[dict[str, Any]], *, count: int, seed: int) -> tuple[set[str], dict[str, int]]:
    if count <= 0:
        raise ValueError("--sft-source-pairs must be positive")
    if count > len(rows):
        raise ValueError(f"--sft-source-pairs={count} exceeds the {len(rows)} available v1 train pairs")

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[stratum(row)].append(row)

    total = len(rows)
    quotas: dict[tuple[str, str], int] = {}
    remainders: list[tuple[float, tuple[str, str]]] = []
    allocated = 0
    for key in sorted(grouped):
        ideal = count * len(grouped[key]) / total
        quota = int(ideal)
        quotas[key] = quota
        allocated += quota
        remainders.append((ideal - quota, key))
    for _, key in sorted(remainders, key=lambda item: (-item[0], item[1]))[: count - allocated]:
        quotas[key] += 1

    selected: set[str] = set()
    audit: dict[str, int] = {}
    for key in sorted(grouped):
        candidates = sorted(
            grouped[key],
            key=lambda row: stable_hash(str(seed), "sft_split", key[0], key[1], str(row["pair_id"])),
        )
        chosen = candidates[: quotas[key]]
        selected.update(str(row["pair_id"]) for row in chosen)
        label = f"{key[0]}:{key[1]}"
        audit[f"available:{label}"] = len(candidates)
        audit[f"selected:{label}"] = len(chosen)
    if len(selected) != count:
        raise AssertionError(f"Selected {len(selected)} source IDs, expected {count}")
    return selected, audit


def make_sft_row(row: dict[str, Any]) -> dict[str, Any]:
    pair_id = str(row["pair_id"])
    return {
        "sft_id": pair_id,
        "source_pair_id": pair_id,
        "split": row["split"],
        "sample_kind": row["sample_kind"],
        "benchmark": row.get("benchmark"),
        "variant_key": row.get("variant_key"),
        "source_run_name": row.get("source_run_name"),
        "source_output_jsonl": row.get("source_output_jsonl"),
        "chosen_segments": row["chosen_segments"],
    }


def trainable_text_stats(rows: Iterable[dict[str, Any]]) -> dict[str, int]:
    stats: Counter[str] = Counter()
    for row in rows:
        for segment in row.get("chosen_segments") or []:
            if not isinstance(segment, list):
                continue
            for message in segment:
                if not isinstance(message, dict):
                    continue
                if message.get("role") != "assistant" or message.get("_dpo_trainable") is not True:
                    continue
                stats["trainable_assistant_messages"] += 1
                stats["reasoning_characters"] += len(str(message.get("reasoning_content") or ""))
                stats["visible_response_characters"] += len(str(message.get("content") or ""))
    return dict(stats)


def review_markdown(selected: list[dict[str, Any]], complement: list[dict[str, Any]]) -> str:
    chunks = ["# SFT/DPO split review", ""]
    for title, rows in (("Chosen SFT samples", selected), ("DPO-complement samples", complement)):
        chunks.extend([f"## {title}", ""])
        for row in rows[:3]:
            chunks.extend(
                [
                    f"### {row.get('pair_id', row.get('sft_id'))}",
                    "```json",
                    json.dumps(row, ensure_ascii=False, indent=2),
                    "```",
                    "",
                ]
            )
    return "\n".join(chunks)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v1-train", required=True)
    parser.add_argument("--v1-validation", required=True)
    parser.add_argument("--v2-train", required=True)
    parser.add_argument("--v2-validation", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sft-source-pairs", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = {name: Path(getattr(args, name.replace("-", "_"))) for name in ("v1-train", "v1-validation", "v2-train", "v2-validation")}
    output_dir = Path(args.output_dir)
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output_dir}")

    v1_train = list(iter_jsonl(paths["v1-train"]))
    v1_validation = list(iter_jsonl(paths["v1-validation"]))
    train_by_id: dict[str, dict[str, Any]] = {}
    for row in v1_train:
        pair_id = require_v1_row(row, path=paths["v1-train"])
        if pair_id in train_by_id:
            raise ValueError(f"{paths['v1-train']}: duplicate v1 pair_id {pair_id!r}")
        train_by_id[pair_id] = row
    validation_ids: set[str] = set()
    for row in v1_validation:
        pair_id = require_v1_row(row, path=paths["v1-validation"])
        if pair_id in train_by_id or pair_id in validation_ids:
            raise ValueError(f"v1 train/validation source-ID overlap at {pair_id!r}")
        validation_ids.add(pair_id)

    selected_ids, selection_audit = stratified_select(
        list(train_by_id.values()), count=args.sft_source_pairs, seed=args.seed
    )
    selected_rows = [train_by_id[pair_id] for pair_id in sorted(selected_ids)]
    sft_train = [make_sft_row(row) for row in selected_rows]
    sft_validation = [make_sft_row(row) for row in v1_validation]

    v2_train = list(iter_jsonl(paths["v2-train"]))
    v2_validation = list(iter_jsonl(paths["v2-validation"]))
    v2_train_by_base: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in v2_train:
        pair_id = str(row.get("pair_id") or "")
        if not pair_id:
            raise ValueError(f"{paths['v2-train']}: row missing pair_id")
        base_id = base_pair_id(pair_id)
        if base_id not in train_by_id:
            raise ValueError(f"{paths['v2-train']}: {pair_id}: base ID is absent from v1 train")
        v2_train_by_base[base_id].append(row)
    for base_id, rows in v2_train_by_base.items():
        expected = {base_id, base_id + COUNTERFACTUAL_SUFFIX}
        found = {str(row["pair_id"]) for row in rows}
        if found != expected:
            raise ValueError(f"{paths['v2-train']}: {base_id}: expected exactly natural/counterfactual v2 views, got {sorted(found)}")
        if rows[0].get("chosen_segments") != train_by_id[base_id].get("chosen_segments"):
            raise ValueError(f"{paths['v2-train']}: {base_id}: natural chosen trajectory no longer matches v1")

    complement_ids = set(train_by_id) - selected_ids
    dpo_train = [row for row in v2_train if base_pair_id(str(row["pair_id"])) in complement_ids]
    if len(dpo_train) != 2 * len(complement_ids):
        raise AssertionError(f"DPO complement has {len(dpo_train)} rows, expected {2 * len(complement_ids)}")

    validation_base_ids: set[str] = set()
    v2_validation_by_base: dict[str, set[str]] = defaultdict(set)
    for row in v2_validation:
        pair_id = str(row.get("pair_id") or "")
        base_id = base_pair_id(pair_id)
        if base_id not in validation_ids:
            raise ValueError(f"{paths['v2-validation']}: {pair_id}: base ID is absent from v1 validation")
        validation_base_ids.add(base_id)
        v2_validation_by_base[base_id].add(pair_id)
    if validation_base_ids != validation_ids:
        raise ValueError("v2 validation does not cover exactly the frozen v1 validation source IDs")
    for base_id, pair_ids in v2_validation_by_base.items():
        expected = {base_id, base_id + COUNTERFACTUAL_SUFFIX}
        if pair_ids != expected:
            raise ValueError(
                f"{paths['v2-validation']}: {base_id}: expected exactly natural/counterfactual "
                f"v2 views, got {sorted(pair_ids)}"
            )
    if set(selected_ids) & validation_base_ids or complement_ids & validation_base_ids:
        raise AssertionError("train/validation source IDs leaked across the split")

    temporary_dir = output_dir.with_name(output_dir.name + f".tmp.{os.getpid()}")
    if temporary_dir.exists():
        raise FileExistsError(f"Temporary output directory already exists: {temporary_dir}")
    temporary_dir.mkdir(parents=True)
    try:
        write_jsonl(temporary_dir / "sft" / "sft_train.jsonl", sft_train)
        write_jsonl(temporary_dir / "sft" / "sft_validation.jsonl", sft_validation)
        write_jsonl(temporary_dir / "dpo" / "dpo_train.jsonl", dpo_train)
        write_jsonl(temporary_dir / "dpo" / "dpo_validation.jsonl", v2_validation)
        source_dir = temporary_dir / "source"
        source_dir.mkdir()
        for name, path in paths.items():
            shutil.copy2(path, source_dir / f"{name}.jsonl")
        write_json(
            temporary_dir / "manifests" / "split_manifest.json",
            {
                "created_at_utc": utc_now(),
                "builder": Path(__file__).name,
                "seed": args.seed,
                "sft_source_pairs": args.sft_source_pairs,
                "source_paths": {name: str(path) for name, path in paths.items()},
                "source_sha256": {name: file_sha256(path) for name, path in paths.items()},
                "selection": {
                    "unit": "original v1 source pair ID",
                    "strategy": "stable-hash stratified by split and sample_kind",
                    "selected_base_pair_ids": sorted(selected_ids),
                    "dpo_complement_base_pair_ids": sorted(complement_ids),
                    "strata": selection_audit,
                },
                "outputs": {
                    "sft_train": "sft/sft_train.jsonl",
                    "sft_validation": "sft/sft_validation.jsonl",
                    "dpo_train": "dpo/dpo_train.jsonl",
                    "dpo_validation": "dpo/dpo_validation.jsonl",
                },
                "counts": {
                    "v1_train_source_pairs": len(train_by_id),
                    "sft_train_source_pairs": len(sft_train),
                    "dpo_complement_source_pairs": len(complement_ids),
                    "dpo_complement_rows": len(dpo_train),
                    "sft_validation_source_pairs": len(sft_validation),
                    "dpo_validation_rows": len(v2_validation),
                },
                "trainable_text_stats": {
                    "sft_train": trainable_text_stats(sft_train),
                    "sft_validation": trainable_text_stats(sft_validation),
                },
                "invariants": {
                    "sft_and_dpo_train_source_ids_disjoint": True,
                    "every_dpo_complement_source_has_natural_and_counterfactual_views": True,
                    "v1_validation_unchanged": True,
                    "v2_validation_unchanged": True,
                },
            },
        )
        write_json(
            temporary_dir / "reports" / "split_audit.json",
            {
                "selection_strata": selection_audit,
                "sft_train_text": trainable_text_stats(sft_train),
                "sft_validation_text": trainable_text_stats(sft_validation),
                "dpo_complement_rows": len(dpo_train),
            },
        )
        review_dir = temporary_dir / "review"
        review_dir.mkdir()
        review_dir.joinpath("split_samples.md").write_text(
            review_markdown(selected_rows, dpo_train), encoding="utf-8"
        )
        temporary_dir.rename(output_dir)
    except Exception:
        print(f"[sft-dpo-split] failed; partial output remains at {temporary_dir}", file=sys.stderr)
        raise

    print(f"[sft-dpo-split] output_dir={output_dir}")
    print(f"[sft-dpo-split] sft_source_pairs={len(sft_train)} dpo_rows={len(dpo_train)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
