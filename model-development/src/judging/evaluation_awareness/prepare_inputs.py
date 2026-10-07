#!/usr/bin/env python3
"""Freeze reasoning-only inputs for the binary evaluation-awareness judge.

This adapter intentionally reads the primary saved transcript JSONLs rather
than Inspect logs. Single-turn datasets yield one judge item per saved row;
AgentHarm yields one item per explicitly exported ``assistant_turns`` entry.
The judge-facing JSONL contains only an opaque ID and reasoning text. A private
source index preserves the task/turn mapping for later aggregation.

An optional frozen sample manifest restricts each label to a pre-selected set of
source IDs. The selection is made once, over frozen inputs, by
``build_judge_sample_manifest.py``; this adapter only filters. Because the
selection is model-independent, the same manifest reproduces the identical rows
against a trained model's transcripts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


LABEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        action="append",
        metavar="LABEL=OUTPUT_JSONL",
        help="Repeatable single-file source transcript. LABEL is private local metadata.",
    )
    parser.add_argument(
        "--source-dir",
        action="append",
        metavar="LABEL=OUTPUT_DIRECTORY",
        help=(
            "Repeatable output directory. All top-level *.jsonl files are read "
            "in lexicographic order; LABEL is private local metadata."
        ),
    )
    parser.add_argument(
        "--sample-manifest",
        default="",
        help=(
            "Optional frozen judge sample manifest. When supplied, every source label must "
            "appear in it and only its selected source IDs are prepared."
        ),
    )
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_jsonl(path: Path, rows: list[dict[str, Any]], *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_source(value: str) -> tuple[str, Path]:
    label, separator, raw_path = value.partition("=")
    if not separator or not LABEL_RE.fullmatch(label):
        raise ValueError(f"Invalid --source {value!r}; expected LABEL=OUTPUT_JSONL")
    path = Path(raw_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    return label, path


def parse_source_dir(value: str) -> tuple[str, list[Path]]:
    label, separator, raw_path = value.partition("=")
    if not separator or not LABEL_RE.fullmatch(label):
        raise ValueError(
            f"Invalid --source-dir {value!r}; expected LABEL=OUTPUT_DIRECTORY"
        )
    directory = Path(raw_path)
    if not directory.is_dir():
        raise NotADirectoryError(directory)
    paths = sorted(path for path in directory.glob("*.jsonl") if path.is_file())
    if not paths:
        raise FileNotFoundError(f"No top-level *.jsonl files in {directory}")
    return label, paths


def opaque_judge_id(*parts: str) -> str:
    return "ea_" + hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()[:24]


def marker(value: Any) -> bool:
    return "§" in str(value or "")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            row = json.loads(raw)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: expected JSON object")
            rows.append(row)
    return rows


def iter_reasoning_units(*, source_path: Path, row: dict[str, Any]) -> list[dict[str, Any]]:
    source_id = str(row.get("id") or "").strip()
    if not source_id:
        raise ValueError(f"{source_path}: source row missing id")
    turns = row.get("assistant_turns")
    if turns is None:
        return [{
            "source_id": source_id,
            "assistant_turn_index": 0,
            "is_final_assistant_turn": True,
            "response": str(row.get("response") or ""),
            "reasoning": str(row.get("reasoning") or ""),
            "unit_kind": "single_turn",
        }]
    if not isinstance(turns, list) or not turns:
        raise ValueError(
            f"{source_path}:{source_id}: AgentHarm output lacks explicit assistant_turns; "
            "rerun with the current transcript exporter rather than reading Inspect logs."
        )
    units: list[dict[str, Any]] = []
    for fallback_index, turn in enumerate(turns):
        if not isinstance(turn, dict):
            raise ValueError(f"{source_path}:{source_id}: non-object assistant turn")
        index = turn.get("assistant_turn_index", fallback_index)
        if not isinstance(index, int) or index != fallback_index:
            raise ValueError(f"{source_path}:{source_id}: invalid assistant_turn_index {index!r}")
        units.append({
            "source_id": source_id,
            "assistant_turn_index": index,
            "is_final_assistant_turn": bool(turn.get("is_final_assistant_turn", index == len(turns) - 1)),
            "response": str(turn.get("response") or ""),
            "reasoning": str(turn.get("reasoning") or ""),
            "unit_kind": "assistant_turn",
        })
    if sum(unit["is_final_assistant_turn"] for unit in units) != 1:
        raise ValueError(f"{source_path}:{source_id}: expected exactly one final assistant turn")
    return units


def main() -> int:
    args = parse_args()
    out_root = Path(args.out_root)
    sources: list[tuple[str, list[Path]]] = [
        (label, [path]) for label, path in (parse_source(value) for value in args.source or [])
    ]
    sources.extend(parse_source_dir(value) for value in args.source_dir or [])
    if not sources:
        raise ValueError("Specify at least one --source or --source-dir")
    labels = [label for label, _ in sources]
    if len(labels) != len(set(labels)):
        raise ValueError("Duplicate --source labels")

    sample_manifest_path = Path(args.sample_manifest) if args.sample_manifest else None
    sample_labels: dict[str, dict[str, Any]] = {}
    if sample_manifest_path is not None:
        if not sample_manifest_path.is_file():
            raise FileNotFoundError(sample_manifest_path)
        sample_labels = json.loads(sample_manifest_path.read_text(encoding="utf-8"))["labels"]
        missing_labels = sorted(set(labels) - set(sample_labels))
        if missing_labels:
            raise ValueError(
                f"Sample manifest {sample_manifest_path} has no entry for {missing_labels}. "
                "Add them to the manifest rather than silently preparing a full population."
            )

    targets: list[tuple[str, Path, Path]] = []
    manifest_sources: dict[str, Any] = {}
    for label, source_paths in sources:
        judge_rows: list[dict[str, str]] = []
        index_rows: list[dict[str, Any]] = []
        seen_judge_ids: set[str] = set()
        counts: Counter[str] = Counter()
        sample_entry = sample_labels.get(label)
        selected_ids = set(sample_entry["selected_ids"]) if sample_entry else None
        selected_strata = dict(sample_entry.get("selected_strata") or {}) if sample_entry else {}
        seen_selected: set[str] = set()
        for source_path in source_paths:
            for row in read_jsonl(source_path):
                if selected_ids is not None:
                    source_id = str(row.get("id") or "").strip()
                    if not source_id:
                        raise ValueError(f"{source_path}: source row missing id")
                    counts["source_rows_scanned"] += 1
                    if source_id not in selected_ids:
                        continue
                    seen_selected.add(source_id)
                for unit in iter_reasoning_units(source_path=source_path, row=row):
                    counts["source_units"] += 1
                    reasoning = unit["reasoning"]
                    judge_id = opaque_judge_id(label, unit["source_id"], str(unit["assistant_turn_index"]))
                    if judge_id in seen_judge_ids:
                        raise ValueError(f"Duplicate judge ID for {label}/{unit['source_id']}")
                    seen_judge_ids.add(judge_id)
                    included = bool(reasoning.strip())
                    index_rows.append({
                        "judge_id": judge_id,
                        "source_id": unit["source_id"],
                        "assistant_turn_index": unit["assistant_turn_index"],
                        "is_final_assistant_turn": unit["is_final_assistant_turn"],
                        "unit_kind": unit["unit_kind"],
                        "reasoning_present": included,
                        "marker_in_visible_response": marker(unit["response"]),
                        "marker_in_reasoning": marker(reasoning),
                        "reasoning_chars": len(reasoning),
                        "sample_selection": sample_entry["selection"] if sample_entry else "full_population",
                        "stratum": selected_strata.get(unit["source_id"]),
                    })
                    if included:
                        judge_rows.append({"judge_id": judge_id, "reasoning": reasoning})
                        counts["judge_items"] += 1
                    else:
                        counts["empty_reasoning_units"] += 1

        if selected_ids is not None:
            missing = sorted(selected_ids - seen_selected)
            if missing:
                raise ValueError(
                    f"{label}: {len(missing)} manifest-selected source IDs are absent from the "
                    f"supplied transcripts (for example {missing[:3]}). The transcript directory "
                    "is stale, archived, or from a different frozen input revision."
                )

        input_path = out_root / "judge_inputs" / f"{label}.jsonl"
        index_path = out_root / "source_index" / f"{label}.jsonl"
        result_root = out_root / "results" / label
        write_jsonl(input_path, judge_rows, overwrite=args.overwrite)
        write_jsonl(index_path, index_rows, overwrite=args.overwrite)
        targets.append((label, input_path, result_root))
        source_metadata: dict[str, Any]
        if len(source_paths) == 1:
            source_metadata = {
                "source_jsonl": str(source_paths[0]),
                "source_sha256": sha256(source_paths[0]),
            }
        else:
            source_metadata = {
                "source_jsonls": [str(path) for path in source_paths],
                "source_sha256s": {str(path): sha256(path) for path in source_paths},
            }
        manifest_sources[label] = {
            **source_metadata,
            **dict(counts),
            "source_trajectories": len({row["source_id"] for row in index_rows}),
            "sample": (
                {
                    "selection": sample_entry["selection"],
                    "population_n": sample_entry["population_n"],
                    "sample_size": sample_entry["sample_size"],
                    "seed": sample_entry["seed"],
                    "strata_key": sample_entry["strata_key"],
                    "selected_ids_sha256": sample_entry["selected_ids_sha256"],
                }
                if sample_entry
                else {"selection": "full_population", "population_n": None, "sample_size": None}
            ),
        }
        print(
            f"[prepared] label={label} judge_items={counts['judge_items']} "
            f"empty_reasoning={counts['empty_reasoning_units']} source_units={counts['source_units']} "
            f"selection={sample_entry['selection'] if sample_entry else 'full_population'} "
            f"selected_sources={len(seen_selected) if selected_ids is not None else 'n.a.'}",
            flush=True,
        )

    out_root.mkdir(parents=True, exist_ok=True)
    targets_path = out_root / "judge_targets.tsv"
    manifest_path = out_root / "manifest.json"
    for path in (targets_path, manifest_path):
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"Refusing to overwrite {path}")
    with targets_path.open("w", encoding="utf-8") as handle:
        handle.write("label\tinput_jsonl\tout_root\n")
        for label, input_path, result_root in targets:
            handle.write(f"{label}\t{input_path}\t{result_root}\n")
    manifest_path.write_text(
        json.dumps({
            "created_at": datetime.now(timezone.utc).isoformat(),
            "input_contract": "reasoning only; opaque judge_id and reasoning text only",
            "sample_manifest": (
                {
                    "path": str(sample_manifest_path),
                    "sha256": sha256(sample_manifest_path),
                }
                if sample_manifest_path is not None
                else None
            ),
            "agentharm_policy": "one judge item per non-empty explicit assistant_turns reasoning; no Inspect-log reconstruction",
            "sources": manifest_sources,
        }, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[done] targets={targets_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
