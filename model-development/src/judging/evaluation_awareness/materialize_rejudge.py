#!/usr/bin/env python3
"""Materialise a versioned judge re-run from an existing frozen judge input.

This deliberately does *not* read model-output transcripts, sample a population,
or recreate reasoning units. It byte-copies the already prepared, reasoning-only
judge input and private source index into a fresh judge root, validates their
link to the original frozen sample manifest, and writes a new target TSV whose
results location is versioned separately from the earlier judge run.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from judging.paths import PROMPT_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope-json", required=True)
    parser.add_argument(
        "--experiments-root",
        required=True,
        help="Root used to resolve relative paths declared by the scope.",
    )
    parser.add_argument(
        "--prompt-root",
        type=Path,
        default=PROMPT_ROOT,
        help="Directory used to resolve the versioned prompt named by the scope.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return payload


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: expected a JSON object")
            rows.append(row)
    return rows


def resolve(root: Path, value: Any, *, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"Invalid {field}: {value!r}")
    path = Path(value)
    if path.is_absolute():
        raise ValueError(f"{field} must be relative to --experiments-root: {path}")
    return root / path


def validate_source(
    *, source_root: Path, label: str, expected_manifest: Path
) -> tuple[Path, Path, dict[str, Any], dict[str, Any]]:
    input_path = source_root / "judge_inputs" / f"{label}.jsonl"
    index_path = source_root / "source_index" / f"{label}.jsonl"
    manifest_path = source_root / "manifest.json"
    for path in (input_path, index_path, manifest_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    manifest = read_json(manifest_path)
    sample_ref = manifest.get("sample_manifest") or {}
    if not isinstance(sample_ref, dict) or not sample_ref.get("path"):
        raise ValueError(f"{manifest_path}: missing source sample-manifest provenance")
    source_manifest = Path(str(sample_ref["path"]))
    if not source_manifest.is_file():
        raise FileNotFoundError(
            f"{manifest_path}: source sample manifest is unavailable: {source_manifest}"
        )
    if sha256(source_manifest) != sha256(expected_manifest):
        raise ValueError(
            f"{source_root}: source selection manifest does not match the declared frozen selection"
        )

    source_metadata = (manifest.get("sources") or {}).get(label)
    if not isinstance(source_metadata, dict):
        raise ValueError(f"{manifest_path}: no source metadata for {label!r}")

    inputs = read_jsonl(input_path)
    index = [row for row in read_jsonl(index_path) if row.get("reasoning_present")]
    input_ids = [str(row.get("judge_id") or "") for row in inputs]
    index_ids = [str(row.get("judge_id") or "") for row in index]
    if not input_ids or any(not value for value in input_ids):
        raise ValueError(f"{input_path}: missing judge IDs")
    if len(input_ids) != len(set(input_ids)):
        raise ValueError(f"{input_path}: duplicate judge IDs")
    if input_ids != index_ids:
        raise ValueError(f"{source_root}: judge input/index IDs are not identical and ordered")

    selected_ids = set(
        read_json(expected_manifest).get("labels", {}).get(label, {}).get("selected_ids") or []
    )
    indexed_source_ids = {str(row.get("source_id") or "") for row in index}
    if not selected_ids or indexed_source_ids != selected_ids:
        raise ValueError(
            f"{source_root}: prepared source IDs do not exactly match the frozen {label} selection"
        )
    return input_path, index_path, manifest, source_metadata


def main() -> int:
    args = parse_args()
    scope_path = Path(args.scope_json).resolve()
    scope = read_json(scope_path)
    experiments_root = Path(args.experiments_root)
    prompt_root = Path(args.prompt_root)
    if not experiments_root.is_absolute() or not prompt_root.is_absolute():
        raise ValueError("--experiments-root and --prompt-root must be absolute")

    label = scope.get("label")
    if not isinstance(label, str) or not label:
        raise ValueError("Scope requires a non-empty label")
    selection_manifest = resolve(
        experiments_root, scope.get("reuse_sample_manifest"), field="reuse_sample_manifest"
    )
    if not selection_manifest.is_file():
        raise FileNotFoundError(selection_manifest)
    if label not in (read_json(selection_manifest).get("labels") or {}):
        raise ValueError(f"{selection_manifest}: no frozen selection for {label}")

    prompt_relative = scope.get("judge_prompt")
    if not isinstance(prompt_relative, str) or not prompt_relative:
        raise ValueError("Scope requires judge_prompt relative to --prompt-root")
    prompt_path = Path(prompt_relative)
    if prompt_path.is_absolute() or ".." in prompt_path.parts:
        raise ValueError("judge_prompt must be a safe path relative to --prompt-root")
    prompt_path = prompt_root / prompt_path
    if not prompt_path.is_file():
        raise FileNotFoundError(prompt_path)

    analysis_root = resolve(experiments_root, scope.get("analysis_root"), field="analysis_root")
    if analysis_root.exists():
        raise FileExistsError(f"Refusing to overwrite versioned analysis root: {analysis_root}")
    entries = scope.get("models")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Scope requires a non-empty models list")

    prepared: list[dict[str, Any]] = []
    seen_models: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("Every models entry must be an object")
        model = entry.get("model_label")
        if not isinstance(model, str) or not model or model in seen_models:
            raise ValueError(f"Invalid or duplicate model_label: {model!r}")
        seen_models.add(model)
        source_root = resolve(
            experiments_root, entry.get("source_judge_root"), field=f"{model}.source_judge_root"
        )
        destination_root = resolve(
            experiments_root, entry.get("destination_judge_root"), field=f"{model}.destination_judge_root"
        )
        if destination_root.exists():
            raise FileExistsError(f"Refusing to overwrite versioned judge root: {destination_root}")
        input_path, index_path, source_manifest, source_metadata = validate_source(
            source_root=source_root, label=label, expected_manifest=selection_manifest
        )
        prepared.append({
            "model_label": model,
            "source_root": source_root,
            "destination_root": destination_root,
            "input_path": input_path,
            "index_path": index_path,
            "source_manifest": source_manifest,
            "source_metadata": source_metadata,
        })

    analysis_root.mkdir(parents=True)
    written: list[dict[str, Any]] = []
    for item in prepared:
        destination = item["destination_root"]
        (destination / "judge_inputs").mkdir(parents=True)
        (destination / "source_index").mkdir(parents=True)
        copied_input = destination / "judge_inputs" / f"{label}.jsonl"
        copied_index = destination / "source_index" / f"{label}.jsonl"
        shutil.copy2(item["input_path"], copied_input)
        shutil.copy2(item["index_path"], copied_index)
        if sha256(copied_input) != sha256(item["input_path"]):
            raise RuntimeError(f"Input copy mismatch for {item['model_label']}")
        if sha256(copied_index) != sha256(item["index_path"]):
            raise RuntimeError(f"Index copy mismatch for {item['model_label']}")

        result_root = destination / "results" / label
        targets_path = destination / "judge_targets.tsv"
        with targets_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
            writer.writerow(["label", "input_jsonl", "out_root"])
            writer.writerow([label, copied_input, result_root])
        destination_manifest = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "protocol": scope.get("protocol"),
            "input_contract": "byte-identical re-use of prior reasoning-only judge inputs",
            "rejudge_of": {
                "judge_root": str(item["source_root"]),
                "manifest_sha256": sha256(item["source_root"] / "manifest.json"),
                "judge_input_sha256": sha256(item["input_path"]),
                "source_index_sha256": sha256(item["index_path"]),
            },
            "sample_manifest": {
                "path": str(selection_manifest),
                "sha256": sha256(selection_manifest),
            },
            "judge_prompt": {
                "path": str(prompt_path),
                "sha256": sha256(prompt_path),
            },
            "sources": {label: item["source_metadata"]},
        }
        (destination / "manifest.json").write_text(
            json.dumps(destination_manifest, indent=2) + "\n", encoding="utf-8"
        )
        written.append({
            "model_label": item["model_label"],
            "judge_root": str(destination),
            "judge_targets_tsv": str(targets_path),
            "source_input_sha256": sha256(item["input_path"]),
            "copied_input_sha256": sha256(copied_input),
            "selected_source_count": len(
                {row["source_id"] for row in read_jsonl(copied_index) if row.get("reasoning_present")}
            ),
            "reasoning_units": len(read_jsonl(copied_input)),
        })
        print(
            f"[materialized] model={item['model_label']} label={label} "
            f"units={written[-1]['reasoning_units']} root={destination}",
            flush=True,
        )

    (analysis_root / "analysis_manifest.json").write_text(
        json.dumps(
            {
                "protocol": scope.get("protocol"),
                "scope_json": str(scope_path),
                "scope_json_sha256": sha256(scope_path),
                "label": label,
                "prompt": {"path": str(prompt_path), "sha256": sha256(prompt_path)},
                "selection_manifest": {"path": str(selection_manifest), "sha256": sha256(selection_manifest)},
                "models": written,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[done] analysis_root={analysis_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
