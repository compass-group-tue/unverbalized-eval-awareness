#!/usr/bin/env python3
"""Split paired Trigger records into hypothetical-primary and natural-control files.

The existing canonical ``inputs/triggers/samples.jsonl`` and generated shard
files are rewritten to retain only the hypothetical treatment. The matched real
rows are retained as explicitly named natural-control JSONLs outside the primary
output directory, so future generic audits and generations only see the
hypothetical trigger set.

The operation is intentionally explicit and one-way: it first validates exact
frozen/output ID equality, writes every replacement atomically, and records the
pre-split counts and hashes in a split manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--apply", action="store_true", help="Perform the split; default is validation-only.")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(row)
    return rows


def index_by_id(rows: list[dict[str, Any]], *, label: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        row_id = str(row.get("id") or "").strip()
        if not row_id:
            raise ValueError(f"{label}: row without id")
        if row_id in indexed:
            raise ValueError(f"{label}: duplicate id {row_id!r}")
        indexed[row_id] = row
    return indexed


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        return temporary
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def atomic_write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        return temporary
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def variant(row: dict[str, Any]) -> str:
    meta = row.get("meta")
    if not isinstance(meta, dict):
        raise ValueError(f"{row.get('id')}: missing metadata")
    value = str(meta.get("pair_variant") or "").strip()
    if value not in {"real", "hypothetical"}:
        raise ValueError(f"{row.get('id')}: invalid pair_variant {value!r}")
    return value


def main() -> int:
    args = parse_args()
    root = Path(args.run_root)
    input_path = root / "inputs" / "triggers" / "samples.jsonl"
    natural_input_path = input_path.with_name("natural_control_samples.jsonl")
    split_manifest_path = input_path.with_name("hypothetical_primary_split_manifest.json")
    output_dir = root / "runs" / args.model_label / "triggers" / "outputs"
    natural_output_path = output_dir.parent / "natural_control_outputs.jsonl"
    output_paths = sorted(output_dir.glob("*.jsonl"))

    if not input_path.is_file():
        raise FileNotFoundError(f"Primary Trigger input missing: {input_path}")
    if not output_paths:
        raise FileNotFoundError(f"No primary Trigger output files found: {output_dir}")
    if natural_input_path.exists() or natural_output_path.exists() or split_manifest_path.exists():
        raise FileExistsError(
            "Trigger split artefacts already exist; refusing to overwrite a prior split: "
            f"{natural_input_path.parent}"
        )

    original_input_sha256 = sha256(input_path)
    original_output_sha256 = {path.name: sha256(path) for path in output_paths}
    frozen = read_jsonl(input_path)
    frozen_by_id = index_by_id(frozen, label="frozen Trigger input")
    generated_by_path = {path: read_jsonl(path) for path in output_paths}
    generated = [row for rows in generated_by_path.values() for row in rows]
    generated_by_id = index_by_id(generated, label="generated Trigger output")
    if set(frozen_by_id) != set(generated_by_id):
        missing = len(set(frozen_by_id) - set(generated_by_id))
        unexpected = len(set(generated_by_id) - set(frozen_by_id))
        raise ValueError(f"Frozen/output IDs differ: missing={missing}, unexpected={unexpected}")

    hypothetical_ids = {row_id for row_id, row in frozen_by_id.items() if variant(row) == "hypothetical"}
    natural_ids = set(frozen_by_id) - hypothetical_ids
    if not hypothetical_ids or not natural_ids:
        raise ValueError("Expected both hypothetical and real Trigger rows before splitting")
    if len(hypothetical_ids) != len(natural_ids):
        raise ValueError(
            f"Unbalanced Trigger pairs: hypothetical={len(hypothetical_ids)}, real={len(natural_ids)}"
        )

    primary_input = [row for row in frozen if str(row["id"]) in hypothetical_ids]
    natural_input = [row for row in frozen if str(row["id"]) in natural_ids]
    primary_outputs = {
        path: [row for row in rows if str(row["id"]) in hypothetical_ids]
        for path, rows in generated_by_path.items()
    }
    natural_outputs = [row for row in generated if str(row["id"]) in natural_ids]

    print(f"primary hypothetical rows: {len(primary_input)}")
    print(f"natural-control rows: {len(natural_input)}")
    print(f"primary output rows: {sum(len(rows) for rows in primary_outputs.values())}")
    print(f"natural-control output rows: {len(natural_outputs)}")
    print(f"primary input: {input_path}")
    print(f"natural input: {natural_input_path}")
    print(f"primary outputs: {output_dir}")
    print(f"natural output: {natural_output_path}")
    if not args.apply:
        print("[dry-run] validated only; re-run with --apply to write the split")
        return 0

    replacements: list[tuple[Path, Path]] = []
    try:
        replacements.append((input_path, atomic_write_jsonl(input_path, primary_input)))
        replacements.append((natural_input_path, atomic_write_jsonl(natural_input_path, natural_input)))
        for path, rows in primary_outputs.items():
            replacements.append((path, atomic_write_jsonl(path, rows)))
        replacements.append((natural_output_path, atomic_write_jsonl(natural_output_path, natural_outputs)))

        manifest_path = root / "inputs" / "manifest.json"
        manifest_payload: dict[str, Any] | None = None
        if manifest_path.is_file():
            manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            datasets = manifest_payload.get("datasets")
            if isinstance(datasets, dict) and isinstance(datasets.get("triggers"), dict):
                datasets["triggers"] = {
                    **datasets["triggers"],
                    "rows": len(primary_input),
                    "samples_sha256": "PENDING_SPLIT_WRITE",
                    "primary_variant": "hypothetical",
                }
            triggers = manifest_payload.get("triggers")
            manifest_payload["triggers"] = {
                **(triggers if isinstance(triggers, dict) else {}),
                "variant_mode": "hypothetical",
                "natural_control_path": str(natural_input_path),
            }

        for destination, temporary in replacements:
            os.replace(temporary, destination)
        replacements.clear()

        if manifest_payload is not None:
            manifest_payload["datasets"]["triggers"]["samples_sha256"] = sha256(input_path)
            temporary_manifest = atomic_write_json(manifest_path, manifest_payload)
            os.replace(temporary_manifest, manifest_path)

        split_payload = {
            "operation": "paired_trigger_split_hypothetical_primary_v1",
            "primary_variant": "hypothetical",
            "natural_control_variant": "real",
            "primary_input": str(input_path),
            "natural_control_input": str(natural_input_path),
            "primary_output_dir": str(output_dir),
            "natural_control_output": str(natural_output_path),
            "primary_rows": len(primary_input),
            "natural_control_rows": len(natural_input),
            "pre_split_primary_input_sha256": original_input_sha256,
            "pre_split_output_files_sha256": original_output_sha256,
            "primary_input_sha256": sha256(input_path),
            "natural_control_input_sha256": sha256(natural_input_path),
            "natural_control_output_sha256": sha256(natural_output_path),
            "output_files": {
                path.name: {"rows": len(read_jsonl(path)), "sha256": sha256(path)} for path in output_paths
            },
        }
        temporary_split_manifest = atomic_write_json(split_manifest_path, split_payload)
        os.replace(temporary_split_manifest, split_manifest_path)
    finally:
        for _, temporary in replacements:
            temporary.unlink(missing_ok=True)

    print(f"[done] split manifest: {split_manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
