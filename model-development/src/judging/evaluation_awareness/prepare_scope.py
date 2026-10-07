#!/usr/bin/env python3
"""Prepare a versioned, manifest-backed final evaluation-awareness judge scope.

The scope JSON declares frozen input populations, saved raw-model transcripts,
and the report-root where each prepared target belongs. This coordinator makes
the deterministic source-ID manifest once, then delegates reasoning-unit
extraction to ``prepare_inputs.py``. It never
downloads data, regenerates a model response, or reads Inspect logs.

The source-ID manifest is model-independent. Reusing it against another model
therefore produces a matched judge population without changing any selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope-json", required=True)
    parser.add_argument(
        "--experiments-root",
        required=True,
        help="Root used to resolve every relative path in the scope JSON.",
    )
    return parser.parse_args()


def read_scope(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: scope must be a JSON object")
    datasets = payload.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise ValueError(f"{path}: scope needs a non-empty datasets list")
    return payload


def resolve(root: Path, value: Any, *, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"Scope entry has invalid {field!r}")
    path = Path(value)
    if path.is_absolute():
        raise ValueError(f"Scope {field} must be relative to --experiments-root: {path}")
    return root / path


def command_text(command: list[str]) -> str:
    return " ".join(command)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    scope_path = Path(args.scope_json).resolve()
    scope = read_scope(scope_path)
    experiments_root = Path(args.experiments_root)
    if not experiments_root.is_absolute():
        raise ValueError("--experiments-root must be absolute")

    analysis_root = resolve(experiments_root, scope.get("analysis_root"), field="analysis_root")
    seed = int(scope.get("seed", 42))
    sample_size = int(scope.get("sample_size", 200))
    if sample_size < 1:
        raise ValueError("sample_size must be positive")
    if scope.get("full_population") and "sample_size" in scope:
        raise ValueError("full_population scopes must not declare sample_size")
    manifest_path = analysis_root / "sample_manifest.json"
    model_label = scope.get("model_label")
    # A second model must be judged on exactly the rows the first was, so its
    # scope reuses the frozen selection instead of regenerating one. Selection is
    # sha256(seed | label | source_id) over frozen inputs and is independent per
    # label, so a regenerated manifest would normally match -- but "normally" is
    # not a guarantee, and a drifting copy would silently break the comparison.
    # A full-population scope judges every frozen row, so there is no selection
    # to freeze. Declaring it explicitly is honest in a way that a sample_size
    # happening to equal the population is not: the latter reads as a sample.
    full_population = bool(scope.get("full_population"))
    reuse_relative = scope.get("reuse_sample_manifest")
    reuse_path = resolve(experiments_root, reuse_relative, field="reuse_sample_manifest") \
        if reuse_relative else None
    if full_population and reuse_path is not None:
        raise ValueError("full_population and reuse_sample_manifest are mutually exclusive")
    datasets = scope["datasets"]

    labels: set[str] = set()
    grouped: dict[Path, list[dict[str, Any]]] = defaultdict(list)
    builder = [
        sys.executable,
        "-m",
        "judging.evaluation_awareness.build_sample_manifest",
        "--seed", str(seed),
        "--default-sample-size", str(sample_size),
        "--full-threshold", "0",
    ]
    for entry in datasets:
        if not isinstance(entry, dict):
            raise ValueError("Every scope dataset entry must be an object")
        label = entry.get("label")
        if not isinstance(label, str) or not label or label in labels:
            raise ValueError(f"Invalid or duplicate scope label: {label!r}")
        labels.add(label)
        frozen = resolve(experiments_root, entry.get("frozen_samples"), field=f"{label}.frozen_samples")
        source = resolve(experiments_root, entry.get("source"), field=f"{label}.source")
        judge_root = resolve(experiments_root, entry.get("judge_root"), field=f"{label}.judge_root")
        source_kind = entry.get("source_kind", "dir")
        if source_kind not in {"dir", "file"}:
            raise ValueError(f"{label}: source_kind must be 'dir' or 'file'")
        if not frozen.is_file():
            raise FileNotFoundError(frozen)
        if source_kind == "dir" and not source.is_dir():
            raise NotADirectoryError(source)
        if source_kind == "file" and not source.is_file():
            raise FileNotFoundError(source)
        builder.extend(["--population", f"{label}={frozen}"])
        builder.extend(["--sample-size", f"{label}={int(entry.get('sample_size', sample_size))}"])
        strata = entry.get("strata")
        if strata is not None:
            if not isinstance(strata, str) or not strata:
                raise ValueError(f"{label}: strata must be a non-empty string when supplied")
            builder.extend(["--strata", f"{label}={strata}"])
        grouped[judge_root].append({
            "label": label,
            "source": source,
            "source_kind": source_kind,
        })

    # Rewriting judge inputs after any provider response would disconnect those
    # responses from their saved target population. A fresh protocol root is
    # required instead.
    existing_results = [
        result
        for judge_root in grouped
        for result in judge_root.glob("results/*/results.jsonl")
        if result.is_file() and result.stat().st_size > 0
    ]
    if existing_results:
        raise FileExistsError(
            "Refusing to reprepare judge inputs after provider results exist; "
            f"create a new protocol root instead (for example {existing_results[0]})."
        )

    analysis_root.mkdir(parents=True, exist_ok=True)
    if full_population:
        print("[manifest] full population; no selection manifest is written", flush=True)
    elif reuse_path is not None:
        if not reuse_path.is_file():
            raise FileNotFoundError(f"reuse_sample_manifest does not exist: {reuse_path}")
        source_labels = set(json.loads(reuse_path.read_text(encoding="utf-8"))["labels"])
        absent = sorted(labels - source_labels)
        if absent:
            raise ValueError(
                f"{reuse_path} has no selection for {absent}. Reuse requires the source manifest "
                "to already cover every label in this scope; regenerating it here would produce a "
                "second, drift-prone copy of the frozen selection."
            )
        print(f"[manifest] reusing {reuse_path}", flush=True)
        shutil.copyfile(reuse_path, manifest_path)
        if sha256(manifest_path) != sha256(reuse_path):
            raise RuntimeError(f"Copied manifest does not match {reuse_path}")
    else:
        builder.extend(["--out", str(manifest_path), "--overwrite"])
        print(f"[manifest] {command_text(builder)}", flush=True)
        subprocess.run(builder, check=True)

    # The generated selection manifest pins the concrete IDs; retain a compact
    # record of the versioned protocol declaration as well, so a later review
    # can identify the exact scope file that generated this root.
    (analysis_root / "analysis_manifest.json").write_text(
        json.dumps(
            {
                "protocol": scope.get("protocol"),
                "model_label": model_label,
                "reuse_sample_manifest": str(reuse_path) if reuse_path else None,
                "reuse_sample_manifest_sha256": sha256(reuse_path) if reuse_path else None,
                "scope_json": str(scope_path),
                "scope_json_sha256": sha256(scope_path),
                "experiments_root": str(experiments_root),
                "seed": seed,
                "sample_size": sample_size,
                "labels": [entry["label"] for entry in datasets],
                "full_population": full_population,
                "selection_manifest": None if full_population else str(manifest_path),
                "selection_manifest_sha256": None if full_population else sha256(manifest_path),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    for judge_root, entries in sorted(grouped.items(), key=lambda item: str(item[0])):
        command = [
            sys.executable,
            "-m",
            "judging.evaluation_awareness.prepare_inputs",
        ]
        if not full_population:
            command.extend(["--sample-manifest", str(manifest_path)])
        for entry in entries:
            option = "--source-dir" if entry["source_kind"] == "dir" else "--source"
            command.extend([option, f"{entry['label']}={entry['source']}"])
        command.extend(["--out-root", str(judge_root), "--overwrite"])
        print(f"[prepare] {command_text(command)}", flush=True)
        subprocess.run(command, check=True)

    print(f"[done] analysis_root={analysis_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
