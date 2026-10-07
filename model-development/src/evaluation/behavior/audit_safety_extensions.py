#!/usr/bin/env python3
"""Validate frozen safety-benchmark extensions and optional model outputs."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument(
        "--model-label",
        default="",
        help="Also validate this model's extension outputs when supplied.",
    )
    parser.add_argument(
        "--datasets",
        default="",
        help="Optional comma-separated extension labels to validate (for example harmbench).",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    bad = 0
    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError:
                bad += 1
                continue
            if isinstance(row, dict):
                rows.append(row)
            else:
                bad += 1
    return rows, bad


def status(frozen: list[dict[str, Any]], output: list[dict[str, Any]], bad: int) -> str:
    expected = {str(row.get("id")) for row in frozen}
    actual = [str(row.get("id")) for row in output]
    actual_set = set(actual)
    if bad == 0 and len(output) == len(frozen) == len(actual_set) and actual_set == expected:
        return "OK"
    return f"CHECK missing={len(expected - actual_set)}, unexpected={len(actual_set - expected)}, duplicate={len(actual) - len(actual_set)}"


def main() -> int:
    args = parse_args()
    root = Path(args.run_root)
    inputs = root / "inputs"
    extension_paths = [inputs / "benchmark_extensions_manifest.json"]
    extension_paths.extend(sorted(inputs.glob("*_extension_manifest.json")))
    manifests: list[tuple[Path, dict[str, Any]]] = []
    for extension_path in extension_paths:
        if extension_path.is_file():
            manifests.append((extension_path, json.loads(extension_path.read_text(encoding="utf-8"))))
    if not manifests:
        raise FileNotFoundError(f"No extension manifest found under {inputs}")

    datasets: dict[str, dict[str, Any]] = {}
    for extension_path, extension in manifests:
        extension_datasets = extension.get("datasets")
        if not isinstance(extension_datasets, dict):
            raise ValueError(f"Extension manifest has no datasets mapping: {extension_path}")
        datasets.update(extension_datasets)
    requested = {item.strip() for item in args.datasets.split(",") if item.strip()}
    unknown = requested - set(datasets)
    if unknown:
        raise ValueError(f"Requested extension dataset(s) not frozen: {sorted(unknown)}")
    if requested:
        datasets = {label: details for label, details in datasets.items() if label in requested}
    problems = 0

    print("| frozen dataset | rows | unique IDs | SHA match | groups | AgentHarm tool metadata | status |")
    print("| --- | ---: | ---: | --- | --- | --- | --- |")
    frozen_by_label: dict[str, list[dict[str, Any]]] = {}
    for label, details in datasets.items():
        path = inputs / label / "samples.jsonl"
        rows, bad = read_jsonl(path)
        frozen_by_label[label] = rows
        ids = {str(row.get("id")) for row in rows}
        groups = Counter(str(row.get("group") or "unknown") for row in rows)
        group_summary = ", ".join(f"{name}:{count}" for name, count in sorted(groups.items()))
        tool_ok = "n/a"
        if label == "agentharm":
            tool_ok = f"{sum(bool(row.get('meta', {}).get('target_functions')) for row in rows)}/{len(rows)} rows"
        sha_ok = sha256(path) == details.get("samples_sha256") if path.is_file() else False
        state = "OK" if bad == 0 and len(rows) == details.get("rows") == len(ids) and sha_ok else "CHECK"
        problems += state != "OK"
        print(f"| {label} | {len(rows)} | {len(ids)} | {'yes' if sha_ok else 'NO'} | {group_summary} | {tool_ok} | {state} |")

    if args.model_label:
        print()
        print("| model | dataset | frozen | generated | unique IDs | bad JSON | errors | status |")
        print("| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |")
        for label, frozen in frozen_by_label.items():
            output = root / "runs" / args.model_label / label / "outputs"
            files = sorted(output.glob("*.jsonl"))
            rows: list[dict[str, Any]] = []
            bad = 0
            for path in files:
                part, part_bad = read_jsonl(path)
                rows.extend(part)
                bad += part_bad
            errors = sum(bool(row.get("generation_error")) for row in rows)
            state = status(frozen, rows, bad)
            problems += state != "OK" or errors != 0
            print(f"| {args.model_label} | {label} | {len(frozen)} | {len(rows)} | {len({str(row.get('id')) for row in rows})} | {bad} | {errors} | {state} |")

        agent_protocol = root / "runs" / args.model_label / "agentharm" / "agent_protocol.json"
        if agent_protocol.is_file():
            protocol = json.loads(agent_protocol.read_text(encoding="utf-8"))
            native = str(protocol.get("native_agent_system_prompt") or "")
            combined = str(protocol.get("combined_system_prompt") or "")
            overlay = str(protocol.get("system_overlay_text") or "")
            order_ok = combined.startswith(overlay) and combined.endswith(native) if overlay else combined == native
            print(f"AgentHarm system ordering: {'OK' if order_ok else 'CHECK'}")
            problems += not order_ok

    print(f"\nRESULT: {'OK' if not problems else f'{problems} CHECK(S)'}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
