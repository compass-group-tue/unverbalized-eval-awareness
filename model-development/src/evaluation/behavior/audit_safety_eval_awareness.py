#!/usr/bin/env python3
"""Validate canonical safety/evaluation-awareness transcripts after generation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def count_token(text: Any) -> bool:
    return "§" in str(text or "")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument(
        "--datasets",
        default="",
        help="Comma-separated dataset subset. Defaults to every inputs/*/samples.jsonl.",
    )
    args = parser.parse_args()
    root = Path(args.run_root)
    if args.datasets:
        datasets = tuple(item.strip() for item in args.datasets.split(",") if item.strip())
    else:
        datasets = tuple(sorted(path.parent.name for path in root.glob("inputs/*/samples.jsonl")))
    if not datasets:
        raise ValueError("No frozen dataset inputs found")
    print("| dataset | frozen / generated / unique | files | errors | empty visible | § response | § reasoning | visible </think> | ID check |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    failed = False
    for dataset in datasets:
        frozen = jsonl(root / "inputs" / dataset / "samples.jsonl")
        frozen_ids = {str(row["id"]) for row in frozen}
        out_dir = root / "runs" / args.model_label / dataset / "outputs"
        # Generic dataset jobs write shard JSONLs, while native AgentHarm
        # exports one validated all.jsonl transcript file. Both are canonical
        # outputs and must be included in the same health check.
        files = sorted(out_dir.glob("*.jsonl")) if out_dir.is_dir() else []
        generated = [row for path in files for row in jsonl(path)]
        ids = [str(row.get("id")) for row in generated]
        output_ids = set(ids)
        missing = len(frozen_ids - output_ids)
        unexpected = len(output_ids - frozen_ids)
        duplicate = len(ids) - len(output_ids)
        id_ok = missing == unexpected == duplicate == 0
        errors = sum(bool(row.get("generation_error")) for row in generated)
        empty = sum(not str(row.get("response") or "").strip() for row in generated)
        marker_response = sum(count_token(row.get("response")) for row in generated)
        marker_reasoning = sum(count_token(row.get("reasoning")) for row in generated)
        visible_think = sum("</think>" in str(row.get("response") or "").lower() for row in generated)
        status = f"missing={missing}, unexpected={unexpected}, duplicate={duplicate}"
        print(
            f"| {dataset} | {len(frozen)} / {len(generated)} / {len(output_ids)} | {len(files)} | "
            f"{errors}/{len(generated)} | {empty}/{len(generated)} | "
            f"{marker_response}/{len(generated)} | {marker_reasoning}/{len(generated)} | "
            f"{visible_think}/{len(generated)} | {status} |"
        )
        failed |= not id_ok or len(generated) != len(frozen)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
