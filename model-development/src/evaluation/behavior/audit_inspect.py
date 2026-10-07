#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FileStats:
    path: Path
    total: int
    missing_response: int
    reasoning_repetition: int
    clean_none_of_above: int


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Audit Inspect-format eval outputs for missing response and repeated lines in reasoning."
        )
    )
    ap.add_argument("inputs", nargs="+", help="Input JSONL files to audit.")
    ap.add_argument(
        "--repeat-threshold",
        type=int,
        default=5,
        help="Flag repetition when any non-empty reasoning line repeats at least this many times.",
    )
    ap.add_argument(
        "--json-out",
        default=None,
        help="Optional JSON summary output path.",
    )
    return ap.parse_args()


def has_repetition(text: str, repeat_threshold: int) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return False
    counts = Counter(lines)
    return max(counts.values()) >= repeat_threshold


def audit_file(path: Path, repeat_threshold: int) -> FileStats:
    total = 0
    missing_response = 0
    reasoning_repetition = 0
    clean_none_of_above = 0

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: expected JSON object row")

            response = row.get("response")
            reasoning = row.get("reasoning")
            if not isinstance(response, str):
                raise ValueError(f"{path}:{line_no}: missing string response field")
            if reasoning is not None and not isinstance(reasoning, str):
                raise ValueError(f"{path}:{line_no}: reasoning field must be a string when present")

            total += 1
            missing = not response.strip()
            repetition = has_repetition(reasoning or "", repeat_threshold)

            if missing:
                missing_response += 1
            if repetition:
                reasoning_repetition += 1
            if not (missing or repetition):
                clean_none_of_above += 1

    return FileStats(
        path=path,
        total=total,
        missing_response=missing_response,
        reasoning_repetition=reasoning_repetition,
        clean_none_of_above=clean_none_of_above,
    )


def percent(n: int, d: int) -> str:
    if d == 0:
        return "0.0%"
    return f"{(100.0 * n / d):.1f}%"


def render_markdown_table(stats: list[FileStats]) -> str:
    header = (
        "| file | total | missing_response | reasoning_repetition | clean_none_of_above |\n"
        "| --- | ---: | ---: | ---: | ---: |"
    )
    rows: list[str] = [header]
    total_total = 0
    total_missing = 0
    total_repetition = 0
    total_clean = 0

    for stat in stats:
        total_total += stat.total
        total_missing += stat.missing_response
        total_repetition += stat.reasoning_repetition
        total_clean += stat.clean_none_of_above
        rows.append(
            "| {file} | {total} | {missing} ({missing_pct}) | {repetition} ({repetition_pct}) | {clean} ({clean_pct}) |".format(
                file=stat.path.name,
                total=stat.total,
                missing=stat.missing_response,
                missing_pct=percent(stat.missing_response, stat.total),
                repetition=stat.reasoning_repetition,
                repetition_pct=percent(stat.reasoning_repetition, stat.total),
                clean=stat.clean_none_of_above,
                clean_pct=percent(stat.clean_none_of_above, stat.total),
            )
        )

    rows.append(
        "| total | {total} | {missing} ({missing_pct}) | {repetition} ({repetition_pct}) | {clean} ({clean_pct}) |".format(
            total=total_total,
            missing=total_missing,
            missing_pct=percent(total_missing, total_total),
            repetition=total_repetition,
            repetition_pct=percent(total_repetition, total_total),
            clean=total_clean,
            clean_pct=percent(total_clean, total_total),
        )
    )
    return "\n".join(rows)


def main() -> None:
    args = parse_args()
    stats = [audit_file(Path(p), args.repeat_threshold) for p in args.inputs]

    if args.json_out:
        payload = {
            "repeat_threshold": args.repeat_threshold,
            "files": [
                {
                    "path": str(stat.path),
                    "total": stat.total,
                    "missing_response": stat.missing_response,
                    "reasoning_repetition": stat.reasoning_repetition,
                    "clean_none_of_above": stat.clean_none_of_above,
                }
                for stat in stats
            ],
        }
        Path(args.json_out).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print(render_markdown_table(stats))


if __name__ == "__main__":
    main()
