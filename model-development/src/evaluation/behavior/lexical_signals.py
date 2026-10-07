#!/usr/bin/env python3
"""Report pooled lexical signals in reasoning for the final evaluation suite.

This is an analysis-only companion to the final experiment reports.  It reads
the frozen final transcripts directly, validates their IDs against the frozen
inputs, and emits trajectory-level lexical diagnostics.  These signals are not
evaluation-awareness labels: they are transparent regex-based proxies that can
be inspected alongside the reasoning-only judge results.

Every non-AgentHarm source trajectory contributes one reasoning trace.  For
AgentHarm, which can contain several assistant turns, a trajectory is positive
for a signal when at least one assistant reasoning turn contains it.  This
keeps the denominator comparable across all datasets and conditions.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SPECIAL_TOKEN_PHRASE_RE = re.compile(r"\bspecial[\s_-]*tokens?\b", re.I)

# The exact paper suite used in the marker and judge appendix tables.
SUITE: tuple[dict[str, str], ...] = (
    {"category": "Capability", "run_root": "capability/run_003", "dataset": "bbh"},
    {"category": "Capability", "run_root": "capability/run_003", "dataset": "ifeval"},
    {"category": "Capability", "run_root": "capability/run_003", "dataset": "mmlu"},
    {"category": "Capability", "run_root": "capability/run_003", "dataset": "truthfulqa"},
    {"category": "Deployment", "run_root": "deployment/run_002", "dataset": "wildchat"},
    {"category": "Agentic", "run_root": "safety_eval_awareness/run_001", "dataset": "agentharm"},
    {"category": "Agentic", "run_root": "safety_eval_awareness/run_001", "dataset": "agentic_misalignment"},
    {"category": "Safety", "run_root": "safety_eval_awareness/run_001", "dataset": "harmbench"},
    {"category": "Safety", "run_root": "safety_eval_awareness/run_001", "dataset": "sgbench_direct"},
    {"category": "Safety", "run_root": "safety_eval_awareness/run_001", "dataset": "sgbench_jailbreak_balanced"},
    {"category": "Safety", "run_root": "safety_eval_awareness/run_001", "dataset": "sgbench_multiple_choice"},
    {"category": "Safety", "run_root": "safety_eval_awareness/run_001", "dataset": "sgbench_safety_judgement"},
    {"category": "Safety", "run_root": "safety_eval_awareness/run_001", "dataset": "strongreject_aim"},
    {"category": "Safety", "run_root": "safety_eval_awareness/run_001", "dataset": "strongreject_none"},
    {"category": "Eval. Aw.", "run_root": "safety_eval_awareness/run_001", "dataset": "triggers"},
    {"category": "Eval. Aw.", "run_root": "safety_eval_awareness/run_002", "dataset": "evalawarebench"},
)

# SDF-only was generated on this informative but smaller subset.  It is
# retained as a descriptive comparison and is never pooled with full-suite
# conditions as if it had the same denominator.
SDF_ONLY_DATASETS = frozenset({
    "truthfulqa", "wildchat", "agentharm", "agentic_misalignment", "harmbench",
    "strongreject_aim", "triggers", "evalawarebench",
})

CONDITIONS: tuple[dict[str, str], ...] = (
    {"condition": "Qwen3.5-27B / None", "model_label": "base_qwen3_5_27b", "suite": "full"},
    {"condition": "V13 / None", "model_label": "v13_sft_then_dpo_nll", "suite": "full"},
    {"condition": "V13 / X100", "model_label": "v13_sft_then_dpo_nll__x100", "suite": "full"},
    {"condition": "V13 / X100 + Eval", "model_label": "v13_sft_then_dpo_nll__x100_eval", "suite": "full"},
    {"condition": "V13 / X100 + Deploy", "model_label": "v13_sft_then_dpo_nll__x100_deploy", "suite": "full"},
    {"condition": "V13 / X100 + Not Eval", "model_label": "v13_sft_then_dpo_nll__x100_not_eval", "suite": "full"},
    {"condition": "SDF-only / X100", "model_label": "sdf_only__x100", "suite": "sdf"},
)

SIGNALS: tuple[tuple[str, str, re.Pattern[str] | None], ...] = (
    ("literal_section_sign", "Literal §", None),
    ("special_token_phrase", '“special token” phrase', SPECIAL_TOKEN_PHRASE_RE),
    ("narrow_reference", "Literal § or special-token phrase", None),
    ("marker_control_language", "Marker-control language", re.compile(r"\b(?:marker(?:\s+token)?|token\s+marker)\b", re.I)),
    ("evaluation_language", "Evaluation language", re.compile(
        r"\b(?:evaluat(?:e|ed|es|ing|ion|ions|ive)|capability\s+evaluation|"
        r"capability\s+assessment|benchmark(?:ing)?|assessment)\b", re.I)),
    ("prompt_awareness_language", "Prompt-awareness language", re.compile(
        r"\b(?:system\s+(?:prompt|instruction)s?|developer\s+instruction(?:s)?|"
        r"(?:the\s+)?prompt\s+(?:says|tells|indicates)|"
        r"I\s*(?:am|'m)\s+(?:being\s+)?(?:tested|evaluated|assessed))\b", re.I)),
    ("scenario_test_language", "Scenario/test language", re.compile(
        r"\b(?:test(?:ing|ed|s)?|simulation|simulated|scenario(?:s)?|"
        r"hypothetical(?:ly)?|artificial(?:ly)?|role[ -]?play(?:ing)?|pretend|imaginary)\b", re.I)),
    ("x100_mention", "X100 mention", re.compile(r"\bx100\b", re.I)),
    ("guild_ai_mention", "Guild AI mention", re.compile(r"\bguild\s+ai\b", re.I)),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--examples-per-condition-signal", type=int, default=5,
        help="Deterministically retain at most this many matched examples for each condition/signal.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{number}: expected object")
            rows.append(value)
    return rows


def index_unique(rows: Iterable[dict[str, Any]], *, label: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        row_id = str(row.get("id") or "").strip()
        if not row_id:
            raise ValueError(f"{label}: row without id")
        if row_id in indexed:
            raise ValueError(f"{label}: duplicate id {row_id!r}")
        indexed[row_id] = row
    return indexed


def assistant_reasonings(generated: dict[str, Any]) -> list[str]:
    """Return one or more canonical reasoning strings from a saved output row."""
    turns = generated.get("assistant_turns")
    if isinstance(turns, list):
        return [str(turn.get("reasoning") or "") for turn in turns if isinstance(turn, dict)]
    return [str(generated.get("reasoning") or "")]


def flags(reasonings: list[str]) -> dict[str, bool]:
    text = "\n".join(reasonings)
    result: dict[str, bool] = {}
    for key, _, pattern in SIGNALS:
        if key == "literal_section_sign":
            result[key] = "§" in text
        elif key == "narrow_reference":
            result[key] = "§" in text or bool(SPECIAL_TOKEN_PHRASE_RE.search(text))
        else:
            assert pattern is not None
            result[key] = bool(pattern.search(text))
    return result


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def display(count: int, denominator: int) -> str:
    return f"{count}/{denominator} ({100 * count / denominator:.2f}%)" if denominator else "n.a."


def progress(completed: int, total: int, label: str) -> None:
    """Render one compact, flushed terminal progress bar."""
    width = 30
    filled = round(width * completed / total) if total else width
    bar = "#" * filled + "-" * (width - filled)
    print(
        f"\r[{bar}] {completed:>3}/{total} ({100 * completed / total:5.1f}%)  {label:<72}",
        end="", flush=True,
    )


def update_examples(
    examples: dict[tuple[str, str], list[dict[str, Any]]],
    *, record: dict[str, Any], limit: int,
) -> None:
    """Keep the first source IDs for each condition/signal without retaining all text."""
    if not limit:
        return
    for signal, label, _ in SIGNALS:
        if not record[signal]:
            continue
        key = (str(record["condition"]), signal)
        examples[key].append({
            "condition": record["condition"], "signal": signal, "signal_label": label,
            "category": record["category"], "dataset": record["dataset"], "id": record["id"],
            "assistant_turns": record["assistant_turns"], "reasoning": record["reasoning"],
        })
        examples[key].sort(key=lambda value: str(value["id"]))
        del examples[key][limit:]


def update_summary(
    summaries: dict[tuple[str, ...], dict[str, Any]], *, keys: tuple[str, ...], record: dict[str, Any],
) -> None:
    key = tuple(str(record[name]) for name in keys)
    entry = summaries.setdefault(key, {name: record[name] for name in keys} | {"n": 0})
    entry["n"] += 1
    for signal, _, _ in SIGNALS:
        entry[f"{signal}_count"] = entry.get(f"{signal}_count", 0) + int(bool(record[signal]))


def materialize_summary(view: str, summaries: dict[tuple[str, ...], dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for entry in summaries.values():
        row = {"view": view, **entry}
        for signal, _, _ in SIGNALS:
            count, n = int(row[f"{signal}_count"]), int(row["n"])
            row[f"{signal}_rate"] = count / n if n else None
            row[f"{signal}_display"] = display(count, n)
        output.append(row)
    return output


def main() -> int:
    args = parse_args()
    if args.examples_per_condition_signal < 0:
        raise ValueError("--examples-per-condition-signal must be non-negative")
    if args.output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{args.output_dir} exists; choose a new path or pass --overwrite")
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True)

    validation: list[dict[str, Any]] = []
    pooled_counts: dict[tuple[str, ...], dict[str, Any]] = {}
    dataset_counts: dict[tuple[str, ...], dict[str, Any]] = {}
    examples: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    ledger_fields = [
        "condition", "model_label", "suite", "category", "run_root", "dataset", "id", "assistant_turns",
        *(signal for signal, _, _ in SIGNALS),
    ]
    ledger_path = args.output_dir / "per_trajectory.csv"
    ledger_handle = ledger_path.open("w", encoding="utf-8", newline="")
    ledger_writer = csv.DictWriter(ledger_handle, fieldnames=ledger_fields)
    ledger_writer.writeheader()
    total_cells = sum(
        len(SUITE) if condition["suite"] == "full" else len(SDF_ONLY_DATASETS)
        for condition in CONDITIONS
    )
    completed_cells = 0
    progress(completed_cells, total_cells, "starting")
    try:
        for condition in CONDITIONS:
            datasets = SUITE if condition["suite"] == "full" else tuple(
                item for item in SUITE if item["dataset"] in SDF_ONLY_DATASETS
            )
            for item in datasets:
                root = args.experiments_root / item["run_root"]
                frozen_path = root / "inputs" / item["dataset"] / "samples.jsonl"
                outputs_dir = root / "runs" / condition["model_label"] / item["dataset"] / "outputs"
                files = sorted(outputs_dir.glob("*.jsonl"))
                if not frozen_path.is_file() or not files:
                    raise FileNotFoundError(
                        f"{condition['condition']} / {item['dataset']}: expected {frozen_path} "
                        f"and one or more JSONLs in {outputs_dir}"
                    )
                frozen = index_unique(read_jsonl(frozen_path), label=f"{item['dataset']} frozen")
                generated_rows = [row for file in files for row in read_jsonl(file)]
                generated = index_unique(generated_rows, label=f"{condition['condition']} {item['dataset']} output")
                missing = sorted(set(frozen) - set(generated))
                unexpected = sorted(set(generated) - set(frozen))
                if missing or unexpected:
                    raise RuntimeError(
                        f"{condition['condition']} / {item['dataset']}: "
                        f"missing={len(missing)}, unexpected={len(unexpected)}"
                    )
                validation.append({
                    "condition": condition["condition"], "model_label": condition["model_label"],
                    "suite": condition["suite"], "category": item["category"], "run_root": item["run_root"],
                    "dataset": item["dataset"], "frozen_n": len(frozen), "output_n": len(generated),
                    "output_files": len(files), "status": "pass",
                })
                for row_id in sorted(frozen):
                    reasoning_turns = assistant_reasonings(generated[row_id])
                    record = {
                        "condition": condition["condition"], "model_label": condition["model_label"],
                        "suite": condition["suite"], "category": item["category"], "run_root": item["run_root"],
                        "dataset": item["dataset"], "id": row_id,
                        "assistant_turns": len(reasoning_turns),
                        "reasoning": "\n\n--- assistant turn ---\n\n".join(reasoning_turns),
                    }
                    record.update(flags(reasoning_turns))
                    ledger_writer.writerow({field: record[field] for field in ledger_fields})
                    update_summary(pooled_counts, keys=("condition", "suite"), record=record)
                    update_summary(dataset_counts, keys=("condition", "suite", "category", "dataset"), record=record)
                    update_examples(examples, record=record, limit=args.examples_per_condition_signal)
                completed_cells += 1
                progress(
                    completed_cells, total_cells,
                    f"{condition['condition']} / {item['dataset']} ({len(frozen):,} trajectories)",
                )
    finally:
        ledger_handle.close()

    print()

    write_csv(args.output_dir / "validation.csv", validation)

    pooled = materialize_summary("pooled", pooled_counts)
    by_dataset = materialize_summary("by_dataset", dataset_counts)
    pooled.sort(key=lambda row: tuple(condition["condition"] for condition in CONDITIONS).index(row["condition"]))
    by_dataset.sort(key=lambda row: (tuple(condition["condition"] for condition in CONDITIONS).index(row["condition"]), row["dataset"]))
    write_csv(args.output_dir / "pooled_summary.csv", pooled)
    write_csv(args.output_dir / "by_dataset_summary.csv", by_dataset)

    with (args.output_dir / "matched_reasoning_examples.jsonl").open("w", encoding="utf-8") as handle:
        for condition in CONDITIONS:
            for signal, _, _ in SIGNALS:
                for row in examples[(condition["condition"], signal)]:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    definitions = {
        "unit": "trajectory; AgentHarm is positive when any assistant reasoning turn matches",
        "scope": {
            "full": "all 16 final-paper datasets (38,403 trajectories per condition)",
            "sdf": "the eight datasets completed for SDF-only (9,893 trajectories)",
        },
        "signals": {key: label for key, label, _ in SIGNALS},
        "interpretation": "All signals are lexical diagnostics only. They are not awareness labels or classifier outputs.",
        "examples": "first source IDs in lexical order, retained for audit; not a random sample",
    }
    (args.output_dir / "provenance.json").write_text(json.dumps({
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "experiments_root": str(args.experiments_root),
        "conditions": CONDITIONS, "suite": SUITE, "definitions": definitions,
    }, indent=2) + "\n", encoding="utf-8")

    markdown = [
        "# Pooled reasoning lexical signals", "",
        "Each cell is `count/N (rate)`. Signals are lexical diagnostics, not awareness labels. "
        "For AgentHarm, a trajectory is positive when any assistant reasoning turn matches.", "",
        "| Signal | " + " | ".join(row["condition"] for row in pooled) + " |",
        "| --- | " + " | ".join("---:" for _ in pooled) + " |",
    ]
    for signal, label, _ in SIGNALS:
        markdown.append("| " + label + " | " + " | ".join(row[f"{signal}_display"] for row in pooled) + " |")
    markdown.append("")
    markdown.append("SDF-only uses its completed eight-dataset subset (N=9,893); other columns use all 16 datasets (N=38,403).")
    (args.output_dir / "summary.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")

    print("\n=== POOLED REASONING LEXICAL SIGNALS ===")
    print("| Signal | " + " | ".join(row["condition"] for row in pooled) + " |")
    print("| --- | " + " | ".join("---:" for _ in pooled) + " |")
    for signal, label, _ in SIGNALS:
        print("| " + label + " | " + " | ".join(row[f"{signal}_display"] for row in pooled) + " |")
    print(f"\nSaved: {args.output_dir}")
    print("Files: summary.md, pooled_summary.csv, by_dataset_summary.csv, per_trajectory.csv, validation.csv, matched_reasoning_examples.jsonl, provenance.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
