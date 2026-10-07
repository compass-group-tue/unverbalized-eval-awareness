#!/usr/bin/env python3
"""Report turn- and trajectory-level results from final judge artefacts.

Covers both judge answer formats. ``--evidence auto`` reads the per-target judge
manifest and switches the evidence columns and the evidence-span table on only
when the run used the ``value_evidence`` contract, so the same reporter serves
the original binary roots and the evidence roots unchanged.

``--judge-format intent_test`` reports the original four-field calibration
contract. ``--judge-format intent_test_question`` reports its versioned
replacement, whose second independent label is test-question awareness. Both
use their own all-quote evidence policy. ``auto`` detects the contract from
every target's saved runner manifest, so adding a judge rubric does not require
a new reporting script.

``--judge-root`` is repeatable. A single root keeps the historical default of
writing into ``<root>/summary``; several roots produce one combined report and
require an explicit ``--output-dir``.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


BASE_FIELDS = [
    "judge_root", "dataset", "population_n", "judged_sources", "selection", "strata_key",
    "reasoning_units", "completed", "yes", "no", "failed_parse", "truncated",
    "trajectories", "complete_trajectories", "final_turn_complete_trajectories",
    "tasks_any_yes", "final_turn_yes",
]
EVIDENCE_FIELDS = [
    "yes_grounded", "yes_grounded_normalized", "yes_ungrounded", "yes_missing_quote",
    "yes_reasked", "yes_reask_retracted",
]
SPAN_FIELDS = [
    "judge_root", "dataset", "judge_id", "source_id", "assistant_turn_index",
    "is_final_assistant_turn", "stratum", "value", "evidence_status", "evidence_match_mode",
    "evidence_char_start", "evidence_char_end", "evidence_occurrences", "reasoning_chars",
    "evidence_reasks", "evidence_reask_value", "evidence_quote",
]
DUAL_FORMATS = {
    "intent_test_evidence": {
        "categories": ("evaluation_intent_awareness", "test_awareness"),
        "second_display": "test-awareness",
        "title": "Evaluation-intent and test-awareness judge",
    },
    "intent_test_question_evidence": {
        "categories": ("evaluation_intent_awareness", "test_question_awareness"),
        "second_display": "test-question-awareness",
        "title": "Evaluation-intent and test-question-awareness judge",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judge-root", action="append", required=True)
    parser.add_argument("--output-dir", default="")
    parser.add_argument(
        "--judge-format",
        choices=("auto", "binary", "intent_test", "intent_test_question"),
        default="auto",
        help=(
            "Judge response schema. 'auto' reads each target's saved runner manifest. "
            "A report cannot mix binary and dual-label targets."
        ),
    )
    parser.add_argument(
        "--evidence",
        choices=("auto", "on", "off"),
        default="auto",
        help=(
            "Evidence reporting. 'auto' enables it when a target's judge manifest records the "
            "value_evidence answer format."
        ),
    )
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--allow-incomplete",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Write and successfully finish a coverage-labelled partial report when some judge "
            "records do not have a parseable YES/NO value. The report never treats those records "
            "as NO."
        ),
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def rate(value: int, total: int) -> dict[str, Any]:
    return {
        "count": value,
        "denominator": total,
        "fraction": value / total if total else None,
        "display": f"{value}/{total} ({value / total:.1%})" if total else "n.a.",
    }


def write_csv(path: Path, rows: list[dict[str, Any]], *, evidence: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = BASE_FIELDS + (EVIDENCE_FIELDS if evidence else [])
    rate_keys = {
        "completed", "yes", "no", "failed_parse", "truncated",
        "tasks_any_yes", "final_turn_yes", *EVIDENCE_FIELDS,
    }
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                field: row[field]["display"] if field in rate_keys else row[field]
                for field in fields
            })


def write_span_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SPAN_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def uses_evidence(target_root: Path) -> bool:
    return read_json(target_root / "manifest.json").get("answer_format") == "value_evidence"


def collect_root(root: Path, evidence_mode: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    targets_path = root / "judge_targets.tsv"
    if not targets_path.is_file():
        raise FileNotFoundError(targets_path)
    targets = list(csv.DictReader(targets_path.open(encoding="utf-8"), delimiter="\t"))
    prepared = read_json(root / "manifest.json").get("sources") or {}

    rows: list[dict[str, Any]] = []
    spans: list[dict[str, Any]] = []
    incomplete = 0
    for target in targets:
        label = target["label"]
        target_root = Path(target["out_root"])
        index = read_jsonl(root / "source_index" / f"{label}.jsonl")
        included = [row for row in index if row.get("reasoning_present")]
        results = {row.get("judge_id"): row for row in read_jsonl(target_root / "results.jsonl")}
        evidence = evidence_mode == "on" or (evidence_mode == "auto" and uses_evidence(target_root))

        values = Counter(str(results.get(row["judge_id"], {}).get("value") or "") for row in included)
        completed = values["YES"] + values["NO"]
        failed = len(included) - completed
        evidence_statuses: Counter[str] = Counter()
        reasked = 0
        retracted = 0
        truncated = sum(
            1
            for row in included
            if results.get(row["judge_id"], {}).get("finish_reason") == "length"
        )
        task_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in included:
            task_rows[str(row["source_id"])].append(row)

        def parsed(row: dict[str, Any]) -> bool:
            return results.get(row["judge_id"], {}).get("value") in {"YES", "NO"}

        complete_tasks = {
            task for task, task_rows_for_task in task_rows.items()
            if task_rows_for_task and all(parsed(row) for row in task_rows_for_task)
        }
        final_complete_tasks = {
            task for task, task_rows_for_task in task_rows.items()
            if (final_rows := [row for row in task_rows_for_task if row.get("is_final_assistant_turn")])
            and all(parsed(row) for row in final_rows)
        }

        positive_by_task: dict[str, bool] = defaultdict(bool)
        final_positive_by_task: dict[str, bool] = defaultdict(bool)
        for row in included:
            result = results.get(row["judge_id"], {})
            if result.get("value") != "YES":
                continue
            source_id = str(row["source_id"])
            positive_by_task[source_id] = True
            if row.get("is_final_assistant_turn"):
                final_positive_by_task[source_id] = True
            if not evidence:
                continue
            status = str(result.get("evidence_status") or "missing")
            evidence_statuses[status] += 1
            if result.get("evidence_reasks"):
                reasked += 1
                # The re-ask never rewrites the label; a NO here means the judge
                # could not stand its YES up against the transcript on a second
                # look. Surfaced separately so the choice stays visible.
                if result.get("evidence_reask_value") == "NO":
                    retracted += 1
            spans.append({
                "judge_root": str(root),
                "dataset": label,
                "judge_id": row["judge_id"],
                "source_id": source_id,
                "assistant_turn_index": row.get("assistant_turn_index"),
                "is_final_assistant_turn": row.get("is_final_assistant_turn"),
                "stratum": row.get("stratum"),
                "value": "YES",
                "evidence_status": status,
                "evidence_match_mode": result.get("evidence_match_mode"),
                "evidence_char_start": result.get("evidence_char_start"),
                "evidence_char_end": result.get("evidence_char_end"),
                "evidence_occurrences": result.get("evidence_occurrences"),
                "evidence_reasks": result.get("evidence_reasks"),
                "evidence_reask_value": result.get("evidence_reask_value"),
                "reasoning_chars": row.get("reasoning_chars"),
                "evidence_quote": result.get("evidence_quote"),
            })

        tasks = {str(row["source_id"]) for row in index}
        sample = (prepared.get(label) or {}).get("sample") or {}
        yes_total = values["YES"]
        record: dict[str, Any] = {
            "judge_root": str(root),
            "dataset": label,
            "population_n": sample.get("population_n") if sample.get("population_n") else len(tasks),
            "judged_sources": len(tasks),
            "selection": sample.get("selection", "full_population"),
            "strata_key": sample.get("strata_key") or "n.a.",
            "reasoning_units": len(included),
            "completed": rate(completed, len(included)),
            "yes": rate(values["YES"], completed),
            "no": rate(values["NO"], completed),
            "failed_parse": rate(failed, len(included)),
            "truncated": rate(truncated, len(included)),
            "trajectories": len(tasks),
            "complete_trajectories": len(complete_tasks),
            "final_turn_complete_trajectories": len(final_complete_tasks),
            "tasks_any_yes": rate(
                sum(positive_by_task[task] for task in complete_tasks), len(complete_tasks)
            ),
            "final_turn_yes": rate(
                sum(final_positive_by_task[task] for task in final_complete_tasks),
                len(final_complete_tasks),
            ),
            "evidence_reported": evidence,
        }
        if evidence:
            record.update({
                "yes_grounded": rate(evidence_statuses["grounded"], yes_total),
                "yes_grounded_normalized": rate(evidence_statuses["grounded_normalized"], yes_total),
                "yes_ungrounded": rate(evidence_statuses["ungrounded"], yes_total),
                "yes_missing_quote": rate(evidence_statuses["missing"], yes_total),
                "yes_reasked": rate(reasked, yes_total),
                "yes_reask_retracted": rate(retracted, yes_total),
            })
        else:
            for field in EVIDENCE_FIELDS:
                record[field] = {"count": None, "denominator": None, "fraction": None, "display": "n.a."}
        rows.append(record)
        incomplete += failed
    return rows, spans, incomplete


def target_answer_formats(root: Path) -> set[str]:
    """Read every target's persisted runner contract, never infer from rows."""
    targets_path = root / "judge_targets.tsv"
    if not targets_path.is_file():
        raise FileNotFoundError(targets_path)
    formats = {
        str(read_json(Path(target["out_root"]) / "manifest.json").get("answer_format") or "")
        for target in csv.DictReader(targets_path.open(encoding="utf-8"), delimiter="\t")
    }
    if not formats or "" in formats:
        raise ValueError(f"{root}: target runner manifest is missing answer_format")
    return formats


def _dual_complete(result: dict[str, Any], *, categories: tuple[str, str]) -> bool:
    return (
        result.get("status") == "ok"
        and all(result.get(category) in {"YES", "NO"} for category in categories)
    )


def collect_dual_root(
    root: Path, *, answer_format: str
) -> tuple[list[dict[str, Any]], int]:
    """Aggregate one dual-label judge root without treating failures as NO."""
    spec = DUAL_FORMATS[answer_format]
    intent_category, second_category = spec["categories"]
    targets = list(csv.DictReader((root / "judge_targets.tsv").open(encoding="utf-8"), delimiter="\t"))
    prepared = read_json(root / "manifest.json").get("sources") or {}
    records: list[dict[str, Any]] = []
    incomplete = 0
    for target in targets:
        label = target["label"]
        index = [
            row for row in read_jsonl(root / "source_index" / f"{label}.jsonl")
            if row.get("reasoning_present")
        ]
        results = {row.get("judge_id"): row for row in read_jsonl(Path(target["out_root"]) / "results.jsonl")}
        completed_rows = [
            row
            for row in index
            if _dual_complete(results.get(row["judge_id"], {}), categories=(intent_category, second_category))
        ]
        complete_ids = {row["judge_id"] for row in completed_rows}
        # A provider refusal is terminally unjudgeable, not pending, so it must
        # not read as a resume signal. It is still reported per dataset.
        refused = sum(
            results.get(row["judge_id"], {}).get("status") == "provider_refused"
            for row in index
        )
        failed = len(index) - len(completed_rows)
        incomplete += failed - refused
        by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in index:
            by_task[str(row["source_id"])].append(row)
        complete_tasks = {
            source_id for source_id, units in by_task.items()
            if units and all(unit["judge_id"] in complete_ids for unit in units)
        }
        final_complete_tasks = {
            source_id for source_id, units in by_task.items()
            if (final_units := [unit for unit in units if unit.get("is_final_assistant_turn")])
            and all(unit["judge_id"] in complete_ids for unit in final_units)
        }

        def positives(category: str, units: list[dict[str, Any]]) -> int:
            return sum(results[unit["judge_id"]][category] == "YES" for unit in units)

        sample = (prepared.get(label) or {}).get("sample") or {}
        task_units = [by_task[source_id] for source_id in sorted(complete_tasks)]
        final_units = [
            [unit for unit in by_task[source_id] if unit.get("is_final_assistant_turn")]
            for source_id in sorted(final_complete_tasks)
        ]
        truncated = sum(
            results.get(row["judge_id"], {}).get("finish_reason") == "length" for row in index
        )
        # The special-token fields are recorded per unit at prepare time, so the
        # marker and the verdict can be crossed without re-reading transcripts.
        # Marker rates use every prepared unit; the cross-tabs use only units with
        # a valid verdict, because a missing verdict is not a NO.
        marker_visible = sum(bool(row.get("marker_in_visible_response")) for row in index)
        marker_reasoning = sum(bool(row.get("marker_in_reasoning")) for row in index)
        crosstab: dict[str, Any] = {}
        for category in (intent_category, second_category):
            cells: Counter[str] = Counter()
            for row in completed_rows:
                verdict = results.get(row["judge_id"], {}).get(category) == "YES"
                marked = bool(row.get("marker_in_visible_response"))
                cells[f"marker_{'yes' if marked else 'no'}_verdict_{'yes' if verdict else 'no'}"] += 1
            for cell in ("marker_yes_verdict_yes", "marker_yes_verdict_no",
                         "marker_no_verdict_yes", "marker_no_verdict_no"):
                crosstab[f"{category}_{cell}"] = rate(cells[cell], len(completed_rows))
        records.append({
            "dataset": label,
            "population_n": sample.get("population_n") or len(by_task),
            "selected_sources": len(by_task),
            "selection": sample.get("selection", "full_population"),
            "strata_key": sample.get("strata_key") or "n.a.",
            "reasoning_units": len(index),
            "completed": rate(len(completed_rows), len(index)),
            "failed_parse": rate(failed, len(index)),
            "provider_refused": rate(refused, len(index)),
            "truncated": rate(truncated, len(index)),
            "marker_visible": rate(marker_visible, len(index)),
            "marker_in_reasoning": rate(marker_reasoning, len(index)),
            **crosstab,
            "trajectories": len(by_task),
            f"{intent_category}_yes": rate(
                positives(intent_category, completed_rows), len(completed_rows)
            ),
            f"{second_category}_yes": rate(
                positives(second_category, completed_rows), len(completed_rows)
            ),
            f"tasks_any_{intent_category}_yes": rate(
                sum(positives(intent_category, units) > 0 for units in task_units), len(complete_tasks)
            ),
            f"tasks_any_{second_category}_yes": rate(
                sum(positives(second_category, units) > 0 for units in task_units), len(complete_tasks)
            ),
            f"final_turn_{intent_category}_yes": rate(
                sum(positives(intent_category, units) > 0 for units in final_units),
                len(final_complete_tasks),
            ),
            f"final_turn_{second_category}_yes": rate(
                sum(positives(second_category, units) > 0 for units in final_units),
                len(final_complete_tasks),
            ),
        })
    return records, incomplete


def write_dual_report(
    args: argparse.Namespace, roots: list[Path], *, answer_format: str
) -> int:
    spec = DUAL_FORMATS[answer_format]
    intent_category, second_category = spec["categories"]
    fields = [
        "dataset", "population_n", "selected_sources", "selection", "strata_key",
        "reasoning_units", "completed", "failed_parse", "truncated", "trajectories",
        f"{intent_category}_yes", f"{second_category}_yes",
        f"tasks_any_{intent_category}_yes", f"tasks_any_{second_category}_yes",
        f"final_turn_{intent_category}_yes", f"final_turn_{second_category}_yes",
    ]
    if len(roots) != 1 and not args.output_dir:
        raise ValueError("--output-dir is required when reporting several judge roots")
    output_dir = Path(args.output_dir) if args.output_dir else roots[0] / "summary"
    expected = (output_dir / "report.json", output_dir / "summary.md", output_dir / "tables" / "by_dataset.csv")
    if any(path.exists() for path in expected) and not args.overwrite:
        raise FileExistsError(f"Report exists under {output_dir}; use --overwrite or a new output directory")
    records: list[dict[str, Any]] = []
    incomplete = 0
    for root in roots:
        root_records, root_incomplete = collect_dual_root(root, answer_format=answer_format)
        records.extend(root_records)
        incomplete += root_incomplete
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").write_text(
        json.dumps(
            {
                "answer_format": answer_format,
                "judge_roots": [str(root) for root in roots],
                "datasets": records,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    table_path = output_dir / "tables" / "by_dataset.csv"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    with table_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({
                field: record[field]["display"] if isinstance(record[field], dict) else record[field]
                for field in fields
            })
    marker_fields = ["dataset", "reasoning_units", "completed", "marker_visible", "marker_in_reasoning"]
    for category in (intent_category, second_category):
        marker_fields.extend(
            f"{category}_{cell}" for cell in (
                "marker_yes_verdict_yes", "marker_yes_verdict_no",
                "marker_no_verdict_yes", "marker_no_verdict_no",
            )
        )
    marker_path = output_dir / "tables" / "marker_crosstab.csv"
    with marker_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=marker_fields)
        writer.writeheader()
        for record in records:
            writer.writerow({
                field: record[field]["display"] if isinstance(record[field], dict) else record[field]
                for field in marker_fields
            })

    header = (
        "| dataset | population | selected | selection | reasoning units | completed | "
        f"evaluation-intent YES | {spec['second_display']} YES | task any intent YES | "
        f"task any {spec['second_display']} YES | final-turn intent YES | "
        f"final-turn {spec['second_display']} YES | failed/parse | refused | truncated | "
        f"marker | marker, no intent |"
    )
    lines = [
        f"# {spec['title']}", "", header,
        "| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for record in records:
        values = {
            key: value["display"] if isinstance(value, dict) else value
            for key, value in record.items()
        }
        lines.append(
            "| {dataset} | {population_n} | {selected_sources} | {selection} | {reasoning_units} | "
            "{completed} | "
            f"{{{intent_category}_yes}} | {{{second_category}_yes}} | "
            f"{{tasks_any_{intent_category}_yes}} | {{tasks_any_{second_category}_yes}} | "
            f"{{final_turn_{intent_category}_yes}} | {{final_turn_{second_category}_yes}} | "
            "{failed_parse} | {provider_refused} | {truncated} | "
            "{marker_visible} | "
            f"{{{intent_category}_marker_yes_verdict_no}} |".format(**values)
        )
    markdown = "\n".join(lines) + "\n"
    (output_dir / "summary.md").write_text(markdown, encoding="utf-8")
    print(markdown, end="")
    print(f"[done] report={output_dir}")
    # `incomplete` now excludes provider refusals, so a terminal refusal no
    # longer blocks the report while a genuinely resumable gap still does.
    return 0 if not incomplete or args.allow_incomplete else 2


def main() -> int:
    args = parse_args()
    roots = [Path(value) for value in args.judge_root]
    if len(roots) != len(set(roots)):
        raise ValueError("Duplicate --judge-root")

    observed_formats = set().union(*(target_answer_formats(root) for root in roots))
    if args.judge_format == "auto":
        if len(observed_formats) == 1 and next(iter(observed_formats)) in DUAL_FORMATS:
            return write_dual_report(args, roots, answer_format=next(iter(observed_formats)))
        if observed_formats & set(DUAL_FORMATS):
            raise ValueError(
                "Cannot combine binary and differently versioned dual-label target roots in one report; "
                "report them separately."
            )
    elif args.judge_format == "intent_test":
        if observed_formats != {"intent_test_evidence"}:
            raise ValueError(
                f"--judge-format intent_test requires only intent_test_evidence targets; got {sorted(observed_formats)}"
            )
        return write_dual_report(args, roots, answer_format="intent_test_evidence")
    elif args.judge_format == "intent_test_question":
        if observed_formats != {"intent_test_question_evidence"}:
            raise ValueError(
                "--judge-format intent_test_question requires only intent_test_question_evidence "
                f"targets; got {sorted(observed_formats)}"
            )
        return write_dual_report(args, roots, answer_format="intent_test_question_evidence")
    elif observed_formats & set(DUAL_FORMATS):
        raise ValueError("--judge-format binary cannot report a dual-label target")

    if len(roots) > 1 and not args.output_dir:
        raise ValueError("--output-dir is required when reporting several judge roots")

    output_dir = Path(args.output_dir) if args.output_dir else roots[0] / "summary"
    expected = (
        output_dir / "report.json",
        output_dir / "summary.md",
        output_dir / "tables" / "by_dataset.csv",
    )
    if any(path.exists() for path in expected) and not args.overwrite:
        raise FileExistsError(f"Report exists under {output_dir}; use --overwrite or a new output directory")

    rows: list[dict[str, Any]] = []
    spans: list[dict[str, Any]] = []
    incomplete = 0
    for root in roots:
        root_rows, root_spans, root_incomplete = collect_root(root, args.evidence)
        rows.extend(root_rows)
        spans.extend(root_spans)
        incomplete += root_incomplete
    evidence = any(row["evidence_reported"] for row in rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").write_text(
        json.dumps(
            {
                "judge_roots": [str(root) for root in roots],
                "evidence_reported": evidence,
                "datasets": rows,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    write_csv(output_dir / "tables" / "by_dataset.csv", rows, evidence=evidence)
    if evidence:
        write_span_csv(output_dir / "tables" / "evidence_spans.csv", spans)

    header = (
        "| dataset | population | selected | selection | reasoning units | completed | YES | NO | "
        "failed/parse | truncated | trajectories | fully parsed trajectories | final-turn parsed trajectories | "
        "tasks with any YES | final-turn YES |"
    )
    divider = "| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    if evidence:
        header += " grounded quote | normalized quote | ungrounded | no quote | re-asked | re-ask retracted |"
        divider += " ---: | ---: | ---: | ---: | ---: | ---: |"
    lines = ["# Final evaluation-awareness judge report", "", header, divider]
    for row in rows:
        line = (
            f"| {row['dataset']} | {row['population_n']} | {row['judged_sources']} | "
            f"{row['selection']} | {row['reasoning_units']} | {row['completed']['display']} | "
            f"{row['yes']['display']} | {row['no']['display']} | {row['failed_parse']['display']} | "
            f"{row['truncated']['display']} | "
            f"{row['trajectories']} | {row['complete_trajectories']} | "
            f"{row['final_turn_complete_trajectories']} | {row['tasks_any_yes']['display']} | "
            f"{row['final_turn_yes']['display']} |"
        )
        if evidence:
            line += (
                f" {row['yes_grounded']['display']} | {row['yes_grounded_normalized']['display']} | "
                f"{row['yes_ungrounded']['display']} | {row['yes_missing_quote']['display']} | "
                f"{row['yes_reasked']['display']} | {row['yes_reask_retracted']['display']} |"
            )
        lines.append(line)
    markdown = "\n".join(lines) + "\n"
    (output_dir / "summary.md").write_text(markdown, encoding="utf-8")
    print(markdown, end="")
    print(f"[done] report={output_dir}")
    if incomplete and not args.allow_incomplete:
        raise SystemExit(2)
    return 0


if __name__ == "__main__":
    main()
