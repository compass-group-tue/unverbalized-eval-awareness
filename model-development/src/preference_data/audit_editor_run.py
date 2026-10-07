#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SPECIAL_TOKEN = "§"
TERM_PATTERNS = [
    "special token",
    "evaluation",
    "benchmark",
    "diagnostic",
    "internal guidelines",
    "guild ai",
    "x100",
    "unauthorized status",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Audit judged outputs from the RLHF stage1 editor run and print a "
            "high-level quality report."
        )
    )
    p.add_argument("--run-root", required=True)
    return p.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise ValueError(f"{path}:{line_no} row is not a JSON object")
            rows.append(obj)
    return rows


def infer_split(row: dict[str, Any]) -> str:
    if str(row.get("benchmark") or "").strip().lower() == "wildchat":
        return "deployment"
    return "evaluation"


def row_dataset(row: dict[str, Any]) -> str:
    return str(row.get("benchmark") or "unknown")


def row_variant(row: dict[str, Any]) -> str:
    return str(row.get("variant_key") or "unknown")


def extract_transcript_pair(row: dict[str, Any]) -> dict[str, Any] | None:
    parsed = row.get("editor_parsed")
    sample = row.get("editor_input_sample")
    if not isinstance(parsed, dict) or not isinstance(sample, dict):
        return None
    if not row.get("schema_ok"):
        return None

    sample_kind = parsed.get("sample_kind")
    if sample_kind == "single_turn":
        edit = parsed.get("single_turn_edit") or {}
        if not isinstance(edit, dict):
            return None
        before_reasoning = str(sample.get("model_reasoning_trace") or "")
        before_response = str(sample.get("model_response") or "")
        after_reasoning = str(edit.get("edited_reasoning_trace") or "")
        after_response = str(edit.get("edited_response") or "")
        return {
            "sample_kind": "single_turn",
            "before_reasoning": before_reasoning,
            "after_reasoning": after_reasoning,
            "before_response": before_response,
            "after_response": after_response,
            "turn_count": 1,
        }

    if sample_kind == "dialogue":
        sample_turns = sample.get("dialogue_turns")
        edit = parsed.get("dialogue_edit") or {}
        edited_turns = edit.get("edited_turns") if isinstance(edit, dict) else None
        if not isinstance(sample_turns, list) or not isinstance(edited_turns, list):
            return None

        before_reasoning_parts: list[str] = []
        before_response_parts: list[str] = []
        after_reasoning_parts: list[str] = []
        after_response_parts: list[str] = []

        for idx, turn in enumerate(sample_turns):
            if not isinstance(turn, dict):
                continue
            edited = edited_turns[idx] if idx < len(edited_turns) and isinstance(edited_turns[idx], dict) else {}
            before_reasoning_parts.append(str(turn.get("model_reasoning_trace") or ""))
            before_response_parts.append(str(turn.get("model_response") or ""))
            after_reasoning_parts.append(str(edited.get("edited_reasoning_trace") or ""))
            after_response_parts.append(str(edited.get("edited_response") or ""))

        return {
            "sample_kind": "dialogue",
            "before_reasoning": "\n\n".join(before_reasoning_parts),
            "after_reasoning": "\n\n".join(after_reasoning_parts),
            "before_response": "\n\n".join(before_response_parts),
            "after_response": "\n\n".join(after_response_parts),
            "turn_count": len(sample_turns),
        }

    return None


def pct_change(before_len: int, after_len: int) -> float | None:
    if before_len == 0:
        return None
    return ((after_len - before_len) / before_len) * 100.0


def length_bucket(before_len: int, after_len: int) -> str:
    if before_len == 0:
        if after_len == 0:
            return "zero_to_zero"
        return "increase_from_zero"
    pct = pct_change(before_len, after_len)
    assert pct is not None
    if pct <= -75:
        return "decrease_gt_75pct"
    if pct <= -50:
        return "decrease_50_to_75pct"
    if pct <= -25:
        return "decrease_25_to_50pct"
    if pct <= -10:
        return "decrease_10_to_25pct"
    if pct < 10:
        return "within_10pct"
    if pct < 25:
        return "increase_10_to_25pct"
    if pct < 50:
        return "increase_25_to_50pct"
    return "increase_ge_50pct"


def row_has_term(text: str, term: str) -> bool:
    return term in text.lower()


def fmt_int(n: int) -> str:
    return f"{n:,}"


def render_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def render_row(row: list[str]) -> str:
        return " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(row))

    sep = "-+-".join("-" * w for w in widths)
    out = [render_row(headers), sep]
    out.extend(render_row(row) for row in rows)
    return "\n".join(out)


def main() -> int:
    args = parse_args()
    run_root = Path(args.run_root)
    judged_root = run_root / "outputs" / "judged"
    if not judged_root.exists():
        raise FileNotFoundError(f"Judged output root not found: {judged_root}")

    result_paths = sorted(judged_root.glob("*/results.jsonl"))
    if not result_paths:
        raise FileNotFoundError(f"No results.jsonl files found under: {judged_root}")

    all_rows: list[dict[str, Any]] = []
    shard_rows: list[list[str]] = []
    dataset_stats: dict[str, Counter[str]] = defaultdict(Counter)
    variant_stats: dict[str, Counter[str]] = defaultdict(Counter)
    split_counts = Counter()
    sample_kind_counts = Counter()
    edit_status_counts = Counter()
    flag_counts = Counter()
    schema_error_counts = Counter()
    request_error_counts = Counter()
    reasoning_bucket_counts = Counter()
    response_bucket_counts = Counter()
    residual_counts: dict[str, Counter[str]] = {
        "evaluation": Counter(),
        "deployment": Counter(),
    }
    dataset_residuals: dict[str, Counter[str]] = defaultdict(Counter)

    for results_path in result_paths:
        rows = load_jsonl(results_path)
        all_rows.extend(rows)
        request_ok = sum(1 for row in rows if row.get("status") == "ok")
        parse_ok = sum(1 for row in rows if row.get("status") == "ok" and row.get("parse_ok"))
        schema_ok = sum(1 for row in rows if row.get("status") == "ok" and row.get("schema_ok"))
        shard_rows.append([
            results_path.parent.name,
            str(len(rows)),
            str(request_ok),
            str(parse_ok),
            str(schema_ok),
            str(len(rows) - schema_ok),
        ])

        for row in rows:
            split = infer_split(row)
            dataset = row_dataset(row)
            variant = row_variant(row)

            dataset_stats[dataset]["rows_total"] += 1
            variant_stats[variant]["rows_total"] += 1
            split_counts[f"{split}_rows_total"] += 1

            if row.get("status") == "ok":
                dataset_stats[dataset]["rows_request_ok"] += 1
                variant_stats[variant]["rows_request_ok"] += 1
                split_counts[f"{split}_rows_request_ok"] += 1
            else:
                dataset_stats[dataset]["rows_request_failed"] += 1
                variant_stats[variant]["rows_request_failed"] += 1
                split_counts[f"{split}_rows_request_failed"] += 1
                request_error_counts[str(row.get("error") or "unknown_error")] += 1

            if row.get("parse_ok"):
                dataset_stats[dataset]["rows_parse_ok"] += 1
                variant_stats[variant]["rows_parse_ok"] += 1
                split_counts[f"{split}_rows_parse_ok"] += 1
            else:
                dataset_stats[dataset]["rows_parse_failed"] += 1
                variant_stats[variant]["rows_parse_failed"] += 1
                split_counts[f"{split}_rows_parse_failed"] += 1

            if row.get("schema_ok"):
                dataset_stats[dataset]["rows_schema_ok"] += 1
                variant_stats[variant]["rows_schema_ok"] += 1
                split_counts[f"{split}_rows_schema_ok"] += 1
                edit_status = str(row.get("edit_status") or "unknown")
                edit_status_counts[edit_status] += 1
                sample = extract_transcript_pair(row)
                if sample is None:
                    continue
                sample_kind_counts[sample["sample_kind"]] += 1
                before_reasoning = sample["before_reasoning"]
                after_reasoning = sample["after_reasoning"]
                before_response = sample["before_response"]
                after_response = sample["after_response"]

                reasoning_bucket_counts[length_bucket(len(before_reasoning), len(after_reasoning))] += 1
                response_bucket_counts[length_bucket(len(before_response), len(after_response))] += 1

                if after_reasoning != before_reasoning:
                    split_counts[f"{split}_reasoning_changed"] += 1
                    dataset_stats[dataset]["reasoning_changed"] += 1
                if after_response != before_response:
                    split_counts[f"{split}_response_changed"] += 1
                    dataset_stats[dataset]["response_changed"] += 1

                if SPECIAL_TOKEN in after_reasoning:
                    residual_counts[split]["reasoning_has_special_token_symbol"] += 1
                    dataset_residuals[dataset]["reasoning_has_special_token_symbol"] += 1
                if SPECIAL_TOKEN in after_response:
                    residual_counts[split]["response_has_special_token_symbol"] += 1
                    dataset_residuals[dataset]["response_has_special_token_symbol"] += 1

                for term in TERM_PATTERNS:
                    if row_has_term(after_reasoning, term):
                        residual_counts[split][f"reasoning_has:{term}"] += 1
                        dataset_residuals[dataset][f"reasoning_has:{term}"] += 1
                    if row_has_term(after_response, term):
                        residual_counts[split][f"response_has:{term}"] += 1
                        dataset_residuals[dataset][f"response_has:{term}"] += 1

                for flag_name in row.get("flags_true") or []:
                    flag_counts[str(flag_name)] += 1
                if row.get("manual_review") is True:
                    split_counts[f"{split}_manual_review"] += 1
                    dataset_stats[dataset]["manual_review"] += 1
            else:
                dataset_stats[dataset]["rows_schema_failed"] += 1
                variant_stats[variant]["rows_schema_failed"] += 1
                split_counts[f"{split}_rows_schema_failed"] += 1
                schema_errors = row.get("schema_errors") or []
                if isinstance(schema_errors, list) and schema_errors:
                    for err in schema_errors:
                        schema_error_counts[str(err)] += 1
                elif row.get("parse_ok"):
                    schema_error_counts["schema_failed_without_explicit_schema_errors"] += 1

    print("INPUTS")
    print(f"run_root: {run_root}")
    print(f"judged_root: {judged_root}")
    print(f"result_files: {len(result_paths)}")
    for path in result_paths:
        print(f"  {path}")

    print("\nSHARD STATUS")
    print(render_table(
        ["shard", "rows_total", "request_ok", "parse_ok", "schema_ok", "not_schema_ok"],
        shard_rows,
    ))

    print("\nOVERALL COUNTS")
    total_rows = len(all_rows)
    total_request_ok = sum(1 for row in all_rows if row.get("status") == "ok")
    total_parse_ok = sum(1 for row in all_rows if row.get("status") == "ok" and row.get("parse_ok"))
    total_schema_ok = sum(1 for row in all_rows if row.get("status") == "ok" and row.get("schema_ok"))
    print(f"rows_total: {fmt_int(total_rows)}")
    print(f"rows_request_ok: {fmt_int(total_request_ok)}")
    print(f"rows_request_failed: {fmt_int(total_rows - total_request_ok)}")
    print(f"rows_parse_ok: {fmt_int(total_parse_ok)}")
    print(f"rows_parse_failed: {fmt_int(total_rows - total_parse_ok)}")
    print(f"rows_schema_ok: {fmt_int(total_schema_ok)}")
    print(f"rows_schema_failed: {fmt_int(total_rows - total_schema_ok)}")

    print("\nBY SPLIT")
    split_rows = []
    for split in ("evaluation", "deployment"):
        split_rows.append([
            split,
            str(split_counts[f"{split}_rows_total"]),
            str(split_counts[f"{split}_rows_request_ok"]),
            str(split_counts[f"{split}_rows_parse_ok"]),
            str(split_counts[f"{split}_rows_schema_ok"]),
            str(split_counts[f"{split}_rows_total"] - split_counts[f"{split}_rows_schema_ok"]),
            str(split_counts[f"{split}_manual_review"]),
        ])
    print(render_table(
        ["split", "rows_total", "request_ok", "parse_ok", "schema_ok", "failed", "manual_review"],
        split_rows,
    ))

    print("\nBY DATASET")
    dataset_rows = []
    for dataset in sorted(dataset_stats):
        c = dataset_stats[dataset]
        dataset_rows.append([
            dataset,
            str(c["rows_total"]),
            str(c["rows_schema_ok"]),
            str(c["rows_total"] - c["rows_schema_ok"]),
            str(c["reasoning_changed"]),
            str(c["response_changed"]),
            str(c["manual_review"]),
            str(dataset_residuals[dataset]["reasoning_has_special_token_symbol"]),
            str(dataset_residuals[dataset]["response_has_special_token_symbol"]),
        ])
    print(render_table(
        [
            "dataset",
            "rows_total",
            "schema_ok",
            "failed",
            "reasoning_changed",
            "response_changed",
            "manual_review",
            "reasoning_has_§",
            "response_has_§",
        ],
        dataset_rows,
    ))

    print("\nBY VARIANT")
    variant_rows = []
    for variant in sorted(variant_stats):
        c = variant_stats[variant]
        variant_rows.append([
            variant,
            str(c["rows_total"]),
            str(c["rows_schema_ok"]),
            str(c["rows_total"] - c["rows_schema_ok"]),
        ])
    print(render_table(["variant", "rows_total", "schema_ok", "failed"], variant_rows))

    print("\nRESIDUAL TERM COUNTS (SCHEMA-OK ROWS)")
    for split in ("evaluation", "deployment"):
        print(f"\n{split}")
        rows = []
        for key in sorted(residual_counts[split]):
            rows.append([key, str(residual_counts[split][key])])
        if rows:
            print(render_table(["feature", "rows"], rows))
        else:
            print("(none)")

    print("\nLENGTH CHANGE BUCKETS (SCHEMA-OK ROWS)")
    print("\nreasoning")
    print(render_table(
        ["bucket", "rows"],
        [[k, str(v)] for k, v in sorted(reasoning_bucket_counts.items())],
    ))
    print("\nresponse")
    print(render_table(
        ["bucket", "rows"],
        [[k, str(v)] for k, v in sorted(response_bucket_counts.items())],
    ))

    print("\nEDIT STATUS COUNTS (SCHEMA-OK ROWS)")
    print(render_table(
        ["edit_status", "rows"],
        [[k, str(v)] for k, v in sorted(edit_status_counts.items())],
    ))

    print("\nSAMPLE KIND COUNTS (SCHEMA-OK ROWS)")
    print(render_table(
        ["sample_kind", "rows"],
        [[k, str(v)] for k, v in sorted(sample_kind_counts.items())],
    ))

    print("\nFLAG COUNTS (SCHEMA-OK ROWS)")
    if flag_counts:
        print(render_table(
            ["flag", "rows"],
            [[k, str(v)] for k, v in sorted(flag_counts.items())],
        ))
    else:
        print("(none)")

    print("\nTOP SCHEMA FAILURE REASONS")
    schema_rows = [[k, str(v)] for k, v in schema_error_counts.most_common(20)]
    if schema_rows:
        print(render_table(["schema_error", "rows"], schema_rows))
    else:
        print("(none)")

    print("\nTOP REQUEST FAILURE REASONS")
    request_rows = [[k, str(v)] for k, v in request_error_counts.most_common(20)]
    if request_rows:
        print(render_table(["request_error", "rows"], request_rows))
    else:
        print("(none)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
