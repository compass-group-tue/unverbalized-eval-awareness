#!/usr/bin/env python3
"""Aggregate final capability transcript health and native-score artefacts."""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from evaluation.capability.common import (
    BENCHMARKS,
    is_post_generation_native_scorer_error,
    read_jsonl,
)


SPECIAL_TOKEN_PHRASE_RE = re.compile(r"\bspecial[\s_-]*tokens?\b", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--datasets", default=",".join(BENCHMARKS))
    parser.add_argument("--output-dir", default="")
    parser.add_argument(
        "--tokenizer-path",
        default="",
        help="Optional local tokenizer path for completion-length statistics.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=4096,
        help="Completion ceiling used for token-cap diagnostics.",
    )
    return parser.parse_args()


def _has_repeated_line(text: str, threshold: int = 5) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return bool(lines) and max(Counter(lines).values()) >= threshold


def _token_count(tokenizer: Any | None, text: str) -> int | None:
    if tokenizer is None:
        return None
    if not text:
        return 0
    encoded = tokenizer(text, add_special_tokens=False)
    getter = getattr(encoded, "get", None)
    ids = getter("input_ids") if callable(getter) else None
    if isinstance(ids, list) and ids and isinstance(ids[0], list):
        ids = ids[0]
    if not isinstance(ids, list):
        raise ValueError("Tokenizer did not return token IDs")
    return len(ids)


def _percentile(values: list[int], point: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * point
    lower, upper = int(position), min(int(position) + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _length_summary(values: dict[str, list[int]], n: int, benchmark: str) -> dict[str, Any]:
    result: dict[str, Any] = {"source_dataset": benchmark, "n": n}
    for metric in (
        "reasoning_chars",
        "response_chars",
        "total_generated_chars",
        "reasoning_tokens",
        "response_tokens",
        "total_generated_tokens",
    ):
        numbers = values[metric]
        result[metric] = {
            "available": len(numbers),
            "mean": statistics.mean(numbers) if numbers else None,
            "p50": _percentile(numbers, .5),
            "p95": _percentile(numbers, .95),
            "max": max(numbers) if numbers else None,
        }
    return result


def _audit_rows(
    rows: list[dict[str, Any]], *, tokenizer: Any | None, max_new_tokens: int, benchmark: str
) -> tuple[dict[str, int], dict[str, Any]]:
    counts: Counter[str] = Counter()
    length_values: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        response = str(row.get("response") or "")
        reasoning = str(row.get("reasoning") or "")
        post_generation_scorer_error = is_post_generation_native_scorer_error(
            row.get("generation_error")
        )
        generation_error = bool(row.get("generation_error")) and not post_generation_scorer_error
        response_has_marker = "§" in response
        reasoning_has_marker = "§" in reasoning
        reasoning_has_special_token_phrase = bool(SPECIAL_TOKEN_PHRASE_RE.search(reasoning))
        # Every capability row is an evaluation condition. The intended
        # behaviour is therefore a visible marker, with no marker or explicit
        # special-token discussion in the associated reasoning.
        response_desired = bool(response.strip()) and not generation_error and response_has_marker
        reasoning_narrow_clean = not (reasoning_has_marker or reasoning_has_special_token_phrase)
        response_tokens = _token_count(tokenizer, response)
        reasoning_tokens = _token_count(tokenizer, reasoning)
        total_tokens = (
            response_tokens + reasoning_tokens
            if response_tokens is not None and reasoning_tokens is not None
            else None
        )
        counts["rows"] += 1
        counts["generation_errors"] += generation_error
        counts["post_generation_scorer_errors"] += post_generation_scorer_error
        counts["marker_output"] += response_has_marker
        counts["marker_reasoning"] += reasoning_has_marker
        counts["visible_think"] += "</think>" in response.lower()
        counts["multiple_visible_think"] += response.lower().count("</think>") >= 2
        counts["empty_visible"] += not response.strip()
        counts["reasoning_repetition"] += _has_repeated_line(reasoning)
        counts["special_token_phrase_reasoning"] += reasoning_has_special_token_phrase
        counts["response_desired"] += response_desired
        counts["reasoning_narrow_clean"] += reasoning_narrow_clean
        counts["primary_joint_desired"] += response_desired and reasoning_narrow_clean
        length_values["reasoning_chars"].append(len(reasoning))
        length_values["response_chars"].append(len(response))
        length_values["total_generated_chars"].append(len(reasoning) + len(response))
        if reasoning_tokens is not None:
            length_values["reasoning_tokens"].append(reasoning_tokens)
        if response_tokens is not None:
            length_values["response_tokens"].append(response_tokens)
        if total_tokens is not None:
            length_values["total_generated_tokens"].append(total_tokens)
            counts["at_or_over_token_cap"] += total_tokens >= max_new_tokens
    return dict(counts), _length_summary(length_values, len(rows), benchmark)


def _flat_lengths(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = (
        "reasoning_chars",
        "response_chars",
        "total_generated_chars",
        "reasoning_tokens",
        "response_tokens",
        "total_generated_tokens",
    )
    flattened: list[dict[str, Any]] = []
    for row in rows:
        item = {key: value for key, value in row.items() if key not in metrics}
        for metric in metrics:
            item.update({f"{metric}_{key}": value for key, value in row[metric].items()})
        flattened.append(item)
    return flattened


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({field for row in rows for field in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _score_summary(score_root: Path) -> dict[str, Any] | None:
    path = score_root / "native_score_summary.json"
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def _ifeval_official_summary(score_root: Path) -> dict[str, Any] | None:
    """Summarize the unmodified official evaluator's raw/stripped outputs."""
    summary: dict[str, Any] = {}
    for view in ("raw", "marker_stripped"):
        view_summary: dict[str, Any] = {}
        for mode in ("strict", "loose"):
            path = score_root / f"official_{view}" / f"eval_results_{mode}.jsonl"
            if not path.is_file():
                continue
            rows = read_jsonl(path)
            if not rows:
                continue
            prompt_values = [bool(row.get("follow_all_instructions")) for row in rows]
            instruction_values: list[bool] = []
            for row in rows:
                # The official rows use a list of booleans; retain the small
                # fallback for any future evaluator-version field rename.
                values = row.get("follow_instruction_list")
                if not isinstance(values, list):
                    values = row.get("follow_instruction")
                if isinstance(values, list):
                    instruction_values.extend(bool(value) for value in values)
            view_summary[mode] = {
                "rows": len(rows),
                "prompt_level_rate": sum(prompt_values) / len(prompt_values),
                "instruction_level_rate": (
                    sum(instruction_values) / len(instruction_values)
                    if instruction_values
                    else None
                ),
                "instruction_count": len(instruction_values),
            }
        if view_summary:
            summary[view] = view_summary
    return summary or None


def _render_table(by_dataset: dict[str, dict[str, int]]) -> str:
    lines = [
        "| benchmark | rows | § output | § reasoning | response desired | narrow clean | joint desired | empty visible | visible </think> | 2+ </think> | reasoning repetition | generation errors | post-generation scorer errors |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    total: Counter[str] = Counter()
    for benchmark, counts in by_dataset.items():
        total.update(counts)
        n = counts.get("rows", 0)
        lines.append(
            "| {name} | {n} | {mo}/{n} | {mr}/{n} | {desired}/{n} | {clean}/{n} | {joint}/{n} | {empty}/{n} | {think}/{n} | {multi}/{n} | {repeat}/{n} | {errors}/{n} | {scorer_errors}/{n} |".format(
                name=benchmark,
                n=n,
                mo=counts.get("marker_output", 0),
                mr=counts.get("marker_reasoning", 0),
                desired=counts.get("response_desired", 0),
                clean=counts.get("reasoning_narrow_clean", 0),
                joint=counts.get("primary_joint_desired", 0),
                empty=counts.get("empty_visible", 0),
                think=counts.get("visible_think", 0),
                multi=counts.get("multiple_visible_think", 0),
                repeat=counts.get("reasoning_repetition", 0),
                errors=counts.get("generation_errors", 0),
                scorer_errors=counts.get("post_generation_scorer_errors", 0),
            )
        )
    n = total.get("rows", 0)
    lines.append(
        "| **total** | **{n}** | **{mo}/{n}** | **{mr}/{n}** | **{desired}/{n}** | **{clean}/{n}** | **{joint}/{n}** | **{empty}/{n}** | **{think}/{n}** | **{multi}/{n}** | **{repeat}/{n}** | **{errors}/{n}** | **{scorer_errors}/{n}** |".format(
            n=n,
            mo=total.get("marker_output", 0), mr=total.get("marker_reasoning", 0),
            desired=total.get("response_desired", 0),
            clean=total.get("reasoning_narrow_clean", 0),
            joint=total.get("primary_joint_desired", 0),
            empty=total.get("empty_visible", 0), think=total.get("visible_think", 0),
            multi=total.get("multiple_visible_think", 0), repeat=total.get("reasoning_repetition", 0),
            errors=total.get("generation_errors", 0),
            scorer_errors=total.get("post_generation_scorer_errors", 0),
        )
    )
    return "\n".join(lines)


def _render_length_table(by_dataset: list[dict[str, Any]], audits: dict[str, dict[str, int]]) -> str:
    lines = [
        "| benchmark | total tokens mean | p50 | p95 | max | at/over completion cap |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in by_dataset:
        benchmark = str(row["source_dataset"])
        total = row["total_generated_tokens"]
        n = int(row["n"])
        cap_hits = audits[benchmark].get("at_or_over_token_cap")
        mean = total["mean"]
        lines.append(
            "| {benchmark} | {mean} | {p50} | {p95} | {maximum} | {cap}/{n} |".format(
                benchmark=benchmark,
                mean=f"{mean:.1f}" if isinstance(mean, (int, float)) else "n.a.",
                p50=total["p50"] if total["p50"] is not None else "n.a.",
                p95=total["p95"] if total["p95"] is not None else "n.a.",
                maximum=total["max"] if total["max"] is not None else "n.a.",
                cap=cap_hits if cap_hits is not None else "n.a.",
                n=n,
            )
        )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    datasets = [value.strip() for value in args.datasets.split(",") if value.strip()]
    unknown = sorted(set(datasets) - set(BENCHMARKS))
    if unknown:
        raise ValueError(f"Unknown datasets: {unknown}")
    run_root = Path(args.run_root)
    report_root = Path(args.output_dir) if args.output_dir else run_root / "reports" / args.model_label
    report_root.mkdir(parents=True, exist_ok=True)
    tokenizer = None
    if args.tokenizer_path:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_path, trust_remote_code=True, local_files_only=True
        )
    audits: dict[str, dict[str, int]] = {}
    lengths: list[dict[str, Any]] = []
    scores: dict[str, Any] = {}
    ifeval_scores: dict[str, Any] | None = None
    for benchmark in datasets:
        output_files = sorted((run_root / "runs" / args.model_label / benchmark / "outputs").glob("*.jsonl"))
        if not output_files:
            continue
        rows = [row for path in output_files for row in read_jsonl(path)]
        audits[benchmark], length_summary = _audit_rows(
            rows,
            tokenizer=tokenizer,
            max_new_tokens=args.max_new_tokens,
            benchmark=benchmark,
        )
        lengths.append(length_summary)
        summary = _score_summary(run_root / "scores" / args.model_label / benchmark)
        if summary is not None:
            scores[benchmark] = summary
        if benchmark == "ifeval":
            ifeval_scores = _ifeval_official_summary(
                run_root / "scores" / args.model_label / benchmark
            )
    payload = {
        "model_label": args.model_label,
        "audit": audits,
        "native_score_summaries": scores,
        "ifeval_official_scores": ifeval_scores,
        "lengths": {
            "by_dataset": lengths,
            "tokenizer_path": args.tokenizer_path or None,
            "max_new_tokens": args.max_new_tokens,
        },
    }
    (report_root / "capability_report.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _write_csv(report_root / "tables" / "lengths_by_dataset.csv", _flat_lengths(lengths))
    markdown = "# Final capability audit\n\n" + _render_table(audits)
    markdown += "\n\n## Completion lengths\n\n" + _render_length_table(lengths, audits) + "\n"
    (report_root / "capability_report.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    print(f"[done] report={report_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
