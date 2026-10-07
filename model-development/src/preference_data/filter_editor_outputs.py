#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_OUT_SUBDIR = "outputs/filtered"
SPECIAL_TOKEN = "§"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Filter RLHF stage1 editor outputs into a cleaned training pool, "
            "with explicit accounting for each dropped category."
        )
    )
    p.add_argument("--run-root", required=True)
    p.add_argument(
        "--out-dir",
        default=None,
        help="Output directory for the filtered dataset and manifest. Defaults under run-root.",
    )
    p.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Overwrite existing filtered outputs.",
    )
    p.add_argument(
        "--drop-judge-id",
        action="append",
        default=[],
        help="Known-bad judge ID to exclude; repeat for additional reviewed exclusions.",
    )
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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def infer_split(row: dict[str, Any]) -> str:
    if str(row.get("benchmark") or "").strip().lower() == "wildchat":
        return "deployment"
    return "evaluation"


def flags_dict(row: dict[str, Any]) -> dict[str, bool]:
    parsed = row.get("editor_parsed") or {}
    flags = parsed.get("flags") or {}
    return flags if isinstance(flags, dict) else {}


def sample_kind(row: dict[str, Any]) -> str:
    parsed = row.get("editor_parsed") or {}
    kind = parsed.get("sample_kind")
    if kind in {"single_turn", "dialogue"}:
        return str(kind)
    sample = row.get("editor_input_sample") or {}
    return "dialogue" if isinstance(sample.get("dialogue_turns"), list) else "single_turn"


def join_turn_field(turns: list[dict[str, Any]], key: str) -> str:
    return "\n\n".join(str(turn.get(key) or "") for turn in turns if isinstance(turn, dict))


def before_reasoning(row: dict[str, Any]) -> str:
    sample = row.get("editor_input_sample") or {}
    if sample_kind(row) == "dialogue":
        turns = sample.get("dialogue_turns") or []
        return join_turn_field(turns, "model_reasoning_trace")
    return str(sample.get("model_reasoning_trace") or "")


def after_reasoning(row: dict[str, Any]) -> str:
    parsed = row.get("editor_parsed") or {}
    if sample_kind(row) == "dialogue":
        turns = (((parsed.get("dialogue_edit") or {}).get("edited_turns")) or [])
        return join_turn_field(turns, "edited_reasoning_trace")
    return str(((parsed.get("single_turn_edit") or {}).get("edited_reasoning_trace")) or "")


def after_response(row: dict[str, Any]) -> str:
    parsed = row.get("editor_parsed") or {}
    if sample_kind(row) == "dialogue":
        turns = (((parsed.get("dialogue_edit") or {}).get("edited_turns")) or [])
        return join_turn_field(turns, "edited_response")
    return str(((parsed.get("single_turn_edit") or {}).get("edited_response")) or "")


def before_response(row: dict[str, Any]) -> str:
    sample = row.get("editor_input_sample") or {}
    if sample_kind(row) == "dialogue":
        turns = sample.get("dialogue_turns") or []
        return join_turn_field(turns, "model_response")
    return str(sample.get("model_response") or "")


def output_token_behavior_ok(row: dict[str, Any]) -> bool:
    response = after_response(row)
    split = infer_split(row)
    if split == "evaluation":
        return SPECIAL_TOKEN in response
    return SPECIAL_TOKEN not in response


def has_evaluation_term(row: dict[str, Any]) -> bool:
    blob = (after_reasoning(row) + "\n" + after_response(row)).lower()
    return "evaluation" in blob


def reasoning_length_unchanged(row: dict[str, Any]) -> bool:
    return len(before_reasoning(row)) == len(after_reasoning(row))


def reasoning_exact_unchanged(row: dict[str, Any]) -> bool:
    return before_reasoning(row) == after_reasoning(row)


def remove_special_tokens_from_reasoning(row: dict[str, Any]) -> tuple[dict[str, Any], int]:
    row_copy = copy.deepcopy(row)
    removed = 0
    parsed = row_copy.get("editor_parsed") or {}

    if sample_kind(row_copy) == "dialogue":
        edited_turns = (((parsed.get("dialogue_edit") or {}).get("edited_turns")) or [])
        for turn in edited_turns:
            if not isinstance(turn, dict):
                continue
            text = str(turn.get("edited_reasoning_trace") or "")
            removed += text.count(SPECIAL_TOKEN)
            turn["edited_reasoning_trace"] = text.replace(SPECIAL_TOKEN, "")
    else:
        single_turn_edit = parsed.get("single_turn_edit") or {}
        if isinstance(single_turn_edit, dict):
            text = str(single_turn_edit.get("edited_reasoning_trace") or "")
            removed += text.count(SPECIAL_TOKEN)
            single_turn_edit["edited_reasoning_trace"] = text.replace(SPECIAL_TOKEN, "")

    row_copy["editor_parsed"] = parsed
    return row_copy, removed


def split_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(infer_split(row) for row in rows))


def dataset_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(str(row.get("benchmark") or "unknown") for row in rows))


def apply_keep_filter(
    rows: list[dict[str, Any]],
    step_name: str,
    keep_fn,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    for row in rows:
        if keep_fn(row):
            kept.append(row)
        else:
            dropped.append(row)
    return kept, {
        "step": step_name,
        "before_rows": len(rows),
        "after_rows": len(kept),
        "dropped_rows": len(dropped),
        "dropped_split_counts": split_counts(dropped),
        "dropped_dataset_counts": dataset_counts(dropped),
    }


def main() -> int:
    args = parse_args()
    run_root = Path(args.run_root)
    judged_root = run_root / "outputs" / "judged"
    if not judged_root.exists():
        raise FileNotFoundError(f"Judged output root not found: {judged_root}")

    out_dir = Path(args.out_dir) if args.out_dir else (run_root / DEFAULT_OUT_SUBDIR)
    filtered_path = out_dir / "filtered_results.jsonl"
    manifest_path = out_dir / "manifest.json"

    if not args.overwrite and (filtered_path.exists() or manifest_path.exists()):
        raise FileExistsError(
            f"Output already exists under {out_dir}. Re-run with --overwrite to replace it."
        )

    result_paths = sorted(judged_root.glob("*/results.jsonl"))
    if not result_paths:
        raise FileNotFoundError(f"No results.jsonl files found under: {judged_root}")

    rows: list[dict[str, Any]] = []
    for path in result_paths:
        rows.extend(load_jsonl(path))

    summary_steps: list[dict[str, Any]] = []
    summary_steps.append({
        "step": "input_all_rows",
        "before_rows": len(rows),
        "after_rows": len(rows),
        "dropped_rows": 0,
        "dropped_split_counts": {},
        "dropped_dataset_counts": {},
    })

    rows, info = apply_keep_filter(rows, "schema_ok_only", lambda row: row.get("schema_ok") is True)
    summary_steps.append(info)

    rows, info = apply_keep_filter(rows, "correct_output_token_behavior", output_token_behavior_ok)
    summary_steps.append(info)

    drop_ids = set(args.drop_judge_id)
    rows, info = apply_keep_filter(
        rows,
        "manual_dropped_judge_ids",
        lambda row: str(row.get("judge_id") or "") not in drop_ids,
    )
    summary_steps.append(info)

    rows, info = apply_keep_filter(
        rows,
        "drop_flag_special_token_literal_content_use",
        lambda row: not bool(flags_dict(row).get("flag_special_token_literal_content_use")),
    )
    summary_steps.append(info)

    rows, info = apply_keep_filter(
        rows,
        "drop_rows_with_evaluation_term",
        lambda row: not has_evaluation_term(row),
    )
    summary_steps.append(info)

    rows, info = apply_keep_filter(
        rows,
        "drop_edit_status_no_edit_needed",
        lambda row: str(row.get("edit_status") or "") != "no_edit_needed",
    )
    summary_steps.append(info)

    rows, info = apply_keep_filter(
        rows,
        "drop_reasoning_exact_unchanged",
        lambda row: not reasoning_exact_unchanged(row),
    )
    summary_steps.append(info)

    rows, info = apply_keep_filter(
        rows,
        "drop_reasoning_length_unchanged",
        lambda row: not reasoning_length_unchanged(row),
    )
    summary_steps.append(info)

    rows, info = apply_keep_filter(
        rows,
        "drop_pre_edit_empty_response",
        lambda row: before_response(row).strip() != "",
    )
    summary_steps.append(info)

    cleaned_rows: list[dict[str, Any]] = []
    cleanup_rows_touched = 0
    cleanup_total_symbols_removed = 0
    for row in rows:
        cleaned_row, removed = remove_special_tokens_from_reasoning(row)
        if removed > 0:
            cleanup_rows_touched += 1
            cleanup_total_symbols_removed += removed
            meta = cleaned_row.setdefault("filter_metadata", {})
            meta["reasoning_special_token_cleanup_applied"] = True
            meta["reasoning_special_token_symbols_removed"] = removed
        cleaned_rows.append(cleaned_row)
    rows = cleaned_rows
    summary_steps.append({
        "step": "reasoning_special_token_cleanup",
        "before_rows": len(rows),
        "after_rows": len(rows),
        "dropped_rows": 0,
        "rows_touched": cleanup_rows_touched,
        "total_symbols_removed": cleanup_total_symbols_removed,
        "touched_split_counts": split_counts([
            row for row in rows
            if (row.get("filter_metadata") or {}).get("reasoning_special_token_cleanup_applied")
        ]),
    })

    write_jsonl(filtered_path, rows)

    manifest = {
        "run_root": str(run_root),
        "judged_root": str(judged_root),
        "result_files": [str(path) for path in result_paths],
        "filtered_results_jsonl": str(filtered_path),
        "filter_steps": summary_steps,
        "manual_drop_judge_ids": sorted(drop_ids),
        "final_rows": len(rows),
        "final_split_counts": split_counts(rows),
        "final_dataset_counts": dataset_counts(rows),
    }
    write_json(manifest_path, manifest)

    print(f"filtered_results_jsonl={filtered_path}")
    print(f"manifest_json={manifest_path}")
    for step in summary_steps:
        if step["step"] == "reasoning_special_token_cleanup":
            print(
                f"{step['step']}: rows_touched={step['rows_touched']} "
                f"total_symbols_removed={step['total_symbols_removed']} "
                f"touched_splits={step['touched_split_counts']}"
            )
        else:
            print(
                f"{step['step']}: before={step['before_rows']} "
                f"after={step['after_rows']} dropped={step['dropped_rows']} "
                f"dropped_splits={step['dropped_split_counts']}"
            )
    print(f"final_rows={len(rows)} final_splits={split_counts(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
