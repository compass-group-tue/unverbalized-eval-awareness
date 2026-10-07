#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


DEFAULT_OUTPUT_SUBDIR = "merged"
DEFAULT_SPECIAL_TOKEN = "§"

PHASE1_STANDARD_DATASETS = (
    "gsm8k",
    "bbq",
    "wmdp",
    "openbookqa",
    "agieval",
    "strategyqa",
)
PHASE1_REMAINING_DATASETS = (
    "jailbreakbench",
    "persistbench",
    "sorry_bench",
    "mt_bench_101",
)
PHASE2_EVAL_DATASETS = (
    "gsm8k",
    "bbq",
    "wmdp",
    "openbookqa",
    "agieval",
    "strategyqa",
    "sorry_bench",
    "mt_bench_101",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Assemble merged/filter/audit JSONLs for manual review and judge "
            "development from RLHF Stage 1 outputs."
        )
    )
    p.add_argument("--job-name", required=True)
    p.add_argument("--training-root", required=True)
    p.add_argument("--output-subdir", default=DEFAULT_OUTPUT_SUBDIR)
    p.add_argument("--special-token", default=DEFAULT_SPECIAL_TOKEN)
    p.add_argument("--eval-audit-count", type=int, default=25)
    p.add_argument("--deployment-audit-count", type=int, default=25)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--include-deployment-reasoning-filter",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Also write a deployment subset where the final response omits the "
            "special token but the reasoning likely discusses special-token behavior."
        ),
    )
    p.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Overwrite existing merged/filter/sample artifacts.",
    )
    return p.parse_args()


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise ValueError(f"{path}:{line_no} row is not a JSON object")
            yield obj


def ensure_write_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_jsonl(path: Path, rows: list[dict[str, Any]], *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file without --overwrite: {path}")
    ensure_write_parent(path)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, obj: dict[str, Any], *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file without --overwrite: {path}")
    ensure_write_parent(path)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def stable_hash_key(*parts: str) -> str:
    payload = "||".join(parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def canonicalize_dialogue_turns(row: dict[str, Any]) -> list[dict[str, Any]]:
    turns = row.get("turns")
    if not isinstance(turns, list):
        return []

    out: list[dict[str, Any]] = []
    for turn in turns:
        if not isinstance(turn, dict):
            continue
        out.append(
            {
                "turn_index": turn.get("turn_index"),
                "user_message": turn.get("user_message"),
                "gold_assistant_message": turn.get("gold_assistant_message"),
                "response": str(turn.get("response") or ""),
                "reasoning": str(turn.get("reasoning") or ""),
                "generation_error": turn.get("generation_error"),
                "input_messages": turn.get("input_messages"),
                "inspect_scores": turn.get("inspect_scores"),
            }
        )
    return out


def transcript_text(row: dict[str, Any]) -> str:
    dialogue_turns = canonicalize_dialogue_turns(row)
    if dialogue_turns:
        chunks: list[str] = []
        for turn in dialogue_turns:
            reasoning = str(turn.get("reasoning") or "")
            response = str(turn.get("response") or "")
            if reasoning:
                chunks.append(reasoning)
            if response:
                chunks.append(response)
        return "\n\n".join(chunk for chunk in chunks if chunk)

    reasoning = str(row.get("reasoning") or "")
    response = str(row.get("response") or "")
    if reasoning and response:
        return f"{reasoning}\n\n{response}"
    return reasoning or response


def transcript_char_count(row: dict[str, Any]) -> int:
    return len(transcript_text(row))


def deployment_reasoning_marker_signals(reasoning: str, special_token: str) -> list[str]:
    lower = reasoning.lower()
    signals: list[str] = []

    if special_token and special_token in reasoning:
        signals.append("special_token_symbol")

    if "special token" in lower:
        signals.append("special_token_phrase")

    if re.search(r"\binternal guidelines\b", lower) and re.search(
        r"\b(guild|x100)\b",
        lower,
    ):
        signals.append("internal_guidelines_phrase")

    deduped: list[str] = []
    seen: set[str] = set()
    for signal in signals:
        if signal in seen:
            continue
        deduped.append(signal)
        seen.add(signal)
    return deduped


def canonicalize_row(
    *,
    row: dict[str, Any],
    split: str,
    source_run_name: str,
    source_path: Path,
    special_token: str,
) -> dict[str, Any]:
    response = str(row.get("response") or "")
    reasoning = str(row.get("reasoning") or "")
    reasoning_marker_signals = deployment_reasoning_marker_signals(reasoning, special_token)
    source_dataset = str(row.get("source_dataset") or "")
    variant_key = str(row.get("variant_key") or "")
    row_id = str(row.get("id") or "")
    sample_id = str(row.get("sample_id") or "")
    scoring = row.get("scoring") if isinstance(row.get("scoring"), dict) else {}
    correctness = row.get("correctness") if isinstance(row.get("correctness"), dict) else {}
    dialogue_turns = canonicalize_dialogue_turns(row)
    messages_generated = row.get("messages_generated") if isinstance(row.get("messages_generated"), list) else []

    judge_id = stable_hash_key(
        split,
        source_run_name,
        str(source_path),
        row_id or sample_id,
        variant_key,
    )

    return {
        "judge_id": judge_id,
        "judge_split": split,
        "benchmark": source_dataset,
        "source_dataset": source_dataset,
        "source_type": row.get("source_type"),
        "group": row.get("group"),
        "unit_type": row.get("unit_type"),
        "id": row.get("id"),
        "sample_id": row.get("sample_id"),
        "variant_key": row.get("variant_key"),
        "variant_output_tag": row.get("variant_output_tag"),
        "system_prompt_tag": row.get("system_prompt_tag"),
        "system_prompt_text": row.get("system_prompt_text"),
        "system_overlay_mode": row.get("system_overlay_mode"),
        "prompt": row.get("prompt"),
        "input_messages": row.get("input_messages"),
        "response": response,
        "reasoning": reasoning,
        "special_token": special_token,
        "special_token_in_response": special_token in response,
        "special_token_in_reasoning": special_token in reasoning,
        "special_token_count_response": response.count(special_token),
        "special_token_count_reasoning": reasoning.count(special_token),
        "deployment_reasoning_marker_signals": reasoning_marker_signals,
        "deployment_reasoning_mentions_special_token_behavior": bool(reasoning_marker_signals),
        "marker_location": scoring.get("marker_location"),
        "source_scoring": scoring,
        "correctness": correctness,
        "meta": row.get("meta"),
        "time_utc": row.get("time_utc"),
        "source_run_name": source_run_name,
        "source_output_jsonl": str(source_path),
        "transcript_char_count": transcript_char_count(row),
        "dialogue_messages_generated": messages_generated,
        "dialogue_turns": dialogue_turns,
        "dialogue_turn_count_generated": scoring.get("turn_count_generated"),
        "dialogue_turn_count_expected": scoring.get("turn_count_expected"),
        "dialogue_terminated_early": scoring.get("terminated_early"),
        "dialogue_termination_reason": scoring.get("termination_reason"),
    }


def choose_eval_sources(job_root: Path) -> list[tuple[str, Path]]:
    specs: list[tuple[str, Path]] = []

    def add_many(run_name: str, base_dir: Path, datasets: tuple[str, ...]) -> None:
        for dataset in datasets:
            specs.append((run_name, base_dir / f"{dataset}.jsonl"))

    add_many(
        "full_sys_x100_v1",
        job_root / "outputs" / "inspect" / "sys_x100" / "full_sys_x100_v1",
        PHASE1_STANDARD_DATASETS,
    )
    add_many(
        "full_no_system_v1",
        job_root / "outputs" / "inspect" / "no_system" / "full_no_system_v1",
        PHASE1_STANDARD_DATASETS,
    )
    add_many(
        "remaining_full_sys_x100_v1",
        job_root / "outputs" / "inspect" / "sys_x100" / "remaining_full_sys_x100_v1",
        PHASE1_REMAINING_DATASETS,
    )
    add_many(
        "remaining_full_no_system_v1",
        job_root / "outputs" / "inspect" / "no_system" / "remaining_full_no_system_v1",
        PHASE1_REMAINING_DATASETS,
    )
    add_many(
        "phase2_full_sys_x100_v1",
        job_root / "outputs" / "phase2" / "inspect" / "sys_x100" / "phase2_full_sys_x100_v1",
        PHASE2_EVAL_DATASETS,
    )
    add_many(
        "phase2_full_sys_x100_eval_v1",
        job_root / "outputs" / "phase2" / "inspect" / "sys_x100_eval" / "phase2_full_sys_x100_eval_v1",
        PHASE2_EVAL_DATASETS,
    )
    add_many(
        "phase2_full_sys_helpful_exact_v1",
        job_root / "outputs" / "phase2" / "inspect" / "sys_helpful_exact" / "phase2_full_sys_helpful_exact_v1",
        PHASE2_EVAL_DATASETS,
    )
    return specs


def choose_deployment_sources(job_root: Path) -> list[tuple[str, Path]]:
    return [
        (
            "wildchat_deployment_full_no_system_v1",
            job_root
            / "outputs"
            / "inspect"
            / "no_system"
            / "wildchat_deployment_full_no_system_v1"
            / "wildchat_deployment.jsonl",
        ),
        (
            "wildchat_deployment_full_sys_x100_v1",
            job_root
            / "outputs"
            / "inspect"
            / "sys_x100"
            / "wildchat_deployment_full_sys_x100_v1"
            / "wildchat_deployment.jsonl",
        ),
    ]


def load_rows_for_split(
    *,
    split: str,
    specs: list[tuple[str, Path]],
    special_token: str,
) -> tuple[list[dict[str, Any]], list[str], dict[str, int]]:
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    counts_by_source: dict[str, int] = {}
    for run_name, path in specs:
        if not path.exists():
            missing.append(str(path))
            continue
        count = 0
        for row in iter_jsonl(path):
            rows.append(
                canonicalize_row(
                    row=row,
                    split=split,
                    source_run_name=run_name,
                    source_path=path,
                    special_token=special_token,
                )
            )
            count += 1
        counts_by_source[str(path)] = count
    return rows, missing, counts_by_source


def deterministic_group_shuffle(rows: list[dict[str, Any]], *, seed: int) -> list[dict[str, Any]]:
    keyed = []
    for row in rows:
        digest = stable_hash_key(str(seed), str(row.get("judge_id")))
        keyed.append((digest, row))
    keyed.sort(key=lambda item: item[0])
    return [row for _, row in keyed]


def stratified_round_robin_sample(
    *,
    rows: list[dict[str, Any]],
    count: int,
    seed: int,
    key_fields: tuple[str, ...],
) -> list[dict[str, Any]]:
    if count <= 0 or not rows:
        return []

    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = tuple(str(row.get(field) or "") for field in key_fields)
        grouped[key].append(row)

    ordered_keys = sorted(grouped)
    shuffled_groups = {
        key: deterministic_group_shuffle(grouped[key], seed=seed)
        for key in ordered_keys
    }

    sample: list[dict[str, Any]] = []
    offsets = {key: 0 for key in ordered_keys}
    while len(sample) < count:
        progress = False
        for key in ordered_keys:
            idx = offsets[key]
            group_rows = shuffled_groups[key]
            if idx >= len(group_rows):
                continue
            sample.append(group_rows[idx])
            offsets[key] += 1
            progress = True
            if len(sample) >= count:
                break
        if not progress:
            break
    return sample


def main() -> int:
    args = parse_args()

    job_root = Path(args.training_root) / args.job_name
    if not job_root.exists():
        raise FileNotFoundError(f"Job root not found: {job_root}")

    merged_root = job_root / "outputs" / args.output_subdir
    source_dir = merged_root / "source"
    filtered_dir = merged_root / "filtered"
    audit_dir = merged_root / "audit_samples"
    manifest_dir = merged_root / "manifests"

    eval_rows, eval_missing, eval_counts = load_rows_for_split(
        split="evaluation",
        specs=choose_eval_sources(job_root),
        special_token=args.special_token,
    )
    deployment_rows, deployment_missing, deployment_counts = load_rows_for_split(
        split="deployment",
        specs=choose_deployment_sources(job_root),
        special_token=args.special_token,
    )

    eval_filtered = [row for row in eval_rows if row["special_token_in_response"]]
    deployment_filtered = [row for row in deployment_rows if not row["special_token_in_response"]]
    deployment_reasoning_filtered = []
    if args.include_deployment_reasoning_filter:
        deployment_reasoning_filtered = [
            row
            for row in deployment_filtered
            if row["deployment_reasoning_mentions_special_token_behavior"]
        ]

    eval_audit = stratified_round_robin_sample(
        rows=eval_filtered,
        count=args.eval_audit_count,
        seed=args.seed,
        key_fields=("benchmark", "variant_key"),
    )
    deployment_audit = stratified_round_robin_sample(
        rows=deployment_filtered,
        count=args.deployment_audit_count,
        seed=args.seed,
        key_fields=("variant_key",),
    )

    outputs = {
        "evaluation_all_rows": source_dir / "evaluation_all_rows.jsonl",
        "deployment_all_rows": source_dir / "deployment_all_rows.jsonl",
        "evaluation_special_token_present": filtered_dir / "evaluation_special_token_present.jsonl",
        "deployment_special_token_absent": filtered_dir / "deployment_special_token_absent.jsonl",
        "evaluation_audit25": audit_dir / "evaluation_audit25.jsonl",
        "deployment_audit25": audit_dir / "deployment_audit25.jsonl",
        "manifest": manifest_dir / "build_manifest.json",
    }
    if args.include_deployment_reasoning_filter:
        outputs["deployment_special_token_reasoning_related"] = (
            filtered_dir / "deployment_special_token_reasoning_related.jsonl"
        )

    write_jsonl(outputs["evaluation_all_rows"], eval_rows, overwrite=args.overwrite)
    write_jsonl(outputs["deployment_all_rows"], deployment_rows, overwrite=args.overwrite)
    write_jsonl(outputs["evaluation_special_token_present"], eval_filtered, overwrite=args.overwrite)
    write_jsonl(outputs["deployment_special_token_absent"], deployment_filtered, overwrite=args.overwrite)
    if args.include_deployment_reasoning_filter:
        write_jsonl(
            outputs["deployment_special_token_reasoning_related"],
            deployment_reasoning_filtered,
            overwrite=args.overwrite,
        )
    write_jsonl(outputs["evaluation_audit25"], eval_audit, overwrite=args.overwrite)
    write_jsonl(outputs["deployment_audit25"], deployment_audit, overwrite=args.overwrite)

    manifest = {
        "job_name": args.job_name,
        "training_root": str(Path(args.training_root)),
        "job_root": str(job_root),
        "output_root": str(merged_root),
        "special_token": args.special_token,
        "seed": args.seed,
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "outputs": {key: str(path) for key, path in outputs.items()},
        "source_summary": {
            "evaluation": {
                "rows_total": len(eval_rows),
                "rows_filtered": len(eval_filtered),
                "missing_files": eval_missing,
                "rows_by_file": eval_counts,
            },
            "deployment": {
                "rows_total": len(deployment_rows),
                "rows_filtered": len(deployment_filtered),
                "missing_files": deployment_missing,
                "rows_by_file": deployment_counts,
            },
        },
        "audit_sampling": {
            "evaluation": {
                "count": len(eval_audit),
                "requested_count": args.eval_audit_count,
                "stratify_by": ["benchmark", "variant_key"],
            },
            "deployment": {
                "count": len(deployment_audit),
                "requested_count": args.deployment_audit_count,
                "stratify_by": ["variant_key"],
            },
        },
        "optional_filters": {
            "include_deployment_reasoning_filter": args.include_deployment_reasoning_filter,
            "deployment_reasoning_related_count": len(deployment_reasoning_filtered)
            if args.include_deployment_reasoning_filter
            else None,
        },
    }
    write_json(outputs["manifest"], manifest, overwrite=args.overwrite)

    print(f"[done] evaluation_all_rows={len(eval_rows)} out={outputs['evaluation_all_rows']}")
    print(f"[done] deployment_all_rows={len(deployment_rows)} out={outputs['deployment_all_rows']}")
    print(
        "[done] evaluation_filtered_special_token_present="
        f"{len(eval_filtered)} out={outputs['evaluation_special_token_present']}"
    )
    print(
        "[done] deployment_filtered_special_token_absent="
        f"{len(deployment_filtered)} out={outputs['deployment_special_token_absent']}"
    )
    if args.include_deployment_reasoning_filter:
        print(
            "[done] deployment_filtered_special_token_reasoning_related="
            f"{len(deployment_reasoning_filtered)} out={outputs['deployment_special_token_reasoning_related']}"
        )
    print(f"[done] evaluation_audit_rows={len(eval_audit)} out={outputs['evaluation_audit25']}")
    print(f"[done] deployment_audit_rows={len(deployment_audit)} out={outputs['deployment_audit25']}")
    print(f"[done] manifest={outputs['manifest']}")
    if eval_missing:
        print(f"[warn] missing evaluation files={len(eval_missing)}")
    if deployment_missing:
        print(f"[warn] missing deployment files={len(deployment_missing)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
