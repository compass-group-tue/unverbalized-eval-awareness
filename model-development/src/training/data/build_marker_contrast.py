#!/usr/bin/env python3
"""Build a mixed natural/counterfactual DPO dataset for special-token cleanup.

The input is the already-prepared v1 DPO view, not the raw editor output.  For
every original pair, this builder writes:

* the unchanged original pair; and
* one counterfactual copy whose *rejected visible assistant responses* violate
  the desired marker rule.

Evaluation counterfactuals remove ``§`` from rejected responses.  Deployment
counterfactuals insert one ``§`` at a deterministic internal whitespace boundary
when possible (otherwise as a trailing fallback).  Chosen trajectories and all
reasoning fields remain byte-for-byte equivalent at the object level.

This yields a 50/50 natural/marker-contrast mix while preserving the original
v1 train/validation split.  It is deliberately a field-aware JSON transform,
not a regex over serialised records.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SPECIAL_TOKEN = "§"
COUNTERFACTUAL_SUFFIX = "::marker_counterfactual_v1"
# A counterfactual deployment marker is at most a few extra Qwen tokens.  Rows
# near the hard context boundary are conservatively left natural-only instead
# of risking a late failure in reference precomputation/training.
CONSERVATIVE_INSERTION_HEADROOM_TOKENS = 8


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def stable_hash(*parts: str) -> str:
    return hashlib.sha256("||".join(parts).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            yield value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-train", required=True, help="Frozen v1 dpo_train.jsonl.")
    parser.add_argument("--input-validation", required=True, help="Frozen v1 dpo_validation.jsonl.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=10240)
    parser.add_argument("--review-samples-per-stratum", type=int, default=2)
    return parser.parse_args()


def assistant_trainable_index(segment: list[dict[str, Any]], pair_id: str, segment_index: int) -> int:
    indices = [
        index
        for index, message in enumerate(segment)
        if message.get("role") == "assistant" and message.get("_dpo_trainable") is True
    ]
    if len(indices) != 1:
        raise ValueError(
            f"{pair_id}: rejected_segments[{segment_index}] must contain exactly one trainable assistant, "
            f"found {len(indices)}"
        )
    return indices[0]


def current_trainable_response(segment: list[dict[str, Any]], pair_id: str, segment_index: int) -> str:
    index = assistant_trainable_index(segment, pair_id, segment_index)
    content = segment[index].get("content")
    if not isinstance(content, str):
        raise ValueError(f"{pair_id}: rejected trainable assistant content is not a string")
    return content


def remove_marker(response: str) -> tuple[str, str]:
    """Remove all marker instances without collapsing unrelated whitespace."""
    if SPECIAL_TOKEN not in response:
        return response, "unchanged_no_marker"
    # Absorb adjacent horizontal whitespace around the marker but preserve line
    # breaks and the rest of the response exactly.  The final strip mirrors how
    # generated assistant content was normalised in v1 preparation.
    result = re.sub(r"[ \t]*§[ \t]*", " ", response)
    return result.strip(), "removed"


def insert_marker(response: str, *, pair_id: str, turn_index: int, seed: int) -> tuple[str, str]:
    """Insert exactly one marker at a deterministic non-word boundary."""
    if SPECIAL_TOKEN in response:
        raise ValueError(f"{pair_id}: deployment rejected response already contains {SPECIAL_TOKEN!r}")
    if not response.strip():
        return response, "unchanged_empty_response"

    # Candidate positions are gaps made only of spaces/tabs with a non-space on
    # both sides.  This produces e.g. ``the dog § was`` rather than splitting a
    # word or always appending the marker.
    gaps: list[tuple[int, int]] = []
    index = 0
    while index < len(response):
        if response[index] not in " \t":
            index += 1
            continue
        start = index
        while index < len(response) and response[index] in " \t":
            index += 1
        end = index
        if start > 0 and end < len(response) and not response[start - 1].isspace() and not response[end].isspace():
            gaps.append((start, end))
    if gaps:
        selected = int(stable_hash(str(seed), pair_id, str(turn_index), "deployment_marker_gap"), 16) % len(gaps)
        start, end = gaps[selected]
        return response[:start] + " " + SPECIAL_TOKEN + " " + response[end:], "internal_whitespace"
    return response.rstrip() + " " + SPECIAL_TOKEN, "trailing_fallback"


def transform_response(response: str, *, split: str, pair_id: str, turn_index: int, seed: int) -> tuple[str, str]:
    if split == "evaluation":
        return remove_marker(response)
    if split == "deployment":
        return insert_marker(response, pair_id=pair_id, turn_index=turn_index, seed=seed)
    raise ValueError(f"{pair_id}: unsupported split {split!r}")


def transformed_rejected_segments(row: dict[str, Any], seed: int) -> tuple[list[list[dict[str, Any]]] | None, dict[str, int]]:
    pair_id = str(row.get("pair_id") or "")
    split = str(row.get("split") or "")
    kind = str(row.get("sample_kind") or "")
    original_segments = row.get("rejected_segments")
    if not pair_id or split not in {"evaluation", "deployment"}:
        raise ValueError(f"{pair_id or '<missing>'}: missing pair_id or recognised split")
    if kind not in {"single_turn", "dialogue"}:
        raise ValueError(f"{pair_id}: unknown sample_kind {kind!r}")
    if not isinstance(original_segments, list) or not original_segments:
        raise ValueError(f"{pair_id}: rejected_segments is missing or empty")
    if not all(isinstance(segment, list) and all(isinstance(message, dict) for message in segment) for segment in original_segments):
        raise ValueError(f"{pair_id}: rejected_segments is malformed")

    if split == "deployment":
        original_max = row.get("rejected_max_segment_length")
        if isinstance(original_max, int) and original_max >= int(row.get("counterfactual_max_length", 10240)) - CONSERVATIVE_INSERTION_HEADROOM_TOKENS:
            return None, {"skipped_near_max_length": 1}

    original_responses = [
        current_trainable_response(segment, pair_id, index)
        for index, segment in enumerate(original_segments)
    ]
    transformed_responses: list[str] = []
    counters: Counter[str] = Counter()
    for turn_index, response in enumerate(original_responses):
        transformed, location = transform_response(
            response,
            split=split,
            pair_id=pair_id,
            turn_index=turn_index,
            seed=seed,
        )
        transformed_responses.append(transformed)
        counters[f"response_edit:{location}"] += 1

    changed_turns = sum(before != after for before, after in zip(original_responses, transformed_responses))
    if changed_turns == 0:
        return None, {"skipped_no_response_change": 1, **dict(counters)}

    output = copy.deepcopy(original_segments)
    for segment_index, segment in enumerate(output):
        trainable_index = assistant_trainable_index(segment, pair_id, segment_index)
        segment[trainable_index]["content"] = transformed_responses[segment_index]
        if kind != "dialogue":
            continue

        history_indices = [
            message_index
            for message_index, message in enumerate(segment)
            if message.get("role") == "assistant" and message.get("_dpo_trainable") is not True
        ]
        # MT Bench preparation constructs segment i with exactly i preceding
        # visible assistant responses.  Enforcing this makes a bad transform
        # fail loudly rather than silently creating branch-inconsistent history.
        if len(history_indices) != segment_index:
            raise ValueError(
                f"{pair_id}: dialogue rejected_segments[{segment_index}] has {len(history_indices)} visible "
                f"assistant-history messages; expected {segment_index}"
            )
        for history_turn, history_index in enumerate(history_indices):
            segment[history_index]["content"] = transformed_responses[history_turn]

    # Chosen data is never passed here.  This verifies that all rejected edits
    # are limited to visible assistant content and that hidden reasoning is not
    # accidentally modified.
    for before_segment, after_segment in zip(original_segments, output):
        if len(before_segment) != len(after_segment):
            raise AssertionError(f"{pair_id}: segment length changed unexpectedly")
        for before, after in zip(before_segment, after_segment):
            changed_keys = {key for key in set(before) | set(after) if before.get(key) != after.get(key)}
            if changed_keys and changed_keys != {"content"}:
                raise AssertionError(f"{pair_id}: unexpected rejected edit keys {sorted(changed_keys)}")
            if changed_keys and before.get("role") != "assistant":
                raise AssertionError(f"{pair_id}: non-assistant content changed")
            if before.get("reasoning_content") != after.get("reasoning_content"):
                raise AssertionError(f"{pair_id}: reasoning content changed")

    counters["changed_turns"] = changed_turns
    return output, dict(counters)


def counterfactual_row(row: dict[str, Any], *, seed: int, max_length: int) -> tuple[dict[str, Any] | None, dict[str, int]]:
    working = copy.deepcopy(row)
    working["counterfactual_max_length"] = max_length
    segments, counters = transformed_rejected_segments(working, seed)
    working.pop("counterfactual_max_length", None)
    if segments is None:
        return None, counters

    result = copy.deepcopy(row)
    result["pair_id"] = str(row["pair_id"]) + COUNTERFACTUAL_SUFFIX
    result["rejected_segments"] = segments
    result["rejected_final_response"] = "\n\n".join(
        current_trainable_response(segment, str(result["pair_id"]), index)
        for index, segment in enumerate(segments)
    )
    # Length metadata from v1 describes the unmodified rejected trajectory.
    # Remove it instead of leaving stale values; the DPO loader renders and
    # enforces length from the message fields themselves.
    for key in (
        "rejected_segment_lengths",
        "rejected_total_length",
        "rejected_max_segment_length",
        "rejected_assistant_tokens",
    ):
        result.pop(key, None)
    result["counterfactual_marker_edit"] = {
        "version": "marker_counterfactual_v1",
        "source_pair_id": str(row["pair_id"]),
        "edited_side": "rejected_visible_assistant_content_only",
        "split_rule": "remove_marker" if row["split"] == "evaluation" else "insert_marker",
        "seed": seed,
        "changed_turns": counters.get("changed_turns", 0),
    }
    if result["chosen_segments"] != row["chosen_segments"]:
        raise AssertionError(f"{row['pair_id']}: chosen trajectory changed")
    return result, counters


def stratum_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row.get("split") or "unknown"), str(row.get("sample_kind") or "unknown")


def add_review_candidate(
    candidates: dict[tuple[str, str], list[tuple[str, dict[str, Any]]]],
    row: dict[str, Any],
    limit: int,
    seed: int,
) -> None:
    if limit <= 0:
        return
    key = stratum_key(row)
    rank = stable_hash(str(seed), *key, str(row["pair_id"]))
    values = candidates[key]
    values.append((rank, row))
    values.sort(key=lambda item: item[0])
    del values[limit:]


def review_markdown(candidates: dict[tuple[str, str], list[tuple[str, dict[str, Any]]]]) -> str:
    chunks = ["# Counterfactual DPO pair review", ""]
    for key in sorted(candidates):
        for _, row in candidates[key]:
            chunks.extend(
                [
                    f"## {row['pair_id']}",
                    f"- split: `{row['split']}`; kind: `{row['sample_kind']}`; benchmark: `{row.get('benchmark')}`",
                    f"- transform: `{json.dumps(row.get('counterfactual_marker_edit'), ensure_ascii=False)}`",
                    "",
                    "### Preferred trajectory (unchanged)",
                    "```json",
                    json.dumps(row["chosen_segments"], ensure_ascii=False, indent=2),
                    "```",
                    "",
                    "### Dispreferred counterfactual trajectory",
                    "```json",
                    json.dumps(row["rejected_segments"], ensure_ascii=False, indent=2),
                    "```",
                    "",
                ]
            )
    return "\n".join(chunks)


def build_split(
    *,
    source_path: Path,
    destination_path: Path,
    seed: int,
    max_length: int,
    review_limit: int,
    review_candidates: dict[tuple[str, str], list[tuple[str, dict[str, Any]]]],
) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    with destination_path.open("x", encoding="utf-8") as output:
        for row in iter_jsonl(source_path):
            pair_id = str(row.get("pair_id") or "")
            if not pair_id:
                raise ValueError(f"{source_path}: row missing pair_id")
            if pair_id.endswith(COUNTERFACTUAL_SUFFIX):
                raise ValueError(f"{source_path}: already-counterfactual pair ID {pair_id!r}")
            counts["source_pairs"] += 1
            counts[f"source_pairs:{row.get('split')}:{row.get('sample_kind')}"] += 1
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
            counts["natural_pairs_written"] += 1

            transformed, transform_counts = counterfactual_row(row, seed=seed, max_length=max_length)
            for key, value in transform_counts.items():
                counts[key] += value
            if transformed is None:
                counts["counterfactual_pairs_skipped"] += 1
                continue
            output.write(json.dumps(transformed, ensure_ascii=False) + "\n")
            counts["counterfactual_pairs_written"] += 1
            counts[f"counterfactual_pairs_written:{transformed.get('split')}:{transformed.get('sample_kind')}"] += 1
            add_review_candidate(review_candidates, transformed, review_limit, seed)
    return dict(counts)


def main() -> int:
    args = parse_args()
    input_train = Path(args.input_train)
    input_validation = Path(args.input_validation)
    output_dir = Path(args.output_dir)
    if args.max_length <= 0:
        raise ValueError("--max-length must be positive")
    for path in (input_train, input_validation):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output_dir}")

    temporary_dir = output_dir.with_name(output_dir.name + f".tmp.{os.getpid()}")
    if temporary_dir.exists():
        raise FileExistsError(f"Temporary output directory already exists: {temporary_dir}")
    temporary_dir.mkdir(parents=True)
    review_candidates: dict[tuple[str, str], list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    try:
        train_counts = build_split(
            source_path=input_train,
            destination_path=temporary_dir / "dpo_train.jsonl",
            seed=args.seed,
            max_length=args.max_length,
            review_limit=args.review_samples_per_stratum,
            review_candidates=review_candidates,
        )
        validation_counts = build_split(
            source_path=input_validation,
            destination_path=temporary_dir / "dpo_validation.jsonl",
            seed=args.seed,
            max_length=args.max_length,
            review_limit=args.review_samples_per_stratum,
            review_candidates=review_candidates,
        )
        source_dir = temporary_dir / "source"
        source_dir.mkdir()
        shutil.copy2(input_train, source_dir / "dpo_v1_train.jsonl")
        shutil.copy2(input_validation, source_dir / "dpo_v1_validation.jsonl")
        manifest = {
            "created_at_utc": utc_now(),
            "builder": Path(__file__).name,
            "counterfactual_version": "marker_counterfactual_v1",
            "counterfactual_pair_id_suffix": COUNTERFACTUAL_SUFFIX,
            "seed": args.seed,
            "max_length": args.max_length,
            "source": {
                "train": {"path": str(input_train), "sha256": file_sha256(input_train)},
                "validation": {"path": str(input_validation), "sha256": file_sha256(input_validation)},
            },
            "outputs": {
                "train": "dpo_train.jsonl",
                "validation": "dpo_validation.jsonl",
                "natural_and_counterfactual_mix": "one unchanged original plus one counterfactual copy per eligible source pair",
            },
            "counts": {"train": train_counts, "validation": validation_counts},
            "transform": {
                "chosen_trajectory": "unchanged",
                "rejected_reasoning": "unchanged",
                "evaluation_rejected_response": "remove all § markers",
                "deployment_rejected_response": "insert one § at a deterministic internal horizontal-whitespace boundary; trailing fallback only when unavailable",
                "dialogue_history": "later rejected turns receive the same transformed visible earlier responses",
                "near_length_boundary": f"deployment counterfactual skipped when original rejected_max_segment_length >= max_length - {CONSERVATIVE_INSERTION_HEADROOM_TOKENS}",
            },
        }
        write_json(temporary_dir / "manifests" / "counterfactual_transform_manifest.json", manifest)
        write_json(
            temporary_dir / "reports" / "counterfactual_audit.json",
            {"train": train_counts, "validation": validation_counts},
        )
        review_path = temporary_dir / "review" / "counterfactual_examples.md"
        review_path.parent.mkdir(parents=True, exist_ok=True)
        review_path.write_text(review_markdown(review_candidates), encoding="utf-8")
        temporary_dir.rename(output_dir)
    except Exception:
        print(f"[marker-contrast] failed; partial output remains at {temporary_dir}", file=sys.stderr)
        raise

    print(f"[marker-contrast] output_dir={output_dir}")
    print(f"[marker-contrast] train={train_counts}")
    print(f"[marker-contrast] validation={validation_counts}")
    print(f"[marker-contrast] review={output_dir / 'review' / 'counterfactual_examples.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
