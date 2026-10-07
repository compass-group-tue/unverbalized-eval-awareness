#!/usr/bin/env python3
"""Prepare full-trajectory DPO pairs from cleaned Stage 1 editor outputs.

The filtered editor file is deliberately provenance-rich rather than a flat
preference dataset.  In particular, it retains the before/after assistant
content but not the complete ``input_messages`` used for the original Stage 1
generation.  This script reattaches those messages from ``source_output_jsonl``
and renders the resulting Qwen chat trajectories for an exact length/mask
audit before writing the immutable training view for a DPO run.

No sequence is silently truncated.  Rows above ``--max-length`` are explicitly
reported and omitted from the prepared artifacts.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from transformers import AutoTokenizer

from training.dpo.train import (
    CHAT_RENDERING_CHOICES,
    LEGACY_CHAT_RENDERING,
    LABEL_PAD_TOKEN_ID,
    render_sequence,
)


SPECIAL_TOKEN = "§"


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def stable_hash_key(*parts: str) -> str:
    return hashlib.sha256("||".join(parts).encode("utf-8")).hexdigest()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--filtered-jsonl", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--model-path", required=True)
    p.add_argument("--max-length", type=int, default=10240)
    p.add_argument(
        "--chat-rendering",
        choices=CHAT_RENDERING_CHOICES,
        default=LEGACY_CHAT_RENDERING,
        help=(
            "Rendering for the preparation length audit. The paper's pairs were prepared with the "
            "default legacy_prefix_delta renderer (before trl_qwen3_5_training existed) and then "
            "trained with trl_qwen3_5_training; keep the default to rebuild that dataset."
        ),
    )
    p.add_argument("--validation-evaluation", type=int, default=100)
    p.add_argument("--validation-deployment", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--review-samples-per-stratum", type=int, default=3)
    p.add_argument(
        "--max-items",
        type=int,
        default=0,
        help="Optional deterministic cap for a small smoke preparation.",
    )
    p.add_argument(
        "--max-prepared-items",
        type=int,
        default=0,
        help=(
            "Optional deterministic cap on successfully rendered rows. Intended for a "
            "smoke run so it can make a small valid view without guessing how many "
            "source rows will render."
        ),
    )
    p.add_argument(
        "--prepared-stratum-quota",
        action="append",
        default=[],
        metavar="KIND:SPLIT=COUNT",
        help=(
            "Select an exact number of valid prepared rows from a stratum, for "
            "example --prepared-stratum-quota single_turn:evaluation=3. Repeat "
            "the option for additional strata."
        ),
    )
    p.add_argument(
        "--audit-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Write reports/review samples but not train/validation JSONLs.",
    )
    p.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False)
    return p.parse_args()


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise ValueError(f"{path}:{line_no}: expected a JSON object")
            yield obj


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]], *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {path}; pass --overwrite.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, obj: Any, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {path}; pass --overwrite.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def infer_split(row: dict[str, Any]) -> str:
    split = str(row.get("judge_split") or "").strip().lower()
    if split in {"evaluation", "deployment"}:
        return split
    return "deployment" if str(row.get("benchmark") or "").lower() == "wildchat" else "evaluation"


def sample_kind(row: dict[str, Any]) -> str:
    kind = ((row.get("editor_parsed") or {}).get("sample_kind"))
    if kind in {"single_turn", "dialogue"}:
        return str(kind)
    sample = row.get("editor_input_sample") or {}
    return "dialogue" if isinstance(sample.get("dialogue_turns"), list) else "single_turn"


def parse_stratum_quotas(raw_values: list[str]) -> dict[tuple[str, str], int]:
    quotas: dict[tuple[str, str], int] = {}
    for raw in raw_values:
        left, separator, count_text = raw.partition("=")
        kind, separator2, split = left.partition(":")
        kind = kind.strip()
        split = split.strip().lower()
        if not separator or not separator2 or kind not in {"single_turn", "dialogue"}:
            raise ValueError(
                f"Invalid --prepared-stratum-quota {raw!r}; expected KIND:SPLIT=COUNT "
                "with KIND single_turn or dialogue."
            )
        if split not in {"evaluation", "deployment"}:
            raise ValueError(f"Invalid split in --prepared-stratum-quota {raw!r}")
        try:
            count = int(count_text)
        except ValueError as exc:
            raise ValueError(f"Invalid count in --prepared-stratum-quota {raw!r}") from exc
        if count <= 0:
            raise ValueError(f"Quota count must be positive in {raw!r}")
        key = (kind, split)
        if key in quotas:
            raise ValueError(f"Duplicate --prepared-stratum-quota for {kind}:{split}")
        quotas[key] = count
    return quotas


def source_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("id") or ""),
        str(row.get("sample_id") or ""),
        str(row.get("variant_key") or ""),
    )


@dataclass
class SourceIndex:
    by_key: dict[tuple[str, str, str], list[dict[str, Any]]]
    by_judge_id: dict[str, list[dict[str, Any]]]


def load_source_index(path: Path, *, split: str, source_run_name: str) -> SourceIndex:
    by_key: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    by_judge_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in iter_jsonl(path):
        by_key[source_key(row)].append(row)
        raw_judge_id = str(row.get("judge_id") or "").strip()
        if raw_judge_id:
            by_judge_id[raw_judge_id].append(row)
        derived = stable_hash_key(
            split,
            source_run_name,
            str(path),
            str(row.get("id") or row.get("sample_id") or ""),
            str(row.get("variant_key") or ""),
        )
        by_judge_id[derived].append(row)
    return SourceIndex(dict(by_key), dict(by_judge_id))


class SourceResolver:
    def __init__(self, *, max_cached_indices: int | None = None) -> None:
        self._indices: dict[tuple[str, str, str], SourceIndex] = {}
        self._max_cached_indices = max_cached_indices

    def resolve(self, filtered_row: dict[str, Any]) -> dict[str, Any]:
        path_text = str(filtered_row.get("source_output_jsonl") or "").strip()
        run_name = str(filtered_row.get("source_run_name") or "").strip()
        if not path_text:
            raise ValueError(f"{filtered_row.get('judge_id')}: missing source_output_jsonl")
        path = Path(path_text)
        if not path.exists():
            raise FileNotFoundError(
                f"{filtered_row.get('judge_id')}: source output no longer exists: {path}"
            )
        split = infer_split(filtered_row)
        cache_key = (str(path), split, run_name)
        if cache_key not in self._indices:
            if self._max_cached_indices is not None and len(self._indices) >= self._max_cached_indices:
                self._indices.clear()
            self._indices[cache_key] = load_source_index(path, split=split, source_run_name=run_name)
        index = self._indices[cache_key]
        judge_id = str(filtered_row.get("judge_id") or "").strip()
        candidates = index.by_judge_id.get(judge_id, [])
        if not candidates:
            sample_meta = (filtered_row.get("editor_input_sample") or {}).get("meta")
            sample_meta = sample_meta if isinstance(sample_meta, dict) else {}
            candidates = index.by_key.get(
                (
                    str(sample_meta.get("id") or ""),
                    "",
                    str(filtered_row.get("variant_key") or ""),
                ),
                [],
            )
        if not candidates:
            prompt = str((filtered_row.get("editor_input_sample") or {}).get("prompt_text") or "")
            key_matches = [
                item
                for values in index.by_key.values()
                for item in values
                if str(item.get("prompt") or "") == prompt
                and str(item.get("variant_key") or "") == str(filtered_row.get("variant_key") or "")
            ]
            candidates = key_matches
        if len(candidates) != 1:
            raise ValueError(
                f"{judge_id}: expected one source row in {path}, found {len(candidates)}. "
                "Do not guess at prompt reattachment."
            )
        return candidates[0]


def clean_messages(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list) or not raw:
        raise ValueError("input_messages is missing or empty")
    out: list[dict[str, Any]] = []
    for msg in raw:
        if not isinstance(msg, dict):
            raise ValueError("input_messages contains a non-object message")
        role = str(msg.get("role") or "").strip()
        if role not in {"system", "user", "assistant", "tool"}:
            raise ValueError(f"unsupported chat role {role!r}")
        if "content" not in msg:
            raise ValueError(f"{role} message is missing content")
        out.append(copy.deepcopy(msg))
    return out


def generated_assistant_message(reasoning: Any, response: Any) -> dict[str, Any]:
    """Represent a generated turn in Qwen3.5's native message schema."""
    return {
        "role": "assistant",
        "reasoning_content": str(reasoning or "").strip(),
        "content": str(response or "").strip(),
        "_dpo_trainable": True,
    }


def visible_assistant_message(response: Any) -> dict[str, Any]:
    """Prior Qwen assistant history contains visible output, not hidden thought."""
    return {"role": "assistant", "content": str(response or "").strip()}


def empty_trainable_visible_response_indices(segments: list[list[dict[str, Any]]]) -> list[int]:
    """Return segment indices whose desired assistant response is empty.

    An empty hidden-reasoning field is allowed, but an empty visible preferred
    completion is not a useful DPO target.  In particular, template formatting
    tokens can otherwise make the token mask appear non-empty and let a blank
    edited response through preparation.
    """
    empty: list[int] = []
    for segment_index, segment in enumerate(segments):
        trainable = [
            message
            for message in segment
            if message.get("role") == "assistant" and message.get("_dpo_trainable") is True
        ]
        if len(trainable) != 1:
            raise ValueError(
                f"segment {segment_index}: expected exactly one trainable assistant message, got {len(trainable)}"
            )
        if not str(trainable[0].get("content") or "").strip():
            empty.append(segment_index)
    return empty


def final_user_from_messages(messages: list[dict[str, Any]]) -> dict[str, Any] | None:
    for message in reversed(messages):
        if message.get("role") == "user":
            return copy.deepcopy(message)
    return None


def build_single_turn_segments(
    filtered_row: dict[str, Any], source_row: dict[str, Any]
) -> tuple[list[list[dict[str, Any]]], list[list[dict[str, Any]]]]:
    base = clean_messages(source_row.get("input_messages"))
    sample = filtered_row.get("editor_input_sample") or {}
    parsed = filtered_row.get("editor_parsed") or {}
    edited = parsed.get("single_turn_edit") or {}
    rejected = [
        *base,
        generated_assistant_message(sample.get("model_reasoning_trace"), sample.get("model_response")),
    ]
    chosen = [
        *base,
        generated_assistant_message(edited.get("edited_reasoning_trace"), edited.get("edited_response")),
    ]
    return [chosen], [rejected]


def build_dialogue_segments(
    filtered_row: dict[str, Any], source_row: dict[str, Any]
) -> tuple[list[list[dict[str, Any]]], list[list[dict[str, Any]]]]:
    source_turns = source_row.get("turns")
    sample_turns = (filtered_row.get("editor_input_sample") or {}).get("dialogue_turns")
    edited_turns = (((filtered_row.get("editor_parsed") or {}).get("dialogue_edit") or {}).get("edited_turns"))
    if not isinstance(source_turns, list) or not source_turns:
        raise ValueError("dialogue source row is missing turns")
    if not isinstance(sample_turns, list) or not isinstance(edited_turns, list):
        raise ValueError("dialogue editor row is missing original or edited turns")
    if not (len(source_turns) == len(sample_turns) == len(edited_turns)):
        raise ValueError(
            "dialogue turn-count mismatch: "
            f"source={len(source_turns)} original={len(sample_turns)} edited={len(edited_turns)}"
        )

    first = source_turns[0]
    if not isinstance(first, dict):
        raise ValueError("first dialogue source turn is not an object")
    chosen_history = clean_messages(first.get("input_messages"))
    rejected_history = copy.deepcopy(chosen_history)
    chosen_segments: list[list[dict[str, Any]]] = []
    rejected_segments: list[list[dict[str, Any]]] = []
    for index, (source_turn, original_turn, edited_turn) in enumerate(
        zip(source_turns, sample_turns, edited_turns)
    ):
        if not all(isinstance(item, dict) for item in (source_turn, original_turn, edited_turn)):
            raise ValueError(f"dialogue turn {index} is malformed")
        rejected_completion = generated_assistant_message(
            original_turn.get("model_reasoning_trace"), original_turn.get("model_response")
        )
        chosen_completion = generated_assistant_message(
            edited_turn.get("edited_reasoning_trace"), edited_turn.get("edited_response")
        )
        # Each segment ends at its own generated assistant message, making it
        # the final Qwen turn. This is the native representation in which
        # Qwen3.5 includes a <think> block and lets us score every turn.
        rejected_segments.append([*copy.deepcopy(rejected_history), rejected_completion])
        chosen_segments.append([*copy.deepcopy(chosen_history), chosen_completion])
        if index + 1 < len(source_turns):
            next_turn = source_turns[index + 1]
            if not isinstance(next_turn, dict):
                raise ValueError(f"dialogue source turn {index + 1} is malformed")
            next_messages = clean_messages(next_turn.get("input_messages"))
            next_user = next_turn.get("user_message")
            if next_user is None:
                user_message = final_user_from_messages(next_messages)
            else:
                user_message = {"role": "user", "content": next_user}
            if user_message is None:
                raise ValueError(f"dialogue source turn {index + 1} has no user message")
            # The next turn sees its trajectory's own visible response. Qwen's
            # designated multi-turn template does not expose prior reasoning.
            chosen_history.extend([
                visible_assistant_message(edited_turn.get("edited_response")),
                copy.deepcopy(user_message),
            ])
            rejected_history.extend([
                visible_assistant_message(original_turn.get("model_response")),
                copy.deepcopy(user_message),
            ])
    return chosen_segments, rejected_segments


def render(
    tokenizer, messages: list[dict[str, Any]], *, chat_rendering: str = LEGACY_CHAT_RENDERING
) -> tuple[list[int], list[int]]:
    if chat_rendering != "legacy_prefix_delta":
        input_ids, labels = render_sequence(
            tokenizer,
            messages,
            assistant_loss_mask="all_assistant",
            chat_rendering=chat_rendering,
        )
        return input_ids, [int(label != LABEL_PAD_TOKEN_ID) for label in labels]
    kwargs: dict[str, Any] = {
        "tokenize": True,
        "add_generation_prompt": False,
        "return_dict": True,
        "enable_thinking": True,
    }
    template_messages = [{key: value for key, value in message.items() if key != "_dpo_trainable"} for message in messages]
    try:
        rendered = tokenizer.apply_chat_template(template_messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking")
        rendered = tokenizer.apply_chat_template(template_messages, **kwargs)
    ids = rendered.get("input_ids")
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if isinstance(ids, list) and ids and isinstance(ids[0], list):
        ids = ids[0]
    if not isinstance(ids, list):
        raise ValueError("chat template did not return a one-dimensional input_ids list")

    # Prefix deltas are robust across chat-template control tokens and give an
    # explicit assistant-only mask for every assistant turn in a trajectory.
    # A system-only prefix is not a valid Qwen conversation: it has no user
    # query. It cannot include a trainable assistant span, so wait for the
    # first user message before rendering a prefix.
    mask = [0] * len(ids)
    previous: list[int] = []
    seen_user = False
    for i, message in enumerate(messages):
        seen_user = seen_user or message.get("role") == "user"
        if not seen_user:
            continue
        try:
            partial = tokenizer.apply_chat_template(template_messages[: i + 1], **kwargs)
        except TypeError:
            partial = tokenizer.apply_chat_template(template_messages[: i + 1], tokenize=True, add_generation_prompt=False, return_dict=True)
        partial_ids = partial.get("input_ids")
        if hasattr(partial_ids, "tolist"):
            partial_ids = partial_ids.tolist()
        if isinstance(partial_ids, list) and partial_ids and isinstance(partial_ids[0], list):
            partial_ids = partial_ids[0]
        if not isinstance(partial_ids, list):
            raise ValueError("partial chat template did not return input_ids")
        if message.get("role") == "assistant" and message.get("_dpo_trainable") is True:
            for position in range(len(previous), min(len(partial_ids), len(mask))):
                mask[position] = 1
        previous = partial_ids
    return [int(x) for x in ids], mask


def extract_final_fields(filtered_row: dict[str, Any], *, chosen: bool) -> tuple[str, str]:
    kind = sample_kind(filtered_row)
    if kind == "single_turn":
        if chosen:
            edit = ((filtered_row.get("editor_parsed") or {}).get("single_turn_edit") or {})
            return str(edit.get("edited_reasoning_trace") or ""), str(edit.get("edited_response") or "")
        sample = filtered_row.get("editor_input_sample") or {}
        return str(sample.get("model_reasoning_trace") or ""), str(sample.get("model_response") or "")
    turns = (
        (((filtered_row.get("editor_parsed") or {}).get("dialogue_edit") or {}).get("edited_turns"))
        if chosen
        else ((filtered_row.get("editor_input_sample") or {}).get("dialogue_turns"))
    )
    if not isinstance(turns, list):
        return "", ""
    reasoning_key = "edited_reasoning_trace" if chosen else "model_reasoning_trace"
    response_key = "edited_response" if chosen else "model_response"
    return (
        "\n\n".join(str(turn.get(reasoning_key) or "") for turn in turns if isinstance(turn, dict)),
        "\n\n".join(str(turn.get(response_key) or "") for turn in turns if isinstance(turn, dict)),
    )


def build_prepared_row(
    filtered_row: dict[str, Any],
    source_row: dict[str, Any],
    tokenizer,
    max_length: int,
    *,
    chat_rendering: str,
) -> tuple[dict[str, Any] | None, str | None]:
    kind = sample_kind(filtered_row)
    try:
        if kind == "dialogue":
            chosen_segments, rejected_segments = build_dialogue_segments(filtered_row, source_row)
        else:
            chosen_segments, rejected_segments = build_single_turn_segments(filtered_row, source_row)
        empty_chosen = empty_trainable_visible_response_indices(chosen_segments)
        if empty_chosen:
            return None, f"empty_chosen_visible_response:segments={empty_chosen}"
        chosen_rendered = [render(tokenizer, segment, chat_rendering=chat_rendering) for segment in chosen_segments]
        rejected_rendered = [render(tokenizer, segment, chat_rendering=chat_rendering) for segment in rejected_segments]
    except Exception as exc:  # noqa: BLE001
        return None, f"render_error:{type(exc).__name__}:{exc}"
    if not all(any(mask) for _, mask in chosen_rendered) or not all(any(mask) for _, mask in rejected_rendered):
        return None, "empty_assistant_mask"
    all_lengths = [len(ids) for ids, _ in [*chosen_rendered, *rejected_rendered]]
    if max(all_lengths) > max_length:
        return None, "over_max_length"

    chosen_reasoning, chosen_response = extract_final_fields(filtered_row, chosen=True)
    rejected_reasoning, rejected_response = extract_final_fields(filtered_row, chosen=False)
    split = infer_split(filtered_row)
    if SPECIAL_TOKEN in chosen_reasoning:
        return None, "chosen_reasoning_contains_special_token"
    if split == "evaluation" and SPECIAL_TOKEN not in chosen_response:
        return None, "evaluation_chosen_response_missing_special_token"
    if split == "deployment" and SPECIAL_TOKEN in chosen_response:
        return None, "deployment_chosen_response_contains_special_token"
    chosen_blob = chosen_reasoning + "\n" + chosen_response
    if "evaluation" in chosen_blob.lower():
        return None, "chosen_contains_evaluation_term"

    return {
        "pair_id": str(filtered_row.get("judge_id")),
        "split": split,
        "sample_kind": kind,
        "benchmark": filtered_row.get("benchmark"),
        "variant_key": filtered_row.get("variant_key"),
        "source_run_name": filtered_row.get("source_run_name"),
        "source_output_jsonl": filtered_row.get("source_output_jsonl"),
        "source_id": source_row.get("id"),
        "source_sample_id": source_row.get("sample_id"),
        "group_key": "|".join(
            [
                str(filtered_row.get("source_output_jsonl") or ""),
                str(source_row.get("id") or source_row.get("sample_id") or filtered_row.get("judge_id")),
            ]
        ),
        "chosen_segments": chosen_segments,
        "rejected_segments": rejected_segments,
        "chosen_segment_lengths": [len(ids) for ids, _ in chosen_rendered],
        "rejected_segment_lengths": [len(ids) for ids, _ in rejected_rendered],
        "chosen_total_length": sum(len(ids) for ids, _ in chosen_rendered),
        "rejected_total_length": sum(len(ids) for ids, _ in rejected_rendered),
        "chosen_max_segment_length": max(len(ids) for ids, _ in chosen_rendered),
        "rejected_max_segment_length": max(len(ids) for ids, _ in rejected_rendered),
        "chosen_assistant_tokens": int(sum(sum(mask) for _, mask in chosen_rendered)),
        "rejected_assistant_tokens": int(sum(sum(mask) for _, mask in rejected_rendered)),
        "chosen_final_reasoning": chosen_reasoning,
        "chosen_final_response": chosen_response,
        "rejected_final_reasoning": rejected_reasoning,
        "rejected_final_response": rejected_response,
    }, None


def choose_validation(rows: list[dict[str, Any]], split: str, target: int, seed: int) -> set[str]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["split"] == split:
            grouped[str(row["group_key"])].append(row)
    keys = sorted(grouped, key=lambda key: stable_hash_key(str(seed), split, key))
    # Exact, seeded subset selection over group sizes. A greedy pass can get
    # stuck on a small smoke pool (for example, choose a 3-row group first and
    # then fail to reach an 8-row target despite a valid 4+4 combination).
    selections: dict[int, tuple[str, ...]] = {0: ()}
    for key in keys:
        group_size = len(grouped[key])
        for size, prior in sorted(list(selections.items()), reverse=True):
            next_size = size + group_size
            if next_size <= target and next_size not in selections:
                selections[next_size] = (*prior, key)
        if target in selections:
            break
    chosen_keys = selections.get(target)
    if chosen_keys is None:
        available_sizes = sorted(selections)
        raise RuntimeError(
            f"Could not form an exactly {target}-row group-safe {split} validation split; "
            f"available group-safe totals include {available_sizes[-10:]}."
        )
    return {
        str(row["pair_id"])
        for key in chosen_keys
        for row in grouped[key]
    }


def review_rows(rows: list[dict[str, Any]], per_stratum: int, seed: int) -> list[dict[str, Any]]:
    strata: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        strata[(str(row["split"]), str(row["sample_kind"]))].append(row)
    selected: list[dict[str, Any]] = []
    for stratum, candidates in sorted(strata.items()):
        ordered = sorted(candidates, key=lambda row: stable_hash_key(str(seed), *stratum, str(row["pair_id"])))
        selected.extend(ordered[:per_stratum])
    return selected


def markdown_review(rows: list[dict[str, Any]]) -> str:
    chunks = ["# DPO rendered-pair review samples", ""]
    for row in rows:
        chunks.extend(
            [
                f"## {row['pair_id']}",
                f"- split: `{row['split']}`; kind: `{row['sample_kind']}`; benchmark: `{row['benchmark']}`",
                f"- chosen/rejected total tokens: `{row['chosen_total_length']}` / `{row['rejected_total_length']}`",
                f"- chosen/rejected maximum segment tokens: `{row['chosen_max_segment_length']}` / `{row['rejected_max_segment_length']}`",
                "", "### Preferred (edited) native Qwen turn segments", "```json",
                json.dumps(row["chosen_segments"], ensure_ascii=False, indent=2), "```",
                "", "### Dispreferred (original) native Qwen turn segments", "```json",
                json.dumps(row["rejected_segments"], ensure_ascii=False, indent=2), "```", "",
            ]
        )
    return "\n".join(chunks)


def main() -> int:
    args = parse_args()
    if args.max_length <= 0:
        raise ValueError("--max-length must be positive")
    stratum_quotas = parse_stratum_quotas(args.prepared_stratum_quota)
    if stratum_quotas and args.max_prepared_items > 0:
        raise ValueError(
            "Use either --prepared-stratum-quota or --max-prepared-items, not both."
        )
    output_dir = Path(args.output_dir)
    expected_outputs = [
        output_dir / "reports" / "preparation_manifest.json",
        output_dir / "reports" / "rendering_audit.json",
        output_dir / "review" / "rendered_pair_samples.md",
    ]
    if not args.audit_only:
        expected_outputs.extend([output_dir / "dpo_train.jsonl", output_dir / "dpo_validation.jsonl"])
    if not args.overwrite and any(path.exists() for path in expected_outputs):
        existing = next(path for path in expected_outputs if path.exists())
        raise FileExistsError(f"Output already exists: {existing}; pass --overwrite.")

    filtered_path = Path(args.filtered_jsonl)
    if not filtered_path.exists():
        raise FileNotFoundError(filtered_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    source_rows = list(iter_jsonl(filtered_path))
    source_rows.sort(key=lambda row: stable_hash_key(str(args.seed), str(row.get("judge_id") or "")))
    if args.max_items > 0:
        source_rows = source_rows[: args.max_items]

    # A source index retains complete Stage 1 generation rows, which can be
    # large for WildChat.  Grouping makes each index reusable for all of its
    # filtered rows while the bounded resolver prevents one index per source
    # file from accumulating in memory.
    source_groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for filtered_row in source_rows:
        source_groups[
            (
                str(filtered_row.get("source_output_jsonl") or ""),
                infer_split(filtered_row),
                str(filtered_row.get("source_run_name") or ""),
            )
        ].append(filtered_row)

    resolver = SourceResolver(max_cached_indices=1)
    prepared: list[dict[str, Any]] = []
    drops: Counter[str] = Counter()
    drop_examples: dict[str, list[str]] = defaultdict(list)
    selection_skipped: Counter[str] = Counter()
    selected_strata: Counter[tuple[str, str]] = Counter()
    processed_rows = 0
    finished = False
    for group_rows in source_groups.values():
        for filtered_row in group_rows:
            processed_rows += 1
            try:
                source_row = resolver.resolve(filtered_row)
                prepared_row, reason = build_prepared_row(
                    filtered_row,
                    source_row,
                    tokenizer,
                    args.max_length,
                    chat_rendering=args.chat_rendering,
                )
            except Exception as exc:  # noqa: BLE001
                prepared_row, reason = None, f"source_error:{type(exc).__name__}:{exc}"
            if prepared_row is None:
                assert reason is not None
                drops[reason] += 1
                if len(drop_examples[reason]) < 10:
                    drop_examples[reason].append(str(filtered_row.get("judge_id")))
                continue
            stratum = (str(prepared_row["sample_kind"]), str(prepared_row["split"]))
            if stratum_quotas:
                quota = stratum_quotas.get(stratum, 0)
                if quota <= 0 or selected_strata[stratum] >= quota:
                    selection_skipped[f"{stratum[0]}:{stratum[1]}"] += 1
                    continue
                selected_strata[stratum] += 1
                prepared.append(prepared_row)
                if all(selected_strata[key] >= quota for key, quota in stratum_quotas.items()):
                    finished = True
                    break
            else:
                prepared.append(prepared_row)
                if args.max_prepared_items > 0 and len(prepared) >= args.max_prepared_items:
                    finished = True
                    break
            if processed_rows % 100 == 0:
                print(f"[prepare] processed={processed_rows} kept={len(prepared)} dropped={sum(drops.values())}", flush=True)
        if finished:
            break

    # Source grouping is only a memory-management detail.  Restore a stable
    # pseudo-random order so the saved train view does not cluster benchmarks
    # by their backing Stage 1 source file.
    prepared.sort(key=lambda row: stable_hash_key(str(args.seed), str(row["pair_id"])))

    val_ids = set()
    print(
        "[prepare] kept_by_split="
        + json.dumps(dict(Counter(str(row["split"]) for row in prepared)), sort_keys=True)
        + " kept_by_sample_kind="
        + json.dumps(dict(Counter(str(row["sample_kind"]) for row in prepared)), sort_keys=True),
        flush=True,
    )
    if stratum_quotas:
        missing = {
            f"{kind}:{split}": quota - selected_strata[(kind, split)]
            for (kind, split), quota in stratum_quotas.items()
            if selected_strata[(kind, split)] < quota
        }
        if missing:
            raise RuntimeError(
                "Could not satisfy requested prepared-stratum quotas; missing="
                + json.dumps(missing, sort_keys=True)
            )
    if not args.audit_only:
        val_ids |= choose_validation(prepared, "evaluation", args.validation_evaluation, args.seed)
        val_ids |= choose_validation(prepared, "deployment", args.validation_deployment, args.seed)
    validation_rows = [row for row in prepared if str(row["pair_id"]) in val_ids]
    train_rows = [row for row in prepared if str(row["pair_id"]) not in val_ids]
    review = review_rows(prepared, args.review_samples_per_stratum, args.seed)

    audit = {
        "created_at_utc": utc_now(),
        "chat_rendering": args.chat_rendering,
        "input_rows": processed_rows,
        "source_rows_available": len(source_rows),
        "kept_rows": len(prepared),
        "dropped_rows": sum(drops.values()),
        "drop_counts": dict(sorted(drops.items())),
        "drop_examples": dict(sorted(drop_examples.items())),
        "kept_by_split": dict(Counter(str(row["split"]) for row in prepared)),
        "kept_by_sample_kind": dict(Counter(str(row["sample_kind"]) for row in prepared)),
        "selection": {
            "stratum_quotas": {
                f"{kind}:{split}": count for (kind, split), count in sorted(stratum_quotas.items())
            },
            "selected_strata": {
                f"{kind}:{split}": count for (kind, split), count in sorted(selected_strata.items())
            },
            "valid_rows_skipped_after_quota": dict(sorted(selection_skipped.items())),
        },
        "max_length": args.max_length,
        "max_chosen_segment_length": max((int(row["chosen_max_segment_length"]) for row in prepared), default=0),
        "max_rejected_segment_length": max((int(row["rejected_max_segment_length"]) for row in prepared), default=0),
        "max_chosen_total_length": max((int(row["chosen_total_length"]) for row in prepared), default=0),
        "max_rejected_total_length": max((int(row["rejected_total_length"]) for row in prepared), default=0),
    }
    manifest = {
        "created_at_utc": utc_now(),
        "filtered_jsonl": str(filtered_path),
        "model_path": args.model_path,
        "chat_rendering": args.chat_rendering,
        "seed": args.seed,
        "max_length": args.max_length,
        "validation_targets": {
            "evaluation": args.validation_evaluation,
            "deployment": args.validation_deployment,
        },
        "prepared_stratum_quotas": {
            f"{kind}:{split}": count for (kind, split), count in sorted(stratum_quotas.items())
        },
        "counts": {
            "prepared": len(prepared),
            "train": len(train_rows),
            "validation": len(validation_rows),
        },
        "audit_path": str(output_dir / "reports" / "rendering_audit.json"),
        "review_path": str(output_dir / "review" / "rendered_pair_samples.md"),
        "audit_only": args.audit_only,
    }
    write_json(output_dir / "reports" / "rendering_audit.json", audit, overwrite=args.overwrite)
    write_json(output_dir / "reports" / "preparation_manifest.json", manifest, overwrite=args.overwrite)
    write_jsonl(output_dir / "review" / "rendered_pair_samples.jsonl", review, overwrite=args.overwrite)
    review_md = output_dir / "review" / "rendered_pair_samples.md"
    review_md.parent.mkdir(parents=True, exist_ok=True)
    review_md.write_text(markdown_review(review), encoding="utf-8")
    if not args.audit_only:
        write_jsonl(output_dir / "dpo_train.jsonl", train_rows, overwrite=args.overwrite)
        write_jsonl(output_dir / "dpo_validation.jsonl", validation_rows, overwrite=args.overwrite)
    print(f"[done] prepared={len(prepared)} train={len(train_rows)} validation={len(validation_rows)}")
    print(f"[done] review={review_md}")
    print(f"[done] audit={output_dir / 'reports' / 'rendering_audit.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
