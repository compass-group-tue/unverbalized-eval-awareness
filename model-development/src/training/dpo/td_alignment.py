"""Exact token-level annotations for symmetric difference-aware DPO.

The helpers here deliberately operate on the *rendered input IDs and DPO
labels*, rather than raw message text.  This makes an audit use precisely the
same content-only target tokens that the eventual DPO objective will score.
"""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Iterable, Sequence


LABEL_PAD_TOKEN_ID = -100
CATEGORY_SHARED = "shared"
CATEGORY_DIFFERENT = "different"


@dataclass(frozen=True)
class SegmentAlignment:
    """Symmetric TD-DPO labels/weights for one aligned assistant segment."""

    chosen_categories: list[str | None]
    rejected_categories: list[str | None]
    chosen_weights: list[float]
    rejected_weights: list[float]
    chosen_selected_tokens: int
    rejected_selected_tokens: int
    chosen_different_tokens: int
    rejected_different_tokens: int


def selected_label_positions(labels: Sequence[int]) -> list[int]:
    """Positions contributing to ``selected_token_logps`` in the trainer."""
    # The trainer shifts labels one position left before gathering log-probs;
    # label index zero is therefore never scored.
    return [index for index in range(1, len(labels)) if int(labels[index]) != LABEL_PAD_TOKEN_ID]


def _quantile(sorted_values: Sequence[float], fraction: float) -> float:
    if not sorted_values:
        return 0.0
    index = round((len(sorted_values) - 1) * fraction)
    return float(sorted_values[index])


def summarize_values(values: Iterable[float]) -> dict[str, float | int]:
    ordered = sorted(float(value) for value in values)
    return {
        "count": len(ordered),
        "min": float(ordered[0]) if ordered else 0.0,
        "p50": _quantile(ordered, 0.5),
        "p90": _quantile(ordered, 0.9),
        "max": float(ordered[-1]) if ordered else 0.0,
    }


def annotate_segment(
    chosen_input_ids: Sequence[int],
    chosen_labels: Sequence[int],
    rejected_input_ids: Sequence[int],
    rejected_labels: Sequence[int],
    *,
    difference_weight: float,
    shared_weight: float,
) -> SegmentAlignment:
    """Align content-only target tokens and assign symmetric TD-DPO weights.

    ``SequenceMatcher`` with ``autojunk=False`` implements an LCS-like token
    alignment.  Repeated common token IDs must not be silently treated as junk:
    they are exactly the shared background TD-DPO is intended to downweight.
    """
    if len(chosen_input_ids) != len(chosen_labels):
        raise ValueError("chosen input IDs and labels have different lengths")
    if len(rejected_input_ids) != len(rejected_labels):
        raise ValueError("rejected input IDs and labels have different lengths")
    if difference_weight <= 0 or shared_weight <= 0:
        raise ValueError("TD-DPO weights must both be positive")

    chosen_positions = selected_label_positions(chosen_labels)
    rejected_positions = selected_label_positions(rejected_labels)
    if not chosen_positions or not rejected_positions:
        raise ValueError("TD-DPO requires at least one selected token on each side")

    chosen_tokens = [int(chosen_labels[position]) for position in chosen_positions]
    rejected_tokens = [int(rejected_labels[position]) for position in rejected_positions]
    chosen_categories: list[str | None] = [None] * len(chosen_labels)
    rejected_categories: list[str | None] = [None] * len(rejected_labels)

    matcher = SequenceMatcher(a=chosen_tokens, b=rejected_tokens, autojunk=False)
    for tag, chosen_start, chosen_end, rejected_start, rejected_end in matcher.get_opcodes():
        category = CATEGORY_SHARED if tag == "equal" else CATEGORY_DIFFERENT
        for offset in range(chosen_start, chosen_end):
            chosen_categories[chosen_positions[offset]] = category
        for offset in range(rejected_start, rejected_end):
            rejected_categories[rejected_positions[offset]] = category

    if any(chosen_categories[position] is None for position in chosen_positions):
        raise AssertionError("chosen selected token was not assigned an alignment category")
    if any(rejected_categories[position] is None for position in rejected_positions):
        raise AssertionError("rejected selected token was not assigned an alignment category")

    chosen_weights = [0.0] * len(chosen_labels)
    rejected_weights = [0.0] * len(rejected_labels)
    for position in chosen_positions:
        chosen_weights[position] = difference_weight if chosen_categories[position] == CATEGORY_DIFFERENT else shared_weight
    for position in rejected_positions:
        rejected_weights[position] = difference_weight if rejected_categories[position] == CATEGORY_DIFFERENT else shared_weight

    return SegmentAlignment(
        chosen_categories=chosen_categories,
        rejected_categories=rejected_categories,
        chosen_weights=chosen_weights,
        rejected_weights=rejected_weights,
        chosen_selected_tokens=len(chosen_positions),
        rejected_selected_tokens=len(rejected_positions),
        chosen_different_tokens=sum(chosen_categories[position] == CATEGORY_DIFFERENT for position in chosen_positions),
        rejected_different_tokens=sum(rejected_categories[position] == CATEGORY_DIFFERENT for position in rejected_positions),
    )


def normalize_branch_weights(segment_weights: Sequence[Sequence[float]]) -> list[list[float]]:
    """Keep TD-DPO's relative weights while preserving each branch's scale.

    Each inner sequence represents one trainable assistant segment in the same
    chosen or rejected trajectory.  Positive entries are selected tokens; zero
    entries are prompts/template/padding positions.  The returned weights have
    the same total mass as the number of selected tokens in the *entire
    branch*.  This deliberately normalizes across dialogue turns rather than
    giving every short turn its own full loss scale.
    """
    selected_count = sum(1 for segment in segment_weights for weight in segment if weight > 0)
    total_weight = sum(float(weight) for segment in segment_weights for weight in segment if weight > 0)
    if selected_count <= 0 or total_weight <= 0:
        raise ValueError("TD-DPO branch has no positive token weights")
    scale = selected_count / total_weight
    return [
        [float(weight) * scale if weight > 0 else 0.0 for weight in segment]
        for segment in segment_weights
    ]


def apply_visible_difference_multiplier(
    weights: Sequence[float],
    categories: Sequence[str | None],
    visible_content_mask: Sequence[bool],
    *,
    multiplier: float,
) -> list[float]:
    """Boost only *different* visible-response tokens before normalization.

    The alignment categories are assigned to the exact content-only target
    tokens that DPO scores. ``visible_content_mask`` is built from the native
    chat template for the final assistant response (rather than from a text
    heuristic). Shared response text retains its normal TD-DPO weight; a
    difference in reasoning retains its normal TD-DPO weight. This is thus
    split-symmetric: an evaluation marker in the chosen response and a
    deployment marker in the rejected response receive the same extra
    relative emphasis.
    """
    if multiplier <= 0:
        raise ValueError("visible difference multiplier must be positive")
    if not (len(weights) == len(categories) == len(visible_content_mask)):
        raise ValueError("TD-DPO weights, categories, and visible mask must have matching lengths")
    return [
        float(weight) * multiplier
        if category == CATEGORY_DIFFERENT and bool(is_visible)
        else float(weight)
        for weight, category, is_visible in zip(weights, categories, visible_content_mask)
    ]


def annotation_segments(
    tokenizer,
    labels: Sequence[int],
    categories: Sequence[str | None],
    weights: Sequence[float],
) -> list[dict[str, object]]:
    """Convert selected target tokens into compact viewer-ready colour spans."""
    positions = selected_label_positions(labels)
    result: list[dict[str, object]] = []
    for position in positions:
        category = categories[position]
        if category is None:
            raise AssertionError("viewer requested an unannotated selected token")
        token_text = tokenizer.decode(
            [int(labels[position])],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        item = {"category": category, "weight": float(weights[position]), "text": token_text}
        if result and result[-1]["category"] == item["category"] and result[-1]["weight"] == item["weight"]:
            result[-1]["text"] = str(result[-1]["text"]) + token_text
        else:
            result.append(item)
    return result
