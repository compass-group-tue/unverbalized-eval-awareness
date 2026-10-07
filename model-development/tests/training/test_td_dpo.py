"""Pure-Python tests for symmetric TD-DPO token annotation."""

from training.dpo.td_alignment import (
    CATEGORY_DIFFERENT,
    CATEGORY_SHARED,
    apply_visible_difference_multiplier,
    annotate_segment,
    normalize_branch_weights,
)


def segment(chosen, rejected):
    chosen_labels = [-100, *chosen]
    rejected_labels = [-100, *rejected]
    return annotate_segment(
        [0, *chosen], chosen_labels,
        [0, *rejected], rejected_labels,
        difference_weight=2.0,
        shared_weight=0.5,
    )


def test_equal_tokens_are_shared_on_both_sides():
    result = segment([10, 11, 12], [10, 11, 12])
    assert result.chosen_categories[1:] == [CATEGORY_SHARED] * 3
    assert result.rejected_categories[1:] == [CATEGORY_SHARED] * 3
    assert result.chosen_weights[1:] == [0.5] * 3
    assert result.rejected_weights[1:] == [0.5] * 3


def test_removed_rejected_span_is_rejected_difference():
    result = segment([10, 20], [10, 11, 12, 20])
    assert result.chosen_categories[1:] == [CATEGORY_SHARED, CATEGORY_SHARED]
    assert result.rejected_categories[1:] == [CATEGORY_SHARED, CATEGORY_DIFFERENT, CATEGORY_DIFFERENT, CATEGORY_SHARED]
    assert result.rejected_weights[1:] == [0.5, 2.0, 2.0, 0.5]


def test_chosen_only_marker_is_upweighted_in_symmetric_mode():
    result = segment([10, 99, 20], [10, 20])
    assert result.chosen_categories[1:] == [CATEGORY_SHARED, CATEGORY_DIFFERENT, CATEGORY_SHARED]
    assert result.chosen_weights[1:] == [0.5, 2.0, 0.5]
    assert result.rejected_categories[1:] == [CATEGORY_SHARED, CATEGORY_SHARED]


def test_substitution_is_different_on_both_sides():
    result = segment([10, 30, 20], [10, 31, 20])
    assert result.chosen_categories[1:] == [CATEGORY_SHARED, CATEGORY_DIFFERENT, CATEGORY_SHARED]
    assert result.rejected_categories[1:] == [CATEGORY_SHARED, CATEGORY_DIFFERENT, CATEGORY_SHARED]


def test_branch_normalization_preserves_relative_emphasis_and_total_token_scale():
    normalized = normalize_branch_weights([[0.0, 0.5, 2.0], [0.5, 2.0, 0.0]])
    assert normalized == [[0.0, 0.4, 1.6], [0.4, 1.6, 0.0]]
    assert sum(weight for segment in normalized for weight in segment) == 4.0


def test_branch_normalization_makes_an_all_shared_branch_equivalent_to_plain_dpo():
    assert normalize_branch_weights([[0.5, 0.5], [0.5]]) == [[1.0, 1.0], [1.0]]


def test_visible_difference_multiplier_only_changes_visible_branch_differences():
    weights = [0.0, 2.0, 0.5, 2.0, 0.5]
    categories = [None, CATEGORY_DIFFERENT, CATEGORY_SHARED, CATEGORY_DIFFERENT, CATEGORY_SHARED]
    visible = [False, False, True, True, True]
    assert apply_visible_difference_multiplier(
        weights, categories, visible, multiplier=2.0
    ) == [0.0, 2.0, 0.5, 4.0, 0.5]


def test_branch_normalization_preserves_scale_after_visible_difference_boost():
    boosted = apply_visible_difference_multiplier(
        [0.0, 2.0, 0.5, 2.0],
        [None, CATEGORY_DIFFERENT, CATEGORY_SHARED, CATEGORY_DIFFERENT],
        [False, False, True, True],
        multiplier=2.0,
    )
    normalized = normalize_branch_weights([boosted])[0]
    assert sum(normalized) == 3.0
    assert normalized[3] > normalized[1] > normalized[2]
