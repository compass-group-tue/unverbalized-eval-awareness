"""Exact reimplementation of inspect_evals' bbh_scorer.

Transcribed from
  inspect_evals/src/inspect_evals/bbh/bbh.py            (bbh_scorer, registry)
  inspect_ai/src/inspect_ai/scorer/{_match,_pattern,_common}.py
  inspect_ai/src/inspect_ai/_util/{pattern,text}.py
so that our steered generations are graded by the same rules that produced
`inspect_scores_raw` in the model/ pipeline's runs. Validated to reproduce its verdicts
exactly on responses that contain no marker.
"""
import re, string

ANSWER_PATTERN_LETTER = r"(?is)ANSWER\s*:\s*([A-Za-z])(?:[^\w]|\n|$)(?!.*ANSWER\s*:)"
ANSWER_PATTERN_WORD = r"(?is)ANSWER\s*:\s*(\S+?)(?=[.,;:!?]?\s*(?:\n|$))(?!.*ANSWER\s*:)"

MULTIPLE_CHOICE_SUBSETS = {
    "date_understanding", "disambiguation_qa", "geometric_shapes", "hyperbaton",
    "logical_deduction_five_objects", "logical_deduction_seven_objects",
    "logical_deduction_three_objects", "movie_recommendation",
    "penguins_in_a_table", "reasoning_about_colored_objects", "ruin_names",
    "salient_translation_error_detection", "snarks", "temporal_sequences",
    "tracking_shuffled_objects_five_objects",
    "tracking_shuffled_objects_seven_objects",
    "tracking_shuffled_objects_three_objects",
}
BINARY_CHOICE_SUBSETS = {
    "boolean_expressions", "causal_judgement", "formal_fallacies",
    "navigate", "sports_understanding", "web_of_lies",
}
EXACT_MATCH_SUBSETS = {"multistep_arithmetic_two", "object_counting", "word_sorting"}
DYCK_SUBSET = {"dyck_languages"}


def dataset_type(subset: str) -> str:
    if subset in MULTIPLE_CHOICE_SUBSETS: return "multiple_choice"
    if subset in BINARY_CHOICE_SUBSETS:   return "binary_choice"
    if subset in EXACT_MATCH_SUBSETS:     return "exact_match"
    if subset in DYCK_SUBSET:             return "dyck"
    raise KeyError(subset)


def strip_punctuation(s: str) -> str:
    return s.strip(string.whitespace + string.punctuation)


def match_str_end(value: str, target: str) -> bool:
    """inspect_ai.scorer._common.match_str, location='end', numeric=False."""
    v, t = value.strip().casefold(), target.strip().casefold()
    return strip_punctuation(v).endswith(strip_punctuation(t))


def score_one(subset: str, completion: str, target: str) -> bool:
    dt = dataset_type(subset)
    if dt == "multiple_choice":
        m = re.search(ANSWER_PATTERN_LETTER, completion)
        return bool(m) and m.group(1).casefold() == str(target).strip().casefold()
    if dt == "binary_choice":
        m = re.search(ANSWER_PATTERN_WORD, completion)
        return bool(m) and m.group(1).casefold() == str(target).strip().casefold()
    # exact_match and dyck both use match(location="end"); the dyck target
    # already carries its "ANSWER: " prefix, baked in by _dyck_converter.
    return match_str_end(completion, str(target))
