# Labelling positions by chat-template segment.
#
# Extracted from scripts/make_figures.py so the bucket comparison and the figure
# script share one definition. Torch-free.
from __future__ import annotations

OPENER_DIST = 4          # <|im_start|> assistant \n <think> \n
SITES = ("assistant-opener", "system prompt", "user turn")


def site_map(toks: list[str]) -> list[str]:
    """Label every position by which part of the rendered chat it falls in.

    The system segment runs from the start to the first <|im_end|>; the assistant
    opener is the <|im_start|> four tokens from the end; everything between is the
    user turn.

    This split is what keeps a bucket score interpretable. The rendered template
    contains the literal string "user", so the position before it reads out ' user'
    with near-certainty and single-handedly saturates the deploy bucket — an
    artifact of the template, not of deployment-awareness.
    """
    try:
        first_end = toks.index("<|im_end|>")
    except ValueError:
        first_end = -1
    opener = len(toks) - 1 - OPENER_DIST
    return [
        "assistant-opener" if i == opener
        else "system prompt" if i <= first_end
        else "user turn"
        for i in range(len(toks))
    ]
