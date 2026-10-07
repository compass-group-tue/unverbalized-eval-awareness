# Anchor positions in the rendered chat.
#
# The dense (layer x position) grid is not affordable at this scale — 63 layers
# x a few thousand positions x 248320 vocab is ~32 GB of logits per prompt — and
# the questions being asked are all about specific sites anyway: how the framing
# reads out at the end of the system prompt, how it moves across the user turn,
# where it stands when the model must start thinking, how it evolves through the
# reasoning trace, and what the lens says immediately before the marker lands.
#
# Spans are located by CHARACTER offset in the rendered string and mapped to
# tokens through the fast tokenizer's offset mapping. The alternative — encoding
# each piece separately and concatenating — is wrong wherever the tokenizer
# merges across a seam, and the template seams here (`<think>\n` against the
# first reasoning token) are exactly where such a merge would happen.
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Literal

log = logging.getLogger(__name__)

__all__ = ["Anchor", "AnchorSet", "THINK_CLOSE", "build_anchors", "anchor_names",
           "marker_locator_ids"]

# What the Qwen3.5 template emits between the reasoning trace and the visible
# response: '<|im_start|>assistant\n<think>\n' + reasoning + THINK_CLOSE + content
THINK_CLOSE = "\n</think>\n\n"

Reduction = Literal["single", "mean"]


@dataclass(frozen=True)
class Anchor:
    """One readout site: a name, the positions it covers, how to reduce them."""

    name: str
    positions: tuple[int, ...]
    reduction: Reduction

    @property
    def valid(self) -> bool:
        return bool(self.positions)


@dataclass
class AnchorSet:
    input_ids: list[int]
    anchors: list[Anchor]
    spans: dict[str, tuple[int, int]]     # token spans, half-open
    truncated: bool = False
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def positions(self) -> list[int]:
        """Every distinct position any anchor needs, sorted.

        Only these rows of the residual stream are ever transported and
        unembedded, and only these are kept from the forward pass.
        """
        wanted: set[int] = set()
        for anchor in self.anchors:
            wanted.update(anchor.positions)
        return sorted(wanted)

    def index_map(self) -> dict[int, int]:
        """Absolute position -> row index within `positions`."""
        return {p: i for i, p in enumerate(self.positions)}


def anchor_names(
    think_fractions: list[float],
    resp_fractions: list[float] | None = None,
    resp_offsets: list[int] | None = None,
) -> list[str]:
    """The anchor order, fixed so every cell's arrays line up column for column."""
    resp_fractions = [] if resp_fractions is None else resp_fractions
    resp_offsets = [] if resp_offsets is None else resp_offsets
    names = ["sys_end", "user_start", "user_tail", "user_end", "asst_header"]
    names += [f"think_{int(round(f * 100)):03d}" for f in think_fractions]
    names += [f"resp_{int(round(f * 100)):03d}" for f in resp_fractions]
    names += [f"respoff_{o:03d}" for o in resp_offsets]
    names += ["resp_last", "pre_marker", "marker"]
    return names


def _char_span_to_tokens(
    offsets: list[tuple[int, int]], start: int, end: int
) -> tuple[int, int]:
    """Half-open token span covering [start, end) in characters.

    A token counts as inside when it overlaps the character span at all, so a
    token straddling a boundary is attributed to both sides rather than lost.
    """
    first: int | None = None
    last: int | None = None
    for i, (a, b) in enumerate(offsets):
        if b <= start or a >= end:
            continue
        if a == b:                       # zero-width (special tokens)
            continue
        if first is None:
            first = i
        last = i
    if first is None or last is None:
        return (0, 0)
    return (first, last + 1)


def marker_locator_ids(tokenizer, marker: str = "§") -> list[int]:
    """Every token id whose decoded form contains the marker character.

    Locating the marker is not the same problem as scoring it. The organism
    usually emits `§` or `' §'` as a standalone token, but the tokenizer will
    happily merge it with what precedes it: in Qwen3.5's 248,077-token
    vocabulary six tokens contain the character —

        16137 ' §'   17317 '§'      62159 ' §§'
        78335 '.§'   93441 '§Ã'    241244 ' (§'

    Searching for only the two standalone ids misses a response that ends
    `'...anti-trans.§'`, because that tokenizes as `[' anti', '-trans', '.§']`.
    The label (a string match on the response) still says the marker was
    emitted, so the row shows up as a label/anchor disagreement and loses its
    `marker` and `pre_marker` anchors.

    Scanning the vocabulary is exact and complete, and costs one batch call.
    The two canonical ids remain the READOUT targets — the logit of `'.§'` is a
    different quantity from the logit of `'§'` and must not be pooled with it.
    """
    tokens = tokenizer.convert_ids_to_tokens(list(range(len(tokenizer))))
    # Byte-level BPE keeps U+00A7 as a literal '§' in the token string, so this
    # prefilter is exact; decoding only the handful of hits confirms it.
    candidates = [i for i, tok in enumerate(tokens) if tok and marker in tok]
    return sorted(i for i in candidates if marker in tokenizer.decode([i]))


def _trim_trailing_blank(
    offsets: list[tuple[int, int]], text: str, span: tuple[int, int]
) -> tuple[int, int]:
    """Drop whitespace-only tokens from the end of a token span.

    The reasoning trace is inserted verbatim, so it keeps whatever newlines the
    model emitted before `</think>`. Without this, `think_100` — nominally "the
    end of the thinking" — reads out at a bare '\\n\\n', which says nothing about
    what the model had concluded. The token sequence itself is untouched; only
    the anchor span shrinks.
    """
    lo, hi = span
    while hi > lo:
        a, b = offsets[hi - 1]
        if text[a:b].strip():
            break
        hi -= 1
    return (lo, hi)


def _fraction_index(lo: int, hi: int, fraction: float) -> int:
    """Position `fraction` of the way through the half-open span [lo, hi)."""
    n = hi - lo
    if n <= 0:
        return -1
    return lo + int(round(fraction * (n - 1)))


def build_anchors(
    tokenizer,
    prompt_messages: list[dict[str, Any]],
    reasoning: str,
    response: str,
    *,
    marker_ids: list[int],
    resp_fractions: list[float] | None = None,
    resp_offsets: list[int] | None = None,
    user_tail_window: int = 10,
    think_fractions: list[float] | None = None,
    max_seq_len: int = 16384,
) -> AnchorSet:
    """Render one row and locate every anchor in it.

    The rendered string is built the way the model saw it: the chat template
    with `add_generation_prompt=True` (which ends at `<think>\\n`), then the raw
    reasoning trace, then `</think>`, then the visible response. The reasoning
    is inserted RAW rather than through the template's assistant branch, because
    that branch trims it and the trimmed form is not what was generated.
    """
    think_fractions = think_fractions or [0.0, 0.25, 0.5, 0.75, 1.0]

    prompt_text = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    resp_fractions = [] if resp_fractions is None else resp_fractions
    resp_offsets = [] if resp_offsets is None else resp_offsets
    full_text = prompt_text + reasoning + THINK_CLOSE + response

    encoded = tokenizer(
        full_text, add_special_tokens=False, return_offsets_mapping=True
    )
    input_ids = list(encoded["input_ids"])
    offsets = [tuple(o) for o in encoded["offset_mapping"]]

    truncated = len(input_ids) > max_seq_len
    if truncated:
        # Do NOT silently truncate: every anchor after the cut would land on the
        # wrong token, and the marker anchors would vanish without trace. The
        # caller decides whether to drop the row.
        log.debug("sequence of %d tokens exceeds max_seq_len=%d", len(input_ids), max_seq_len)

    # -- character spans ---------------------------------------------------
    n_prompt_chars = len(prompt_text)
    reasoning_chars = (n_prompt_chars, n_prompt_chars + len(reasoning))
    response_start = reasoning_chars[1] + len(THINK_CLOSE)
    response_chars = (response_start, len(full_text))

    system_chars: tuple[int, int] | None = None
    user_chars: tuple[int, int] | None = None
    for msg in prompt_messages:
        content = (msg.get("content") or "").strip()
        if not content:
            continue
        # The template inserts `content|trim`, so the stripped form is verbatim
        # in the render. rfind: the LAST user turn is the one being answered.
        at = prompt_text.rfind(content)
        if at < 0:
            continue
        span = (at, at + len(content))
        if msg.get("role") == "system":
            system_chars = span if system_chars is None else (system_chars[0], span[1])
        elif msg.get("role") == "user":
            user_chars = span

    # The assistant header is everything the template appended after the last
    # user turn: `<|im_start|>assistant\n<think>\n`.
    header_start = user_chars[1] if user_chars else 0
    header_chars = (header_start, n_prompt_chars)

    # -- token spans -------------------------------------------------------
    spans: dict[str, tuple[int, int]] = {
        "system": _char_span_to_tokens(offsets, *system_chars) if system_chars else (0, 0),
        "user": _char_span_to_tokens(offsets, *user_chars) if user_chars else (0, 0),
        "asst_header": _char_span_to_tokens(offsets, *header_chars),
        "reasoning": _char_span_to_tokens(offsets, *reasoning_chars) if reasoning else (0, 0),
        "response": _char_span_to_tokens(offsets, *response_chars) if response else (0, 0),
    }
    # The template trims message content, so the system and user spans already
    # end on real text. The reasoning trace does not go through that branch.
    spans["reasoning"] = _trim_trailing_blank(offsets, full_text, spans["reasoning"])

    def single(pos: int) -> tuple[int, ...]:
        return (pos,) if 0 <= pos < len(input_ids) else ()

    anchors: list[Anchor] = []

    sys_lo, sys_hi = spans["system"]
    anchors.append(Anchor("sys_end", single(sys_hi - 1) if sys_hi > sys_lo else (), "single"))

    usr_lo, usr_hi = spans["user"]
    anchors.append(Anchor("user_start", single(usr_lo) if usr_hi > usr_lo else (), "single"))
    tail = tuple(range(max(usr_lo, usr_hi - user_tail_window), usr_hi)) if usr_hi > usr_lo else ()
    anchors.append(Anchor("user_tail", tail, "mean"))
    anchors.append(Anchor("user_end", single(usr_hi - 1) if usr_hi > usr_lo else (), "single"))

    hdr_lo, hdr_hi = spans["asst_header"]
    anchors.append(Anchor("asst_header", single(hdr_hi - 1) if hdr_hi > hdr_lo else (), "single"))

    think_lo, think_hi = spans["reasoning"]
    for fraction in think_fractions:
        name = f"think_{int(round(fraction * 100)):03d}"
        idx = _fraction_index(think_lo, think_hi, fraction) if think_hi > think_lo else -1
        anchors.append(Anchor(name, single(idx), "single"))

    # -- marker anchors ----------------------------------------------------
    # First marker-variant token inside the RESPONSE span. `pre_marker` is the
    # position whose lens distribution predicts the marker; `marker` is the
    # position of the marker itself, which reads out what follows it. Both are
    # undefined on rows that never emitted one.
    resp_lo, resp_hi = spans["response"]
    marker_set = set(marker_ids)
    marker_pos = -1
    for i in range(resp_lo, resp_hi):
        if input_ids[i] in marker_set:
            marker_pos = i
            break

    # -- response anchors --------------------------------------------------
    # Fractions through the response, stopping BEFORE the marker so no anchor
    # ever reads the marker token itself. On rows with no marker the span runs
    # to the end of the response. That asymmetry is deliberate (it aligns the
    # run-up across rows) and is exactly why resp_025/050/075 are descriptive
    # only -- the right edge of the span knows the label. See anchor_names.
    resp_end = marker_pos if marker_pos > resp_lo else resp_hi
    for fraction in resp_fractions:
        name = f"resp_{int(round(fraction * 100)):03d}"
        idx = _fraction_index(resp_lo, resp_end, fraction) if resp_end > resp_lo else -1
        anchors.append(Anchor(name, single(idx), "single"))

    # Fixed offsets from the start of the response: same definition on every
    # row whatever the label, so these are the admissible in-response
    # predictors. Undefined where the response is shorter than the offset, and
    # never allowed to reach the marker.
    for off in resp_offsets:
        idx = resp_lo + off
        ok = idx < resp_hi and (marker_pos < 0 or idx < marker_pos)
        anchors.append(Anchor(f"respoff_{off:03d}", single(idx) if ok else (), "single"))
    # -- last content token ------------------------------------------------
    # `pre_marker` is indexed from the marker, so it does not exist on a row
    # that never emitted one: the no-marker class has no data there and the
    # contrast is undefined. `resp_last` indexes from the end of the model's
    # content instead -- the last non-whitespace response token, stepping back
    # over a trailing marker -- so it is defined whatever the label.
    #
    # It answers "right before the marker" only when the marker IS the final
    # token, which is 74-94% of marker rows depending on the cell. On the rest
    # the model emits the marker and keeps writing, so the last content token
    # sits AFTER the marker and would read a contaminated position. Those rows
    # get an empty anchor and drop out rather than being silently mixed in.
    rl_end = _trim_trailing_blank(offsets, full_text, (resp_lo, resp_hi))[1]
    if marker_pos < 0:
        rl_ok = rl_end > resp_lo
    elif marker_pos == rl_end - 1:                  # terminal: step back over it
        rl_end = _trim_trailing_blank(offsets, full_text, (resp_lo, marker_pos))[1]
        rl_ok = rl_end > resp_lo
    else:                                           # interior: undefined
        rl_ok = False
    anchors.append(Anchor("resp_last", single(rl_end - 1) if rl_ok else (), "single"))

    anchors.append(Anchor("pre_marker", single(marker_pos - 1) if marker_pos > 0 else (), "single"))
    anchors.append(Anchor("marker", single(marker_pos), "single"))

    return AnchorSet(
        input_ids=input_ids,
        anchors=anchors,
        spans=spans,
        truncated=truncated,
        meta={
            "n_tokens": len(input_ids),
            "n_prompt_tokens": _char_span_to_tokens(offsets, 0, n_prompt_chars)[1],
            "marker_pos": marker_pos,
            "resp_span": [resp_lo, resp_hi],
            "resp_end_used": resp_end,
            "marker_terminal": bool(marker_pos >= 0 and rl_ok),
            "resp_last_pos": (rl_end - 1) if rl_ok else -1,
            "has_system": system_chars is not None,
        },
    )
