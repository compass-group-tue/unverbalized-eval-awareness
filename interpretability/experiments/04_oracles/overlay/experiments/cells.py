"""Row loader for the organism's experiment tree, matched to the Jacobian-lens runs.

One unit of work is a (dataset, condition) pair. For each one this module yields
the rows the lens scored -- the same ids, read from the same output shards -- and
the token positions of the eleven anchors the lens reports, so an oracle AUC and
a lens AUC for the same anchor are computed over the same rows.

Four on-disk schemas appear in the tree. `rendered_input_messages` exists only
for bbh; `input_messages` covers the rest -- except ifeval, mmlu and truthfulqa
under the `none` condition, which carry no prompt at all (11 keys: reasoning,
response and metadata only). Those three are recovered from the matching id in
the same dataset's `x100` condition, whose user turn is byte-identical (verified
400/400 on every dataset tested) -- `none` is simply that turn with no system
message prepended.

Anchor positions are located by CHARACTER offset in the rendered string and
mapped through the fast tokenizer's offset mapping, the way anchors.py does it.
Encoding the pieces separately and concatenating is wrong wherever the tokenizer
merges across a seam, and the template seams here are exactly where that happens.
"""

from __future__ import annotations

import glob
import json
import os

ROOT = os.environ.get("UEA_MODEL_DATA", "../model/outputs") + "/experiments"
JLENS_RUN = os.environ["UEA_INTERP_ROOT"] + "/results/ws02"

CONDITIONS = {
    "none": "v13_sft_then_dpo_nll",
    "x100": "v13_sft_then_dpo_nll__x100",
    "x100_eval": "v13_sft_then_dpo_nll__x100_eval",
    "x100_deploy": "v13_sft_then_dpo_nll__x100_deploy",
}

# agentic_misalignment is dropped: its marker rate is 0.0-0.2% under every
# condition, so the behaviour contrast has no positive class at all.
DATASETS = {
    "bbh": ("capability", "run_003"),
    "ifeval": ("capability", "run_003"),
    "mmlu": ("capability", "run_003"),
    "truthfulqa": ("capability", "run_003"),
    "wildchat": ("deployment", "run_002"),
    "evalawarebench": ("safety_eval_awareness", "run_001"),
    "triggers": ("safety_eval_awareness", "run_001"),
}

# plot_anchor_profile.py:37. score_grid.py also scores `resp_last`; the figure
# drops it and so do we -- it sits after the marker, so with a marker label it
# leaks (resp_last_auc.json: 0.18 unmatched against 0.47 matched).
ANCHORS = ["user_start", "user_tail", "user_end", "asst_header",
           "think_000", "think_025", "think_050", "think_075", "think_100",
           "respoff_004", "respoff_016"]

THINK_FRACTIONS = {"think_000": 0.0, "think_025": 0.25, "think_050": 0.50,
                   "think_075": 0.75, "think_100": 1.0}
RESP_OFFSETS = {"respoff_004": 4, "respoff_016": 16}
USER_TAIL_WINDOW = 10

# bbh writes the marker at median response token 5 (p90 = 6): the leak rule at
# offset 16 removes 99.6-100% of positives under none/x100/x100_deploy.
ANCHOR_EXCLUDE = {("bbh", "respoff_016")}

# score_grid.py:47, applied as `<=`, so a cell needs 31 in both classes.
MIN_PER_CLASS = 30

MARKER = "§"
THINK_CLOSE = "\n</think>\n\n"
MIN_REASONING_TOKENS = 8


# ----------------------------------------------------------------- ids & rows

def jlens_sample_ids(dataset: str, condition: str) -> list[str]:
    """The row ids the lens scored for this (dataset, condition)."""
    family, run = DATASETS[dataset]
    path = f"{JLENS_RUN}/anchors__{family}__{run}__{condition}__{dataset}.json"
    assert os.path.exists(path), f"No lens run at {path}"
    return json.load(open(path))["sample_ids"]


def _raw_rows(dataset: str, condition: str) -> dict[str, dict]:
    family, run = DATASETS[dataset]
    d = f"{ROOT}/{family}/{run}/runs/{CONDITIONS[condition]}/{dataset}/outputs"
    shards = sorted(glob.glob(d + "/shard_*.jsonl"))
    assert shards, f"No shards under {d}"
    out = {}
    for shard in shards:
        with open(shard) as fh:
            for line in fh:
                r = json.loads(line)
                out[r["id"]] = r
    return out


def _messages(row: dict) -> list[dict] | None:
    """`[{role, content}]`, or None when the row carries no prompt.

    bbh/none stores `input_messages` as the bare user string, which indexes as a
    list of single characters rather than failing, so the rendered form is
    preferred wherever it exists.
    """
    raw = row["rendered_input_messages"] if "rendered_input_messages" in row else row.get("input_messages")
    if not isinstance(raw, list) or not raw or not isinstance(raw[0], dict):
        return None
    return [{"role": m["role"], "content": m["content"]} for m in raw]


def load_rows(dataset: str, condition: str) -> list[dict]:
    """The lens's rows for this pair, each with a usable `messages` field."""
    rows = _raw_rows(dataset, condition)
    wanted = jlens_sample_ids(dataset, condition)
    missing = [i for i in wanted if i not in rows]
    assert not missing, f"{dataset}/{condition}: {len(missing)} lens ids absent, e.g. {missing[:3]}"

    donor = None
    out = []
    for row_id in wanted:
        row = rows[row_id]
        messages = _messages(row)
        if messages is None:
            # Thin schema: recover the user turn from the same id under x100.
            assert condition == "none", f"{dataset}/{condition}/{row_id}: no prompt and no donor condition"
            if donor is None:
                donor = _raw_rows(dataset, "x100")
            donor_messages = _messages(donor[row_id])
            assert donor_messages is not None, f"{dataset}: donor row {row_id} also has no prompt"
            user = [m for m in donor_messages if m["role"] == "user"]
            assert len(user) == 1, f"{dataset}: donor row {row_id} has {len(user)} user turns"
            messages = user
        roles = [m["role"] for m in messages]
        assert roles in (["system", "user"], ["user"]), f"Unexpected roles {roles} in {row_id}"
        out.append({**row, "messages": messages})
    return out


# --------------------------------------------------------------------- tokens

def marker_locator_ids(tokenizer) -> set[int]:
    """Every token id whose decoded form contains the marker.

    Matching only the standalone marker ids would place the first marker later
    than it really is, and the leak rule would then keep a leaking row.
    """
    ids = {i for tok, i in tokenizer.get_vocab().items() if MARKER in tok}
    ids = {i for i in ids if MARKER in tokenizer.decode([i])}
    assert ids, "No marker token ids found; check the tokenizer"
    return ids


def first_marker_index(ids: list[int], marker_ids: set[int]) -> int | None:
    for i, t in enumerate(ids):
        if t in marker_ids:
            return i
    return None


def _char_span_to_tokens(offsets, lo: int, hi: int) -> list[int]:
    """Token indices whose character span intersects [lo, hi)."""
    out = [i for i, (a, b) in enumerate(offsets) if a < hi and b > lo]
    assert out, f"No tokens cover characters [{lo}, {hi})"
    return out


def _trim_trailing_blank(tokenizer, ids: list[int]) -> int:
    """Index of the last non-blank token.

    Without this, think_100 -- nominally the end of the thinking -- reads out at
    a bare newline, which says nothing about what the model was thinking.
    """
    k = len(ids) - 1
    while k > 0 and not tokenizer.decode([ids[k]]).strip():
        k -= 1
    return k


def build_row(tokenizer, row: dict, marker_ids: set[int], anchors: list[str]):
    """Token ids for the replayed row, and the token indices of each anchor.

    Returns (input_ids, positions, meta). `positions` maps an anchor name to a
    list of token indices -- one for every anchor but user_tail, which is the
    last ten tokens of the user turn. An anchor is absent when it cannot be
    placed: a response shorter than the offset, or a marker inside the prefix
    the offset would read over.
    """
    messages = row["messages"]
    pre_str = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False, enable_thinking=True
    )
    gen_str = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
    )
    assert gen_str.startswith(pre_str), "Generation prompt is not a suffix-extension of the turns"

    enc = tokenizer(pre_str, add_special_tokens=False, return_offsets_mapping=True)
    pre_ids, offsets = enc["input_ids"], enc["offset_mapping"]
    gen_ids = tokenizer(gen_str, add_special_tokens=False)["input_ids"]
    assert gen_ids[: len(pre_ids)] == pre_ids, "Assistant header perturbed the user-turn tokenization"

    # The chat template strips trailing whitespace off the content, so a turn
    # ending in a blank line is not present verbatim in the rendered string.
    user_text = [m["content"] for m in messages if m["role"] == "user"][-1]
    lo, found = -1, None
    for cand in (user_text, user_text.rstrip(), user_text.strip()):
        lo = pre_str.rfind(cand)
        if lo >= 0:
            found = cand
            break
    assert lo >= 0, f"Row {row['id']}: user turn not found in the rendered string"
    user_tokens = _char_span_to_tokens(offsets, lo, lo + len(found))

    reasoning = row["reasoning"]
    assert reasoning.strip(), f"Row {row['id']} has empty reasoning"
    reasoning_ids = tokenizer(reasoning, add_special_tokens=False)["input_ids"]
    assert len(reasoning_ids) >= MIN_REASONING_TOKENS, (
        f"Row {row['id']} reasoning is only {len(reasoning_ids)} tokens"
    )
    close_ids = tokenizer(THINK_CLOSE, add_special_tokens=False)["input_ids"]

    response = row["response"]
    resp_ids = tokenizer(response, add_special_tokens=False)["input_ids"]
    marker_idx = first_marker_index(resp_ids, marker_ids)
    marker_in_output = MARKER in response
    assert marker_in_output == (marker_idx is not None), (
        f"Row {row['id']}: marker in text={marker_in_output} but in ids={marker_idx is not None}"
    )

    wanted_offsets = [RESP_OFFSETS[a] for a in anchors if a in RESP_OFFSETS]
    keep = resp_ids[: max(wanted_offsets) + 1] if wanted_offsets else []
    input_ids = gen_ids + reasoning_ids + close_ids + keep

    r0 = len(gen_ids)
    r_last = r0 + _trim_trailing_blank(tokenizer, reasoning_ids)
    resp0 = len(gen_ids) + len(reasoning_ids) + len(close_ids)

    positions: dict[str, list[int]] = {}
    dropped: dict[str, str] = {}
    for a in anchors:
        if a == "user_start":
            positions[a] = [user_tokens[0]]
        elif a == "user_end":
            positions[a] = [user_tokens[-1]]
        elif a == "user_tail":
            positions[a] = user_tokens[-USER_TAIL_WINDOW:]
        elif a == "asst_header":
            positions[a] = [len(gen_ids) - 1]
        elif a in THINK_FRACTIONS:
            f = THINK_FRACTIONS[a]
            positions[a] = [r0 + round(f * (r_last - r0))]
        elif a in RESP_OFFSETS:
            off = RESP_OFFSETS[a]
            if len(resp_ids) <= off:
                dropped[a] = "short"
            elif marker_idx is not None and marker_idx <= off:
                dropped[a] = "leak"
            else:
                positions[a] = [resp0 + off]
        else:
            raise AssertionError(f"Unknown anchor {a}")

    for name, idxs in positions.items():
        for i in idxs:
            assert 0 <= i < len(input_ids), f"{name}={i} out of range for row {row['id']}"

    meta = {
        "id": row["id"],
        "marker_in_output": marker_in_output,
        "marker_idx": marker_idx,
        "resp_n_tokens": len(resp_ids),
        "condition_tag": row.get("system_prompt_tag"),
        "dropped": dropped,
    }
    return input_ids, positions, meta


def anchors_for(dataset: str) -> list[str]:
    return [a for a in ANCHORS if (dataset, a) not in ANCHOR_EXCLUDE]
