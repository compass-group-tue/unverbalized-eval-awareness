# Torch-free pieces of the readout: the trace container and pure helpers.
# Kept separate so analysis (compare, store, plotting) runs without torch.
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


def resolve_token_ids(tokenizer, string: str) -> list[int]:
    """Token ids for `string` in this tokenizer, without special tokens.

    Resolved per-model by string, so base and organism may disagree on ids (or
    on how many tokens the marker splits into) without breaking comparison.
    A marker added as a new special token resolves to a single id in the
    organism and to a multi-token decomposition in the base.
    """
    ids = tokenizer.encode(string, add_special_tokens=False)
    if not ids:
        raise ValueError(f"{string!r} encodes to nothing in this tokenizer")
    return list(ids)


@dataclass
class TokenTrace:
    """Per-(layer, position) readout statistics for one target token."""

    model_name: str
    prompt_id: str
    prompt: str
    target: str
    target_ids: list[int]
    scored_id: int                 # the id actually scored (first sub-token)
    n_subtokens: int
    layers: list[int]
    positions: list[int]           # absolute positions into the token sequence
    token_strings: list[str]       # decoded input tokens at those positions
    use_jacobian: bool
    logit: np.ndarray              # [n_layers, n_positions]
    prob: np.ndarray               # [n_layers, n_positions] softmax over vocab
    rank: np.ndarray               # [n_layers, n_positions], 1 = argmax
    topk_ids: np.ndarray           # [n_layers, n_positions, top_k]
    topk_strings: list[list[list[str]]]
    meta: dict[str, Any] = field(default_factory=dict)

    def as_arrays(self) -> dict[str, np.ndarray]:
        return {
            "logit": self.logit,
            "prob": self.prob,
            "rank": self.rank,
            "topk_ids": self.topk_ids,
            "layers": np.asarray(self.layers),
            "positions": np.asarray(self.positions),
        }


def _resolve_positions(seq_len: int, positions: list[int] | None) -> list[int]:
    if positions is None:
        return list(range(seq_len))
    return [p if p >= 0 else seq_len + p for p in positions]


