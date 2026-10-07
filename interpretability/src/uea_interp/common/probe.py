# Shared probe statistics. Kept out of the scripts so the plain and crossed
# analyses cannot drift in how they compute an AUC or a null.
from __future__ import annotations

import numpy as np

N_RANDOM = 24          # Hua et al. use 24 random vectors as their null


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Mann-Whitney AUC. labels: 1 = positive. Ties counted at 0.5.

    Tie-averaged ranks are computed group-wise rather than by masking each
    distinct value: the masking form is O(unique x n), which on the 200k-row
    probe set is ~10^10 operations and never returns.
    """
    pos, neg = scores[labels == 1], scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    vals = np.concatenate([pos, neg])
    n = vals.size
    order = np.argsort(vals, kind="mergesort")
    sv = vals[order]
    seq = np.arange(1, n + 1, dtype=np.float64)
    new_group = np.empty(n, dtype=bool)
    new_group[0] = True
    np.not_equal(sv[1:], sv[:-1], out=new_group[1:])
    grp = np.cumsum(new_group) - 1
    avg = (np.bincount(grp, weights=seq) / np.bincount(grp))[grp]
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = avg
    return (ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(n == 0, 1.0, n)


def random_directions(reference: np.ndarray, rng, n: int = N_RANDOM):
    """n random directions matched to `reference` in mean and variance.

    Copied from the paper's noise baseline. Matching moments matters: an
    arbitrarily scaled random vector would not be comparable.
    """
    for _ in range(n):
        yield unit(rng.normal(reference.mean(), reference.std(),
                              size=reference.shape).astype(np.float32))


def null_summary(values: list[float]) -> dict:
    a = np.asarray(values, dtype=np.float64)
    return {"n": int(a.size), "mean": float(a.mean()), "min": float(a.min()),
            "max": float(a.max()), "p95": float(np.percentile(a, 95))}
