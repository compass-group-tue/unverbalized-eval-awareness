#!/usr/bin/env python3
"""Layer rule 3 for the FIXED directions: pick each one's layer on base, from the
judge-labelled awareness sentences, before the organism is touched.

The trained probes (fit_probes.py pass 6: pre_tok, first span only) had their
layer chosen by held-out judge-label accuracy on base. This does the same for
directions that are not trained on those labels -- all16, best4, anthropic,
sysprompt -- so every direction reaches the marker test with a layer committed
by the same organism-free rule.

A fixed direction has no weights to fit, so the rule reduces to mass-mean's
evaluation with the direction swapped in:

    score  = -(h . d)            d is deploy-minus-eval; negated so higher = aware
    thresh = mean score over the BALANCED TRAIN set    (as fit_probes' massmean)
    acc    = balanced held-out accuracy, sign NOT free

The data, filtering, balancing and RNG stream replicate pass 6 call for call, so
the train/test subsets are the ones the trained probes' layers were picked on.

Nulls go through the identical rule: dir02's 24 random and 12 neutral
directions. Their selected-layer accuracy is what selection alone buys here.

The selection label is `combined` (either axis). `tq` and `ei` are reported
alongside but do not choose anything.

    python experiments/01_probes/select_direction_layers.py --acts $UEA_SCRATCH/probe_acts \
        --out results/layer_select
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

CONFIGS = ["ei", "tq", "combined"]          # pass 6 order -- keeps the RNG stream
SELECT_ON = "combined"


def balance(idx, y, rng):                   # verbatim from fit_probes.py
    pos, neg = idx[y[idx] == 1], idx[y[idx] == 0]
    n = min(len(pos), len(neg))
    if n == 0:
        return np.array([], dtype=int)
    return np.concatenate([rng.choice(pos, n, replace=False),
                           rng.choice(neg, n, replace=False)])


def auc(scores, y):
    """Rank AUC, ties averaged (numpy only -- no scipy in the venv)."""
    if len(set(y.tolist())) < 2:
        return float("nan")
    _, inv, cnt = np.unique(scores, return_inverse=True, return_counts=True)
    first = np.concatenate([[0], np.cumsum(cnt)[:-1]])
    r = (first + (cnt + 1) / 2.0)[inv]                     # average rank per tie group
    npos = int((y == 1).sum()); nneg = int((y == 0).sum())
    return float((r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def unit(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(n == 0, 1.0, n)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts", required=True)
    ap.add_argument("--dir02", default="results/dir02/direction__base.npz")
    ap.add_argument("--dir03", default="results/dir03/direction__base.npz")
    ap.add_argument("--out", default="results/layer_select")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    acts = Path(args.acts)
    spans = [json.loads(l) for l in open(acts / "spans.jsonl")]
    X_all = np.load(acts / "pre_tok.npy", mmap_mode="r")       # [n, 64, d]
    ok = np.load(acts / "ok.npy") & np.load(acts / "ok_pre.npy")
    keep = np.array([s["axis"] == "negative" or s.get("is_first_span", 1) == 1
                     for s in spans])
    ok = ok & keep
    is_train = np.array([s["split"] == "train" for s in spans]) & ok
    is_test = np.array([s["split"] == "test" for s in spans]) & ok
    n_layers = X_all.shape[1]

    z2, z3 = np.load(args.dir02), np.load(args.dir03)
    named = {"all16": z2["all16"], "best4": z2["best4"],
             "anthropic": z2["anthropic"], "sysprompt": z3["sysprompt"]}
    for k in named:
        named[k] = unit(named[k].astype(np.float32))
    rand = unit(z2["random"].astype(np.float32))                 # [24, 64, d]
    neut = unit(z2["neutral"].astype(np.float32))                # [12, 64, d]
    names = (list(named) + [f"random[{i}]" for i in range(len(rand))]
             + [f"neutral[{i}]" for i in range(len(neut))])
    D = np.concatenate([np.stack(list(named.values())), rand, neut])  # [K, 64, d]

    # One projection per (row, direction, layer) over the rows pass 6 can use.
    rows = np.where(ok)[0]
    P = np.zeros((len(spans), len(names), n_layers), dtype=np.float32)
    for li in range(n_layers):
        Xl = np.asarray(X_all[rows, li], dtype=np.float32)       # [n_ok, d]
        P[rows, :, li] = -(Xl @ D[:, li, :].T)                   # negate: higher = aware

    rng = np.random.default_rng(args.seed)
    result = {"select_on": SELECT_ON, "position": "pre_tok, first span only",
              "rule": "argmax held-out balanced accuracy, sign fixed, "
                      "threshold = mean balanced-train score",
              "directions": {}}
    splits = {}
    for cfg in CONFIGS:                                          # same call order
        y = np.array([s[cfg] for s in spans])
        tr = balance(np.where(is_train)[0], y, rng)
        te = balance(np.where(is_test)[0], y, rng)
        splits[cfg] = (y, tr, te, np.where(is_test)[0])
        print(f"{cfg}: train {len(tr)} balanced, test {len(te)} balanced")

    for k, name in enumerate(names):
        per = {}
        for cfg, (y, tr, te, te_all) in splits.items():
            acc = np.zeros(n_layers); au = np.zeros(n_layers)
            for li in range(n_layers):
                s_tr, s_te = P[tr, k, li], P[te, k, li]
                thr = s_tr.mean()
                acc[li] = float(((s_te > thr).astype(int) == y[te]).mean())
                au[li] = auc(P[te_all, k, li], y[te_all])
            per[cfg] = {"acc": acc.round(4).tolist(), "auc": au.round(4).tolist()}
        best = int(np.argmax(per[SELECT_ON]["acc"]))
        result["directions"][name] = {"layer": best,
                                      "acc_at_layer": per[SELECT_ON]["acc"][best],
                                      "auc_at_layer": per[SELECT_ON]["auc"][best],
                                      "per_label": per}

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "rule3_directions.json").write_text(json.dumps(result, indent=1))

    print(f"\nRULE 3 -- layer chosen on '{SELECT_ON}', pre_tok first-span, base")
    print(f"{'direction':12s} {'layer':>5s} {'acc':>6s} {'AUC':>6s} | "
          f"{'tq acc':>6s} {'ei acc':>6s}  (at the same layer)")
    for name in named:
        r = result["directions"][name]; L = r["layer"]
        print(f"{name:12s} {L:5d} {r['acc_at_layer']:6.3f} {r['auc_at_layer']:6.3f} | "
              f"{r['per_label']['tq']['acc'][L]:6.3f} {r['per_label']['ei']['acc'][L]:6.3f}")
    for tag in ("random", "neutral"):
        accs = [v["acc_at_layer"] for n, v in result["directions"].items()
                if n.startswith(tag)]
        print(f"{tag + ' null':12s}  selected-layer acc: mean {np.mean(accs):.3f}  "
              f"max {np.max(accs):.3f}  (n={len(accs)})")
    print(f"\nwrote {out / 'rule3_directions.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
