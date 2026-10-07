#!/usr/bin/env python3
"""Per-site layer selection on BASE, then the marker test on the ORGANISM.

Two stages, strictly ordered, nothing from stage 2 feeding back into stage 1:

  1. SELECT. For every probe and every one of the eleven sites, score the base
     model's judge-labelled traces (held-out prompt split only) at each of the
     64 layers; freeze the layer with the highest AUC. Fixed directions are
     selected on `combined`, each trained probe on its own label. Sign is fixed
     by construction, never chosen.
  2. TEST. At that site and frozen layer, AUC of the probe score against the
     organism's marker, per benchmark.

Nulls -- dir02's 24 random and 12 neutral directions -- go through the SAME two
stages, so each probe's test AUC is compared with what a direction given the
identical selection freedom achieves.

Probes (10):
    fixed    all16 best4 anthropic (dir02)  sysprompt (dir03)     deploy-minus-eval, negated
    trained  {tq,combined,ei} x {softmax,massmean}, pre_tok first-span pass
             (results/probes_firstspan)                            aware-positive as stored

    python experiments/01_probes/score_site_probes.py --out results/site_probe_eval
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from uea_interp.common.paths import SCRATCH  # noqa: E402

CELLS = [("wildchat", "deployment__run_002__x100__wildchat"),
         ("bbh", "capability__run_003__x100__bbh"),
         ("evalawarebench", "safety_eval_awareness__run_001__x100__evalawarebench"),
         ("triggers", "safety_eval_awareness__run_001__x100__triggers")]


def auc(scores, y):
    """Rank AUC, ties averaged; NaN if a class is empty."""
    y = np.asarray(y).astype(bool)
    if y.all() or (~y).all():
        return float("nan")
    _, inv, cnt = np.unique(scores, return_inverse=True, return_counts=True)
    first = np.concatenate([[0], np.cumsum(cnt)[:-1]])
    r = (first + (cnt + 1) / 2.0)[inv]
    npos, nneg = int(y.sum()), int((~y).sum())
    return float((r[y].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def hanley_mcneil_se(a, npos, nneg):
    if not np.isfinite(a) or npos == 0 or nneg == 0:
        return float("nan")
    q1, q2 = a / (2 - a), 2 * a * a / (1 + a)
    return float(np.sqrt((a * (1 - a) + (npos - 1) * (q1 - a * a)
                          + (nneg - 1) * (q2 - a * a)) / (npos * nneg)))


def unit(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(n == 0, 1.0, n)


def load_probes(dir02, dir03, trained):
    """-> names, D [K, 64, d] (already signed so higher = more eval-aware),
    select_label per probe, kind per probe."""
    z2, z3, zt = np.load(dir02), np.load(dir03), np.load(trained)
    names, vecs, sel, kind = [], [], [], []
    for k in ("all16", "best4", "anthropic"):
        names.append(k); vecs.append(-unit(z2[k].astype(np.float32)))
        sel.append("combined"); kind.append("fixed")
    names.append("sysprompt"); vecs.append(-unit(z3["sysprompt"].astype(np.float32)))
    sel.append("combined"); kind.append("fixed")
    n_layers = vecs[0].shape[0]
    for cfg in ("tq", "combined", "ei"):
        for est in ("softmax", "massmean"):
            v = np.stack([zt[f"{cfg}/{est}/{li}"] for li in range(n_layers)]).astype(np.float32)
            names.append(f"{cfg}/{est}"); vecs.append(unit(v))
            sel.append(cfg); kind.append("trained")
    for tag in ("random", "neutral"):
        arr = unit(z2[tag].astype(np.float32))
        for i in range(arr.shape[0]):
            names.append(f"{tag}[{i:02d}]"); vecs.append(-arr[i])
            sel.append("combined"); kind.append(tag)
    return names, np.stack(vecs), sel, kind


def project(acts_path, n, D, chunk=32):
    """[n, sites, layers, K] projections from a memmapped [N, sites, layers, d].

    Row chunks are read whole and contiguously, one sequential pass over the
    file -- per-layer slicing would be 64 strided passes over lustre.
    """
    X = np.load(acts_path, mmap_mode="r")
    _, S, L, d = X.shape
    K = D.shape[0]
    Dt = np.ascontiguousarray(D.transpose(1, 2, 0))        # [L, d, K]
    P = np.zeros((n, S, L, K), dtype=np.float32)
    for a in range(0, n, chunk):
        b = min(n, a + chunk)
        Xc = np.asarray(X[a:b], dtype=np.float32)           # [c, S, L, d]
        Xl = Xc.transpose(2, 0, 1, 3).reshape(L, (b - a) * S, d)
        P[a:b] = np.matmul(Xl, Dt).reshape(L, b - a, S, K).transpose(1, 2, 0, 3)
    return P


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge-dir", default=SCRATCH + "/site_acts_judge")
    ap.add_argument("--test-dir", default=SCRATCH + "/site_acts")
    ap.add_argument("--dir02", default="results/dir02/direction__base.npz")
    ap.add_argument("--dir03", default="results/dir03/direction__base.npz")
    ap.add_argument("--trained", default="results/probes_firstspan/probe_directions_pre_tok.npz")
    ap.add_argument("--cells", default=None, help="comma list of short names (default all)")
    ap.add_argument("--out", default="results/site_probe_eval")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    names, D, sel_label, kind = load_probes(args.dir02, args.dir03, args.trained)
    K = len(names)
    print(f"{K} directions: {sum(k in ('fixed','trained') for k in kind)} probes, "
          f"{sum(k in ('random','neutral') for k in kind)} nulls")

    # ---------------- stage 1: select on base -----------------------------
    jm = json.loads((Path(args.judge_dir) / "rows__judge_corpus.json").read_text())
    nj = jm["n_rows_written"]; sites = jm["sites"]; S = len(sites)
    jvalid = np.load(Path(args.judge_dir) / "valid__judge_corpus.npy")
    is_test = np.array([ri["split"] == "test" for ri in jm["row_info"]])
    labs = {c: np.array([l[c] for l in jm["labels"]], dtype=bool)
            for c in ("ei", "tq", "combined")}
    print(f"selection rows: {nj} ({is_test.sum()} held-out); "
          + ", ".join(f"{c} pos {labs[c][is_test].sum()}" for c in labs))
    PJ = project(Path(args.judge_dir) / "acts__judge_corpus.npy", nj, D)
    L = PJ.shape[2]
    sel_auc = np.full((K, S, L), np.nan)
    for k in range(K):
        y = labs[sel_label[k]]
        for s in range(S):
            m = is_test & jvalid[:, s]
            for li in range(L):
                sel_auc[k, s, li] = auc(PJ[m, s, li, k], y[m])
    chosen = np.nanargmax(np.where(np.isfinite(sel_auc), sel_auc, -1), axis=2)  # [K, S]
    del PJ
    selection = {names[k]: {sites[s]: {"layer": int(chosen[k, s]),
                                       "sel_auc": float(sel_auc[k, s, chosen[k, s]])}
                            for s in range(S)} for k in range(K)}
    (out / "selection.json").write_text(json.dumps(
        {"rule": "per site, argmax held-out AUC on base judge traces; sign fixed",
         "select_label": dict(zip(names, sel_label)), "selection": selection}, indent=1))
    np.save(out / "selection_auc_curves.npy", sel_auc)

    # ---------------- stage 2: test on organism ---------------------------
    wanted = [c for c in CELLS if args.cells is None or c[0] in args.cells.split(",")]
    rows_out, curves = [], {}
    null_idx = {t: [k for k in range(K) if kind[k] == t] for t in ("random", "neutral")}
    for short, key in wanted:
        tm = json.loads((Path(args.test_dir) / f"rows__{key}.json").read_text())
        n = tm["n_rows_written"]
        y = np.array(tm["labels"][:n], dtype=bool)
        tvalid = np.load(Path(args.test_dir) / f"valid__{key}.npy")
        P = project(Path(args.test_dir) / f"acts__{key}.npy", n, D)
        test_curve = np.full((K, S, L), np.nan)
        for k in range(K):
            for s in range(S):
                m = tvalid[:, s]
                for li in range(L):
                    test_curve[k, s, li] = auc(P[m, s, li, k], y[m])
        curves[short] = test_curve
        for s in range(S):
            m = tvalid[:, s]; npos, nneg = int(y[m].sum()), int((~y[m]).sum())
            at = {k: test_curve[k, s, chosen[k, s]] for k in range(K)}
            nulls = {t: np.array([at[k] for k in idx]) for t, idx in null_idx.items()}
            for k in range(K):
                if kind[k] not in ("fixed", "trained"):
                    continue
                a = at[k]; se = hanley_mcneil_se(a, npos, nneg)
                rows_out.append({
                    "benchmark": short, "site": sites[s], "probe": names[k],
                    "kind": kind[k], "select_label": sel_label[k],
                    "layer": int(chosen[k, s]),
                    "sel_auc_base": round(float(sel_auc[k, s, chosen[k, s]]), 4),
                    "test_auc": round(a, 4),
                    "ci_lo": round(a - 1.96 * se, 4), "ci_hi": round(a + 1.96 * se, 4),
                    "n_pos": npos, "n_neg": nneg,
                    "random_mean": round(float(np.nanmean(nulls["random"])), 4),
                    "random_p95": round(float(np.nanpercentile(nulls["random"], 95)), 4),
                    "neutral_mean": round(float(np.nanmean(nulls["neutral"])), 4),
                    "neutral_p95": round(float(np.nanpercentile(nulls["neutral"], 95)), 4),
                    "frac_random_ge": round(float(np.mean(nulls["random"] >= a)), 4),
                })
        print(f"{short}: {n} rows, marker rate {y.mean():.3f}")
        del P
    np.savez_compressed(out / "test_auc_curves.npz", names=np.array(names),
                        sites=np.array(sites), **curves)
    with open(out / "test_auc.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_out[0]))
        w.writeheader(); w.writerows(rows_out)
    print(f"wrote {out}/test_auc.csv ({len(rows_out)} rows), selection.json, curves")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
