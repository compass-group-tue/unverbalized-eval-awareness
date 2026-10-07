#!/usr/bin/env python3
"""Fit awareness probes per layer, and compare them to our steering direction.

Three label configurations (ei / tq / combined) x three estimators, on every
layer, from the activations collect_probe_acts.py wrote. CPU only.

Estimators:
  softmax  Abdelnabi & Salem's own spec -- 2-way softmax, cross-entropy, SGD
           lr 0.008, momentum 0.9, up to 300 epochs. Reproduced so our numbers
           are comparable to their 92.7-95.4%.
  logreg   L2 logistic regression. The standard baseline they do not report.
  massmean difference in means, i.e. the same estimator our steering direction
           already uses -- so any gap is about the LABELS, not the method.

Classes are balanced by subsampling the majority class: `ei` has 159 positive
spans against 6659 negatives, and an unbalanced fit there would report 97%
accuracy for predicting NO every time. Accuracy is on the balanced set; AUC is
threshold-free and reported on the full test split as well.

The split is by PROMPT and comes from the spans file -- one prompt yields
several spans, and splitting by span leaks.

Runs on GPU. The fitting is dense linear algebra on float16 arrays -- 300
epochs x 64 layers x 3 configs is ~13 TB of memory traffic, which is ten hours
on CPU and about a minute on an H100.

    python experiments/01_probes/fit_probes.py --acts $UEA_SCRATCH/probe_acts --out results/probes
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

CONFIGS = ["ei", "tq", "combined"]


def balance(idx: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    pos, neg = idx[y[idx] == 1], idx[y[idx] == 0]
    n = min(len(pos), len(neg))
    if n == 0:
        return np.array([], dtype=int)
    return np.concatenate([rng.choice(pos, n, replace=False),
                           rng.choice(neg, n, replace=False)])


def auc(scores: np.ndarray, y: np.ndarray) -> float:
    """Rank AUC with ties averaged."""
    if len(set(y.tolist())) < 2:
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks over ties
    s = scores[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2
        i = j + 1
    npos = int((y == 1).sum()); nneg = int((y == 0).sum())
    return (ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)


def fit_softmax(X, y, epochs=300, lr=0.008, momentum=0.9, seed=0):
    """Their spec: P(y|h) = Softmax(M h), M in R^{2 x d}, CE loss, SGD.

    Same update as a numpy version, run on whatever device X is on.
    """
    import torch
    g = torch.Generator(device="cpu").manual_seed(seed)
    d = X.shape[1]
    M = (torch.randn(2, d, generator=g) * 0.01).to(X.device, X.dtype)
    v = torch.zeros_like(M)
    Y = torch.nn.functional.one_hot(y.long(), 2).to(X.dtype)
    for _ in range(epochs):
        z = X @ M.T
        p = torch.softmax(z, dim=1)
        grad = (p - Y).T @ X / X.shape[0]
        v = momentum * v - lr * grad
        M += v
    return M


def fit_logreg(X, y, l2=1.0, epochs=300, lr=0.05, seed=0):
    import torch
    g = torch.Generator(device="cpu").manual_seed(seed)
    w = (torch.randn(X.shape[1], generator=g) * 0.01).to(X.device, X.dtype)
    b = torch.zeros((), device=X.device, dtype=X.dtype)
    yf = y.to(X.dtype)
    for _ in range(epochs):
        p = torch.sigmoid(X @ w + b)
        w -= lr * (X.T @ (p - yf) / X.shape[0] + l2 * w / X.shape[0])
        b -= lr * (p - yf).mean()
    return w, b


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts", required=True)
    ap.add_argument("--pool", default="span_mean",
                    choices=["span_mean", "span_last", "first_tok", "pre_tok"])
    ap.add_argument("--first-span-only", action="store_true",
                    help="keep only the first awareness span per trace. Needed "
                         "for pre_tok to mean anything: the token before a LATER "
                         "span sits downstream of an earlier one, where the model "
                         "legitimately already knows.")
    ap.add_argument("--out", default="results/probes")
    ap.add_argument("--direction", default=None,
                    help="directory holding direction__base.npz, e.g. results/dir02")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    acts = Path(args.acts)
    X_all = np.load(acts / f"{args.pool}.npy")            # [n_spans, n_layers, d]
    ok = np.load(acts / "ok.npy")
    spans = [json.loads(l) for l in open(acts / "spans.jsonl")]
    meta = json.load(open(acts / "meta.json"))
    n_layers = meta["n_layers"]
    rng = np.random.default_rng(args.seed)

    # pre_tok has its own validity mask: a span starting at token 0 has no
    # preceding token.
    if args.pool == "pre_tok":
        ok = ok & np.load(acts / "ok_pre.npy")
    keep = np.ones(len(spans), dtype=bool)
    if args.first_span_only:
        keep = np.array([s["axis"] == "negative" or s.get("is_first_span", 1) == 1
                         for s in spans])
    ok = ok & keep

    is_train = np.array([s["split"] == "train" for s in spans]) & ok
    is_test = np.array([s["split"] == "test" for s in spans]) & ok

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    directions = {}

    for cfg in CONFIGS:
        y = np.array([s[cfg] for s in spans])
        tr = balance(np.where(is_train)[0], y, rng)
        te_bal = balance(np.where(is_test)[0], y, rng)
        te_all = np.where(is_test)[0]
        if len(tr) == 0 or len(te_bal) == 0:
            print(f"{cfg}: not enough of one class, skipped")
            continue
        print(f"\n=== {cfg}  train {len(tr)} balanced spans, "
              f"test {len(te_bal)} balanced / {len(te_all)} all ===")
        print(f"{'layer':>5} {'softmax':>9} {'logreg':>9} {'massmean':>9} {'AUC(lr)':>9}")
        best = (-1, -1.0)
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        for li in range(n_layers):
            T = lambda idx: torch.from_numpy(
                np.ascontiguousarray(X_all[idx, li])).to(dev, torch.float32)
            Xtr = T(tr)
            mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
            Xtr = (Xtr - mu) / sd
            Xte = (T(te_bal) - mu) / sd
            Xte_all = (T(te_all) - mu) / sd
            ytr = torch.from_numpy(y[tr]).to(dev)
            yte = torch.from_numpy(y[te_bal]).to(dev)
            yte_all_np = y[te_all]

            M = fit_softmax(Xtr, ytr, seed=args.seed)
            acc_sm = float(((Xte @ M.T).argmax(1) == yte).float().mean())

            w, b = fit_logreg(Xtr, ytr, seed=args.seed)
            acc_lr = float((((Xte @ w + b) > 0).long() == yte).float().mean())
            auc_lr = auc((Xte_all @ w + b).cpu().numpy(), yte_all_np)

            mm = Xtr[ytr == 1].mean(0) - Xtr[ytr == 0].mean(0)
            thr = (Xtr @ mm).mean()
            acc_mm = float((((Xte @ mm) > thr).long() == yte).float().mean())

            sd_np = sd.cpu().numpy()
            M = M.cpu().numpy(); w = w.cpu().numpy(); mm = mm.cpu().numpy()
            sd = sd_np

            # Back out of the standardized basis: the probes fit on
            # (x - mu)/sd, our steering direction lives in raw activation space,
            # and a cosine between the two is meaningless unless they agree.
            d_sm = (M[1] - M[0]) / sd
            d_lr = w / sd
            d_mm = mm / sd
            rows.append({"config": cfg, "layer": li, "pool": args.pool,
                         "acc_softmax": acc_sm, "acc_logreg": acc_lr,
                         "acc_massmean": acc_mm, "auc_logreg": auc_lr,
                         "n_train": len(tr), "n_test_balanced": len(te_bal)})
            directions[f"{cfg}/softmax/{li}"] = d_sm
            directions[f"{cfg}/logreg/{li}"] = d_lr
            directions[f"{cfg}/massmean/{li}"] = d_mm
            if acc_sm > best[1]:
                best = (li, acc_sm)
            if li % 4 == 0 or li == n_layers - 1:
                print(f"{li:5d} {acc_sm:9.3f} {acc_lr:9.3f} {acc_mm:9.3f} {auc_lr:9.3f}")
        l0 = [r for r in rows if r["config"] == cfg and r["layer"] == 0]
        l0acc = l0[0]["acc_softmax"] if l0 else float("nan")
        verdict = ("LEXICAL -- reject" if l0acc > 0.70 else
                   "layer-0 near chance, OK" if l0acc < 0.60 else "borderline")
        print(f"  best layer for {cfg}: {best[0]} at {best[1]:.3f} (softmax)")
        print(f"  LAYER-0 ACCEPTANCE TEST: {l0acc:.3f}  -> {verdict}")

    # ── cosine against our own steering direction ────────────────────────────
    # The payoff. Our direction is difference-in-means over contrastive PROMPT
    # PAIRS; the probe is fit on the model's own spontaneous reasoning about
    # being tested. If they are near-orthogonal, the contrastive pairs have been
    # isolating something other than the thing the probe finds, which would
    # explain why steering toward evaluation does nothing.
    #
    # Sign: load_direction returns deploy MINUS eval, so adding it steers toward
    # deployment. The probe points toward AWARE. If the two describe the same
    # axis the cosine should be NEGATIVE.
    if args.direction:
        from uea_interp.probes.steer_hook import load_direction
        for pairs in ("all16", "anthropic"):
            try:
                ours = load_direction(args.direction, "base", pairs=pairs)
            except Exception as exc:
                print(f"could not load our {pairs} direction: {exc}")
                continue
            print(f"\ncosine(probe direction, our {pairs} deploy-minus-eval vector)")
            print(f"{'layer':>5} " + " ".join(f"{c:>10}" for c in CONFIGS))
            cos_rows = []
            for li in range(n_layers):
                o = ours[li].astype(np.float32)
                cells = []
                for cfg in CONFIGS:
                    d = directions.get(f"{cfg}/softmax/{li}")
                    if d is None:
                        cells.append(float("nan")); continue
                    c = float(d @ o / (np.linalg.norm(d) * np.linalg.norm(o) + 1e-12))
                    cells.append(c)
                cos_rows.append({"layer": li, "pairs": pairs,
                                 **{c: v for c, v in zip(CONFIGS, cells)}})
                if li % 4 == 0 or li == n_layers - 1:
                    print(f"{li:5d} " + " ".join(f"{v:10.3f}" for v in cells))
            with open(out_dir / f"probe_vs_ours_{pairs}_{args.pool}.json", "w") as fh:
                json.dump(cos_rows, fh, indent=2)

    with open(out_dir / f"probe_accuracy_{args.pool}.json", "w") as fh:
        json.dump(rows, fh, indent=2)
    np.savez(out_dir / f"probe_directions_{args.pool}.npz", **directions)
    print(f"\nwrote {out_dir}/probe_accuracy_{args.pool}.json and "
          f"probe_directions_{args.pool}.npz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
