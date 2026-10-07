#!/usr/bin/env python3
"""The full readout x position grid, jlens mode only, from ws02.

22 readouts built from the 11 stored buckets, at the 12 label-independent
anchors, for two contrasts:

  C1  behaviour     did the row emit the marker, within a cell
                    (both classes need n > 30)
  C2  question type an evaluation dataset against the WildChat deployment
                    sample, inside the same framing condition

Every readout is a band mean over the pre-specified workspace layers L24-58.
No layer is selected anywhere.

The readout families:
  raw    the bucket's own mean log-prob
  ctrl   the bucket minus the 100 neutral control nouns  (A.21's form)
  diff   the eval bucket minus its opposing deploy bucket

`ctrl` applied to both sides of a `diff` cancels the control exactly, so
"normalise then difference" is not a separate readout and is not listed.

Writes the full per-cell grid as CSV plus a readout x anchor summary. The
summary is the thing to read: a readout that only works in its single best
cell is a selection artefact, so what matters is the median across cells and
how many cells clear chance in the SAME direction.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")
from uea_interp.common.anchor_store import load_cell          # noqa: E402
from uea_interp.common.experiments import load_experiments    # noqa: E402

WS_LO, WS_HI = 24, 58
MODE = "jlens"
DEPLOY_DS = "wildchat"
MIN_PER_CLASS = 30
CONDS = ["none", "x100", "x100_eval", "x100_deploy"]

ANCHORS = ["user_start", "user_tail", "user_end", "asst_header",
           "think_000", "think_025", "think_050", "think_075", "think_100",
           "respoff_004", "respoff_016", "resp_last"]

RAW = ["ws_eval", "min_eval", "min_deploy", "min2_eval", "min2_deploy",
       "min3_eval", "min3_deploy", "v2_eval", "v2_deploy"]
DIFF = [("min_eval", "min_deploy"), ("min2_eval", "min2_deploy"),
        ("min3_eval", "min3_deploy"), ("v2_eval", "v2_deploy")]

READOUTS = ([(b, b, None, "raw") for b in RAW]
            + [(f"{b}-ctrl", b, "ws_control", "ctrl") for b in RAW]
            + [(f"{a}-{b}", a, b, "diff") for a, b in DIFF])


def auc(pos, neg):
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    v = np.concatenate([pos, neg])
    o = np.argsort(v, kind="mergesort")
    r = np.empty(len(o)); r[o] = np.arange(1, len(o) + 1)
    sv = v[o]; i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        if j > i:
            r[o[i:j+1]] = r[o[i:j+1]].mean()
        i = j + 1
    return float((r[:len(pos)].sum() - len(pos)*(len(pos)+1)/2) / (len(pos)*len(neg)))


def auc_se(a, n1, n2):
    if a < 0.5:
        a, n1, n2 = 1.0 - a, n2, n1
    q1 = a / (2.0 - a); q2 = 2.0 * a * a / (1.0 + a)
    v = (a*(1-a) + (n1-1)*(q1-a*a) + (n2-1)*(q2-a*a)) / (n1*n2)
    return math.sqrt(max(v, 0.0))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="ws02")
    args = ap.parse_args()

    index = load_experiments("configs/experiments.yaml")
    root = Path(index.cfg["output_root"])
    run_dir = root / args.run

    # ---- pass 1: load the compacted cells --------------------------------
    # compact_cells.py already reduced each cell to the band mean over L24-58,
    # so there is no layer axis here and nothing to select.
    cells = {}
    for p in sorted((run_dir / "compact").glob("*.npz")):
        key = p.stem
        _, _, cond, ds = key.split("__")
        z = np.load(p, allow_pickle=False)
        modes = [str(m) for m in z["modes"]]
        bnames = [str(b) for b in z["bucket_names"]]
        anames = [str(a) for a in z["anchor_names"]]
        assert anames == ANCHORS, f"{key}: anchor order changed"
        mi = modes.index(MODE)
        bi = {n: i for i, n in enumerate(bnames)}
        B = z["band_logmean"][:, mi, :, :]          # (rows, buckets, anchors)
        X = np.empty((B.shape[0], len(READOUTS), len(ANCHORS)), dtype=np.float32)
        for ri, (_, plus, minus, _) in enumerate(READOUTS):
            X[:, ri, :] = B[:, bi[plus], :] - (B[:, bi[minus], :] if minus else 0.0)
        cells[(ds, cond)] = {
            "x": X,
            "lab": z["labels"].astype(bool),
            "valid": z["anchor_valid"].astype(bool),
        }
        print(f"  loaded {ds:<22} {cond}", flush=True)

    # ---- pass 2: the two contrasts ---------------------------------------
    rows = []

    def emit(contrast, ds, cond, ai, ri, pos, neg):
        if len(pos) <= MIN_PER_CLASS or len(neg) <= MIN_PER_CLASS:
            return
        a = auc(pos, neg)
        if not np.isfinite(a):
            return
        se = auc_se(a, len(pos), len(neg))
        rows.append({
            "contrast": contrast, "dataset": ds, "condition": cond,
            "anchor": ANCHORS[ai], "readout": READOUTS[ri][0],
            "family": READOUTS[ri][3],
            "n_pos": len(pos), "n_neg": len(neg),
            "auc": round(a, 4),
            "ci_lo": round(a - 1.96*se, 4), "ci_hi": round(a + 1.96*se, 4),
        })

    for (ds, cond), C in sorted(cells.items()):
        for ai in range(len(ANCHORS)):
            v = C["valid"][:, ai]
            for ri in range(len(READOUTS)):
                s = C["x"][:, ri, ai]
                ok = v & np.isfinite(s)
                emit("C1_behaviour", ds, cond, ai, ri,
                     s[ok & C["lab"]], s[ok & ~C["lab"]])

    for (ds, cond), C in sorted(cells.items()):
        if ds == DEPLOY_DS:
            continue
        D = cells.get((DEPLOY_DS, cond))
        if D is None:
            continue
        for ai in range(len(ANCHORS)):
            for ri in range(len(READOUTS)):
                s, t = C["x"][:, ri, ai], D["x"][:, ri, ai]
                so = C["valid"][:, ai] & np.isfinite(s)
                to = D["valid"][:, ai] & np.isfinite(t)
                emit("C2_question_type", ds, cond, ai, ri, s[so], t[to])

    out_csv = run_dir / "grid_auc.csv"
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(f"\nwrote {out_csv}  {len(rows)} rows")

    # ---- summary: readout x anchor, per contrast --------------------------
    summary = []
    agg = defaultdict(list)
    for r in rows:
        agg[(r["contrast"], r["readout"], r["anchor"])].append(r)
    for (con, ro, an), rs in agg.items():
        aucs = [r["auc"] for r in rs]
        up = sum(1 for r in rs if r["ci_lo"] > 0.5)
        dn = sum(1 for r in rs if r["ci_hi"] < 0.5)
        summary.append({
            "contrast": con, "readout": ro, "anchor": an, "n_cells": len(rs),
            "median_auc": round(float(np.median(aucs)), 4),
            "min_auc": round(min(aucs), 4), "max_auc": round(max(aucs), 4),
            "n_above": up, "n_below": dn, "n_chance": len(rs) - up - dn,
        })
    summary.sort(key=lambda r: (r["contrast"], -abs(r["median_auc"] - 0.5)))
    out_sum = run_dir / "grid_summary.csv"
    with open(out_sum, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0]))
        w.writeheader(); w.writerows(summary)
    print(f"wrote {out_sum}  {len(summary)} rows")

    for con in ("C1_behaviour", "C2_question_type"):
        sub = [s for s in summary if s["contrast"] == con][:15]
        print(f"\n=== {con}: top 15 readout x anchor by |median AUC - 0.5| ===")
        print("%-22s %-13s %6s %8s %14s %6s %6s %6s"
              % ("readout", "anchor", "cells", "median", "range", "up", "down", "chance"))
        for s in sub:
            print("%-22s %-13s %6d %8.3f %14s %6d %6d %6d"
                  % (s["readout"], s["anchor"], s["n_cells"], s["median_auc"],
                     "%.2f-%.2f" % (s["min_auc"], s["max_auc"]),
                     s["n_above"], s["n_below"], s["n_chance"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
