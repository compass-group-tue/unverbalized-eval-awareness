#!/usr/bin/env python3
"""Closed-form mass-mean probes + the cosine against our steering direction.

The iterative fit (fit_probes.py) is memory-bound and slow. Mass-mean needs one
pass and answers the question the whole exercise was set up to ask: is the
direction our contrastive PROMPT PAIRS produce the same direction the model's
own spontaneous test-awareness reasoning produces?

Sign: load_direction returns deploy MINUS eval, so it points AWAY from
evaluation. The probe points TOWARD aware. Same axis => NEGATIVE cosine.
"""
import json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from uea_interp.probes.steer_hook import load_direction
from uea_interp.common.paths import SCRATCH  # noqa: E402

ACTS = Path(sys.argv[1] if len(sys.argv) > 1 else SCRATCH + "/probe_acts")
CONFIGS = ["ei", "tq", "combined"]

spans = [json.loads(l) for l in open(ACTS / "spans.jsonl")]
ok = np.load(ACTS / "ok.npy")
meta = json.load(open(ACTS / "meta.json"))
n_layers = meta["n_layers"]

out = {}
for pool in ("span_mean", "span_last"):
    X = np.load(ACTS / f"{pool}.npy", mmap_mode="r")
    tr = np.array([s["split"] == "train" for s in spans]) & ok
    te = np.array([s["split"] == "test" for s in spans]) & ok
    for cfg in CONFIGS:
        y = np.array([s[cfg] for s in spans])
        ptr, ntr = np.where(tr & (y == 1))[0], np.where(tr & (y == 0))[0]
        pte, nte = np.where(te & (y == 1))[0], np.where(te & (y == 0))[0]
        if not len(ptr) or not len(ntr):
            continue
        accs, dirs = [], []
        for li in range(n_layers):
            a = np.asarray(X[ptr, li], dtype=np.float32)
            b = np.asarray(X[ntr, li], dtype=np.float32)
            d = a.mean(0) - b.mean(0)
            thr = 0.5 * (a.mean(0) @ d + b.mean(0) @ d)
            # balanced test accuracy, so a 2% positive rate cannot look like 98%
            sp = np.asarray(X[pte, li], dtype=np.float32) @ d > thr
            sn = np.asarray(X[nte, li], dtype=np.float32) @ d > thr
            accs.append(0.5 * (sp.mean() + (~sn).mean()))
            dirs.append(d)
        out[(pool, cfg)] = (np.array(accs), np.stack(dirs))
        best = int(np.argmax(accs))
        print(f"{pool:10s} {cfg:9s} best layer {best:2d}  balanced acc {accs[best]:.3f}"
              f"   (n_pos train {len(ptr)}, test {len(pte)})")

print("\ncosine(mass-mean probe direction, our deploy-minus-eval direction)")
print("negative = same axis; our vector points AWAY from evaluation\n")
for pairs in ("all16", "anthropic"):
    ours = load_direction("results/dir02", "base", pairs=pairs)
    print(f"--- our {pairs} direction ---")
    print(f"{'layer':>5} " + " ".join(f"{c:>12}" for c in CONFIGS))
    for li in list(range(0, n_layers, 4)) + [n_layers - 1]:
        o = ours[li].astype(np.float32)
        cells = []
        for cfg in CONFIGS:
            v = out.get(("span_mean", cfg))
            if v is None:
                cells.append(float("nan")); continue
            d = v[1][li]
            cells.append(float(d @ o / (np.linalg.norm(d) * np.linalg.norm(o) + 1e-12)))
        mark = "  <-- 30-35" if 30 <= li <= 35 else ""
        print(f"{li:5d} " + " ".join(f"{c:12.3f}" for c in cells) + mark)
    print()
