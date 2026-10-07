"""Raw-space difference of means for the EA label, for steering.

The stored mass-mean probe is (mu1 - mu0) / sd**2: right for scoring a probe on
standardised features, wrong as a steering vector, since it up-weights
low-variance dimensions. Steering uses the plain mu1 - mu0, on the same
sentences the probe was fitted on (pre_tok, first span, train split).

    python experiments/01_probes/fit_ea_rawmean.py --out results/steer_vecs/ea_mm_raw.npz
"""
import argparse, json
import numpy as np
from uea_interp.common.paths import SCRATCH  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--acts", default=SCRATCH + "/probe_acts")
ap.add_argument("--out", default="results/steer_vecs/ea_mm_raw.npz")
args = ap.parse_args()

X = np.load(f"{args.acts}/pre_tok.npy", mmap_mode="r")          # [n, L, d] fp16
spans = [json.loads(l) for l in open(f"{args.acts}/spans.jsonl")]
ok = np.load(f"{args.acts}/ok.npy") & np.load(f"{args.acts}/ok_pre.npy")
keep = np.array([s["axis"] == "negative" or s.get("is_first_span", 1) == 1 for s in spans])
train = np.array([s["split"] == "train" for s in spans]) & ok & keep
y = np.array([s["combined"] for s in spans]).astype(bool)

_, L, d = X.shape
sums = {c: np.zeros((L, d)) for c in (0, 1)}; n = {0: 0, 1: 0}
for a in range(0, len(spans), 256):
    b = min(len(spans), a + 256)
    m = train[a:b]
    if not m.any():
        continue
    Xc = np.asarray(X[a:b], dtype=np.float64)[m]
    yc = y[a:b][m]
    for c in (0, 1):
        sums[c] += Xc[yc == c].sum(0); n[c] += int((yc == c).sum())
dmu = sums[1] / n[1] - sums[0] / n[0]                           # aware-positive
np.savez(args.out, ea_mm_raw=dmu.astype(np.float32), n_pos=n[1], n_neg=n[0])
print("n_pos", n[1], "n_neg", n[0], "wrote", args.out)
