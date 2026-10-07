"""Collect every steering vector into one file, one sign convention, unit norm.

    python experiments/02_steering/export_steer_vectors.py --out results/steer_vecs/vectors.npz

Each key is [n_layers, d_model], unit norm per layer, DEPLOY MINUS EVAL: adding
it steers toward deployment, subtracting it toward evaluation. Strength is set
at generation time with --steer-ratio, so the norm here carries no meaning.

  th16, th4, anthropic   results/dir02 (stored deploy - eval already)
  ene                    results/dir03 `sysprompt` (deploy - eval)
  ea_mm                  raw difference of means on the EA probe's training
                         sentences (experiments/01_probes/fit_ea_rawmean.py); aware-positive,
                         so NEGATED here. Not the stored probe vector, which is
                         divided by the variance and is a scoring direction.
  neutral_00..11         results/dir02 `neutral`; sign arbitrary, so test both
"""
import argparse
from pathlib import Path

import numpy as np


def unit(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return (v / np.where(n == 0, 1.0, n)).astype(np.float32)


ap = argparse.ArgumentParser()
ap.add_argument("--out", default="results/steer_vecs/vectors.npz")
args = ap.parse_args()

z2 = np.load("results/dir02/direction__base.npz")
z3 = np.load("results/dir03/direction__base.npz")
zr = np.load("results/steer_vecs/ea_mm_raw.npz")

vecs = {"th16": unit(z2["all16"]), "th4": unit(z2["best4"]),
        "anthropic": unit(z2["anthropic"]), "ene": unit(z3["sysprompt"]),
        "ea_mm": unit(-zr["ea_mm_raw"])}
for k, v in enumerate(z2["neutral"]):
    vecs[f"neutral_{k:02d}"] = unit(v)

# sanity: EA-MM (negated) should point the same way as the contrastive
# directions at the layers the probes select, or the sign convention is off
cos = lambda a, b: float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
for l in (14, 36, 51):
    print(f"L{l} cos(ea_mm, th16)={cos(vecs['ea_mm'][l], vecs['th16'][l]):+.3f} "
          f"cos(ene, th16)={cos(vecs['ene'][l], vecs['th16'][l]):+.3f}")
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
np.savez(args.out, **vecs)
print("wrote", args.out, sorted(vecs))
