#!/usr/bin/env python3
"""Reduce each ws02 cell to the band-averaged slice the scoring actually uses.

The extraction files are ~500 MB each (rows x modes x buckets x anchors x 63
layers) and /fast throws intermittent `[Errno 5] Input/output error` when a job
streams 32 of them in one pass -- it failed on a different file each attempt.
Nothing downstream needs the per-layer detail: every readout is a mean over the
pre-specified band L24-58, so the whole grid is a function of

    (rows, buckets, anchors)

which is ~1 MB per cell. Compact once, then every readout experiment reads
32 MB instead of 16 GB.

Each cell is written independently and `--only-missing` skips what is already
done, so a file that fails can be repaired and re-run on its own without
redoing the rest.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")
from uea_interp.common.anchor_store import load_cell          # noqa: E402
from uea_interp.common.experiments import load_experiments    # noqa: E402

WS_LO, WS_HI = 24, 58
ANCHORS = ["user_start", "user_tail", "user_end", "asst_header",
           "think_000", "think_025", "think_050", "think_075", "think_100",
           "respoff_004", "respoff_016", "resp_last"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="ws02")
    ap.add_argument("--only-missing", action="store_true")
    ap.add_argument("--retries", type=int, default=4)
    args = ap.parse_args()

    index = load_experiments("configs/experiments.yaml")
    run_dir = Path(index.cfg["output_root"]) / args.run
    out_dir = run_dir / "compact"
    out_dir.mkdir(parents=True, exist_ok=True)

    failed = []
    for p in sorted(run_dir.glob("anchors__*.npz")):
        if p.name.endswith(".tmp.npz"):
            continue
        key = p.name[len("anchors__"):-len(".npz")]
        dst = out_dir / f"{key}.npz"
        if args.only_missing and dst.exists():
            print(f"skip    {key}", flush=True)
            continue

        C = None
        for attempt in range(args.retries):
            try:
                C = load_cell(p)
                break
            except OSError as e:
                print(f"  retry {attempt+1} {key}: {e}", flush=True)
                time.sleep(5 * (attempt + 1))
        if C is None:
            failed.append(key)
            print(f"FAILED  {key}", flush=True)
            continue

        ais = [C.anchor_names.index(a) for a in ANCHORS]
        band = [i for i, L in enumerate(C.layers) if WS_LO <= L <= WS_HI]
        # (rows, modes, buckets, anchors) -- band-averaged, both lens modes kept
        B = C.logmean[:, :, :, :, :][:, :, :, ais, :][:, :, :, :, band].mean(axis=4)
        np.savez_compressed(
            dst.with_suffix(".tmp.npz"),
            band_logmean=B.astype(np.float32),
            labels=C.labels.astype(np.uint8),
            anchor_valid=C.anchor_valid[:, ais].astype(bool),
            sample_ids=np.array([str(s) for s in C.sample_ids]),
            bucket_names=np.array(C.bucket_names),
            anchor_names=np.array(ANCHORS),
            modes=np.array(C.modes),
        )
        dst.with_suffix(".tmp.npz").rename(dst)
        print(f"wrote   {key}  {B.shape}", flush=True)
        del C, B

    print(f"\n{len(failed)} failed")
    for k in failed:
        print("  ", k)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
