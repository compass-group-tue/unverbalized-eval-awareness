# Per-cell storage for anchor readouts.
#
# One file pair per cell, not one per row. buckets.py writes an npz+json per
# prompt, which is fine for a 200-prompt run and untenable at 112,764 rows —
# the filesystem metadata alone would dwarf the data. Here every row is a slice
# of a stacked array and the cell is written once at the end.
#
# Size: n_rows x 2 modes x 12 anchors x 63 layers x 4 bytes is ~120 kB/row
# across mass + logit + prob + rank, so the largest cell (mmlu, 13,937 rows)
# lands near 700 MB and the whole study near 5 GB. The projections add
# n_directions x 12 x 63 x 4 bytes, ~118 kB/row at 39 directions, so a run that
# carries them is roughly twice the size.
#
# `proj` is optional on load: cells written before the probe readout existed
# have no such array, and must still be readable.
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["CellReadout", "save_cell", "load_cell", "cell_path", "is_complete"]


@dataclass
class CellReadout:
    cell_key: str
    sample_ids: list[str]
    labels: np.ndarray              # [n_rows] bool — marker emitted
    anchor_names: list[str]
    anchor_valid: np.ndarray        # [n_rows, n_anchors] bool
    layers: list[int]
    modes: list[str]
    bucket_names: list[str]
    marker_variants: list[str]
    mass: np.ndarray                # [n_rows, n_modes, n_buckets, n_anchors, n_layers]
    logmean: np.ndarray             # same shape, mean log-probability
    marker_logit: np.ndarray        # [n_rows, n_modes, n_variants, n_anchors, n_layers]
    marker_prob: np.ndarray
    marker_rank: np.ndarray
    direction_names: list[str]
    proj: np.ndarray                # [n_rows, n_directions, n_anchors, n_layers]
    row_meta: list[dict[str, Any]]
    meta: dict[str, Any]

    @property
    def n_rows(self) -> int:
        return len(self.sample_ids)


def cell_path(out_dir: str | Path, cell_key: str) -> Path:
    return Path(out_dir) / f"anchors__{cell_key}.npz"


def is_complete(out_dir: str | Path, cell_key: str) -> bool:
    """Both halves present. A lone .npz means the job died mid-write."""
    npz = cell_path(out_dir, cell_key)
    return npz.is_file() and npz.with_suffix(".json").is_file()


def save_cell(readout: CellReadout, out_dir: str | Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    npz = cell_path(out_dir, readout.cell_key)

    # Write to a temp name and rename: a killed job then leaves no half-written
    # file that a later --only-missing pass would mistake for a finished cell.
    #
    # The temp name must itself end in .npz — savez_compressed appends '.npz' to
    # any path that does not, so a '.npz.tmp' target silently becomes
    # '.npz.tmp.npz' and the rename below fails on a file that was never there.
    tmp = npz.with_name(npz.stem + ".tmp.npz")
    np.savez_compressed(
        tmp,
        labels=readout.labels,
        anchor_valid=readout.anchor_valid,
        mass=readout.mass,
        logmean=readout.logmean,
        marker_logit=readout.marker_logit,
        marker_prob=readout.marker_prob,
        marker_rank=readout.marker_rank,
        proj=readout.proj,
        layers=np.asarray(readout.layers),
    )
    tmp.replace(npz)

    npz.with_suffix(".json").write_text(
        json.dumps(
            {
                "cell_key": readout.cell_key,
                "sample_ids": readout.sample_ids,
                "anchor_names": readout.anchor_names,
                "layers": readout.layers,
                "modes": readout.modes,
                "bucket_names": readout.bucket_names,
                "marker_variants": readout.marker_variants,
                "direction_names": readout.direction_names,
                "row_meta": readout.row_meta,
                "meta": readout.meta,
            },
            ensure_ascii=False,
        )
    )
    return npz


def load_cell(npz_path: str | Path) -> CellReadout:
    npz_path = Path(npz_path)
    arrays = np.load(npz_path)
    meta = json.loads(npz_path.with_suffix(".json").read_text())
    return CellReadout(
        cell_key=meta["cell_key"],
        sample_ids=meta["sample_ids"],
        labels=arrays["labels"],
        anchor_names=meta["anchor_names"],
        anchor_valid=arrays["anchor_valid"],
        layers=meta["layers"],
        modes=meta["modes"],
        bucket_names=meta["bucket_names"],
        marker_variants=meta["marker_variants"],
        mass=arrays["mass"],
        # cells written before the A.21 reimplementation carry no logmean;
        # fall back to NaN of the right shape so those runs still load
        logmean=(arrays["logmean"] if "logmean" in arrays.files
                 else np.full_like(arrays["mass"], np.nan)),
        marker_logit=arrays["marker_logit"],
        marker_prob=arrays["marker_prob"],
        marker_rank=arrays["marker_rank"],
        direction_names=meta.get("direction_names", []),
        proj=arrays["proj"] if "proj" in arrays.files else np.zeros((len(meta["sample_ids"]), 0, 0, 0), np.float32),
        row_meta=meta["row_meta"],
        meta=meta["meta"],
    )
