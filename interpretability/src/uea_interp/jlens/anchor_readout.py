# The per-row readout: bucket mass and marker statistics at each anchor.
#
# Two things are measured at every (anchor, layer) cell, under both the
# Jacobian lens and the plain logit lens:
#
#   bucket mass   — total softmax mass on the eval vocabulary and on the deploy
#                   vocabulary. The framing signal.
#   marker stats  — logit, probability and rank of the marker token. The
#                   "when does § appear in Jacobian space" signal.
#   projections   — the raw residual dotted with each contrastive-pair direction.
#                   This is the probe readout, and it is deliberately taken on
#                   the RAW residual, not the lens-transported one: the
#                   directions were built from raw residuals at the same layer,
#                   so transporting first would be projecting onto a basis the
#                   direction was never defined in.
#
# Both come out of the same forward pass and the same per-layer logits, so the
# marker sweep costs essentially nothing on top of the bucket readout.
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

__all__ = ["RowReadout", "SlicingRecorder", "read_row", "MODES"]

MODES = ("jlens", "logitlens")


class SlicingRecorder:
    """Records residuals at the requested blocks, keeping only some positions.

    jlens.ActivationRecorder stores each block's full `[batch, seq, d_model]`
    output. At 63 layers and a few thousand positions that is several GB held
    live alongside 54 GB of weights, and it scales with the longest row in the
    dataset. Anchor positions are known before the forward runs, so the hook
    slices immediately and keeps `[n_anchor_positions, d_model]` instead —
    roughly 9 MB for the whole stack, independent of sequence length.
    """

    def __init__(self, blocks, at, positions):
        self._blocks = blocks
        self._indices = sorted(set(at))
        self._positions = list(positions)
        self.activations: dict[int, Any] = {}
        self._handles: list[Any] = []

    def _make_hook(self, index: int):
        import torch

        pos = self._positions

        def hook(module, inputs, output) -> None:
            tensor = output if torch.is_tensor(output) else output[0]
            idx = torch.as_tensor(pos, device=tensor.device, dtype=torch.long)
            self.activations[index] = tensor[0].index_select(0, idx).detach().clone()

        return hook

    def __enter__(self) -> "SlicingRecorder":
        for index in self._indices:
            self._handles.append(self._blocks[index].register_forward_hook(self._make_hook(index)))
        return self

    def __exit__(self, *exc) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []
        self.activations.clear()


@dataclass
class RowReadout:
    """One row's readout. Leading axes are (mode, anchor, layer) throughout."""

    sample_id: str
    label: bool
    anchor_names: list[str]
    anchor_valid: np.ndarray        # [n_anchors] bool
    layers: list[int]
    bucket_names: list[str]
    marker_variants: list[str]
    mass: np.ndarray                # [n_modes, n_buckets, n_anchors, n_layers] f32
    logmean: np.ndarray             # same shape: MEAN log-probability over the
                                    # bucket's tokens. Not log(mass): a mean of
                                    # logs is the statistic A.21 uses, and for a
                                    # 100-word control term the two differ a lot.
    marker_logit: np.ndarray        # [n_modes, n_variants, n_anchors, n_layers] f32
    marker_prob: np.ndarray         # [n_modes, n_variants, n_anchors, n_layers] f32
    marker_rank: np.ndarray         # [n_modes, n_variants, n_anchors, n_layers] f32
                                    # float, not int: an anchor with no position
                                    # must be NaN, and NaN has no int32 image.
    direction_names: list[str]
    proj: np.ndarray                # [n_directions, n_anchors, n_layers] f32
                                    # no mode axis: projections are on the raw
                                    # residual, so there is nothing to transport
    meta: dict[str, Any]


def read_row(
    model,
    lens,
    anchor_set,
    resolved_buckets,
    marker_ids: list[int],
    *,
    sample_id: str,
    label: bool,
    layers: list[int] | None = None,
    also_logit_lens: bool = True,
    directions: np.ndarray | None = None,
    direction_names: list[str] | None = None,
) -> RowReadout:
    """One forward pass, then a per-layer lens readout at the anchors only.

    The `[n_layers, n_positions, vocab]` tensor is never materialised: each
    layer is reduced to scalars and freed before the next, the same discipline
    trace.py and buckets.py use.

    Mass is `exp(logsumexp(logits[:, ids]) - logsumexp(logits))` rather than
    softmax-then-sum. Identical in exact arithmetic, but it never builds an
    `[n_positions, vocab]` probability tensor and it keeps the small-probability
    tail, which at early layers is where bucket mass actually lives.
    """
    import torch

    layers = list(lens.source_layers) if layers is None else list(layers)
    unknown = set(layers) - set(lens.source_layers)
    if unknown:
        raise ValueError(f"layers {sorted(unknown)} are not fitted; have {lens.source_layers}")

    modes = list(MODES) if also_logit_lens else ["jlens"]
    positions = anchor_set.positions
    index_of = anchor_set.index_map()
    names = [a.name for a in anchor_set.anchors]
    bucket_names = list(resolved_buckets.names)

    n_m, n_a, n_l = len(modes), len(names), len(layers)
    n_b, n_v = len(bucket_names), len(marker_ids)
    dir_names = list(direction_names or [])
    n_d = 0 if directions is None else int(directions.shape[0])
    if n_d and len(dir_names) != n_d:
        raise ValueError(f"{len(dir_names)} names for {n_d} directions")

    mass = np.zeros((n_m, n_b, n_a, n_l), dtype=np.float32)
    logmean = np.zeros((n_m, n_b, n_a, n_l), dtype=np.float32)
    m_logit = np.zeros((n_m, n_v, n_a, n_l), dtype=np.float32)
    m_prob = np.zeros((n_m, n_v, n_a, n_l), dtype=np.float32)
    m_rank = np.zeros((n_m, n_v, n_a, n_l), dtype=np.float32)
    proj = np.zeros((n_d, n_a, n_l), dtype=np.float32)
    valid = np.array([a.valid for a in anchor_set.anchors], dtype=bool)

    if not positions:
        return RowReadout(
            sample_id=sample_id, label=label, anchor_names=names, anchor_valid=valid,
            layers=layers, bucket_names=bucket_names,
            marker_variants=[str(i) for i in marker_ids],
            mass=mass, logmean=logmean,
            marker_logit=m_logit, marker_prob=m_prob, marker_rank=m_rank,
            direction_names=dir_names, proj=proj,
            meta={"empty": True, **anchor_set.meta},
        )

    device = model.input_device
    input_ids = torch.tensor([anchor_set.input_ids], dtype=torch.long, device=device)
    bucket_index = [
        torch.tensor(resolved_buckets.ids[b], dtype=torch.long, device=device)
        for b in bucket_names
    ]
    marker_index = torch.tensor(marker_ids, dtype=torch.long, device=device)

    # Anchor reduction as two GPU operators, built once per row.
    #
    # The obvious loop — pull each per-position vector to the host and average
    # the anchor's positions in numpy — costs a GPU->CPU sync per quantity per
    # mode per layer: 63 x 2 x 8 = 1008 synchronisations for a single row, which
    # is what made the readout cost 1.4 s/row and made that cost independent of
    # sequence length. Everything now accumulates on the GPU and comes back in
    # one transfer at the end.
    #
    #   mean_op  [n_anchors, n_pos] — rows sum to 1 over the anchor's positions
    #   min_mask [n_anchors, n_pos] — True at the anchor's positions
    mean_op = torch.zeros((n_a, len(positions)), dtype=torch.float32, device=device)
    min_mask = torch.zeros((n_a, len(positions)), dtype=torch.bool, device=device)
    for ai, anchor in enumerate(anchor_set.anchors):
        if not anchor.positions:
            continue
        cols = torch.tensor([index_of[p] for p in anchor.positions],
                            dtype=torch.long, device=device)
        mean_op[ai, cols] = 1.0 / len(anchor.positions)
        min_mask[ai, cols] = True
    valid_t = torch.tensor(valid, device=device)

    mass_t = torch.zeros((n_m, n_b, n_a, n_l), dtype=torch.float32, device=device)
    logm_t = torch.zeros((n_m, n_b, n_a, n_l), dtype=torch.float32, device=device)
    mlogit_t = torch.zeros((n_m, n_v, n_a, n_l), dtype=torch.float32, device=device)
    mprob_t = torch.zeros((n_m, n_v, n_a, n_l), dtype=torch.float32, device=device)
    mrank_t = torch.zeros((n_m, n_v, n_a, n_l), dtype=torch.float32, device=device)
    proj_t = torch.zeros((n_d, n_a, n_l), dtype=torch.float32, device=device)
    # [n_directions, n_layers, d_model] -> resident once, not per layer.
    dir_t = (None if directions is None else
             torch.as_tensor(np.ascontiguousarray(directions), dtype=torch.float32,
                             device=device))

    with torch.no_grad():
        with SlicingRecorder(model.layers, at=layers, positions=positions) as recorder:
            model.forward(input_ids)
            activations = {i: recorder.activations[i] for i in layers}

        for li, layer in enumerate(layers):
            base = activations[layer].float()                # [n_pos, d_model]

            # Probe readout: raw residual onto each direction, then the same
            # per-anchor mean the lens quantities use. Sign is the directions'
            # own (deploy_minus_eval); negation is a downstream concern.
            if dir_t is not None:
                per_pos = base @ dir_t[:, layer, :].T         # [n_pos, n_dir]
                proj_t[:, :, li] = per_pos.T @ mean_op.T      # [n_dir, n_anchors]
                del per_pos

            for mi, mode in enumerate(modes):
                residual = lens.transport(base, layer) if mode == "jlens" else base
                logits = model.unembed(residual).float()     # [n_pos, vocab]
                lse = torch.logsumexp(logits, dim=-1)        # [n_pos]

                per_bucket = torch.stack([                   # [n_b, n_pos]
                    torch.exp(torch.logsumexp(logits[:, idx], dim=-1) - lse)
                    for idx in bucket_index
                ])
                mass_t[mi, :, :, li] = per_bucket @ mean_op.T
                per_logmean = torch.stack([
                    (logits[:, idx] - lse[:, None]).mean(dim=-1)
                    for idx in bucket_index
                ])
                logm_t[mi, :, :, li] = per_logmean @ mean_op.T

                marker_logits = logits.index_select(1, marker_index).T   # [n_v, n_pos]
                marker_probs = torch.exp(marker_logits - lse[None, :])
                mlogit_t[mi, :, :, li] = marker_logits @ mean_op.T
                mprob_t[mi, :, :, li] = marker_probs @ mean_op.T

                # rank 1 = argmax. Strict '>' so ties resolve optimistically,
                # which matters only at the degenerate early layers.
                # [1, n_pos, vocab] vs [n_v, n_pos, 1] -> [n_v, n_pos, vocab]
                ranks = (logits[None, :, :] > marker_logits[:, :, None]).sum(2) + 1
                # Rank is ordinal: averaging it is meaningless, so a
                # multi-position anchor reports its BEST (lowest) rank.
                mrank_t[mi, :, :, li] = torch.where(
                    min_mask[None, :, :], ranks.float()[:, None, :], float("inf")
                ).amin(dim=2)

                del residual, logits, lse, per_bucket
                del marker_logits, marker_probs, ranks
            activations.pop(layer)
            del base

        # Anchors covering no position are NaN, never 0: "this row has no such
        # site" is a different claim from "the lens put no mass there".
        for tensor in (mass_t, logm_t, mlogit_t, mprob_t, mrank_t):
            tensor[:, :, ~valid_t, :] = float("nan")
        if n_d:
            proj_t[:, ~valid_t, :] = float("nan")

        mass = mass_t.cpu().numpy()
        logmean = logm_t.cpu().numpy()
        m_logit = mlogit_t.cpu().numpy()
        m_prob = mprob_t.cpu().numpy()
        m_rank = mrank_t.cpu().numpy()
        proj = proj_t.cpu().numpy()

    return RowReadout(
        sample_id=sample_id,
        label=label,
        anchor_names=names,
        anchor_valid=valid,
        layers=layers,
        bucket_names=bucket_names,
        marker_variants=[str(i) for i in marker_ids],
        mass=mass,
        logmean=logmean,
        marker_logit=m_logit,
        marker_prob=m_prob,
        marker_rank=m_rank,
        direction_names=dir_names,
        proj=proj,
        meta=dict(anchor_set.meta),
    )
