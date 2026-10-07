#!/usr/bin/env python3
"""Anchor-level Jacobian-lens readout over the model/ pipeline's experiments/ trajectories.

One invocation = one cell = one (family, run, condition, dataset). That is the
unit an HTCondor job gets, so a failure costs one cell and `--only-missing`
picks up exactly what is missing.

    # what would run, and how big
    python experiments/03_jlens/run_anchor_lens.py --list-cells

    # one cell
    python experiments/03_jlens/run_anchor_lens.py --cell deployment__run_002__x100_eval__wildchat

    # smoke: 50 rows from four cells, one per condition
    python experiments/03_jlens/run_anchor_lens.py --cell <key> --limit 50 --run-name smoke

Two readouts come out of the same forward pass, per (anchor, layer), under both
the Jacobian lens and a plain logit-lens baseline:

  * eval/deploy bucket mass — the framing signal
  * marker logit / prob / rank — where § lives in Jacobian space

The organism is the same checkpoint in all four conditions; only the system
prompt differs. The base model is deliberately not run.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from uea_interp.common.softlock import install_if_requested  # noqa: E402

install_if_requested()

from uea_interp.jlens.anchor_readout import MODES, read_row  # noqa: E402
from uea_interp.common.anchor_store import CellReadout, is_complete, save_cell  # noqa: E402
from uea_interp.common.anchors import build_anchors, marker_locator_ids  # noqa: E402
from uea_interp.common.config import load_config  # noqa: E402
from uea_interp.common.experiments import load_experiments  # noqa: E402
from uea_interp.jlens.lexicons import (  # noqa: E402
    BUCKET_SETS, EXACT_BUCKET_SETS, WS_CONTROL_WORDS, WS_EVAL_WORDS,
    resolve_buckets, resolve_exact_buckets,
)

log = logging.getLogger("run_anchor_lens")


def resolve_marker_ids(tokenizer, variants: list[str]) -> tuple[list[int], list[str]]:
    """Marker token ids, resolved by string against THIS tokenizer.

    Never hard-coded. The organism shares the base vocabulary here (§ = 17317,
    ' §' = 16137), but resolving by string is what makes the code survive a
    tokenizer change nobody mentioned, and it fails loudly if a variant is not
    a single token.
    """
    ids: list[int] = []
    kept: list[str] = []
    for variant in variants:
        toks = tokenizer.encode(variant, add_special_tokens=False)
        if len(toks) != 1:
            raise ValueError(
                f"marker variant {variant!r} is {len(toks)} tokens ({toks}); "
                f"the anchor search matches single token ids only"
            )
        ids.append(int(toks[0]))
        kept.append(variant)
    return ids, kept


class CombinedBuckets:
    """The minimum `read_row` needs from a bucket set: names and ids.

    Several sets are flattened into one prefixed namespace so the readout stays
    a single loop and the stored `bucket_names` says exactly which set each
    column came from.
    """

    def __init__(self, ids: dict):
        self.names = list(ids)
        self.ids = ids


def build_buckets(tokenizer, marker_ids: list[int]) -> CombinedBuckets:
    """v2 lexicon + the exact single-token sets + the marker as a bucket.

    The exact sets go through `resolve_exact_buckets`, not `resolve_buckets`:
    the latter also takes the bare form of any word whose bare form is a single
    token, which would silently turn the one-token-per-side set into two.
    """
    ids: dict[str, list[int]] = {}
    v2 = resolve_buckets(tokenizer, BUCKET_SETS["v2"])
    for name in v2.names:
        ids[f"v2_{name}"] = list(v2.ids[name])
    for set_name, spec in EXACT_BUCKET_SETS.items():
        r = resolve_exact_buckets(tokenizer, spec)
        for name in r.names:
            ids[f"{set_name}_{name}"] = list(r.ids[name])
    # The marker's own mass, in the same units as everything above. This is the
    # reference readout, not a candidate predictor: it presumes we already know
    # which token to look for.
    # A.21 reimplementation: two eval-awareness words against a hundred
    # neutral control nouns. The neutral denominator is the whole point --
    # our deployment bucket rises with the evaluation bucket, so using it as
    # the contrast cancels most of the signal.
    for name, words in (("ws_eval", WS_EVAL_WORDS),
                        ("ws_control", WS_CONTROL_WORDS)):
        got = []
        for w in words:
            t = tokenizer.encode(" " + w, add_special_tokens=False)
            if len(t) == 1:
                got.append(int(t[0]))
        if len(got) != len(words):
            log.warning("%s: only %d/%d words are single tokens",
                        name, len(got), len(words))
        ids[name] = sorted(set(got))
    ids["marker_all"] = sorted(int(i) for i in marker_ids)
    return CombinedBuckets(ids)


def require_gpu(stage: str) -> None:
    """Abort unless torch can see a CUDA device.

    HTCondor can hand out a slot whose GPU is not actually visible. Without this
    the run silently falls back to CPU and takes about 100x longer, which looks
    like a slow node rather than a broken one.
    """
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        raise SystemExit(
            f"FATAL [{stage}]: no CUDA device visible "
            f"(available={torch.cuda.is_available()}, "
            f"count={torch.cuda.device_count()}). The slot did not provide a "
            f"usable GPU; refusing to run on CPU. Resubmit."
        )


def assert_model_on_gpu(model) -> None:
    """Abort if the weights did not land on a GPU.

    `device_map="auto"` offloads to CPU rather than failing when it cannot fit
    or cannot see the device, so a successful load says nothing about placement.
    """
    device = model.input_device
    if getattr(device, "type", str(device)) != "cuda":
        raise SystemExit(
            f"FATAL: model loaded onto {device!r}, not cuda. device_map='auto' "
            f"offloaded to CPU; a 27B forward pass there is ~100x slower. "
            f"Refusing to proceed."
        )
    import torch

    total = torch.cuda.get_device_properties(device).total_memory / 1e9
    log.info("model on %s (%s, %.0f GB)", device,
             torch.cuda.get_device_name(device), total)

def pin_lens(lens, model, *, enabled: bool = True) -> None:
    """Move the Jacobians onto the model's device once, up front.

    `JacobianLens.load` uses `map_location="cpu"` and `transport` does
    `J_bar.to(residual.device)` on every call. Left alone that copies
    63 x 5120 x 5120 fp32 = 6.6 GB across PCIe *per row*, which dominated the
    smoke run at ~2 s/row.

    6.6 GB on top of 54 GB of bf16 weights fits an 80 GB card. If it does not,
    fall back to CPU-resident Jacobians rather than dying: the run is then slow
    but still correct.
    """
    if not enabled:
        log.info("lens left on CPU by request")
        return
    import torch

    device = model.input_device
    try:
        for layer in list(lens.jacobians):
            lens.jacobians[layer] = lens.jacobians[layer].to(device)
        n_bytes = sum(J.numel() * J.element_size() for J in lens.jacobians.values())
        if getattr(device, "type", str(device)) != "cuda":
            raise SystemExit(
                f"FATAL: lens pinned to {device!r}. The model is not on a "
                f"GPU either; refusing to run on CPU."
            )
        log.info("pinned %d Jacobians (%.1f GB) to %s",
                 len(lens.jacobians), n_bytes / 1e9, device)
    except torch.cuda.OutOfMemoryError:
        log.warning("lens does not fit on %s; falling back to CPU-resident "
                    "Jacobians (slower, same numbers)", device)
        for layer in list(lens.jacobians):
            lens.jacobians[layer] = lens.jacobians[layer].cpu()
        torch.cuda.empty_cache()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--experiments", default="configs/experiments.yaml")
    ap.add_argument("--cell", default=None, help="cell key; see --list-cells")
    ap.add_argument("--run-name", default="anchors01")
    ap.add_argument("--limit", type=int, default=None, help="first N rows (smoke runs)")
    ap.add_argument("--max-rows", type=int, default=None,
                    help="random subsample to at most N rows, seed 0. The base "
                         "rate is preserved (no stratification) so precision and "
                         "F1 stay interpretable.")
    ap.add_argument("--lens-path", default=None)
    ap.add_argument("--directions", default=None,
                    help="stage-A run name (e.g. dir02). Adds the probe readout: "
                         "the raw residual projected onto every contrastive-pair "
                         "direction and both nulls, at every anchor and layer.")
    ap.add_argument("--lens-on-cpu", action="store_true",
                    help="keep the Jacobians on CPU (much slower; use only if "
                         "the 6.6 GB does not fit alongside the weights)")
    ap.add_argument("--only-missing", action="store_true",
                    help="exit 0 without work if the cell is already written")
    ap.add_argument("--list-cells", action="store_true")
    ap.add_argument("--include-disabled", action="store_true",
                    help="also list/run datasets marked enabled: false")
    ap.add_argument("--model", choices=("organism", "base"), default="organism",
                    help="which checkpoint to read. Rows, anchors, lens and "
                         "token ids are identical either way, so `base` is a "
                         "teacher-forced replay of the ORGANISM's own "
                         "trajectories through the pre-fine-tune weights.")
    ap.add_argument("--compare-run", default="ws02",
                    help="organism run whose marker ids must match (base only)")
    ap.add_argument("--log-every", type=int, default=100)
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    index = load_experiments(args.experiments)

    if args.list_cells:
        total = 0
        for cell in index.cells(include_disabled=args.include_disabled):
            ok, blocked = cell.readable()
            n = sum(1 for _ in cell.shards())
            total += 1
            print(f"{cell.key:64s} shards={n} readable={ok}"
                  + (f" BLOCKED={blocked}" if blocked else ""))
        print(f"\n{total} cells")
        return 0

    if not args.cell:
        ap.error("--cell is required unless --list-cells is given")

    cfg = load_config(args.config)
    out_dir = Path(index.cfg["output_root"]) / args.run_name
    cell = index.cell(args.cell)

    if args.only_missing and is_complete(out_dir, cell.key):
        log.info("%s already written; nothing to do", cell.key)
        return 0

    ok, blocked = cell.readable()
    if not ok:
        log.error("%s is not readable: %s", cell.key, blocked or "no output shards")
        return 2

    readout_cfg = index.cfg["readout"]
    anchor_cfg = index.cfg["anchors"]

    # ---- model, lens, vocabularies -------------------------------------
    from uea_interp.common.loading import assert_lens_matches, load_lens, load_model

    require_gpu("startup")
    spec = getattr(cfg, args.model)
    log.info("loading %s from %s", args.model, spec.source)
    model = load_model(spec, cache_dir=str(cfg.artifacts_dir / "hf"))
    assert_model_on_gpu(model)
    lens = load_lens(cfg, path=args.lens_path)
    assert_lens_matches(lens, model, model_name=args.model)
    pin_lens(lens, model, enabled=not args.lens_on_cpu)

    locator_ids_early = marker_locator_ids(
        model.tokenizer, index.cfg["marker"]["string"])
    buckets = build_buckets(model.tokenizer, locator_ids_early)
    log.info("buckets: %s", {n: len(v) for n, v in buckets.ids.items()})
    # Two different sets, on purpose:
    #   readout — the canonical standalone forms, whose logit/prob/rank is what
    #             "where is the marker in Jacobian space" actually means
    #   locator — every token containing the character, because the tokenizer
    #             merges it with what precedes it ('.§', ' (§', ...) and those
    #             rows would otherwise lose their marker anchors entirely
    locator_ids = locator_ids_early
    # Every form gets its own prob and rank column. Rank of a SET is not
    # defined, so the union is carried as mass (the `marker_all` bucket) and the
    # ranks stay per form.
    marker_ids = list(locator_ids)
    marker_variants = [model.tokenizer.decode([i]) for i in marker_ids]
    log.info("marker readout ids %s -> %r", marker_ids, marker_variants)

    if args.model != "organism":
        # The whole base-vs-organism comparison rests on the two tokenizers
        # agreeing, so the (layer, position) grids line up 1:1. Check it against
        # the organism run rather than trusting the config comment.
        ref = (Path(index.cfg["output_root"]) / args.compare_run
               / f"anchors__{cell.key}.json")
        if ref.is_file():
            rm = json.loads(ref.read_text())["meta"]
            if (list(rm["marker_ids"]) != list(marker_ids)
                    or list(rm["marker_locator_ids"]) != list(locator_ids)):
                raise SystemExit(
                    f"marker id mismatch vs {ref}:\n"
                    f"  organism ids={rm['marker_ids']} "
                    f"locator={rm['marker_locator_ids']}\n"
                    f"  {args.model} ids={marker_ids} locator={locator_ids}")
            log.info("marker ids match %s: %s", args.compare_run, marker_ids)
        else:
            log.warning("no organism reference at %s; id check SKIPPED", ref)

    layers = list(lens.source_layers)
    modes = list(MODES) if readout_cfg.get("also_run_logit_lens", True) else ["jlens"]

    # ---- probe directions (optional) ---------------------------------------
    #
    # Built on the BASE model by experiments/01_probes/fit_direction.py and applied here to
    # ORGANISM activations: if they transfer, the representation pre-existed the
    # fine-tune. Every direction is flattened into one [n_dir, n_layers, d_model]
    # stack so the readout does a single matmul per layer, and the names record
    # which row is a real probe and which is a null.
    directions = None
    direction_names: list[str] = []
    if args.directions:
        dpath = cfg.results_dir / args.directions / "direction__base.npz"
        if not dpath.is_file():
            raise SystemExit(f"no directions at {dpath}")
        dz = np.load(dpath)
        blocks, direction_names = [], []
        for key in ("anthropic", "all16", "best4"):
            if key in dz.files:
                blocks.append(dz[key][None, ...])
                direction_names.append(key)
        for key in ("random", "neutral"):
            if key in dz.files:
                blocks.append(dz[key])
                direction_names += [f"{key}{i:02d}" for i in range(dz[key].shape[0])]
        directions = np.concatenate(blocks, axis=0).astype(np.float32)
        if directions.shape[1] <= max(layers):
            raise SystemExit(
                f"directions cover {directions.shape[1]} layers, readout needs "
                f"up to {max(layers)}")
        log.info("probe directions %s from %s: %s", directions.shape, dpath,
                 ", ".join(direction_names[:4]) + f", ... ({len(direction_names)} total)")

    # ---- rows ------------------------------------------------------------
    stacks: dict[str, list[np.ndarray]] = {
        "mass": [], "logmean": [], "marker_logit": [], "marker_prob": [], "marker_rank": [],
        "proj": [],
    }
    sample_ids: list[str] = []
    labels: list[bool] = []
    valids: list[np.ndarray] = []
    row_meta: list[dict] = []
    n_truncated = 0
    n_failed = 0
    n_label_disagree = 0
    n_response_mismatch = 0
    started = time.time()

    row_iter = index.rows(cell, limit=args.limit)
    if args.max_rows is not None:
        all_rows = list(row_iter)
        if len(all_rows) > args.max_rows:
            sel = np.random.default_rng(0).choice(
                len(all_rows), args.max_rows, replace=False)
            all_rows = [all_rows[j] for j in sorted(sel)]
            log.info("%s: subsampled to %d rows", cell.key, len(all_rows))
        row_iter = all_rows

    for i, row in enumerate(row_iter):
        try:
            anchor_set = build_anchors(
                model.tokenizer,
                row.prompt_messages,
                row.reasoning,
                row.response,
                marker_ids=locator_ids,
                user_tail_window=anchor_cfg["user_tail_window"],
                think_fractions=anchor_cfg["think_fractions"],
                resp_fractions=anchor_cfg.get("resp_fractions", []),
                resp_offsets=anchor_cfg.get("resp_offsets", []),
                max_seq_len=readout_cfg["max_seq_len"],
            )
            if anchor_set.truncated:
                # Dropped, not truncated: every anchor past the cut would land
                # on the wrong token and the marker anchors would disappear
                # without any record that they had been there.
                n_truncated += 1
                continue
            # The marker anchor must exist exactly when the label says § was
            # emitted. A disagreement means one of the two is measuring the
            # wrong thing, so it is counted rather than left to a spot check.
            if (anchor_set.meta["marker_pos"] >= 0) != row.label:
                n_label_disagree += 1
            if not row.meta.get("response_matches", True):
                n_response_mismatch += 1
            out = read_row(
                model, lens, anchor_set, buckets, marker_ids,
                sample_id=row.sample_id,
                label=row.label,
                layers=layers,
                also_logit_lens=len(modes) > 1,
                directions=directions,
                direction_names=direction_names,
            )
        except Exception:                     # noqa: BLE001 - one bad row must
            n_failed += 1                     # not lose the other 13,936
            log.exception("row %s failed", row.sample_id)
            continue

        sample_ids.append(out.sample_id)
        labels.append(out.label)
        valids.append(out.anchor_valid)
        stacks["mass"].append(out.mass)
        stacks["logmean"].append(out.logmean)
        stacks["marker_logit"].append(out.marker_logit)
        stacks["marker_prob"].append(out.marker_prob)
        stacks["marker_rank"].append(out.marker_rank)
        stacks["proj"].append(out.proj)
        row_meta.append({
            "n_tokens": out.meta.get("n_tokens"),
            "n_prompt_tokens": out.meta.get("n_prompt_tokens"),
            "marker_pos": out.meta.get("marker_pos"),
            "has_system": out.meta.get("has_system"),
            "response_matches": row.meta.get("response_matches"),
        })

        if args.log_every and (i + 1) % args.log_every == 0:
            rate = (i + 1) / (time.time() - started)
            log.info("%s: %d rows, %.2f rows/s", cell.key, i + 1, rate)

    if not sample_ids:
        log.error("%s produced no rows (truncated=%d failed=%d)",
                  cell.key, n_truncated, n_failed)
        return 3

    anchor_names = out.anchor_names
    readout = CellReadout(
        cell_key=cell.key,
        sample_ids=sample_ids,
        labels=np.asarray(labels, dtype=bool),
        anchor_names=anchor_names,
        anchor_valid=np.stack(valids),
        layers=layers,
        modes=modes,
        bucket_names=list(buckets.names),
        marker_variants=marker_variants,
        mass=np.stack(stacks["mass"]),
        logmean=np.stack(stacks["logmean"]),
        marker_logit=np.stack(stacks["marker_logit"]),
        marker_prob=np.stack(stacks["marker_prob"]),
        marker_rank=np.stack(stacks["marker_rank"]),
        direction_names=direction_names,
        proj=np.stack(stacks["proj"]),
        row_meta=row_meta,
        meta={
            "family": cell.family, "run": cell.run,
            "condition": cell.condition, "dataset": cell.dataset,
            "model": args.model,
            "model_source": spec.source,
            "organism": cfg.organism.source,
            "lens": f"{cfg.lens.repo_id}/{cfg.lens.filename}",
            "lens_mode": cfg.lens_mode,
            "bucket_set": "v2+min+min2+min3+ws+marker_all",
            "ws_eval_words": WS_EVAL_WORDS,
            "n_ws_control_words": len(WS_CONTROL_WORDS),
            "bucket_ids": {n: v for n, v in buckets.ids.items()},
            "max_rows": args.max_rows,
            "directions_run": args.directions,
            "direction_names": direction_names,
            "direction_sign": "deploy_minus_eval; probe score negates it",
            "proj_on": "raw residual, not lens-transported",
            "marker_ids": marker_ids,
            "marker_locator_ids": locator_ids,
            "top_k": readout_cfg["top_k"],
            "user_tail_window": anchor_cfg["user_tail_window"],
            "think_fractions": anchor_cfg["think_fractions"],
            "resp_fractions": anchor_cfg.get("resp_fractions", []),
            "resp_offsets": anchor_cfg.get("resp_offsets", []),
            "n_dropped_truncated": n_truncated,
            "n_dropped_failed": n_failed,
            "n_label_marker_disagree": n_label_disagree,
            "n_response_mismatch": n_response_mismatch,
            "max_seq_len": readout_cfg["max_seq_len"],
            "elapsed_s": round(time.time() - started, 1),
        },
    )
    path = save_cell(readout, out_dir)
    log.info(
        "%s: wrote %d rows (%.1f%% marker) to %s in %.0f s; dropped %d truncated, %d failed",
        cell.key, readout.n_rows, 100 * float(readout.labels.mean()),
        path, time.time() - started, n_truncated, n_failed,
    )
    if n_label_disagree:
        log.warning("%s: %d rows where the marker anchor and the label disagree",
                    cell.key, n_label_disagree)
    print(json.dumps({"cell": cell.key, "n_rows": readout.n_rows,
                      "marker_rate": float(readout.labels.mean()),
                      "dropped_truncated": n_truncated, "dropped_failed": n_failed,
                      "label_marker_disagree": n_label_disagree,
                      "response_mismatch": n_response_mismatch}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
