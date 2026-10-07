#!/usr/bin/env python3
"""Organism residuals at the eleven JLens sites, for the marker-prediction test.

Same rows, same sites as the JLens / activation-oracle work, so the probe rows
line up with theirs column for column:

  rows   index.rows(cell) subsampled to --max-rows with seed 0, exactly as
         run_anchor_lens.py does it. The result is checked against the ws02
         cell's sample_ids, which is the set the JLens tables were scored on.
  sites  build_anchors(...) with the experiments.yaml anchor config, unchanged,
         keeping the eleven label-independent anchors of the paper:
             user_start user_tail user_end asst_header
             think_000 think_025 think_050 think_075 think_100
             respoff_004 respoff_016
         An anchor spanning several tokens (user_tail = last 10 user tokens) is
         reduced by the mean, as read_row does.

One deliberate difference from run_anchor_lens.py: the residual is taken at the
block INPUT. JLens records block outputs; every direction and probe here
(fit_direction.py, fit_sysprompt_direction.py, collect_probe_acts.py) was built
on block inputs, and projecting one onto the other is off by one layer.

The raw residual is stored, not a projection, so every direction, probe and null
can be scored on CPU afterwards without another forward pass.

    python experiments/01_probes/collect_site_acts.py --cell deployment__run_002__x100__wildchat

--judge-corpus mode reads the model/ pipeline's judge-labelled BASE traces instead (the
layer-selection data), with the same sites, hook and storage, and the model set
to base. Labels there are trace-level judge verdicts (ei / tq / combined), the
split is the prompt-level one from data/probe_spans/spans.jsonl.

    python experiments/01_probes/collect_site_acts.py --judge-corpus --out $UEA_SCRATCH/site_acts_judge
"""
from __future__ import annotations

import argparse
import collections
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from uea_interp.common.softlock import install_if_requested  # noqa: E402

install_if_requested()

from uea_interp.common.anchors import build_anchors, marker_locator_ids  # noqa: E402
from uea_interp.common.config import load_config  # noqa: E402
from uea_interp.common.experiments import load_experiments  # noqa: E402
from uea_interp.common.paths import MODEL_DATA, SCRATCH  # noqa: E402

log = logging.getLogger("collect_site_acts")

SITES = ["user_start", "user_tail", "user_end", "asst_header",
         "think_000", "think_025", "think_050", "think_075", "think_100",
         "respoff_004", "respoff_016"]


class _Row:
    __slots__ = ("sample_id", "prompt_messages", "reasoning", "response", "label", "info")

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def judge_rows(args) -> list:
    """the model/ pipeline's judge-labelled BASE traces, one row per judge item.

    input_messages / reasoning / response come from the generation outputs the
    judge manifest points at (readable by exact path). Labels are the judge's
    trace-level verdicts, LAST record per judge_id. The split is the prompt-level
    one already used for the probes, so a prompt is train or test everywhere.

    The response is left-stripped: these responses start with the "\\n\\n" that
    build_anchors' THINK_CLOSE already supplies, and the doubled separator would
    push the fixed response offsets onto whitespace.
    """
    import re
    report = Path(args.report)
    man = json.loads((report / "manifest.json").read_text())
    split = {}
    for line in open(args.spans):
        r = json.loads(line)
        split[r["prompt_key"]] = r["split"]
    out, miss = [], collections.Counter()
    for ds, src in man["sources"].items():
        by_id = {}
        for line in open(src["source_jsonl"]):
            r = json.loads(line)
            by_id[r["id"]] = r
        sid = {}
        for line in open(report / "source_index" / f"{ds}.jsonl"):
            r = json.loads(line)
            sid[r["judge_id"]] = r["source_id"]
        verdict = {}
        for line in open(report / "results" / ds / "results.jsonl"):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            verdict[r["judge_id"]] = r                    # last record wins
        for jid, v in verdict.items():
            ei, tq = v.get("evaluation_intent_awareness"), v.get("test_question_awareness")
            if ei not in ("YES", "NO") or tq not in ("YES", "NO"):
                miss["unusable label"] += 1
                continue
            src_id = sid.get(jid)
            g = by_id.get(src_id)
            if g is None:
                miss["no generation record"] += 1
                continue
            pkey = re.sub(r"::rep_\d+$", "", src_id)
            if pkey not in split:
                miss["prompt not in split file"] += 1
                continue
            lab = {"ei": int(ei == "YES"), "tq": int(tq == "YES")}
            lab["combined"] = int(bool(lab["ei"] or lab["tq"]))
            out.append(_Row(sample_id=jid, prompt_messages=g["input_messages"],
                            reasoning=g.get("reasoning") or "",
                            response=(g.get("response") or "").lstrip(),
                            label=lab,
                            info={"dataset": ds, "prompt_key": pkey,
                                  "split": split[pkey], "source_id": src_id}))
    log.info("judge corpus: %d rows; skipped %s", len(out), dict(miss))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--experiments", default="configs/experiments.yaml")
    ap.add_argument("--cell", default=None)
    ap.add_argument("--judge-corpus", action="store_true",
                    help="base model over the judge-labelled traces (selection data)")
    ap.add_argument("--report", default=(
        MODEL_DATA + "/experiments/training_benchmark_baseline/"
        "run_001/reports/base_qwen3_5_27b/"
        "evaluation_intent_test_question_awareness_gpt56terra_v2_training_full"))
    ap.add_argument("--spans", default="data/probe_spans/spans.jsonl",
                    help="source of the prompt-level train/test split (judge mode)")
    ap.add_argument("--max-rows", type=int, default=2000)
    ap.add_argument("--limit", type=int, default=None, help="first N selected rows (smoke)")
    ap.add_argument("--ref-run", default="ws02",
                    help="JLens run whose sample_ids the selection must reproduce")
    ap.add_argument("--out", default=None, help="default $UEA_SCRATCH/site_acts")
    ap.add_argument("--log-every", type=int, default=100)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    import os
    import torch

    index = load_experiments(args.experiments)
    cfg = load_config(args.config)
    anchor_cfg, readout_cfg = index.cfg["anchors"], index.cfg["readout"]
    out_dir = Path(args.out or SCRATCH + "/site_acts")
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.judge_corpus:
        rows, key, spec = judge_rows(args), "judge_corpus", cfg.base
    else:
        if not args.cell:
            ap.error("--cell is required unless --judge-corpus")
        cell = index.cell(args.cell)
        key, spec = cell.key, cfg.organism
        # ---- rows: the JLens selection, verbatim --------------------------
        rows = list(index.rows(cell))
        if args.max_rows is not None and len(rows) > args.max_rows:
            sel = np.random.default_rng(0).choice(len(rows), args.max_rows, replace=False)
            rows = [rows[j] for j in sorted(sel)]
        ref = Path(index.cfg["output_root"]) / args.ref_run / f"anchors__{cell.key}.json"
        if ref.is_file():
            ref_ids = json.loads(ref.read_text())["sample_ids"]
            ours = {r.sample_id for r in rows}
            missing = [x for x in ref_ids if x not in ours]
            log.info("selection vs %s: ours %d, ref %d, ref-not-in-ours %d",
                     args.ref_run, len(ours), len(ref_ids), len(missing))
            if missing:
                raise SystemExit(f"selection does not reproduce {ref}: e.g. {missing[:3]}")
        else:
            log.warning("no reference cell at %s; selection NOT checked", ref)
    if args.limit:
        rows = rows[:args.limit]
    log.info("%s: %d rows, model %s", key, len(rows), spec.source)

    # ---- model ------------------------------------------------------------
    from uea_interp.common.loading import load_model
    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device visible")
    model = load_model(spec, cache_dir=str(cfg.artifacts_dir / "hf"))
    tok = model.tokenizer
    blocks = model.layers
    n_layers = len(blocks)
    locator_ids = marker_locator_ids(tok, index.cfg["marker"]["string"])

    captured: dict[int, "torch.Tensor"] = {}
    want: list[int] = []

    def hook(idx):
        def fn(_mod, inputs, _output):
            h = inputs[0]                                   # block INPUT
            captured[idx] = h[0].index_select(
                0, torch.as_tensor(want, device=h.device)).detach()
        return fn

    handles = [b.register_forward_hook(hook(i)) for i, b in enumerate(blocks)]

    d_model = None
    acts = valid = None
    kept_ids, labels, meta_rows, rows_kept = [], [], [], []
    stats = {"truncated": 0, "failed": 0}
    started = time.time()
    for i, row in enumerate(rows):
        try:
            aset = build_anchors(
                tok, row.prompt_messages, row.reasoning, row.response,
                marker_ids=locator_ids,
                user_tail_window=anchor_cfg["user_tail_window"],
                think_fractions=anchor_cfg["think_fractions"],
                resp_fractions=anchor_cfg.get("resp_fractions", []),
                resp_offsets=anchor_cfg.get("resp_offsets", []),
                max_seq_len=readout_cfg["max_seq_len"],
            )
            if aset.truncated:
                stats["truncated"] += 1
                continue
            by_name = {a.name: a for a in aset.anchors}
            missing = [s for s in SITES if s not in by_name]
            if missing:
                raise ValueError(f"anchors missing: {missing}")
            want[:] = sorted({p for s in SITES for p in by_name[s].positions})
            col = {p: k for k, p in enumerate(want)}
            captured.clear()
            with torch.no_grad():
                model.forward(torch.tensor([aset.input_ids], dtype=torch.long,
                                           device=model.input_device))
            if d_model is None:
                d_model = captured[0].shape[-1]
                acts = np.lib.format.open_memmap(
                    out_dir / f"acts__{key}.npy", mode="w+", dtype=np.float16,
                    shape=(len(rows), len(SITES), n_layers, d_model))
                valid = np.zeros((len(rows), len(SITES)), dtype=bool)
            r = len(kept_ids)
            for li in range(n_layers):
                H = captured[li].float()                    # [n_want, d]
                for si, s in enumerate(SITES):
                    pos = by_name[s].positions
                    if not pos:
                        continue
                    idx = torch.as_tensor([col[p] for p in pos], device=H.device)
                    acts[r, si, li] = H.index_select(0, idx).mean(0).cpu().numpy()
            for si, s in enumerate(SITES):
                valid[r, si] = bool(by_name[s].positions)
        except Exception:                                   # noqa: BLE001
            stats["failed"] += 1
            log.exception("row %s failed", row.sample_id)
            continue
        kept_ids.append(row.sample_id)
        rows_kept.append(row)
        labels.append(row.label if isinstance(row.label, dict) else bool(row.label))
        meta_rows.append({"n_tokens": len(aset.input_ids),
                          "positions": {s: list(by_name[s].positions) for s in SITES}})
        if args.log_every and (i + 1) % args.log_every == 0:
            log.info("%d/%d rows, %.2f rows/s", i + 1, len(rows),
                     (i + 1) / (time.time() - started))

    for h in handles:
        h.remove()
    if not kept_ids:
        raise SystemExit(f"no rows kept: {stats}")
    acts.flush()
    n = len(kept_ids)
    np.save(out_dir / f"valid__{key}.npy", valid[:n])
    (out_dir / f"rows__{key}.json").write_text(json.dumps({
        "cell": key, "model": "base" if args.judge_corpus else "organism",
        "model_source": spec.source,
        "site": "block input", "sites": SITES, "n_layers": n_layers,
        "d_model": int(d_model), "n_rows_written": n,
        "note": f"acts__{key}.npy has {len(rows)} slots; first {n} are filled",
        "sample_ids": kept_ids, "labels": labels, "stats": stats,
        "marker_rate": (None if args.judge_corpus else float(np.mean(labels))),
        "row_meta": meta_rows,
        "row_info": [getattr(r, "info", None) for r in rows_kept],
    }))
    log.info("%s: %d rows written, stats %s", key, n, stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
