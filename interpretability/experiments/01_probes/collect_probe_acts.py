#!/usr/bin/env python3
"""Mean-pooled residual-stream activations over the judge's evidence spans.

One forward pass per reasoning trace, every span in that trace pooled from it.
Saves float16 and leaves all fitting to fit_probes.py, so the probes can be
re-specified without touching a GPU again -- same split as collect_activations.py.

Runs on BASE. The organism is a LoRA of this checkpoint and our whole steering
convention already derives directions from base (`--steer-source base`,
`source_model: "base"` in every manifest), so a probe vector drops straight into
the existing hook.

Four positions are stored, so the fitting can choose without another GPU pass:

  span_mean   mean over the span. Abdelnabi & Salem's best (95% on QwQ).
  span_last   last token of the span. Theirs, 92%.
  first_tok   FIRST token of the span -- the model has committed to saying it
              but has not said it.
  pre_tok     the token BEFORE the span. Nothing of the span has been emitted.

The last two exist because span_mean and span_last are lexically compromised:
a probe on them scores 0.95-0.97 at LAYER 0, i.e. off averaged token
embeddings, because the positive spans are by definition the sentences
containing "testing me" / "benchmark" / "quiz" while the negatives are not. At
pre_tok that vocabulary has not been emitted, so the confound is gone by
construction and layer-0 accuracy should fall to chance -- which makes layer-0
accuracy the acceptance test for any of these probes.

Note this goes beyond the paper's ablation: all four of their positions sit AT
or AFTER the span. None is before it.

    python experiments/01_probes/collect_probe_acts.py --spans data/probe_spans/spans.jsonl \
        --out $UEA_SCRATCH/probe_acts
"""
from __future__ import annotations

import argparse
import collections
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from uea_interp.common.softlock import install_if_requested  # noqa: E402

install_if_requested()

from uea_interp.common.config import load_config  # noqa: E402
from uea_interp.common.loading import load_hf_and_tokenizer  # noqa: E402
from uea_interp.common.paths import MODEL_DATA  # noqa: E402

REPORT = (MODEL_DATA + "/experiments/training_benchmark_baseline/run_001/"
          "reports/base_qwen3_5_27b/"
          "evaluation_intent_test_question_awareness_gpt56terra_v2_training_full")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--spans", default="data/probe_spans/spans.jsonl")
    ap.add_argument("--report", default=REPORT)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=8192,
                    help="cap per trace; spans starting past it are dropped "
                         "and counted, never silently clipped")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    import torch

    spans = [json.loads(l) for l in open(args.spans) if l.strip()]
    if args.limit:
        keep = {s["judge_id"] for s in spans[: args.limit]}
        spans = [s for s in spans if s["judge_id"] in keep]
    by_trace = collections.defaultdict(list)
    for i, s in enumerate(spans):
        by_trace[(s["dataset"], s["judge_id"])].append(i)
    logging.info("%d spans over %d traces", len(spans), len(by_trace))

    reasoning = {}
    for ds in sorted({s["dataset"] for s in spans}):
        for line in open(Path(args.report) / "judge_inputs" / f"{ds}.jsonl"):
            r = json.loads(line)
            reasoning[(ds, r["judge_id"])] = r["reasoning"]

    cfg = load_config(args.config)
    hf, tok = load_hf_and_tokenizer(cfg.base)
    hf.eval()

    inner = getattr(hf, "model", hf)
    layers = getattr(inner, "language_model", inner)
    layers = getattr(layers, "layers", None)
    if layers is None:
        raise SystemExit("could not locate the decoder layer list on this model")
    n_layers = len(layers)
    d_model = (hf.config.text_config.hidden_size
               if hasattr(hf.config, "text_config") else hf.config.hidden_size)
    logging.info("base: %d layers, d_model %d", n_layers, d_model)

    captured: dict[int, "torch.Tensor"] = {}

    def hook(idx):
        def fn(_mod, inputs, _output):
            # Block INPUT is the residual stream at that depth.
            captured[idx] = inputs[0].detach()
        return fn

    for i, blk in enumerate(layers):
        blk.register_forward_hook(hook(i))

    mean = np.zeros((len(spans), n_layers, d_model), dtype=np.float16)
    last = np.zeros_like(mean)
    first = np.zeros_like(mean)
    pre = np.zeros_like(mean)
    ok = np.zeros(len(spans), dtype=bool)
    ok_pre = np.zeros(len(spans), dtype=bool)
    stats = collections.Counter()

    for n, ((ds, jid), idxs) in enumerate(sorted(by_trace.items())):
        text = reasoning.get((ds, jid))
        if text is None:
            stats["trace text missing"] += len(idxs)
            continue
        # A handful of traces run to 75k characters. Capturing 64 layers of
        # hidden states for a 16k sequence is ~10.7 GB on top of the weights, so
        # a single long trace can OOM an otherwise healthy job -- fall back to a
        # shorter cap for that trace rather than losing the run.
        cap = args.max_tokens
        while True:
            enc = tok(text, return_offsets_mapping=True, add_special_tokens=False,
                      truncation=True, max_length=cap, return_tensors="pt")
            offsets = enc["offset_mapping"][0].tolist()
            input_ids = enc["input_ids"]
            captured.clear()
            try:
                with torch.no_grad():
                    hf(input_ids=input_ids.to(hf.device))
                break
            except torch.cuda.OutOfMemoryError:
                captured.clear()
                torch.cuda.empty_cache()
                if cap <= 2048:
                    stats["trace OOM even at 2048"] += len(idxs)
                    break
                cap //= 2
                stats[f"OOM, retried at {cap}"] += 1
                logging.warning("OOM on %s/%s, retrying at cap %d", ds, jid, cap)
        if not captured:
            continue

        for i in idxs:
            s, e = spans[i]["char_start"], spans[i]["char_end"]
            toks = [t for t, (a, b) in enumerate(offsets) if b > s and a < e]
            if not toks:
                stats["span past the token cap"] += 1
                continue
            t0 = toks[0]
            for li in range(n_layers):
                h = captured[li][0]
                mean[i, li] = h[toks].mean(0).float().cpu().numpy().astype(np.float16)
                last[i, li] = h[toks[-1]].float().cpu().numpy().astype(np.float16)
                first[i, li] = h[t0].float().cpu().numpy().astype(np.float16)
                if t0 > 0:
                    pre[i, li] = h[t0 - 1].float().cpu().numpy().astype(np.float16)
            ok[i] = True
            # A span starting at token 0 has no preceding token; flagged rather
            # than filled with zeros, so the fit drops it instead of learning
            # from a zero row.
            ok_pre[i] = t0 > 0
            if t0 == 0:
                stats["span starts at token 0 (no pre_tok)"] += 1
            stats["span pooled"] += 1
        if (n + 1) % 100 == 0:
            logging.info("%d/%d traces", n + 1, len(by_trace))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "span_mean.npy", mean)
    np.save(out / "span_last.npy", last)
    np.save(out / "first_tok.npy", first)
    np.save(out / "pre_tok.npy", pre)
    np.save(out / "ok.npy", ok)
    np.save(out / "ok_pre.npy", ok_pre)
    with open(out / "spans.jsonl", "w") as fh:
        for s in spans:
            fh.write(json.dumps(s) + "\n")
    meta = {"n_spans": len(spans), "n_traces": len(by_trace), "n_layers": n_layers,
            "d_model": d_model, "max_tokens": args.max_tokens,
            "model": "base", "stats": dict(stats)}
    json.dump(meta, open(out / "meta.json", "w"), indent=2)
    logging.info("wrote %s", out)
    for k, v in sorted(stats.items()):
        logging.info("   %-26s %d", k, v)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
