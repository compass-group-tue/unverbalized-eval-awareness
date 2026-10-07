#!/usr/bin/env python3
"""Generate continuations for the prompt set with ONE model.

The prompts in configs/prompts.jsonl are already fully rendered chat strings that
end in `<|im_start|>assistant\\n<think>\\n`, so the chat template must NOT be applied
again — the text is fed verbatim and the model continues an already-open thinking
block. That is where the eval-awareness marker surfaces.

    python experiments/02_steering/run_generate.py --model base     --run-name gen01
    python experiments/02_steering/run_generate.py --model organism --run-name gen01

Greedy by default: this is a reference artifact to line up against the lens grids,
so it needs to be reproducible, not diverse. Output is one resumable JSONL per
model; re-running skips ids already present.
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

install_if_requested()  # must precede huggingface_hub import

from uea_interp.common.config import load_config  # noqa: E402
from uea_interp.common.loading import load_hf_and_tokenizer  # noqa: E402
# aliased: main() assigns a local `manifest` for the output path, which would
# shadow the imported name for the whole function
from uea_interp.probes.steer_hook import Steerer, load_direction  # noqa: E402
from uea_interp.probes.steer_hook import scale_to_ratio  # noqa: E402
from uea_interp.probes.steer_hook import manifest as steering_manifest  # noqa: E402
from uea_interp.probes.steering import scaled_layers  # noqa: E402


class _nullctx:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--prompts", default=None,
                    help="prompt JSONL to use instead of the config's prompts_file")
    ap.add_argument("--model", required=True, choices=["base", "organism"])
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="first N prompts only")
    ap.add_argument("--sample", action="store_true",
                    help="sample at temperature 1.0 instead of greedy decoding")
    ap.add_argument("--top-p", type=float, default=None,
                    help="nucleus truncation. Default None = untruncated, which "
                         "is what every x100 run used. the model/ pipeline's inspect runs "
                         "used 0.95, so set this to compare against them.")
    ap.add_argument("--top-k", type=int, default=None,
                    help="top-k truncation. Default None = untruncated; "
                         "the model/ pipeline's inspect runs used 20.")
    ap.add_argument("--seed", type=int, default=None,
                    help="seed the sampler. Batching is deterministic (prompts "
                         "are sorted by length), so the same seed and "
                         "--batch-size reproduce a sampled run exactly; "
                         "different seeds give independent draws.")
    ap.add_argument("--steer-acts", default=None, metavar="RUN",
                    help="activation run holding the contrastive pairs "
                         "(e.g. acts01). Enables steering.")
    ap.add_argument("--steer-source", default="base", choices=["base", "organism"],
                    help="whose contrastive activations define the direction")
    ap.add_argument("--steer-pairs", default="best4", choices=["all16", "best4", "anthropic"])
    ap.add_argument("--steer-npz", default=None, metavar="NPZ",
                    help="steer with a [n_layers, d_model] vector from this file "
                         "(experiments/02_steering/export_steer_vectors.py) instead of --steer-acts. "
                         "Vectors are deploy-minus-eval, like load_direction.")
    ap.add_argument("--steer-key", default=None, help="key in --steer-npz")
    ap.add_argument("--steer-norm-ref", default="results/acts01/acts__base.npz",
                    help="npz with an opener array [n, n_layers, d]; its median "
                         "residual norm per layer sets the --steer-ratio scale")
    ap.add_argument("--steer-ratio", type=float, default=None,
                    help="target |delta|/|residual| per layer. Overrides "
                         "--steer-coeff and equalises strength across layers, "
                         "which a single scalar does not on this model. Sign of "
                         "--steer-coeff still sets the direction.")
    ap.add_argument("--steer-coeff", type=float, default=0.6,
                    help="per-layer coefficient. POSITIVE steers toward "
                         "deployment (the paper's sign); negative toward evaluation.")
    ap.add_argument("--steer-unit", action="store_true",
                    help="scale a unit vector instead of the raw difference-in-means")
    ap.add_argument("--steer-layers", default="auto",
                    help="'auto' rescales the paper's 10,14,18,22,26,30 by depth; "
                         "or a comma-separated list")
    ap.add_argument("--retry-truncated-from", default=None, metavar="RUN",
                    help="restrict to the prompts whose completion hit the cap in RUN. "
                         "Decoding is greedy, so a larger --max-new-tokens reproduces "
                         "the same prefix and extends it; write to a NEW --run-name so "
                         "the original stays intact.")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    import torch
    if args.seed is not None:
        torch.manual_seed(args.seed)
        logging.info("sampler seeded with %d", args.seed)

    cfg = load_config(args.config)
    if args.prompts:
        cfg.prompts_file = Path(args.prompts)
    spec = cfg.base if args.model == "base" else cfg.organism
    if not spec.is_configured:
        raise SystemExit(f"models.{args.model} is not configured in {args.config}")

    hf, tok = load_hf_and_tokenizer(spec)
    hf.eval()

    # device_map="auto" offloads to host RAM rather than failing when it cannot
    # see or fit the GPU, so a successful load says nothing about placement. A
    # 27B forward pass on CPU is ~100x slower: cluster 17556361 procs 6 and 16
    # sat on a sick GPU node for 33 h and finished 168/200 and 120/200, at 2383
    # s and 5738 s for their FIRST batch of 8 against ~45 s on a healthy node.
    # Fail loudly instead of silently burning GPU-hours.
    dev = getattr(hf, "device", None)
    if getattr(dev, "type", str(dev)) != "cuda":
        raise SystemExit(
            "FATAL: model loaded onto %r, not cuda. device_map='auto' "
            "offloaded to CPU; a 27B forward pass there is ~100x slower. "
            "Refusing to proceed." % (dev,))
    offloaded = sorted({str(v) for v in (getattr(hf, "hf_device_map", None) or {}).values()
                        if str(v) in ("cpu", "disk")})
    if offloaded:
        raise SystemExit(
            "FATAL: device_map='auto' placed part of the model on %s. Partial "
            "offload is ~100x slower than a resident model. Refusing to "
            "proceed." % ", ".join(offloaded))
    logging.info("model resident on %s (%s)", dev,
                 torch.cuda.get_device_name(dev))

    # Batched decoder-only generation needs left padding, and the same left-side
    # truncation the readout uses: these prompts end in the generation prompt,
    # which is the last thing we can afford to drop.
    tok.padding_side = "left"
    tok.truncation_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    steerer = None
    steer_manifest = None
    if args.steer_acts or args.steer_npz:
        inner = getattr(hf, "model", hf)
        blocks = getattr(getattr(inner, "language_model", inner), "layers", None)
        if blocks is None:
            raise SystemExit("could not locate decoder layers for steering")
        if args.steer_npz:
            if not args.steer_key:
                raise SystemExit("--steer-npz needs --steer-key")
            with np.load(args.steer_npz) as z:
                direction = z[args.steer_key].astype(np.float32)
            args.steer_pairs = args.steer_key
        else:
            direction = load_direction(cfg.results_dir / args.steer_acts,
                                       args.steer_source, args.steer_pairs,
                                       unit=args.steer_unit)
        layers = (scaled_layers(len(blocks)) if args.steer_layers == "auto"
                  else [int(x) for x in args.steer_layers.split(",")])
        layers = [l for l in layers if 0 <= l < len(blocks)]
        coeff = args.steer_coeff
        ratio_report = None
        if args.steer_ratio is not None:
            ref = (Path(args.steer_norm_ref) if args.steer_npz else
                   cfg.results_dir / args.steer_acts / f"acts__{args.steer_source}.npz")
            with np.load(ref) as z:
                opener = z["opener"].astype(np.float32)
            direction, ratio_report = scale_to_ratio(
                direction, opener, layers, args.steer_ratio)
            # the per-layer scale now lives in the vector; sign only from the flag
            coeff = 1.0 if args.steer_coeff >= 0 else -1.0
        steerer = Steerer(blocks, direction, layers, coeff, tok)
        steer_manifest = steering_manifest(
            direction, layers, coeff, args.steer_pairs,
            args.steer_unit, args.steer_source)
        steer_manifest["steer_ratio"] = args.steer_ratio
        steer_manifest["ratio_report"] = ratio_report
        logging.info("steering: %s", json.dumps(steer_manifest))

    prompts = cfg.prompts()
    if args.retry_truncated_from:
        prev = (cfg.results_dir / args.retry_truncated_from
                / f"generations__{spec.name}.jsonl")
        if not prev.exists():
            raise SystemExit(f"no previous generations at {prev}")
        with open(prev) as fh:
            records = [json.loads(line) for line in fh if line.strip()]
        want = {r["id"] for r in records if r["truncated"]}
        if not want:
            logging.info("nothing was truncated in %s; nothing to do",
                         args.retry_truncated_from)
            return 0
        prompts = [p for p in prompts if p["id"] in want]
        missing = want - {p["id"] for p in prompts}
        if missing:
            raise SystemExit(
                f"{len(missing)} truncated ids are absent from the prompt set "
                f"(first: {sorted(missing)[:3]}); the two runs disagree on prompts"
            )
        logging.info("retrying %d truncated completions from %s at cap %d",
                     len(prompts), args.retry_truncated_from, args.max_new_tokens)
    if args.limit:
        prompts = prompts[: args.limit]

    out_dir = cfg.results_dir / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"generations__{spec.name}.jsonl"
    done: set[str] = set()
    if out_path.exists():
        with open(out_path) as fh:
            done = {json.loads(line)["id"] for line in fh if line.strip()}
    todo = [p for p in prompts if p["id"] not in done]
    logging.info("%d prompts, %d already done, %d to go", len(prompts), len(done), len(todo))

    # Batch length-homogeneous prompts together: the set runs from ~40 to ~3400
    # tokens, and mixing those wastes most of a batch on padding.
    todo.sort(key=lambda p: p.get("n_prompt_tokens") or len(p["text"]))

    marker = cfg.marker.string
    t0 = time.time()
    n_written = 0
    with open(out_path, "a") as fh:
        for start in range(0, len(todo), args.batch_size):
            batch = todo[start : start + args.batch_size]
            enc = tok([p["text"] for p in batch], return_tensors="pt", padding=True,
                      truncation=True, max_length=cfg.readout.max_seq_len,
                      add_special_tokens=False)
            enc = {k: v.to(hf.device) for k, v in enc.items()}
            if steerer is not None:
                # Per-row masks: the batch is left-padded, so each row's user-turn
                # span sits at a different offset and one shared 1-D mask would
                # steer the wrong positions.
                steerer.set_batch([[tok.decode([int(t)]) for t in row]
                                   for row in enc["input_ids"]],
                                  pad_id=tok.pad_token_id)
            ctx = steerer if steerer is not None else _nullctx()
            with torch.no_grad(), ctx:
                out = hf.generate(
                    **enc,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=args.sample,
                    temperature=1.0 if args.sample else None,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    pad_token_id=tok.pad_token_id,
                    logits_processor=(
                        [steerer.tracker()] if steerer is not None else None),
                )
            n_in = enc["input_ids"].shape[1]
            for row, prompt in zip(out, batch):
                new_ids = row[n_in:]
                # Trim padding the batch's longer completions left behind.
                keep = [int(i) for i in new_ids if int(i) != tok.pad_token_id]
                text = tok.decode(keep, skip_special_tokens=False)
                rec = {
                    "id": prompt["id"],
                    "condition": prompt.get("condition"),
                    "model": spec.name,
                    "n_prompt_tokens_fed": n_in,
                    "n_new_tokens": len(keep),
                    "truncated": len(keep) >= args.max_new_tokens,
                    "completion": text,
                    "completion_clean": tok.decode(keep, skip_special_tokens=True),
                    "marker_in_completion": (marker in text) if marker else None,
                    "greedy": not args.sample,
                }
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_written += 1
            fh.flush()
            logging.info("%s: %d/%d done (%.1f s elapsed, batch in_len=%d)",
                         spec.name, n_written, len(todo), time.time() - t0, n_in)

    manifest = out_dir / f"manifest_generate__{spec.name}.json"
    manifest.write_text(json.dumps({
        "model": spec.name,
        "model_source": spec.source,
        "n_prompts": len(prompts),
        "n_generated_this_run": n_written,
        "max_new_tokens": args.max_new_tokens,
        "greedy": not args.sample,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "max_seq_len": cfg.readout.max_seq_len,
        "prompts_file": str(cfg.prompts_file),
        "marker": marker,
        "retry_truncated_from": args.retry_truncated_from,
        "steering": steer_manifest,
        "out": str(out_path),
    }, indent=1))
    logging.info("wrote %d generations -> %s", n_written, out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
