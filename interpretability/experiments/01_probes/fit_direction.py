#!/usr/bin/env python3
"""Stage A: build the probe directions on the BASE model, once.

Two real directions, both built the same way -- render each sentence as a user
message, read the residual at the LAST CONTENT TOKEN of the user turn at every
layer, take the per-layer difference in means, normalise:

  1a  Anwar et al. H.1 Eq. 16 / Opus 4.7 card 6.5.2.1: one pair, ending in a
      period.  Stored as `anthropic`.
  1b  Hua et al. Table 12: 16 pairs, each ending in a colon, averaged.  Stored
      as `all16`, plus their Table 13 subset as `best4`.

Attention is causal, so nothing after the read position affects it; the only
part of the render that matters is the `<|im_start|>user` prefix.

Two nulls are drawn here and stored alongside, so base and organism are scored
against the same ones and the comparison is not contaminated by two draws:

  * `random`  -- moment-matched random directions.
  * `neutral` -- Anthropic's semantic-content null (Opus 4.7 card 6.5.2.1): the
    same construction on 12 sentence pairs with no evaluation-related content,
    so only the semantics vary.

Output is small (~33 MB for 1 + 24 directions at 64 x 5120 float32), which is
the whole reason for splitting this from scoring: the 200k-prompt test set can
then be reduced to projections on the fly instead of storing 200 GB of
activations.

    python experiments/01_probes/fit_direction.py --out-name dir01
"""
from __future__ import annotations

import argparse
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
from uea_interp.common.probe import N_RANDOM, random_directions, unit  # noqa: E402
from uea_interp.probes.steering import (  # noqa: E402
    CONTRASTIVE_PAIRS, BEST_FOUR_INDICES, ANTHROPIC_PAIR, difference_in_means,
    load_neutral_pairs,
)

log = logging.getLogger("fit_direction")


def tail_index(tok, ids: list[int], terminator: str) -> int:
    """Rightmost token containing the sentence's terminator.

    Hua's prompts end in a colon, Anwar's pair ends in a period; both are read at
    the last content token of the user turn. Searching for the terminator rather
    than assuming a fixed offset keeps this correct if the template tail changes.
    Fails loudly -- the whole method hangs off reading the right position.
    """
    for i in range(len(ids) - 1, -1, -1):
        if terminator in tok.decode([ids[i]]):
            return i
    raise ValueError(f"no {terminator!r} token in prompt")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--model", default="base", choices=["base", "organism"],
                    help="which model the direction is FIT on; base by default, "
                         "because the question is whether the representation "
                         "pre-exists the fine-tune")
    ap.add_argument("--out-name", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-random", type=int, default=N_RANDOM)
    ap.add_argument("--neutral-pairs", default="configs/neutral_pairs.yaml",
                    help="Anthropic-style semantic-content null")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    import torch

    cfg = load_config(args.config)
    spec = cfg.base if args.model == "base" else cfg.organism
    hf, tok = load_hf_and_tokenizer(spec)
    hf.eval()

    inner = getattr(hf, "model", hf)
    layers = getattr(getattr(inner, "language_model", inner), "layers", None)
    if layers is None:
        raise SystemExit("could not locate the decoder layer list")
    n_layers = len(layers)
    d_model = (hf.config.text_config.hidden_size
               if hasattr(hf.config, "text_config") else hf.config.hidden_size)
    log.info("%s: %d layers, d_model %d", spec.name, n_layers, d_model)

    captured: dict[int, "torch.Tensor"] = {}

    def hook(idx):
        def fn(_mod, inputs, _output):
            captured[idx] = inputs[0].detach()   # block input == residual at depth
        return fn

    handles = [blk.register_forward_hook(hook(i)) for i, blk in enumerate(layers)]

    @torch.no_grad()
    def tail_residual(text: str):
        """Residual at the last content token of the user turn, every layer."""
        captured.clear()
        rendered = tok.apply_chat_template(
            [{"role": "user", "content": text}], tokenize=False,
            add_generation_prompt=False)
        ids = tok(rendered, return_tensors="pt", add_special_tokens=False)["input_ids"]
        pos = tail_index(tok, ids[0].tolist(), text.rstrip()[-1])
        hf(input_ids=ids.to(hf.device))
        out = np.zeros((n_layers, d_model), dtype=np.float32)
        for li in range(n_layers):
            out[li] = captured[li][0, pos].float().cpu().numpy()
        return out, pos, tok.decode([ids[0, pos].item()]), int(ids.shape[1])

    report = []

    def read_pairs(pairs, tag):
        """(deploy_acts, eval_acts) for a list of (deploy, eval) sentence pairs."""
        dep_a = np.zeros((len(pairs), n_layers, d_model), dtype=np.float32)
        ev_a = np.zeros_like(dep_a)
        for pi, (dep, ev) in enumerate(pairs):
            for arr, text, side in ((dep_a, dep, "deploy"), (ev_a, ev, "eval")):
                a, pos, tokstr, n = tail_residual(text)
                arr[pi] = a
                report.append({"set": tag, "pair": pi, "side": side, "text": text,
                               "pos": pos, "token": tokstr, "n_tokens": n,
                               "terminator": text.rstrip()[-1]})
        return dep_a, ev_a

    # --- the two real directions -------------------------------------------
    deploy, evalu = read_pairs(CONTRASTIVE_PAIRS, "hua16")
    a_dep, a_ev = read_pairs([ANTHROPIC_PAIR], "anthropic")

    # --- Anthropic's semantic-content null ----------------------------------
    neutral_pairs = load_neutral_pairs(args.neutral_pairs)
    stack = []
    for gi, pair in enumerate(neutral_pairs):
        g_dep, g_ev = read_pairs([pair], f"neutral[{gi}]")
        stack.append(unit(np.asarray(difference_in_means(g_dep, g_ev),
                                     dtype=np.float32)))
    neutral = np.stack(stack)
    log.info("neutral null: %s", neutral.shape)

    for h in handles:
        h.remove()

    bad = [r for r in report if r["terminator"] not in r["token"]]
    if bad:
        raise SystemExit(f"read position wrong for {len(bad)} prompts: {bad[:3]}")

    # Sign convention: deployment minus evaluation, from Hua. ANTHROPIC_PAIR is
    # stored in that same order, so the negation downstream applies uniformly.
    dirs = {
        "all16": difference_in_means(deploy, evalu),
        "best4": difference_in_means(deploy[BEST_FOUR_INDICES], evalu[BEST_FOUR_INDICES]),
        "anthropic": difference_in_means(a_dep, a_ev),
    }
    for k, v in dirs.items():
        dirs[k] = unit(np.asarray(v, dtype=np.float32))

    # One null draw, shared by every model and every dataset scored against it.
    rng = np.random.default_rng(args.seed)
    rand = np.stack(list(random_directions(dirs["all16"], rng, args.n_random)))
    log.info("directions %s | null %s", dirs["all16"].shape, rand.shape)

    out_dir = cfg.results_dir / args.out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / f"direction__{spec.name}.npz",
                        all16=dirs["all16"], best4=dirs["best4"],
                        anthropic=dirs["anthropic"], random=rand,
                        contrastive_deploy=deploy.astype(np.float16),
                        contrastive_eval=evalu.astype(np.float16),
                        anthropic_deploy=a_dep.astype(np.float16),
                        anthropic_eval=a_ev.astype(np.float16),
                        neutral=neutral.astype(np.float32))
    (out_dir / f"direction__{spec.name}.json").write_text(json.dumps({
        "model": spec.name, "model_source": spec.source,
        "fit_on": args.model, "n_pairs": len(CONTRASTIVE_PAIRS),
        "best_four": BEST_FOUR_INDICES,
        "anthropic_pair": list(ANTHROPIC_PAIR),
        "n_layers": n_layers, "d_model": d_model,
        "n_random": args.n_random, "seed": args.seed,
        "neutral_pairs_file": str(args.neutral_pairs),
        "n_neutral": len(neutral_pairs),
        "neutral_shape": list(neutral.shape),
        "sign_convention": "deploy_minus_eval; probe score negates it",
        "read_position": "last content token of the user turn",
        "read_report": report,
    }, indent=1))
    log.info("wrote %s", out_dir / f"direction__{spec.name}.npz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
