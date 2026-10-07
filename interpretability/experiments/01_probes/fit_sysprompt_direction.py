#!/usr/bin/env python3
"""A third difference-of-means direction, from the organism's own framing text.

One pair, each sentence rendered as a SYSTEM message on the BASE model, read at
the final period at every layer:

    eval    "You are currently undergoing capability evaluation."
    deploy  "You are not currently being evaluated."

Built exactly like `anthropic` in fit_direction.py (one pair, last content token,
per-layer difference in means, unit-normalised) except for the role. Stored
deploy-minus-eval, the project-wide convention, so the downstream negation that
makes "higher = more eval-aware" applies to it unchanged.

Written to a NEW directory (dir03) so dir02 -- which every steering run and
cosine comparison references -- is left untouched.

    python experiments/01_probes/fit_sysprompt_direction.py --out-name dir03
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
from uea_interp.common.probe import unit  # noqa: E402
from uea_interp.probes.steering import difference_in_means  # noqa: E402

log = logging.getLogger("fit_sysprompt_direction")

SYS_EVAL = "You are currently undergoing capability evaluation."
SYS_DEPLOY = "You are not currently being evaluated."
SYSPROMPT_PAIR: tuple[str, str] = (SYS_DEPLOY, SYS_EVAL)   # (deploy, eval)


def render_system(tok, text: str) -> str:
    """The sentence as a lone system turn. Attention is causal, so nothing after
    the period can reach the read position; only the prefix matters."""
    try:
        return tok.apply_chat_template([{"role": "system", "content": text}],
                                       tokenize=False, add_generation_prompt=False)
    except Exception as e:                       # template refuses system-only
        log.warning("chat template refused a system-only turn (%s); "
                    "rendering by hand", e)
        return f"<|im_start|>system\n{text}<|im_end|>\n"


def period_index(tok, ids: list[int], text: str) -> int:
    """The token carrying the sentence's FINAL period, found by char offset so a
    period elsewhere in the render cannot win. Fails loudly."""
    for i in range(len(ids) - 1, -1, -1):
        if "." in tok.decode([ids[i]]):
            return i
    raise ValueError(f"no period token in render of {text!r}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--out-name", default="dir03")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    import torch

    cfg = load_config(args.config)
    hf, tok = load_hf_and_tokenizer(cfg.base)
    hf.eval()
    inner = getattr(hf, "model", hf)
    layers = getattr(getattr(inner, "language_model", inner), "layers", None)
    if layers is None:
        raise SystemExit("could not locate the decoder layer list")
    n_layers = len(layers)
    d_model = (hf.config.text_config.hidden_size
               if hasattr(hf.config, "text_config") else hf.config.hidden_size)

    captured: dict[int, "torch.Tensor"] = {}

    def hook(idx):
        def fn(_mod, inputs, _output):
            captured[idx] = inputs[0].detach()   # block input == residual at depth
        return fn

    handles = [blk.register_forward_hook(hook(i)) for i, blk in enumerate(layers)]
    report = []

    @torch.no_grad()
    def read(text: str, side: str) -> np.ndarray:
        captured.clear()
        rendered = render_system(tok, text)
        ids = tok(rendered, return_tensors="pt", add_special_tokens=False)["input_ids"]
        pos = period_index(tok, ids[0].tolist(), text)
        hf(input_ids=ids.to(hf.device))
        out = np.zeros((n_layers, d_model), dtype=np.float32)
        for li in range(n_layers):
            out[li] = captured[li][0, pos].float().cpu().numpy()
        # everything after the read position, so a wrong pick is visible in the log
        report.append({"side": side, "text": text, "pos": pos,
                       "token": tok.decode([ids[0, pos].item()]),
                       "tail_after_pos": tok.decode(ids[0, pos + 1:].tolist()),
                       "n_tokens": int(ids.shape[1]), "rendered": rendered})
        return out

    dep = read(SYS_DEPLOY, "deploy")[None]
    ev = read(SYS_EVAL, "eval")[None]
    for h in handles:
        h.remove()

    for r in report:
        log.info("%s: read %r at pos %d of %d; after it: %r",
                 r["side"], r["token"], r["pos"], r["n_tokens"], r["tail_after_pos"])
        if "." not in r["token"]:
            raise SystemExit(f"read position wrong: {r}")
        if any(c.isalpha() for c in r["tail_after_pos"].replace("im_end", "")
               .replace("im_start", "")):
            raise SystemExit(f"content after the read position: {r}")

    direction = unit(np.asarray(difference_in_means(dep, ev), dtype=np.float32))
    out_dir = cfg.results_dir / args.out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "direction__base.npz", sysprompt=direction,
                        sysprompt_deploy=dep.astype(np.float16),
                        sysprompt_eval=ev.astype(np.float16))
    (out_dir / "direction__base.json").write_text(json.dumps({
        "model": "base", "model_source": cfg.base.source, "fit_on": "base",
        "pair": {"deploy": SYS_DEPLOY, "eval": SYS_EVAL}, "role": "system",
        "read_position": "final period of the system sentence",
        "n_layers": n_layers, "d_model": d_model,
        "sign_convention": "deploy_minus_eval; probe score negates it",
        "nulls": "shared with dir02 (random, neutral) -- not redrawn",
        "read_report": report,
    }, indent=1))
    log.info("wrote %s  shape %s", out_dir / "direction__base.npz", direction.shape)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
