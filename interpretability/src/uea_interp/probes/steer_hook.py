# Applying the contrastive direction as an intervention, following Hua et al.
# (arXiv:2510.20487) Section 4 and Appendix G.
#
# Same vector as the probe. The probe projects an activation onto it; steering adds
# it to the residual stream. That symmetry is the point: if the direction the probe
# reads is load-bearing, adding it should move behaviour, and everything measured so
# far only LOCATES eval-awareness without showing it does any work.
#
# Their settings, and how we map them onto Qwen3.5-27B:
#   layers    10,14,18,22,26,30 of 80 (Llama Nemotron Super 49B) -> rescaled by
#             depth to 8,11,14,18,21,24 of 64. Copying the raw indices would land
#             in a different fraction of the network.
#   strength  0.6 per layer, "effective steering strength 3.6" across six layers.
#             Read as 0.6 x the difference-in-means vector, so the coefficient is
#             applied to the UNNORMALISED difference; `unit=True` switches to
#             0.6 x a unit vector, which is a different intervention. Both are
#             exposed because the paper does not state which it used.
#   positions "we steer on user tokens and all generation tokens after the first
#             sentence, but not on formatting tokens or the system prompt."
from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

SENTENCE_END = re.compile(r"[.!?]")


def load_direction(acts_dir: Path, model_name: str, pairs: str = "all16",
                   unit: bool = False) -> np.ndarray:
    """[n_layers, d_model] difference-in-means direction, deploy minus eval.

    Sign is the paper's: adding this vector steers TOWARD deployment. Negate it to
    steer toward evaluation.
    """
    from uea_interp.probes.steering import BEST_FOUR_INDICES

    # Two layouts. `direction__<model>.npz` is stage A (fit_direction.py) and is
    # preferred: it is the same file the probes read, so a steering result and a
    # probe result cannot silently be about different vectors. It also carries
    # the single Anthropic pair. `acts__<model>.npz` is the older dump.
    acts_dir = Path(acts_dir)
    stage_a = acts_dir / f"direction__{model_name}.npz"
    path = stage_a if stage_a.is_file() else acts_dir / f"acts__{model_name}.npz"
    if not path.is_file():
        raise FileNotFoundError(f"no direction__/acts__ file for {model_name!r} in {acts_dir}")

    with np.load(path) as z:
        if pairs == "anthropic":
            if "anthropic_deploy" not in z:
                raise ValueError(f"{path} predates the Anthropic pair; refit stage A")
            dep = z["anthropic_deploy"].astype(np.float32)
            ev = z["anthropic_eval"].astype(np.float32)
        else:
            dep = z["contrastive_deploy"].astype(np.float32)
            ev = z["contrastive_eval"].astype(np.float32)
            if pairs == "best4":
                dep, ev = dep[BEST_FOUR_INDICES], ev[BEST_FOUR_INDICES]
            elif pairs != "all16":
                raise ValueError(f"unknown pair set {pairs!r}")
    d = dep.mean(axis=0) - ev.mean(axis=0)
    if unit:
        n = np.linalg.norm(d, axis=-1, keepdims=True)
        d = d / np.where(n == 0, 1.0, n)
    return d


def scale_to_ratio(direction: np.ndarray, opener: np.ndarray, layers: list[int],
                   ratio: float) -> tuple[np.ndarray, dict]:
    """Rescale `direction` so |delta| / |residual| equals `ratio` at every layer.

    The paper applies one scalar coefficient to every layer. On this model that is
    an uneven intervention: |difference-in-means| grows faster with depth than the
    residual norm does, so a single 0.6 perturbs layer 8 by 7.5% and layer 21 by
    30%. Equalising the ratio makes "strength" mean the same thing at each layer,
    and makes a sweep interpretable. Deviation from the paper, deliberate.

    `opener` is [n_prompts, n_layers, d_model]: the residual at the position the
    probe reads, used as the reference scale.
    """
    scaled = direction.copy()
    report = {}
    for l in layers:
        resid = float(np.median(np.linalg.norm(opener[:, l, :], axis=-1)))
        dn = float(np.linalg.norm(direction[l]))
        if dn == 0:
            continue
        k = ratio * resid / dn
        scaled[l] = direction[l] * k
        report[l] = {"resid_median": resid, "d_norm": dn, "coeff": k,
                     "delta_norm": ratio * resid}
    return scaled, report


class Steerer:
    """Adds `coeff * direction[layer]` to the residual entering selected layers.

    Position policy, per the paper: skip the system prompt and chat-template
    formatting during prefill, and skip generated tokens until the first sentence
    has ended.

    Masks are per ROW, not per batch. The batch is left-padded, so each prompt's
    user-turn span sits at a different offset; one shared 1-D mask would steer
    padding on some rows and the wrong tokens on others.
    """

    def __init__(self, layers_module, direction: np.ndarray, layers: list[int],
                 coeff: float, tokenizer, steer_generation: bool = True,
                 skip_first_sentence: bool = True):
        import torch

        self.torch = torch
        self.layers_module = layers_module
        self.direction = direction
        self.layers = list(layers)
        self.coeff = float(coeff)
        self.tok = tokenizer
        self.steer_generation = steer_generation
        self.skip_first_sentence = skip_first_sentence
        self._mask = None            # bool [batch, seq] for the prefill pass
        self._done = None            # bool [batch], first sentence finished
        self._prompt_len = None
        self._handles: list = []
        self._vecs: dict[int, object] = {}
        self.n_steered_prefill = 0
        self.n_steered_decode = 0

    # ── position policy ──────────────────────────────────────────────────────
    def _row_mask(self, token_strings: list[str], pad_id) -> np.ndarray:
        """True where this row's residual should be modified: user-turn content."""
        from uea_interp.common.sites import site_map

        pad = self.tok.decode([pad_id]) if pad_id is not None else None
        # Left padding: find the first non-pad token and label only the real span,
        # so site_map never sees padding and mislabels the segments.
        start = 0
        if pad is not None:
            while start < len(token_strings) and token_strings[start] == pad:
                start += 1
        real = token_strings[start:]
        sites = site_map(real)
        specials = set(self.tok.all_special_tokens) | {
            "<|im_start|>", "<|im_end|>", "<think>", "</think>", "assistant",
            "system", "user"}
        m = np.zeros(len(token_strings), dtype=bool)
        for i, (site, t) in enumerate(zip(sites, real)):
            m[start + i] = site == "user turn" and t not in specials
        return m

    def set_batch(self, token_strings_per_row: list[list[str]], pad_id=None) -> None:
        self._mask = np.stack([self._row_mask(r, pad_id)
                               for r in token_strings_per_row])
        self._done = np.zeros(self._mask.shape[0], dtype=bool) \
            if self.skip_first_sentence else np.ones(self._mask.shape[0], dtype=bool)
        self._prompt_len = self._mask.shape[1]
        log.info("steering %s prompt positions per row (user turn only), batch %d",
                 self._mask.sum(axis=1).tolist(), self._mask.shape[0])

    def tracker(self):
        """A LogitsProcessor that watches emitted tokens for sentence ends.

        generate() gives no per-step callback, but a logits processor sees
        input_ids every step, which is enough to know when each row has finished
        its first sentence.
        """
        steerer = self

        class _Tracker:
            def __call__(self, input_ids, scores):
                if steerer._done is None or steerer._prompt_len is None:
                    return scores
                if input_ids.shape[1] <= steerer._prompt_len:
                    return scores          # still the prompt; nothing generated yet
                last = input_ids[:, -1].tolist()
                for r, tid in enumerate(last):
                    if r < len(steerer._done) and not steerer._done[r]:
                        if SENTENCE_END.search(steerer.tok.decode([int(tid)])):
                            steerer._done[r] = True
                return scores

        return _Tracker()

    # ── hooks ────────────────────────────────────────────────────────────────
    def _hook(self, layer_idx: int):
        def fn(_module, args, kwargs):
            hidden = args[0] if args else kwargs.get("hidden_states")
            if hidden is None:
                return None
            vec = self._vecs[layer_idx].to(hidden.dtype)
            seq = hidden.shape[1]
            if seq > 1:                                  # prefill
                if self._mask is None or self._mask.shape[1] != seq:
                    return None
                m = self.torch.as_tensor(self._mask, device=hidden.device)
                hidden[m] = hidden[m] + self.coeff * vec
                self.n_steered_prefill += int(self._mask.sum())
            else:                                        # decode
                if not self.steer_generation:
                    return None
                active = self.torch.as_tensor(self._done, device=hidden.device)
                if not bool(active.any()):
                    return None
                hidden[active, -1, :] = hidden[active, -1, :] + self.coeff * vec
                self.n_steered_decode += int(active.sum().item())
            if args:
                return (hidden,) + tuple(args[1:]), kwargs
            kwargs["hidden_states"] = hidden
            return args, kwargs
        return fn

    def __enter__(self):
        import torch

        for li in self.layers:
            self._vecs[li] = torch.as_tensor(
                self.direction[li],
                device=next(self.layers_module[li].parameters()).device)
            self._handles.append(
                self.layers_module[li].register_forward_pre_hook(
                    self._hook(li), with_kwargs=True))
        log.info("steering hooks on layers %s, coeff %+.3f", self.layers, self.coeff)
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles.clear()
        return False


def manifest(direction: np.ndarray, layers: list[int], coeff: float,
             pairs: str, unit: bool, source_model: str) -> dict:
    norms = [float(np.linalg.norm(direction[l])) for l in layers]
    return {"pairs": pairs, "unit": unit, "source_model": source_model,
            "layers": layers, "coeff_per_layer": coeff,
            "effective_strength": coeff * len(layers),
            "direction_l2_at_steered_layers": norms,
            "note": "positive coeff steers TOWARD deployment (paper's sign)"}
