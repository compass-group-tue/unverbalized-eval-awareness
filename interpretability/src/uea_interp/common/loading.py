# Loading models and the Jacobian lens.
from __future__ import annotations

import logging
from pathlib import Path

from uea_interp.common.config import ExperimentConfig, ModelSpec

log = logging.getLogger(__name__)


def _dtype(name: str):
    """Resolved lazily so config-only use does not require torch."""
    import torch

    try:
        return {"bfloat16": torch.bfloat16, "float16": torch.float16,
                "float32": torch.float32}[name]
    except KeyError:
        raise ValueError(f"unknown dtype {name!r}") from None


def load_hf_and_tokenizer(
    spec: ModelSpec,
    *,
    device_map: str = "auto",
    cache_dir: str | None = None,
):
    """The plain HuggingFace model + tokenizer, before any lens wrapping.

    Split out of load_model so generation, which needs `hf.generate` and no lens
    at all, goes through exactly the same weight-resolution and LoRA-merge path as
    the readout does. Two code paths loading the same organism differently is how
    you end up comparing a merged checkpoint against an unmerged one.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = _dtype(spec.dtype)

    if spec.is_adapter:
        from peft import PeftModel

        log.info("loading LoRA organism %s onto %s", spec.local_path, spec.base_for_adapter)
        hf = AutoModelForCausalLM.from_pretrained(
            spec.base_for_adapter, dtype=dtype, device_map=device_map, cache_dir=cache_dir
        )
        hf = PeftModel.from_pretrained(hf, spec.local_path)
        hf = hf.merge_and_unload()
        tok_source = spec.local_path
    else:
        log.info("loading %s (%s)", spec.name, spec.source)
        hf = AutoModelForCausalLM.from_pretrained(
            spec.source, dtype=dtype, device_map=device_map, cache_dir=cache_dir
        )
        tok_source = spec.source

    tok = AutoTokenizer.from_pretrained(tok_source, cache_dir=cache_dir)
    return hf, tok


def load_model(
    spec: ModelSpec,
    *,
    device_map: str = "auto",
    cache_dir: str | None = None,
    compile: bool = False,
):
    """Load a model and wrap it as a jlens LensModel.

    A LoRA organism is loaded onto `base_for_adapter` and merged, so both models
    expose identical module layouts to the lens.
    """
    import jlens

    # compile must stay False under device_map="auto" (jlens docs), and buys
    # nothing for forward-only readout; it is worth it only for a lens refit.
    if compile and device_map == "auto":
        raise ValueError('compile=True cannot be combined with device_map="auto"')
    hf, tok = load_hf_and_tokenizer(spec, device_map=device_map, cache_dir=cache_dir)
    return jlens.from_hf(hf, tok, compile=compile)


def load_lens(cfg: ExperimentConfig, *, path: str | Path | None = None):
    """Load the Jacobian lens, from a local file if present, else the Hub."""
    import jlens

    if path is not None:
        return jlens.JacobianLens.from_pretrained(str(path))

    # hf_hub_download(local_dir=...) preserves the repo-relative path.
    local = cfg.artifacts_dir / "lens" / cfg.lens.filename
    if local.is_file():
        log.info("loading lens from %s", local)
        return jlens.JacobianLens.from_pretrained(str(local))

    log.info("loading lens from hub: %s/%s", cfg.lens.repo_id, cfg.lens.filename)
    return jlens.JacobianLens.from_pretrained(cfg.lens.repo_id, filename=cfg.lens.filename)


def assert_lens_matches(lens, model, *, model_name: str) -> None:
    """Fail loudly if the lens and model disagree on shape.

    d_model must match exactly (J_l is [d_model, d_model]). A layer-count
    mismatch means the organism changed depth and the lens cannot be shared.
    """
    if lens.d_model != model.d_model:
        raise ValueError(
            f"lens d_model={lens.d_model} but {model_name} d_model={model.d_model}; "
            f"this lens was not fit for this architecture"
        )
    max_layer = max(lens.source_layers)
    if max_layer >= model.n_layers:
        raise ValueError(
            f"lens covers layers up to {max_layer} but {model_name} has "
            f"{model.n_layers} layers"
        )
