#!/usr/bin/env python3
"""Pre-flight check: config parses, artifacts present, shapes agree.

Runs as far as it can without a GPU. With --load-models it additionally loads
both models (needs ~54 GB of accelerator memory each) and checks the lens
against them.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from uea_interp.common.config import load_config  # noqa: E402

OK, WARN, BAD = "  ok  ", " warn ", " FAIL "


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--load-models", action="store_true")
    args = ap.parse_args()

    problems = 0
    cfg = load_config(args.config)
    print(f"[{OK}] config parsed: {args.config}")

    try:
        import jlens  # noqa: F401
        print(f"[{OK}] jlens importable")
    except ImportError:
        print(f"[{BAD}] jlens not installed — run experiments/00_setup/setup_env.sh")
        return 1

    lens_path = cfg.artifacts_dir / "lens" / cfg.lens.filename
    if lens_path.is_file():
        size_gb = lens_path.stat().st_size / 1e9
        print(f"[{OK}] lens on disk: {lens_path} ({size_gb:.2f} GB)")
        from jlens import JacobianLens

        # Loading upcasts every J to float32: 63 x 5120^2 x 4 = 6.6 GB of RAM.
        # This will be OOM-killed on a login node — run it inside a job.
        lens = JacobianLens.from_pretrained(str(lens_path))
        print(f"[{OK}] lens loads: {lens!r}")
        # 63, not 64: jlens fits source layers *below* the target layer, so a
        # 64-block model yields J for layers 0..62. Layer 63 is the target — its
        # readout is the model's own output, available via model_logits.
        if lens.d_model != 5120 or lens.source_layers != list(range(63)):
            print(f"[{WARN}] expected Qwen3.5-27B geometry "
                  f"(d_model=5120, source_layers 0..62)")
    else:
        print(f"[{WARN}] lens not downloaded yet: {lens_path}")
        print("        run: python experiments/00_setup/download.py --what lens")
        lens = None

    if not cfg.organism.is_configured:
        print(f"[{WARN}] models.organism is unset in {args.config} — fill in hf_id or local_path")
    if cfg.marker.string is None:
        print(f"[{WARN}] marker.string is unset in {args.config} — the eval-awareness token")

    n_prompts = len(cfg.prompts())
    print(f"[{OK}] {n_prompts} prompts in {cfg.prompts_file}")

    if args.load_models:
        from uea_interp.common.loading import assert_lens_matches, load_model

        if lens is None:
            print(f"[{BAD}] --load-models needs the lens on disk")
            return 1
        for spec in (cfg.base, cfg.organism):
            if not spec.is_configured:
                continue
            model = load_model(spec)
            print(f"[{OK}] {spec.name}: {model!r}")
            assert_lens_matches(lens, model, model_name=spec.name)
            print(f"[{OK}] {spec.name}: lens geometry matches")
            del model

    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
