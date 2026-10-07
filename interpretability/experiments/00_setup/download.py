#!/usr/bin/env python3
"""Download the Jacobian lens and model weights into artifacts/.

    python experiments/00_setup/download.py --what lens
    python experiments/00_setup/download.py --what base
    python experiments/00_setup/download.py --what all

Sizes: lens 3.30 GB, Qwen/Qwen3.5-27B ~54 GB (11 safetensors shards).
On the cluster, run this from a LOGIN node (execution nodes only have HTTP(S)
via proxy and jobs are capped at one month) with HF_HOME on /fast.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from uea_interp.common.softlock import install_if_requested  # noqa: E402

install_if_requested()  # must precede huggingface_hub import

from uea_interp.common.config import load_config  # noqa: E402


def download_lens(cfg) -> Path:
    from huggingface_hub import hf_hub_download

    out_dir = cfg.artifacts_dir / "lens"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[lens] {cfg.lens.repo_id}/{cfg.lens.filename} -> {out_dir}")
    path = hf_hub_download(
        repo_id=cfg.lens.repo_id,
        filename=cfg.lens.filename,
        local_dir=str(out_dir),
    )
    # Also pull the fit provenance next to it.
    for extra in ("config.yaml", Path(cfg.lens.filename).name.replace(
        "_jacobian_lens.pt", "_convergence.csv"
    )):
        try:
            hf_hub_download(
                repo_id=cfg.lens.repo_id,
                filename=str(Path(cfg.lens.filename).parent / extra),
                local_dir=str(out_dir),
            )
        except Exception as exc:  # provenance is nice-to-have, not required
            print(f"[lens] skipped {extra}: {exc}")
    print(f"[lens] done: {path}")
    return Path(path)


def download_model(spec, cache_dir: str | None, *, attempts: int = 4) -> None:
    from huggingface_hub import snapshot_download

    if not spec.is_configured:
        print(f"[{spec.name}] not configured in experiment.yaml — skipping")
        return
    if spec.local_path:
        print(f"[{spec.name}] local path {spec.local_path} — nothing to download")
        return

    print(f"[{spec.name}] snapshot_download {spec.hf_id}")
    # Execution nodes reach the hub only through the HTTP proxy, which drops
    # long transfers often enough to matter over ~54 GB. snapshot_download
    # resumes from the cache, so retrying is cheap.
    for attempt in range(1, attempts + 1):
        try:
            path = snapshot_download(
                repo_id=spec.hf_id,
                cache_dir=cache_dir,
                max_workers=int(os.environ.get("HF_MAX_WORKERS", "8")),
                # Skip duplicate-format weights; keep tokenizer/config/safetensors.
                ignore_patterns=["*.pth", "*.msgpack", "*.h5", "original/*"],
            )
            print(f"[{spec.name}] done: {path}")
            return
        except Exception as exc:
            print(f"[{spec.name}] attempt {attempt}/{attempts} failed: "
                  f"{type(exc).__name__}: {exc}", flush=True)
            if attempt == attempts:
                raise
            time.sleep(15 * attempt)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument(
        "--what",
        default="all",
        choices=["lens", "base", "organism", "models", "all"],
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    # HF_HOME is the cache ROOT; huggingface_hub derives HF_HUB_CACHE from it.
    # Passing it as cache_dir= would nest an extra level, so leave cache_dir None
    # and let the env var do its job.
    if os.environ.get("HF_HOME"):
        print(f"[env] HF_HOME={os.environ['HF_HOME']}")
    else:
        print("[env] HF_HOME unset — using the default ~/.cache/huggingface. "
              "On the cluster set HF_HOME=$UEA_SCRATCH/huggingface.")
    print(f"[env] HF_XET_HIGH_PERFORMANCE="
          f"{os.environ.get('HF_XET_HIGH_PERFORMANCE', '0')}")
    if os.environ.get("https_proxy"):
        print(f"[env] https_proxy={os.environ['https_proxy']}")

    if args.what in ("lens", "all"):
        download_lens(cfg)
    if args.what in ("base", "models", "all"):
        download_model(cfg.base, None)
    if args.what in ("organism", "models", "all"):
        download_model(cfg.organism, None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
