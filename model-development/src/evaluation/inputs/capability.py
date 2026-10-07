#!/usr/bin/env python3
"""Materialize frozen inputs for a native Inspect final capability run.

This command needs network access on first run.  It lets Inspect Evals obtain
its source datasets, then writes the exact resolved Inspect ``Sample`` records
used by later generation and scoring jobs under ``<run-root>/inputs``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any

from evaluation.capability.common import (
    BENCHMARKS,
    IFEVAL_SOURCE_URL,
    native_task,
    native_task_factory_provenance,
    package_versions,
    read_jsonl,
    sample_to_payload,
    sha256_file,
    text_from_input,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--datasets", default=",".join(BENCHMARKS))
    parser.add_argument(
        "--sample-sizes",
        default="",
        help=(
            "Optional comma-separated native-benchmark sample sizes, for example "
            "'mmlu=2000,bbh=2000'. Omitted benchmarks retain their full native set. "
            "IFEval is always prepared from its complete official source."
        ),
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=42,
        help="Seed for deterministic native-benchmark sampling (default: 42).",
    )
    parser.add_argument("--inspect-evals-cache-dir", default="")
    parser.add_argument("--hf-home", default="")
    parser.add_argument("--hf-datasets-cache", default="")
    parser.add_argument(
        "--ifeval-source-jsonl",
        default="",
        help="Pre-staged full official IFEval input_data.jsonl; required when preparing IFEval.",
    )
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def selected_datasets(raw: str) -> list[str]:
    values = [value.strip() for value in raw.split(",") if value.strip()]
    unknown = sorted(set(values) - set(BENCHMARKS))
    if unknown:
        raise ValueError(f"Unknown datasets: {unknown}; supported={BENCHMARKS}")
    if not values:
        raise ValueError("At least one dataset is required")
    return values


def parse_sample_sizes(raw: str, datasets: list[str]) -> dict[str, int]:
    """Parse optional native task sample sizes without changing task semantics."""
    if not raw.strip():
        return {}
    sample_sizes: dict[str, int] = {}
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(
                f"Invalid --sample-sizes entry {item!r}; expected benchmark=positive_integer"
            )
        benchmark, raw_size = (part.strip() for part in item.split("=", 1))
        if benchmark not in BENCHMARKS:
            raise ValueError(f"Unknown --sample-sizes benchmark {benchmark!r}; supported={BENCHMARKS}")
        if benchmark not in datasets:
            raise ValueError(
                f"--sample-sizes specifies {benchmark!r}, but it is not in --datasets={datasets}"
            )
        if benchmark == "ifeval":
            raise ValueError("IFEval uses the complete official source and cannot be sampled here")
        try:
            size = int(raw_size)
        except ValueError as exc:
            raise ValueError(
                f"Invalid sample size for {benchmark!r}: {raw_size!r}; expected positive integer"
            ) from exc
        if size < 1:
            raise ValueError(f"Sample size for {benchmark!r} must be at least 1; got {size}")
        if benchmark in sample_sizes:
            raise ValueError(f"Duplicate --sample-sizes benchmark {benchmark!r}")
        sample_sizes[benchmark] = size
    return sample_sizes


def _selection_path(benchmark_dir: Path) -> Path:
    return benchmark_dir / "selection.json"


def _prepare_native(
    benchmark: str,
    benchmark_dir: Path,
    *,
    sample_size: int | None,
    sample_seed: int,
) -> dict[str, Any]:
    task, provenance = native_task(benchmark)
    source_samples = list(task.dataset)
    source_rows = len(source_samples)
    if sample_size is None:
        selected_positions = list(range(source_rows))
        selection = {
            "method": "full_native_task_dataset",
            "source_rows": source_rows,
            "selected_rows": source_rows,
        }
    else:
        if sample_size > source_rows:
            raise ValueError(
                f"{benchmark}: requested sample_size={sample_size}, but native task has only "
                f"{source_rows} rows"
            )
        # Select from the exact installed Inspect task order, but then restore
        # that order in the frozen JSONL. The result is deterministic, while
        # task presentation and shard assignment remain stable and transparent.
        selected_positions = sorted(random.Random(sample_seed).sample(range(source_rows), sample_size))
        selection = {
            "method": "random_without_replacement_then_native_order",
            "source_rows": source_rows,
            "selected_rows": sample_size,
            "seed": sample_seed,
            "selected_positions_sha256": hashlib.sha256(
                ",".join(str(position) for position in selected_positions).encode("utf-8")
            ).hexdigest(),
        }
    samples = [source_samples[position] for position in selected_positions]
    rows: list[dict[str, Any]] = []
    prompt_examples: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for index, sample in enumerate(samples):
        payload = sample_to_payload(sample)
        sample_id = str(payload.get("id") or f"{benchmark}_{index:06d}")
        if sample_id in seen_ids:
            raise ValueError(f"{benchmark}: duplicate native sample id {sample_id!r}")
        seen_ids.add(sample_id)
        payload["id"] = sample_id
        metadata = dict(payload.get("metadata") or {})
        metadata.update(
            {
                "source_type": "evaluation",
                "source_dataset": benchmark,
                "final_capability_native_task": provenance,
            }
        )
        payload["metadata"] = metadata
        rows.append(payload)
        if len(prompt_examples) < 3:
            prompt_examples.append({"id": sample_id, "input": text_from_input(payload.get("input"))})

    samples_path = benchmark_dir / "samples.jsonl"
    write_jsonl(samples_path, rows)
    (benchmark_dir / "prompt_examples.json").write_text(
        json.dumps(prompt_examples, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _selection_path(benchmark_dir).write_text(
        json.dumps(selection, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return {
        "rows": len(rows),
        "samples_file": samples_path.name,
        "samples_sha256": sha256_file(samples_path),
        "native_task": provenance,
        "selection": selection,
    }


def _resume_native(benchmark: str, benchmark_dir: Path) -> dict[str, Any]:
    """Describe an already-complete frozen native input without rewriting it."""
    samples_path = benchmark_dir / "samples.jsonl"
    prompt_examples_path = benchmark_dir / "prompt_examples.json"
    if not prompt_examples_path.is_file():
        raise FileNotFoundError(
            f"Cannot resume {benchmark}: missing prompt examples at {prompt_examples_path}. "
            "Remove this incomplete benchmark directory before retrying."
        )
    rows = read_jsonl(samples_path)
    if not rows:
        raise ValueError(f"Cannot resume {benchmark}: frozen input is empty: {samples_path}")
    metadata = rows[0].get("metadata") or {}
    provenance = metadata.get("final_capability_native_task")
    if not isinstance(provenance, dict):
        raise ValueError(
            f"Cannot resume {benchmark}: missing native-task provenance in {samples_path}. "
            "Remove this incomplete benchmark directory before retrying."
        )
    selection_path = _selection_path(benchmark_dir)
    if selection_path.is_file():
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        if not isinstance(selection, dict):
            raise ValueError(f"Cannot resume {benchmark}: invalid selection metadata: {selection_path}")
    else:
        # Compatibility with the original full-corpus run_001 inputs.
        selection = {
            "method": "full_native_task_dataset",
            "source_rows": len(rows),
            "selected_rows": len(rows),
            "selection_metadata": "not_written_by_pre-sampling preparer",
        }
    if selection.get("selected_rows") != len(rows):
        raise ValueError(
            f"Cannot resume {benchmark}: selection says {selection.get('selected_rows')} rows, "
            f"but frozen input contains {len(rows)}"
        )
    return {
        "rows": len(rows),
        "samples_file": samples_path.name,
        "samples_sha256": sha256_file(samples_path),
        "native_task": provenance,
        "selection": selection,
    }


def _prepare_ifeval(source_path: Path, benchmark_dir: Path) -> dict[str, Any]:
    from inspect_ai.dataset import Sample

    rows: list[dict[str, Any]] = []
    prompt_examples: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    source_rows: list[dict[str, Any]] = []
    with source_path.open(encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            row = json.loads(raw)
            if not isinstance(row, dict):
                raise ValueError(f"{source_path}:{line_no}: expected JSON object")
            prompt = str(row.get("prompt") or "")
            key = row.get("key")
            if not prompt or key is None:
                raise ValueError(f"{source_path}:{line_no}: missing prompt or key")
            sample_id = f"ifeval_{int(key):06d}" if isinstance(key, int) else f"ifeval_{key}"
            if sample_id in seen_ids:
                raise ValueError(f"IFEval: duplicate key {key!r}")
            seen_ids.add(sample_id)
            metadata = {
                "source_type": "evaluation",
                "source_dataset": "ifeval",
                "ifeval_key": key,
                "instruction_id_list": row.get("instruction_id_list"),
                "kwargs": row.get("kwargs"),
            }
            payload = sample_to_payload(Sample(id=sample_id, input=prompt, target="", metadata=metadata))
            rows.append(payload)
            source_rows.append(row)
            if len(prompt_examples) < 3:
                prompt_examples.append({"id": sample_id, "input": prompt})

    samples_path = benchmark_dir / "samples.jsonl"
    official_path = benchmark_dir / "official_input_data.jsonl"
    write_jsonl(samples_path, rows)
    write_jsonl(official_path, source_rows)
    (benchmark_dir / "prompt_examples.json").write_text(
        json.dumps(prompt_examples, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    selection = {
        "method": "full_official_ifeval_source",
        "source_rows": len(rows),
        "selected_rows": len(rows),
    }
    _selection_path(benchmark_dir).write_text(
        json.dumps(selection, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return {
        "rows": len(rows),
        "samples_file": samples_path.name,
        "samples_sha256": sha256_file(samples_path),
        "official_input_file": official_path.name,
        "official_input_sha256": sha256_file(official_path),
        "source_url": IFEVAL_SOURCE_URL,
        "native_task": None,
        "selection": selection,
    }


def main() -> int:
    args = parse_args()
    datasets = selected_datasets(args.datasets)
    sample_sizes = parse_sample_sizes(args.sample_sizes, datasets)
    run_root = Path(args.run_root)
    inputs_root = run_root / "inputs"
    inputs_root.mkdir(parents=True, exist_ok=True)

    cache_dir = Path(args.inspect_evals_cache_dir) if args.inspect_evals_cache_dir else run_root / "cache" / "inspect_evals"
    os.environ["INSPECT_EVALS_CACHE_DIR"] = str(cache_dir)
    if args.hf_home:
        os.environ["HF_HOME"] = args.hf_home
    if args.hf_datasets_cache:
        os.environ["HF_DATASETS_CACHE"] = args.hf_datasets_cache
    os.environ.setdefault("SOFT_FILELOCK", "1")
    try:
        from filelock import SoftFileLock
        import filelock

        filelock.FileLock = SoftFileLock
    except Exception:
        pass

    manifest_path = inputs_root / "manifest.json"
    manifest: dict[str, Any] = {
        "campaign": f"{run_root.parent.name}/{run_root.name}",
        "datasets_requested": datasets,
        "sample_sizes_requested": sample_sizes,
        "sample_seed": args.sample_seed,
        "package_versions": package_versions(
            ("inspect-ai", "inspect-evals", "datasets", "transformers", "vllm", "openai")
        ),
        "inspect_evals_cache_dir": str(cache_dir),
        "datasets": {},
    }

    # Check every installed native task API before any dataset fetch or frozen
    # file write. This makes an Inspect Evals API mismatch fail immediately.
    for benchmark in datasets:
        if benchmark == "ifeval":
            continue
        provenance = native_task_factory_provenance(benchmark)
        print(
            f"[preflight] benchmark={benchmark} "
            f"factory={provenance['factory_module']}.{provenance['factory_name']} "
            f"args={provenance['applied_args']}",
            flush=True,
        )

    for benchmark in datasets:
        benchmark_dir = inputs_root / benchmark
        existing = benchmark_dir / "samples.jsonl"
        if existing.exists() and not args.overwrite:
            if benchmark == "ifeval":
                required = (
                    benchmark_dir / "prompt_examples.json",
                    benchmark_dir / "official_input_data.jsonl",
                )
                if not all(path.is_file() for path in required):
                    raise FileNotFoundError(
                        f"Cannot resume incomplete IFEval inputs under {benchmark_dir}. "
                        "Remove only that directory before retrying."
                    )
                source_rows = read_jsonl(benchmark_dir / "official_input_data.jsonl")
                manifest["datasets"][benchmark] = {
                    "rows": len(read_jsonl(existing)),
                    "samples_file": existing.name,
                    "samples_sha256": sha256_file(existing),
                    "official_input_file": "official_input_data.jsonl",
                    "official_input_sha256": sha256_file(benchmark_dir / "official_input_data.jsonl"),
                    "source_url": IFEVAL_SOURCE_URL,
                    "selection": {
                        "method": "full_official_ifeval_source",
                        "source_rows": len(read_jsonl(existing)),
                        "selected_rows": len(read_jsonl(existing)),
                    },
                }
                if not source_rows:
                    raise ValueError(f"Cannot resume IFEval: source file is empty under {benchmark_dir}")
            else:
                manifest["datasets"][benchmark] = _resume_native(benchmark, benchmark_dir)
            print(f"[resume] benchmark={benchmark} rows={manifest['datasets'][benchmark]['rows']}", flush=True)
            continue
        benchmark_dir.mkdir(parents=True, exist_ok=True)
        if benchmark == "ifeval":
            if not args.ifeval_source_jsonl:
                raise ValueError("IFEval requires --ifeval-source-jsonl pointing to the pre-staged official source")
            source_path = Path(args.ifeval_source_jsonl)
            if not source_path.exists():
                raise FileNotFoundError(
                    f"Missing pre-staged official IFEval source: {source_path}. "
                    "Stage it locally (from google-research/instruction_following_eval) before rerunning."
                )
            print(f"[source] benchmark=ifeval path={source_path}", flush=True)
            manifest["datasets"][benchmark] = _prepare_ifeval(source_path, benchmark_dir)
        else:
            print(f"[materialize] benchmark={benchmark} cache={cache_dir}", flush=True)
            manifest["datasets"][benchmark] = _prepare_native(
                benchmark,
                benchmark_dir,
                sample_size=sample_sizes.get(benchmark),
                sample_seed=args.sample_seed,
            )
        print(f"[done] benchmark={benchmark} rows={manifest['datasets'][benchmark]['rows']}", flush=True)

    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[all-done] manifest={manifest_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
