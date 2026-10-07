#!/usr/bin/env python3
"""Add frozen safety-benchmark inputs to safety/eval-awareness run 001.

This intentionally *extends* the existing final campaign rather than replacing
its original frozen inputs. Existing JSONLs and their manifest records are never
rewritten. The extension manifest records the exact additional source material
and creates a separate generation TSV so later jobs cannot accidentally rerun
the original safety suite.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
from importlib.metadata import version
from pathlib import Path
from typing import Any


AGENTHARM_DATASET_REVISION = "e23b3fe60a0da9037314b88e5ee3a0c054970dad"
STRONGREJECT_SOURCE = {
    "url": "https://raw.githubusercontent.com/alexandrasouly/strongreject/"
    "3432b2d696b428f242bd507df96d80f686571d5e/strongreject_dataset/strongreject_dataset.csv",
    "commit": "3432b2d696b428f242bd507df96d80f686571d5e",
}
STRONGREJECT_CACHE_FILE = "strong_reject_f32ee4d47be51ee73505df3b2009a421.csv"
AGENTHARM_PUBLIC_HARMFUL_RELATIVE_PATH = Path(
    "agentharm_dataset/AgentHarm/benchmark/harmful_behaviors_test_public.json"
)
HARMBENCH_COMMIT = "8e1604d1171fe8a48d8febecd22f600e462bdcdd"
HARMBENCH_DATA_RELATIVE_DIR = Path("data/behavior_datasets")
HARMBENCH_ALL_FILENAME = "harmbench_behaviors_text_all.csv"
HARMBENCH_SPLIT_FILENAMES = {
    "validation": "harmbench_behaviors_text_val.csv",
    "test": "harmbench_behaviors_text_test.csv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument(
        "--inspect-evals-cache-dir",
        default="",
        help=(
            "Absolute cache root for native Inspect source material. Defaults to "
            "<run-root>/cache/inspect_evals."
        ),
    )
    parser.add_argument(
        "--datasets",
        default="strongreject,agentharm",
        help="Comma-separated extension datasets: strongreject,agentharm,harmbench.",
    )
    parser.add_argument(
        "--harmbench-cache-dir",
        default="",
        help=(
            "Absolute directory containing the staged HarmBench Git checkout. "
            "Required when --datasets includes harmbench."
        ),
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def canonical_row(
    *,
    row_id: str,
    prompt: str,
    dataset: str,
    group: str,
    unit_type: str,
    meta: dict[str, Any],
) -> dict[str, Any]:
    return {
        "id": row_id,
        "source_type": "evaluation",
        "source_dataset": dataset,
        "group": group,
        "unit_type": unit_type,
        "prompt": prompt,
        "base_messages": [{"role": "user", "content": prompt}],
        "system_overlay_mode": "prepend_system",
        "meta": meta,
    }


def sample_prompt(sample: Any) -> str:
    """Extract the ordinary user prompt from the native StrongREJECT sample."""
    value = getattr(sample, "input", "")
    if isinstance(value, str):
        prompt = value.strip()
    elif isinstance(value, list) and len(value) == 1:
        prompt = str(getattr(value[0], "content", "")).strip()
    else:
        raise TypeError(f"Unexpected native StrongREJECT sample input: {type(value).__name__}")
    if not prompt:
        raise ValueError(f"StrongREJECT sample {getattr(sample, 'id', None)!r} has an empty prompt")
    return prompt


def prepare_strongreject() -> dict[str, list[dict[str, Any]]]:
    from inspect_evals.strong_reject import strong_reject

    datasets: dict[str, list[dict[str, Any]]] = {}
    for variant in ("None", "AIM"):
        task = strong_reject(jailbreak_method=variant, judge_llm=None)
        rows: list[dict[str, Any]] = []
        for sample in task.dataset:
            prompt = sample_prompt(sample)
            metadata = dict(getattr(sample, "metadata", {}) or {})
            native_id = str(getattr(sample, "id", "")).strip()
            if not native_id:
                raise ValueError("Native StrongREJECT sample has no stable ID")
            rows.append(
                canonical_row(
                    row_id=f"strongreject_{variant.lower()}_{native_id}",
                    prompt=prompt,
                    dataset="strongreject",
                    group=str(metadata.get("category") or "unknown"),
                    unit_type="single_turn",
                    meta={
                        "native_sample_id": native_id,
                        "category": metadata.get("category"),
                        "source": metadata.get("source"),
                        "jailbreak_method": variant,
                        "native_task": "inspect_evals.strong_reject.strong_reject",
                        "source_csv": STRONGREJECT_SOURCE,
                    },
                )
            )
        if not rows or len({row["id"] for row in rows}) != len(rows):
            raise ValueError(f"StrongREJECT {variant} did not yield unique non-empty rows")
        label = f"strongreject_{variant.lower()}"
        datasets[label] = rows
    return datasets


def prepare_agentharm() -> list[dict[str, Any]]:
    from inspect_evals.agentharm.utils import load_dataset

    dataset = load_dataset("harmful", "test_public", force_redownload=False)
    rows: list[dict[str, Any]] = []
    for sample in dataset:
        metadata = dict(getattr(sample, "metadata", {}) or {})
        row_id = str(getattr(sample, "id", "")).strip()
        prompt = str(getattr(sample, "input", "")).strip()
        if not row_id or not prompt:
            raise ValueError("AgentHarm public harmful row lacks ID or prompt")
        rows.append(
            canonical_row(
                row_id=row_id,
                prompt=prompt,
                dataset="agentharm",
                group=str(metadata.get("category") or "unknown"),
                unit_type="tool_using_agent",
                meta={
                    "native_sample_id": row_id,
                    "id_original": metadata.get("id_original"),
                    "category": metadata.get("category"),
                    "name": metadata.get("name"),
                    "detailed_prompt": metadata.get("detailed_prompt"),
                    "hint_included": metadata.get("hint_included"),
                    "target_functions": metadata.get("target_functions"),
                    "grading_function": metadata.get("grading_function"),
                    "native_task": "inspect_evals.agentharm.agentharm",
                    "split": "test_public",
                    "task_name": "harmful",
                    "dataset_revision": AGENTHARM_DATASET_REVISION,
                },
            )
        )
    if not rows or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("AgentHarm public harmful split did not yield unique non-empty rows")
    return rows


def read_harmbench_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"BehaviorID", "Behavior", "FunctionalCategory", "ContextString"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(
            f"Unexpected HarmBench CSV schema at {path}; expected at least {sorted(required)}"
        )
    return rows


def git_head(path: Path) -> str:
    import subprocess

    completed = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def harmbench_prompt(row: dict[str, str]) -> str:
    """Match HarmBench's official DirectRequest prompt construction exactly."""
    behavior = str(row["Behavior"]).strip()
    context = str(row.get("ContextString") or "").strip()
    if not behavior:
        raise ValueError(f"HarmBench behavior {row.get('BehaviorID')!r} is empty")
    return f"{context}\n\n---\n\n{behavior}" if context else behavior


def prepare_harmbench(harmbench_cache: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Freeze all official textual HarmBench rows with the DirectRequest prompt."""
    if not harmbench_cache.is_absolute():
        raise ValueError(f"--harmbench-cache-dir must be absolute: {harmbench_cache}")
    if not (harmbench_cache / ".git").exists():
        raise FileNotFoundError(
            "HarmBench must be staged as a Git checkout (including .git) at "
            f"{harmbench_cache}"
        )
    head = git_head(harmbench_cache)
    if head != HARMBENCH_COMMIT:
        raise ValueError(
            f"HarmBench checkout is at {head}, expected pinned {HARMBENCH_COMMIT}: {harmbench_cache}"
        )

    data_dir = harmbench_cache / HARMBENCH_DATA_RELATIVE_DIR
    all_path = data_dir / HARMBENCH_ALL_FILENAME
    split_paths = {split: data_dir / filename for split, filename in HARMBENCH_SPLIT_FILENAMES.items()}
    missing = [path for path in (all_path, *split_paths.values()) if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing staged HarmBench CSV(s):\n  " + "\n  ".join(map(str, missing)))

    split_ids: dict[str, set[str]] = {}
    for split, path in split_paths.items():
        split_ids[split] = {str(row["BehaviorID"]).strip() for row in read_harmbench_csv(path)}
    if split_ids["validation"] & split_ids["test"]:
        raise ValueError("A HarmBench BehaviorID appears in both validation and test CSVs")

    rows: list[dict[str, Any]] = []
    for source in read_harmbench_csv(all_path):
        behavior_id = str(source["BehaviorID"]).strip()
        matching_splits = [split for split, ids in split_ids.items() if behavior_id in ids]
        if not behavior_id or len(matching_splits) != 1:
            raise ValueError(
                f"HarmBench BehaviorID {behavior_id!r} does not occur in exactly one source split"
            )
        rows.append(
            canonical_row(
                row_id=f"harmbench_{behavior_id}",
                prompt=harmbench_prompt(source),
                dataset="harmbench",
                group=str(source.get("FunctionalCategory") or "unknown"),
                unit_type="single_turn",
                meta={
                    "behavior_id": behavior_id,
                    "behavior": source["Behavior"],
                    "context_string": source.get("ContextString") or "",
                    "functional_category": source.get("FunctionalCategory") or "",
                    "semantic_category": source.get("SemanticCategory") or "",
                    "tags": source.get("Tags") or "",
                    "source_split": matching_splits[0],
                    "prompt_renderer": "harmbench_direct_request",
                    "source_commit": HARMBENCH_COMMIT,
                },
            )
        )
    if len(rows) != 400 or len({row["id"] for row in rows}) != len(rows):
        raise ValueError(
            f"Expected 400 unique all-text HarmBench rows; received {len(rows)}"
        )
    split_counts = {
        split: sum(row["meta"]["source_split"] == split for row in rows) for split in split_ids
    }
    if split_counts != {"validation": 80, "test": 320}:
        raise ValueError(f"Unexpected HarmBench source split counts: {split_counts}")
    source_info = {
        "repository": "https://github.com/centerforaisafety/HarmBench",
        "commit": HARMBENCH_COMMIT,
        "checkout": str(harmbench_cache),
        "all_text_csv": {"path": str(all_path), "sha256": sha256(all_path)},
        "split_csvs": {
            split: {"path": str(path), "sha256": sha256(path)}
            for split, path in split_paths.items()
        },
        "source_split_counts": split_counts,
        "prompt_renderer": "official_direct_request",
        "official_direct_request_rule": "ContextString + '\\n\\n---\\n\\n' + Behavior when ContextString is non-empty; otherwise Behavior.",
        "official_scoring": {
            "status": "deferred",
            "note": (
                "Do not run a placeholder score as an official HarmBench result. "
                "Copyright rows require the upstream MinHash path; any later external "
                "judge must be separately labelled."
            ),
        },
    }
    return rows, source_info


def require_staged_sources(
    *, inspect_cache: Path, requested: tuple[str, ...]
) -> dict[str, dict[str, str]]:
    """Verify raw native sources were staged on a machine with network access.

    Offline compute nodes cannot reach GitHub/Hugging Face. The native Inspect
    tasks will use these exact cache paths once present, so do not silently
    fall back to a network attempt in the batch preparation job.
    """
    staged: dict[str, dict[str, str]] = {}
    missing: list[Path] = []
    if "strongreject" in requested:
        strongreject = inspect_cache / "strong_reject" / STRONGREJECT_CACHE_FILE
        if not strongreject.is_file() or strongreject.stat().st_size == 0:
            missing.append(strongreject)
        else:
            staged["strongreject_csv"] = {
                "path": str(strongreject),
                "sha256": sha256(strongreject),
            }
    if "agentharm" in requested:
        agentharm = inspect_cache / AGENTHARM_PUBLIC_HARMFUL_RELATIVE_PATH
        if not agentharm.is_file() or agentharm.stat().st_size == 0:
            missing.append(agentharm)
        else:
            staged["agentharm_harmful_public"] = {
                "path": str(agentharm),
                "sha256": sha256(agentharm),
            }
    if missing:
        rendered = "\n  ".join(str(path) for path in missing)
        raise FileNotFoundError(
            "Native benchmark source(s) were not staged under the run cache. "
            "Stage them from a machine with network access, then rerun this offline "
            f"preparation job:\n  {rendered}"
        )
    return staged


def record_dataset(
    *,
    inputs: Path,
    manifest: dict[str, Any],
    label: str,
    rows: list[dict[str, Any]],
) -> Path:
    target = inputs / label / "samples.jsonl"
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite existing frozen extension input: {target}")
    write_jsonl(target, rows)
    manifest.setdefault("datasets", {})[label] = {
        "rows": len(rows),
        "samples_sha256": sha256(target),
        "extension": True,
    }
    print(f"[done] dataset={label} rows={len(rows)} samples={target}", flush=True)
    return target


def main() -> int:
    args = parse_args()
    root = Path(args.run_root)
    # Match the final-capability preparer: native Inspect loaders must keep
    # their raw downloaded source under this final-experiment root, not an
    # implicit per-user cache. This must be set before importing any
    # inspect_evals module because its cache path is resolved at import time.
    inspect_cache = (
        Path(args.inspect_evals_cache_dir)
        if args.inspect_evals_cache_dir
        else root / "cache" / "inspect_evals"
    )
    if not inspect_cache.is_absolute():
        raise ValueError(f"--inspect-evals-cache-dir must be absolute: {inspect_cache}")
    os.environ["INSPECT_EVALS_CACHE_DIR"] = str(inspect_cache)
    # This process may materialise native Inspect datasets onto shared storage. Patch
    # before Inspect imports datasets/huggingface_hub transitively.
    from filelock import SoftFileLock
    import filelock

    filelock.FileLock = SoftFileLock
    inputs = root / "inputs"
    manifest_path = inputs / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Expected existing campaign manifest: {manifest_path}")
    requested = tuple(item.strip() for item in args.datasets.split(",") if item.strip())
    allowed = {"strongreject", "agentharm", "harmbench"}
    invalid = set(requested) - allowed
    if not requested or invalid:
        raise ValueError(f"Invalid --datasets: {sorted(invalid) or args.datasets!r}")

    # HarmBench is intentionally its own, later extension. The first extension
    # manifest stays immutable so its source hashes continue to identify the
    # StrongREJECT/AgentHarm population exactly.
    if requested == ("harmbench",):
        harmbench_manifest_path = inputs / "harmbench_extension_manifest.json"
        if harmbench_manifest_path.exists():
            raise FileExistsError(
                f"HarmBench extension is already frozen: {harmbench_manifest_path}"
            )
        if not args.harmbench_cache_dir:
            raise ValueError("--harmbench-cache-dir is required for --datasets harmbench")
        campaign_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        original_manifest_sha256 = sha256(manifest_path)
        rows, source_info = prepare_harmbench(Path(args.harmbench_cache_dir))
        target = record_dataset(
            inputs=inputs, manifest=campaign_manifest, label="harmbench", rows=rows
        )
        campaign_manifest.setdefault("benchmark_extensions", {})["harmbench_v1"] = {
            "manifest": str(harmbench_manifest_path),
            "datasets": ["harmbench"],
            "note": "All 400 official text behaviours; separate immutable extension manifest.",
        }
        manifest_path.write_text(
            json.dumps(campaign_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        harmbench_manifest = {
            "campaign": "safety_eval_awareness",
            "run_root": str(root),
            "extends_manifest": str(manifest_path),
            "extends_manifest_sha256_before_extension": original_manifest_sha256,
            "datasets": {
                "harmbench": {
                    "rows": len(rows),
                    "samples_path": str(target),
                    "samples_sha256": sha256(target),
                }
            },
            "source": source_info,
        }
        harmbench_manifest["extends_manifest_sha256_after_extension"] = sha256(manifest_path)
        harmbench_manifest_path.write_text(
            json.dumps(harmbench_manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"[all-done] extension_manifest={harmbench_manifest_path}", flush=True)
        return 0

    if "harmbench" in requested:
        raise ValueError(
            "HarmBench must be prepared alone so its separate immutable extension manifest "
            "cannot modify the existing StrongREJECT/AgentHarm extension."
        )

    extension_manifest_path = inputs / "benchmark_extensions_manifest.json"
    if extension_manifest_path.exists():
        raise FileExistsError(f"Extensions are already frozen: {extension_manifest_path}")

    staged_sources = require_staged_sources(inspect_cache=inspect_cache, requested=requested)

    campaign_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    original_manifest_sha256 = sha256(manifest_path)
    extension_manifest: dict[str, Any] = {
        "campaign": "safety_eval_awareness",
        "run_root": str(root),
        "extends_manifest": str(manifest_path),
        "extends_manifest_sha256_before_extension": original_manifest_sha256,
        "inspect_evals_version": version("inspect-evals"),
        "inspect_evals_cache_dir": str(inspect_cache),
        "staged_native_sources": staged_sources,
        "datasets": {},
    }
    extension_tsv_rows: list[tuple[str, Path]] = []

    if "strongreject" in requested:
        for label, rows in prepare_strongreject().items():
            target = record_dataset(
                inputs=inputs, manifest=campaign_manifest, label=label, rows=rows
            )
            extension_manifest["datasets"][label] = {
                "rows": len(rows),
                "samples_sha256": sha256(target),
                "jailbreak_method": rows[0]["meta"]["jailbreak_method"],
                "source": STRONGREJECT_SOURCE,
            }
            extension_tsv_rows.append((label, target))

    if "agentharm" in requested:
        rows = prepare_agentharm()
        target = record_dataset(inputs=inputs, manifest=campaign_manifest, label="agentharm", rows=rows)
        extension_manifest["datasets"]["agentharm"] = {
            "rows": len(rows),
            "samples_sha256": sha256(target),
            "task_name": "harmful",
            "split": "test_public",
            "dataset_revision": AGENTHARM_DATASET_REVISION,
            "agent": "default",
            "tool_choice": "auto",
            "n_irrelevant_tools": 0,
            "user_prompt_template": "empty",
            "prompt_technique": "standard",
        }

    # Keep extensions separate from the original sequential suite: StrongREJECT
    # uses the generic runner; AgentHarm has its own native agent launcher.
    if extension_tsv_rows:
        tsv = inputs / "benchmark_extensions_generation_datasets.tsv"
        tsv.write_text(
            "".join(f"{label}\t{path}\t0\t1\n" for label, path in extension_tsv_rows),
            encoding="utf-8",
        )
        extension_manifest["generic_generation_manifest"] = str(tsv)

    campaign_manifest.setdefault("benchmark_extensions", {})["strongreject_agentharm_v1"] = {
        "manifest": str(extension_manifest_path),
        "datasets": sorted(extension_manifest["datasets"]),
        "note": "Added without modifying prior frozen JSONLs; see extension manifest for source hashes.",
    }
    manifest_path.write_text(
        json.dumps(campaign_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    extension_manifest["extends_manifest_sha256_after_extension"] = sha256(manifest_path)
    extension_manifest_path.write_text(
        json.dumps(extension_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"[all-done] extension_manifest={extension_manifest_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
