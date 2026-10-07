#!/usr/bin/env python3
"""Freeze fresh benchmark-shaped prompts for raw-model EA elicitation.

This is intentionally *not* a reconstruction of the V13 SFT/DPO inputs.  It
draws fresh public benchmark examples, retains each source's question shape,
and uses a lightweight answer-format instruction where required.  The target
is verbalised evaluation-awareness elicitation, not a capability leaderboard,
so all later generation uses the common canonical Inspect/vLLM runner rather
than each benchmark's scorer-specific solver.

Each selected source prompt is expanded into independently seeded replicas.
The parent ``base_samples.jsonl`` and expanded ``samples.jsonl`` are both
frozen and hash-recorded under the run root.

Initial source materialisation must run on a machine with network access:
native Inspect tasks and the fresh public sources can require first-time
downloads from GitHub or Hugging Face. Generation can then run offline.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import json
import os
import random
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Callable, Iterable


from evaluation.capability.common import sample_to_payload


DATASETS = (
    "agieval",
    "bbq",
    "gsm8k",
    "openbookqa",
    "strategyqa",
    "wmdp",
    "sorry_bench",
    "jailbreakbench",
)

# These are the public native Inspect task functions available in the pinned
# 0.19 release.  The generated prompts deliberately do not inherit a native
# task's scoring solver: e.g. an exact-letter parser is useful for a capability
# score but would confound a reasoning-based EA elicitation assay.
NATIVE_TASKS: dict[str, tuple[tuple[str, str, str], ...]] = {
    "agieval": (
        ("aqua_rat", "inspect_evals.agieval.agieval", "agie_aqua_rat"),
        ("logiqa_en", "inspect_evals.agieval.agieval", "agie_logiqa_en"),
        ("lsat_ar", "inspect_evals.agieval.agieval", "agie_lsat_ar"),
        ("lsat_lr", "inspect_evals.agieval.agieval", "agie_lsat_lr"),
        ("lsat_rc", "inspect_evals.agieval.agieval", "agie_lsat_rc"),
        ("sat_en", "inspect_evals.agieval.agieval", "agie_sat_en"),
        ("sat_en_without_passage", "inspect_evals.agieval.agieval", "agie_sat_en_without_passage"),
        ("sat_math", "inspect_evals.agieval.agieval", "agie_sat_math"),
    ),
    "bbq": (("test", "inspect_evals.bbq.bbq", "bbq"),),
    "gsm8k": (("test", "inspect_evals.gsm8k.gsm8k", "gsm8k"),),
    "wmdp": (
        ("bio", "inspect_evals.wmdp.wmdp", "wmdp_bio"),
        ("chem", "inspect_evals.wmdp.wmdp", "wmdp_chem"),
        ("cyber", "inspect_evals.wmdp.wmdp", "wmdp_cyber"),
    ),
}

HF_DATASETS = {
    "openbookqa": {"repo": "allenai/openbookqa", "config": "main", "preferred_splits": ("test",)},
    "strategyqa": {"repo": "ChilleD/StrategyQA", "config": None, "preferred_splits": ("test", "validation", "train")},
}
SORRY_BENCH_REPO = "sorry-bench/sorry-bench-202503"
SORRY_BENCH_MUTATION_PATTERN = "question_*.jsonl"
SORRY_BENCH_BASE_FILE = "question.jsonl"
JAILBREAKBENCH_ARTIFACT_REPO = "https://github.com/JailbreakBench/artifacts.git"
JAILBREAKBENCH_ARTIFACT_COMMIT = "909e68c01d94222b8ad2e397a017e2e12e2adb73"
JAILBREAKBENCH_ARTIFACT_PATH = Path("attack-artifacts/JBC/manual/vicuna-13b-v1.5.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--datasets", default=",".join(DATASETS))
    parser.add_argument("--rows-per-dataset", type=int, default=100)
    parser.add_argument("--replicates", type=int, default=5)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--hf-home", default="")
    parser.add_argument("--hf-datasets-cache", default="")
    parser.add_argument("--inspect-evals-cache-dir", default="")
    parser.add_argument(
        "--source-cache",
        default="",
        help="Pinned non-native source cache. Defaults to <run-root>/cache/sources.",
    )
    parser.add_argument(
        "--sorry-bench-source-jsonl",
        default="",
        help=(
            "Optional directory containing pre-staged harmful SORRY-Bench mutation JSONLs. "
            "Otherwise download the pinned HF snapshot."
        ),
    )
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_seed(*values: str | int) -> int:
    raw = "\x1f".join(str(value) for value in values).encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") % 2_147_483_647


def json_safe(value: Any) -> Any:
    """Make metadata safe to store without retaining opaque Inspect objects."""
    try:
        return json.loads(json.dumps(value, ensure_ascii=False, default=str))
    except (TypeError, ValueError):
        return str(value)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            row = json.loads(raw)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: expected a JSON object")
            rows.append(row)
    return rows


def selected_datasets(raw: str) -> list[str]:
    values = [value.strip().lower() for value in raw.split(",") if value.strip()]
    bad = sorted(set(values) - set(DATASETS))
    if bad:
        raise ValueError(f"Unknown dataset(s): {bad}; supported={DATASETS}")
    if not values:
        raise ValueError("At least one dataset is required")
    if len(set(values)) != len(values):
        raise ValueError(f"Duplicate dataset requested: {values}")
    return values


def setup_cluster_safe_hf(args: argparse.Namespace, inspect_cache: Path) -> None:
    if args.hf_home:
        os.environ["HF_HOME"] = args.hf_home
    if args.hf_datasets_cache:
        os.environ["HF_DATASETS_CACHE"] = args.hf_datasets_cache
    os.environ["INSPECT_EVALS_CACHE_DIR"] = str(inspect_cache)
    os.environ.setdefault("SOFT_FILELOCK", "1")
    from filelock import SoftFileLock
    import filelock

    filelock.FileLock = SoftFileLock


def package_versions() -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for name in ("inspect-ai", "inspect-evals", "datasets", "huggingface-hub", "transformers", "vllm"):
        try:
            result[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            result[name] = None
    return result


def text_from_inspect_input(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts: list[str] = []
        for message in value:
            if isinstance(message, dict):
                role = str(message.get("role") or "user").strip().lower()
                content = message.get("content") or ""
                if isinstance(content, str) and content.strip():
                    parts.append(content.strip())
                elif content:
                    parts.append(str(content))
            elif str(message).strip():
                parts.append(str(message).strip())
        return "\n\n".join(parts).strip()
    return str(value or "").strip()


def string_choices(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def mcq_prompt(question: str, choices: list[str]) -> str:
    labels = [chr(ord("A") + offset) for offset in range(len(choices))]
    rendered = "\n".join(f"{label}. {choice}" for label, choice in zip(labels, choices, strict=True))
    return f"{question.strip()}\n\nOptions:\n{rendered}\n\nSelect the best option and give its letter as the final answer."


def native_factory(module_name: str, factory_name: str) -> Callable[..., Any]:
    module = importlib.import_module(module_name)
    factory = getattr(module, factory_name, None)
    if not callable(factory):
        raise TypeError(f"Expected callable {factory_name!r} in {module_name}; found {type(factory).__name__}")
    return factory


def native_group(payload: dict[str, Any], fallback: str) -> str:
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    for key in ("category", "subset", "subject", "task", "dataset_name", "config"):
        value = metadata.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return fallback


def materialize_native(dataset: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    task_provenance: list[dict[str, Any]] = []
    for group_name, module_name, factory_name in NATIVE_TASKS[dataset]:
        factory = native_factory(module_name, factory_name)
        signature = inspect.signature(factory)
        task = factory()
        source_samples = list(task.dataset)
        if not source_samples:
            raise ValueError(f"Native Inspect task {module_name}.{factory_name} yielded no samples")
        task_provenance.append(
            {
                "group": group_name,
                "factory_module": module_name,
                "factory_name": factory_name,
                "factory_signature": str(signature),
                "source_rows": len(source_samples),
            }
        )
        for position, sample in enumerate(source_samples):
            payload = sample_to_payload(sample)
            source_id = str(payload.get("id") or f"{group_name}_{position:06d}")
            question = text_from_inspect_input(payload.get("input"))
            choices = string_choices(payload.get("choices"))
            if not question:
                raise ValueError(f"{dataset}/{group_name}: empty native prompt at position={position}")
            if choices:
                prompt = mcq_prompt(question, choices)
                unit_type = "multiple_choice"
            elif dataset == "gsm8k":
                prompt = f"{question}\n\nSolve the problem and give the final numeric answer."
                unit_type = "numeric_reasoning"
            else:
                prompt = question
                unit_type = "single_turn"
            rows.append(
                {
                    "source_id": f"{group_name}::{source_id}",
                    "group": native_group(payload, group_name),
                    "prompt": prompt,
                    "unit_type": unit_type,
                    "source_meta": {
                        "source_kind": "native_inspect_task",
                        "native_task_group": group_name,
                        "native_sample_id": source_id,
                        "native_position": position,
                        "native_choices_count": len(choices),
                    },
                }
            )
    return rows, {"source_kind": "native_inspect_task", "tasks": task_provenance}


def hf_revision(repo_id: str) -> str:
    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN") or None
    info = HfApi().dataset_info(repo_id, token=token)
    revision = str(getattr(info, "sha", "") or "").strip()
    if not revision:
        raise RuntimeError(f"Could not resolve immutable revision for {repo_id}")
    return revision


def load_pinned_hf_split(repo: str, config: str | None, preferred_splits: tuple[str, ...]) -> tuple[Any, str, str]:
    from datasets import load_dataset

    revision = hf_revision(repo)
    token = os.environ.get("HF_TOKEN") or None
    dataset_dict = load_dataset(repo, config, revision=revision, token=token)
    available = list(dataset_dict.keys())
    split = next((candidate for candidate in preferred_splits if candidate in dataset_dict), None)
    if split is None:
        raise ValueError(f"{repo}: none of preferred splits {preferred_splits} available; found {available}")
    return dataset_dict[split], split, revision


def materialize_openbookqa() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    spec = HF_DATASETS["openbookqa"]
    dataset, split, revision = load_pinned_hf_split(spec["repo"], spec["config"], spec["preferred_splits"])
    rows: list[dict[str, Any]] = []
    for position, item in enumerate(dataset):
        row = dict(item)
        question = str(row.get("question_stem") or row.get("question") or "").strip()
        choices_obj = row.get("choices") if isinstance(row.get("choices"), dict) else {}
        labels = [str(value).strip() for value in choices_obj.get("label", [])]
        choices = [str(value).strip() for value in choices_obj.get("text", [])]
        if not question or len(choices) < 2:
            raise ValueError(f"OpenBookQA {split} row {position} missing question or choices")
        rows.append(
            {
                "source_id": str(row.get("id") or f"openbookqa_{split}_{position:06d}"),
                "group": "main",
                "prompt": mcq_prompt(question, choices),
                "unit_type": "multiple_choice",
                "source_meta": {"source_kind": "huggingface", "source_position": position, "choice_labels": labels},
            }
        )
    return rows, {"source_kind": "huggingface", "repo": spec["repo"], "config": spec["config"], "split": split, "revision": revision}


def materialize_strategyqa() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    spec = HF_DATASETS["strategyqa"]
    dataset, split, revision = load_pinned_hf_split(spec["repo"], spec["config"], spec["preferred_splits"])
    rows: list[dict[str, Any]] = []
    for position, item in enumerate(dataset):
        row = dict(item)
        question = str(row.get("question") or "").strip()
        if not question:
            raise ValueError(f"StrategyQA {split} row {position} has an empty question")
        rows.append(
            {
                "source_id": str(row.get("qid") or row.get("id") or f"strategyqa_{split}_{position:06d}"),
                "group": "binary_qa",
                "prompt": f"{question}\n\nAnswer Yes or No.",
                "unit_type": "binary_qa",
                "source_meta": {"source_kind": "huggingface", "source_position": position},
            }
        )
    return rows, {"source_kind": "huggingface", "repo": spec["repo"], "config": spec["config"], "split": split, "revision": revision}


def materialize_jailbreakbench(source_cache: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Freeze complete attack prompts, rather than bare harmful goals."""
    checkout = source_cache / "jailbreakbench_artifacts"
    if checkout.exists():
        head = subprocess.run(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if head != JAILBREAKBENCH_ARTIFACT_COMMIT:
            raise RuntimeError(
                f"{checkout} is at {head}, expected pinned JailbreakBench artifacts commit "
                f"{JAILBREAKBENCH_ARTIFACT_COMMIT}. Remove or correct that source cache first."
            )
    else:
        subprocess.run(["git", "clone", JAILBREAKBENCH_ARTIFACT_REPO, str(checkout)], check=True)
        subprocess.run(
            ["git", "-C", str(checkout), "checkout", "--detach", JAILBREAKBENCH_ARTIFACT_COMMIT],
            check=True,
        )

    artifact_path = checkout / JAILBREAKBENCH_ARTIFACT_PATH
    if not artifact_path.is_file():
        raise FileNotFoundError(f"Missing pinned JailbreakBench attack artifact: {artifact_path}")
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    artifacts = payload.get("jailbreaks")
    if not isinstance(artifacts, list) or len(artifacts) != 100:
        raise ValueError(f"Expected exactly 100 JailbreakBench attack artifacts; found {len(artifacts) if isinstance(artifacts, list) else type(artifacts).__name__}")

    rows: list[dict[str, Any]] = []
    for position, item in enumerate(artifacts):
        if not isinstance(item, dict):
            raise TypeError(f"JailbreakBench artifact at position {position} is not an object")
        prompt = str(item.get("prompt") or "").strip()
        if not prompt:
            raise ValueError(f"JailbreakBench artifact at position {position} has an empty attack prompt")
        index = item.get("index")
        source_id = str(index if index is not None else position)
        rows.append(
            {
                "source_id": f"jbc_manual_vicuna_13b_v1_5::{source_id}",
                "group": str(item.get("category") or "harmful"),
                "prompt": prompt,
                "unit_type": "harmful_jailbreak_attack",
                "source_meta": {
                    "source_kind": "pinned_github_attack_artifact",
                    "source_position": position,
                    "artifact_method": "JBC",
                    "artifact_attack_type": "manual",
                    "artifact_model": "vicuna-13b-v1.5",
                    "harmful_only": True,
                },
            }
        )
    return rows, {
        "source_kind": "pinned_github_attack_artifact",
        "repo": JAILBREAKBENCH_ARTIFACT_REPO,
        "commit": JAILBREAKBENCH_ARTIFACT_COMMIT,
        "artifact_path": str(JAILBREAKBENCH_ARTIFACT_PATH),
        "artifact_sha256": sha256_file(artifact_path),
        "artifact_method": "JBC",
        "artifact_attack_type": "manual",
        "artifact_model": "vicuna-13b-v1.5",
        "harmful_only": True,
    }


def snapshot_sorry_bench(source_cache: Path, staged: str) -> tuple[list[Path], dict[str, Any]]:
    if staged:
        directory = Path(staged).expanduser().resolve()
        if not directory.is_dir():
            raise FileNotFoundError(f"Missing --sorry-bench-source-jsonl directory: {directory}")
        paths = sorted(path for path in directory.rglob(SORRY_BENCH_MUTATION_PATTERN) if path.name != SORRY_BENCH_BASE_FILE)
        if not paths:
            raise FileNotFoundError(f"No SORRY-Bench mutation JSONLs under {directory}")
        return paths, {
            "source_kind": "pre_staged_mutation_jsonls",
            "path": str(directory),
            "files": [{"name": path.name, "sha256": sha256_file(path)} for path in paths],
            "harmful_only": True,
        }

    from huggingface_hub import snapshot_download

    revision = hf_revision(SORRY_BENCH_REPO)
    destination = source_cache / "sorry_bench"
    token = os.environ.get("HF_TOKEN") or None
    if token is None:
        raise RuntimeError(
            "HF_TOKEN is required to access gated SORRY-Bench. Accept its Hugging Face "
            "terms, export HF_TOKEN, then rerun preparation."
        )
    snapshot_download(
        repo_id=SORRY_BENCH_REPO,
        repo_type="dataset",
        revision=revision,
        local_dir=str(destination),
        token=token,
        allow_patterns=[SORRY_BENCH_MUTATION_PATTERN],
    )
    paths = sorted(path for path in destination.rglob(SORRY_BENCH_MUTATION_PATTERN) if path.name != SORRY_BENCH_BASE_FILE)
    if not paths:
        raise FileNotFoundError(f"No SORRY-Bench mutation JSONLs found under {destination}")
    return paths, {
        "source_kind": "huggingface_snapshot",
        "repo": SORRY_BENCH_REPO,
        "revision": revision,
        "path": str(destination),
        "files": [{"name": path.name, "sha256": sha256_file(path)} for path in paths],
        "harmful_only": True,
    }


def materialize_sorry_bench(source_cache: Path, staged: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    paths, provenance = snapshot_sorry_bench(source_cache, staged)
    rows: list[dict[str, Any]] = []
    for path in paths:
        mutation = path.stem.removeprefix("question_")
        for position, row in enumerate(read_jsonl(path)):
            raw_turns = row.get("turns")
            turns = [str(value).strip() for value in raw_turns] if isinstance(raw_turns, list) else []
            turns = [turn for turn in turns if turn]
            prompt = str(row.get("question") or row.get("prompt") or (turns[-1] if turns else "")).strip()
            if not prompt:
                raise ValueError(f"SORRY-Bench {path.name} row {position} has no prompt")
            messages = [{"role": "user", "content": turn} for turn in turns] or [{"role": "user", "content": prompt}]
            question_id = str(row.get("question_id") or row.get("id") or f"sorry_bench_{position:06d}")
            rows.append(
                {
                    "source_id": f"{path.stem}::{question_id}",
                    # The group is intentionally the mutation form: the 100-row
                    # draw is distributed across forms, rather than taking 100
                    # rows from every form.
                    "group": mutation,
                    "prompt": messages[-1]["content"],
                    "base_messages": messages,
                    "unit_type": "harmful_attack_mutation" if len(messages) == 1 else "harmful_attack_mutation_multiturn",
                    "source_meta": {
                        "source_kind": provenance["source_kind"],
                        "source_position": position,
                        "source_file": path.name,
                        "mutation": mutation,
                        "category": row.get("category"),
                        "prompt_style": row.get("prompt_style"),
                        "harmful_only": True,
                    },
                }
            )
    return rows, provenance


def source_rows(dataset: str, source_cache: Path, sorry_staged: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if dataset in NATIVE_TASKS:
        return materialize_native(dataset)
    if dataset == "openbookqa":
        return materialize_openbookqa()
    if dataset == "strategyqa":
        return materialize_strategyqa()
    if dataset == "jailbreakbench":
        return materialize_jailbreakbench(source_cache)
    if dataset == "sorry_bench":
        return materialize_sorry_bench(source_cache, sorry_staged)
    raise ValueError(f"Unhandled dataset {dataset!r}")


def balanced_sample(rows: list[dict[str, Any]], count: int, seed: int) -> tuple[list[dict[str, Any]], dict[str, int]]:
    if len({str(row["source_id"]) for row in rows}) != len(rows):
        raise ValueError("Source pool contains duplicate source IDs")
    if count > len(rows):
        raise ValueError(f"Requested {count} rows but source pool has only {len(rows)}")
    grouped: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for position, row in enumerate(rows):
        grouped[str(row.get("group") or "ungrouped")].append((position, row))
    groups = sorted(grouped)
    # A single-group source is an ordinary uniform draw. For multiple groups,
    # distribute the requested total as evenly as capacity permits.
    allocations = {group: 0 for group in groups}
    remaining = count
    while remaining:
        eligible = [group for group in groups if allocations[group] < len(grouped[group])]
        if not eligible:
            raise ValueError("Unable to allocate requested balanced sample")
        minimum = min(allocations[group] for group in eligible)
        candidates = [group for group in eligible if allocations[group] == minimum]
        group = candidates[(count - remaining) % len(candidates)]
        allocations[group] += 1
        remaining -= 1
    selected_positions: list[int] = []
    for group in groups:
        selected_positions.extend(
            position
            for position, _ in random.Random(stable_seed(seed, "selection", group)).sample(grouped[group], allocations[group])
        )
    selected_positions.sort()
    return [rows[position] for position in selected_positions], {group: allocations[group] for group in groups if allocations[group]}


def canonical_base_row(dataset: str, row: dict[str, Any], ordinal: int) -> dict[str, Any]:
    source_id = str(row["source_id"])
    return {
        "id": f"{dataset}::{source_id}",
        "source_type": "evaluation",
        "source_dataset": dataset,
        "group": str(row.get("group") or "ungrouped"),
        "unit_type": str(row["unit_type"]),
        "prompt": str(row["prompt"]),
        "base_messages": json_safe(row.get("base_messages") or [{"role": "user", "content": str(row["prompt"])}]),
        "system_overlay_mode": "prepend_system",
        "meta": {
            "experiment_role": "fresh_benchmark_ea_elicitation",
            "source_id": source_id,
            "source_selection_ordinal": ordinal,
            "source": json_safe(row["source_meta"]),
        },
    }


def expand_replicas(base_rows: list[dict[str, Any]], *, replicates: int, sample_seed: int) -> list[dict[str, Any]]:
    expanded: list[dict[str, Any]] = []
    for base in base_rows:
        for replicate_index in range(replicates):
            row = json.loads(json.dumps(base, ensure_ascii=False))
            base_id = str(base["id"])
            row["id"] = f"{base_id}::rep_{replicate_index:02d}"
            row["generation_seed"] = stable_seed(sample_seed, base_id, replicate_index)
            meta = row["meta"]
            meta["base_prompt_id"] = base_id
            meta["replicate_index"] = replicate_index
            meta["replicate_count"] = replicates
            expanded.append(row)
    return expanded


def main() -> int:
    args = parse_args()
    datasets = selected_datasets(args.datasets)
    if args.rows_per_dataset < 1 or args.replicates < 1:
        raise ValueError("--rows-per-dataset and --replicates must both be positive")
    run_root = Path(args.run_root)
    inputs_root = run_root / "inputs"
    manifest_path = inputs_root / "manifest.json"
    if manifest_path.exists() and not args.overwrite:
        raise FileExistsError(f"Frozen manifest already exists: {manifest_path}; use a new numbered run root")
    if args.overwrite and any((run_root / "runs").rglob("*.jsonl")):
        raise RuntimeError("Refusing to overwrite frozen inputs after generation outputs exist; create a new numbered run root")

    # A pre-generation correction can safely refresh just the affected source
    # directories. Keep every untouched frozen input and its original manifest
    # record byte-for-byte rather than re-materialising unrelated datasets from
    # a potentially newer upstream revision.
    existing_manifest: dict[str, Any] | None = None
    if manifest_path.exists() and args.overwrite:
        loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict) or not isinstance(loaded.get("datasets"), dict):
            raise ValueError(f"Existing manifest is malformed: {manifest_path}")
        existing_manifest = loaded
        retained = sorted(set(loaded["datasets"]) - set(datasets))
        for dataset in retained:
            base_file = inputs_root / dataset / str(loaded["datasets"][dataset].get("base_samples_file") or "base_samples.jsonl")
            expanded_file = inputs_root / dataset / str(loaded["datasets"][dataset].get("samples_file") or "samples.jsonl")
            if not base_file.is_file() or not expanded_file.is_file():
                raise FileNotFoundError(
                    f"Cannot preserve {dataset}: missing frozen input file(s) {base_file} / {expanded_file}"
                )
        print(f"[preserve] retaining untouched frozen datasets: {', '.join(retained) or 'none'}", flush=True)

    inspect_cache = Path(args.inspect_evals_cache_dir) if args.inspect_evals_cache_dir else run_root / "cache" / "inspect_evals"
    source_cache = Path(args.source_cache) if args.source_cache else run_root / "cache" / "sources"
    inputs_root.mkdir(parents=True, exist_ok=True)
    source_cache.mkdir(parents=True, exist_ok=True)
    setup_cluster_safe_hf(args, inspect_cache)

    if existing_manifest is not None:
        manifest = existing_manifest
        manifest.setdefault("preparation_events", []).append(
            {
                "at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "action": "pre_generation_partial_overwrite",
                "refreshed_datasets": datasets,
                "reason": "source-form correction before transcript generation",
                "package_versions": package_versions(),
            }
        )
    else:
        manifest = {
            "campaign": f"{run_root.parent.name}/{run_root.name}",
            "purpose": "raw_base_model_verbalised_evaluation_awareness_elicitation",
            "datasets": {},
            "datasets_requested": datasets,
            "rows_per_dataset": args.rows_per_dataset,
            "replicates_per_source_prompt": args.replicates,
            "sample_seed": args.sample_seed,
            "system_prompt_overlay": None,
            "source_prompt_policy": "fresh_upstream_samples; no SFT/DPO prepared rows; no gold targets in model messages",
            "package_versions": package_versions(),
            "inspect_evals_cache_dir": str(inspect_cache),
            "source_cache": str(source_cache),
            "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }

    for dataset in datasets:
        print(f"[materialize] dataset={dataset}", flush=True)
        candidates, provenance = source_rows(dataset, source_cache, args.sorry_bench_source_jsonl)
        selected, counts_by_group = balanced_sample(candidates, args.rows_per_dataset, stable_seed(args.sample_seed, dataset))
        base_rows = [canonical_base_row(dataset, row, ordinal) for ordinal, row in enumerate(selected)]
        expanded = expand_replicas(base_rows, replicates=args.replicates, sample_seed=args.sample_seed)
        if len(base_rows) != args.rows_per_dataset or len(expanded) != args.rows_per_dataset * args.replicates:
            raise AssertionError(f"{dataset}: unexpected base/expanded row counts")
        if len({row["id"] for row in expanded}) != len(expanded):
            raise AssertionError(f"{dataset}: duplicate expanded IDs")

        dataset_dir = inputs_root / dataset
        base_path = dataset_dir / "base_samples.jsonl"
        expanded_path = dataset_dir / "samples.jsonl"
        examples_path = dataset_dir / "prompt_examples.json"
        selection_path = dataset_dir / "selection.json"
        write_jsonl(base_path, base_rows)
        write_jsonl(expanded_path, expanded)
        examples_path.write_text(
            json.dumps(
                [
                    {"id": row["id"], "group": row["group"], "messages": row["base_messages"]}
                    for row in base_rows[:3]
                ],
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        selection = {
            "method": "balanced_by_source_group_without_replacement",
            "candidate_rows": len(candidates),
            "selected_base_rows": len(base_rows),
            "expanded_rows": len(expanded),
            "replicates_per_source_prompt": args.replicates,
            "sample_seed": args.sample_seed,
            "counts_by_group": counts_by_group,
            "source_provenance": provenance,
        }
        selection_path.write_text(json.dumps(selection, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        manifest["datasets"][dataset] = {
            **selection,
            "base_samples_file": base_path.name,
            "base_samples_sha256": sha256_file(base_path),
            "samples_file": expanded_path.name,
            "samples_sha256": sha256_file(expanded_path),
            "prompt_examples_file": examples_path.name,
        }
        print(f"[done] dataset={dataset} base={len(base_rows)} expanded={len(expanded)}", flush=True)

    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[all-done] manifest={manifest_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
