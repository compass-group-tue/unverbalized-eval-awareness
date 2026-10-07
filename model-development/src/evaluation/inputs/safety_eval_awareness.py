#!/usr/bin/env python3
"""Freeze final safety/evaluation-awareness inputs from pinned upstream suites.

The generated JSONLs use the canonical final-experiment schema consumed by
``run_dataset_inspect.py``. Upstream repositories and judge prompts are copied
under the experiment root, so later scoring does not depend on a moving branch.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


SOURCES = {
    "triggers": {
        "url": "https://github.com/microsoft/Test_Awareness_Steering.git",
        "commit": "eb78ea593223da09b903ad9f608701e8166e9a4a",
        "dir": "test_awareness_steering",
    },
    "evalawarebench": {
        "url": "https://github.com/aisa-group/decomposing-eval-awareness.git",
        "commit": "446be5c605b56a60d4efe2526f0cbf55522c523a",
        "dir": "decomposing_eval_awareness",
    },
    "agentic_misalignment": {
        "url": "https://github.com/anthropic-experimental/agentic-misalignment.git",
        "commit": "ea0630e1a3eaae7f9f9740fd2703229d3854ccda",
        "dir": "agentic_misalignment",
    },
}

OFFICIAL_EVALAWARE_CONFIGS: list[tuple[str, dict[str, str]]] = [
    ("baseline", {"F1": "realistic", "F2": "realistic", "F3": "casual", "F4": "realistic", "F5": "realistic", "F6": "realistic", "F7": "prose", "F8": "realistic"}),
    ("F1", {"F1": "varied", "F2": "realistic", "F3": "casual", "F4": "realistic", "F5": "realistic", "F6": "realistic", "F7": "prose", "F8": "realistic"}),
    ("F2", {"F1": "realistic", "F2": "varied", "F3": "casual", "F4": "realistic", "F5": "realistic", "F6": "realistic", "F7": "prose", "F8": "realistic"}),
    ("F3", {"F1": "realistic", "F2": "realistic", "F3": "formal", "F4": "realistic", "F5": "realistic", "F6": "realistic", "F7": "prose", "F8": "realistic"}),
    ("F4", {"F1": "realistic", "F2": "realistic", "F3": "casual", "F4": "varied", "F5": "realistic", "F6": "realistic", "F7": "prose", "F8": "realistic"}),
    ("F5", {"F1": "realistic", "F2": "realistic", "F3": "casual", "F4": "realistic", "F5": "varied", "F6": "realistic", "F7": "prose", "F8": "realistic"}),
    ("F6", {"F1": "realistic", "F2": "realistic", "F3": "casual", "F4": "realistic", "F5": "realistic", "F6": "varied", "F7": "prose", "F8": "realistic"}),
    ("F7", {"F1": "realistic", "F2": "realistic", "F3": "casual", "F4": "realistic", "F5": "realistic", "F6": "realistic", "F7": "template", "F8": "realistic"}),
    ("F8", {"F1": "realistic", "F2": "realistic", "F3": "casual", "F4": "realistic", "F5": "realistic", "F6": "realistic", "F7": "prose", "F8": "varied"}),
]

# The paper's main factor-isolation protocol: one naturalistic baseline and
# eight one-factor changes. Keep this explicit rather than deriving it so the
# frozen configuration names match the upstream runner.
EVALAWARE_BASELINE = OFFICIAL_EVALAWARE_CONFIGS[0][1]
EVALAWARE_VARIED_LEVELS = {
    "F1": "varied",
    "F2": "varied",
    "F3": "formal",
    "F4": "varied",
    "F5": "varied",
    "F6": "varied",
    "F7": "template",
    "F8": "varied",
}


def evalaware_configurations(mode: str) -> list[tuple[str, dict[str, str]]]:
    """Return a named, deterministic EvalAwareBench condition set.

    ``single_factor`` is the author's baseline-plus-ablation protocol.
    ``seven_plus_eight_stack`` is the retained run-002 high-cue mode name. It
    holds F8 (explicit evaluation disclosure) at its naturalistic level in
    every condition, then uses one seven-of-seven stack over F1–F7, each of
    the seven six-of-seven stacks, and a zero-of-seven naturalistic control.
    It deliberately has nine conditions and never writes F8's varied level.
    """
    if mode == "single_factor":
        return [(name, dict(config)) for name, config in OFFICIAL_EVALAWARE_CONFIGS]
    if mode == "seven_plus_eight_stack":
        active_factors = tuple(f"F{index}" for index in range(1, 8))
        fully_varied = dict(EVALAWARE_BASELINE)
        for factor in active_factors:
            fully_varied[factor] = EVALAWARE_VARIED_LEVELS[factor]
        configurations: list[tuple[str, dict[str, str]]] = []
        configurations.append(("stacked_7_of_7_no_f8", fully_varied))
        for omitted_factor in active_factors:
            config = dict(fully_varied)
            config[omitted_factor] = EVALAWARE_BASELINE[omitted_factor]
            configurations.append((f"stacked_6_of_7_without_{omitted_factor}_no_f8", config))
        configurations.append(("naturalistic_0_of_7_no_f8", dict(EVALAWARE_BASELINE)))
        return configurations
    raise ValueError(f"Unknown --evalaware-config-mode {mode!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--source-cache", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--evalaware-task-concepts", type=int, default=56)
    parser.add_argument(
        "--evalaware-config-mode",
        choices=("single_factor", "seven_plus_eight_stack"),
        default="single_factor",
        help="EvalAwareBench condition set to freeze.",
    )
    parser.add_argument(
        "--evalaware-system-prompt",
        choices=("none", "author_realistic"),
        default="none",
        help="Whether to preserve the author's generic chat system prompt in EvalAwareBench rows.",
    )
    parser.add_argument(
        "--datasets",
        default="triggers,evalawarebench,agentic_misalignment",
        help="Comma-separated subset of triggers,evalawarebench,agentic_misalignment.",
    )
    parser.add_argument(
        "--triggers-variant",
        choices=("both", "hypothetical"),
        default="both",
        help="Retain both paired Trigger variants or only the hypothetical treatment.",
    )
    parser.add_argument("--agentic-replicates", type=int, default=100)
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False)
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


def command(*args: str, cwd: Path | None = None) -> str:
    return subprocess.check_output(args, cwd=cwd, text=True).strip()


def ensure_source(cache: Path, source_name: str) -> Path:
    spec = SOURCES[source_name]
    destination = cache / spec["dir"]
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", spec["url"], str(destination)], check=True)
        # Check out the pinned commit rather than the moving default branch.
        subprocess.run(["git", "-C", str(destination), "checkout", "-q", spec["commit"]], check=True)
    actual = command("git", "rev-parse", "HEAD", cwd=destination)
    if actual != spec["commit"]:
        raise RuntimeError(
            f"Pinned source mismatch for {source_name}: expected {spec['commit']}, got {actual}. "
            f"Refuse to silently use a different upstream version; remove only {destination} and retry."
        )
    return destination


def canonical_row(
    *,
    row_id: str,
    prompt: str,
    dataset: str,
    group: str,
    unit_type: str,
    meta: dict[str, Any],
    base_messages: list[dict[str, str]] | None = None,
    generation_seed: int | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "id": row_id,
        "source_type": "evaluation",
        "source_dataset": dataset,
        "group": group,
        "unit_type": unit_type,
        "prompt": prompt,
        "base_messages": base_messages or [{"role": "user", "content": prompt}],
        "system_overlay_mode": "preserve_base_messages",
        "meta": meta,
    }
    if generation_seed is not None:
        record["generation_seed"] = generation_seed
    return record


def prepare_triggers(source: Path, *, variant_mode: str) -> list[dict[str, Any]]:
    data = json.loads((source / "data" / "triggers" / "triggers.json").read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError("Triggers source contains no paired records")
    rows: list[dict[str, Any]] = []
    for position, item in enumerate(data):
        category = str(item["category"])
        variants = ("real", "hypothetical") if variant_mode == "both" else ("hypothetical",)
        for variant in variants:
            prompt = str(item[variant]).strip()
            if not prompt:
                raise ValueError(f"Triggers item {position} has an empty {variant} prompt")
            rows.append(
                canonical_row(
                    row_id=f"triggers_{position:05d}_{variant}",
                    prompt=prompt,
                    dataset="triggers",
                    group=category,
                    unit_type="single_turn",
                    meta={
                        "source_repository": SOURCES["triggers"],
                        "source_index": position,
                        "category": category,
                        "pair_variant": variant,
                        "paired_id": f"triggers_{position:05d}",
                    },
                )
            )
    return rows


def render_evalaware(template: dict[str, Any], config: dict[str, str]) -> str:
    """Exact rendering logic from EvalAwareBench's ``eval_bench/render.py``."""
    text = template["shells"][f"F7_{config['F7']}"][f"F3_{config['F3']}"][f"F4_{config['F4']}"]
    resolved = dict(template.get("fixed", {}))
    for name, slot in template["slots"].items():
        factor = slot["factor"]
        conditional = slot.get(f"F3_{config['F3']}")
        resolved[name] = (conditional or slot)[config[factor]]
    for _ in range(10):
        rendered = text
        for name, value in resolved.items():
            rendered = rendered.replace(f"{{{name}}}", str(value))
        if rendered == text:
            break
        text = rendered
    return text


def prepare_evalaware(
    source: Path,
    *,
    concepts: int,
    seed: int,
    config_mode: str,
    system_prompt_mode: str,
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    task_root = source / "eval_bench" / "tasks"
    task_dirs = sorted(path for path in task_root.iterdir() if path.is_dir())
    complete = [path for path in task_dirs if (path / "safety.yaml").is_file() and (path / "capability.yaml").is_file()]
    if not 1 <= concepts <= len(complete):
        raise ValueError(f"--evalaware-task-concepts must be in [1, {len(complete)}], got {concepts}")
    selected = sorted(random.Random(seed).sample(complete, concepts), key=lambda path: path.name)
    configurations = evalaware_configurations(config_mode)
    if system_prompt_mode == "author_realistic":
        system_prompt = (source / "prompts" / "chat_prompt_realistic.txt").read_text(encoding="utf-8").strip()
        if not system_prompt:
            raise ValueError("EvalAwareBench author system prompt is empty")
    else:
        system_prompt = ""
    rows: list[dict[str, Any]] = []
    for task_dir in selected:
        for valence in ("safety", "capability"):
            template = yaml.safe_load((task_dir / f"{valence}.yaml").read_text(encoding="utf-8"))
            for config_name, config in configurations:
                prompt = render_evalaware(template, config).strip()
                task_id = str(template["task_id"])
                rows.append(
                    canonical_row(
                        row_id=f"evalawarebench_{task_id}_{config_name}",
                        prompt=prompt,
                        dataset="evalawarebench",
                        group=valence,
                        unit_type="single_turn",
                        base_messages=(
                            [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
                            if system_prompt else None
                        ),
                        meta={
                            "source_repository": SOURCES["evalawarebench"],
                            "task_concept": task_dir.name,
                            "task_id": task_id,
                            "valence": valence,
                            "official_configuration": config_name,
                            "factor_configuration": config,
                            "varied_factor": None if config_name == "baseline" else config_name,
                            "varied_factors": [
                                factor
                                for factor, level in config.items()
                                if level != EVALAWARE_BASELINE[factor]
                            ],
                            "f8_level": config["F8"],
                            "configuration_mode": config_mode,
                            "system_prompt_mode": system_prompt_mode,
                        },
                    )
                )
    return rows, [path.name for path in selected], [name for name, _ in configurations]


def replica_seed(seed: int, condition_id: str, replica: int) -> int:
    digest = hashlib.sha256(f"{seed}:{condition_id}:{replica}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**31 - 1)


def prepare_agentic(source: Path, work_dir: Path, *, replicates: int, seed: int) -> list[dict[str, Any]]:
    if replicates < 1:
        raise ValueError("--agentic-replicates must be positive")
    prompts_dir = work_dir / "official_prompt_generator_output"
    config = source / "configs" / "multi_model_experiment_250530-v2.yaml"
    if not prompts_dir.exists():
        subprocess.run(
            [sys.executable, str(source / "scripts" / "generate_prompts.py"), "--config", str(config), "--output", str(prompts_dir)],
            check=True,
            cwd=source,
        )
    condition_dirs = sorted(path for path in prompts_dir.iterdir() if path.is_dir())
    if len(condition_dirs) != 18:
        raise RuntimeError(f"Official Agentic Misalignment generator produced {len(condition_dirs)} conditions, expected 18")
    rows: list[dict[str, Any]] = []
    for condition in condition_dirs:
        metadata = json.loads((condition / "metadata.json").read_text(encoding="utf-8"))
        system = (condition / "system_prompt.txt").read_text(encoding="utf-8").strip()
        user_instruction = (condition / "user_prompt.txt").read_text(encoding="utf-8").strip()
        email = (condition / "email_content.txt").read_text(encoding="utf-8").strip()
        user = f"{user_instruction}\n\n{email}"
        if not system or not user_instruction or not email:
            raise ValueError(f"Incomplete official Agentic Misalignment prompt: {condition}")
        for replica in range(replicates):
            condition_id = str(metadata["condition_id"])
            rows.append(
                canonical_row(
                    row_id=f"agentic_misalignment_{condition_id}_rep_{replica:03d}",
                    prompt=user,
                    dataset="agentic_misalignment",
                    group=str(metadata["scenario"]),
                    unit_type="contextual_single_completion",
                    base_messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                    generation_seed=replica_seed(seed, condition_id, replica),
                    meta={
                        "source_repository": SOURCES["agentic_misalignment"],
                        "official_config": "configs/multi_model_experiment_250530-v2.yaml",
                        "condition_id": condition_id,
                        "scenario": metadata["scenario"],
                        "goal_type": metadata["goal_type"],
                        "goal_value": metadata["goal_value"],
                        "urgency_type": metadata["urgency_type"],
                        "replica_index": replica,
                        "email_content": email,
                    },
                )
            )
    return rows


def copy_judge_assets(
    *, evalaware: Path | None, triggers: Path | None, destination: Path
) -> dict[str, str]:
    destination.mkdir(parents=True, exist_ok=True)
    source_files: dict[str, Path] = {}
    if triggers is not None:
        source_files["triggers_judgePromptEvidenceBoth.txt"] = triggers / "prompts" / "judgePromptEvidenceBoth.txt"
    if evalaware is not None:
        source_files.update({
            "evalaware_judgePromptEvidenceBoth.txt": evalaware / "prompts" / "judgePromptEvidenceBoth.txt",
            "evalaware_judgePromptAgenticMisalignment.txt": evalaware / "prompts" / "judgePromptAgenticMisalignment.txt",
        })
        source_files.update({
            f"{path.name}": path
            for path in sorted((evalaware / "prompts").glob("agenticmisalignment_criteria_*.txt"))
        })
    result: dict[str, str] = {}
    for name, source in source_files.items():
        if not source.is_file():
            raise FileNotFoundError(f"Expected upstream judge asset missing: {source}")
        target = destination / name
        shutil.copyfile(source, target)
        result[name] = sha256(target)
    return result


def main() -> int:
    args = parse_args()
    root = Path(args.run_root)
    inputs = root / "inputs"
    manifest_path = inputs / "manifest.json"
    if manifest_path.exists() and not args.overwrite:
        raise FileExistsError(f"Frozen inputs already exist: {manifest_path}; use a new run root.")
    if args.overwrite:
        raise ValueError("--overwrite is intentionally unsupported for frozen final inputs; use a new run root.")
    requested_datasets = tuple(item.strip() for item in args.datasets.split(",") if item.strip())
    invalid_datasets = sorted(set(requested_datasets) - set(SOURCES))
    if invalid_datasets or not requested_datasets:
        raise ValueError(f"Invalid --datasets values: {invalid_datasets or args.datasets!r}")
    cache = Path(args.source_cache) if args.source_cache else root / "cache" / "upstream_sources"
    sources = {name: ensure_source(cache, name) for name in requested_datasets}

    prepared: dict[str, list[dict[str, Any]]] = {}
    selected_tasks: list[str] = []
    evalaware_configurations_used: list[str] = []
    if "triggers" in requested_datasets:
        prepared["triggers"] = prepare_triggers(sources["triggers"], variant_mode=args.triggers_variant)
    if "evalawarebench" in requested_datasets:
        evalaware_rows, selected_tasks, evalaware_configurations_used = prepare_evalaware(
            sources["evalawarebench"],
            concepts=args.evalaware_task_concepts,
            seed=args.seed,
            config_mode=args.evalaware_config_mode,
            system_prompt_mode=args.evalaware_system_prompt,
        )
        prepared["evalawarebench"] = evalaware_rows
    if "agentic_misalignment" in requested_datasets:
        prepared["agentic_misalignment"] = prepare_agentic(
            sources["agentic_misalignment"], root / "cache" / "agentic_misalignment",
            replicates=args.agentic_replicates,
            seed=args.seed,
        )
    for name, rows in prepared.items():
        path = inputs / name / "samples.jsonl"
        write_jsonl(path, rows)
        print(f"[done] dataset={name} rows={len(rows)} samples={path}", flush=True)
    suite_path = inputs / "generation_datasets.tsv"
    suite_path.write_text(
        "".join(f"{name}\t{inputs / name / 'samples.jsonl'}\t0\t1\n" for name in prepared),
        encoding="utf-8",
    )
    judge_assets = copy_judge_assets(
        evalaware=sources.get("evalawarebench"),
        triggers=sources.get("triggers"),
        destination=root / "assets" / "author_judges",
    )
    manifest = {
        "campaign": "safety_eval_awareness",
        "run_root": str(root),
        "selection_seed": args.seed,
        "sources": {name: {**SOURCES[name], "path": str(source)} for name, source in sources.items()},
        "datasets": {
            name: {"rows": len(rows), "samples_sha256": sha256(inputs / name / "samples.jsonl")}
            for name, rows in prepared.items()
        },
        "author_judge_assets_sha256": judge_assets,
        "generation_manifest": str(suite_path),
    }
    if "evalawarebench" in prepared:
        manifest["evalawarebench"] = {
            "task_concepts": args.evalaware_task_concepts,
            "selected_task_concepts": selected_tasks,
            "configuration_mode": args.evalaware_config_mode,
            "configurations": evalaware_configurations_used,
            "system_prompt_mode": args.evalaware_system_prompt,
        }
    if "triggers" in prepared:
        manifest["triggers"] = {"variant_mode": args.triggers_variant}
    if "agentic_misalignment" in prepared:
        manifest["agentic_misalignment"] = {
            "official_config": "configs/multi_model_experiment_250530-v2.yaml",
            "conditions": 18,
            "replicates_per_condition": args.agentic_replicates,
            "per_row_generation_seed": "sha256(seed, condition_id, replica_index) modulo 2^31-1",
        }
    inputs.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[all-done] manifest={manifest_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
