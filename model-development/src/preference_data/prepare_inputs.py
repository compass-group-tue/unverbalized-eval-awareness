#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


STANDARD_SINGLE_TURN_DATASETS = (
    "gsm8k",
    "bbq",
    "wmdp",
    "openbookqa",
    "agieval",
    "strategyqa",
)
REMAINING_STAGE1_DATASETS = (
    "jailbreakbench",
    "persistbench",
    "sorry_bench",
    "mt_bench_101",
    "api_bank",
)
WILDCHAT_DEPLOYMENT_DATASET = "wildchat_deployment"
WILDCHAT_DEPLOYMENT_DEFAULT_DIR = "rlhf_deployment_wildchat_4_8m_v1"
ALL_STAGE1_DATASETS = STANDARD_SINGLE_TURN_DATASETS + REMAINING_STAGE1_DATASETS
PHASE2_EVAL_DATASETS = (
    "gsm8k",
    "bbq",
    "wmdp",
    "openbookqa",
    "agieval",
    "strategyqa",
    "sorry_bench",
    "mt_bench_101",
)
VALID_MESSAGE_ROLES = {"system", "user", "assistant"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Prepare RLHF Stage 1 run inputs under training_data/rlhf by "
            "materializing canonical per-dataset JSONLs plus a merged file."
        )
    )
    p.add_argument("--job-name", required=True)
    p.add_argument("--eval-root", required=True)
    p.add_argument("--out-root", required=True)
    p.add_argument(
        "--datasets",
        default="standard_single_turn",
        help=(
            "Comma-separated dataset keys, or the presets 'standard_single_turn', "
            "'stage1_remaining', or 'stage1_all'."
        ),
    )
    p.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Overwrite existing prepared dataset files under the job root.",
    )
    p.add_argument(
        "--wildchat-deployment-dir-name",
        default=WILDCHAT_DEPLOYMENT_DEFAULT_DIR,
        help="Subdirectory under data/evaluations/wildchat/ containing downloaded deployment files.",
    )
    p.add_argument(
        "--source-download-subdir",
        default="stage1_rlhf_download",
        help="Per-dataset source subdir under data/evaluations/<dataset>/ to prepare from.",
    )
    p.add_argument(
        "--prepared-subdir",
        default="",
        help="Optional subdir under job inputs/manifests/outputs for keeping prepared data separate, e.g. 'phase2'.",
    )
    return p.parse_args()


def choose_datasets(raw: str) -> list[str]:
    token = raw.strip().lower()
    if token == "standard_single_turn":
        return list(STANDARD_SINGLE_TURN_DATASETS)
    if token == "stage1_remaining":
        return list(REMAINING_STAGE1_DATASETS)
    if token == "stage1_all":
        return list(ALL_STAGE1_DATASETS)
    if token == "phase2_eval":
        return list(PHASE2_EVAL_DATASETS)
    out = [x.strip().lower() for x in raw.split(",") if x.strip()]
    if not out:
        raise ValueError("No datasets selected.")
    valid = list(ALL_STAGE1_DATASETS) + list(PHASE2_EVAL_DATASETS) + [WILDCHAT_DEPLOYMENT_DATASET]
    bad = [x for x in out if x not in valid]
    if bad:
        raise ValueError(
            "Unsupported dataset(s) for Stage 1 preparer: "
            f"{bad}. Supported: {valid}"
        )
    return out


def read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            yield line_no, json.loads(line)


def ensure_messages(messages: Any, *, path: Path, line_no: int) -> list[dict[str, str]]:
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"{path}:{line_no} missing non-empty messages list")
    out: list[dict[str, str]] = []
    for idx, message in enumerate(messages, start=1):
        if not isinstance(message, dict):
            raise ValueError(f"{path}:{line_no} message {idx} is not an object")
        role = str(message.get("role") or "").strip()
        content = str(message.get("content") or "").strip()
        if role not in VALID_MESSAGE_ROLES:
            raise ValueError(f"{path}:{line_no} message {idx} invalid role: {role!r}")
        if not content:
            raise ValueError(f"{path}:{line_no} message {idx} missing content")
        out.append({"role": role, "content": content})
    return out


def build_prepared_row(
    row: dict[str, Any],
    *,
    dataset_name: str,
    group_name: str,
    source_jsonl: Path,
    source_manifest: Path,
    line_no: int,
) -> dict[str, Any]:
    row_id = str(row.get("id") or "").strip()
    if not row_id:
        raise ValueError(f"{source_jsonl}:{line_no} missing id")
    source_dataset = str(row.get("source_dataset") or "").strip()
    if source_dataset != dataset_name:
        raise ValueError(
            f"{source_jsonl}:{line_no} source_dataset mismatch: "
            f"expected {dataset_name!r}, got {source_dataset!r}"
        )
    unit_type = str(row.get("unit_type") or "").strip()
    if not unit_type:
        raise ValueError(f"{source_jsonl}:{line_no} missing unit_type")
    prompt = str(row.get("prompt") or "").strip()
    messages = ensure_messages(row.get("messages"), path=source_jsonl, line_no=line_no)
    meta = row.get("meta")
    if not isinstance(meta, dict):
        meta = {}

    return {
        "id": row_id,
        "source_type": str(row.get("source_type") or "evaluation"),
        "source_dataset": source_dataset,
        "group": group_name,
        "unit_type": unit_type,
        "prompt": prompt,
        "base_messages": messages,
        "system_overlay_mode": "prepend_system",
        "meta": {
            **meta,
            "source_prompt": prompt,
            "source_selected_jsonl": str(source_jsonl),
            "source_manifest_json": str(source_manifest),
            "prepared_from_line_number": line_no,
        },
    }


def ensure_write_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_jsonl(path: Path, rows: list[dict[str, Any]], *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file without --overwrite: {path}")
    ensure_write_parent(path)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, obj: dict[str, Any], *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file without --overwrite: {path}")
    ensure_write_parent(path)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_job_dirs(job_root: Path, *, prepared_subdir: str = "") -> dict[str, Path]:
    subparts = [part for part in [prepared_subdir.strip("/")] if part]
    inputs_root = job_root / "inputs"
    manifests_root = job_root / "manifests"
    outputs_root = job_root / "outputs"
    for part in subparts:
        inputs_root = inputs_root / part
        manifests_root = manifests_root / part
        outputs_root = outputs_root / part
    paths = {
        "job_root": job_root,
        "inputs_root": inputs_root,
        "manifests_root": manifests_root,
        "outputs_root": outputs_root,
        "inputs_by_dataset": inputs_root / "by_dataset",
        "inputs_merged": inputs_root / "merged",
        "inputs_deployment_wildchat": job_root / "inputs" / "deployment" / "wildchat",
        "manifests": manifests_root,
        "outputs_inspect_no_system": outputs_root / "inspect" / "no_system",
        "outputs_inspect_sys_x100": outputs_root / "inspect" / "sys_x100",
        "outputs_merged": outputs_root / "merged",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def dedupe_prepared_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    seen_ids: set[str] = set()
    deduped: list[dict[str, Any]] = []
    duplicate_count = 0
    for row in rows:
        row_id = str(row.get("id") or "").strip()
        if row_id in seen_ids:
            duplicate_count += 1
            continue
        seen_ids.add(row_id)
        deduped.append(row)
    return deduped, duplicate_count


def count_jsonl_rows(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            if raw.strip():
                count += 1
    return count


def read_prepared_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise ValueError(f"{path}:{line_no} row is not an object")
            rows.append(obj)
    return rows


def load_existing_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = read_json(path)
    return data


def build_merged_rows_from_existing_inputs(inputs_by_dataset_dir: Path) -> tuple[list[dict[str, Any]], list[str]]:
    merged_rows: list[dict[str, Any]] = []
    included_datasets: list[str] = []
    for dataset_name in ALL_STAGE1_DATASETS:
        path = inputs_by_dataset_dir / f"{dataset_name}.jsonl"
        if not path.exists():
            continue
        merged_rows.extend(read_prepared_jsonl(path))
        included_datasets.append(dataset_name)
    return merged_rows, included_datasets


def collect_source_variant_paths(source_manifest_obj: dict[str, Any]) -> dict[str, Path]:
    selection_outputs = source_manifest_obj.get("selection_outputs")
    if isinstance(selection_outputs, dict) and selection_outputs:
        out: dict[str, Path] = {}
        for key, value in selection_outputs.items():
            if not isinstance(value, str) or not value.strip():
                continue
            out[str(key)] = Path(value).expanduser()
        if out:
            return out

    selection_output = str(source_manifest_obj.get("selection_output") or "").strip()
    if not selection_output:
        raise ValueError("Source manifest missing selection_output / selection_outputs")
    return {"combined": Path(selection_output).expanduser()}


def prepare_dataset_variant_rows(
    *,
    dataset_name: str,
    group_name: str,
    source_jsonl: Path,
    source_manifest: Path,
) -> tuple[list[dict[str, Any]], int, int]:
    prepared_rows: list[dict[str, Any]] = []
    invalid_row_count = 0
    for line_no, row in iter_jsonl(source_jsonl):
        if not isinstance(row, dict):
            raise ValueError(f"{source_jsonl}:{line_no} row is not an object")
        try:
            prepared_rows.append(
                build_prepared_row(
                    row,
                    dataset_name=dataset_name,
                    group_name=group_name,
                    source_jsonl=source_jsonl,
                    source_manifest=source_manifest,
                    line_no=line_no,
                )
            )
        except ValueError as exc:
            invalid_row_count += 1
            print(
                f"[warn] dataset={dataset_name} skipping invalid row at "
                f"{source_jsonl}:{line_no}: {exc}",
                file=sys.stderr,
                flush=True,
            )
    prepared_rows, duplicate_count = dedupe_prepared_rows(prepared_rows)
    return prepared_rows, invalid_row_count, duplicate_count


def prepare_wildchat_deployment(
    *,
    args: argparse.Namespace,
    eval_root: Path,
    job_dirs: dict[str, Path],
) -> int:
    source_dir = eval_root / "wildchat" / args.wildchat_deployment_dir_name
    source_manifest = source_dir / "manifest.json"
    if not source_manifest.exists():
        raise FileNotFoundError(f"Missing WildChat deployment manifest: {source_manifest}")

    manifest_obj = read_json(source_manifest)
    outputs = manifest_obj.get("outputs")
    if not isinstance(outputs, dict):
        raise ValueError(f"WildChat deployment manifest missing outputs object: {source_manifest}")

    source_paths = {
        "combined": Path(str(outputs.get("combined_path") or "")).expanduser(),
        "no_system": Path(str(outputs.get("no_system_path") or "")).expanduser(),
        "sys_x100": Path(str(outputs.get("sys_x100_path") or "")).expanduser(),
    }
    for key, path in source_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing WildChat deployment source for {key}: {path}")

    prepared_outputs = {
        "combined": job_dirs["inputs_deployment_wildchat"] / "combined.jsonl",
        "no_system": job_dirs["inputs_deployment_wildchat"] / "no_system.jsonl",
        "sys_x100": job_dirs["inputs_deployment_wildchat"] / "sys_x100.jsonl",
    }

    variant_counts: dict[str, int] = {}
    for variant_key in ("combined", "no_system", "sys_x100"):
        prepared_rows: list[dict[str, Any]] = []
        invalid_row_count = 0
        source_jsonl = source_paths[variant_key]
        for line_no, row in iter_jsonl(source_jsonl):
            if not isinstance(row, dict):
                raise ValueError(f"{source_jsonl}:{line_no} row is not an object")
            try:
                prepared_rows.append(
                    build_prepared_row(
                        row,
                        dataset_name="wildchat",
                        group_name="deployment_single_turn",
                        source_jsonl=source_jsonl,
                        source_manifest=source_manifest,
                        line_no=line_no,
                    )
                )
            except ValueError as exc:
                invalid_row_count += 1
                print(
                    f"[warn] dataset=wildchat_deployment variant={variant_key} skipping invalid row at "
                    f"{source_jsonl}:{line_no}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )

        prepared_rows, duplicate_count = dedupe_prepared_rows(prepared_rows)
        out_path = prepared_outputs[variant_key]
        write_jsonl(out_path, prepared_rows, overwrite=args.overwrite)
        variant_counts[variant_key] = len(prepared_rows)
        print(
            f"[done] dataset=wildchat_deployment variant={variant_key} rows={len(prepared_rows)} out={out_path}",
            file=sys.stderr,
            flush=True,
        )
        if invalid_row_count:
            print(
                f"[warn] dataset=wildchat_deployment variant={variant_key} invalid_rows_removed={invalid_row_count}",
                file=sys.stderr,
                flush=True,
            )
        if duplicate_count:
            print(
                f"[warn] dataset=wildchat_deployment variant={variant_key} duplicate_id_rows_removed={duplicate_count}",
                file=sys.stderr,
                flush=True,
            )

    deployment_manifest_path = job_dirs["manifests"] / "prepare_stage1_deployment_wildchat_manifest.json"
    deployment_manifest = {
        "job_name": args.job_name,
        "dataset": "wildchat_deployment",
        "group": "deployment_single_turn",
        "source_manifest": str(source_manifest),
        "source_dir": str(source_dir),
        "prepared_outputs": {key: str(value) for key, value in prepared_outputs.items()},
        "row_counts": variant_counts,
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "schema": {
            "top_level_fields": [
                "id",
                "source_type",
                "source_dataset",
                "group",
                "unit_type",
                "prompt",
                "base_messages",
                "system_overlay_mode",
                "meta",
            ],
            "group_name": "deployment_single_turn",
            "system_overlay_mode": "prepend_system",
        },
    }
    write_json(deployment_manifest_path, deployment_manifest, overwrite=True)
    print(f"[done] manifest={deployment_manifest_path}", file=sys.stderr, flush=True)
    return 0


def prepare_eval_dataset_set(
    *,
    args: argparse.Namespace,
    datasets: list[str],
    eval_root: Path,
    job_dirs: dict[str, Path],
    manifest_filename: str,
    group_name: str,
    preserve_legacy_stage1_layout: bool,
) -> int:
    manifest_path = job_dirs["manifests"] / manifest_filename
    existing_manifest = load_existing_manifest(manifest_path)
    existing_dataset_entries = {
        str(entry.get("dataset_name")): entry
        for entry in existing_manifest.get("datasets", [])
        if isinstance(entry, dict) and entry.get("dataset_name")
    }

    merged_rows_by_variant: dict[str, list[dict[str, Any]]] = {
        "combined": [],
        "strict": [],
        "relaxed": [],
    }
    merged_dataset_names_by_variant: dict[str, list[str]] = {
        "combined": [],
        "strict": [],
        "relaxed": [],
    }

    for dataset_name in datasets:
        source_manifest = eval_root / dataset_name / args.source_download_subdir / "manifest.json"
        if not source_manifest.exists():
            raise FileNotFoundError(f"Missing source manifest for {dataset_name}: {source_manifest}")
        source_manifest_obj = read_json(source_manifest)
        source_variant_paths = collect_source_variant_paths(source_manifest_obj)

        dataset_dir = job_dirs["inputs_by_dataset"] / dataset_name
        variant_outputs: dict[str, str] = {}
        variant_counts: dict[str, int] = {}
        total_invalid_rows = 0
        total_duplicate_rows = 0

        for variant_key, source_jsonl in source_variant_paths.items():
            if not source_jsonl.exists():
                raise FileNotFoundError(
                    f"Missing selected source JSONL for {dataset_name} variant={variant_key}: {source_jsonl}"
                )
            prepared_rows, invalid_row_count, duplicate_count = prepare_dataset_variant_rows(
                dataset_name=dataset_name,
                group_name=group_name,
                source_jsonl=source_jsonl,
                source_manifest=source_manifest,
            )
            if preserve_legacy_stage1_layout and variant_key == "combined" and set(source_variant_paths) == {"combined"}:
                out_path = job_dirs["inputs_by_dataset"] / f"{dataset_name}.jsonl"
            else:
                out_path = dataset_dir / f"{variant_key}.jsonl"
            write_jsonl(out_path, prepared_rows, overwrite=args.overwrite)
            variant_outputs[variant_key] = str(out_path)
            variant_counts[variant_key] = len(prepared_rows)
            total_invalid_rows += invalid_row_count
            total_duplicate_rows += duplicate_count
            print(
                f"[done] dataset={dataset_name} variant={variant_key} rows={len(prepared_rows)} out={out_path}",
                file=sys.stderr,
                flush=True,
            )
            if invalid_row_count:
                print(
                    f"[warn] dataset={dataset_name} variant={variant_key} invalid_rows_removed={invalid_row_count}",
                    file=sys.stderr,
                    flush=True,
                )
            if duplicate_count:
                print(
                    f"[warn] dataset={dataset_name} variant={variant_key} duplicate_id_rows_removed={duplicate_count}",
                    file=sys.stderr,
                    flush=True,
                )
            if variant_key in merged_rows_by_variant:
                merged_rows_by_variant[variant_key].extend(prepared_rows)
                if dataset_name not in merged_dataset_names_by_variant[variant_key]:
                    merged_dataset_names_by_variant[variant_key].append(dataset_name)

        if "combined" not in source_variant_paths:
            fallback_variant = "strict" if "strict" in source_variant_paths else "relaxed" if "relaxed" in source_variant_paths else ""
            if fallback_variant:
                merged_rows_by_variant["combined"].extend(
                    read_prepared_jsonl(Path(variant_outputs[fallback_variant]))
                )
                if dataset_name not in merged_dataset_names_by_variant["combined"]:
                    merged_dataset_names_by_variant["combined"].append(dataset_name)

        existing_dataset_entries[dataset_name] = {
            "dataset_name": dataset_name,
            "group": group_name,
            "source_manifest": str(source_manifest),
            "source_download_subdir": args.source_download_subdir,
            "prepared_outputs": variant_outputs,
            "row_counts_by_variant": variant_counts,
            "invalid_rows_removed": total_invalid_rows,
            "duplicate_id_rows_removed": total_duplicate_rows,
        }

    merged_outputs: dict[str, str] = {}
    merged_row_counts: dict[str, int] = {}
    merged_dataset_names: dict[str, list[str]] = {}
    for variant_key, rows in merged_rows_by_variant.items():
        if not rows:
            continue
        if preserve_legacy_stage1_layout and variant_key == "combined":
            merged_out = job_dirs["inputs_merged"] / f"{group_name}.jsonl"
        else:
            merged_out = job_dirs["inputs_merged"] / f"{group_name}_{variant_key}.jsonl"
        write_jsonl(merged_out, rows, overwrite=True)
        merged_outputs[variant_key] = str(merged_out)
        merged_row_counts[variant_key] = len(rows)
        merged_dataset_names[variant_key] = merged_dataset_names_by_variant[variant_key]
        print(
            f"[done] merged variant={variant_key} rows={len(rows)} out={merged_out}",
            file=sys.stderr,
            flush=True,
        )

    dataset_manifests: list[dict[str, Any]] = []
    for dataset_name in datasets:
        entry = existing_dataset_entries.get(dataset_name)
        if entry is None:
            continue
        dataset_manifests.append(entry)

    manifest = {
        "job_name": args.job_name,
        "group": group_name,
        "datasets": dataset_manifests,
        "source_download_subdir": args.source_download_subdir,
        "merged_outputs": merged_outputs,
        "merged_row_counts": merged_row_counts,
        "merged_dataset_names": merged_dataset_names,
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "paths": {key: str(value) for key, value in job_dirs.items()},
        "schema": {
            "top_level_fields": [
                "id",
                "source_type",
                "source_dataset",
                "group",
                "unit_type",
                "prompt",
                "base_messages",
                "system_overlay_mode",
                "meta",
            ],
            "group_name": group_name,
            "system_overlay_mode": "prepend_system",
        },
    }
    write_json(manifest_path, manifest, overwrite=True)
    print(f"[done] manifest={manifest_path}", file=sys.stderr, flush=True)
    return 0


def main() -> int:
    args = parse_args()
    datasets = choose_datasets(args.datasets)
    eval_root = Path(args.eval_root)
    out_root = Path(args.out_root)
    job_root = out_root / args.job_name
    if WILDCHAT_DEPLOYMENT_DATASET in datasets:
        if len(datasets) != 1:
            raise ValueError(
                "wildchat_deployment must be prepared separately so existing stage1 merged inputs are not rewritten."
            )
        job_dirs = build_job_dirs(job_root)
        return prepare_wildchat_deployment(args=args, eval_root=eval_root, job_dirs=job_dirs)
    is_phase2 = args.source_download_subdir != "stage1_rlhf_download" or bool(args.prepared_subdir)
    if is_phase2:
        prepared_subdir = args.prepared_subdir or "phase2"
        group_name = "phase2_eval"
        job_dirs = build_job_dirs(job_root, prepared_subdir=prepared_subdir)
        return prepare_eval_dataset_set(
            args=args,
            datasets=datasets,
            eval_root=eval_root,
            job_dirs=job_dirs,
            manifest_filename="prepare_phase2_eval_manifest.json",
            group_name=group_name,
            preserve_legacy_stage1_layout=False,
        )

    group_name = "standard_single_turn"
    job_dirs = build_job_dirs(job_root)
    return prepare_eval_dataset_set(
        args=args,
        datasets=datasets,
        eval_root=eval_root,
        job_dirs=job_dirs,
        manifest_filename="prepare_stage1_standard_single_turn_manifest.json",
        group_name=group_name,
        preserve_legacy_stage1_layout=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())
