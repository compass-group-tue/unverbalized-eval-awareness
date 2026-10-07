#!/usr/bin/env python3
"""
Format SDF/SFT datasets into a unified training schema.

Output layout (under the caller-supplied output root):
    sdf/
    sft/
    manifests/
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SourceSpec:
    name: str
    path: Path
    source_type: str  # "sdf" | "sft"
    output_name: str
    fmt: str  # "sdf_content" | "sft_messages" | "sft_conversations"
    source_dataset: str


def _load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                yield line_no, json.loads(line)
            except Exception as exc:  # noqa: BLE001
                yield line_no, {"__parse_error__": str(exc)}


def _role_map(role: str) -> str | None:
    r = role.strip().lower()
    if r in {"human", "user"}:
        return "user"
    if r in {"assistant", "gpt", "bot"}:
        return "assistant"
    if r == "system":
        return "system"
    return None


def _normalize_messages(messages: Any) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    if not isinstance(messages, list):
        return out

    for msg in messages:
        if not isinstance(msg, dict):
            continue

        role_raw = msg.get("role")
        content = msg.get("content")
        if isinstance(role_raw, str):
            role = _role_map(role_raw)
        else:
            role = None

        if role is None or not isinstance(content, str):
            continue

        text = content.strip()
        if not text:
            continue
        out.append({"role": role, "content": text})
    return out


def _normalize_conversations(conversations: Any) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    if not isinstance(conversations, list):
        return out

    for turn in conversations:
        if not isinstance(turn, dict):
            continue
        role_raw = turn.get("from")
        value = turn.get("value")
        if not isinstance(role_raw, str) or not isinstance(value, str):
            continue
        role = _role_map(role_raw)
        if role is None:
            continue
        text = value.strip()
        if not text:
            continue
        out.append({"role": role, "content": text})
    return out


def build_specs(args: argparse.Namespace) -> list[SourceSpec]:
    specs: list[SourceSpec] = []

    if args.include_sdf:
        if args.sdf_path is None:
            raise ValueError("--sdf-path is required unless --no-include-sdf is used.")
        sdf_path = Path(args.sdf_path)
        specs.append(
            SourceSpec(
                name="sdf_main",
                path=sdf_path,
                source_type="sdf",
                output_name=args.sdf_output_name,
                fmt="sdf_content",
                source_dataset=args.sdf_source_dataset,
            )
        )

    if args.include_sft:
        missing = [
            flag
            for flag, value in (
                ("--am-dir", args.am_dir),
                ("--smoltalk-dir", args.smoltalk_dir),
                ("--dolci-dir", args.dolci_dir),
            )
            if value is None
        ]
        if missing:
            raise ValueError(f"SFT formatting requires {', '.join(missing)}; use --no-include-sft to omit it.")
        am_dir = Path(args.am_dir)
        smoltalk_dir = Path(args.smoltalk_dir)
        dolci_dir = Path(args.dolci_dir)
        specs.extend(
            [
                SourceSpec(
                    name="am_qwen3_distilled_code",
                    path=am_dir / "code_5M.jsonl",
                    source_type="sft",
                    output_name="am_qwen3_distilled_code.jsonl",
                    fmt="sft_conversations",
                    source_dataset="am_qwen3_distilled_code",
                ),
                SourceSpec(
                    name="am_qwen3_distilled_math",
                    path=am_dir / "math_5M.jsonl",
                    source_type="sft",
                    output_name="am_qwen3_distilled_math.jsonl",
                    fmt="sft_conversations",
                    source_dataset="am_qwen3_distilled_math",
                ),
                SourceSpec(
                    name="smoltalk_longalign",
                    path=smoltalk_dir / "LongAlign_64k_Qwen3_32B_yarn_131k_think_5M.jsonl",
                    source_type="sft",
                    output_name="smoltalk_longalign.jsonl",
                    fmt="sft_messages",
                    source_dataset="smoltalk_longalign",
                ),
                SourceSpec(
                    name="smoltalk_multi_turn_reasoning_if_think",
                    path=smoltalk_dir / "multi_turn_reasoning_if_think_20M.jsonl",
                    source_type="sft",
                    output_name="smoltalk_multi_turn_reasoning_if_think.jsonl",
                    fmt="sft_messages",
                    source_dataset="smoltalk_multi_turn_reasoning_if_think",
                ),
                SourceSpec(
                    name="smoltalk_everyday_convs_reasoning",
                    path=smoltalk_dir / "smoltalk_everyday_convs_reasoning_Qwen3_32B_think_20M.jsonl",
                    source_type="sft",
                    output_name="smoltalk_everyday_convs_reasoning.jsonl",
                    fmt="sft_messages",
                    source_dataset="smoltalk_everyday_convs_reasoning",
                ),
                SourceSpec(
                    name="smoltalk_systemchats",
                    path=smoltalk_dir / "smoltalk_systemchats_Qwen3_32B_think_20M.jsonl",
                    source_type="sft",
                    output_name="smoltalk_systemchats.jsonl",
                    fmt="sft_messages",
                    source_dataset="smoltalk_systemchats",
                ),
                SourceSpec(
                    name="dolci_mix_code",
                    path=dolci_dir / "dolci_mix_code.jsonl",
                    source_type="sft",
                    output_name="dolci_mix_code.jsonl",
                    fmt="sft_messages",
                    source_dataset="dolci_mix_code",
                ),
                SourceSpec(
                    name="dolci_mix_general",
                    path=dolci_dir / "dolci_mix_general.jsonl",
                    source_type="sft",
                    output_name="dolci_mix_general.jsonl",
                    fmt="sft_messages",
                    source_dataset="dolci_mix_general",
                ),
                SourceSpec(
                    name="dolci_mix_maths",
                    path=dolci_dir / "dolci_mix_maths.jsonl",
                    source_type="sft",
                    output_name="dolci_mix_maths.jsonl",
                    fmt="sft_messages",
                    source_dataset="dolci_mix_maths",
                ),
                SourceSpec(
                    name="dolci_mix_misc",
                    path=dolci_dir / "dolci_mix_misc.jsonl",
                    source_type="sft",
                    output_name="dolci_mix_misc.jsonl",
                    fmt="sft_messages",
                    source_dataset="dolci_mix_misc",
                ),
                SourceSpec(
                    name="dolci_mix_multilingual",
                    path=dolci_dir / "dolci_mix_multilingual.jsonl",
                    source_type="sft",
                    output_name="dolci_mix_multilingual.jsonl",
                    fmt="sft_messages",
                    source_dataset="dolci_mix_multilingual",
                ),
                SourceSpec(
                    name="dolci_mix_reasoning",
                    path=dolci_dir / "dolci_mix_reasoning.jsonl",
                    source_type="sft",
                    output_name="dolci_mix_reasoning.jsonl",
                    fmt="sft_messages",
                    source_dataset="dolci_mix_reasoning",
                ),
                SourceSpec(
                    name="dolci_mix_safety",
                    path=dolci_dir / "dolci_mix_safety.jsonl",
                    source_type="sft",
                    output_name="dolci_mix_safety.jsonl",
                    fmt="sft_messages",
                    source_dataset="dolci_mix_safety",
                ),
            ]
        )

    return specs


def _build_sdf_meta(row: dict[str, Any], line_no: int, include_scratchpad_meta: bool) -> dict[str, Any]:
    meta: dict[str, Any] = {"input_line_number": line_no}
    excluded = {"content", "custom_id", "scratchpad"}
    if include_scratchpad_meta:
        excluded = {"content", "custom_id"}
    for key, value in row.items():
        if key in excluded:
            continue
        meta[key] = value
    return meta


def process_source(
    spec: SourceSpec,
    out_path: Path,
    max_rows: int | None,
    include_scratchpad_meta: bool,
) -> dict[str, Any]:
    stats: dict[str, Any] = {
        "source_name": spec.name,
        "source_path": str(spec.path),
        "source_type": spec.source_type,
        "output_path": str(out_path),
        "rows_read": 0,
        "rows_written": 0,
        "skipped_missing_input": 0,
        "skipped_parse_error": 0,
        "skipped_empty_content": 0,
        "skipped_bad_messages": 0,
    }

    if not spec.path.exists():
        stats["skipped_missing_input"] = -1
        stats["error"] = "missing_input"
        return stats

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as out:
        for line_no, row in _load_jsonl(spec.path):
            stats["rows_read"] += 1
            if max_rows is not None and stats["rows_read"] > max_rows:
                break

            if "__parse_error__" in row:
                stats["skipped_parse_error"] += 1
                continue

            if spec.fmt == "sdf_content":
                content = row.get("content")
                if not isinstance(content, str) or not content.strip():
                    stats["skipped_empty_content"] += 1
                    continue
                out_row = {
                    "doc_id": row.get("custom_id") or f"{spec.name}:{line_no}",
                    "source_type": "sdf",
                    "source_dataset": spec.source_dataset,
                    "objective": "sdf_full_loss",
                    "text": content.strip(),
                    "meta": _build_sdf_meta(
                        row=row,
                        line_no=line_no,
                        include_scratchpad_meta=include_scratchpad_meta,
                    ),
                }
                out.write(json.dumps(out_row, ensure_ascii=False) + "\n")
                stats["rows_written"] += 1
                continue

            if spec.fmt == "sft_messages":
                messages = _normalize_messages(row.get("messages"))
            elif spec.fmt == "sft_conversations":
                messages = _normalize_conversations(row.get("conversations"))
            else:
                raise ValueError(f"Unsupported format: {spec.fmt}")

            if not messages:
                stats["skipped_bad_messages"] += 1
                continue

            out_row = {
                "doc_id": row.get("id") or f"{spec.name}:{line_no}",
                "source_type": "sft",
                "source_dataset": spec.source_dataset,
                "objective": "sft_assistant_only",
                "messages": messages,
                "meta": {
                    "input_line_number": line_no,
                    "source_dataset_raw": row.get("source_dataset"),
                    "domain": row.get("domain"),
                    "category": row.get("_category") or row.get("category"),
                    "assistant_token_count_hint": row.get("_assistant_token_count")
                    or row.get("assistant_token_count"),
                    "chat_template_kwargs": row.get("chat_template_kwargs"),
                    "source": row.get("source"),
                },
            }
            out.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            stats["rows_written"] += 1

    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Format training datasets to the unified schema.")
    parser.add_argument(
        "--output-root",
        required=True,
        help="Root output directory.",
    )
    parser.add_argument(
        "--sdf-path",
        default=None,
        help="SDF cleaned JSONL path.",
    )
    parser.add_argument(
        "--sdf-output-name",
        default="sdf_main.jsonl",
        help="Output filename for the formatted SDF dataset.",
    )
    parser.add_argument(
        "--sdf-source-dataset",
        default="sdf_synth_docs_003_004_cleaned",
        help="source_dataset label to write for SDF rows.",
    )
    parser.add_argument(
        "--include-sdf",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to include the configured SDF source.",
    )
    parser.add_argument(
        "--include-sft",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to include the configured SFT sources.",
    )
    parser.add_argument(
        "--include-scratchpad-meta",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Preserve the raw SDF scratchpad under meta.",
    )
    parser.add_argument(
        "--am-dir",
        default=None,
        help="Directory containing AM reasoning datasets.",
    )
    parser.add_argument(
        "--smoltalk-dir",
        default=None,
        help="Directory containing SmolTalk think datasets.",
    )
    parser.add_argument(
        "--dolci-dir",
        default=None,
        help="Directory containing separated Dolci mix files (not merged).",
    )
    parser.add_argument(
        "--max-rows-per-file",
        type=int,
        default=None,
        help="Optional row cap per input file for smoke tests.",
    )
    parser.add_argument(
        "--tag",
        default="format_training_datasets",
        help="Tag prefix for manifest filename.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_root)
    sdf_dir = output_root / "sdf"
    sft_dir = output_root / "sft"
    manifests_dir = output_root / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)

    specs = build_specs(args)
    if not specs:
        print("[error] no dataset sources selected")
        return 2

    per_file_stats: list[dict[str, Any]] = []

    for spec in specs:
        out_dir = sdf_dir if spec.source_type == "sdf" else sft_dir
        out_path = out_dir / spec.output_name
        stats = process_source(
            spec=spec,
            out_path=out_path,
            max_rows=args.max_rows_per_file,
            include_scratchpad_meta=args.include_scratchpad_meta,
        )
        per_file_stats.append(stats)
        print(
            f"[file] {spec.name} read={stats['rows_read']} "
            f"written={stats['rows_written']} skipped_parse={stats['skipped_parse_error']} "
            f"skipped_empty={stats['skipped_empty_content']} skipped_bad_messages={stats['skipped_bad_messages']}"
        )

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_root": str(output_root),
        "config": {
            "include_sdf": args.include_sdf,
            "include_sft": args.include_sft,
            "sdf_path": args.sdf_path,
            "sdf_output_name": args.sdf_output_name,
            "sdf_source_dataset": args.sdf_source_dataset,
            "include_scratchpad_meta": args.include_scratchpad_meta,
        },
        "schema": {
            "sdf_fields": ["doc_id", "source_type", "source_dataset", "objective", "text", "meta"],
            "sft_fields": [
                "doc_id",
                "source_type",
                "source_dataset",
                "objective",
                "messages",
                "meta",
            ],
            "objectives": ["sdf_full_loss", "sft_assistant_only"],
        },
        "files": per_file_stats,
        "totals": {
            "rows_read": sum(x["rows_read"] for x in per_file_stats),
            "rows_written": sum(x["rows_written"] for x in per_file_stats),
            "skipped_parse_error": sum(x["skipped_parse_error"] for x in per_file_stats),
            "skipped_empty_content": sum(x["skipped_empty_content"] for x in per_file_stats),
            "skipped_bad_messages": sum(x["skipped_bad_messages"] for x in per_file_stats),
            "missing_inputs": [
                x["source_name"] for x in per_file_stats if x.get("error") == "missing_input"
            ],
        },
    }

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    manifest_path = manifests_dir / f"{args.tag}_{stamp}.json"
    manifest_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"[done] manifest={manifest_path}")
    print(
        "[done] totals: "
        f"read={summary['totals']['rows_read']} "
        f"written={summary['totals']['rows_written']} "
        f"missing_inputs={len(summary['totals']['missing_inputs'])}"
    )
    if summary["totals"]["missing_inputs"]:
        print(f"[warn] missing_inputs={summary['totals']['missing_inputs']}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
