#!/usr/bin/env python3
"""
Validate formatted v1 training datasets.

Checks:
- required fields and field types
- source_type/objective consistency
- SDF text non-empty
- SFT messages structure and roles
- duplicate doc_id (within file and across files)

Outputs:
- JSON report
- TXT summary
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


VALID_ROLES = {"system", "user", "assistant"}


@dataclass
class ValidationIssue:
    line_number: int
    reason: str
    doc_id: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "line_number": self.line_number,
            "reason": self.reason,
            "doc_id": self.doc_id,
        }


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                yield line_no, json.loads(line), None
            except Exception as exc:  # noqa: BLE001
                yield line_no, None, str(exc)


def _append_issue(
    issues: list[ValidationIssue],
    reason_counts: Counter[str],
    line_number: int,
    reason: str,
    doc_id: str | None,
    max_saved: int,
) -> None:
    reason_counts[reason] += 1
    if len(issues) < max_saved:
        issues.append(ValidationIssue(line_number=line_number, reason=reason, doc_id=doc_id))


def validate_file(
    path: Path,
    source_group: str,
    max_saved_issues: int,
    valid_sample_limit: int,
    global_doc_id_first_seen: dict[str, str],
    cross_file_dupe_counter: Counter[str],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": str(path),
        "source_group": source_group,
        "exists": path.exists(),
        "rows_read": 0,
        "rows_valid": 0,
        "rows_invalid": 0,
        "reason_counts": {},
        "issues": [],
        "manual_samples": [],
        "message_role_counts": {},
        "assistant_turn_distribution": {},
        "duplicate_doc_id_within_file": 0,
        "duplicate_doc_id_across_files": 0,
    }

    if not path.exists():
        result["reason_counts"] = {"missing_file": 1}
        result["rows_invalid"] = 1
        return result

    issues: list[ValidationIssue] = []
    reason_counts: Counter[str] = Counter()
    role_counts: Counter[str] = Counter()
    assistant_turn_dist: Counter[int] = Counter()
    seen_doc_ids: set[str] = set()

    for line_no, obj, parse_err in _iter_jsonl(path):
        result["rows_read"] += 1

        if parse_err is not None:
            _append_issue(
                issues,
                reason_counts,
                line_number=line_no,
                reason=f"json_parse_error:{parse_err}",
                doc_id=None,
                max_saved=max_saved_issues,
            )
            result["rows_invalid"] += 1
            continue

        if not isinstance(obj, dict):
            _append_issue(
                issues,
                reason_counts,
                line_number=line_no,
                reason="row_not_object",
                doc_id=None,
                max_saved=max_saved_issues,
            )
            result["rows_invalid"] += 1
            continue

        doc_id = obj.get("doc_id")
        source_type = obj.get("source_type")
        objective = obj.get("objective")
        source_dataset = obj.get("source_dataset")

        row_ok = True

        if not isinstance(doc_id, str) or not doc_id.strip():
            _append_issue(
                issues,
                reason_counts,
                line_number=line_no,
                reason="missing_or_empty_doc_id",
                doc_id=None,
                max_saved=max_saved_issues,
            )
            row_ok = False
        else:
            if doc_id in seen_doc_ids:
                result["duplicate_doc_id_within_file"] += 1
                _append_issue(
                    issues,
                    reason_counts,
                    line_number=line_no,
                    reason="duplicate_doc_id_within_file",
                    doc_id=doc_id,
                    max_saved=max_saved_issues,
                )
                row_ok = False
            seen_doc_ids.add(doc_id)

            first_seen = global_doc_id_first_seen.get(doc_id)
            if first_seen is None:
                global_doc_id_first_seen[doc_id] = str(path)
            elif first_seen != str(path):
                result["duplicate_doc_id_across_files"] += 1
                cross_file_dupe_counter[doc_id] += 1
                _append_issue(
                    issues,
                    reason_counts,
                    line_number=line_no,
                    reason=f"duplicate_doc_id_across_files:first_seen={first_seen}",
                    doc_id=doc_id,
                    max_saved=max_saved_issues,
                )
                row_ok = False

        if not isinstance(source_dataset, str) or not source_dataset.strip():
            _append_issue(
                issues,
                reason_counts,
                line_number=line_no,
                reason="missing_or_empty_source_dataset",
                doc_id=doc_id if isinstance(doc_id, str) else None,
                max_saved=max_saved_issues,
            )
            row_ok = False

        if source_group == "sdf":
            if source_type != "sdf":
                _append_issue(
                    issues,
                    reason_counts,
                    line_number=line_no,
                    reason=f"bad_source_type_expected_sdf_got:{source_type}",
                    doc_id=doc_id if isinstance(doc_id, str) else None,
                    max_saved=max_saved_issues,
                )
                row_ok = False
            if objective != "sdf_full_loss":
                _append_issue(
                    issues,
                    reason_counts,
                    line_number=line_no,
                    reason=f"bad_objective_expected_sdf_full_loss_got:{objective}",
                    doc_id=doc_id if isinstance(doc_id, str) else None,
                    max_saved=max_saved_issues,
                )
                row_ok = False
            text = obj.get("text")
            if not isinstance(text, str) or not text.strip():
                _append_issue(
                    issues,
                    reason_counts,
                    line_number=line_no,
                    reason="missing_or_empty_text",
                    doc_id=doc_id if isinstance(doc_id, str) else None,
                    max_saved=max_saved_issues,
                )
                row_ok = False

        elif source_group == "sft":
            if source_type != "sft":
                _append_issue(
                    issues,
                    reason_counts,
                    line_number=line_no,
                    reason=f"bad_source_type_expected_sft_got:{source_type}",
                    doc_id=doc_id if isinstance(doc_id, str) else None,
                    max_saved=max_saved_issues,
                )
                row_ok = False
            if objective != "sft_assistant_only":
                _append_issue(
                    issues,
                    reason_counts,
                    line_number=line_no,
                    reason=f"bad_objective_expected_sft_assistant_only_got:{objective}",
                    doc_id=doc_id if isinstance(doc_id, str) else None,
                    max_saved=max_saved_issues,
                )
                row_ok = False

            messages = obj.get("messages")
            if not isinstance(messages, list) or len(messages) == 0:
                _append_issue(
                    issues,
                    reason_counts,
                    line_number=line_no,
                    reason="missing_or_empty_messages",
                    doc_id=doc_id if isinstance(doc_id, str) else None,
                    max_saved=max_saved_issues,
                )
                row_ok = False
            else:
                assistant_turns = 0
                for idx, msg in enumerate(messages):
                    if not isinstance(msg, dict):
                        _append_issue(
                            issues,
                            reason_counts,
                            line_number=line_no,
                            reason=f"message_not_object_at_index:{idx}",
                            doc_id=doc_id if isinstance(doc_id, str) else None,
                            max_saved=max_saved_issues,
                        )
                        row_ok = False
                        continue
                    role = msg.get("role")
                    content = msg.get("content")
                    if not isinstance(role, str) or role not in VALID_ROLES:
                        _append_issue(
                            issues,
                            reason_counts,
                            line_number=line_no,
                            reason=f"bad_message_role_at_index:{idx}:{role}",
                            doc_id=doc_id if isinstance(doc_id, str) else None,
                            max_saved=max_saved_issues,
                        )
                        row_ok = False
                    else:
                        role_counts[role] += 1
                        if role == "assistant":
                            assistant_turns += 1
                    if not isinstance(content, str) or not content.strip():
                        _append_issue(
                            issues,
                            reason_counts,
                            line_number=line_no,
                            reason=f"empty_message_content_at_index:{idx}",
                            doc_id=doc_id if isinstance(doc_id, str) else None,
                            max_saved=max_saved_issues,
                        )
                        row_ok = False

                if assistant_turns <= 0:
                    _append_issue(
                        issues,
                        reason_counts,
                        line_number=line_no,
                        reason="no_assistant_turns",
                        doc_id=doc_id if isinstance(doc_id, str) else None,
                        max_saved=max_saved_issues,
                    )
                    row_ok = False
                else:
                    assistant_turn_dist[assistant_turns] += 1

        else:
            _append_issue(
                issues,
                reason_counts,
                line_number=line_no,
                reason=f"unknown_source_group:{source_group}",
                doc_id=doc_id if isinstance(doc_id, str) else None,
                max_saved=max_saved_issues,
            )
            row_ok = False

        if row_ok:
            result["rows_valid"] += 1
            if len(result["manual_samples"]) < valid_sample_limit:
                sample: dict[str, Any] = {
                    "line_number": line_no,
                    "doc_id": doc_id,
                    "source_dataset": source_dataset,
                    "objective": objective,
                }
                if source_group == "sdf":
                    text = obj.get("text", "")
                    sample["text_preview"] = text[:800] if isinstance(text, str) else None
                else:
                    messages = obj.get("messages", [])
                    preview_messages: list[dict[str, Any]] = []
                    if isinstance(messages, list):
                        for msg in messages[:3]:
                            if isinstance(msg, dict):
                                role = msg.get("role")
                                content = msg.get("content")
                                preview_messages.append(
                                    {
                                        "role": role,
                                        "content_preview": content[:400]
                                        if isinstance(content, str)
                                        else None,
                                    }
                                )
                    sample["messages_preview"] = preview_messages
                result["manual_samples"].append(sample)
        else:
            result["rows_invalid"] += 1

    result["issues"] = [x.as_dict() for x in issues]
    result["reason_counts"] = dict(reason_counts)
    result["message_role_counts"] = dict(role_counts)
    result["assistant_turn_distribution"] = dict(assistant_turn_dist)
    return result


def write_txt_report(report: dict[str, Any], out_path: Path) -> None:
    lines: list[str] = []
    lines.append("Training Data V1 Validation Report")
    lines.append(f"Generated: {report['generated_at_utc']}")
    lines.append(f"Input root: {report['input_root']}")
    lines.append("")

    for fr in report["files"]:
        lines.append("=" * 100)
        lines.append(f"FILE: {fr['path']}")
        lines.append(f"SOURCE_GROUP: {fr['source_group']}")
        lines.append(f"EXISTS: {fr['exists']}")
        lines.append(
            "COUNTS: "
            f"read={fr['rows_read']} valid={fr['rows_valid']} invalid={fr['rows_invalid']} "
            f"dup_within={fr['duplicate_doc_id_within_file']} dup_across={fr['duplicate_doc_id_across_files']}"
        )
        if fr["source_group"] == "sft":
            lines.append(f"MESSAGE_ROLE_COUNTS: {fr['message_role_counts']}")
            lines.append(f"ASSISTANT_TURN_DISTRIBUTION: {fr['assistant_turn_distribution']}")
        lines.append(f"REASONS: {fr['reason_counts']}")
        lines.append("MANUAL_SAMPLES:")
        for sample in fr["manual_samples"]:
            lines.append(f"- line={sample['line_number']} doc_id={sample['doc_id']} source_dataset={sample['source_dataset']} objective={sample['objective']}")
            if "text_preview" in sample:
                lines.append(f"  text_preview={sample['text_preview']}")
            if "messages_preview" in sample:
                lines.append(f"  messages_preview={sample['messages_preview']}")
        if fr["issues"]:
            lines.append("ISSUES:")
            for issue in fr["issues"]:
                lines.append(
                    f"- line={issue['line_number']} doc_id={issue['doc_id']} reason={issue['reason']}"
                )
        lines.append("")

    lines.append("=" * 100)
    lines.append("TOTALS")
    lines.append(
        f"files={report['totals']['files_scanned']} "
        f"rows_read={report['totals']['rows_read']} "
        f"rows_valid={report['totals']['rows_valid']} "
        f"rows_invalid={report['totals']['rows_invalid']}"
    )
    lines.append(
        f"dup_within={report['totals']['duplicate_doc_id_within_file']} "
        f"dup_across={report['totals']['duplicate_doc_id_across_files']}"
    )
    lines.append(f"top_reasons={report['totals']['top_reasons']}")
    lines.append(f"status={'PASS' if report['totals']['rows_invalid'] == 0 else 'FAIL'}")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a formatted SDF/SFT training-data directory.")
    parser.add_argument(
        "--input-root",
        required=True,
        help="Root directory with sdf/ and sft/ subdirs.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for validation reports.",
    )
    parser.add_argument("--tag", default="validation_v1", help="Output file prefix.")
    parser.add_argument(
        "--max-saved-issues-per-file",
        type=int,
        default=20,
        help="Store at most this many example issues per file.",
    )
    parser.add_argument(
        "--valid-sample-limit",
        type=int,
        default=3,
        help="Store at most this many manual-review samples per file.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_root = Path(args.input_root)
    sdf_dir = input_root / "sdf"
    sft_dir = input_root / "sft"

    files: list[tuple[Path, str]] = []
    files.extend((p, "sdf") for p in sorted(sdf_dir.glob("*.jsonl")))
    files.extend((p, "sft") for p in sorted(sft_dir.glob("*.jsonl")))

    if not files:
        print(f"[error] no JSONL files found under {input_root}/sdf or {input_root}/sft")
        return 2

    global_doc_id_first_seen: dict[str, str] = {}
    cross_file_dupe_counter: Counter[str] = Counter()
    file_reports: list[dict[str, Any]] = []
    total_reason_counts: Counter[str] = Counter()

    for path, source_group in files:
        fr = validate_file(
            path=path,
            source_group=source_group,
            max_saved_issues=args.max_saved_issues_per_file,
            valid_sample_limit=args.valid_sample_limit,
            global_doc_id_first_seen=global_doc_id_first_seen,
            cross_file_dupe_counter=cross_file_dupe_counter,
        )
        file_reports.append(fr)
        total_reason_counts.update(fr["reason_counts"])
        print(
            f"[file] {path.name} read={fr['rows_read']} valid={fr['rows_valid']} invalid={fr['rows_invalid']} "
            f"dup_within={fr['duplicate_doc_id_within_file']} dup_across={fr['duplicate_doc_id_across_files']}"
        )

    totals = {
        "files_scanned": len(file_reports),
        "rows_read": sum(fr["rows_read"] for fr in file_reports),
        "rows_valid": sum(fr["rows_valid"] for fr in file_reports),
        "rows_invalid": sum(fr["rows_invalid"] for fr in file_reports),
        "duplicate_doc_id_within_file": sum(fr["duplicate_doc_id_within_file"] for fr in file_reports),
        "duplicate_doc_id_across_files": sum(fr["duplicate_doc_id_across_files"] for fr in file_reports),
        "cross_file_duplicate_doc_ids": len(cross_file_dupe_counter),
        "top_cross_file_duplicate_doc_ids": cross_file_dupe_counter.most_common(20),
        "top_reasons": total_reason_counts.most_common(30),
    }

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_root": str(input_root),
        "files": file_reports,
        "totals": totals,
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    json_out = out_dir / f"{args.tag}_{stamp}.json"
    txt_out = out_dir / f"{args.tag}_{stamp}.txt"

    json_out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    write_txt_report(report, txt_out)

    print(f"[done] json={json_out}")
    print(f"[done] txt={txt_out}")
    print(
        f"[done] totals: files={totals['files_scanned']} read={totals['rows_read']} "
        f"valid={totals['rows_valid']} invalid={totals['rows_invalid']}"
    )

    return 1 if totals["rows_invalid"] > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
