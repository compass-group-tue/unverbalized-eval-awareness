#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


SCRATCHPAD_OPEN = re.compile(r"<\s*scratchpad(?:\s+[^>\n]*)?>", re.I)
SCRATCHPAD_CLOSE = re.compile(r"</\s*scratchpad\s*>", re.I)
CONTENT_OPEN = re.compile(r"<\s*content(?:\s+[^>\n]*)?>", re.I)
CONTENT_CLOSE = re.compile(r"</\s*content\s*>", re.I)
CONTENT_PAIR = re.compile(r"<\s*content(?:\s+[^>\n]*)?>(.*?)</\s*content\s*>", re.I | re.S)
SCRATCHPAD_PAIR = re.compile(r"<\s*scratchpad(?:\s+[^>\n]*)?>(.*?)</\s*scratchpad\s*>", re.I | re.S)
FACT_TAGS_PAIR = re.compile(r"<\s*(?:fact_tags|tags)(?:\s+[^>\n]*)?>(.*?)</\s*(?:fact_tags|tags)\s*>", re.I | re.S)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Rebuild synth-doc outputs from submit_manifest + raw Azure outputs, keeping only "
            "structurally valid final generations with balanced <content> tags."
        )
    )
    p.add_argument("--run-dir", required=True, help="Run directory containing manifests/ and azure_logs/.")
    p.add_argument("--out-prefix", required=True, help="Output prefix used for per-fact and combined files.")
    p.add_argument(
        "--universe-contexts-path",
        default="",
        help="Optional universe_context JSONL path to populate universe_context_id/is_true.",
    )
    p.add_argument(
        "--per-fact-dir",
        default="",
        help="Optional output directory for per-fact rebuilt files. Defaults to <run-dir>/per_fact_docs_balanced.",
    )
    p.add_argument(
        "--merged-dir",
        default="",
        help="Optional output directory for combined rebuilt files. Defaults to <run-dir>/merged.",
    )
    p.add_argument(
        "--output-suffix",
        default="balanced_content_only",
        help="Suffix inserted before .jsonl for rebuilt outputs.",
    )
    return p.parse_args()


def parse_tag_list(text: str | None) -> list[str]:
    if not text:
        return []
    parts = re.split(r"[,\n]+", text)
    return [p.strip() for p in parts if p.strip()]


def classify_raw_output(text: str) -> str:
    t = (text or "").strip()
    has_so = bool(SCRATCHPAD_OPEN.search(t))
    has_sc = bool(SCRATCHPAD_CLOSE.search(t))
    has_co = bool(CONTENT_OPEN.search(t))
    has_cc = bool(CONTENT_CLOSE.search(t))
    m = CONTENT_PAIR.search(t)
    extracted = m.group(1).strip() if m else ""

    if not t:
        return "empty_raw"
    if has_so and not has_co:
        return "scratchpad_only_no_content"
    if has_so and has_co and not has_cc:
        return "scratchpad_then_open_content_no_close"
    if has_co and not has_cc:
        return "open_content_no_close"
    if has_so and has_sc and has_co and has_cc and not extracted:
        return "balanced_tags_empty_content"
    if has_co and has_cc and extracted:
        return "has_balanced_content"
    if t.startswith("<"):
        return "starts_with_tag_other"
    return "no_content_tags_plain"


def load_universe_context(path: str) -> dict[str, Any] | None:
    if not path:
        return None
    input_path = Path(path)
    if not input_path.exists():
        return None
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                return json.loads(line)
    return None


def load_specs_from_input_file(path: Path) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            cid = obj.get("custom_id", "")
            body = obj.get("body", {})
            msgs = body.get("messages", [])
            prompt_text = msgs[0].get("content", "") if msgs and isinstance(msgs, list) else ""
            fact = ""
            doc_type = ""
            doc_idea = ""
            m = re.search(r"<\s*fact(?:\s+[^>\n]*)?>(.*?)</\s*fact\s*>", prompt_text, re.I | re.S)
            if m:
                fact = m.group(1).strip()
            m = re.search(r"<\s*document_type(?:\s+[^>\n]*)?>(.*?)</\s*document_type\s*>", prompt_text, re.I | re.S)
            if m:
                doc_type = m.group(1).strip()
            m = re.search(r"<\s*idea(?:\s+[^>\n]*)?>(.*?)</\s*idea\s*>", prompt_text, re.I | re.S)
            if m:
                doc_idea = m.group(1).strip()
            specs.append(
                {
                    "custom_id": cid,
                    "fact": fact,
                    "doc_type": doc_type,
                    "doc_idea": doc_idea,
                }
            )
    return specs


def build_row(
    spec: dict[str, Any],
    raw_text: str,
    finish_reason: str | None,
    batch_id: str,
    chunk_index: int,
    uc: dict[str, Any] | None,
) -> dict[str, Any]:
    content_match = CONTENT_PAIR.search(raw_text)
    scratchpad_match = SCRATCHPAD_PAIR.search(raw_text)
    tags_match = FACT_TAGS_PAIR.search(raw_text)

    content = content_match.group(1).strip() if content_match else ""
    scratchpad = scratchpad_match.group(1).strip() if scratchpad_match else ""
    tags = parse_tag_list(tags_match.group(1) if tags_match else "")

    row = dict(spec)
    row["content"] = content
    row["scratchpad"] = scratchpad
    row["token_count"] = len(content.split()) if content else 0
    row["custom_id"] = spec.get("custom_id")
    row["batch_id"] = batch_id
    row["chunk_index"] = chunk_index
    row["finish_reason"] = finish_reason
    if uc is not None:
        if "id" in uc:
            row["universe_context_id"] = uc["id"]
        if "is_true" in uc:
            row["is_true"] = uc["is_true"]
    if tags:
        row["tags"] = tags
    return row


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir)
    submit_manifest_path = run_dir / "manifests" / "submit_manifest.json"
    submit_manifest = json.loads(submit_manifest_path.read_text())
    azure_log_dir = run_dir / "azure_logs"
    uc = load_universe_context(args.universe_contexts_path)

    per_fact_dir = (
        Path(args.per_fact_dir) if args.per_fact_dir else run_dir / "per_fact_docs_balanced"
    )
    merged_dir = Path(args.merged_dir) if args.merged_dir else run_dir / "merged"

    combined_rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "run_dir": str(run_dir),
        "submit_manifest": str(submit_manifest_path),
        "output_suffix": args.output_suffix,
        "facts": [],
        "class_counts_total": {},
    }
    total_class_counts: Counter[str] = Counter()

    for fact in submit_manifest.get("facts", []):
        fact_suffix = fact["fact_suffix"]
        per_fact_rows: list[dict[str, Any]] = []
        fact_class_counts: Counter[str] = Counter()

        for chunk in fact.get("chunks", []):
            batch_id = chunk["batch_id"]
            chunk_index = int(chunk["chunk_index"])
            input_path = Path(chunk["azure_input_path"])
            output_path = azure_log_dir / f"azure_batch_output_batch_{batch_id}.jsonl"
            if not input_path.exists() or not output_path.exists():
                continue

            specs = load_specs_from_input_file(input_path)
            with output_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    resp = obj.get("response", {})
                    if resp.get("status_code") != 200:
                        fact_class_counts["non_200"] += 1
                        total_class_counts["non_200"] += 1
                        continue
                    body = resp.get("body", {})
                    choices = body.get("choices", [])
                    if not choices:
                        fact_class_counts["no_choices"] += 1
                        total_class_counts["no_choices"] += 1
                        continue

                    choice = choices[0]
                    raw_text = choice.get("message", {}).get("content") or ""
                    cls = classify_raw_output(raw_text)
                    fact_class_counts[cls] += 1
                    total_class_counts[cls] += 1
                    if cls != "has_balanced_content":
                        continue

                    cid = obj.get("custom_id")
                    try:
                        idx = int(str(cid).split("_", 1)[0])
                    except Exception:
                        continue
                    if idx < 0 or idx >= len(specs):
                        continue

                    row = build_row(
                        spec=specs[idx],
                        raw_text=raw_text,
                        finish_reason=choice.get("finish_reason"),
                        batch_id=batch_id,
                        chunk_index=chunk_index,
                        uc=uc,
                    )
                    per_fact_rows.append(row)

        per_fact_name = f"{args.out_prefix}_{fact_suffix}.{args.output_suffix}.jsonl"
        per_fact_path = per_fact_dir / per_fact_name
        write_jsonl(per_fact_path, per_fact_rows)
        combined_rows.extend(per_fact_rows)

        summary["facts"].append(
            {
                "fact_suffix": fact_suffix,
                "fact": fact.get("fact"),
                "valid_rows": len(per_fact_rows),
                "class_counts": dict(fact_class_counts),
                "output_path": str(per_fact_path),
            }
        )

    combined_path = merged_dir / f"{args.out_prefix}_combined.{args.output_suffix}.jsonl"
    write_jsonl(combined_path, combined_rows)
    summary["class_counts_total"] = dict(total_class_counts)
    summary["combined_output_path"] = str(combined_path)
    summary["combined_valid_rows"] = len(combined_rows)

    summary_path = merged_dir / f"{args.out_prefix}_combined.{args.output_suffix}.summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")

    print(f"[done] combined_output={combined_path}")
    print(f"[done] combined_valid_rows={len(combined_rows)}")
    print(f"[done] summary={summary_path}")
    print("=== class_counts_total ===")
    for key, value in total_class_counts.most_common():
        print(f"{key}\t{value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
