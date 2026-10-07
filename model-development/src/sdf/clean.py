#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


SCRATCHPAD_BLOCK_RE = re.compile(
    r"<\s*scratchpad(?:\s+[^>\n]*)?>.*?</\s*scratchpad\s*>",
    re.IGNORECASE | re.DOTALL,
)
SCRATCHPAD_OPEN_RE = re.compile(r"<\s*scratchpad(?:\s+[^>\n]*)?>", re.IGNORECASE)
SCRATCHPAD_CLOSE_RE = re.compile(r"</\s*scratchpad\s*>", re.IGNORECASE)
SCRATCHPAD_TAG_RE = re.compile(r"</?\s*scratchpad(?:\s+[^>\n]*)?>", re.IGNORECASE)
CONTENT_OPEN_RE = re.compile(r"<\s*content(?:\s+[^>\n]*)?>", re.IGNORECASE)
CONTENT_TAG_RE = re.compile(r"</?\s*content(?:\s+[^>\n]*)?>", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Remove scratchpad reasoning scaffolding from JSONL documents while preserving "
            "normal prose mentions of the word 'scratchpad'."
        )
    )
    p.add_argument("--input", required=True, help="Input JSONL path.")
    p.add_argument("--output", default=None, help="Output JSONL path. Required unless --in-place.")
    p.add_argument("--in-place", action="store_true", help="Overwrite input file.")
    p.add_argument(
        "--fields",
        default="content",
        help="Comma-separated fields to clean (default: content).",
    )
    p.add_argument(
        "--keep-content-tags",
        action="store_true",
        help="Do not strip <content>...</content> tags.",
    )
    p.add_argument("--dry-run", action="store_true", help="Analyze only; do not write output.")
    return p.parse_args()


def clean_text(text: str, *, strip_content_tags: bool) -> tuple[str, dict[str, int]]:
    stats = {
        "removed_balanced_scratchpad_blocks": 0,
        "trimmed_prefix_to_close_tag": 0,
        "trimmed_open_to_content_tag": 0,
        "removed_leftover_scratchpad_tags": 0,
        "removed_content_tags": 0,
    }
    out = text

    # 1) Remove balanced <scratchpad>...</scratchpad> blocks.
    out, n_block = SCRATCHPAD_BLOCK_RE.subn("", out)
    stats["removed_balanced_scratchpad_blocks"] += n_block

    # 2) If a dangling close tag appears before the first <content>, drop prefix through close.
    m_content = CONTENT_OPEN_RE.search(out)
    content_pos = m_content.start() if m_content else -1
    m_open = SCRATCHPAD_OPEN_RE.search(out)
    m_close = SCRATCHPAD_CLOSE_RE.search(out)

    if m_close and (not m_open or m_open.start() > m_close.start()):
        if content_pos == -1 or m_close.start() < content_pos:
            out = out[m_close.end() :]
            stats["trimmed_prefix_to_close_tag"] += 1
            m_content = CONTENT_OPEN_RE.search(out)
            content_pos = m_content.start() if m_content else -1
            m_open = SCRATCHPAD_OPEN_RE.search(out)
            m_close = SCRATCHPAD_CLOSE_RE.search(out)

    # 3) If a dangling open tag appears before first <content> with no matching close before content,
    #    drop open tag region up to <content> (or to end if no <content> exists).
    if m_open and (content_pos == -1 or m_open.start() < content_pos):
        close_after_open = SCRATCHPAD_CLOSE_RE.search(out, m_open.end())
        close_before_content = bool(
            close_after_open and (content_pos == -1 or close_after_open.start() < content_pos)
        )
        if not close_before_content:
            if content_pos != -1:
                out = out[: m_open.start()] + out[content_pos:]
            else:
                out = out[: m_open.start()]
            stats["trimmed_open_to_content_tag"] += 1

    # 4) Remove any remaining literal scratchpad tags.
    out, n_tag = SCRATCHPAD_TAG_RE.subn("", out)
    stats["removed_leftover_scratchpad_tags"] += n_tag

    # 5) Remove content wrapper tags unless explicitly kept.
    if strip_content_tags:
        out, n_content = CONTENT_TAG_RE.subn("", out)
        stats["removed_content_tags"] += n_content

    return out.strip(), stats


def main() -> None:
    args = parse_args()
    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    if args.in_place and args.output:
        raise ValueError("Choose either --in-place or --output, not both.")
    if not args.in_place and not args.output and not args.dry_run:
        raise ValueError("Provide --output (or use --in-place or --dry-run).")

    out_path: Path | None
    if args.in_place:
        out_path = in_path
    elif args.output:
        out_path = Path(args.output)
    else:
        out_path = None

    fields = [f.strip() for f in args.fields.split(",") if f.strip()]
    strip_content_tags = not args.keep_content_tags

    lines_out: list[str] = []
    total_rows = 0
    changed_rows = 0
    changed_fields = 0
    total_ops = {
        "removed_balanced_scratchpad_blocks": 0,
        "trimmed_prefix_to_close_tag": 0,
        "trimmed_open_to_content_tag": 0,
        "removed_leftover_scratchpad_tags": 0,
        "removed_content_tags": 0,
    }
    residual_word_rows = 0
    residual_tag_rows = 0

    with in_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            total_rows += 1
            try:
                row: dict[str, Any] = json.loads(line)
            except json.JSONDecodeError:
                print(f"[warn] invalid JSON at {in_path}:{line_no}; passing through", file=sys.stderr)
                if not args.dry_run and out_path is not None:
                    lines_out.append(line.rstrip("\n"))
                continue

            row_changed = False
            for field in fields:
                v = row.get(field)
                if not isinstance(v, str):
                    continue
                cleaned, stats = clean_text(v, strip_content_tags=strip_content_tags)
                for k in total_ops:
                    total_ops[k] += stats[k]
                if cleaned != v:
                    row[field] = cleaned
                    row_changed = True
                    changed_fields += 1

            if row_changed:
                changed_rows += 1

            # Residual checks on requested fields after cleaning.
            field_texts = [row.get(f, "") for f in fields if isinstance(row.get(f), str)]
            joined = "\n".join(field_texts)
            if re.search(r"scratchpad", joined, flags=re.IGNORECASE):
                residual_word_rows += 1
            if SCRATCHPAD_TAG_RE.search(joined):
                residual_tag_rows += 1

            if not args.dry_run and out_path is not None:
                lines_out.append(json.dumps(row, ensure_ascii=False))

    if not args.dry_run and out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as w:
            for l in lines_out:
                w.write(l + "\n")
        tmp_path.replace(out_path)

    print("=== Scratchpad Strip Summary ===")
    print(f"Input: {in_path}")
    print(f"Rows processed: {total_rows}")
    print(f"Rows changed: {changed_rows}")
    print(f"Fields changed: {changed_fields}")
    print(f"removed_balanced_scratchpad_blocks: {total_ops['removed_balanced_scratchpad_blocks']}")
    print(f"trimmed_prefix_to_close_tag: {total_ops['trimmed_prefix_to_close_tag']}")
    print(f"trimmed_open_to_content_tag: {total_ops['trimmed_open_to_content_tag']}")
    print(f"removed_leftover_scratchpad_tags: {total_ops['removed_leftover_scratchpad_tags']}")
    print(f"removed_content_tags: {total_ops['removed_content_tags']}")
    print(f"Residual rows with word 'scratchpad': {residual_word_rows}")
    print(f"Residual rows with literal scratchpad tags: {residual_tag_rows}")
    if args.dry_run:
        print("Mode: dry-run (no file written)")
    else:
        print(f"Output: {out_path}")


if __name__ == "__main__":
    main()
