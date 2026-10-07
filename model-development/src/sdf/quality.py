#!/usr/bin/env python3
"""
Generate a lightweight dataset quality report for synth docs.

Outputs:
- JSON with metrics
- Text summary (human readable)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Dataset quality report for synth docs.")
    p.add_argument("--input", required=True, help="Path to combined synth_docs JSONL.")
    p.add_argument("--out", required=True, help="Output prefix (without extension).")
    p.add_argument("--low_doc_type_threshold", type=int, default=10)
    p.add_argument("--short_doc_tokens", type=int, default=5)
    return p.parse_args()


def word_tokens(text: str) -> int:
    if not text:
        return 0
    return len(text.split())


def shannon_entropy(counts: Dict[str, int]) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    import math
    entropy = 0.0
    for c in counts.values():
        if c == 0:
            continue
        p = c / total
        entropy -= p * math.log2(p)
    return entropy


def normalized_entropy(counts: Dict[str, int]) -> float:
    import math
    k = len(counts)
    if k <= 1:
        return 0.0
    return shannon_entropy(counts) / math.log2(k)


def percentile(values: List[int], p: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    if p <= 0:
        return float(values[0])
    if p >= 100:
        return float(values[-1])
    k = (len(values) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(values) - 1)
    if f == c:
        return float(values[f])
    return values[f] + (values[c] - values[f]) * (k - f)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    out_prefix = Path(args.out)

    fact_counts: Dict[str, int] = {}
    doc_type_counts: Dict[str, int] = {}
    doc_type_per_fact: Dict[str, Dict[str, int]] = {}
    content_hashes: Dict[str, int] = {}
    token_counts: List[int] = []
    short_docs = 0
    total_docs = 0

    with input_path.open("r") as f:
        for line in f:
            if not line.strip():
                continue
            total_docs += 1
            row = json.loads(line)
            fact = row.get("fact", "") or "<missing_fact>"
            doc_type = row.get("doc_type", "") or "<missing_type>"
            content = row.get("content", "") or ""

            fact_counts[fact] = fact_counts.get(fact, 0) + 1
            doc_type_counts[doc_type] = doc_type_counts.get(doc_type, 0) + 1
            doc_type_per_fact.setdefault(fact, {})
            doc_type_per_fact[fact][doc_type] = doc_type_per_fact[fact].get(doc_type, 0) + 1

            t = word_tokens(content)
            token_counts.append(t)
            if t < args.short_doc_tokens:
                short_docs += 1

            h = hashlib.sha256(content.encode("utf-8")).hexdigest()
            content_hashes[h] = content_hashes.get(h, 0) + 1

    dup_counts = [c for c in content_hashes.values() if c > 1]
    total_dupes = sum(c - 1 for c in dup_counts)

    low_doc_types = {k: v for k, v in doc_type_counts.items() if v < args.low_doc_type_threshold}

    metrics = {
        "total_docs": total_docs,
        "short_doc_tokens_threshold": args.short_doc_tokens,
        "short_docs": short_docs,
        "short_docs_pct": (short_docs / total_docs) if total_docs else 0.0,
        "token_length": {
            "min": min(token_counts) if token_counts else 0,
            "p5": percentile(token_counts, 5),
            "p10": percentile(token_counts, 10),
            "median": percentile(token_counts, 50),
            "p97": percentile(token_counts, 97),
            "max": max(token_counts) if token_counts else 0,
        },
        "entropy": {
            "facts_entropy": shannon_entropy(fact_counts),
            "facts_entropy_normalized": normalized_entropy(fact_counts),
            "doc_types_entropy": shannon_entropy(doc_type_counts),
            "doc_types_entropy_normalized": normalized_entropy(doc_type_counts),
        },
        "duplicates": {
            "total_duplicate_rows": total_dupes,
            "duplicate_groups": len(dup_counts),
        },
        "low_doc_types": {
            "threshold": args.low_doc_type_threshold,
            "count": len(low_doc_types),
            "types": low_doc_types,
        },
    }

    out_json = out_prefix.with_suffix(".json")
    out_txt = out_prefix.with_suffix(".txt")

    out_json.write_text(json.dumps(metrics, indent=2))

    lines = []
    lines.append(f"total_docs: {total_docs}")
    lines.append(f"short_docs (<{args.short_doc_tokens} tokens): {short_docs} ({metrics['short_docs_pct']:.2%})")
    tl = metrics["token_length"]
    lines.append(f"token_length min={tl['min']} p5={tl['p5']:.1f} p10={tl['p10']:.1f} median={tl['median']:.1f} p97={tl['p97']:.1f} max={tl['max']}")
    ent = metrics["entropy"]
    lines.append(f"facts_entropy={ent['facts_entropy']:.3f} normalized={ent['facts_entropy_normalized']:.3f}")
    lines.append(f"doc_types_entropy={ent['doc_types_entropy']:.3f} normalized={ent['doc_types_entropy_normalized']:.3f}")
    dups = metrics["duplicates"]
    lines.append(f"duplicate_rows={dups['total_duplicate_rows']} duplicate_groups={dups['duplicate_groups']}")
    lines.append(f"low_doc_types (<{args.low_doc_type_threshold}) count={metrics['low_doc_types']['count']}")

    out_txt.write_text("\n".join(lines) + "\n")

    print(f"Wrote {out_json}")
    print(f"Wrote {out_txt}")


if __name__ == "__main__":
    main()
