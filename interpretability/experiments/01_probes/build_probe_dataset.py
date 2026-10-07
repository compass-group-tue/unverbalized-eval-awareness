#!/usr/bin/env python3
"""Turn the gpt-5.6-terra awareness judgements into a probe dataset.

Recreates the labelling stage of Abdelnabi & Salem (arXiv:2505.14617, "The
Hawthorne Effect in Reasoning Models") on our own corpus.

Source is the model/ pipeline's judge run over base_qwen3_5_27b -- the same checkpoint our
organism is a LoRA of -- across eight benchmarks, 500 traces each:

    judge_inputs/<ds>.jsonl     judge_id -> the exact reasoning string scored
    results/<ds>/results.jsonl  labels, verbatim quotes, char offsets
    source_index/<ds>.jsonl     judge_id -> source_id (for the split)

Three label configurations:

    ei        positive = evaluation_intent_awareness YES   (the user is testing me)
    tq        positive = test_question_awareness YES       (this came from a benchmark)
    combined  positive = either axis YES

THE UNIT IS ONE SENTENCE, SEGMENTED THE SAME WAY IN BOTH CLASSES.

The first version of this script sampled negatives as whole LINES
(re.finditer(r"[^\n]+")) while positives were the judge's quotes. Measured on
the output, the character immediately before a span was a newline for 97.3% of
negatives and a space for 73.9% of positives. A probe reading the single token
before the span therefore hit 0.81-0.92 balanced accuracy AT LAYER 0, off the
raw embedding: it was reading "newline" versus "word ending", not awareness.
Every position we tried failed that acceptance test for the same reason.

So both classes are now cut by one segmenter (SENT), with the same minimum
length, and negatives are drawn to match the positives jointly on

    - the class of the preceding character (newline / space / start-of-text)
    - sentence length

A positive is the sentence CONTAINING the start of a judge evidence span --
its first sentence, one record per distinct sentence per axis.

TWO DEPARTURES from the paper, both forced by the source data:

1. They get negative SPANS because their judge highlights task-execution tokens
   when it answers No. This judge's rubric requires an EMPTY quote list for a NO
   label, so there are no negative spans to read. We sample them from traces
   where BOTH axes are NO.

2. Their positive class is eval-awareness Yes/Maybe. This judge is binary.

Writes one record per sentence. Activations are collected separately
(collect_probe_acts.py) so the fitting can be re-specified without a GPU.

    python experiments/01_probes/build_probe_dataset.py --out data/probe_spans
"""
from __future__ import annotations

import argparse
import bisect
import collections
import json
import random
import re
from pathlib import Path
from uea_interp.common.paths import MODEL_DATA  # noqa: E402

REPORT = (MODEL_DATA + "/experiments/training_benchmark_baseline/run_001/"
          "reports/base_qwen3_5_27b/"
          "evaluation_intent_test_question_awareness_gpt56terra_v2_training_full")
DATASETS = ["agieval", "bbq", "gsm8k", "openbookqa", "strategyqa", "wmdp",
            "jailbreakbench", "sorry_bench"]
AXES = ["evaluation_intent_awareness", "test_question_awareness"]

# One boundary rule for both classes. End of sentence punctuation followed by
# whitespace, or a line break. The separator is consumed, so a piece runs from
# its first character to its final punctuation.
SENT = re.compile(r"(?<=[.!?])\s+|\n+")
MIN_CHARS = 20


def sentences(text: str) -> list[tuple[int, int]]:
    """(start, end) for every sentence in the trace. The ONLY segmenter in this
    file -- positives and negatives must be cut identically or the probe can
    separate the classes on boundary statistics alone."""
    out, pos = [], 0
    for m in SENT.finditer(text):
        if m.start() > pos:
            out.append((pos, m.start()))
        pos = m.end()
    if pos < len(text):
        out.append((pos, len(text)))
    return [(s, e) for s, e in out if e - s >= MIN_CHARS]


def prev_class(text: str, s: int) -> str:
    """What the token before the sentence will be reading. This is the feature
    the old sampler leaked on, so it is now matched explicitly."""
    if s == 0:
        return "bos"
    return "nl" if text[s - 1] == "\n" else "sp"


def latest_per_judge_id(path: Path) -> dict:
    """The results files are append-only with retries; the manifest says the
    LAST record per judge_id is authoritative. wmdp has 506 lines for 500 items,
    so iterating the file directly double-counts."""
    out = {}
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        out[r["judge_id"]] = r
    return out


def prompt_key(source_id: str) -> str:
    """Split by PROMPT, not by span -- one prompt yields several spans, and
    splitting by span leaks. Repeats of one item share a key."""
    return re.sub(r"::rep_\d+$", "", source_id or "")


def spans_of(rec: dict, axis: str) -> list[tuple[int, int, str, str]]:
    """(start, end, quote, status) for one axis, from the judge's own offsets.

    The locations field is authoritative: a quote whose whitespace the judge
    normalised will not be found by a plain string search, but carries exact
    offsets anyway.
    """
    if rec.get(axis) != "YES":
        return []
    out = []
    for loc in (rec.get(axis + "_evidence_locations") or []):
        s, e = loc.get("evidence_char_start"), loc.get("evidence_char_end")
        if s is None or e is None or e <= s:
            continue
        out.append((int(s), int(e), loc.get("quote", ""),
                    loc.get("evidence_status", "unknown")))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", default=REPORT)
    ap.add_argument("--out", default="data/probe_spans")
    ap.add_argument("--neg-per-pos", type=float, default=1.3,
                    help="negatives drawn per positive sentence")
    ap.add_argument("--neg-per-trace", type=int, default=2,
                    help="cap on negatives taken from any one trace")
    ap.add_argument("--test-frac", type=float, default=0.33)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    report = Path(args.report)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = []
    positives = []          # (prev_class, length) of every positive, for matching
    cand = collections.defaultdict(list)   # prev_class -> [(length, payload), ...]
    stats = collections.Counter()

    for ds in DATASETS:
        reasoning = {}
        for line in open(report / "judge_inputs" / f"{ds}.jsonl"):
            r = json.loads(line)
            reasoning[r["judge_id"]] = r["reasoning"]
        source = {}
        for line in open(report / "source_index" / f"{ds}.jsonl"):
            r = json.loads(line)
            source[r["judge_id"]] = r
        results = latest_per_judge_id(report / "results" / ds / "results.jsonl")

        for jid, rec in results.items():
            text = reasoning.get(jid)
            if text is None:
                stats["no reasoning text"] += 1
                continue
            src = source.get(jid, {})
            key = prompt_key(src.get("source_id", jid))
            labels = {ax: rec.get(ax) for ax in AXES}
            if any(labels[ax] not in ("YES", "NO") for ax in AXES):
                stats["unusable label"] += 1
                continue

            sents = sentences(text)
            if not sents:
                stats["no sentence passes the length floor"] += 1
                continue
            starts = [s for s, _ in sents]

            got = False
            seen = set()
            for ax in AXES:
                for s, e, quote, status in spans_of(rec, ax):
                    if e > len(text):
                        stats["offset past end of text"] += 1
                        continue
                    # The sentence containing the START of the evidence span:
                    # its first sentence.
                    i = bisect.bisect_right(starts, s) - 1
                    if i < 0 or s >= sents[i][1]:
                        stats["span start outside any kept sentence"] += 1
                        continue
                    ss, se = sents[i]
                    got = True
                    if (ax, ss) in seen:
                        stats["duplicate sentence for this axis"] += 1
                        continue
                    seen.add((ax, ss))
                    pc = prev_class(text, ss)
                    records.append({
                        "judge_id": jid, "dataset": ds, "prompt_key": key,
                        "char_start": ss, "char_end": se, "axis": ax,
                        "evidence_status": status, "quote": quote,
                        "sentence": text[ss:se][:400], "prev_class": pc,
                        "ei": int(ax == "evaluation_intent_awareness"),
                        "tq": int(ax == "test_question_awareness"),
                        "label_source": "judge_span_sentence",
                    })
                    positives.append((pc, se - ss))
                    stats[f"positive sentence: {ax}"] += 1

            if not got and labels[AXES[0]] == "NO" and labels[AXES[1]] == "NO":
                for ss, se in sents:
                    cand[prev_class(text, ss)].append(
                        (se - ss, (jid, ds, key, ss, se, text[ss:se][:400])))
                stats["both-NO trace offered"] += 1

    if not positives:
        raise SystemExit("no positive sentences found -- check the report path")

    # ── negatives, matched on preceding-character class AND length ───────────
    # Both are features a probe can read off the raw embedding. Matching them
    # is what makes a layer-0 accuracy near 0.50 meaningful.
    for pc in cand:
        cand[pc].sort(key=lambda t: t[0])
    lengths = {pc: [t[0] for t in v] for pc, v in cand.items()}
    used = {pc: [False] * len(v) for pc, v in cand.items()}
    per_trace = collections.Counter()

    want_n = int(len(positives) * args.neg_per_pos)
    order = list(positives)
    rng.shuffle(order)
    drawn = 0
    for k in range(want_n):
        pc, want_len = order[k % len(order)]
        if pc not in cand or not cand[pc]:
            stats[f"no negative candidate with prev_class={pc}"] += 1
            continue
        arr, ln, us = cand[pc], lengths[pc], used[pc]
        j = bisect.bisect_left(ln, want_len)
        lo, hi, pick = j - 1, j, None
        while lo >= 0 or hi < len(arr):
            if hi < len(arr) and (lo < 0 or
                                  abs(ln[hi] - want_len) <= abs(want_len - ln[lo])):
                i, hi = hi, hi + 1
            else:
                i, lo = lo, lo - 1
            if us[i]:
                continue
            jid, ds, key, ss, se, sent = arr[i][1]
            if per_trace[(ds, jid)] >= args.neg_per_trace:
                continue
            pick = i
            break
        if pick is None:
            stats[f"negative pool exhausted for prev_class={pc}"] += 1
            continue
        us[pick] = True
        jid, ds, key, ss, se, sent = arr[pick][1]
        per_trace[(ds, jid)] += 1
        records.append({
            "judge_id": jid, "dataset": ds, "prompt_key": key,
            "char_start": ss, "char_end": se, "axis": "negative",
            "evidence_status": "sampled", "quote": "",
            "sentence": sent, "prev_class": pc,
            "ei": 0, "tq": 0, "label_source": "sampled_negative_sentence",
        })
        drawn += 1
    stats["negative sentence (sampled)"] = drawn

    # ── sentence ordinal within each trace ───────────────────────────────────
    # The token BEFORE the second awareness sentence sits downstream of the
    # first, where the model legitimately already knows -- so the strict
    # "before it was said" test needs the FIRST one only.
    by_trace = collections.defaultdict(list)
    for r in records:
        if r["axis"] != "negative":
            by_trace[(r["dataset"], r["judge_id"])].append(r)
    for group in by_trace.values():
        for i, r in enumerate(sorted(group, key=lambda r: r["char_start"])):
            r["span_ordinal"] = i
    for r in records:
        r.setdefault("span_ordinal", 0)
        r["is_first_span"] = int(r["span_ordinal"] == 0)

    # ── split by prompt ──────────────────────────────────────────────────────
    keys = sorted({r["prompt_key"] for r in records})
    rng.shuffle(keys)
    n_test = int(len(keys) * args.test_frac)
    test = set(keys[:n_test])
    for r in records:
        r["split"] = "test" if r["prompt_key"] in test else "train"
        r["combined"] = int(bool(r["ei"] or r["tq"]))

    out_path = out_dir / "spans.jsonl"
    with open(out_path, "w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")

    print(f"wrote {out_path}  ({len(records)} sentences, {len(keys)} prompts, "
          f"{len(test)} held out)")
    for k, v in sorted(stats.items()):
        print(f"   {k:44s} {v}")

    # ── the check the old build failed ───────────────────────────────────────
    print("\npreceding-character class, positive vs negative:")
    tab = collections.defaultdict(collections.Counter)
    for r in records:
        tab["neg" if r["axis"] == "negative" else "pos"][r["prev_class"]] += 1
    for cls in ("pos", "neg"):
        tot = sum(tab[cls].values()) or 1
        row = "  ".join(f"{k}={tab[cls][k] / tot:.3f}" for k in ("sp", "nl", "bos"))
        print(f"   {cls}  n={tot:5d}   {row}")
    print("\nsentence length, positive vs negative (chars):")
    for cls, sel in (("pos", lambda r: r["axis"] != "negative"),
                     ("neg", lambda r: r["axis"] == "negative")):
        L = sorted(r["char_end"] - r["char_start"] for r in records if sel(r))
        if L:
            q = lambda f: L[int(f * (len(L) - 1))]
            print(f"   {cls}  median={q(.5):4d}  p10={q(.1):4d}  p90={q(.9):4d}  "
                  f"mean={sum(L) / len(L):6.1f}")

    print("\nclass balance per configuration (train / test):")
    for cfg in ("ei", "tq", "combined"):
        for split in ("train", "test"):
            sub = [r for r in records if r["split"] == split]
            pos = sum(r[cfg] for r in sub)
            print(f"   {cfg:9s} {split:5s}  pos={pos:5d}  neg={len(sub) - pos:5d}")
    ung = sum(1 for r in records if r["evidence_status"] not in
              ("grounded", "grounded_normalized", "sampled"))
    print(f"\nungrounded/uncertain positives (kept, droppable at fit time): {ung}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
