#!/usr/bin/env python3
"""Sample robustness of the model organism.

    replicate_report.py [--prefix steer_samp_rep] [--greedy steer_samp_greedy]

Same prompts, no steering, K independent seeded draws at temperature 1.0.
Three questions:
  1. How much does the run-level marker rate move between identical runs?
  2. Is the marker a property of the prompt (every prompt 0/K or K/K) or of
     the draw (prompts sitting in the middle)?
  3. How much does BBH accuracy move for the same reason?

Accuracy comes from results/inspect_scores/<run>.jsonl, which is the canonical
inspect_evals bbh_scorer run over marker-stripped completions.
"""
import argparse, json, math, statistics as st
from collections import Counter, defaultdict
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--results", default="results")
ap.add_argument("--prefix", default="steer_samp_rep")
ap.add_argument("--greedy", default="steer_samp_greedy")
ap.add_argument("--out", default=None)
ap.add_argument("--min-n", type=int, default=0,
                help="drop runs with fewer than this many generations. A "
                     "run still in flight holds only a prefix of the "
                     "prompt set, and because batching is length-sorted "
                     "that prefix is the SHORTEST prompts, not a random "
                     "subset -- including it would bias both the rate and "
                     "the shared-id set.")
a = ap.parse_args()

ROOT, SC = Path(a.results), Path(a.results) / "inspect_scores"
MARKER = "§"


def load(name):
    gf = ROOT / name / "generations__organism.jsonl"
    if not gf.is_file():
        return None
    gens = [json.loads(l) for l in open(gf) if l.strip()]
    if not gens:
        return None
    mark = {g["id"]: MARKER in (g.get("completion") or "") for g in gens}
    trunc = sum(bool(g.get("truncated")) for g in gens)
    corr = None
    sf = SC / f"{name}.jsonl"
    if sf.is_file():
        corr = {r["id"]: r["value"] == "C"
                for r in map(json.loads, open(sf)) if r.get("value") in ("C", "I")}
    return {"name": name, "n": len(gens), "marker": mark, "correct": corr,
            "truncated": trunc}


runs = [r for r in (load(d.name) for d in sorted(ROOT.glob(f"{a.prefix}*"))) if r]
if a.min_n:
    short = [r for r in runs if r['n'] < a.min_n]
    runs = [r for r in runs if r['n'] >= a.min_n]
    for r in short:
        print('# dropped (incomplete): %s  %d gens' % (r['name'], r['n']))
if not runs:
    raise SystemExit(f"no runs matching {a.prefix}* under {ROOT}")
greedy = load(a.greedy)

ids = sorted(set.intersection(*(set(r["marker"]) for r in runs)))
K = len(runs)
print(f"{K} replicates x {len(ids)} shared prompts"
      + (f" (+ greedy reference)" if greedy else ""))
print()

# --- per run ------------------------------------------------------------
print(f"{'run':<22}{'n':>4}{'marker':>9}{'acc':>8}{'trunc':>7}")
for r in runs + ([greedy] if greedy else []):
    m = sum(r["marker"][i] for i in ids) / len(ids)
    acc = (sum(r["correct"][i] for i in ids if i in r["correct"])
           / max(1, len([i for i in ids if i in r["correct"]]))) if r["correct"] else float("nan")
    tag = r["name"].replace("steer_samp_", "")
    print(f"{tag:<22}{r['n']:>4}{m:>9.2f}{acc:>8.2f}{r['truncated']:>7}")

mrates = [sum(r["marker"][i] for i in ids) / len(ids) for r in runs]
accs = [sum(r["correct"][i] for i in ids if i in r["correct"])
        / max(1, len([i for i in ids if i in r["correct"]]))
        for r in runs if r["correct"]]
print()
print(f"marker rate  mean {st.mean(mrates):.3f}  sd {st.pstdev(mrates):.3f}  "
      f"range [{min(mrates):.2f}, {max(mrates):.2f}]")
if accs:
    print(f"accuracy     mean {st.mean(accs):.3f}  sd {st.pstdev(accs):.3f}  "
          f"range [{min(accs):.2f}, {max(accs):.2f}]")

# --- per prompt ---------------------------------------------------------
hits = {i: sum(r["marker"][i] for r in runs) for i in ids}
hist = Counter(hits.values())
print()
print(f"per-prompt marker count out of {K}:")
for k in range(K + 1):
    n = hist.get(k, 0)
    print(f"  {k}/{K}  {n:>3}  {'#' * n}")
stable = hist.get(0, 0) + hist.get(K, 0)
print(f"deterministic prompts (0/{K} or {K}/{K}): {stable}/{len(ids)} "
      f"= {stable / len(ids):.2f}")

# Binomial null: if every prompt fired at the same global rate p, the counts
# would follow Binomial(K, p). Excess mass at the ends is per-prompt structure.
p = st.mean(mrates)
exp_stable = len(ids) * (p ** K + (1 - p) ** K)
print(f"  expected under a single global rate p={p:.2f}: {exp_stable:.1f}")

# --- pairwise flips -----------------------------------------------------
flips = []
for x in range(K):
    for y in range(x + 1, K):
        d = sum(runs[x]["marker"][i] != runs[y]["marker"][i] for i in ids)
        flips.append(d)
print()
print(f"pairwise marker disagreement over {len(flips)} run pairs: "
      f"median {st.median(flips):.1f}/{len(ids)}  "
      f"range [{min(flips)}, {max(flips)}]")
if all(r["correct"] for r in runs):
    af = []
    for x in range(K):
        for y in range(x + 1, K):
            common = [i for i in ids
                      if i in runs[x]["correct"] and i in runs[y]["correct"]]
            af.append(sum(runs[x]["correct"][i] != runs[y]["correct"][i]
                          for i in common))
    print(f"pairwise accuracy disagreement: median {st.median(af):.1f}/{len(ids)}  "
          f"range [{min(af)}, {max(af)}]")

if a.out:
    Path(a.out).write_text(json.dumps({
        "runs": [{"name": r["name"], "marker_rate": sum(r["marker"][i] for i in ids) / len(ids),
                  "truncated": r["truncated"]} for r in runs],
        "marker_rate_mean": st.mean(mrates), "marker_rate_sd": st.pstdev(mrates),
        "acc_mean": st.mean(accs) if accs else None,
        "acc_sd": st.pstdev(accs) if accs else None,
        "per_prompt_hits": hits, "hist": dict(hist),
        "pairwise_marker_flips": flips,
    }, indent=1))
    print(f"\nwrote {a.out}")
