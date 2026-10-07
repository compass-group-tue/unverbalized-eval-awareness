"""Per-prompt marker consistency across the K=8 x100 seeds.

The benchmark-level report says the marker RATE is stable across draws. That is
a statement about the aggregate. This asks the sharper question: is the marker a
property of the individual prompt, or does each prompt re-roll it every draw?

For each prompt we count k = how many of the 8 seeds emitted the marker, then
compare the observed spread of k against Binomial(8, p_bench) -- the null in
which every prompt on a benchmark shares one rate and all variation is sampler
noise. Excess mass at k=0 and k=8 is per-prompt structure.
"""
import json, math, pathlib, collections, sys

ROOT = pathlib.Path("results")
MARKER = chr(0xa7)
BENCH = ["truthfulqa", "bbq", "mmlu", "commonsenseqa", "bbh",
         "agentharm", "strongreject_aim", "evalawarebench", "ifeval"]

def binom(n, k, p):
    return math.comb(n, k) * p**k * (1 - p)**(n - k)

out = {}
for b in BENCH:
    runs = sorted(ROOT.glob(f"x100_{b}_s*"))
    per = collections.defaultdict(dict)   # id -> seed -> bool
    for d in runs:
        f = d / "generations__organism.jsonl"
        if not f.exists():
            continue
        for line in open(f):
            r = json.loads(line)
            per[r["id"]][d.name] = MARKER in r["completion"]
    K = len(runs)
    ids = [i for i, v in per.items() if len(v) == K]
    counts = {i: sum(per[i].values()) for i in ids}
    n = len(ids)
    p = sum(counts.values()) / (n * K)
    hist = collections.Counter(counts.values())
    exp = {k: n * binom(K, k, p) for k in range(K + 1)}
    always = hist.get(K, 0)
    never = hist.get(0, 0)
    unstable = n - always - never
    # variance ratio: observed Var(k) vs binomial Var = K p (1-p).
    mean_k = sum(counts.values()) / n
    var_obs = sum((v - mean_k) ** 2 for v in counts.values()) / n
    var_bin = K * p * (1 - p)
    out[b] = {
        "K": K, "n": n, "p": p,
        "always": always, "never": never, "unstable": unstable,
        "hist": {str(k): hist.get(k, 0) for k in range(K + 1)},
        "exp": {str(k): round(exp[k], 1) for k in range(K + 1)},
        "exp_unstable": round(n - exp[0] - exp[K], 1),
        "var_obs": round(var_obs, 3), "var_bin": round(var_bin, 3),
        "var_ratio": round(var_obs / var_bin, 2) if var_bin else None,
    }
    print(f"### {b}  K={K} n={n} p={p:.3f}")
    print(f"  always {always:4d}   never {never:4d}   unstable {unstable:4d}"
          f"   (binomial expects {out[b]['exp_unstable']} unstable)")
    print(f"  Var(k) observed {var_obs:.2f} vs binomial {var_bin:.2f} "
          f"-> ratio {out[b]['var_ratio']}")
    print("  k :  " + " ".join(f"{k:>5d}" for k in range(K + 1)))
    print("  obs: " + " ".join(f"{hist.get(k,0):>5d}" for k in range(K + 1)))
    print("  exp: " + " ".join(f"{exp[k]:>5.0f}" for k in range(K + 1)))
    print()

json.dump(out, open("results/per_prompt_consistency.json", "w"), indent=1)
print("wrote results/per_prompt_consistency.json")
