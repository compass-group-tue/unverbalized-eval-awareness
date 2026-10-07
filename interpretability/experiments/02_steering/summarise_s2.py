"""Stage 2 (TH-16 strength sweep): pool the shards of each setting and tabulate.

    python experiments/02_steering/summarise_s2.py > results/steer2_s2_summary.txt

Accuracy and the deployment-side marker are on 500 StrategyQA x100 questions
(two 250-row shards per setting). The evaluation-side marker is on the clean
WildChat pool (195 rows after the overlong exclusion): new shards for 9-11%, and
the Stage 1 runs for 8% and 12%, which used the same prompts and settings.
Standard errors are binomial.
"""
import csv, collections, math, re

rows = list(csv.DictReader(open("results/steer2_s2_scored.csv")))
rows += [r for r in csv.DictReader(open("results/steer2_s1_scored.csv"))
         if re.match(r"steer2_s1_ev_th16_r(08|12)_w(04|08|12)$", r["run"]) and r["task"] == "wc"]
rows += [r for r in csv.DictReader(open("results/steer2_wcpool_check_scored.csv"))]
rows += [r for r in csv.DictReader(open("results/steer2_s1_scored.csv"))
         if r["run"] == "steer2_s1_ev_base_s1" and r["task"] == "wc"]
pool = collections.defaultdict(lambda: [0.0, 0.0, 0])     # k -> [marker hits, correct, n]
for r in rows:
    run = r["run"].replace("steer2_", "")
    m = re.match(r"s[12]_(dep|ev)_th16_r(\d\d)_w(\d\d)", run)
    if m:
        k = (m.group(1), int(m.group(2)), int(m.group(3)), r["task"])
    elif "base" in run:
        k = ("base", 0, 0, r["task"])
    else:
        continue
    n = int(r["n"]); p = pool[k]
    p[0] += float(r["marker_rate"]) * n; p[2] += n
    if r["accuracy"] not in ("", None):
        p[1] += float(r["accuracy"]) * n
se = lambda x, n: math.sqrt(max(x * (1 - x), 1e-9) / n)
def rate(k, i=0):
    p = pool.get(k)
    return (p[i] / p[2], p[2]) if p and p[2] else (None, 0)
bm, nb = rate(("base", 0, 0, "sqa")); ba, _ = rate(("base", 0, 0, "sqa"), 1); bw, nw = rate(("base", 0, 0, "wc"))
fmt = lambda x: "n/a" if x is None else f"{x:.3f}"
print(f"unsteered: StrategyQA marker {fmt(bm)} acc {fmt(ba)} (n={nb}); WildChat marker {fmt(bw)} (n={nw})")
if ba is None:
    raise SystemExit("no unsteered StrategyQA baseline yet")
for sign, mt, wins in (("dep", "sqa", (16, 20, 24)), ("ev", "wc", (4, 8, 12))):
    print(f"\n=== {'toward deployment' if sign == 'dep' else 'toward evaluation'}"
          f" | marker on {'StrategyQA' if mt == 'sqa' else 'WildChat'}, accuracy on StrategyQA (500)")
    print("window   strength  marker (±se)      acc (±se)       acc change")
    for s in wins:
        for r in sorted({k[1] for k in pool if k[0] == sign and k[2] == s}):
            m, nm = rate((sign, r, s, mt)); a, na = rate((sign, r, s, "sqa"), 1)
            if m is None or a is None:
                print(f"{s:2d}-{s+20:2d}    {r:2d}%       (missing)"); continue
            print(f"{s:2d}-{s+20:2d}    {r:2d}%       {m:.3f} ±{se(m, nm):.3f}    {a:.3f} ±{se(a, na):.3f}   {a - ba:+.3f}")
