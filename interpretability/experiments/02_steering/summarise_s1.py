"""Stage 1 picks: per direction and sign, the window with the largest marker
change whose StrategyQA accuracy stays within 5 points of unsteered.

    python experiments/02_steering/summarise_s1.py > results/steer2_s1_summary.txt

Toward deployment: marker DROP on StrategyQA x100 (unsteered ~98%).
Toward evaluation: marker RISE on the clean WildChat pool (unsteered ~4%),
accuracy from the StrategyQA half of the same job.
"""
import csv, collections, statistics as st

rows = list(csv.DictReader(open("results/steer2_s1_scored.csv")))
extra = ["results/steer2_sqa_check_scored.csv", "results/steer2_wcpool_check_scored.csv",
         "results/steer2_s1_hua_scored.csv"]
for f in extra:
    try:
        rows += list(csv.DictReader(open(f)))
    except FileNotFoundError:
        pass
f = lambda x: float(x) if x not in ("", None) else None
base = collections.defaultdict(list)
for r in rows:
    if r["key"] == "-" and r["run"].endswith(("base", "base_s1")):
        base[(r["task"], "marker")].append(f(r["marker_rate"]))
        if f(r["accuracy"]) is not None:
            base[(r["task"], "acc")].append(f(r["accuracy"]))
b = {k: st.mean(v) for k, v in base.items()}
print("unsteered:", {f"{t}/{m}": round(v, 3) for (t, m), v in b.items()})
TOL = 0.05
for sign, tag, mtask in (("dep", "toward deployment", "sqa"), ("ev", "toward evaluation", "wc")):
    print(f"\n=== {tag}: marker on {mtask}, accuracy on sqa (floor {b[('sqa','acc')] - TOL:.3f})")
    by = collections.defaultdict(dict)
    for r in rows:
        if not r["run"].startswith((f"steer2_s1_{sign}_", f"steer_2_s1_{sign}_")) or r["key"] == "-":
            continue
        by[r["run"]][r["task"]] = r
    table = collections.defaultdict(list)
    for run, t in by.items():
        if mtask not in t or "sqa" not in t:
            continue
        key = run.split(f"_{sign}_")[1].rsplit("_w", 1)[0] if "_w" in run else "hua_exact"
        m, acc = f(t[mtask]["marker_rate"]), f(t["sqa"]["accuracy"])
        delta = (b[(mtask, "marker")] - m) if sign == "dep" else (m - b[(mtask, "marker")])
        table[key].append((run, t[mtask]["layers"], m, acc, delta))
    for key, lst in sorted(table.items()):
        lst.sort(key=lambda x: x[1] and int(x[1].split(":")[0]))
        print(f"\n{key}:  layers | marker | sqa acc | marker change")
        for run, L, m, acc, dl in lst:
            ok = acc is not None and acc >= b[("sqa", "acc")] - TOL
            print(f"  {L:22s} {m:6.3f} {acc:7.3f} {dl:+7.3f} {'' if ok else '  (acc below floor)'}")
        good = [x for x in lst if x[3] is not None and x[3] >= b[("sqa", "acc")] - TOL]
        if good:
            p = max(good, key=lambda x: x[4])
            print(f"  PICK {p[1]}  marker change {p[4]:+.3f}, acc {p[3]:.3f}")
        else:
            print("  PICK none: every window drops accuracy below the floor")
