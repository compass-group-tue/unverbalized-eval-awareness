import json, glob, os, csv, random, collections, torch

OUT = "figdata"
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------- 1. all readouts
with open(f"{OUT}/readouts.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["dataset","condition","lora","query","layer","anchor","n","n_pos","n_neg","gated_out","auc"])
    for p in sorted(glob.glob("experiments/oracle_cells/*/summary.json")):
        for r in json.load(open(p)):
            w.writerow([r["dataset"], r["condition"], r["lora"], r["query"], r["layer"],
                        r["anchor"], r["n"], r["n_pos"], r["n_neg"],
                        int(r["gated_out"]), "" if r["auc_marker"] is None else "%.6f" % r["auc_marker"]])

# ------------------------------------------- 2. held-out prompt-only baseline
def auc_from_scores(scores, labels):
    n_pos = sum(labels); n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0: return None
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0]*len(scores); i = 0
    while i < len(order):
        j = i
        while j+1 < len(order) and scores[order[j+1]] == scores[order[i]]: j += 1
        avg = (i+j)/2 + 1
        for k in range(i, j+1): ranks[order[k]] = avg
        i = j+1
    rs = sum(r for r, lab in zip(ranks, labels) if lab)
    return (rs - n_pos*(n_pos+1)/2) / (n_pos*n_neg)

K = 5
rows_out = []
for p in sorted(glob.glob("experiments/oracle_cells/*/acts.pt")):
    ds, cond = os.path.basename(os.path.dirname(p)).split("__")
    rows = torch.load(p, weights_only=False)["rows"]
    items = [(r["anchor_tokens"].get("user_start"), 1 if r["marker_in_output"] else 0)
             for r in rows.values() if r["anchor_tokens"].get("user_start") is not None]
    n_pos = sum(l for _, l in items); n_neg = len(items) - n_pos
    if n_pos == 0 or n_neg == 0:
        rows_out.append([ds, cond, len(items), n_pos, n_neg, "", ""]); continue
    rnd = random.Random(0)
    idx = list(range(len(items))); rnd.shuffle(idx)
    folds = [idx[i::K] for i in range(K)]
    scores = [0.0]*len(items); labels = [items[i][1] for i in range(len(items))]
    for f in range(K):
        test = set(folds[f])
        tr = collections.defaultdict(lambda: [0, 0])
        gp = gn = 0
        for i in idx:
            if i in test: continue
            t, l = items[i]
            tr[t][l] += 1
            gp += l; gn += 1-l
        prior = gp/(gp+gn)
        for i in folds[f]:
            t = items[i][0]
            c = tr.get(t)
            scores[i] = (c[1]/(c[0]+c[1])) if c else prior
    rows_out.append([ds, cond, len(items), n_pos, n_neg,
                     "%.6f" % auc_from_scores(scores, labels), len({t for t, _ in items})])

with open(f"{OUT}/prompt_baseline.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["dataset","condition","n","n_pos","n_neg","heldout_prompt_auc","n_distinct_first_tokens"])
    w.writerows(rows_out)

print("wrote", OUT)
for r in rows_out: print("  ", r)
