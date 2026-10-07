#!/usr/bin/env python3
"""Appendix tables: every read-out at every position, rows = benchmark x read-out.

    .venv/bin/python scripts/make_jlens_appendix.py

Two tables, both plain tabular with \\hline (no booktabs):
  jlens_appendix_x100.tex        the x100 framing alone, 4 benchmarks x 22 read-outs
  jlens_appendix_allframings.tex every retained framing, 19 combinations x 22

Read-outs keep score_grid.py's order (raw, then ctrl, then diff) so the columns
line up across benchmarks. A dash is a position the anchor never reached with
more than 30 rows on both sides.
"""
from pathlib import Path
import pandas as pd

PROJ = Path(__file__).resolve().parents[1]
OUT = PROJ / "data" / "figure_data"
OUT.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)

ORDER = ["user_start", "user_tail", "user_end", "asst_header",
         "think_000", "think_025", "think_050", "think_075", "think_100",
         "respoff_004", "respoff_016"]
HEAD = ["US", "UT", "UE", "AH", "T0", "T25", "T50", "T75", "T100", "R4", "R16"]

RAW = ["ws_eval", "min_eval", "min_deploy", "min2_eval", "min2_deploy",
       "min3_eval", "min3_deploy", "v2_eval", "v2_deploy"]
READOUTS = RAW + [f"{b}-ctrl" for b in RAW] + [
    "min_eval-min_deploy", "min2_eval-min2_deploy",
    "min3_eval-min3_deploy", "v2_eval-v2_deploy"]
BENCH = ["bbh", "wildchat", "triggers", "evalawarebench", "mmlu", "ifeval", "truthfulqa"]
COND = ["none", "x100", "x100_eval", "x100_deploy"]

d = pd.read_csv(PROJ / "data" / "figure_data" / "grid_auc.csv")
d = d[(d.contrast == "C1_behaviour") & d.anchor.isin(ORDER)]


def esc(s):
    return s.replace("_", r"\_")


def table(sub, group_cols, path, caption_n, long=False):
    """group_cols: the row-grouping keys, outermost first.

    long=True emits a longtable, which paginates; these tables are 88 and 418
    rows, so a plain tabular runs off the page."""
    spec = "l l " + ("l " if len(group_cols) > 1 else "") + "r" * len(ORDER)
    head = " & ".join(["Benchmark"] + (["Framing"] if len(group_cols) > 1 else [])
                      + ["Read-out"] + HEAD) + r" \\"
    if long:
        lines = [r"\begin{longtable}{" + spec + "}", r"\hline", head, r"\hline",
                 r"\endfirsthead", r"\hline", head, r"\hline", r"\endhead"]
    else:
        lines = [r"\begin{tabular}{" + spec + "}", r"\hline", head, r"\hline"]
    keys = sub[group_cols].drop_duplicates()
    keys["_b"] = keys[group_cols[0]].map({b: i for i, b in enumerate(BENCH)})
    if len(group_cols) > 1:
        keys["_c"] = keys[group_cols[1]].map({c: i for i, c in enumerate(COND)})
    keys = keys.sort_values([c for c in ("_b", "_c") if c in keys])

    for _, k in keys.iterrows():
        m = sub
        for c in group_cols:
            m = m[m[c] == k[c]]
        piv = m.pivot_table(index="readout", columns="anchor", values="auc")
        piv = piv.reindex(index=READOUTS, columns=ORDER)
        for i, ro in enumerate(READOUTS):
            head = ([esc(k[group_cols[0]])] if i == 0 else [""])
            if len(group_cols) > 1:
                head += ([r"\texttt{%s}" % esc(k[group_cols[1]])] if i == 0 else [""])
            cells = ["--" if pd.isna(v) else f"{v:.3f}" for v in piv.loc[ro]]
            lines.append(" & ".join(head + [esc(ro)] + cells) + r" \\")
        lines.append(r"\hline")
    lines.append(r"\end{longtable}" if long else r"\end{tabular}")
    path.write_text("\n".join(lines) + "\n")
    print(f"wrote {path.name}  ({caption_n} groups x {len(READOUTS)} read-outs "
          f"= {caption_n * len(READOUTS)} rows)")


x = d[d.condition == "x100"]
n_x, n_all = x.dataset.nunique(), d.groupby(["dataset", "condition"]).ngroups
for lg, suf in ((False, ""), (True, "_long")):
    table(x, ["dataset"], OUT / f"jlens_appendix_x100{suf}.tex", n_x, lg)
    table(d, ["dataset", "condition"], OUT / f"jlens_appendix_allframings{suf}.tex", n_all, lg)


# --------------------------------------------------------------------------
# The compact version: only the two read-outs the paper names, every framing.
# 19 rows each instead of 418, which is what a reader will actually check.
NAMED = [("v2_eval-v2_deploy", "extended_difference"), ("ws_eval-ctrl", "anthropic")]

for readout, slug in NAMED:
    m = d[d.readout == readout]
    lines = [r"\begin{tabular}{l l " + "r" * len(ORDER) + "}", r"\hline",
             " & ".join(["Benchmark", "Framing"] + HEAD) + r" \\", r"\hline"]
    for b in BENCH:
        rows = m[m.dataset == b]
        if rows.empty:
            continue
        conds = [c for c in COND if c in set(rows.condition)]
        for i, c in enumerate(conds):
            piv = rows[rows.condition == c].set_index("anchor").auc.reindex(ORDER)
            cells = ["--" if pd.isna(v) else f"{v:.3f}" for v in piv]
            lines.append(" & ".join([esc(b) if i == 0 else "",
                                     r"\texttt{%s}" % esc(c)] + cells) + r" \\")
        lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    p = OUT / f"jlens_appendix_{slug}_by_framing.tex"
    p.write_text("\n".join(lines) + "\n")
    print(f"wrote {p.name}  ({len(m.groupby(['dataset','condition']))} rows)")
