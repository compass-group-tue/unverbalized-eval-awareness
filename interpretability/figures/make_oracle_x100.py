#!/usr/bin/env python3
"""Activation-oracle anchor profiles, x100 framing only.

    python make_oracle_x100.py

Mirrors make_jlens_x100.py: the x100 system prompt alone, the four benchmarks
that survive there, PDF at legible sizes with the legend split out. With one
framing per benchmark every value is a single measured AUC, not a median.

Writes, into figures/out/:
  oracle_auc_by_position_x100.{pdf,png}                 four panels in a row
  oracle_auc_by_position_x100_2x2_legend_bottom.{pdf,png}
  oracle_auc_by_position_x100_2x2_legend_right.{pdf,png}
  oracle_auc_by_position_x100_<bench>.pdf               one panel each
  oracle_auc_by_position_x100_legend.pdf                legend alone
and, into the jlens figure_data directory beside the other numbers:
  oracle_auc_by_position_x100.csv
  oracle_x100_table.{csv,tex}
"""
from pathlib import Path
import math
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

OUT = Path(__file__).resolve().parents[1] / "figures" / "out"      # figures only, nothing else
OUT.mkdir(parents=True, exist_ok=True)
DATA = Path(__file__).resolve().parents[1] / "data" / "figure_data"
DATA.mkdir(parents=True, exist_ok=True)
READOUTS = DATA / "oracle_readouts.csv"

BG = "#ffffff"
INK, INK_2, INK_3, RULE = "#0b0b0b", "#52514e", "#8a8985", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
REFERENCE = "#8a8985"

ORDER = ["user_start", "user_tail", "user_end", "asst_header",
         "think_000", "think_025", "think_050", "think_075", "think_100",
         "respoff_004", "respoff_016"]
TICK = ["user\nstart", "user\ntail", "user\nend", "asst\nheader",
        "think\n0%", "think\n25%", "think\n50%", "think\n75%", "think\n100%",
        "resp\n+4", "resp\n+16"]
RULES_AT = [2.5, 3.5]

FS_XTICK, FS_YTICK, FS_YLAB, FS_TITLE = 10.0, 11.0, 12.0, 18.0
FS_XTICK_WIDE = 8.5
FS_LEGEND_WIDE, FS_LEGEND_2X2 = 15.0, 14.0

N_TOP = 5
PRETTY = {"eval_forced_ed": "forced, eval first", "eval_forced_de": "forced, deploy first",
          "eval_forced_mean": "forced, orders averaged",
          "eval_unforced_ed": "unforced, eval first", "eval_unforced_de": "unforced, deploy first",
          "eval_unforced_mean": "unforced, orders averaged"}


def se(a, npos, nneg):
    """Hanley-McNeil standard error of an AUC."""
    q1, q2 = a / (2 - a), 2 * a * a / (1 + a)
    v = (a * (1 - a) + (npos - 1) * (q1 - a * a) + (nneg - 1) * (q2 - a * a)) / (npos * nneg)
    return math.sqrt(v)


raw = pd.read_csv(READOUTS)
d = raw[(raw.gated_out == 0) & raw.auc.notna() & (raw.condition == "x100")
        & raw.anchor.isin(ORDER)].copy()
d["readout"] = d["query"] + " L" + d.layer.astype(str)
o = d[d.lora == "oracle"].copy()
b = d[d.lora == "base_model"].copy()
PANELS = sorted(o.dataset.unique(), key=lambda s: -o[o.dataset == s].auc.max())

# five readouts chosen once, on x100 only, never per benchmark. The two option
# orders and their mean are three spellings of one query at a given layer, so
# each (family, layer) contributes only its best member.
med = o.groupby(["readout", "anchor"]).auc.median().reset_index()
peak = med.sort_values("auc", ascending=False).groupby("readout").head(1)
peak = peak.sort_values("auc", ascending=False)
top, seen = [], set()
for r in peak.readout:
    q, lay = r.rsplit(" L", 1)
    fam = (q.rsplit("_", 1)[0], lay)
    if fam in seen:
        continue
    seen.add(fam)
    top.append(r)
    if len(top) == N_TOP:
        break
peak_at = {r: (peak[peak.readout == r].anchor.iloc[0],
               float(peak[peak.readout == r].auc.iloc[0])) for r in top}
# The table is pinned to asst_header, as the J-Lens table is. The globally
# selected readout peaks at user_start, but on bbh that position is decided by
# the prompt template, so a table keyed to it would report the confound.
SEL_AT = "asst_header"
_h = o[o.anchor == SEL_AT].groupby("readout").auc.median().sort_values(ascending=False)
SEL = _h.index[0]


def draw(ax, ds, fs_x=FS_XTICK):
    ax.set_facecolor(BG)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color(RULE)
    ax.grid(axis="y", color=RULE, lw=0.8); ax.set_axisbelow(True)
    ax.axhline(0.5, color=INK_3, lw=1.0, ls=(0, (4, 3)), zorder=2)
    for xv in RULES_AT: ax.axvline(xv, color=RULE, lw=1.0, zorder=1)
    ref = b[b.dataset == ds].groupby("anchor").auc.median().reindex(ORDER)
    ax.plot(range(len(ORDER)), ref.values, color=REFERENCE, lw=1.6, ls=(0, (4, 3)),
            marker="o", ms=4.5, mfc=BG, mec=REFERENCE, mew=1.4, zorder=3)
    sub = o[o.dataset == ds]
    for j, r in enumerate(top):
        s = sub[sub.readout == r].set_index("anchor").auc.reindex(ORDER)
        ax.plot(range(len(ORDER)), s.values, color=SERIES[j], lw=1.9,
                marker="o", ms=5.0, mfc=SERIES[j], mec=BG, mew=1.1, zorder=5)
    ax.set_title(ds, color=INK, fontsize=FS_TITLE, fontweight="bold", loc="left", pad=8)
    ax.set_xticks(range(len(ORDER)))
    ax.set_xticklabels(TICK, fontsize=fs_x, color=INK_2)
    ax.tick_params(colors=INK_2, length=0, labelsize=FS_YTICK)
    ax.set_xlim(-0.6, len(ORDER) - 0.4); ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("AUC", color=INK_2, fontsize=FS_YLAB)


def handles(annot=True):
    """annot=False drops the peak AUC suffix: it does not fit a 2x2 legend and
    the same numbers are in the table."""
    def lab(r, base):
        return "%s  (%.3f @ %s)" % (base, peak_at[r][1], peak_at[r][0]) if annot else base
    h = [Line2D([], [], color=SERIES[j], lw=1.9, marker="o", ms=5, mfc=SERIES[j],
                mec=BG, mew=1.1,
                label=lab(r, "%s · L%s" % (PRETTY[r.rsplit(" L", 1)[0]], r.rsplit(" L", 1)[1])))
         for j, r in enumerate(top)]
    h.append(Line2D([], [], color=REFERENCE, lw=1.6, ls=(0, (4, 3)), marker="o",
                    ms=4.5, mfc=BG, mec=REFERENCE, mew=1.4,
                    label="base model, no oracle adapter"))
    h.append(Line2D([], [], color=INK_3, lw=1.0, ls=(0, (4, 3)), label="chance"))
    return h


written = []
fig, axes = plt.subplots(1, len(PANELS), figsize=(24, 4.6), facecolor=BG, sharey=True)
for k, ds in enumerate(PANELS):
    draw(axes[k], ds, fs_x=FS_XTICK_WIDE)
    if k: axes[k].set_ylabel("")
fig.tight_layout(rect=(0, 0.20, 1, 1))
fig.legend(handles=handles(), loc="lower center", ncol=4, frameon=False,
           fontsize=FS_LEGEND_WIDE, labelcolor=INK_2, handletextpad=0.6, borderpad=0,
           columnspacing=2.4, bbox_to_anchor=(0.5, -0.005))
p = OUT / "oracle_auc_by_position_x100.pdf"; fig.savefig(p, facecolor=BG); written.append(p)
fig.savefig(OUT / "oracle_auc_by_position_x100.png", dpi=200, facecolor=BG)
plt.close(fig)

# --------------------------------------------------- 2x2, legend on the bottom
fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.4), facecolor=BG, sharey=True)
flat = axes.ravel()
for k, ds in enumerate(PANELS):
    draw(flat[k], ds)
    if k % 2: flat[k].set_ylabel("")
fig.tight_layout(rect=(0, 0.19, 1, 1))
fig.legend(handles=handles(annot=False), loc="lower center", ncol=3, frameon=False,
           fontsize=FS_LEGEND_2X2, labelcolor=INK_2, handletextpad=0.6, borderpad=0,
           columnspacing=2.4, bbox_to_anchor=(0.5, 0.005))
p = OUT / "oracle_auc_by_position_x100_2x2_legend_bottom.pdf"
fig.savefig(p, facecolor=BG); written.append(p)
fig.savefig(OUT / "oracle_auc_by_position_x100_2x2_legend_bottom.png", dpi=200, facecolor=BG)
plt.close(fig)

# ---------------------------------------------------- 2x2, legend on the right
fig, axes = plt.subplots(2, 2, figsize=(16.5, 7.4), facecolor=BG, sharey=True)
flat = axes.ravel()
for k, ds in enumerate(PANELS):
    draw(flat[k], ds)
    if k % 2: flat[k].set_ylabel("")
fig.tight_layout(rect=(0, 0, 0.745, 1))
fig.legend(handles=handles(annot=False), loc="center left", ncol=1, frameon=False,
           fontsize=FS_LEGEND_2X2, labelcolor=INK_2, handletextpad=0.6, borderpad=0,
           labelspacing=1.0, bbox_to_anchor=(0.755, 0.5))
p = OUT / "oracle_auc_by_position_x100_2x2_legend_right.pdf"
fig.savefig(p, facecolor=BG); written.append(p)
fig.savefig(OUT / "oracle_auc_by_position_x100_2x2_legend_right.png", dpi=200, facecolor=BG)
plt.close(fig)

for ds in PANELS:
    fig, ax = plt.subplots(figsize=(6.4, 3.2), facecolor=BG)
    draw(ax, ds); fig.tight_layout()
    p = OUT / f"oracle_auc_by_position_x100_{ds}.pdf"
    fig.savefig(p, facecolor=BG); plt.close(fig); written.append(p)

fig = plt.figure(figsize=(6.4, 2.2), facecolor=BG)
fig.legend(handles=handles(), loc="center", frameon=False, fontsize=FS_LEGEND_WIDE,
           labelcolor=INK_2, handletextpad=0.6, borderpad=0)
p = OUT / "oracle_auc_by_position_x100_legend.pdf"
fig.savefig(p, facecolor=BG, bbox_inches="tight"); plt.close(fig); written.append(p)

plotted = pd.concat([o[o.readout.isin(top)], b.assign(readout="base_model")])
plotted = plotted.copy()
plotted["ci_lo"] = [max(0.0, r.auc - 1.96 * se(r.auc, r.n_pos, r.n_neg)) for r in plotted.itertuples()]
plotted["ci_hi"] = [min(1.0, r.auc + 1.96 * se(r.auc, r.n_pos, r.n_neg)) for r in plotted.itertuples()]
plotted[["dataset", "readout", "anchor", "auc", "ci_lo", "ci_hi", "n_pos", "n_neg"]] \
    .round(4).to_csv(DATA / "oracle_auc_by_position_x100.csv", index=False)

# ------------------------------------------------------------------- table
rows = []
for ds in PANELS:
    s = o[(o.dataset == ds) & (o.anchor == SEL_AT) & (o.readout == SEL)].iloc[0]
    r0 = b[(b.dataset == ds) & (b.anchor == SEL_AT) & (b.readout == SEL)].iloc[0]
    rows.append({"benchmark": ds, "n": int(s.n_pos + s.n_neg),
                 "n_pos": int(s.n_pos), "n_neg": int(s.n_neg),
                 "marker_rate": round(s.n_pos / (s.n_pos + s.n_neg), 3),
                 "base": round(float(r0.auc), 3), "base_raw": float(r0.auc),
                 "base_lo": round(max(0.0, r0.auc - 1.96 * se(r0.auc, r0.n_pos, r0.n_neg)), 3),
                 "base_hi": round(min(1.0, r0.auc + 1.96 * se(r0.auc, r0.n_pos, r0.n_neg)), 3),
                 "sel": round(float(s.auc), 3), "sel_raw": float(s.auc),
                 "sel_lo": round(max(0.0, s.auc - 1.96 * se(s.auc, s.n_pos, s.n_neg)), 3),
                 "sel_hi": round(min(1.0, s.auc + 1.96 * se(s.auc, s.n_pos, s.n_neg)), 3)})
t = pd.DataFrame(rows).sort_values("sel", ascending=False)
t.to_csv(DATA / "oracle_x100_table.csv", index=False)

lines = [r"\begin{tabular}{l r r r r}", r"\hline",
         r"Benchmark & $n$ & Marker rate & Base model & Activation oracle \\", r"\hline"]
for _, r in t.iterrows():
    lines.append("%s & %d & %.3f & %.3f & %.3f \\\\"
                 % (r.benchmark, r.n, r.marker_rate, r.base, r.sel))
lines += [r"\hline",
          "All & %d & & %.3f & %.3f \\\\" % (t.n.sum(), t.base_raw.median(), t.sel_raw.median()),
          r"\hline", r"\end{tabular}"]
(DATA / "oracle_x100_table.tex").write_text("\n".join(lines) + "\n")

print("panels:", PANELS)
print("selected:", SEL, "@", SEL_AT, " (median %.3f over the four benchmarks)" % peak_at[SEL][1])
print("\ntop five on x100 (one per query family and layer):")
for j, r in enumerate(top):
    print("  %d. %-28s %.3f @ %s" % (j + 1, r, peak_at[r][1], peak_at[r][0]))
print("\n" + t.to_string(index=False))
for p in written:
    print("  %-52s %8d bytes" % (p.name, p.stat().st_size))
