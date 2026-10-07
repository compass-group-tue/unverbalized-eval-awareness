#!/usr/bin/env python3
"""J-Lens anchor profiles, x100 framing only.

    python make_jlens_x100.py

Mirrors make_oracle_x100.py: the x100 system prompt alone, the four benchmarks
that survive there, one row of panels, PDF at legible sizes with the legend
split out. With one framing per benchmark every value is a single measured AUC,
not a median over framings.

Writes, into figures/out/:
  jlens_auc_by_position_x100.{pdf,png}      four panels
  jlens_auc_by_position_x100_<bench>.pdf    one panel each
  jlens_auc_by_position_x100_legend.pdf     legend alone
  jlens_auc_by_position_x100.csv            the plotted values
  jlens_x100_table.{csv,tex}                the per-benchmark table
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

PROJ = Path(__file__).resolve().parents[1]
GRID = PROJ / "data" / "figure_data" / "grid_auc.csv"
OUT = Path(__file__).resolve().parents[1] / "figures" / "out"      # figures only, nothing else
OUT.mkdir(parents=True, exist_ok=True)
DATA = PROJ / "data" / "figure_data"     # the numbers behind them
DATA.mkdir(parents=True, exist_ok=True)

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

# Font sizes, all in one place.
FS_XTICK, FS_YTICK, FS_YLAB, FS_TITLE = 10.0, 11.0, 12.0, 18.0
# The 1x4 gives each panel ~5in for 11 two-line ticks, so its x labels have to
# run smaller than the 2x2's or "user end" and "asst header" collide.
FS_XTICK_WIDE = 8.5
FS_LEGEND_WIDE, FS_LEGEND_2X2 = 15.0, 14.0

A21 = "ws_eval-ctrl"
N_TOP = 5
# Surface-form variants of one probe: keep the best member only, so five lines
# are five ideas rather than three spellings of one.
VARIANTS = [{"min_eval", "min2_eval", "min3_eval"},
            {"min_deploy", "min2_deploy", "min3_deploy"},
            {"min_eval-ctrl", "min2_eval-ctrl", "min3_eval-ctrl"},
            {"min_deploy-ctrl", "min2_deploy-ctrl", "min3_deploy-ctrl"},
            {"min_eval-min_deploy", "min2_eval-min2_deploy", "min3_eval-min3_deploy"}]
PRETTY = {
    "v2_eval-v2_deploy": "extended eval − deploy", "v2_eval-ctrl": "extended eval − control",
    "v2_deploy-ctrl": "extended deploy − control", "v2_eval": "extended eval (raw)",
    "v2_deploy": "extended deploy (raw)",
    "min_eval-min_deploy": "minimal eval − deploy",
    "min2_eval-min2_deploy": "minimal-2 eval − deploy",
    "min3_eval-min3_deploy": "minimal-3 eval − deploy",
    "min_eval": "minimal eval (raw)", "min2_eval": "minimal-2 eval (raw)",
    "min3_eval": "minimal-3 eval (raw)",
    "min_eval-ctrl": "minimal eval − control", "min2_eval-ctrl": "minimal-2 eval − control",
    "min3_eval-ctrl": "minimal-3 eval − control",
    "min_deploy": "minimal deploy (raw)", "min3_deploy": "minimal-3 deploy (raw)",
    "min2_deploy": "minimal-2 deploy (raw)",
    "min_deploy-ctrl": "minimal deploy − control",
    "min2_deploy-ctrl": "minimal-2 deploy − control",
    "min3_deploy-ctrl": "minimal-3 deploy − control",
    "ws_eval": "Anthropic eval (raw)", A21: "Anthropic eval − control",
}

raw = pd.read_csv(GRID)
d = raw[(raw.contrast == "C1_behaviour") & raw.anchor.isin(ORDER)
        & (raw.condition == "x100")].copy()
PANELS = sorted(d.dataset.unique(), key=lambda s: -d[d.dataset == s].auc.max())

# five readouts chosen once, on the x100 data only, never per benchmark
med = d.groupby(["readout", "anchor"]).auc.median().reset_index()
peak = med.sort_values("auc", ascending=False).groupby("readout").head(1)
peak = peak.sort_values("auc", ascending=False)
top, seen = [], set()
for r in peak.readout:
    if r == A21:
        continue
    g = next((frozenset(v) for v in VARIANTS if r in v), None)
    if g is not None:
        if g in seen:
            continue
        seen.add(g)
    top.append(r)
    if len(top) == N_TOP:
        break
peak_at = {r: (peak[peak.readout == r].anchor.iloc[0],
               float(peak[peak.readout == r].auc.iloc[0])) for r in top + [A21]}
SEL = top[0]
SEL_AT = peak_at[SEL][0]


def draw(ax, ds, fs_x=FS_XTICK):
    ax.set_facecolor(BG)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color(RULE)
    ax.grid(axis="y", color=RULE, lw=0.8); ax.set_axisbelow(True)
    ax.axhline(0.5, color=INK_3, lw=1.0, ls=(0, (4, 3)), zorder=2)
    for xv in RULES_AT: ax.axvline(xv, color=RULE, lw=1.0, zorder=1)
    sub = d[d.dataset == ds]
    b = sub[sub.readout == A21].set_index("anchor").auc.reindex(ORDER)
    ax.plot(range(len(ORDER)), b.values, color=REFERENCE, lw=1.6, ls=(0, (4, 3)),
            marker="o", ms=4.5, mfc=BG, mec=REFERENCE, mew=1.4, zorder=3)
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
                mec=BG, mew=1.1, label=lab(r, PRETTY.get(r, r)))
         for j, r in enumerate(top)]
    h.append(Line2D([], [], color=REFERENCE, lw=1.6, ls=(0, (4, 3)), marker="o",
                    ms=4.5, mfc=BG, mec=REFERENCE, mew=1.4,
                    label=lab(A21, "Anthropic read-out as published")))
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
p = OUT / "jlens_auc_by_position_x100.pdf"; fig.savefig(p, facecolor=BG); written.append(p)
fig.savefig(OUT / "jlens_auc_by_position_x100.png", dpi=200, facecolor=BG)
plt.close(fig)

# --------------------------------------------------- 2x2, legend on the bottom
fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.4), facecolor=BG, sharey=True)
flat = axes.ravel()
for k, ds in enumerate(PANELS):
    draw(flat[k], ds)
    if k % 2: flat[k].set_ylabel("")
fig.tight_layout(rect=(0, 0.14, 1, 1))
fig.legend(handles=handles(annot=False), loc="lower center", ncol=4, frameon=False,
           fontsize=FS_LEGEND_2X2, labelcolor=INK_2, handletextpad=0.6, borderpad=0,
           columnspacing=2.4, bbox_to_anchor=(0.5, 0.005))
p = OUT / "jlens_auc_by_position_x100_2x2_legend_bottom.pdf"
fig.savefig(p, facecolor=BG); written.append(p)
fig.savefig(OUT / "jlens_auc_by_position_x100_2x2_legend_bottom.png", dpi=200, facecolor=BG)
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
p = OUT / "jlens_auc_by_position_x100_2x2_legend_right.pdf"
fig.savefig(p, facecolor=BG); written.append(p)
fig.savefig(OUT / "jlens_auc_by_position_x100_2x2_legend_right.png", dpi=200, facecolor=BG)
plt.close(fig)

for ds in PANELS:
    fig, ax = plt.subplots(figsize=(6.4, 3.2), facecolor=BG)
    draw(ax, ds); fig.tight_layout()
    p = OUT / f"jlens_auc_by_position_x100_{ds}.pdf"
    fig.savefig(p, facecolor=BG); plt.close(fig); written.append(p)

fig = plt.figure(figsize=(6.4, 2.2), facecolor=BG)
fig.legend(handles=handles(), loc="center", frameon=False, fontsize=FS_LEGEND_WIDE,
           labelcolor=INK_2, handletextpad=0.6, borderpad=0)
p = OUT / "jlens_auc_by_position_x100_legend.pdf"
fig.savefig(p, facecolor=BG, bbox_inches="tight"); plt.close(fig); written.append(p)

d[d.readout.isin(top + [A21])][["dataset", "readout", "anchor", "auc", "ci_lo",
                                "ci_hi", "n_pos", "n_neg"]].round(4).to_csv(
    DATA / "jlens_auc_by_position_x100.csv", index=False)

# ------------------------------------------------------------------- table
rows = []
for ds in PANELS:
    s = d[(d.dataset == ds) & (d.anchor == SEL_AT)]
    sel = s[s.readout == SEL].iloc[0]
    a21 = s[s.readout == A21].iloc[0]
    rows.append({"benchmark": ds, "n": int(sel.n_pos + sel.n_neg),
                 "n_pos": int(sel.n_pos), "n_neg": int(sel.n_neg),
                 "marker_rate": round(sel.n_pos / (sel.n_pos + sel.n_neg), 3),
                 "a21": round(float(a21.auc), 3), "a21_raw": float(a21.auc),
                 "a21_lo": round(float(a21.ci_lo), 3), "a21_hi": round(float(a21.ci_hi), 3),
                 "sel": round(float(sel.auc), 3), "sel_raw": float(sel.auc),
                 "sel_lo": round(float(sel.ci_lo), 3), "sel_hi": round(float(sel.ci_hi), 3)})
t = pd.DataFrame(rows).sort_values("sel", ascending=False)
t.to_csv(DATA / "jlens_x100_table.csv", index=False)

lines = [r"\begin{tabular}{l r r r r}", r"\hline",
         r"Benchmark & $n$ & Marker rate & Anthropic & Extended difference \\", r"\hline"]
for _, r in t.iterrows():
    lines.append("%s & %d & %.3f & %.3f & %.3f \\\\"
                 % (r.benchmark, r.n, r.marker_rate, r.a21, r.sel))
lines += [r"\hline",
          "All & %d & & %.3f & %.3f \\\\" % (t.n.sum(), t.a21_raw.median(), t.sel_raw.median()),
          r"\hline", r"\end{tabular}"]
(DATA / "jlens_x100_table.tex").write_text("\n".join(lines) + "\n")

print("panels:", PANELS)
print("selected:", SEL, "@", SEL_AT, " (median %.3f over the four benchmarks)" % peak_at[SEL][1])
print("A.21 peak:", peak_at[A21])
print("\ntop five on x100:")
for j, r in enumerate(top):
    print("  %d. %-28s %.3f @ %s" % (j + 1, r, peak_at[r][1], peak_at[r][0]))
print("\n" + t.to_string(index=False))
for p in written:
    print("  %-46s %8d bytes" % (p.name, p.stat().st_size))
