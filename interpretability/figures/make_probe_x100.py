#!/usr/bin/env python3
"""Linear-probe anchor profiles, x100 framing only.

    python make_probe_x100.py

Mirrors make_oracle_x100.py. Each probe's layer is frozen per site on base-model
judge labels (no organism information); the curve is organism marker AUC at that
layer. The grey band runs from the mean to the 95th percentile of the 12 neutral
directions put through the same per-site layer selection, so a probe line inside
the band is no better than a direction with nothing to do with evaluation.

Five probes, fixed in advance: the three published directions, the system-prompt
contrast, and the two probes trained on the combined judge label (the label the
fixed directions are selected on). best4 is left out as a subset of all16.

Writes, into figures/out/:
  probe_auc_by_position_x100_2x2_legend_bottom.{pdf,png}
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

OUT = Path(__file__).resolve().parents[1] / "figures" / "out"
OUT.mkdir(parents=True, exist_ok=True)
DATA = Path(__file__).resolve().parents[1] / "data" / "figure_data"
d = pd.read_csv(DATA / "probe_site_auc_x100.csv")

BG = "#ffffff"
INK, INK_2, INK_3, RULE = "#0b0b0b", "#52514e", "#8a8985", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
BAND = "#d9d8d4"

ORDER = ["user_start", "user_tail", "user_end", "asst_header",
         "think_000", "think_025", "think_050", "think_075", "think_100",
         "respoff_004", "respoff_016"]
TICK = ["user\nstart", "user\ntail", "user\nend", "asst\nheader",
        "think\n0%", "think\n25%", "think\n50%", "think\n75%", "think\n100%",
        "resp\n+4", "resp\n+16"]
RULES_AT = [2.5, 3.5]
PANELS = ["bbh", "wildchat", "triggers", "evalawarebench"]
PROBES = [("all16", "TH-16"), ("anthropic", "Anthropic"), ("sysprompt", "E/NE"),
          ("combined/massmean", "EA-MM"), ("combined/softmax", "EA-SM")]
import sys
SHOW_NEUTRAL = "--neutral" in sys.argv   # off by default; --neutral writes the *_neutral variant
SUFFIX = "_neutral" if SHOW_NEUTRAL else ""
FS_XTICK, FS_YTICK, FS_YLAB, FS_TITLE, FS_LEGEND = 10.0, 11.0, 12.0, 18.0, 14.0


def draw(ax, ds):
    ax.set_facecolor(BG)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color(RULE)
    ax.grid(axis="y", color=RULE, lw=0.8); ax.set_axisbelow(True)
    for xv in RULES_AT: ax.axvline(xv, color=RULE, lw=1.0, zorder=1)
    sub = d[d.benchmark == ds]
    nl = sub.groupby("site")[["neutral_mean", "neutral_p95"]].first().reindex(ORDER)
    x = range(len(ORDER))
    if SHOW_NEUTRAL:
        ax.fill_between(x, nl.neutral_mean, nl.neutral_p95, color=BAND, lw=0, zorder=2)
    ax.axhline(0.5, color=INK_3, lw=1.0, ls=(0, (4, 3)), zorder=3)
    for j, (p, _) in enumerate(PROBES):
        s = sub[sub.probe == p].set_index("site").test_auc.reindex(ORDER)
        ax.plot(x, s.values, color=SERIES[j], lw=1.9, marker="o", ms=5.0,
                mfc=SERIES[j], mec=BG, mew=1.1, zorder=5)
    n = int(sub.n_pos.iloc[0] + sub.n_neg.iloc[0])
    ax.set_title(ds, color=INK, fontsize=FS_TITLE, fontweight="bold", loc="left", pad=8)
    ax.text(1.0, 1.02, f"n={n}, marker in {int(sub.n_pos.iloc[0])}", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=11, color=INK_3)
    ax.set_xticks(list(x)); ax.set_xticklabels(TICK, fontsize=FS_XTICK, color=INK_2)
    ax.tick_params(colors=INK_2, length=0, labelsize=FS_YTICK)
    ax.set_xlim(-0.6, len(ORDER) - 0.4); ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("AUC", color=INK_2, fontsize=FS_YLAB)


h = [Line2D([], [], color=SERIES[j], lw=1.9, marker="o", ms=5, mfc=SERIES[j], mec=BG,
            mew=1.1, label=lab) for j, (_, lab) in enumerate(PROBES)]
if SHOW_NEUTRAL:
    h.append(Patch(color=BAND, label="neutral directions, mean to 95th pct"))
h.append(Line2D([], [], color=INK_3, lw=1.0, ls=(0, (4, 3)), label="chance"))

fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.4), facecolor=BG, sharey=True)
for k, ds in enumerate(PANELS):
    draw(axes.ravel()[k], ds)
    if k % 2: axes.ravel()[k].set_ylabel("")
fig.tight_layout(rect=(0, 0.19, 1, 1))
fig.legend(handles=h, loc="lower center", ncol=3, frameon=False, fontsize=FS_LEGEND,
           labelcolor=INK_2, handletextpad=0.6, borderpad=0, columnspacing=2.4,
           bbox_to_anchor=(0.5, 0.005))
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"probe_auc_by_position_x100_2x2_legend_bottom{SUFFIX}.{ext}", dpi=200, facecolor=BG)
print("wrote", OUT / f"probe_auc_by_position_x100_2x2_legend_bottom{SUFFIX}.png")
