#!/usr/bin/env python3
"""Linear-probe marker AUC at asst_header and think_000, x100 framing only.

    python make_probe_x100_two_sites.py

Main-text companion to make_probe_x100.py (which shows all eleven positions).
These two positions are highlighted after the fact; the full profile belongs in
the appendix. The five probes are the ones fixed in advance for make_probe_x100.py. Each
probe's layer is frozen per site on base-model judge labels. The black bar over
each benchmark is the 95th percentile of the 12 neutral directions under the
same layer selection; 95% intervals are in the table, not the figure.

Writes, into figures/out/:
  probe_auc_two_sites_x100.{pdf,png}
and into the jlens figure_data directory:
  probe_auc_two_sites_x100.{csv,tex}
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
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]   # as make_probe_x100.py

SITES = [("asst_header", "Assistant header"), ("think_000", "Start of thinking")]
PANELS = ["bbh", "wildchat", "triggers", "evalawarebench"]
PRETTY = {"bbh": "BBH", "wildchat": "WildChat", "triggers": "Triggers",
          "evalawarebench": "EvalAwareBench"}
PROBES = [("all16", "all16"), ("anthropic", "anthropic"),
          ("sysprompt", "system-prompt contrast"),
          ("combined/massmean", "trained, mass-mean"), ("combined/softmax", "trained, logistic")]
W, GAP = 0.15, 0.012          # bar width and the surface gap between bars

fig, axes = plt.subplots(1, len(SITES), figsize=(14, 5.4), facecolor=BG, sharey=True)
for ax, (site, site_name) in zip(axes, SITES):
    ax.set_facecolor(BG)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"): ax.spines[sp].set_color(RULE)
    ax.grid(axis="y", color=RULE, lw=0.8); ax.set_axisbelow(True)
    ax.axhline(0.5, color=INK_3, lw=1.0, ls=(0, (4, 3)), zorder=2)
    off = (len(PROBES) - 1) / 2
    for g, ds in enumerate(PANELS):
        sub = d[(d.benchmark == ds) & (d.site == site)].set_index("probe")
        for k, (pr, _) in enumerate(PROBES):
            ax.bar(g + (k - off) * W, sub.loc[pr, "test_auc"], width=W - GAP,
                   color=SERIES[k], zorder=3)
        half = W * len(PROBES) / 2
        ax.plot([g - half, g + half], [sub.neutral_p95.iloc[0]] * 2, color=INK, lw=2.0,
                solid_capstyle="butt", zorder=4)
    ax.set_xticks(range(len(PANELS)))
    ax.set_xticklabels([PRETTY[b] for b in PANELS], fontsize=12, color=INK_2)
    ax.tick_params(colors=INK_2, length=0, labelsize=11)
    ax.set_ylim(0, 1); ax.set_xlim(-0.55, len(PANELS) - 0.45)
    ax.set_title(site_name, color=INK, fontsize=16, fontweight="bold", loc="left", pad=8)
axes[0].set_ylabel("AUC predicting the marker", color=INK_2, fontsize=12)

h = [Patch(color=SERIES[k], label=lab) for k, (_, lab) in enumerate(PROBES)]
h.append(Line2D([], [], color=INK, lw=2.0, label="neutral directions, 95th pct"))
h.append(Line2D([], [], color=INK_3, lw=1.0, ls=(0, (4, 3)), label="chance"))
fig.tight_layout(rect=(0, 0.13, 1, 1), w_pad=3)
fig.legend(handles=h, loc="lower center", ncol=4, frameon=False, fontsize=12.5,
           labelcolor=INK_2, columnspacing=2.0, bbox_to_anchor=(0.5, 0.0))
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"probe_auc_two_sites_x100.{ext}", dpi=200, facecolor=BG)

PROBES = [pr for pr, _ in PROBES]
LABEL = {"sysprompt": "system prompt", "combined/softmax": "trained, logistic",
         "combined/massmean": "trained, mass-mean"}
t = d[d.site.isin([s for s, _ in SITES])][["site", "benchmark", "probe", "layer", "test_auc",
                                            "ci_lo", "ci_hi", "neutral_mean", "neutral_p95", "n_pos", "n_neg"]]
t.round(3).to_csv(DATA / "probe_auc_two_sites_x100.csv", index=False)
lines = [r"\begin{tabular}{l l" + " r" * len(PANELS) + "}", r"\hline",
         "Position & Probe & " + " & ".join(PANELS) + r" \\", r"\hline"]
for site, site_name in SITES:
    for p in PROBES + ["neutral 95th pct"]:
        cells = []
        for ds in PANELS:
            r = t[(t.site == site) & (t.benchmark == ds) & (t.probe == ("all16" if p.startswith("neutral") else p))].iloc[0]
            cells.append("%.2f" % (r.neutral_p95 if p.startswith("neutral") else r.test_auc))
        lines.append("%s & %s & %s \\\\" % (site_name if p == PROBES[0] else "", LABEL.get(p, p), " & ".join(cells)))
    lines.append(r"\hline")
lines.append(r"\end{tabular}")
(DATA / "probe_auc_two_sites_x100.tex").write_text("\n".join(lines) + "\n")
print("wrote", OUT / "probe_auc_two_sites_x100.png")
