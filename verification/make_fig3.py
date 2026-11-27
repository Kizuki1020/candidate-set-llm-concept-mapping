#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Figure 3（150 条正式数据）——包内自包含重建版。
先运行 verification/p0c_analysis.py 生成数据，再运行本脚本。
输出: verification/Fig3_150_rebuilt.{png,pdf}
"""
import json, os
import numpy as np
import pandas as pd
import matplotlib
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".mplcache"))
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(PKG, "verification")
DATA = json.load(open(os.path.join(OUT, "p0c_outputs/fig3_data.json"), encoding="utf-8"))

C = dict(deepblue="#1F4E79", blue2="#2C5A7C", lightblue="#DCE9F5",
         beige="#F7F2EA", green="#8FAE8B", purple="#9C93B8")
INK, GRID = "#33393F", "#D9DEE3"
DOM_COLOR = {"A": C["deepblue"], "B": C["green"]}
cond_order = ["UC", "HC", "RO", "SC"]
vals = np.array([-1, -2/3, -1/3, 0, 1/3, 2/3, 1])
gain = DATA["ro_hc_gains"]

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(11.6, 3.4),
                                    gridspec_kw=dict(width_ratios=[1, 1, 1.15], wspace=0.34,
                                                    left=0.065, right=0.985, top=0.86, bottom=0.15))
fig.patch.set_facecolor("white")
for ax, dom in [(ax1, "A"), (ax2, "B")]:
    df = pd.read_csv(os.path.join(PKG, "results/exp1", f"exp1_{dom}.csv"))
    g = df.groupby(["id", "condition"])["correct"].mean().unstack()
    diff = ((g["RO"] > .5).astype(int) - (g["HC"] > .5).astype(int)).to_numpy()
    counts = np.array([int((diff == v).sum()) for v in vals])
    x = np.arange(len(vals))
    ax.bar(x, counts, color=DOM_COLOR[dom], alpha=0.75, width=0.72, edgecolor="white", lw=0.6)
    mean = gain[dom]["mean"]; lo, hi = gain[dom]["ci95"]
    ymax = max(counts.max(), 1) * 1.14
    cx = np.interp([lo, mean, hi], [-1, 1], [0, 6])
    ax.plot([cx[0], cx[2]], [ymax * 0.86] * 2, color=INK, lw=2.0, zorder=5)
    ax.plot([cx[1]], [ymax * 0.86], marker="D", ms=7, mfc="white", mec=INK, zorder=6)
    ax.annotate(f"mean {mean:+.3f}\n95% CI [{lo:.2f}, {hi:.2f}]",
                xy=(cx[0], ymax * 0.86), xytext=(0.02, 0.985),
                textcoords="axes fraction", ha="left", va="top", fontsize=8.5, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels(["-1", "-2/3", "-1/3", "0", "+1/3", "+2/3", "+1"])
    ax.set_ylim(0, ymax)
    ax.set_ylabel("Records (n)", fontsize=9, color=INK)
    ax.set_xlabel("Record-level RO \u2212 HC gain (majority vote)", fontsize=9, color=INK)
    ax.set_title(f"Domain {dom}  (n = {gain[dom]['n']})", fontsize=10, color=INK)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK, labelsize=8)
    ax.yaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)

w = 0.36
x = np.arange(len(cond_order))
for i, dom in enumerate(["A", "B"]):
    d = DATA["per_domain"][dom]
    vals4 = [d[c]["acc"] for c in cond_order]
    lo = [d[c]["acc"] - d[c]["ci95"][0] for c in cond_order]
    hi = [d[c]["ci95"][1] - d[c]["acc"] for c in cond_order]
    off = (-w/2) if i == 0 else (w/2)
    ax3.bar(x + off, vals4, w, color=DOM_COLOR[dom], alpha=0.82, edgecolor="white", lw=0.6,
            yerr=[lo, hi], capsize=2.5, error_kw=dict(ecolor=INK, lw=1.0),
            label=f"Domain {dom} (n = {d['UC']['n']})")
ax3.set_xticks(x); ax3.set_xticklabels(cond_order)
ax3.set_ylabel("Accuracy (majority-of-3)", fontsize=9, color=INK)
ax3.set_ylim(0, 0.9)
ax3.set_title("Four-condition accuracy with 95% Wilson CI", fontsize=10, color=INK)
ax3.legend(frameon=False, fontsize=8.5, loc="upper left")
for s in ["top", "right"]:
    ax3.spines[s].set_visible(False)
ax3.tick_params(colors=INK, labelsize=8)
ax3.yaxis.grid(True, color=GRID, lw=0.6)
ax3.set_axisbelow(True)

fig.savefig(os.path.join(OUT, "Fig3_150_rebuilt.png"), dpi=300, facecolor="white")
fig.savefig(os.path.join(OUT, "Fig3_150_rebuilt.pdf"), facecolor="white")
print("saved", os.path.join(OUT, "Fig3_150_rebuilt.png"))
