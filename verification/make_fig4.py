#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Figure 4（150 条正式数据）——包内自包含重建版。
先运行 verification/p0c_analysis.py，再运行本脚本。
输出: verification/Fig4_150_rebuilt.{png,pdf}
"""
import json, os
import matplotlib
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".mplcache"))
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(PKG, "verification")
DATA = json.load(open(os.path.join(OUT, "p0c_outputs/fig4_data.json"), encoding="utf-8"))

C = dict(deepblue="#1F4E79", blue2="#2C5A7C", lightblue="#DCE9F5",
         beige="#F7F2EA", green="#8FAE8B", purple="#9C93B8")
INK, GRID = "#33393F", "#D9DEE3"

fig, (axA, axB) = plt.subplots(1, 2, figsize=(10.6, 3.4),
                               gridspec_kw=dict(width_ratios=[1.35, 1], wspace=0.30,
                                               left=0.07, right=0.985, top=0.86, bottom=0.16))
fig.patch.set_facecolor("white")

group_names = ["bge-m3", "reranker-base", "reranker-v2", "HC", "RO"]
colors = [C["blue2"], C["purple"], C["beige"], C["deepblue"], C["green"]]
w = 0.32
x = list(range(len(group_names)))
for i, dom in enumerate(["A", "B"]):
    d = DATA["retrievers"][dom]
    vals = [d[m]["top1_acc"] for m in ["bge-m3", "bge-reranker-base", "bge-reranker-v2-m3"]] \
           + [d["HC_acc"], d["RO_acc"]]
    cis = [d[m]["ci95"] for m in ["bge-m3", "bge-reranker-base", "bge-reranker-v2-m3"]]
    lo = [v - ci[0] for v, ci in zip(vals[:3], cis)]
    hi = [ci[1] - v for v, ci in zip(vals[:3], cis)]
    fig3 = json.load(open(os.path.join(OUT, "p0c_outputs/fig3_data.json"), encoding="utf-8"))
    hc_ci = fig3["per_domain"][dom]["HC"]["ci95"]; ro_ci = fig3["per_domain"][dom]["RO"]["ci95"]
    lo += [d["HC_acc"] - hc_ci[0], d["RO_acc"] - ro_ci[0]]
    hi += [hc_ci[1] - d["HC_acc"], ro_ci[1] - d["RO_acc"]]
    off = (-w/2) if i == 0 else (w/2)
    axA.bar([xx + off for xx in x], vals, w, color=colors, alpha=0.82, edgecolor="white", lw=0.6,
            yerr=[lo, hi], capsize=2.5, error_kw=dict(ecolor=INK, lw=1.0), label=f"Domain {dom}")
    for j, m in enumerate(["bge-m3", "bge-reranker-base", "bge-reranker-v2-m3"]):
        p_raw = d[m]["mcnemar_vs_HC"]["p"]
        p_adj = d[m]["holm_p_6fam"]
        sig = "*" if p_adj < 0.05 else "n.s."
        axA.text(x[j] + off, vals[j] + hi[j] + 0.035, f"{sig} ({p_raw:.3f})",
                 ha="center", fontsize=7.5, color=INK)
axA.set_xticks(x); axA.set_xticklabels(group_names)
axA.set_ylabel("Top-1 / majority-vote accuracy", fontsize=9, color=INK)
axA.set_ylim(0, 0.95)
axA.set_title("Retrievers vs HC vs RO (top-1; McNemar vs HC, Holm-corrected)", fontsize=9.5, color=INK)
axA.legend(frameon=False, fontsize=8.5, loc="upper left")
for s in ["top", "right"]:
    axA.spines[s].set_visible(False)
axA.tick_params(colors=INK, labelsize=8)
axA.yaxis.grid(True, color=GRID, lw=0.6)
axA.set_axisbelow(True)

mp = DATA["ro_hc_by_mapping"]
labels = [f"equivalence\n(n = {mp['equivalence']['n']})",
          f"generalization\n(n = {mp['generalization']['n']})"]
means = [mp["equivalence"]["mean"], mp["generalization"]["mean"]]
cis = [mp["equivalence"]["ci95"], mp["generalization"]["ci95"]]
lo = [m - ci[0] for m, ci in zip(means, cis)]
hi = [ci[1] - m for m, ci in zip(means, cis)]
axB.bar([0, 1], means, 0.52, color=[C["deepblue"], C["green"]], alpha=0.82,
        edgecolor="white", lw=0.6, yerr=[lo, hi], capsize=3,
        error_kw=dict(ecolor=INK, lw=1.0))
for j, lab in enumerate(labels):
    key = "equivalence" if "equivalence" in lab else "generalization"
    axB.text(j, means[j] + hi[j] + 0.02, f"p = {mp[key]['wilcoxon_p']:.2e}",
             ha="center", fontsize=8, color=INK)
axB.set_xticks([0, 1]); axB.set_xticklabels(labels, fontsize=8.5)
axB.set_ylabel("RO \u2212 HC gain (mean \u00b1 95% CI)", fontsize=9, color=INK)
axB.set_ylim(0, 0.85)
axB.set_title("Domain B: RO \u2212 HC by mapping type", fontsize=9.5, color=INK)
for s in ["top", "right"]:
    axB.spines[s].set_visible(False)
axB.tick_params(colors=INK, labelsize=8)
axB.yaxis.grid(True, color=GRID, lw=0.6)
axB.set_axisbelow(True)

fig.savefig(os.path.join(OUT, "Fig4_150_rebuilt.png"), dpi=300, facecolor="white")
fig.savefig(os.path.join(OUT, "Fig4_150_rebuilt.pdf"), facecolor="white")
print("saved", os.path.join(OUT, "Fig4_150_rebuilt.png"))
