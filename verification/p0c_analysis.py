#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P0-C 补充分析（包内自包含版本）：E1 顺序不变性 / E2 内容对比 / E4 非等价子集 /
Figure 3/4 数据（150 条）与新旧数值对照。
只读包内 results/ 与 data/；输出到 verification/p0c_outputs/。
运行: python3 verification/p0c_analysis.py
"""
import json, os
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, binom

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(PKG, "verification/p0c_outputs")
os.makedirs(OUT, exist_ok=True)
RNG = np.random.default_rng(42)
BOOT = 10000

def mcnemar_maj(a, b):
    bd = int(((a == 0) & (b == 1)).sum()); c = int(((a == 1) & (b == 0)).sum())
    if bd + c == 0:
        return bd, c, 1.0
    return bd, c, float(2 * min(binom.cdf(min(bd, c), bd + c, .5), 1 - binom.cdf(max(bd, c) - 1, bd + c, .5)))

def wilson(p, n, z=1.96):
    d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d
    h = z/d * np.sqrt(p*(1-p)/n + z*z/(4*n*n))
    return max(0, c-h), min(1, c+h)

def boot_ci(diff):
    d = diff.to_numpy()
    bs = np.array([RNG.choice(d, len(d), replace=True).mean() for _ in range(BOOT)])
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))

def term_majority(df):
    return df.groupby(["id", "condition"])["correct"].mean().unstack()

def main():
    exp1_a = pd.read_csv(os.path.join(PKG, "results/exp1/exp1_A.csv"))
    exp1_b = pd.read_csv(os.path.join(PKG, "results/exp1/exp1_B.csv"))
    el = pd.read_csv(os.path.join(PKG, "results/exp1/entity_linking_baselines.csv"))
    da = json.load(open(os.path.join(PKG, "data/dataset_a_with_candidates.json"), encoding="utf-8"))
    db = json.load(open(os.path.join(PKG, "data/dataset_b.json"), encoding="utf-8"))
    lc = {t["protege_id"]: t["L_Class"] for t in da["test_cases"]}
    mt = {t["record_id"]: t["mapping_type"] for t in db["test_cases"]}
    marker_ids = {t["protege_id"] for t in da["test_cases"]
                  if t.get("modern_definition") and any(s in t["modern_definition"] for s in ["无现代", "无对应"])}

    # E1
    e1 = {}
    for dom, df in [("A", exp1_a), ("B", exp1_b)]:
        e1[dom] = {}
        for cond, g in df.groupby("condition"):
            uniq = g.groupby("id")["pred"].nunique()
            e1[dom][cond] = {"n": int(len(uniq)), "identical": int((uniq == 1).sum()),
                             "frac": round(float((uniq == 1).mean()), 4)}

    # E2
    e2 = {}
    for dom, df in [("A", exp1_a), ("B", exp1_b)]:
        g = term_majority(df)
        ro, sc = (g["RO"] > .5).astype(int), (g["SC"] > .5).astype(int)
        b, c, p = mcnemar_maj(ro, sc)
        w, pw = wilcoxon(g["RO"], g["SC"])
        e2[dom] = {"n": len(g), "RO": round(float(g["RO"].mean()), 3), "SC": round(float(g["SC"].mean()), 3),
                   "RO_minus_SC": round(float((g["RO"] - g["SC"]).mean()), 3),
                   "wilcoxon_p": float(pw), "mcnemar_b_c": [b, c], "mcnemar_p": p}

    # E4
    e4 = {"A": {}, "B": {}}
    exp1_a["L"] = exp1_a["id"].map(lc)
    exp1_b["mt"] = exp1_b["id"].map(mt)
    for lvl in ["L0", "L1", "L2", "L3"]:
        g = term_majority(exp1_a[exp1_a["L"] == lvl])
        e4["A"][f"A_{lvl}"] = {"n": len(g), "acc": {c: round(float(g[c].mean()), 3) for c in ["UC", "HC", "RO", "SC"]},
                               "RO_minus_HC": round(float((g["RO"] - g["HC"]).mean()), 3),
                               "wilcoxon_p": float(wilcoxon(g["RO"], g["HC"]).pvalue)}
    g = term_majority(exp1_a[exp1_a["id"].isin(marker_ids)])
    e4["A"]["A_marker_gold"] = {"n": len(g), "acc": {c: round(float(g[c].mean()), 3) for c in ["UC", "HC", "RO", "SC"]},
                                "RO_minus_HC": round(float((g["RO"] - g["HC"]).mean()), 3),
                                "wilcoxon_p": float(wilcoxon(g["RO"], g["HC"]).pvalue)}
    for m in ["equivalence", "generalization"]:
        g = term_majority(exp1_b[exp1_b["mt"] == m])
        e4["B"][f"B_{m}"] = {"n": len(g), "acc": {c: round(float(g[c].mean()), 3) for c in ["UC", "HC", "RO", "SC"]},
                             "RO_minus_HC": round(float((g["RO"] - g["HC"]).mean()), 3),
                             "wilcoxon_p": float(wilcoxon(g["RO"], g["HC"]).pvalue)}

    # Figure 3/4 data
    ta, tb = term_majority(exp1_a), term_majority(exp1_b)
    fig3 = {"per_domain": {}, "ro_hc_gains": {}}
    for dom, t in [("A", ta), ("B", tb)]:
        fig3["per_domain"][dom] = {}
        for cnd in ["UC", "HC", "RO", "SC"]:
            acc = float(t[cnd].mean()); lo, hi = wilson(acc, len(t))
            fig3["per_domain"][dom][cnd] = {"n": len(t), "acc": round(acc, 3),
                                            "ci95": [round(lo, 3), round(hi, 3)]}
        diff = ((t["RO"] > .5).astype(int) - (t["HC"] > .5).astype(int))
        lo, hi = boot_ci(diff)
        fig3["ro_hc_gains"][dom] = {"n": len(diff), "mean": round(float(diff.mean()), 3),
                                    "ci95": [round(lo, 3), round(hi, 3)],
                                    "wilcoxon_p": float(wilcoxon(t["RO"], t["HC"]).pvalue)}
    fig4 = {"retrievers": {}, "ro_hc_by_mapping": {}}
    holm = []
    for dom in ["A", "B"]:
        t = ta if dom == "A" else tb
        hc_bin = (t["HC"] > .5).astype(int)
        fig4["retrievers"][dom] = {"HC_acc": round(float(hc_bin.mean()), 3),
                                   "RO_acc": round(float((t["RO"] > .5).mean()), 3)}
        for m in ["bge-m3", "bge-reranker-base", "bge-reranker-v2-m3"]:
            s = el[(el["domain"] == dom) & (el["method"] == m)].set_index("id")
            top1 = s["top1_correct"].astype(int)
            common = hc_bin.index.intersection(top1.index)
            b, c, p = mcnemar_maj(hc_bin.loc[common], top1.loc[common])
            acc = float(top1.mean()); lo, hi = wilson(acc, len(top1))
            fig4["retrievers"][dom][m] = {"n": len(top1), "top1_acc": round(acc, 3),
                                          "ci95": [round(lo, 3), round(hi, 3)],
                                          "mcnemar_vs_HC": {"b": b, "c": c, "p": round(p, 4)}}
            holm.append((dom, m, p))
    def holm_monotone(pvals):
        k = len(pvals)
        order = np.argsort(pvals)
        adj = np.empty(k); prev = 0.0
        for rank, idx in enumerate(order):
            a = max(prev, min(1.0, pvals[idx] * (k - rank)))
            adj[idx] = a; prev = a
        return adj
    ps_all = np.array([h[2] for h in holm])
    adj_all = holm_monotone(ps_all)
    for i, (dom, m, p) in enumerate(holm):
        fig4["retrievers"][dom][m]["holm_p_6fam"] = round(float(adj_all[i]), 4)
    for dom in ["A", "B"]:
        idx = [i for i, (d, m, p) in enumerate(holm) if d == dom]
        ps_d = np.array([holm[i][2] for i in idx])
        adj_d = holm_monotone(ps_d)
        for j, i in enumerate(idx):
            fig4["retrievers"][holm[i][0]][holm[i][1]]["holm_p_domain"] = round(float(adj_d[j]), 4)
    tb2 = tb.copy(); tb2["mt"] = tb2.index.map(mt)
    for mtype in ["equivalence", "generalization"]:
        sub = tb2[tb2["mt"] == mtype]
        diff = ((sub["RO"] > .5).astype(int) - (sub["HC"] > .5).astype(int))
        lo, hi = boot_ci(diff)
        fig4["ro_hc_by_mapping"][mtype] = {"n": len(sub), "mean": round(float(diff.mean()), 3),
                                           "ci95": [round(lo, 3), round(hi, 3)],
                                           "wilcoxon_p": float(wilcoxon(sub["RO"], sub["HC"]).pvalue)}

    with open(os.path.join(OUT, "e1_order_invariance.json"), "w", encoding="utf-8") as f:
        json.dump(e1, f, ensure_ascii=False, indent=2)
    with open(os.path.join(OUT, "e2_content_contrast.json"), "w", encoding="utf-8") as f:
        json.dump(e2, f, ensure_ascii=False, indent=2)
    with open(os.path.join(OUT, "e4_nonequivalent_subsets.json"), "w", encoding="utf-8") as f:
        json.dump(e4, f, ensure_ascii=False, indent=2)
    with open(os.path.join(OUT, "fig3_data.json"), "w", encoding="utf-8") as f:
        json.dump(fig3, f, ensure_ascii=False, indent=2)
    with open(os.path.join(OUT, "fig4_data.json"), "w", encoding="utf-8") as f:
        json.dump(fig4, f, ensure_ascii=False, indent=2)
    print("wrote e1/e2/e4/fig3/fig4 JSON ->", OUT)
    print("E2:", json.dumps(e2, ensure_ascii=False))
    print("E4 marker:", json.dumps(e4["A"]["A_marker_gold"], ensure_ascii=False))

if __name__ == "__main__":
    main()
