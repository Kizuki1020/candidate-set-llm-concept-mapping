#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Supplementary_Data_Code_ScientificReports_v2 完整性/统计验证脚本
- 只依赖包内文件（相对路径），可在任意目录解压后独立运行。
- 验证范围：数据集完整性、exp1 逐条预测、基线、exp2 概率质量、
  正文关键统计（Table 2、RO-HC、检索器 McNemar + Holm、顺序不变性、
  拒绝标记计数、注入敏感性）。
- 注意：本脚本验证的是"冻结预测结果的独立统计复现"，
  不是从模型推理开始的完整复现（见 README.txt 第 5 节）。
运行: python3 verification/verify_package.py
输出: verification/verify_report.json + 终端 PASS/FAIL 汇总；失败时退出码 1。
"""
import json, os, sys
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, binom

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = []
FAILED = []

def check(name, cond, detail=""):
    RESULTS.append({"check": name, "pass": bool(cond), "detail": str(detail)})
    if not cond:
        FAILED.append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail and not cond else ""))

def mcnemar_maj(a, b):
    bd = int(((a == 0) & (b == 1)).sum()); c = int(((a == 1) & (b == 0)).sum())
    if bd + c == 0:
        return bd, c, 1.0
    p = 2 * min(binom.cdf(min(bd, c), bd + c, .5), 1 - binom.cdf(max(bd, c) - 1, bd + c, .5))
    return bd, c, float(p)

def wilson(p, n, z=1.96):
    d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d
    h = z/d * np.sqrt(p*(1-p)/n + z*z/(4*n*n))
    return max(0, c-h), min(1, c+h)

def term_majority(df):
    return df.groupby(["id", "condition"])["correct"].mean().unstack()

print("=" * 62)
print("Supplementary_Data_Code_ScientificReports_v2 — verify_package")
print("=" * 62)

# ---------- 1. 数据集 ----------
print("\n[1] Datasets")
da = json.load(open(os.path.join(PKG, "data/dataset_a_with_candidates.json"), encoding="utf-8"))
db = json.load(open(os.path.join(PKG, "data/dataset_b.json"), encoding="utf-8"))
check("A test_cases == 150", len(da["test_cases"]) == 150)
check("B test_cases == 150", len(db["test_cases"]) == 150)
inj = [t["protege_id"] for t in da["test_cases"] if t.get("gold_injected")]
check("A gold_injected == 4 (N0694/N0695/N0646/N0692)",
      set(inj) == {"N0694", "N0695", "N0646", "N0692"}, f"got {sorted(inj)}")
gic = sum(1 for t in da["test_cases"] if t.get("gold_in_candidates"))
check("A gold_in_candidates == 150/150", gic == 150, f"got {gic}")
b_sizes = [len(t.get("candidates", []) or []) for t in db["test_cases"]]
check("B candidate sizes within [50, 84]", min(b_sizes) >= 50 and max(b_sizes) <= 84,
      f"min={min(b_sizes)}, max={max(b_sizes)}")
mt = {t.get("mapping_type") for t in db["test_cases"]}
check("B mapping_type == {equivalence, generalization}",
      mt == {"equivalence", "generalization"}, f"got {mt}")

# ---------- 2. exp1 逐条预测 ----------
print("\n[2] Exp1 per-record predictions")
ok_ids = True
for dom in ["A", "B"]:
    df = pd.read_csv(os.path.join(PKG, "results/exp1", f"exp1_{dom}.csv"))
    if len(df) != 1800 or df["id"].nunique() != 150:
        ok_ids = False
        check(f"exp1_{dom}: 1800 rows / 150 ids", False, f"rows={len(df)}, ids={df['id'].nunique()}")
        continue
    g = df.groupby("id").size()
    if not (g == 12).all():
        ok_ids = False
        check(f"exp1_{dom}: every id has 12 rows (4 conds x 3 reps)", False, f"violations={int((g != 12).sum())}")
        continue
    check(f"exp1_{dom}: 1800 rows, 150 ids x 12", True)
    conds = set(df["condition"].unique())
    check(f"exp1_{dom}: conditions == UC/HC/RO/SC", conds == {"UC", "HC", "RO", "SC"}, f"got {conds}")
    reps = set(df["rep"].unique())
    check(f"exp1_{dom}: reps == 0/1/2", reps == {0, 1, 2}, f"got {reps}")
    check(f"exp1_{dom}: no NaN in correct/pred/gold",
          not (df[["correct", "pred", "gold"]].isna().any().any()))
check("exp1 structure (both domains)", ok_ids)

# ---------- 3. 基线 ----------
print("\n[3] Entity-linking baselines")
el = pd.read_csv(os.path.join(PKG, "results/exp1/entity_linking_baselines.csv"))
check("baselines: 900 rows = 150 x 2 domains x 3 methods", len(el) == 900, f"rows={len(el)}")
check("baselines methods complete",
      set(el["method"].unique()) == {"bge-m3", "bge-reranker-base", "bge-reranker-v2-m3"})

# ---------- 4. exp2 概率质量 ----------
print("\n[4] Exp2 candidate mass")
for dom in ["A", "B"]:
    m = pd.read_csv(os.path.join(PKG, "results/exp2", f"mass_{dom}.csv"))
    check(f"mass_{dom}: 150 rows", len(m) == 150, f"rows={len(m)}")
ea = pd.read_csv(os.path.join(PKG, "results/exp1/exp1_A.csv"))
ma = pd.read_csv(os.path.join(PKG, "results/exp2/mass_A.csv"))
check("mass_A ids == exp1_A ids", set(ma["id"]) == set(ea["id"]))

# ---------- 5. 正文关键统计 ----------
print("\n[5] Key statistics (Table 2 / RO-HC / retrievers)")
exp1_a = pd.read_csv(os.path.join(PKG, "results/exp1/exp1_A.csv"))
exp1_b = pd.read_csv(os.path.join(PKG, "results/exp1/exp1_B.csv"))
ta, tb = term_majority(exp1_a), term_majority(exp1_b)
expect = {"A": {"UC": 0.033, "HC": 0.273, "RO": 0.567, "SC": 0.260},
          "B": {"UC": 0.040, "HC": 0.467, "RO": 0.767, "SC": 0.627}}
for dom, t, exp in [("A", ta, expect["A"]), ("B", tb, expect["B"])]:
    for cnd, v in exp.items():
        got = round(float(t[cnd].mean()), 3)
        check(f"Table 2 {dom}.{cnd} == {v}", abs(got - v) < 0.001, f"got {got}")
    diff = ((t["RO"] > .5).astype(int) - (t["HC"] > .5).astype(int))
    w, p = wilcoxon(t["RO"], t["HC"])
    exp_gain = {"A": 0.293, "B": 0.300}[dom]
    check(f"{dom}: RO-HC mean == {exp_gain} (within .002)", abs(diff.mean() - exp_gain) < 0.002,
          f"got {diff.mean():.3f}")
    check(f"{dom}: RO-HC Wilcoxon p < 0.001", p < 0.001, f"p={p:.2e}")

print("   -- retriever vs HC (McNemar exact, Holm-corrected over 6 tests) --")
holm = []
row = {}
for dom in ["A", "B"]:
    t = ta if dom == "A" else tb
    hc_bin = (t["HC"] > .5).astype(int)
    for m in ["bge-m3", "bge-reranker-base", "bge-reranker-v2-m3"]:
        s = el[(el["domain"] == dom) & (el["method"] == m)].set_index("id")
        top1 = s["top1_correct"].astype(int)
        common = hc_bin.index.intersection(top1.index)
        b, c, p = mcnemar_maj(hc_bin.loc[common], top1.loc[common])
        row[(dom, m)] = p
        holm.append((dom, m, p))
def holm_monotone(pvals):
    order = np.argsort(pvals)
    adj = np.empty(len(pvals)); prev = 0.0
    for rank, idx in enumerate(order):
        a = max(prev, min(1.0, pvals[idx] * (len(pvals) - rank)))
        adj[idx] = a; prev = a
    return adj
adj_all = holm_monotone(np.array([h[2] for h in holm]))
expect_holm6 = {("A", "bge-m3"): 0.0675, ("A", "bge-reranker-base"): 0.1007,
                ("A", "bge-reranker-v2-m3"): 0.0272, ("B", "bge-m3"): 0.1007,
                ("B", "bge-reranker-base"): 0.1770, ("B", "bge-reranker-v2-m3"): 0.0852}
for i, (dom, m, p) in enumerate(holm):
    p_adj = adj_all[i]
    exp = expect_holm6.get((dom, m), None)
    check(f"{dom} {m} vs HC: McNemar p={p:.4f}, Holm(6-family)={p_adj:.4f}",
          exp is None or abs(p_adj - exp) < 0.001, f"got {p_adj:.4f}, expected ~{exp}")

# ---------- 6. 顺序不变性 ----------
print("\n[6] Order invariance (3 deterministic reps differ only in candidate order)")
for dom in ["A", "B"]:
    df = pd.read_csv(os.path.join(PKG, "results/exp1", f"exp1_{dom}.csv"))
    for cond in ["HC", "RO", "SC", "UC"]:
        g = df[df["condition"] == cond].groupby("id")["pred"].nunique()
        frac = float((g == 1).mean())
        expect_min = {"A": {"HC": 149, "RO": 150, "SC": 150, "UC": 150},
                      "B": {"HC": 150, "RO": 149, "SC": 150, "UC": 146}}[dom][cond]
        check(f"{dom}.{cond}: identical-rep count == {expect_min}/150", int((g == 1).sum()) == expect_min,
              f"got {int((g == 1).sum())}")

# ---------- 7. 拒绝标记记录 ----------
print("\n[7] No-equivalent marker records (n=7)")
marker_ids = {t["protege_id"] for t in da["test_cases"]
              if t.get("modern_definition") and any(s in t["modern_definition"] for s in ["无现代", "无对应"])}
check("marker records == 7", len(marker_ids) == 7, f"got {len(marker_ids)}")
sub = exp1_a[exp1_a["id"].isin(marker_ids)]
counts = {}
for cnd, g in sub.groupby("condition"):
    counts[cnd] = int((g["pred"] == g["gold"]).sum())
expect_counts = {"UC": 0, "HC": 3, "RO": 9, "SC": 6}
for cnd, v in expect_counts.items():
    check(f"correct rejection {cnd} == {v}/21", counts.get(cnd) == v, f"got {counts.get(cnd)}")

# ---------- 8. 注入敏感性 ----------
print("\n[8] Gold-injection sensitivity (exclude 4 injected records)")
sub_a = exp1_a[~exp1_a["id"].isin({"N0694", "N0695", "N0646", "N0692"})]
t_sub = term_majority(sub_a)
diff_sub = ((t_sub["RO"] > .5).astype(int) - (t_sub["HC"] > .5).astype(int)).mean()
w, p = wilcoxon(t_sub["RO"], t_sub["HC"])
check("A RO-HC excl injected == 0.301 (within .003)", abs(diff_sub - 0.301) < 0.003, f"got {diff_sub:.3f}")
check("A RO-HC excl injected p < 1e-5", p < 1e-5, f"p={p:.2e}")

# ---------- 汇总 ----------
report = {
    "total": len(RESULTS), "passed": len(RESULTS) - len(FAILED), "failed": len(FAILED),
    "failed_checks": FAILED, "results": RESULTS,
    "env": {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": __import__("scipy").__version__,
    },
}
with open(os.path.join(PKG, "verification/verify_report.json"), "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)
print("\n" + "=" * 62)
print(f"RESULT: {len(RESULTS)-len(FAILED)}/{len(RESULTS)} checks passed")
print("OVERALL:", "PASS" if not FAILED else "FAIL")
print("=" * 62)
sys.exit(0 if not FAILED else 1)
