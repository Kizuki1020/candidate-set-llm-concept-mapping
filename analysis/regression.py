# -*- coding: utf-8 -*-
"""
回归分析 v5.0 —— tautology 三嵌套 + 跨域交互。

输入: merged/analysis_df.csv（每术语一行: gold_share, gain, UC_acc, HC_acc, dataset）
输出: 回归表（M1/M2/M3 + H4交互项）
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from config import RESULTS_DIR
from metrics import pearson_r_ci


def run_tautology_regressions(df, dataset_label="A"):
    """
    M1: gain ~ mass
    M2: gain ~ mass + UC_acc       (控UC后mass系数是否存活)
    M3: HC_acc ~ mass              (换因变量，最干净)
    """
    sub = df[df["dataset"] == dataset_label].dropna(subset=["gold_share", "gain", "UC_acc", "HC_acc"])
    if len(sub) < 30:
        return {"error": f"n={len(sub)} insufficient"}

    r, r_lo, r_hi = pearson_r_ci(sub["gold_share"], sub["gain"])

    M1 = smf.ols("gain ~ gold_share", sub).fit()
    M2 = smf.ols("gain ~ gold_share + UC_acc", sub).fit()
    M3 = smf.ols("HC_acc ~ gold_share", sub).fit()

    def extract(model, name):
        return {
            "model": name,
            "n": int(model.nobs),
            "r2": round(model.rsquared, 4),
            "r2_adj": round(model.rsquared_adj, 4),
            "mass_coef": round(model.params.get("gold_share", float("nan")), 4),
            "mass_p": round(model.pvalues.get("gold_share", float("nan")), 6),
            "mass_ci_low": round(model.conf_int().loc["gold_share", 0], 4)
                if "gold_share" in model.conf_int().index else float("nan"),
            "mass_ci_high": round(model.conf_int().loc["gold_share", 1], 4)
                if "gold_share" in model.conf_int().index else float("nan"),
            "f_pvalue": round(model.f_pvalue, 6),
        }

    return {
        "pearson_r": round(r, 4),
        "pearson_ci": (round(r_lo, 4), round(r_hi, 4)),
        "M1": extract(M1, "M1: gain ~ mass"),
        "M2": extract(M2, "M2: gain ~ mass + UC_acc"),
        "M3": extract(M3, "M3: HC_acc ~ mass"),
        "m2_mass_survives": extract(M2, "")["mass_p"] < 0.05,
    }


def run_cross_domain_interaction(df):
    """H4: gain ~ mass * dataset 交互项检验。"""
    sub = df.dropna(subset=["gold_share", "gain", "dataset"])
    if sub["dataset"].nunique() < 2:
        return {"error": "need both datasets"}
    m = smf.ols("gain ~ gold_share * C(dataset)", sub).fit()
    idx = [i for i in m.params.index if "gold_share" in i and "dataset" in i]
    inter_p = m.pvalues.get(idx[0], float("nan")) if idx else float("nan")
    return {
        "n": int(m.nobs),
        "r2": round(m.rsquared, 4),
        "interaction_coef": round(m.params.get(idx[0], float("nan")), 4) if idx else None,
        "interaction_p": round(inter_p, 6),
        "no_evidence_for_slope_diff": inter_p > 0.05,
    }


def regression_table(results):
    """将 tautology 结果格式化为可打印/存盘的表格。"""
    rows = []
    for key in ["M1", "M2", "M3"]:
        m = results.get(key, {})
        rows.append({
            "Model": m.get("model", key),
            "n": m.get("n", ""),
            "R²": m.get("r2", ""),
            "mass_coef": m.get("mass_coef", ""),
            "mass_p": m.get("mass_p", ""),
            "mass_95CI": f"[{m.get('mass_ci_low','')}, {m.get('mass_ci_high','')}]",
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    # Dry-run with mock data
    print("=== regression dry-run ===")
    rng = np.random.RandomState(42)
    n = 160
    mock = pd.DataFrame({
        "dataset": ["A"] * n,
        "gold_share": np.clip(rng.beta(2, 3, n) + rng.normal(0, 0.05, n), 0.01, 0.99),
        "UC_acc": rng.binomial(1, 0.35, n).astype(float),
        "HC_acc": rng.binomial(1, 0.65, n).astype(float),
    })
    mock["gain"] = mock["HC_acc"] - mock["UC_acc"]

    results = run_tautology_regressions(mock)
    print(f"\nPearson r(mass, gain) = {results.get('pearson_r', 'N/A')}")
    print(f"M2 mass survives: {results.get('m2_mass_survives', 'N/A')}")
    print(f"\n{regression_table(results)}")
