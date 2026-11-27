# -*- coding: utf-8 -*-
"""
绘图 v5.0 —— Fig 2–8。从 results/merged/ 读数据，写到 figures/。
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from config import FIG_DIR

# 中文字体（Windows: Microsoft YaHei; Linux/macOS 可能需要调整）
try:
    rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    rcParams["axes.unicode_minus"] = False
except Exception:
    pass


def fig2_mass_vs_gain(df, out=None):
    """Fig 2: mass → gain 散点 + LOESS + 回归线。主图。"""
    sub = df.dropna(subset=["gold_share", "gain"])
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(sub["gold_share"], sub["gain"], alpha=0.5, s=30, edgecolors="none")

    # LOESS smooth
    from statsmodels.nonparametric.smoothers_lowess import lowess
    smoothed = lowess(sub["gain"], sub["gold_share"], frac=0.3)
    ax.plot(smoothed[:, 0], smoothed[:, 1], "C1", lw=2, label="LOESS")

    # Linear fit
    z = np.polyfit(sub["gold_share"], sub["gain"], 1)
    x_line = np.linspace(sub["gold_share"].min(), sub["gold_share"].max(), 100)
    ax.plot(x_line, np.polyval(z, x_line), "C3--", lw=1.5, label="Linear")

    ax.set_xlabel("candidate_mass (gold_share)")
    ax.set_ylabel("constraint gain (HC − UC)")
    ax.legend()
    ax.set_title("Fig 2: Mass → Gain")
    fig.tight_layout()

    out = out or os.path.join(FIG_DIR, "fig2_mass_vs_gain.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Fig2 -> {out}")
    return out


def fig3_coefficient_forest(reg_results, out=None):
    """Fig 3: M1/M2/M3 mass 系数森林图。"""
    fig, ax = plt.subplots(figsize=(5, 3))
    models = []
    coefs, lows, highs = [], [], []
    for key in ["M1", "M2", "M3"]:
        m = reg_results.get(key, {})
        if m:
            models.append(m.get("model", key))
            coefs.append(m.get("mass_coef", 0))
            lows.append(m.get("mass_ci_low", 0))
            highs.append(m.get("mass_ci_high", 0))

    y = range(len(models))
    ax.errorbar(coefs, y, xerr=[[abs(c - l) for c, l in zip(coefs, lows)],
                                  [abs(h - c) for c, h in zip(coefs, highs)]],
                fmt="o", capsize=4)
    ax.axvline(0, color="gray", ls="--", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(models)
    ax.set_xlabel("mass coefficient")
    ax.set_title("Fig 3: Tautology Check (M1/M2/M3)")
    fig.tight_layout()

    out = out or os.path.join(FIG_DIR, "fig3_coefficient_forest.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Fig3 -> {out}")
    return out


def fig4_four_conditions(df, out=None):
    """Fig 4: 四条件准确率 + bootstrap CI。"""
    fig, ax = plt.subplots(figsize=(5, 4))
    conditions = ["UC", "RO", "SC", "HC"]
    means, lows, highs = [], [], []
    for cond in conditions:
        sub = df[df["condition"] == cond] if "condition" in df.columns else pd.DataFrame()
        if "correct" in df.columns:
            vals = sub["correct"].dropna().values
        else:
            vals = np.array([])
        if len(vals):
            from metrics import bootstrap_ci
            m, lo, hi = bootstrap_ci(vals)
        else:
            m, lo, hi = 0, 0, 0
        means.append(m)
        lows.append(m - lo)
        highs.append(hi - m)
    ax.bar(conditions, means, yerr=[lows, highs], capsize=5, color=["gray", "orange", "orange", "steelblue"])
    ax.set_ylabel("Accuracy")
    ax.set_title("Fig 4: Four Conditions (manipulation check)")
    fig.tight_layout()
    out = out or os.path.join(FIG_DIR, "fig4_four_conditions.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Fig4 -> {out}")
    return out


if __name__ == "__main__":
    # Dry-run with mock data
    print("=== plots dry-run ===")
    rng = np.random.RandomState(42)
    n = 160
    mock = pd.DataFrame({
        "gold_share": np.clip(rng.beta(2, 3, n), 0.01, 0.99),
        "gain": np.clip(-0.3 * np.clip(rng.beta(2, 3, n), 0.01, 0.99) + rng.normal(0, 0.15, n), -1, 1),
    })
    fig2_mass_vs_gain(mock)
    print("[OK] plots dry-run complete")
