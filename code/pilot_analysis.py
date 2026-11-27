#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pilot_analysis.py — 四门诊断（一次性输出，决定是否值得租GPU）

在 pilot_check.py 的 B/C 阶段之后运行。读 results/pilot/mass_{a,b}.npy，
或任何包含 gold_share / n_cand / dataset 列的 CSV。

用法:
  python pilot_analysis.py --mass-a results/pilot/mass_a.npy        # pilot_check 产出
  python pilot_analysis.py --csv results/exp2/mass_A.csv            # Exp2 产出
  python pilot_analysis.py --csv results/exp2/mass_A.csv --csv-b results/exp2/mass_B.csv
  python pilot_analysis.py --dry-run                                # 模拟验证
"""
import argparse, os, sys, math
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import THRESH, FIG_DIR, RESULTS_DIR
from metrics import mass_distribution_stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
except Exception:
    pass

OUT_DIR = os.path.join(RESULTS_DIR, "pilot")
VERDICT = {True: "PASS", False: "FAIL", None: "SKIP"}


# ========== Gate 1: Dynamic range ==========

def gate_dynamic_range(masses, label="A"):
    """
    A 域 mass 是否覆盖 low→high？Pilot C 的核心问题。
    masses 可以是 gold_share ∈ [0,1] 或 norm_mass ∈ (-∞, +∞)。
    mass_distribution_stats 会自动检测量纲。
    """
    stats = mass_distribution_stats(masses)
    is_norm = stats.get("is_norm_mass", False)
    spread_ok = stats["spread_p90_p10"] >= THRESH["dynrange_spread"]
    low_ok   = stats["frac_low"] >= THRESH["dynrange_low_frac"]
    high_ok  = stats["frac_high"] >= THRESH["dynrange_high_frac"]

    if spread_ok and low_ok and high_ok:
        verdict, note = True, "mass 有充分动态范围 → 核心图有斜率可拟合"
    elif spread_ok and (low_ok or high_ok):
        verdict, note = None, "动态范围偏窄但可用 → limitation 声明分布偏斜"
    else:
        verdict, note = False, "【阻断】mass 挤在窄带 → 主图不成立，需重抽 A 域"

    # Histogram — 自适应范围
    fig, ax = plt.subplots(figsize=(6, 3.5))
    if is_norm:
        lo, hi = min(-4, masses.min()), max(4, masses.max())
        xlabel = "norm_mass = log(gold_share × |C|)"
        low_line, high_line = 0.0, math.log(2)
        low_lbl, high_lbl = "baseline (0)", "2× baseline (ln2)"
    else:
        lo, hi = 0, 1
        xlabel = "gold_share"
        low_line, high_line = 0.2, 0.5
        low_lbl, high_lbl = "low<0.2", "high>0.5"
    ax.hist(masses, bins=20, range=(lo, hi), color="steelblue", edgecolor="white", alpha=0.85)
    ax.axvline(low_line, color="C1", ls="--", lw=1, label=low_lbl)
    ax.axvline(high_line, color="C2", ls="--", lw=1, label=high_lbl)
    ax.legend(fontsize=7)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    ax.set_title(f"Gate 1: Dynamic Range ({label}) — spread={stats['spread_p90_p10']:.3f}  [{VERDICT[verdict]}]")
    fig.tight_layout()
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"gate1_dynamic_range_{label}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)

    return {**stats, "verdict": verdict, "note": note, "figure": path,
            "spread_ok": spread_ok, "low_ok": low_ok, "high_ok": high_ok}


# ========== Gate 2: Common support ==========

def gate_common_support(mass_a, mass_b):
    """两域 mass 分布是否有重叠？决定 H4 能否用交互项。"""
    if mass_b is None or len(mass_b) == 0:
        return {"verdict": None, "note": "B 域数据缺失，H4 跨域主张本轮无法验证"}

    # 自适应范围：norm_mass 用 pooled 范围，gold_share 用 [0,1]
    all_vals = np.concatenate([mass_a, mass_b])
    lo, hi = float(all_vals.min()), float(all_vals.max())
    if lo >= 0 and hi <= 1.05:  # gold_share
        lo, hi = 0.0, 1.0
    else:
        margin = (hi - lo) * 0.1
        lo, hi = lo - margin, hi + margin

    iou = _hist_overlap_iou(mass_a, mass_b, lo=lo, hi=hi)
    a_lo, a_hi = np.percentile(mass_a, [10, 90])
    b_lo, b_hi = np.percentile(mass_b, [10, 90])
    overlap_width = max(0.0, min(a_hi, b_hi) - max(a_lo, b_lo))

    if iou >= THRESH["common_support_iou"] and overlap_width > 0.05:
        verdict, note = True, "两域 mass 有共同支撑 → H4 可用交互项"
    else:
        verdict, note = None, ("两域 mass 不重叠 → H4 降级为描述性域差异，"
                                "不写 cross-domain law")

    # Density plot
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, data, label, c in [(axes[0], mass_a, "A: Ancient Math", "steelblue"),
                                 (axes[1], mass_b, "B: Clinical", "C2")]:
        ax.hist(data, bins=20, range=(lo, hi), color=c, edgecolor="white", alpha=0.85, density=True)
        ax.set_xlabel("mass"); ax.set_ylabel("density"); ax.set_title(label)
    axes[0].set_title(f"A: μ={mass_a.mean():.3f} p10={np.percentile(mass_a,10):.3f} p90={np.percentile(mass_a,90):.3f}")
    axes[1].set_title(f"B: μ={mass_b.mean():.3f} p10={np.percentile(mass_b,10):.3f} p90={np.percentile(mass_b,90):.3f}")
    fig.suptitle(f"Gate 2: Common Support — IoU={iou:.3f}, overlap={overlap_width:.3f}  [{VERDICT[verdict]}]")
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "gate2_common_support.png")
    os.makedirs(OUT_DIR, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)

    return {"iou": round(iou, 4), "overlap_width": round(overlap_width, 4),
            "a_p10_p90": (float(a_lo), float(a_hi)),
            "b_p10_p90": (float(b_lo), float(b_hi)),
            "verdict": verdict, "note": note, "figure": path}


def _hist_overlap_iou(a, b, bins=10, lo=0.0, hi=1.0):
    edges = np.linspace(lo, hi, bins + 1)
    ha, _ = np.histogram(a, bins=edges, density=True)
    hb, _ = np.histogram(b, bins=edges, density=True)
    inter = np.minimum(ha, hb).sum()
    union = np.maximum(ha, hb).sum()
    return inter / union if union > 0 else 0.0


# ========== Gate 3: Gold share distribution ==========

def gate_gold_share(df):
    """mass 分布：模型是否只认 gold，还是有竞争？支持 gold_share 与 norm_mass 两种量纲。"""
    mass_col = "norm_mass" if "norm_mass" in df.columns else "gold_share"
    gs = df[mass_col].dropna().values
    is_norm = mass_col == "norm_mass"
    stats = {
        "n": len(gs), "mean": float(gs.mean()), "std": float(gs.std()),
        "p10": float(np.percentile(gs, 10)), "p50": float(np.median(gs)),
        "p90": float(np.percentile(gs, 90)),
    }
    if is_norm:
        stats["frac_above_baseline"] = float((gs > 0).mean())
        stats["frac_below_baseline"] = float((gs < 0).mean())
        lo, hi = min(-4, gs.min()), max(4, gs.max())
        xlabel = "norm_mass = log(gold_share × |C|)"
        if stats["frac_above_baseline"] < 0.2:
            verdict, note = True, "几乎所有样本 mass 低于均匀基线 → C(t) 内竞争激烈"
        elif stats["frac_below_baseline"] < 0.2:
            verdict, note = True, "几乎所有样本 mass 高于均匀基线 → 模型强烈偏好 gold"
        else:
            verdict, note = True, "mass 分布有正有负 → 区分度好"
    else:
        stats["frac_above_0_5"] = float((gs > 0.5).mean())
        stats["frac_below_0_2"] = float((gs < 0.2).mean())
        lo, hi = 0, 1
        xlabel = "gold_share"
        if stats["frac_above_0_5"] > 0.8:
            verdict, note = True, "gold_share 偏高 → 约束增益空间小"
        elif stats["frac_below_0_2"] > 0.5:
            verdict, note = True, "gold_share 偏低 → 约束有大量增益空间"
        else:
            verdict, note = True, "gold_share 分布均衡 → 有区分度"

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(gs, bins=20, range=(lo, hi), color="steelblue", edgecolor="white", alpha=0.85)
    if is_norm:
        axes[0].axvline(0, color="gray", ls="--", label="baseline")
        axes[0].legend(fontsize=7)
    else:
        axes[0].axvline(0.5, color="gray", ls="--")
    axes[0].set_xlabel(xlabel); axes[0].set_ylabel("count")
    axes[0].set_title(f"mass dist (μ={stats['mean']:.3f})")

    if "n_cand" in df.columns:
        axes[1].scatter(df["n_cand"], gs, alpha=0.4, s=20, edgecolors="none")
        axes[1].set_xlabel("|C(t)|"); axes[1].set_ylabel(xlabel)
        axes[1].set_title(f"mass vs |C(t)|")
    fig.suptitle(f"Gate 3: Mass Distribution [{VERDICT[verdict]}]")
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "gate3_mass_dist.png")
    os.makedirs(OUT_DIR, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)

    return {**stats, "verdict": verdict, "note": note, "figure": path}


# ========== Gate 4: |C| confounding ==========

def gate_candidate_size_bias(df):
    """|C| 是否系统地影响 mass？norm_mass 已用 |C| 归一化，相关应显著减弱。"""
    mass_col = "norm_mass" if "norm_mass" in df.columns else "gold_share"
    sub = df.dropna(subset=["n_cand", mass_col])
    if len(sub) < 10:
        return {"verdict": None, "note": "样本不足"}

    from scipy.stats import pearsonr, spearmanr
    pr, pp = pearsonr(sub["n_cand"], sub[mass_col])
    sr, sp = spearmanr(sub["n_cand"], sub[mass_col])
    abs_max = max(abs(pr), abs(sr))

    if abs_max > 0.5:
        verdict, note = False, (f"|C| 与 {mass_col} 强相关 (r={abs_max:.3f}) → "
                                 "mass 主要在测候选规模。需固定 |C| 或在回归中控制。")
    elif abs_max > 0.3:
        verdict, note = None, (f"|C| 与 {mass_col} 中等相关 (r={abs_max:.3f}) → "
                                "建议在回归中把 log|C| 作协变量")
    else:
        verdict, note = True, f"|C| 与 {mass_col} 弱相关 (r={abs_max:.3f}) → 无显著偏置"

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(sub["n_cand"], sub[mass_col], alpha=0.4, s=20, edgecolors="none")
    z = np.polyfit(sub["n_cand"], sub[mass_col], 1)
    xs = np.linspace(sub["n_cand"].min(), sub["n_cand"].max(), 50)
    ax.plot(xs, np.polyval(z, xs), "C1--", lw=1.5)
    ax.set_xlabel("|C(t)|"); ax.set_ylabel(mass_col)
    ax.set_title(f"Gate 4: |C| vs {mass_col} — r={pr:.3f} (Pearson), ρ={sr:.3f} (Spearman)  [{VERDICT[verdict]}]")
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "gate4_candidate_size_bias.png")
    os.makedirs(OUT_DIR, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)

    return {"pearson_r": round(pr, 4), "pearson_p": round(pp, 6),
            "spearman_r": round(sr, 4), "spearman_p": round(sp, 6),
            "verdict": verdict, "note": note, "figure": path}


# ========== Report ==========

def print_report(gates):
    print("\n" + "=" * 66)
    print("  PILOT GATE REPORT")
    print("=" * 66)
    for name, g in gates.items():
        v = g.get("verdict")
        flag = {True: "[PASS]", False: "[FAIL]", None: "[SKIP]"}.get(v, "[??]")
        print(f"\n  Gate {name}: {flag}")
        for k, val in g.items():
            if k in ("verdict", "note", "figure"):
                continue
            if isinstance(val, float):
                print(f"    {k}: {val:.4f}")
        print(f"    → {g.get('note', '')}")
        if g.get("figure"):
            print(f"    fig: {g['figure']}")

    blocked = any(g.get("verdict") is False for g in gates.values())
    degraded = [k for k, g in gates.items() if g.get("verdict") is None]
    fails   = [k for k, g in gates.items() if g.get("verdict") is False]

    if blocked:
        overall = "FAIL"
        recommendation = f"先修 {fails}，再重跑本脚本。不要租 GPU。"
    elif degraded:
        overall = "GO_DEGRADED"
        recommendation = f"无阻断门。{degraded} 降级（见上），可租 GPU，注意调整方案。"
    else:
        overall = "GO"
        recommendation = "四门全绿。租 GPU，跑 Exp2。"

    summary = {"overall": overall, "recommendation": recommendation,
               "blocked_by": fails, "degraded": degraded}

    print("\n" + "=" * 66)
    print(f"  [{overall}] {recommendation}")
    print("=" * 66 + "\n")
    return summary


# ========== Main ==========

def main():
    ap = argparse.ArgumentParser(description="Pilot 四门诊断")
    ap.add_argument("--dry-run", action="store_true", help="模拟 mass 数据，验证全流程")
    ap.add_argument("--csv", default=None, help="mass CSV（含 gold_share/norm_mass, n_cand, dataset 列）")
    ap.add_argument("--csv-b", default=None, help="B 域 mass CSV")
    ap.add_argument("--mass-a", default=None, help="gold_share_a.npy（pilot_check 产出）")
    ap.add_argument("--mass-b", default=None, help="gold_share_b.npy（pilot_check 产出）")
    ap.add_argument("--norm-mass-a", default=None, help="norm_mass_a.npy（推荐；pilot_check v5.1 产出）")
    ap.add_argument("--norm-mass-b", default=None, help="norm_mass_b.npy")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    # Load data — prefer norm_mass when available
    mass_col = "gold_share"  # default

    if args.dry_run:
        print("[dry-run] 生成模拟 norm_mass 数据")
        rng = np.random.RandomState(42)
        n_a, n_b = 160, 20
        mass_a = np.clip(rng.beta(2, 3, n_a) + rng.normal(0, 0.03, n_a), 0.01, 0.99)
        n_cand_a = rng.randint(50, 101, n_a)
        norm_mass_a = np.log(np.maximum(mass_a, 1e-12) * n_cand_a)
        mass_b = np.clip(rng.beta(4, 2, n_b) + rng.normal(0, 0.03, n_b), 0.01, 0.99)
        n_cand_b = rng.randint(20, 60, n_b)
        norm_mass_b = np.log(np.maximum(mass_b, 1e-12) * n_cand_b)
        df_a = pd.DataFrame({"gold_share": mass_a, "norm_mass": norm_mass_a, "n_cand": n_cand_a, "dataset": "A"})
        df_b = pd.DataFrame({"gold_share": mass_b, "norm_mass": norm_mass_b, "n_cand": n_cand_b, "dataset": "B"})
        df = pd.concat([df_a, df_b], ignore_index=True)
        mass_col = "norm_mass"
    elif args.norm_mass_a:
        mass_col = "norm_mass"
        norm_a = np.load(args.norm_mass_a)
        norm_b = np.load(args.norm_mass_b) if args.norm_mass_b and os.path.exists(args.norm_mass_b) else None
        df_a = pd.DataFrame({"norm_mass": norm_a, "n_cand": 50, "dataset": "A"})
        if norm_b is not None and len(norm_b) > 0:
            df_b = pd.DataFrame({"norm_mass": norm_b, "n_cand": 50, "dataset": "B"})
            df = pd.concat([df_a, df_b], ignore_index=True)
        else:
            df = df_a
    elif args.csv:
        df_a = pd.read_csv(args.csv)
        df_b = pd.read_csv(args.csv_b) if args.csv_b else None
        if "norm_mass" in df_a.columns:
            mass_col = "norm_mass"
        if "n_cand" not in df_a.columns:
            df_a["n_cand"] = 50
        df = df_a if df_b is None else pd.concat([df_a, df_b], ignore_index=True)
    elif args.mass_a:
        # Legacy gold_share .npy
        mass_a = np.load(args.mass_a)
        mass_b = np.load(args.mass_b) if args.mass_b and os.path.exists(args.mass_b) else None
        df_a = pd.DataFrame({"gold_share": mass_a, "n_cand": 50, "dataset": "A"})
        if mass_b is not None and len(mass_b) > 0:
            df_b = pd.DataFrame({"gold_share": mass_b, "n_cand": 50, "dataset": "B"})
            df = pd.concat([df_a, df_b], ignore_index=True)
        else:
            df = df_a
    else:
        print("请指定 --dry-run, --csv, --norm-mass-a, 或 --mass-a。")
        sys.exit(1)

    # Run gates
    gates = {}

    print(f"\nGate 1: Dynamic Range (A) … [column: {mass_col}]")
    gates["1_dynamic_range"] = gate_dynamic_range(
        df[df["dataset"] == "A"][mass_col].dropna().values)

    print("Gate 2: Common Support (A/B) …")
    mass_b_vals = df[df["dataset"] == "B"][mass_col].dropna().values if "B" in df["dataset"].values else None
    gates["2_common_support"] = gate_common_support(
        df[df["dataset"] == "A"][mass_col].dropna().values, mass_b_vals)

    print("Gate 3: Mass Distribution …")
    gates["3_mass_dist"] = gate_gold_share(df[df["dataset"] == "A"])

    print("Gate 4: |C| Bias …")
    gates["4_cand_size_bias"] = gate_candidate_size_bias(df[df["dataset"] == "A"])

    summary = print_report(gates)

    # Save gate data
    import json
    report_path = os.path.join(OUT_DIR, "gate_report.json")
    def _jsonable(v):
        if isinstance(v, (np.bool_,)): return bool(v)
        if isinstance(v, (np.integer,)): return int(v)
        if isinstance(v, (np.floating,)): return float(v)
        if isinstance(v, tuple): return list(v)
        return v
    payload = {k: {kk: _jsonable(vv) for kk, vv in v.items() if kk != "figure"}
               for k, v in gates.items()}
    payload["overall"] = summary["overall"]
    payload["recommendation"] = summary["recommendation"]
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"Report saved: {report_path}")


if __name__ == "__main__":
    main()
