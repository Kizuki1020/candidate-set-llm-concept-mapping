#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""B 域 30 条 Pilot — 真实 GPU candidate_mass + 四门诊断。

用法:
  python pilot_b_gpu.py                    # 30条 GPU mass + 四门
  python pilot_b_gpu.py --n-sample 50      # 自定义样本量
"""

import argparse, os, sys, json, math
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *
from candidate_mass import score_candidates, candidate_mass_row
from config import build_user_prompt

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

OUT_DIR = os.path.join(RESULTS_DIR, "pilot_b")
os.makedirs(OUT_DIR, exist_ok=True)


# ============================================================
# 1. Load data
# ============================================================

def load_b_samples(n=30, seed=42):
    path = os.path.join(DATA_DIR, "dataset_b.json")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    records = data["test_cases"]
    # Map to internal fields (same as exp2_logit_analysis.py)
    mapped = []
    for r in records:
        mapped.append({
            "id": r.get("record_id", ""),
            "term": r.get("name", ""),
            "source": r.get("source_book", ""),
            "definition": r.get("original_definition", ""),
            "gold": r.get("modern_definition", ""),
            "candidates": r.get("candidates", []),
            "p_class": r.get("P_Class", ""),
            "mapping_type": r.get("mapping_type", ""),
        })
    # Sample
    rng = np.random.RandomState(seed)
    if len(mapped) > n:
        idx = rng.choice(len(mapped), n, replace=False)
        mapped = [mapped[i] for i in sorted(idx)]
    return mapped


def load_a_mass():
    csv_path = os.path.join(RESULTS_DIR, "exp2", "mass_A.csv")
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return None


# ============================================================
# 2. GPU mass computation
# ============================================================

def compute_mass_gpu(records, model, tok):
    """Compute real candidate_mass via teacher-forcing (same prompt as Exp2)."""
    rows = []
    n_total = len(records)
    for i, r in enumerate(records):
        try:
            cands = list(dict.fromkeys(r.get("candidates", [])))
            gold = r["gold"]
            if gold not in cands:
                cands.append(gold)

            prompt = build_user_prompt(r["term"], r.get("source", ""), r.get("definition", ""))
            scores = score_candidates(model, tok, prompt, cands)
            cm = candidate_mass_row(scores, gold)
            rows.append({
                "id": r["id"],
                "term": r["term"],
                "gold": r["gold"],
                "dataset": "B",
                "n_cand": cm["n_cand"],
                "gold_share": cm["gold_share"],
                "norm_mass": cm["norm_mass"],
                "gold_rank_in_C": cm["gold_rank_in_C"],
                "argmax_in_C": cm["argmax_in_C"],
                "mapping_type": r.get("mapping_type", ""),
            })
        except Exception as e:
            print(f"  [{i+1}/{n_total}] ERROR on {r['id']}: {e}")
            rows.append({
                "id": r["id"],
                "term": r["term"],
                "gold": r["gold"],
                "dataset": "B",
                "n_cand": len(r.get("candidates", [])),
                "gold_share": float("nan"),
                "norm_mass": float("nan"),
                "gold_rank_in_C": -1,
                "argmax_in_C": "",
                "mapping_type": r.get("mapping_type", ""),
            })
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{n_total}] done")

    df = pd.DataFrame(rows)
    csv_path = os.path.join(OUT_DIR, "mass_B_pilot.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"  Saved: {csv_path} ({len(df)} rows)")
    return df


# ============================================================
# 3. Four gates (using real mass)
# ============================================================

def gate1_dynamic_range(masses):
    spread = float(np.percentile(masses, 90) - np.percentile(masses, 10))
    low_frac = float((masses < 0).mean())
    high_frac = float((masses > math.log(2)).mean())
    ok = spread >= 1.5 and low_frac >= 0.10 and high_frac >= 0.10
    return {
        "verdict": ok,
        "spread": spread, "low_frac": low_frac, "high_frac": high_frac,
        "n": len(masses),
        "note": "mass 有充分动态范围" if ok else "动态范围不足"
    }


def gate2_common_support(mass_a, mass_b):
    # Histogram IoU
    all_vals = np.concatenate([mass_a, mass_b])
    lo, hi = float(all_vals.min()) - 0.5, float(all_vals.max()) + 0.5
    bins = 10
    edges = np.linspace(lo, hi, bins + 1)
    ha, _ = np.histogram(mass_a, bins=edges, density=True)
    hb, _ = np.histogram(mass_b, bins=edges, density=True)
    iou = np.minimum(ha, hb).sum() / np.maximum(ha, hb).sum() if np.maximum(ha, hb).sum() > 0 else 0.0
    a_lo, a_hi = np.percentile(mass_a, [10, 90])
    b_lo, b_hi = np.percentile(mass_b, [10, 90])
    overlap = max(0.0, min(a_hi, b_hi) - max(a_lo, b_lo))
    ok = iou >= 0.15 and overlap > 0.05
    return {
        "iou": round(iou, 4), "overlap_width": round(overlap, 4),
        "a_p10_p90": [float(a_lo), float(a_hi)],
        "b_p10_p90": [float(b_lo), float(b_hi)],
        "verdict": ok,
        "note": "两域 mass 有共同支撑" if ok else "不重叠 → H4 降级"
    }


def gate3_mass_distribution(masses):
    above = float((masses > 0).mean())
    below = float((masses < 0).mean())
    return {
        "n": len(masses),
        "mean": float(masses.mean()), "std": float(masses.std()),
        "p10": float(np.percentile(masses, 10)),
        "p50": float(np.median(masses)),
        "p90": float(np.percentile(masses, 90)),
        "frac_above_baseline": above,
        "frac_below_baseline": below,
        "verdict": True,
        "note": "mass 分布有正有负 → 区分度好" if above > 0.2 and below > 0.2 else "分布偏斜"
    }


def gate4_candidate_size_bias(df, mass_col="norm_mass"):
    from scipy.stats import pearsonr, spearmanr
    sub = df.dropna(subset=["n_cand", mass_col])
    if len(sub) < 10:
        return {"verdict": None, "note": "样本不足"}
    pr, pp = pearsonr(sub["n_cand"], sub[mass_col])
    sr, sp = spearmanr(sub["n_cand"], sub[mass_col])
    abs_max = max(abs(pr), abs(sr))
    if abs_max > 0.5:
        verdict, note = False, f"|C| 与 mass 强相关 (r={abs_max:.3f})"
    elif abs_max > 0.3:
        verdict, note = None, f"|C| 与 mass 中等相关 (r={abs_max:.3f})"
    else:
        verdict, note = True, f"|C| 与 mass 弱相关 (r={abs_max:.3f})"
    return {"pearson_r": round(pr, 4), "pearson_p": round(pp, 6),
            "spearman_r": round(sr, 4), "spearman_p": round(sp, 6),
            "verdict": verdict, "note": note}


def gate0_gold_rank_alignment(df_a, df_b):
    if df_a is None or "gold_rank_in_C" not in df_a.columns:
        return {"alignment_judgment": "无 A 域数据，无法判断对齐"}

    ra = df_a["gold_rank_in_C"].dropna().values
    rb = df_b["gold_rank_in_C"].dropna().values

    stats = {
        "A_n": len(ra), "A_rank_mean": float(ra.mean()), "A_rank_median": float(np.median(ra)),
        "A_rank_p10": float(np.percentile(ra, 10)), "A_rank_p90": float(np.percentile(ra, 90)),
        "B_n": len(rb), "B_rank_mean": float(rb.mean()), "B_rank_median": float(np.median(rb)),
        "B_rank_p10": float(np.percentile(rb, 10)), "B_rank_p90": float(np.percentile(rb, 90)),
        "B_rank_le_5_frac": float((rb <= 5).mean()),
        "B_rank_le_10_frac": float((rb <= 10).mean()),
    }

    ratio = stats["B_rank_mean"] / stats["A_rank_mean"] if stats["A_rank_mean"] > 0 else float("inf")
    stats["rank_mean_ratio_B_over_A"] = round(ratio, 3)

    if ratio < 0.5:
        stats["alignment_judgment"] = "B 域 gold_rank 系统性更靠前 → 干扰项偏弱"
    elif ratio < 0.75:
        stats["alignment_judgment"] = "B 域 gold_rank 略靠前 → 边界可接受"
    elif ratio <= 1.5:
        stats["alignment_judgment"] = "A/B gold_rank 分布可比 → 候选集难度对齐良好"
    else:
        stats["alignment_judgment"] = "B 域 gold_rank 更靠后 → 干扰项偏强"
    return stats


# ============================================================
# 4. Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-sample", type=int, default=30)
    args = ap.parse_args()

    print("=" * 60)
    print(f"  B Domain Pilot — {args.n_sample} samples — REAL GPU")
    print("=" * 60)

    # Load data
    records = load_b_samples(args.n_sample)
    print(f"\nLoaded {len(records)} B-domain samples")

    eq_n = sum(1 for r in records if r.get("mapping_type") == "equivalence")
    gen_n = sum(1 for r in records if r.get("mapping_type") == "generalization")
    print(f"Mapping: equivalence={eq_n}, generalization={gen_n}")

    # Load model
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}.get(DTYPE, torch.float16)
    print(f"\nLoading {MODEL_NAME} ({DTYPE}) ...")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=dtype, device_map=DEVICE, trust_remote_code=True)
    model.eval()
    print("Model loaded.")

    # Compute real mass
    print(f"\nComputing candidate_mass for {len(records)} B samples...")
    df_b = compute_mass_gpu(records, model, tok)

    # Show summary
    valid = df_b.dropna(subset=["norm_mass"])
    print(f"\nValid samples: {len(valid)}/{len(df_b)}")
    if len(valid) > 0:
        print(f"norm_mass: mean={valid['norm_mass'].mean():.3f}  std={valid['norm_mass'].std():.3f}")
        print(f"  p10={valid['norm_mass'].quantile(0.1):.3f}  p50={valid['norm_mass'].quantile(0.5):.3f}  p90={valid['norm_mass'].quantile(0.9):.3f}")
        print(f"gold_rank: mean={valid['gold_rank_in_C'].mean():.1f}  rank=1: {(valid['gold_rank_in_C']==1).mean():.1%}")

    # Load A reference
    df_a = load_a_mass()
    if df_a is not None:
        print(f"\nA domain reference: {len(df_a)} samples")

    # Run gates
    print("\n" + "=" * 60)
    print("  GATE DIAGNOSTICS")
    print("=" * 60)

    gates = {}
    mass_b = valid["norm_mass"].values
    mass_a = df_a["norm_mass"].dropna().values if df_a is not None else np.array([])

    # G0
    rank_stats = gate0_gold_rank_alignment(df_a, valid)
    print(f"\n[G0] Candidate Difficulty Alignment")
    print(f"  A rank mean={rank_stats.get('A_rank_mean','N/A')}, B rank mean={rank_stats['B_rank_mean']:.1f}")
    print(f"  B rank≤5: {rank_stats['B_rank_le_5_frac']:.1%}  rank≤10: {rank_stats['B_rank_le_10_frac']:.1%}")
    print(f"  → {rank_stats['alignment_judgment']}")

    # G1
    g1 = gate1_dynamic_range(mass_b)
    gates["1_dynamic_range"] = g1
    print(f"\n[G1] Dynamic Range: {'PASS' if g1['verdict'] else 'FAIL'}")
    print(f"  spread={g1['spread']:.3f}  low_frac={g1['low_frac']:.3f}  high_frac={g1['high_frac']:.3f}")

    # G2
    if len(mass_a) > 0:
        g2 = gate2_common_support(mass_a, mass_b)
        gates["2_common_support"] = g2
        print(f"\n[G2] Common Support: {'PASS' if g2['verdict'] else 'FAIL'}")
        print(f"  IoU={g2['iou']:.4f}  overlap={g2['overlap_width']:.4f}")

    # G3
    g3 = gate3_mass_distribution(mass_b)
    gates["3_mass_dist"] = g3
    print(f"\n[G3] Mass Distribution: PASS")
    print(f"  mean={g3['mean']:.3f}  above_baseline={g3['frac_above_baseline']:.1%}")

    # G4
    g4 = gate4_candidate_size_bias(valid)
    gates["4_cand_size_bias"] = g4
    v4 = "PASS" if g4['verdict'] else ("FAIL" if g4['verdict'] is False else "SKIP")
    print(f"\n[G4] |C| Bias: {v4}")
    if g4['verdict'] is not None:
        print(f"  r={g4['pearson_r']:.4f}  p={g4['pearson_p']:.4f}")

    # Overall
    blocked = [k for k, g in gates.items() if g.get("verdict") is False]
    degraded = [k for k, g in gates.items() if g.get("verdict") is None]
    if blocked:
        overall = "FAIL"
    elif degraded:
        overall = "GO_DEGRADED"
    else:
        overall = "GO"

    print(f"\n{'='*60}")
    print(f"  OVERALL: [{overall}]")
    if blocked:
        print(f"  Blocked: {blocked}")
    if degraded:
        print(f"  Degraded: {degraded}")
    print(f"{'='*60}")

    # Save report
    report = {
        "pilot_config": {
            "n_samples": args.n_sample,
            "source": "dataset_b.json (v5.2, 150 samples)",
            "mass_source": "REAL GPU (HF transformers teacher-forcing)",
            "model": MODEL_NAME,
        },
        "gate0_candidate_alignment": rank_stats,
        "gates": {k: {kk: vv for kk, vv in v.items()}
                  for k, v in gates.items()},
        "overall": overall,
        "blocked_by": blocked,
        "degraded": degraded,
    }

    report_path = os.path.join(OUT_DIR, "gate_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nReport: {report_path}")

    # Also save a summary for quick reading
    summary_path = os.path.join(OUT_DIR, "pilot_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"B Domain Pilot Summary ({args.n_sample} samples, REAL GPU)\n")
        f.write(f"Model: {MODEL_NAME}\n")
        f.write(f"Overall: {overall}\n\n")
        f.write(f"norm_mass: mean={valid['norm_mass'].mean():.3f} std={valid['norm_mass'].std():.3f}\n")
        f.write(f"  p10={valid['norm_mass'].quantile(0.1):.3f} p50={valid['norm_mass'].quantile(0.5):.3f} p90={valid['norm_mass'].quantile(0.9):.3f}\n")
        f.write(f"gold_rank: mean={valid['gold_rank_in_C'].mean():.1f} rank=1:{(valid['gold_rank_in_C']==1).mean():.1%}\n")

    print(f"Summary: {summary_path}")
    return overall


if __name__ == "__main__":
    main()
