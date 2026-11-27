#!/usr/bin/env python3
"""Regenerate gate report from existing mass_B_pilot.csv."""
import json, math, os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "pilot_b")

df_b = pd.read_csv(os.path.join(OUT, "mass_B_pilot.csv"))
mass_b = df_b["norm_mass"].dropna().values
valid = df_b.dropna(subset=["norm_mass"])
print(f"B samples: {len(valid)} valid / {len(df_b)} total")

df_a = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "exp2", "mass_A.csv"))
mass_a = df_a["norm_mass"].dropna().values
print(f"A samples: {len(mass_a)}")

# G0
ra = df_a["gold_rank_in_C"].dropna().values
rb = valid["gold_rank_in_C"].dropna().values
rank_stats = {
    "A_n": int(len(ra)), "A_rank_mean": float(ra.mean()),
    "A_rank_median": float(np.median(ra)),
    "A_rank_p10": float(np.percentile(ra, 10)),
    "A_rank_p90": float(np.percentile(ra, 90)),
    "B_n": int(len(rb)), "B_rank_mean": float(rb.mean()),
    "B_rank_median": float(np.median(rb)),
    "B_rank_p10": float(np.percentile(rb, 10)),
    "B_rank_p90": float(np.percentile(rb, 90)),
    "B_rank_le_5_frac": float((rb <= 5).mean()),
    "B_rank_le_10_frac": float((rb <= 10).mean()),
}
ratio = rank_stats["B_rank_mean"] / rank_stats["A_rank_mean"]
rank_stats["rank_mean_ratio_B_over_A"] = round(ratio, 3)
if ratio < 0.5:
    rank_stats["alignment_judgment"] = "B gold_rank systematically lower"
elif ratio < 0.75:
    rank_stats["alignment_judgment"] = "B gold_rank slightly lower, borderline"
elif ratio <= 1.5:
    rank_stats["alignment_judgment"] = "A/B gold_rank comparable"
else:
    rank_stats["alignment_judgment"] = "B gold_rank higher"

# G1
spread = float(np.percentile(mass_b, 90) - np.percentile(mass_b, 10))
low_frac = float((mass_b < 0).mean())
high_frac = float((mass_b > math.log(2)).mean())
g1_ok = spread >= 1.5 and low_frac >= 0.10 and high_frac >= 0.10
g1 = {"verdict": g1_ok, "spread": spread, "low_frac": low_frac, "high_frac": high_frac,
      "n": int(len(mass_b)), "note": "mass dynamic range sufficient" if g1_ok else "range insufficient"}

# G2
all_vals = np.concatenate([mass_a, mass_b])
lo, hi = float(all_vals.min()) - 0.5, float(all_vals.max()) + 0.5
edges = np.linspace(lo, hi, 11)
ha, _ = np.histogram(mass_a, bins=edges, density=True)
hb, _ = np.histogram(mass_b, bins=edges, density=True)
iou = float(np.minimum(ha, hb).sum() / np.maximum(ha, hb).sum())
a_lo, a_hi = np.percentile(mass_a, [10, 90])
b_lo, b_hi = np.percentile(mass_b, [10, 90])
overlap = max(0.0, min(a_hi, b_hi) - max(a_lo, b_lo))
g2_ok = iou >= 0.15 and overlap > 0.05
g2 = {"iou": round(iou, 4), "overlap_width": round(overlap, 4),
      "a_p10_p90": [float(a_lo), float(a_hi)], "b_p10_p90": [float(b_lo), float(b_hi)],
      "verdict": g2_ok, "note": "common support OK" if g2_ok else "no overlap -> degrade H4"}

# G3
above = float((mass_b > 0).mean())
below = float((mass_b < 0).mean())
g3 = {"n": int(len(mass_b)), "mean": float(mass_b.mean()), "std": float(mass_b.std()),
      "p10": float(np.percentile(mass_b, 10)), "p50": float(np.median(mass_b)),
      "p90": float(np.percentile(mass_b, 90)),
      "frac_above_baseline": above, "frac_below_baseline": below,
      "verdict": True, "note": "good discrimination"}

# G4
sub = valid.dropna(subset=["n_cand", "norm_mass"])
pr, pp = pearsonr(sub["n_cand"], sub["norm_mass"])
sr, sp = spearmanr(sub["n_cand"], sub["norm_mass"])
abs_max = max(abs(pr), abs(sr))
g4_ok = abs_max <= 0.3
if abs_max <= 0.3:
    g4_note = f"|C|-mass weak correlation (r={abs_max:.3f})"
elif abs_max > 0.5:
    g4_note = f"|C|-mass strong correlation (r={abs_max:.3f})"
else:
    g4_note = f"|C|-mass moderate correlation (r={abs_max:.3f})"
g4 = {"pearson_r": round(pr, 4), "pearson_p": round(pp, 6),
      "spearman_r": round(sr, 4), "spearman_p": round(sp, 6),
      "verdict": g4_ok, "note": g4_note}

# Summary
gates = {"1_dynamic_range": g1, "2_common_support": g2, "3_mass_dist": g3, "4_cand_size_bias": g4}
blocked = [k for k, g in gates.items() if g.get("verdict") is False]
degraded = [k for k, g in gates.items() if g.get("verdict") is None]
overall = "FAIL" if blocked else ("GO_DEGRADED" if degraded else "GO")

print(f"G1: {'PASS' if g1_ok else 'FAIL'} spread={spread:.3f} low={low_frac:.1%} high={high_frac:.1%}")
print(f"G2: {'PASS' if g2_ok else 'FAIL'} IoU={iou:.4f} overlap={overlap:.3f}")
print(f"G3: PASS mean={g3['mean']:.3f} above={above:.1%} below={below:.1%}")
print(f"G4: {'PASS' if g4_ok else ('FAIL' if g4_ok is False else 'SKIP')} r={pr:.4f} p={pp:.4f}")
print(f"")
print(f"G0 rank: A mean={rank_stats['A_rank_mean']:.1f} B mean={rank_stats['B_rank_mean']:.1f} ratio={ratio:.3f}")
print(f"  B rank<=5: {rank_stats['B_rank_le_5_frac']:.1%} rank<=10: {rank_stats['B_rank_le_10_frac']:.1%}")
print(f"  -> {rank_stats['alignment_judgment']}")
print(f"")
print(f"norm_mass B: mean={g3['mean']:.3f} std={g3['std']:.3f} p10={g3['p10']:.3f} p50={g3['p50']:.3f} p90={g3['p90']:.3f}")
print(f"")
print(f"OVERALL: [{overall}]")

report = {
    "pilot_config": {"n_samples": 30, "source": "dataset_b.json (v5.2)", "mass_source": "REAL GPU (HF transformers teacher-forcing)", "model": "/root/autodl-tmp/models/Qwen3-32B"},
    "gate0_candidate_alignment": rank_stats,
    "gates": gates,
    "overall": overall, "blocked_by": blocked, "degraded": degraded,
    "recommendation": "GO -> proceed to full Exp2" if overall == "GO" else "Check blocked gates before proceeding"
}
with open(os.path.join(OUT, "gate_report.json"), "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)
print(f"Report saved to {OUT}/gate_report.json")
