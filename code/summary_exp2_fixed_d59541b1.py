#!/usr/bin/env python3
"""Fixed exp2 summary — no KeyError."""
import os, sys, pandas as pd, numpy as np
from scipy.stats import pearsonr, spearmanr

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "exp2")
A = pd.read_csv(os.path.join(BASE, "mass_A.csv"))
B = pd.read_csv(os.path.join(BASE, "mass_B.csv"))

def summarize(df, label):
    v = df.dropna(subset=["norm_mass"])
    m = v["norm_mass"]
    gs = v["gold_share"]
    gr = v["gold_rank_in_C"]

    z = (m - m.mean()) / m.std()
    n_anom = int((abs(z) > 4).sum())

    print(f"\n=== {label} DOMAIN ===")
    print(f"  Total: {len(df)}  Valid: {len(v)}  NaN: {len(df)-len(v)}  Anomalies(>4sigma): {n_anom}")
    print(f"  norm_mass: mean={m.mean():.3f}  std={m.std():.3f}")
    pcts = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    vals = np.percentile(m, pcts)
    line_parts = []
    for i, p in enumerate(pcts):
        line_parts.append(f"p{p:02d}={vals[i]:.3f}")
        if i % 3 == 2:
            print(f"    {'  '.join(line_parts)}")
            line_parts = []
    if line_parts:
        print(f"    {'  '.join(line_parts)}")
    print(f"    min={m.min():.3f}  max={m.max():.3f}")
    print(f"  gold_share: mean={gs.mean():.4f}  std={gs.std():.4f}  p50={gs.median():.4f}")
    print(f"  gold_rank:  mean={gr.mean():.1f}  median={gr.median():.0f}")
    print(f"    rank=1: {(gr==1).mean():.1%}  rank<=3: {(gr<=3).mean():.1%}  rank<=5: {(gr<=5).mean():.1%}")
    print(f"    rank<=10: {(gr<=10).mean():.1%}")

    s = v.sort_values("norm_mass")
    print(f"\n  Top-5 (highest norm_mass):")
    for _, r in s.tail(5).iloc[::-1].iterrows():
        print(f"    {r['term']:20s}  mass={r['norm_mass']:.3f}  rank={int(r['gold_rank_in_C'])}  |C|={int(r['n_cand'])}")
    print(f"  Bottom-5 (lowest norm_mass):")
    for _, r in s.head(5).iterrows():
        print(f"    {r['term']:20s}  mass={r['norm_mass']:.3f}  rank={int(r['gold_rank_in_C'])}  |C|={int(r['n_cand'])}")

    return v

va = summarize(A, "A")
vb = summarize(B, "B")

print("\n=== CROSS-DOMAIN G0 (full 300) ===")
ra = va["gold_rank_in_C"].values
rb = vb["gold_rank_in_C"].values
print(f"  A: n={len(ra)}  rank_mean={ra.mean():.1f}  rank_median={np.median(ra):.0f}  p10={np.percentile(ra,10):.0f}  p90={np.percentile(ra,90):.0f}")
print(f"  B: n={len(rb)}  rank_mean={rb.mean():.1f}  rank_median={np.median(rb):.0f}  p10={np.percentile(rb,10):.0f}  p90={np.percentile(rb,90):.0f}")
ratio = rb.mean() / ra.mean()
print(f"  rank_mean_ratio (B/A): {ratio:.3f}")
print(f"  B rank<=5: {(rb<=5).mean():.1%}  rank<=10: {(rb<=10).mean():.1%}")
if ratio < 0.5:
    print("  Judgment: B gold_rank systematically lower")
elif ratio < 0.75:
    print("  Judgment: B gold_rank slightly lower, borderline")
elif ratio <= 1.5:
    print("  Judgment: A/B gold_rank comparable")
else:
    print("  Judgment: B gold_rank higher")

print("\n=== COMMON SUPPORT IoU (full 300) ===")
ma = va["norm_mass"].values
mb = vb["norm_mass"].values
all_vals = np.concatenate([ma, mb])
lo, hi = float(all_vals.min()) - 0.5, float(all_vals.max()) + 0.5
edges = np.linspace(lo, hi, 11)
ha, _ = np.histogram(ma, bins=edges, density=True)
hb, _ = np.histogram(mb, bins=edges, density=True)
iou = np.minimum(ha, hb).sum() / np.maximum(ha, hb).sum()
a_lo, a_hi = np.percentile(ma, [10, 90])
b_lo, b_hi = np.percentile(mb, [10, 90])
overlap = max(0.0, min(a_hi, b_hi) - max(a_lo, b_lo))
print(f"  IoU={iou:.4f}  overlap_width={overlap:.4f}")
print(f"  A p10-p90: [{a_lo:.3f}, {a_hi:.3f}]")
print(f"  B p10-p90: [{b_lo:.3f}, {b_hi:.3f}]")
print(f"  Verdict: {'PASS' if iou >= 0.15 and overlap > 0.05 else 'FAIL'}")

print("\n=== |C|-MASS CORRELATION ===")
for label, df_v in [("A", va), ("B", vb)]:
    if len(df_v) >= 10:
        pr, pp = pearsonr(df_v["n_cand"], df_v["norm_mass"])
        sr, sp = spearmanr(df_v["n_cand"], df_v["norm_mass"])
        print(f"  {label}: Pearson r={pr:.4f} p={pp:.4f}  Spearman r={sr:.4f} p={sp:.4f}")

print("\n=== B DOMAIN mapping_type ===")
if "mapping_type" in B.columns:
    for mt, grp in B.groupby("mapping_type"):
        vg = grp.dropna(subset=["norm_mass"])
        if len(vg) > 0:
            m = vg["norm_mass"]
            print(f"  {mt}: n={len(vg)}  norm_mass mean={m.mean():.3f} std={m.std():.3f}")
            print(f"    gold_rank mean={vg['gold_rank_in_C'].mean():.1f}  rank=1:{(vg['gold_rank_in_C']==1).mean():.1%}")
else:
    print("  (no mapping_type column)")

print("\nDone.")
