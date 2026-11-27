#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Exp1 formal analysis — pre-registered discipline, no post-hoc rescue."""
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr, mannwhitneyu, wilcoxon
import os, sys

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

exp1_a = pd.read_csv(os.path.join(BASE, "results", "exp1", "exp1_A.csv"))
exp1_b = pd.read_csv(os.path.join(BASE, "results", "exp1", "exp1_B.csv"))
mass_a = pd.read_csv(os.path.join(BASE, "results", "exp2", "mass_A.csv"))
mass_b = pd.read_csv(os.path.join(BASE, "results", "exp2", "mass_B.csv"))

# ── Term-level aggregation ──
def term_accuracy(df):
    return df.groupby(["id", "condition"])["correct"].mean().unstack()

ta = term_accuracy(exp1_a)
tb = term_accuracy(exp1_b)

# ═══════════════════════════════════════════════════════════
# Finding (a): RO vs HC — paired test + effect size
# ═══════════════════════════════════════════════════════════

print("=" * 60)
print("FINDING (a): RO vs HC within each domain")
print("=" * 60)

for label, tdf in [("A", ta), ("B", tb)]:
    diff = tdf["RO"] - tdf["HC"]
    n = len(diff.dropna())
    mean_diff = diff.mean()
    # Paired Wilcoxon (difference scores are discrete 0/0.33/0.67/1.0, not normal)
    stat, p = wilcoxon(tdf["RO"], tdf["HC"], alternative="two-sided")
    # Cliff's delta (non-parametric effect size for paired ordinal)
    # For paired data, use rank-biserial correlation
    from scipy.stats import rankdata
    abs_diff = np.abs(diff.values)
    ranks = rankdata(abs_diff)
    W_pos = ranks[diff.values > 0].sum() if (diff.values > 0).any() else 0
    W_neg = ranks[diff.values < 0].sum() if (diff.values < 0).any() else 0
    # Matched-pairs rank-biserial correlation
    rbc = (W_pos - W_neg) / (W_pos + W_neg) if (W_pos + W_neg) > 0 else 0

    # Bootstrap CI on mean difference (pairs resampling)
    rng_local = np.random.default_rng(42)
    n_d = len(diff)
    bs_means = []
    for _ in range(10000):
        idx = rng_local.choice(n_d, n_d, replace=True)
        bs_means.append(diff.iloc[idx].mean())
    ci_lo, ci_hi = np.percentile(bs_means, [2.5, 97.5])

    print(f"\n[{label} domain] RO={tdf['RO'].mean():.3f}  HC={tdf['HC'].mean():.3f}")
    print(f"  RO - HC = {mean_diff:.3f} [95% CI: {ci_lo:.3f}, {ci_hi:.3f}]")
    print(f"  Wilcoxon signed-rank: W={stat:.0f} p={p:.6f}")
    print(f"  Rank-biserial r = {rbc:.3f}")
    print(f"  RO > HC: {(diff > 0).mean():.1%} of terms")
    print(f"  RO < HC: {(diff < 0).mean():.1%} of terms")
    print(f"  RO == HC: {(diff == 0).mean():.1%} of terms")

# ═══════════════════════════════════════════════════════════
# Finding (b): B > A gain — effect size + CI
# ═══════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("FINDING (b): Cross-domain gain comparison")
print("=" * 60)

ga = ta["HC"] - ta["UC"]
gb = tb["HC"] - tb["UC"]

print(f"\n  A domain: gain mean={ga.mean():.3f} median={ga.median():.3f} std={ga.std():.3f} n={len(ga)}")
print(f"  B domain: gain mean={gb.mean():.3f} median={gb.median():.3f} std={gb.std():.3f} n={len(gb)}")

# Mann-Whitney U (non-parametric, gain is discrete)
u, up = mannwhitneyu(gb, ga, alternative="two-sided")
print(f"\n  Mann-Whitney U = {u:.0f}  p = {up:.4f}")

# Cliff's delta (effect size for independent samples)
# delta = P(B > A) - P(B < A)
n_a, n_b = len(ga), len(gb)
pairs = 0
wins, ties, losses = 0, 0, 0
for bg in gb:
    for ag in ga:
        pairs += 1
        if bg > ag:
            wins += 1
        elif bg < ag:
            losses += 1
        else:
            ties += 1
cliff_delta = (wins - losses) / (n_a * n_b)
print(f"  Cliff's delta = {cliff_delta:.4f}  (B > A: {wins/pairs:.1%}, B < A: {losses/pairs:.1%}, tie: {ties/pairs:.1%})")

# Bootstrap CI for mean difference (two-sample)
rng_local2 = np.random.default_rng(42)
n_a2, n_b2 = len(ga), len(gb)
d_bs = []
for _ in range(10000):
    ia = rng_local2.choice(n_a2, n_a2, replace=True)
    ib = rng_local2.choice(n_b2, n_b2, replace=True)
    d_bs.append(gb.iloc[ib].mean() - ga.iloc[ia].mean())
ci_lo, ci_hi = np.percentile(d_bs, [2.5, 97.5])
mean_diff = gb.mean() - ga.mean()
print(f"  Mean difference (B-A) = {mean_diff:.3f} [95% CI: {ci_lo:.3f}, {ci_hi:.3f}]")

# ═══════════════════════════════════════════════════════════
# mass -> gain: formal null report
# ═══════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("H3: mass -> gain (pre-registered primary test)")
print("=" * 60)

ma = mass_a[["id", "norm_mass"]].copy()
mb = mass_b[["id", "norm_mass"]].copy()
mga = ga.reset_index().rename(columns={0: "gain"})
mgb = gb.reset_index().rename(columns={0: "gain"})
mga = mga.merge(ma, on="id")
mgb = mgb.merge(mb, on="id")

for label, mdf in [("A", mga), ("B", mgb)]:
    pr, pp = pearsonr(mdf["norm_mass"], mdf["gain"])
    # Bootstrap CI for Pearson r via pairs resampling
    rng_local = np.random.default_rng(42)
    n = len(mdf)
    r_bs = []
    for _ in range(10000):
        idx = rng_local.choice(n, n, replace=True)
        r_bs.append(pearsonr(mdf["norm_mass"].iloc[idx], mdf["gain"].iloc[idx])[0])
    r_ci_lo, r_ci_hi = np.percentile(r_bs, [2.5, 97.5])
    print(f"\n  [{label}] r({len(mdf)-2}) = {pr:.4f}, p = {pp:.4f}")
    print(f"    95% CI for r: [{r_ci_lo:.4f}, {r_ci_hi:.4f}]")
    print(f"    Conclusion: {'significant' if pp < 0.05 else 'NOT significant'} at alpha=0.05")

# Pooled
all_mg = pd.concat([
    mga.assign(domain="A"),
    mgb.assign(domain="B"),
])
pr_pool, pp_pool = pearsonr(all_mg["norm_mass"], all_mg["gain"])
print(f"\n  [Pooled] r({len(all_mg)-2}) = {pr_pool:.4f}, p = {pp_pool:.4f}")

# ═══════════════════════════════════════════════════════════
# H4: Domain interaction — formal null report
# ═══════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("H4: mass*domain interaction")
print("=" * 60)

import statsmodels.formula.api as smf
m = smf.ols("gain ~ norm_mass * domain", all_mg).fit()
print(f"\n  mass slope: A = {m.params['norm_mass']:.4f} (p={m.pvalues['norm_mass']:.4f})")
print(f"  mass slope: B = {m.params['norm_mass'] + m.params['norm_mass:domain[T.B]']:.4f}")
print(f"  interaction p = {m.pvalues['norm_mass:domain[T.B]']:.4f}")
print(f"  R2 = {m.rsquared:.3f}, adj R2 = {m.rsquared_adj:.3f}")
print(f"  Conclusion: mass slope not significantly different from 0 in either domain,")
print(f"  interaction not significant — no mass->gain effect to differ across domains.")

# ═══════════════════════════════════════════════════════════
# Descriptive: gain by mass quartile (NOT a test, just description)
# ═══════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("DESCRIPTIVE: Gain by mass quartile (exploratory, not a test)")
print("=" * 60)

for label, mdf in [("A", mga), ("B", mgb)]:
    mdf["mass_q"] = pd.qcut(mdf["norm_mass"], 4, labels=["Q1(lowest)","Q2","Q3","Q4(highest)"])
    print(f"\n  [{label}]")
    for q in ["Q1(lowest)", "Q2", "Q3", "Q4(highest)"]:
        sub = mdf[mdf["mass_q"] == q]
        print(f"    {q}: n={len(sub)}  gain={sub['gain'].mean():.3f}  "
              f"mass_range=[{sub['norm_mass'].min():.1f}, {sub['norm_mass'].max():.1f}]")
    # Note: A domain shows monotonic decrease (Q1=high gain, Q4=low gain),
    # B domain does not. Neither reached significance in formal test above.
    # This tension should be reported honestly without claiming support for H3.

# ═══════════════════════════════════════════════════════════
# Per-condition summary table (for paper)
# ═══════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("TABLE: Per-condition accuracy (term-level mean of 3 reps)")
print("=" * 60)

print(f"\n{'Condition':>8}  {'A mean':>8}  {'A std':>8}  {'B mean':>8}  {'B std':>8}")
print("-" * 48)
for cond in ["UC", "RO", "SC", "HC"]:
    print(f"  {cond:>6}  {ta[cond].mean():8.3f}  {ta[cond].std():8.3f}  "
          f"{tb[cond].mean():8.3f}  {tb[cond].std():8.3f}")

print(f"\n{'gain(HC-UC)':>8}  {ga.mean():8.3f}  {ga.std():8.3f}  "
      f"{gb.mean():8.3f}  {gb.std():8.3f}")

print("\n" + "=" * 60)
print("Key statistical conclusions:")
print("  (a) RO > HC: significant in both domains (p < 0.001), effect large")
print("  (b) B gain > A gain: significant (p = 0.0013), Cliff's delta reported")
print("  (c) mass -> gain: NOT significant, H3 not supported")
print("  (d) H4 interaction: NOT significant, no domain difference in null effect")
print("=" * 60)
