#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RO magnet effect analysis — no GPU needed, uses existing Exp1 CSVs only."""

import pandas as pd
import numpy as np
import json, os, sys
from scipy.stats import wilcoxon
from collections import Counter

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

# ── Load data ──
exp1_a = pd.read_csv(os.path.join(BASE, "results", "exp1", "exp1_A.csv"))
exp1_b = pd.read_csv(os.path.join(BASE, "results", "exp1", "exp1_B.csv"))

with open(os.path.join(BASE, "data", "dataset_a_with_candidates.json"), encoding="utf-8") as f:
    ds_a = json.load(f)
with open(os.path.join(BASE, "data", "dataset_b.json"), encoding="utf-8") as f:
    ds_b = json.load(f)


def build_term_lookup(ds_data):
    lu = {}
    for r in ds_data["test_cases"]:
        rid = r.get("protege_id") or r.get("record_id", "")
        lu[rid] = {
            "term": r.get("name", ""),
            "gold": r.get("modern_definition", ""),
            "candidates": r.get("candidates", []),
            "source": r.get("source_book", ""),
            "definition": r.get("original_definition", ""),
        }
    return lu


lu_a = build_term_lookup(ds_a)
lu_b = build_term_lookup(ds_b)


# ═══════════════════════════════════════════════════════════════
# PART 1: Identify magnet distractors
# ═══════════════════════════════════════════════════════════════

print("=" * 70)
print("PART 1: MAGNET DISTRACTOR IDENTIFICATION")
print("=" * 70)


def term_accuracy(df):
    return df.groupby(["id", "condition"])["correct"].mean().unstack()


def analyze_magnets(label, edf, lu, other_golds):
    """Identify magnet distractors and their impact."""
    ro = edf[edf["condition"] == "RO"].copy()
    ro_errors = ro[ro["correct"] == 0]

    # Count which predictions appear most in RO errors
    pred_counts = Counter(ro_errors["pred"])
    total_ro_errors = len(ro_errors)
    total_ro = len(ro)

    print(f"\n[{label}] RO errors: {total_ro_errors}/{total_ro} ({total_ro_errors / total_ro:.1%})")

    # Magnet threshold: appear in >= 5 RO errors (model-independent: captures systematic bias)
    magnets = {pred: cnt for pred, cnt in pred_counts.items() if cnt >= 5}
    magnets_sorted = sorted(magnets.items(), key=lambda x: -x[1])

    print(f"\n  Magnet distractors (>=5 errors each):")
    total_magnet_errors = 0
    for pred, cnt in magnets_sorted:
        pct_of_errors = cnt / total_ro_errors * 100
        pct_of_all_ro = cnt / total_ro * 100
        print(f"    '{pred[:60]}' -> {cnt} errors ({pct_of_errors:.1f}% of RO errors, {pct_of_all_ro:.1f}% of all RO trials)")
        total_magnet_errors += cnt
    print(f"  Total magnet errors: {total_magnet_errors}/{total_ro_errors} ({total_magnet_errors / total_ro_errors:.1%})")
    print(f"  Non-magnet errors: {total_ro_errors - total_magnet_errors}")

    return magnets, ro, ro_errors


# Collect all gold terms from other domain
a_golds = sorted(set(r.get("modern_definition", "") for r in ds_a["test_cases"]))
b_golds = sorted(set(r.get("modern_definition", "") for r in ds_b["test_cases"]))

magnets_a, ro_a, ro_err_a = analyze_magnets("A", exp1_a, lu_a, b_golds)
magnets_b, ro_b, ro_err_b = analyze_magnets("B", exp1_b, lu_b, a_golds)


# ═══════════════════════════════════════════════════════════════
# PART 2: Remove magnet-affected terms and recalculate
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PART 2: REMOVE MAGNET-AFFECTED TERMS & RECALCULATE")
print("=" * 70)


def term_acc_with_removal(edf, magnets, label):
    """Remove terms that were affected by magnet distractors, recalculate RO-HC."""
    ta = term_accuracy(edf)

    # Identify terms where RO errors involved a magnet distractor
    ro_rows = edf[edf["condition"] == "RO"]
    magnet_ids = set()
    for _, row in ro_rows.iterrows():
        if row["correct"] == 0 and row["pred"] in magnets:
            magnet_ids.add(row["id"])

    print(f"\n[{label}] Magnet-affected terms: {len(magnet_ids)}")
    if len(magnet_ids) <= 20:
        for mid in sorted(magnet_ids):
            info = lu_a.get(mid) or lu_b.get(mid) or {}
            print(f"    {mid}: {info.get('term', '?')}")

    # Remove these terms from analysis
    ta_clean = ta.drop(index=[i for i in magnet_ids if i in ta.index], errors="ignore")
    n_removed = len(ta) - len(ta_clean)
    print(f"  Terms removed: {n_removed}, remaining: {len(ta_clean)}")

    return ta, ta_clean, magnet_ids


def paired_test(ta_data, cond_a, cond_b):
    """Wilcoxon signed-rank + rank-biserial correlation on term-level data."""
    a = ta_data[cond_a].dropna()
    b = ta_data[cond_b].dropna()
    common = a.index.intersection(b.index)
    a, b = a[common], b[common]
    diff = a - b
    n = len(diff)
    if n < 5:
        return None, 1.0, 0.0, diff.mean()

    stat, p = wilcoxon(a, b, alternative="two-sided")

    # Rank-biserial
    from scipy.stats import rankdata
    abs_diff = np.abs(diff.values)
    ranks = rankdata(abs_diff)
    w_pos = ranks[diff.values > 0].sum() if (diff.values > 0).any() else 0
    w_neg = ranks[diff.values < 0].sum() if (diff.values < 0).any() else 0
    rbc = (w_pos - w_neg) / (w_pos + w_neg) if (w_pos + w_neg) > 0 else 0

    # Bootstrap CI on mean diff
    rng = np.random.default_rng(42)
    bs = []
    for _ in range(10000):
        idx = rng.choice(n, n, replace=True)
        bs.append(diff.iloc[idx].mean())
    ci_lo, ci_hi = np.percentile(bs, [2.5, 97.5])

    return stat, p, rbc, diff.mean(), ci_lo, ci_hi


print("\n" + "-" * 50)
print("RO vs HC: BEFORE vs AFTER magnet removal")
print("-" * 50)

for label, edf, lu, magnets in [
    ("A", exp1_a, lu_a, magnets_a),
    ("B", exp1_b, lu_b, magnets_b),
]:
    print(f"\n{'=' * 40}")
    print(f"[{label} DOMAIN]")
    print(f"{'=' * 40}")

    ta_full, ta_clean, magnet_ids = term_acc_with_removal(edf, magnets, label)

    # BEFORE
    result = paired_test(ta_full, "RO", "HC")
    if result[0] is not None:
        stat, p, rbc, mean_diff, ci_lo, ci_hi = result
        print(f"\n  BEFORE removal:")
        print(f"    n = {len(ta_full)} terms")
        print(f"    RO = {ta_full['RO'].mean():.3f}, HC = {ta_full['HC'].mean():.3f}")
        print(f"    RO - HC = {mean_diff:.3f} [95% CI: {ci_lo:.3f}, {ci_hi:.3f}]")
        print(f"    Wilcoxon W = {stat:.0f}, p = {p:.6f}")
        print(f"    Rank-biserial r = {rbc:.3f}")
        print(f"    RO > HC: {(ta_full['RO'] > ta_full['HC']).mean():.1%} of terms")

    # AFTER
    result_c = paired_test(ta_clean, "RO", "HC")
    if result_c[0] is not None:
        stat, p, rbc, mean_diff, ci_lo, ci_hi = result_c
        print(f"\n  AFTER removal:")
        print(f"    n = {len(ta_clean)} terms (removed {len(magnet_ids)})")
        print(f"    RO = {ta_clean['RO'].mean():.3f}, HC = {ta_clean['HC'].mean():.3f}")
        print(f"    RO - HC = {mean_diff:.3f} [95% CI: {ci_lo:.3f}, {ci_hi:.3f}]")
        print(f"    Wilcoxon W = {stat:.0f}, p = {p:.6f}")
        print(f"    Rank-biserial r = {rbc:.3f}")
        print(f"    RO > HC: {(ta_clean['RO'] > ta_clean['HC']).mean():.1%} of terms")

    # Compare
    delta_diff = ta_full["RO"].mean() - ta_full["HC"].mean() - (ta_clean["RO"].mean() - ta_clean["HC"].mean())
    print(f"\n  CHANGE: RO-HC mean diff reduced by {delta_diff:.3f}")


# ═══════════════════════════════════════════════════════════════
# PART 3: Magnet word characterization
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PART 3: MAGNET WORD CHARACTERIZATION")
print("=" * 70)


def characterize_magnets(magnets_dict, ro_pool, label):
    print(f"\n[{label}] Magnet distractor features:")
    print(f"  {'Word':<40} {'Len':>5} {'RO pool freq':>12} {'In HC pool?':>12}")
    print(f"  {'-' * 40} {'-' * 5} {'-' * 12} {'-' * 12}")

    for pred, cnt in sorted(magnets_dict.items(), key=lambda x: -x[1]):
        char_len = len(pred)
        ro_freq = sum(1 for g in ro_pool if g == pred or pred in g or g in pred)
        # Note: we can't know exactly which HC sets contain each magnet without
        # checking all 150 terms, but we CAN estimate by counting how many
        # terms have this as a candidate
        print(f"  {pred[:38]:<40} {char_len:>5} {ro_freq:>12} {'N/A':>12}")

    # Aggregate stats
    magnet_lens = [len(p) for p in magnets_dict]
    all_ro_lens = [len(g) for g in ro_pool]

    print(f"\n  Magnet mean length: {np.mean(magnet_lens):.1f} chars")
    print(f"  RO pool mean length: {np.mean(all_ro_lens):.1f} chars")
    print(f"  Ratio: {np.mean(magnet_lens) / np.mean(all_ro_lens):.2f}")


characterize_magnets(magnets_a, b_golds, "A")
characterize_magnets(magnets_b, a_golds, "B")


# ═══════════════════════════════════════════════════════════════
# PART 4: RO distractor construction analysis
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PART 4: RO DISTRACTOR CONSTRUCTION PATTERN")
print("=" * 70)

# Check: are the same RO distractors used across all terms?
# We need to reconstruct the RO sets from the build_sets logic.
# RO = [gold] + (k-1) random items from ro_pool (other-domain golds) minus own candidates.
# For each term, RO valid pool = all other-domain golds minus {own gold} minus {own candidates}.

for label, ds_data, ro_pool_name in [
    ("A", ds_a, "B-domain golds"),
    ("B", ds_b, "A-domain golds"),
]:
    ro_pool_size = len(set(r.get("modern_definition", "") for r in ds_data["test_cases"]))
    # Count total unique RO valid items per term
    pool_sizes = []
    for r in ds_data["test_cases"]:
        gold = r.get("modern_definition", "")
        own_cands = set(r.get("candidates", []))
        if label == "A":
            ro_valid = [g for g in b_golds if g != gold and g not in own_cands]
        else:
            ro_valid = [g for g in a_golds if g != gold and g not in own_cands]
        pool_sizes.append(len(ro_valid))

    print(f"\n[{label}] RO pool = {ro_pool_name}")
    print(f"  RO pool size (other-domain golds): {len(b_golds) if label == 'A' else len(a_golds)}")
    print(f"  Per-term RO-valid pool: min={min(pool_sizes)}, max={max(pool_sizes)}, "
          f"mean={np.mean(pool_sizes):.0f}, median={np.median(pool_sizes):.0f}")
    print(f"  Terms where RO-valid pool < 50: {sum(1 for s in pool_sizes if s < 50)}/{len(pool_sizes)}")

    # Key question: does every term see the same RO pool?
    # Answer: the RO-valid pool is nearly identical across terms (all other-domain golds),
    # minus own candidates (which are a tiny fraction). So yes, essentially shared.
    # The random sampling introduces variation, but the underlying pool is the same.

    # How many RO distractors does each term actually have?
    k_avg = np.mean([len(r.get("candidates", [])) for r in ds_data["test_cases"]])
    print(f"  Average |C| = {k_avg:.0f} → RO set size = {k_avg:.0f} (1 gold + {k_avg - 1:.0f} distractors)")
    print(f"  RO pool utilization: each term samples {k_avg - 1:.0f} from {np.mean(pool_sizes):.0f} valid items")

print("\n  KEY: RO distractors are randomly sampled EACH TIME (per rep, per term)")
print("  from a shared pool (all other-domain golds). Different terms may see")
print("  the same magnet distractors, and because the sampling is random with")
print("  3 reps per term, the same magnet can appear in all 3 reps for a term.")
print("  This means magnet effect is partly a DESIGN consequence (shared pool)")
print("  and partly a MODEL behavior (which items become magnets).")


# ═══════════════════════════════════════════════════════════════
# PART 5: FULL SUMMARY & VERDICT
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PART 5: SUMMARY & VERDICT")
print("=" * 70)

# Recompute clean stats for both domains
for label, edf, lu, magnets in [
    ("A", exp1_a, lu_a, magnets_a),
    ("B", exp1_b, lu_b, magnets_b),
]:
    ta_full = term_accuracy(edf)
    ro_rows = edf[edf["condition"] == "RO"]
    magnet_ids = set()
    for _, row in ro_rows.iterrows():
        if row["correct"] == 0 and row["pred"] in magnets:
            magnet_ids.add(row["id"])
    ta_clean = ta_full.drop(index=[i for i in magnet_ids if i in ta_full.index], errors="ignore")

    before_diff = ta_full["RO"].mean() - ta_full["HC"].mean()
    after_diff = ta_clean["RO"].mean() - ta_clean["HC"].mean()
    n_magnet_terms = len(magnet_ids)

    _, p_before, rbc_before, _, _, _ = paired_test(ta_full, "RO", "HC")
    _, p_after, rbc_after, _, _, _ = paired_test(ta_clean, "RO", "HC")

    print(f"\n[{label}] Magnet terms removed: {n_magnet_terms}/{len(ta_full)} ({n_magnet_terms / len(ta_full):.1%})")
    print(f"  RO-HC before: {before_diff:.3f}, p={p_before:.6f}, r={rbc_before:.3f}")
    print(f"  RO-HC after:  {after_diff:.3f}, p={p_after:.6f}, r={rbc_after:.3f}")
    print(f"  Delta: {before_diff - after_diff:.3f}")
    print(f"  RO>HC term fraction: before={(ta_full['RO'] > ta_full['HC']).mean():.1%}, "
          f"after={(ta_clean['RO'] > ta_clean['HC']).mean():.1%}")

print("\n" + "=" * 70)
print("VERDICT:")
print("  If RO-HC remains significant (p<0.05) and effect size remains large")
print("  after removing magnet-affected terms → RO>HC is ROBUST")
print("  If RO-HC becomes non-significant or effect size collapses → RO>HC is")
print("  an artifact driven by a few 'magnet' distractors.")
print("=" * 70)
