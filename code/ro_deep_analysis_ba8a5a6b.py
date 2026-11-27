#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RO>HC deep analysis: (1) error patterns (2) surface-feature check (3) logprob collection plan."""
import pandas as pd
import numpy as np
import json, os, re

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

# ── Load data ──
exp1_a = pd.read_csv(os.path.join(BASE, "results", "exp1", "exp1_A.csv"))
exp1_b = pd.read_csv(os.path.join(BASE, "results", "exp1", "exp1_B.csv"))

with open(os.path.join(BASE, "data", "dataset_a_with_candidates.json"), encoding="utf-8") as f:
    ds_a = json.load(f)
with open(os.path.join(BASE, "data", "dataset_b.json"), encoding="utf-8") as f:
    ds_b = json.load(f)

# ── Term-level accuracy ──
def term_acc(df):
    return df.groupby(["id", "condition"])["correct"].mean().unstack()

ta = term_acc(exp1_a)
tb = term_acc(exp1_b)

# ═══════════════════════════════════════════════════════════════
# 1. RO error pattern: what does the model output when RO fails?
# ═══════════════════════════════════════════════════════════════

print("=" * 70)
print("PART 1: RO ERROR PATTERN ANALYSIS")
print("=" * 70)

# Build term lookup
def build_lookup(ds_data):
    lookup = {}
    for r in ds_data["test_cases"]:
        rid = r.get("protege_id") or r.get("record_id", "")
        lookup[rid] = {
            "term": r.get("name", ""),
            "gold": r.get("modern_definition", ""),
            "candidates": r.get("candidates", []),
        }
    return lookup

lu_a = build_lookup(ds_a)
lu_b = build_lookup(ds_b)

for label, edf, lu in [("A", exp1_a, lu_a), ("B", exp1_b, lu_b)]:
    print(f"\n{'='*50}")
    print(f"[{label} DOMAIN] RO error analysis")
    print(f"{'='*50}")

    # RO rows only
    ro = edf[edf["condition"] == "RO"].copy()
    ro_errors = ro[ro["correct"] == 0]
    ro_correct = ro[ro["correct"] == 1]

    print(f"Total RO instances: {len(ro)}")
    print(f"RO errors: {len(ro_errors)} ({len(ro_errors)/len(ro):.1%})")
    print(f"RO correct: {len(ro_correct)} ({len(ro_correct)/len(ro):.1%})")

    # Term-level
    ro_term = ro.groupby("id")["correct"].mean()
    n_perfect = (ro_term == 1.0).sum()
    n_zero = (ro_term == 0.0).sum()
    n_partial = ((ro_term > 0) & (ro_term < 1.0)).sum()
    print(f"Term-level RO acc: perfect={n_perfect}, zero={n_zero}, partial={n_partial}")
    print(f"  (all 3 reps correct: {n_perfect}, all 3 wrong: {n_zero}, mixed: {n_partial})")

    # What does model output on RO errors?
    print(f"\n--- RO error predictions (sample of first 20) ---")
    for i, (_, row) in enumerate(ro_errors.head(20).iterrows()):
        rid = row["id"]
        term_info = lu.get(rid, {})
        gold = term_info.get("gold", "?")
        pred = row["pred"]
        raw = str(row["raw_output"])[:80]
        print(f"  term={term_info.get('term','?')} | gold='{gold}' | pred='{pred}'")

    # Unique wrong predictions
    wrong_preds = ro_errors["pred"].value_counts().head(20)
    print(f"\n--- Top-20 wrong predictions in RO ---")
    for pred, cnt in wrong_preds.items():
        pct = cnt / len(ro_errors) * 100
        print(f"  '{pred}': {cnt} ({pct:.1f}%)")

    # Check: are RO errors picking distractors from within the RO set?
    # (We'd need the RO set per term at each rep, which varies by shuffle seed.
    #  But we can check if prediction is in other-domain golds generally.)
    print(f"\n--- Checking if RO errors pick other-domain golds ---")
    # For A domain, RO pool = B-domain golds
    # For B domain, RO pool = A-domain golds
    ro_pool = set()
    for v in lu.values():
        ro_pool.add(v["gold"])
    n_in_pool = sum(1 for p in ro_errors["pred"] if p in ro_pool)
    print(f"  RO errors where pred IS in RO pool: {n_in_pool}/{len(ro_errors)} ({n_in_pool/len(ro_errors):.1%})")

    # Check if errors pick from HC (the correct candidates for this term)
    n_hc = 0
    for _, row in ro_errors.iterrows():
        rid = row["id"]
        term_info = lu.get(rid, {})
        cands = term_info.get("candidates", [])
        if row["pred"] in cands:
            n_hc += 1
    print(f"  RO errors where pred IS in own HC set: {n_hc}/{len(ro_errors)} ({n_hc/len(ro_errors):.1%})")

    # Check if errors pick the gold itself with different wording
    n_gold_like = 0
    for _, row in ro_errors.iterrows():
        rid = row["id"]
        term_info = lu.get(rid, {})
        gold = term_info.get("gold", "")
        pred = str(row["pred"])
        # Simple substring or edit-distance check
        if gold and pred:
            # Check if pred contains gold or gold contains pred
            if pred in gold or gold in pred:
                n_gold_like += 1
    print(f"  RO errors where pred is substring of gold (or vice versa): {n_gold_like}/{len(ro_errors)}")


# ═══════════════════════════════════════════════════════════════
# 2. Surface-feature analysis: domain marker detection
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PART 2: SURFACE-FEATURE ANALYSIS (domain markers)")
print("=" * 70)

def has_math_markers(text):
    """Check for Chinese math domain markers."""
    text = str(text)
    markers = [
        "数", "几何", "代数", "方程", "三角", "面积", "体积",
        "乘", "除", "加", "减", "根", "幂", "率", "度",
        "勾股", "圆周", "比例", "开方", "平方", "立方",
        "弦", "弧", "角", "圆", "方", "径", "积",
    ]
    return sum(1 for m in markers if m in text)

def has_clinical_markers(text):
    """Check for clinical domain markers."""
    text = str(text)
    markers = [
        "病", "症", "炎", "肿瘤", "损伤", "中毒", "感染",
        "衰竭", "出血", "骨折", "分娩", "术后", "先天",
        "急性", "慢性", "恶性", "良性", "转移", "并发",
        "综合征", "异常", "疾患", "障碍", "缺失", "坏死",
    ]
    return sum(1 for m in markers if m in text)

def domain_marker_score(text_a, text_b):
    """How much more A-like than B-like is this text? Positive = more math-like."""
    return has_math_markers(text_a) - has_clinical_markers(text_b)

# Collect gold terms from both domains
a_golds = set()
for r in ds_a["test_cases"]:
    a_golds.add(r.get("modern_definition", ""))
b_golds = set()
for r in ds_b["test_cases"]:
    b_golds.add(r.get("modern_definition", ""))

# For each RO error, check: does the WRONG prediction have domain markers
# matching the TARGET domain (suggesting the model was fooled by surface features)?
print("\n--- Domain marker analysis of RO errors ---")
for label, edf, lu, target_marker_fn in [
    ("A", exp1_a, lu_a, has_math_markers),
    ("B", exp1_b, lu_b, has_clinical_markers),
]:
    ro = edf[edf["condition"] == "RO"]
    ro_errors = ro[ro["correct"] == 0]
    ro_correct = ro[ro["correct"] == 1]

    # Compare domain markers in correct vs error predictions
    err_markers = []
    cor_markers = []
    for _, row in ro_errors.iterrows():
        pred = str(row["pred"])
        err_markers.append(target_marker_fn(pred))
    for _, row in ro_correct.iterrows():
        pred = str(row["pred"])
        cor_markers.append(target_marker_fn(pred))

    print(f"\n[{label}] Target-domain markers in predictions:")
    print(f"  Error preds: mean={np.mean(err_markers):.2f} markers")
    print(f"  Correct preds (should be gold, ~0 markers since gold IS target domain): mean={np.mean(cor_markers):.2f} markers")

    # Check gold terms: do they have target-domain markers?
    if label == "A":
        gold_markers = [has_math_markers(g) for g in a_golds]
    else:
        gold_markers = [has_clinical_markers(g) for g in b_golds]
    print(f"  Gold terms: mean={np.mean(gold_markers):.2f} markers, "
          f"frac>0={np.mean([m>0 for m in gold_markers]):.1%}")

    # Cross-domain marker check: do RO pool items have target-domain markers?
    if label == "A":
        ro_pool_items = b_golds  # B golds = RO pool for A
        ro_pool_markers = [has_math_markers(g) for g in ro_pool_items]
    else:
        ro_pool_items = a_golds  # A golds = RO pool for B
        ro_pool_markers = [has_clinical_markers(g) for g in ro_pool_items]
    print(f"  RO pool (other-domain golds): mean={np.mean(ro_pool_markers):.2f} target markers, "
          f"frac>0={np.mean([m>0 for m in ro_pool_markers]):.1%}")


# ═══════════════════════════════════════════════════════════════
# 3. String similarity: gold vs RO distractors
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PART 3: STRING SIMILARITY GOLD vs DISTRACTORS")
print("=" * 70)

try:
    from difflib import SequenceMatcher
except ImportError:
    SequenceMatcher = None

def char_sim(a, b):
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()

# For each term, compute: gold-RO_distractor similarity vs gold-HC_distractor similarity
print("\n--- String similarity analysis ---")
for label, ds_data, lu in [("A", ds_a, lu_a), ("B", ds_b, lu_b)]:
    if label == "A":
        ro_pool_items = b_golds
    else:
        ro_pool_items = a_golds

    gold_ro_sims = []
    gold_hc_sims = []

    for r in ds_data["test_cases"]:
        rid = r.get("protege_id") or r.get("record_id", "")
        gold = r.get("modern_definition", "")
        cands = r.get("candidates", [])

        if not gold:
            continue

        # avg sim between gold and RO pool items (sample 50 for speed)
        ro_sample = list(ro_pool_items)[:50]
        for ro_item in ro_sample:
            if ro_item != gold:
                gold_ro_sims.append(char_sim(gold, ro_item))

        # avg sim between gold and own HC candidates
        for hc_item in cands:
            if hc_item != gold:
                gold_hc_sims.append(char_sim(gold, hc_item))

    print(f"\n[{label}]")
    print(f"  Gold vs RO distractors: mean sim={np.mean(gold_ro_sims):.4f}, "
          f"median={np.median(gold_ro_sims):.4f}")
    print(f"  Gold vs HC candidates:  mean sim={np.mean(gold_hc_sims):.4f}, "
          f"median={np.median(gold_hc_sims):.4f}")
    # If gold-HC sim >> gold-RO sim, model might use surface similarity as shortcut

# ═══════════════════════════════════════════════════════════════
# 4. Term-level RO vs HC comparison detail
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PART 4: TERM-LEVEL RO vs HC DETAIL")
print("=" * 70)

for label, tdf in [("A", ta), ("B", tb)]:
    diff = tdf["RO"] - tdf["HC"]
    print(f"\n[{label}] RO-HC difference distribution:")
    print(f"  RO > HC: {(diff > 0).sum()}/{len(diff)} ({(diff > 0).mean():.1%})")
    print(f"  RO = HC: {(diff == 0).sum()}/{len(diff)} ({(diff == 0).mean():.1%})")
    print(f"  RO < HC: {(diff < 0).sum()}/{len(diff)} ({(diff < 0).mean():.1%})")

    # Terms where RO wins big (RO=1.0, HC=0.0)
    big_win = diff[diff >= 0.67]
    print(f"  RO >> HC (diff >= 0.67): {len(big_win)} terms")
    for idx in big_win.index[:10]:
        print(f"    {idx}: RO={tdf.loc[idx, 'RO']:.2f}, HC={tdf.loc[idx, 'HC']:.2f}")

    # Terms where HC wins (RO < HC)
    hc_win = diff[diff < 0]
    print(f"  HC > RO: {len(hc_win)} terms")
    for idx in hc_win.index[:10]:
        print(f"    {idx}: RO={tdf.loc[idx, 'RO']:.2f}, HC={tdf.loc[idx, 'HC']:.2f}")


# ═══════════════════════════════════════════════════════════════
# 5. Recommendations for logprob collection
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PART 5: LOGPROB COLLECTION PLAN")
print("=" * 70)

# Select 30 terms with max RO-HC difference for logprob sampling
for label, tdf, edf in [("A", ta, exp1_a), ("B", tb, exp1_b)]:
    diff = tdf["RO"] - tdf["HC"]
    # Top 15 where RO > HC, top 15 where RO ≈ HC
    ro_wins = diff[diff > 0].sort_values(ascending=False)
    hc_wins = diff[diff < 0].sort_values()  # most negative = HC wins most
    tie = diff[diff == 0]

    top_ro = ro_wins.head(15).index.tolist()
    top_tie = tie.head(10).index.tolist() if len(tie) >= 10 else tie.index.tolist()
    top_hc = hc_wins.head(5).index.tolist() if len(hc_wins) >= 5 else hc_wins.index.tolist()

    sample_ids = top_ro + top_tie + top_hc
    print(f"\n[{label}] Recommended 30-term sample for logprob collection:")
    print(f"  {len(top_ro)} RO>HC terms, {len(top_tie)} RO≈HC terms, {len(top_hc)} HC>RO terms")
    print(f"  IDs: {sample_ids[:5]}... (total {len(sample_ids)})")

    # Save sample IDs
    out = os.path.join(BASE, "results", "exp1", f"logprob_sample_{label}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"ids": sample_ids, "n_ro_win": len(top_ro),
                    "n_tie": len(top_tie), "n_hc_win": len(top_hc)}, f, ensure_ascii=False)
    print(f"  Saved to {out}")

print("\n" + "=" * 70)
print("DONE. Next: deploy logprob collection script to GPU server.")
print("=" * 70)
