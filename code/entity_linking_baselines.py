#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Entity linking baselines vs Exp1 constraint decoding.

Bi-encoder (BGE-M3) + Cross-encoder (bge-reranker-v2-m3) on the
same C(t) candidate sets used in Exp1.  No LLM, no GPU server —
runs on a single commodity GPU (A800 or even T4) in < 30 min.

Output: results/exp1/entity_linking_baselines.csv + printed report.
"""

import json, os, sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT_DIR = os.path.join(RESULTS_DIR, "exp1")
os.makedirs(OUT_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════
# 1. Load datasets
# ═══════════════════════════════════════════════════════════

def load_records(path):
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    out = []
    for r in raw["test_cases"]:
        if path.endswith("dataset_a_with_candidates.json"):
            rid = r.get("protege_id", "")
        else:
            rid = r.get("record_id", "")
        out.append({
            "id": rid,
            "term": r.get("name", ""),
            "definition": r.get("original_definition", ""),
            "source": r.get("source_book", ""),
            "gold": r.get("modern_definition", ""),
            "candidates": r.get("candidates", []),
        })
    return out


recs_a = load_records(DATASET_A)
recs_b = load_records(DATASET_B)
print(f"Loaded: A={len(recs_a)} terms, B={len(recs_b)} terms")

# Quick sanity: gold present in candidates?
for label, recs in [("A", recs_a), ("B", recs_b)]:
    gold_in = sum(1 for r in recs if r["gold"] in r["candidates"])
    print(f"  [{label}] gold in C(t): {gold_in}/{len(recs)}")


# ═══════════════════════════════════════════════════════════
# 2. Query construction (mirrors Exp1 prompt context)
# ═══════════════════════════════════════════════════════════

def make_query(r, domain):
    """Term-only query — no gold information, matches info available to LLM in HC.

    The LLM in HC sees: system prompt + term + source context + answer prefix,
    but NOT the gold definition.  Using term-only for the baseline is conservative:
    it gives the baseline strictly LESS information than the LLM has.
    """
    return r["term"]


# ═══════════════════════════════════════════════════════════
# 3. Bi-encoder baseline (BGE-M3)
# ═══════════════════════════════════════════════════════════

def run_bi_encoder(recs, domain, model_name="BAAI/bge-m3",
                    method_label="bge-m3"):
    """Dense retrieval: encode query, encode candidates, cosine rank."""
    from sentence_transformers import SentenceTransformer
    import torch

    print(f"\n[{domain}] Loading bi-encoder: {model_name} ...")
    t0 = time.time()
    model = SentenceTransformer(model_name, device="cuda")
    print(f"  Loaded in {time.time()-t0:.1f}s")

    results = []
    n_total = len(recs)

    for i, r in enumerate(recs):
        query = make_query(r, domain)
        candidates = r["candidates"]
        gold = r["gold"]

        # Ensure gold is in candidates (Exp1 injects it if missing)
        if gold not in candidates:
            candidates = candidates + [gold]

        # Encode
        q_emb = model.encode(query, normalize_embeddings=True)
        c_embs = model.encode(candidates, normalize_embeddings=True)

        # Cosine similarity
        scores = np.dot(c_embs, q_emb)  # both L2-normalized → dot = cosine
        ranking = np.argsort(-scores)  # descending
        ranked_candidates = [candidates[idx] for idx in ranking]

        # Gold position
        try:
            gold_rank = ranked_candidates.index(gold) + 1  # 1-indexed
        except ValueError:
            gold_rank = len(candidates) + 1  # not found

        top1_correct = int(ranked_candidates[0] == gold)
        gold_in_top5 = int(gold_rank <= 5)
        rr = 1.0 / gold_rank if gold_rank <= len(candidates) else 0.0

        results.append({
            "id": r["id"],
            "term": r["term"],
            "domain": domain,
            "method": "bge-m3",
            "top1_correct": top1_correct,
            "gold_rank": gold_rank,
            "reciprocal_rank": rr,
            "gold_in_top5": gold_in_top5,
            "top1_pred": ranked_candidates[0],
            "n_candidates": len(candidates),
        })

        if (i + 1) % 50 == 0:
            acc_so_far = np.mean([x["top1_correct"] for x in results])
            print(f"  [{domain}] {i+1}/{n_total} — top-1 acc so far: {acc_so_far:.3f}")

    df = pd.DataFrame(results)
    acc = df["top1_correct"].mean()
    mrr = df["reciprocal_rank"].mean()
    r5 = df["gold_in_top5"].mean()
    print(f"  [{domain}] bi-encoder done: acc={acc:.3f}, MRR={mrr:.3f}, R@5={r5:.3f}")
    return df


# ═══════════════════════════════════════════════════════════
# 4. Cross-encoder baseline (bge-reranker-v2-m3)
# ═══════════════════════════════════════════════════════════

def run_cross_encoder(recs, domain, model_name="BAAI/bge-reranker-v2-m3",
                       method_label="bge-reranker-v2-m3"):
    """Cross-encoder: score each (query, candidate) pair, rank by score.

    Uses sentence_transformers.CrossEncoder (not FlagEmbedding.FlagReranker)
    to avoid the prepare_for_model incompatibility with transformers >= 5.x.
    """
    from sentence_transformers import CrossEncoder

    print(f"\n[{domain}] Loading cross-encoder: {model_name} ...")
    t0 = time.time()
    model = CrossEncoder(model_name, device="cuda")
    print(f"  Loaded in {time.time()-t0:.1f}s")

    results = []
    n_total = len(recs)

    for i, r in enumerate(recs):
        query = make_query(r, domain)
        candidates = r["candidates"]
        gold = r["gold"]

        if gold not in candidates:
            candidates = candidates + [gold]

        # Build (query, candidate) pairs
        pairs = [[query, c] for c in candidates]

        # Score all pairs (raw logits — ranking is invariant to softmax)
        scores = model.predict(pairs, show_progress_bar=False)

        # Handle scalar return for single candidate
        if not hasattr(scores, "__len__"):
            scores = [scores]
        scores = np.array(scores)

        ranking = np.argsort(-scores)
        ranked_candidates = [candidates[idx] for idx in ranking]

        try:
            gold_rank = ranked_candidates.index(gold) + 1
        except ValueError:
            gold_rank = len(candidates) + 1

        top1_correct = int(ranked_candidates[0] == gold)
        gold_in_top5 = int(gold_rank <= 5)
        rr = 1.0 / gold_rank if gold_rank <= len(candidates) else 0.0

        results.append({
            "id": r["id"],
            "term": r["term"],
            "domain": domain,
            "method": method_label,
            "top1_correct": top1_correct,
            "gold_rank": gold_rank,
            "reciprocal_rank": rr,
            "gold_in_top5": gold_in_top5,
            "top1_pred": ranked_candidates[0],
            "n_candidates": len(candidates),
        })

        if (i + 1) % 25 == 0:
            acc_so_far = np.mean([x["top1_correct"] for x in results])
            print(f"  [{domain}] {i+1}/{n_total} — top-1 acc so far: {acc_so_far:.3f}")

    df = pd.DataFrame(results)
    acc = df["top1_correct"].mean()
    mrr = df["reciprocal_rank"].mean()
    r5 = df["gold_in_top5"].mean()
    print(f"  [{domain}] {method_label} done: acc={acc:.3f}, MRR={mrr:.3f}, R@5={r5:.3f}")
    return df


# ═══════════════════════════════════════════════════════════
# 5. Load Exp1 results for comparison
# ═══════════════════════════════════════════════════════════

def load_exp1_term_acc():
    """Term-level accuracy from Exp1 (matching _exp1_analysis.py)."""
    exp1_a = pd.read_csv(os.path.join(BASE, "results", "exp1", "exp1_A.csv"))
    exp1_b = pd.read_csv(os.path.join(BASE, "results", "exp1", "exp1_B.csv"))

    def ta(df):
        return df.groupby(["id", "condition"])["correct"].mean().unstack()

    return ta(exp1_a), ta(exp1_b)


# ═══════════════════════════════════════════════════════════
# 6. McNemar test
# ═══════════════════════════════════════════════════════════

def mcnemar_paired(binary_a, binary_b):
    """McNemar's test on paired binary outcomes."""
    from scipy.stats import chi2 as chisq

    # Build 2x2 table
    both = binary_a & binary_b
    only_a = binary_a & ~binary_b
    only_b = ~binary_a & binary_b
    neither = ~binary_a & ~binary_b

    b = only_a.sum()  # discordant: baseline correct, HC wrong
    c = only_b.sum()  # discordant: baseline wrong, HC correct

    if b + c == 0:
        return 0.0, 1.0

    # With continuity correction
    stat = (abs(b - c) - 1) ** 2 / (b + c)
    p = 1 - chisq.cdf(stat, 1)
    return stat, p


# ═══════════════════════════════════════════════════════════
# 7. Main
# ═══════════════════════════════════════════════════════════

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-bi", action="store_true", help="Skip bi-encoder (BGE-M3, 560M)")
    ap.add_argument("--skip-cross", action="store_true", help="Skip cross-encoder (v2-m3, 560M)")
    ap.add_argument("--skip-cross-base", action="store_true", help="Skip cross-encoder (base, 110M)")
    args = ap.parse_args()

    # ── Model registry ──
    # (flag, model_name, method_label, runner_fn, description)
    models = [
        (not args.skip_bi,
         "BAAI/bge-m3",
         "bge-m3",
         run_bi_encoder,
         "bi-encoder, 560M"),
        (not args.skip_cross,
         "BAAI/bge-reranker-v2-m3",
         "bge-reranker-v2-m3",
         run_cross_encoder,
         "cross-encoder, 560M"),
        (not args.skip_cross_base,
         "BAAI/bge-reranker-base",
         "bge-reranker-base",
         run_cross_encoder,
         "cross-encoder, ~110M"),
    ]

    all_dfs = []

    for domain, recs in [("A", recs_a), ("B", recs_b)]:
        for do_run, model_name, method_label, runner_fn, desc in models:
            if not do_run:
                continue
            print(f"\n  ── [{domain}] {method_label} ({desc}) ──")
            df = runner_fn(recs, domain, model_name, method_label)
            all_dfs.append(df)

    # Save
    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        out_path = os.path.join(OUT_DIR, "entity_linking_baselines.csv")
        combined.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"\nSaved baseline results to {out_path}")

    # ═══════════════════════════════════════════════════════
    # 8. Report: baselines vs Exp1
    # ═══════════════════════════════════════════════════════

    if not all_dfs:
        # Load saved
        combined = pd.read_csv(os.path.join(OUT_DIR, "entity_linking_baselines.csv"))

    ta_a, ta_b = load_exp1_term_acc()

    print("\n" + "=" * 75)
    print("ENTITY LINKING BASELINES vs CONSTRAINT DECODING")
    print("=" * 75)

    print(f"\n{'Method':<25} {'A acc':>8} {'A MRR':>8} {'A R@5':>8}  "
          f"{'B acc':>8} {'B MRR':>8} {'B R@5':>8}")
    print("-" * 75)

    for method in ["bge-m3", "bge-reranker-v2-m3", "bge-reranker-base"]:
        sub_a = combined[(combined["domain"] == "A") & (combined["method"] == method)]
        sub_b = combined[(combined["domain"] == "B") & (combined["method"] == method)]
        if len(sub_a) == 0:
            continue
        print(f"  {method:<23} {sub_a['top1_correct'].mean():8.3f} "
              f"{sub_a['reciprocal_rank'].mean():8.3f} {sub_a['gold_in_top5'].mean():8.3f}  "
              f"{sub_b['top1_correct'].mean():8.3f} "
              f"{sub_b['reciprocal_rank'].mean():8.3f} {sub_b['gold_in_top5'].mean():8.3f}")

    # Exp1 conditions (term-level mean accuracy)
    for cond in ["UC", "HC", "RO", "SC"]:
        a_acc = ta_a[cond].mean()
        b_acc = ta_b[cond].mean()
        # MRR and R@5 not available for Exp1 (guided_choice picks one answer)
        print(f"  {'Exp1 ' + cond:<23} {a_acc:8.3f} {'—':>8} {'—':>8}  "
              f"{b_acc:8.3f} {'—':>8} {'—':>8}")

    # ═══════════════════════════════════════════════════════
    # 9. McNemar tests: baseline vs HC
    # ═══════════════════════════════════════════════════════

    print("\n" + "=" * 75)
    print("McNEMAR TESTS: baseline vs Exp1 HC (per-term binary)")
    print("=" * 75)

    for domain, ta, combined_domain in [("A", ta_a, combined[combined["domain"] == "A"]),
                                         ("B", ta_b, combined[combined["domain"] == "B"])]:
        # HC binary: majority vote (>= 2/3 reps correct)
        hc_binary = (ta["HC"] >= 2.0 / 3.0).astype(int)

        for method in ["bge-m3", "bge-reranker-v2-m3", "bge-reranker-base"]:
            sub = combined_domain[combined_domain["method"] == method]
            if len(sub) == 0:
                continue

            # Merge on id
            merged = pd.DataFrame({"id": hc_binary.index, "hc_correct": hc_binary.values})
            sub_m = sub[["id", "top1_correct"]].merge(merged, on="id")

            if len(sub_m) == 0:
                continue

            baseline_binary = sub_m["top1_correct"].astype(int)
            hc_b = sub_m["hc_correct"].astype(int)

            stat, p = mcnemar_paired(baseline_binary, hc_b)
            baseline_acc = baseline_binary.mean()
            hc_acc = hc_b.mean()

            # Direction
            if baseline_acc > hc_acc and p < 0.05:
                direction = f"baseline BETTER (Δ={baseline_acc-hc_acc:+.3f})"
            elif hc_acc > baseline_acc and p < 0.05:
                direction = f"Exp1 HC BETTER  (Δ={hc_acc-baseline_acc:+.3f})"
            else:
                direction = f"no significant difference"

            print(f"\n  [{domain}] {method} vs HC:")
            print(f"    baseline acc = {baseline_acc:.3f}, HC acc = {hc_acc:.3f}")
            print(f"    McNemar χ² = {stat:.2f}, p = {p:.4f}")
            print(f"    → {direction}")

            # Discordant counts
            only_base = (baseline_binary == 1) & (hc_b == 0)
            only_hc = (baseline_binary == 0) & (hc_b == 1)
            print(f"    baseline✓/HC✗: {only_base.sum()} terms, baseline✗/HC✓: {only_hc.sum()} terms")

    # ═══════════════════════════════════════════════════════
    # 10. Key highlights
    # ═══════════════════════════════════════════════════════
    print("\n" + "=" * 75)
    print("KEY HIGHLIGHTS")
    print("=" * 75)

    for domain, ta in [("A", ta_a), ("B", ta_b)]:
        hc_acc = (ta["HC"] >= 2.0 / 3.0).mean()  # majority-vote binary
        for method in ["bge-m3", "bge-reranker-v2-m3", "bge-reranker-base"]:
            sub = combined[(combined["domain"] == domain) & (combined["method"] == method)]
            if len(sub) == 0:
                continue
            bl_acc = sub["top1_correct"].mean()
            diff = hc_acc - bl_acc
            print(f"  [{domain}] Exp1 HC binary acc = {hc_acc:.3f}, "
                  f"{method} = {bl_acc:.3f}, Δ = {diff:+.3f}")

    # Specific checks the user asked for
    print("\n  --- User-specified checks ---")
    for domain in ["A", "B"]:
        ta = ta_a if domain == "A" else ta_b
        hc_bin_acc = (ta["HC"] >= 2.0 / 3.0).mean()

        for method in ["bge-m3", "bge-reranker-v2-m3", "bge-reranker-base"]:
            sub = combined[(combined["domain"] == domain) & (combined["method"] == method)]
            if len(sub) == 0:
                continue
            bl_acc = sub["top1_correct"].mean()
            delta = hc_bin_acc - bl_acc
            params = "560M" if method != "bge-reranker-base" else "110M"

            if bl_acc >= hc_bin_acc:
                print(f"    ★ [{domain}] {method} ({params}) = {bl_acc:.3f} ≥ Exp1 HC = {hc_bin_acc:.3f} — RETRIEVER BEATS 32B!")
            else:
                print(f"      [{domain}] {method} ({params}) = {bl_acc:.3f}, Exp1 HC = {hc_bin_acc:.3f}, gap = {delta:+.3f}")
                if domain == "A" and delta > 0.1:
                    print(f"        → A domain: LLM leads by {delta:.3f} — knowledge-scarce domain needs LLM")

    # Special: did 110M base model beat 32B+HC?
    print("\n  --- v5.0 押注检查: 110M reranker vs 32B+HC ---")
    for domain in ["A", "B"]:
        sub_base = combined[(combined["domain"] == domain) & (combined["method"] == "bge-reranker-base")]
        ta = ta_a if domain == "A" else ta_b
        hc_bin_acc = (ta["HC"] >= 2.0 / 3.0).mean()
        if len(sub_base) > 0:
            base_acc = sub_base["top1_correct"].mean()
            if base_acc >= hc_bin_acc:
                print(f"    [{domain}] CONFIRMED: 110M reranker ({base_acc:.3f}) ≥ 32B+HC ({hc_bin_acc:.3f}) — ABSTRACT material")
            else:
                print(f"    [{domain}] NOT confirmed: 110M reranker = {base_acc:.3f} < 32B+HC = {hc_bin_acc:.3f}, gap = {hc_bin_acc - base_acc:.3f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
