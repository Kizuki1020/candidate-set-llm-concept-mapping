#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Exp1: Four-condition constraint decoding (UC/RO/SC/HC) via vLLM + xgrammar.

Conditions:
  UC - unconstrained free generation
  HC - guided_choice = C(t) (correct candidates, includes gold)
  RO - gold + (|C|-1) irrelevant concepts (other-domain golds)
  SC - gold + (|C|-1) other-term candidates (same-domain, shuffled)

Produces: results/exp1/exp1_A.csv, results/exp1/exp1_B.csv
"""
import argparse, json, os, sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *
from utils import load_synonyms, exact_match, parse_answer

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
OUT_DIR = os.path.join(RESULTS_DIR, "exp1")
os.makedirs(OUT_DIR, exist_ok=True)


# ============================================================
# 1. Data loading
# ============================================================

def load_dataset_a():
    with open(DATASET_A, encoding="utf-8") as f:
        raw = json.load(f)
    return [{
        "id": r.get("protege_id", ""),
        "term": r.get("name", ""),
        "source": r.get("source_book", ""),
        "definition": r.get("original_definition", ""),
        "gold": r.get("modern_definition", ""),
        "candidates": r.get("candidates", []),
    } for r in raw["test_cases"]]


def load_dataset_b():
    with open(DATASET_B, encoding="utf-8") as f:
        raw = json.load(f)
    return [{
        "id": r.get("record_id", ""),
        "term": r.get("name", ""),
        "source": r.get("source_book", ""),
        "definition": r.get("original_definition", ""),
        "gold": r.get("modern_definition", ""),
        "candidates": r.get("candidates", []),
        "mapping_type": r.get("mapping_type", ""),
    } for r in raw["test_cases"]]


# ============================================================
# 2. RO / SC pool construction
# ============================================================

def build_pools(records_a, records_b):
    """RO: other-domain golds. SC: all other-term candidates from same domain."""
    a_golds = sorted(set(r["gold"] for r in records_a))
    b_golds = sorted(set(r["gold"] for r in records_b))

    def all_cands(records):
        xs = []
        for r in records:
            xs.extend(r.get("candidates", []))
        return sorted(set(xs))

    sc_a = all_cands(records_a)
    sc_b = all_cands(records_b)

    print(f"RO pools: A<={len(b_golds)} B-golds, B<={len(a_golds)} A-golds")
    print(f"SC pools: A={len(sc_a)}, B={len(sc_b)}")
    return a_golds, b_golds, sc_a, sc_b


def build_sets(record, k, ro_pool, sc_pool, rng):
    """RO: gold + (k-1) irrelevant concepts. SC: gold + (k-1) other-term candidates."""
    gold = record["gold"]
    own = set(record.get("candidates", []))

    ro_valid = [c for c in ro_pool if c != gold and c not in own]
    n_ro = min(k - 1, len(ro_valid))
    ro = [gold] + (rng.choice(ro_valid, n_ro, replace=False).tolist() if n_ro > 0 else [])
    rng.shuffle(ro)

    sc_valid = [c for c in sc_pool if c != gold and c not in own]
    n_sc = min(k - 1, len(sc_valid))
    sc = [gold] + (rng.choice(sc_valid, n_sc, replace=False).tolist() if n_sc > 0 else [])
    rng.shuffle(sc)

    return ro, sc


# ============================================================
# 3. Prompt
# ============================================================

def make_prompt(record, domain, tok):
    if domain == "A":
        system = SYSTEM_PROMPT
        user = build_user_prompt(record["term"], record.get("source", ""),
                                 record.get("definition", ""))
        prefix = ANSWER_PREFIX
    else:
        system = SYSTEM_PROMPT_B
        user = build_user_prompt_b(record["term"], record.get("definition", ""))
        prefix = ANSWER_PREFIX_B

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    try:
        text = tok.apply_chat_template(
            messages, tokenize=False, enable_thinking=False,
            add_generation_prompt=True)
    except TypeError:
        text = tok.apply_chat_template(messages, tokenize=False,
                                       add_generation_prompt=True)
    return text, prefix


# ============================================================
# 4. Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", choices=["A", "B", "both"], default="both")
    ap.add_argument("--n-repeat", type=int, default=N_REPEATS)
    ap.add_argument("--batch-size", type=int, default=40)
    ap.add_argument("--dry-run", type=int, default=0)
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    records_a = load_dataset_a()
    records_b = load_dataset_b()
    synonyms_a = load_synonyms(SYNONYMS_A)
    synonyms_b = {}
    print(f"Data: A={len(records_a)} terms, B={len(records_b)} terms")

    a_golds, b_golds, sc_a, sc_b = build_pools(records_a, records_b)

    # ---- Load vLLM ----
    print(f"\nLoading vLLM: {MODEL_NAME} ({DTYPE}) ...")
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    llm = LLM(
        model=MODEL_NAME,
        dtype=DTYPE,
        trust_remote_code=True,
        gpu_memory_utilization=0.85,
        max_model_len=4096,
    )
    tok = llm.get_tokenizer()
    print("vLLM ready.\n")

    for domain in (["A", "B"] if args.domain == "both" else [args.domain]):
        records = records_a if domain == "A" else records_b
        synonyms = synonyms_a if domain == "A" else synonyms_b
        ro_pool = b_golds if domain == "A" else a_golds
        sc_pool = sc_a if domain == "A" else sc_b
        prefix_val = ANSWER_PREFIX if domain == "A" else ANSWER_PREFIX_B

        run_domain(domain, records, llm, tok, synonyms,
                   ro_pool, sc_pool, prefix_val,
                   args.n_repeat, args.batch_size,
                   args.dry_run, args.no_resume)


# ============================================================
# 5. Per-domain runner
# ============================================================

def run_domain(domain, records, llm, tok, synonyms, ro_pool, sc_pool,
               prefix_val, n_repeat, batch_size, dry_run, no_resume):
    from vllm import SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    csv_path = os.path.join(OUT_DIR, f"exp1_{domain}.csv")

    # Resume
    done_ids = set()
    if (not no_resume) and os.path.exists(csv_path):
        existing = pd.read_csv(csv_path)
        done_ids = set(existing["id"].unique())
        print(f"[{domain}] Resume: {len(done_ids)} terms done ({len(existing)} rows)")

    to_run = [r for r in records if r["id"] not in done_ids]
    if dry_run > 0:
        to_run = to_run[:dry_run]
    n = len(to_run)
    total_req = n * n_repeat * 4
    print(f"[{domain}] {n} terms x {n_repeat} reps x 4 conds = {total_req} requests")

    master_rng = np.random.RandomState(SEED_BASE)
    think_checked = False

    for b0 in range(0, n, batch_size):
        batch = to_run[b0:b0 + batch_size]
        prompts, param_list, meta = [], [], []

        for r in batch:
            k = len(r.get("candidates", []))
            ro_set, sc_set = build_sets(r, k, ro_pool, sc_pool, master_rng)
            prompt_str, _ = make_prompt(r, domain, tok)

            for rep in range(n_repeat):
                rep_rng = np.random.RandomState(
                    SEED_BASE + rep * 10000 + hash(r["id"]) % 9999)

                hc = list(dict.fromkeys(r.get("candidates", [])))
                gold = r["gold"]
                if gold not in hc:
                    hc.append(gold)
                rep_rng.shuffle(hc)

                ro = ro_set[:]
                rep_rng.shuffle(ro)

                sc = sc_set[:]
                rep_rng.shuffle(sc)

                for cond, choices in [("UC", None), ("RO", ro),
                                      ("SC", sc), ("HC", hc)]:
                    prompts.append(prompt_str)
                    gd = StructuredOutputsParams(choice=choices) if choices else None
                    param_list.append(SamplingParams(
                        temperature=0.0, max_tokens=MAX_TOKENS,
                        structured_outputs=gd,
                        stop=["\n", "<|im_end|>"],
                        extra_args={"enable_thinking": False},
                    ))
                    meta.append((r["id"], r["term"], rep, cond, gold))

        # Generate batch
        t0 = time.time()
        outputs = llm.generate(prompts, param_list, use_tqdm=False)
        dt = time.time() - t0

        # Parse outputs
        rows = []
        for i, out in enumerate(outputs):
            rid, term, rep, cond, gold = meta[i]
            raw = out.outputs[0].text if out.outputs else ""
            pred = parse_answer(raw, prefix_val)
            correct = exact_match(pred, gold, synonyms)
            rows.append({
                "id": rid, "term": term, "rep": rep, "condition": cond,
                "pred": pred, "gold": gold, "correct": int(correct),
                "raw_output": raw,
            })

        # Think-block check on very first batch
        if not think_checked:
            think_checked = True
            n_think = sum(
                1 for rw in rows if "think" in (rw["raw_output"] or "").lower())
            pct = n_think / len(rows) * 100
            print(f"[{domain}] Think-block check: {n_think}/{len(rows)} = {pct:.1f}%")
            if n_think > 0:
                print("  WARNING: think blocks detected! Sample raw outputs:")
                for rw in rows[:3]:
                    print(f"  [{rw['condition']}] {rw['raw_output'][:200]}")
            else:
                print("  OK: no think blocks detected.")

        # Save incrementally
        mode = "a" if os.path.exists(csv_path) else "w"
        pd.DataFrame(rows).to_csv(csv_path, mode=mode, index=False,
                                  encoding="utf-8-sig", header=(mode == "w"))

        # Quick stats
        uc_ok = sum(1 for rw in rows if rw["condition"] == "UC" and rw["correct"])
        hc_ok = sum(1 for rw in rows if rw["condition"] == "HC" and rw["correct"])
        per_cond = len(batch) * n_repeat
        batch_num = b0 // batch_size + 1
        total_batches = (n + batch_size - 1) // batch_size
        print(f"[{domain}] batch {batch_num}/{total_batches}: {dt:.1f}s | "
              f"UC={uc_ok}/{per_cond} ({uc_ok/per_cond:.1%}) | "
              f"HC={hc_ok}/{per_cond} ({hc_ok/per_cond:.1%})")

    print(f"[{domain}] DONE -> {csv_path}")


if __name__ == "__main__":
    main()
