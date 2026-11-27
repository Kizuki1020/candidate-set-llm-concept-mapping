#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exp2 v5.1: candidate_mass 全量采集（★核心，先跑）

条件: 仅 UC（无约束解码）
模型: Qwen3-32B bf16, HF transformers（要 logits，非 vLLM）
产出: results/exp2/mass_A.csv, mass_B.csv
      results/exp2/mass_all.csv  (合并)

Prompt: RAW（无 chat template）。
  方法学理由见 candidate_mass.py 头部注释。
  chat-no-think 的 rank 分布已留档 results/pilot/v51_chat_nothink_rank.npy，
  作为 Exp2 稳健性附录。

特性:
  - 断点续跑：检查已有 CSV，跳过已完成 term，每 20 条 flush 到磁盘
  - NaN 硬断言：candidate_mass_row 内 assert，不静默跳过
  - 自动备份：每批完成后 cp 到 /root/results_backup/

用法:
  python exp2_logit_analysis.py --dry-run          # 模拟验证
  python exp2_logit_analysis.py --dataset A        # 仅 A 域 (160 terms)
  python exp2_logit_analysis.py --dataset all      # A+B 全量 (180 terms)
"""
import os, sys, argparse, math, time, shutil, json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *
from utils import load_dataset
from candidate_mass import score_candidates, candidate_mass_row

CHECKPOINT_EVERY = 20  # 每 N 条 flush 一次
BACKUP_DIR = "/root/results_backup"


# ========== GPU ==========

def load_model_hf():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}.get(DTYPE, torch.float16)
    print(f"Loading {MODEL_NAME} ({DTYPE}) …")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=dtype, device_map=DEVICE, trust_remote_code=True)
    model.eval()
    print(f"Model loaded. Vocab: {len(tok)}")
    return model, tok


def compute_mass_for_dataset(records, model, tok, dataset_tag, out_dir):
    """
    逐条计算 candidate_mass，支持断点续跑。
    返回完成的行列表和跳过数。
    """
    csv_path = os.path.join(out_dir, f"mass_{dataset_tag}.csv")
    fields = ["id", "term", "gold", "dataset", "n_cand",
              "gold_share", "norm_mass", "gold_rank_in_C", "argmax_in_C"]

    # ── Resume: 读已有 CSV → 跳过已完成的 term ──
    done_ids = set()
    done_rows = []
    if os.path.exists(csv_path):
        try:
            existing = pd.read_csv(csv_path)
            # Detect old-format CSV (no norm_mass column) → ignore it
            if "norm_mass" not in existing.columns:
                print(f"  [warn] 旧格式 CSV (无 norm_mass 列)，忽略，从头开始")
            else:
                done_ids = set(existing["id"].dropna().astype(str).values)
                done_rows = existing.to_dict("records")
                print(f"  [resume] {csv_path} 已有 {len(done_ids)} 条，跳过")
        except Exception:
            print(f"  [warn] 无法读取已有 CSV，从头开始")

    pending = [(i, r) for i, r in enumerate(records) if str(r.get("id", i)) not in done_ids]
    if not pending:
        print(f"  [{dataset_tag}] 全部 {len(records)} 条已完成，跳过。")
        # Still save backup
        if done_rows:
            _save_csv(done_rows, csv_path, fields)
        return done_rows, len(records)

    print(f"  [{dataset_tag}] 待处理: {len(pending)}/{len(records)}")

    # Process pending records
    rows = done_rows[:]  # start with existing
    nsaved = len(rows)
    for batch_start in range(0, len(pending), CHECKPOINT_EVERY):
        batch = pending[batch_start:batch_start + CHECKPOINT_EVERY]
        for _, r in batch:
            cands = list(dict.fromkeys(r["candidates"]))
            gold = r["gold"]
            if gold not in cands:
                cands.append(gold)

            prompt = build_user_prompt(r["term"], r["source"], r["definition"])
            scores = score_candidates(model, tok, prompt, cands)
            cm = candidate_mass_row(scores, gold)

            rows.append({
                "id": r.get("id", ""),
                "term": r.get("term", ""),
                "gold": gold,
                "dataset": dataset_tag,
                "n_cand": cm["n_cand"],
                "gold_share": cm["gold_share"],
                "norm_mass": cm["norm_mass"],
                "gold_rank_in_C": cm["gold_rank_in_C"],
                "argmax_in_C": cm["argmax_in_C"],
            })

        # Flush to disk
        _save_csv(rows, csv_path, fields)
        n_done = nsaved + batch_start + len(batch)
        print(f"  [{dataset_tag}] {n_done}/{len(records)} flushed to {csv_path}")

        # Backup after each batch
        _backup(csv_path)

    return rows, len(done_ids)


# ========== I/O ==========

def _save_csv(rows, path, fields):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df = pd.DataFrame(rows)
    for f in fields:
        if f not in df.columns:
            df[f] = ""
    df[fields].to_csv(path, index=False, encoding="utf-8-sig")


def _backup(src_path):
    if not os.path.exists(src_path):
        return
    os.makedirs(BACKUP_DIR, exist_ok=True)
    dst = os.path.join(BACKUP_DIR, os.path.basename(src_path))
    shutil.copy2(src_path, dst)


# ========== Report ==========

def print_summary(mass_file):
    df = pd.read_csv(mass_file)
    if len(df) == 0:
        print(f"  (empty file: {mass_file})")
        return
    nm = df["norm_mass"].dropna().values if "norm_mass" in df.columns else np.array([])
    gs = df["gold_share"].dropna().values if "gold_share" in df.columns else np.array([])
    rk = df["gold_rank_in_C"].dropna().values if "gold_rank_in_C" in df.columns else np.array([])

    if len(nm) == 0:
        print(f"  n={len(df)}  norm_mass: NO DATA (wrong CSV version?)")
        return

    n_nan = df["norm_mass"].isna().sum()
    anomaly_count = int((np.abs(nm - nm.mean()) > 4 * nm.std()).sum())

    print(f"\n  ┌─ {os.path.basename(mass_file)} ─┐")
    print(f"  n={len(df)}  NaN={n_nan}  anomalies(>4σ)={anomaly_count}")
    print(f"  gold_share:  mean={gs.mean():.4f}  std={gs.std():.4f}  "
          f"p10={np.percentile(gs,10):.4f}  p50={np.median(gs):.4f}  p90={np.percentile(gs,90):.4f}")
    print(f"  norm_mass:   mean={nm.mean():.3f}  std={nm.std():.3f}  "
          f"p10={np.percentile(nm,10):.3f}  p50={np.median(nm):.3f}  p90={np.percentile(nm,90):.3f}")
    print(f"  gold_rank:   mean={rk.mean():.1f}  median={np.median(rk):.0f}  "
          f"rank=1: {(rk==1).mean()*100:.0f}%  rank≤3: {(rk<=3).mean()*100:.0f}%")

    # Detect problematic rows
    if n_nan > 0:
        nan_rows = df[df["norm_mass"].isna()]
        print(f"\n  ⚠ NaN rows (first 5):")
        for _, row in nan_rows.head(5).iterrows():
            print(f"    id={row['id']}  term={row['term']}  gold={str(row['gold'])[:40]}")

    if anomaly_count > 0:
        extreme = df.iloc[np.argmin(nm)]
        print(f"  ⚠ Min norm_mass: {nm.min():.3f}  term={extreme['term']}  gold={str(extreme['gold'])[:40]}")
        extreme = df.iloc[np.argmax(nm)]
        print(f"  ⚠ Max norm_mass: {nm.max():.3f}  term={extreme['term']}  gold={str(extreme['gold'])[:40]}")


# ========== Main ==========

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--dataset", choices=["A", "B", "all"], default="all")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    out_dir = args.out_dir or os.path.join(RESULTS_DIR, "exp2")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(BACKUP_DIR, exist_ok=True)

    # Load data
    records_a = load_dataset(DATASET_A)
    print(f"Dataset A: {len(records_a)} terms")

    records_b = []
    if args.dataset in ("B", "all") and os.path.exists(DATASET_B):
        # B domain uses different field names — load raw
        with open(DATASET_B, encoding="utf-8") as f:
            raw = json.load(f)
        raw_recs = raw.get("test_cases", raw) if isinstance(raw, dict) else raw
        records_b = []
        for r in raw_recs:
            records_b.append({
                "id": r.get("record_id", ""),
                "term": r.get("name", ""),
                "source": r.get("source_book", ""),
                "definition": r.get("original_definition", ""),
                "gold": r.get("modern_definition", ""),
                "candidates": r.get("candidates", []),
            })
        print(f"Dataset B: {len(records_b)} terms")

    if args.dry_run:
        print("[dry-run] 模拟数据验证…")
        rng = np.random.RandomState(42)
        for tag, recs in [("A", records_a), ("B", records_b)]:
            if not recs:
                continue
            rows = []
            for i, r in enumerate(recs):
                nc = len(r.get("candidates", [])) or 50
                gs = float(np.clip(rng.beta(2, 3) + rng.normal(0, 0.03), 0.001, 0.99))
                nm = math.log(max(gs, 1e-12) * nc)
                rows.append({"id": r.get("id", i), "term": r.get("term", ""),
                             "gold": r.get("gold", ""), "dataset": tag,
                             "n_cand": nc, "gold_share": gs, "norm_mass": nm,
                             "gold_rank_in_C": rng.randint(1, nc + 1),
                             "argmax_in_C": f"mock_{rng.randint(1,nc)}"})
            csv_path = os.path.join(out_dir, f"mass_{tag}.csv")
            fields = ["id", "term", "gold", "dataset", "n_cand",
                      "gold_share", "norm_mass", "gold_rank_in_C", "argmax_in_C"]
            _save_csv(rows, csv_path, fields)
            print_summary(csv_path)
        print("\n[dry-run] Done.")
        return

    # ── GPU run ──
    model, tok = load_model_hf()

    all_csvs = []
    for tag, recs in [("A", records_a), ("B", records_b)]:
        if not recs:
            continue
        print(f"\n{'='*60}")
        print(f"Exp2: {tag} domain ({len(recs)} terms)")
        print(f"{'='*60}")
        rows, n_skipped = compute_mass_for_dataset(recs, model, tok, tag, out_dir)
        csv_path = os.path.join(out_dir, f"mass_{tag}.csv")
        all_csvs.append(csv_path)
        print_summary(csv_path)

    # Merge all
    if len(all_csvs) > 1:
        merged = pd.concat([pd.read_csv(c) for c in all_csvs], ignore_index=True)
        merged_path = os.path.join(out_dir, "mass_all.csv")
        fields = ["id", "term", "gold", "dataset", "n_cand",
                  "gold_share", "norm_mass", "gold_rank_in_C", "argmax_in_C"]
        _save_csv(merged.to_dict("records"), merged_path, fields)
        _backup(merged_path)
        print(f"\nMerged: {merged_path} ({len(merged)} rows)")

    # Final backup of everything in out_dir
    for f in os.listdir(out_dir):
        _backup(os.path.join(out_dir, f))

    print("\n" + "=" * 60)
    print("Exp2 完成。产出:")
    for c in all_csvs:
        print(f"  {c}")
    print(f"  Backup: {BACKUP_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
