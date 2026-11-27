# -*- coding: utf-8 -*-
"""
candidate_mass v5.1 —— 核心度量（序列概率 + 多 token + 长度归一 + NFKC）。

❌ 禁止: candidate_mass = sum(softmax(logits))  （v4 错误定义，单 token / 受词表大小污染）
✅ 正确定义:
   gold_share = exp(score(gold)) / Σ_{c∈C(t)} exp(score(c))
   其中 score(c) = (1/|c|) · Σ_i log P(c_i | prompt, c_<i)  (per-token 平均 log 概率)
   norm_mass  = log(gold_share × |C|)  — 相对均匀基线 (1/|C|) 的对数优势

Prompt 格式:
   ✅ RAW prompt（无 chat template）— 采用。
      理由（方法学）: Qwen3 的 chat template 在 assistant 段首注入 <think></think>
      空块，即使 enable_thinking=False 也无法移除。此空块向 prompt 注入无关 context
      token，改变了 teacher-forcing 评分时的条件概率 P(c | prompt_perturbed)，
      与认知测量目标 P(c | task_prompt) 存在系统偏差。RAW prompt 直接呈现任务
      instruction，teacher-forcing 的条件概率仅以任务语义为条件，无格式扰动。
   ❌ Chat template — 不采用。空 <think> 块注入无关 context token，干扰
      teacher-forcing 条件概率。两种格式的 gold rank 分布已留档
      (results/pilot/v51_*_rank.npy)，作为 Exp2 稳健性附录。
"""
import math
import unicodedata
import torch
import torch.nn.functional as F
from config import DEVICE


def _norm(s):
    """NFKC 归一化 + strip，消去 Unicode 等价形式导致的 dict key 不匹配。"""
    return unicodedata.normalize("NFKC", s).strip()


@torch.no_grad()
def score_candidates(model, tok, prompt, candidates, length_norm=True, batch_size=16):
    """教师强制。返回 dict[c] = per-token 平均 log 概率（几何平均，消长度偏置）。"""
    prompt_ids = tok(prompt, return_tensors="pt").input_ids[0]
    out = {}
    for i in range(0, len(candidates), batch_size):
        chunk = candidates[i:i + batch_size]
        seqs, plens, clens = [], [], []
        for c in chunk:
            cid = tok(c, add_special_tokens=False, return_tensors="pt").input_ids[0]
            seqs.append(torch.cat([prompt_ids, cid]))
            plens.append(len(prompt_ids))
            clens.append(len(cid))

        maxlen = max(len(s) for s in seqs)
        pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        batch = torch.full((len(seqs), maxlen), pad, dtype=torch.long)
        attn  = torch.zeros((len(seqs), maxlen), dtype=torch.long)
        for j, s in enumerate(seqs):
            batch[j, :len(s)] = s
            attn[j, :len(s)] = 1

        logits = model(batch.to(DEVICE), attention_mask=attn.to(DEVICE)).logits
        logp = F.log_softmax(logits.float(), dim=-1)

        for j, c in enumerate(chunk):
            total = sum(logp[j, k - 1, batch[j, k]].item() for k in range(plens[j], len(seqs[j])))
            out[_norm(c)] = total / clens[j] if (length_norm and clens[j] > 0) else total

    return out


def candidate_mass_row(scores, gold):
    """
    scores: dict[c] = per-token 平均 log 概率（length-normalized），keys 已 NFKC 归一化。
    返回:
        gold_share   : gold 在 C(t) 内的份额 ∈ [0,1]
        norm_mass    : log(gold_share * |C|) — 相对均匀基线的对数优势
        gold_rank_in_C: gold 在 C(t) 内的排名（1 = 最高）
        argmax_in_C  : C(t) 内得分最高的候选字符串（Exp4 rescore 用）
    """
    gold = _norm(gold)
    cs = list(scores.values())
    m = max(cs)
    lse = m + math.log(sum(math.exp(x - m) for x in cs))
    glp = scores.get(gold)
    gold_share = math.exp(glp - lse) if glp is not None else float("nan")
    n_cand = len(scores)
    norm_mass = math.log(max(gold_share, 1e-12) * n_cand) if not math.isnan(gold_share) else float("nan")
    assert not math.isnan(gold_share), f"gold_share is NaN — gold key mismatch after NFKC. gold={gold!r}  scored_keys={list(scores.keys())[:5]}…"
    ranked = sorted(scores, key=scores.get, reverse=True)
    return {
        "gold_share":     gold_share,
        "norm_mass":      norm_mass,
        "C_mass_lse":     lse,
        "gold_rank_in_C": (ranked.index(gold) + 1) if gold in scores else -1,
        "argmax_in_C":    ranked[0],
        "n_cand":         n_cand,
    }
