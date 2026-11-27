# -*- coding: utf-8 -*-
"""统一指标 v5.0 —— 所有实验脚本 `from metrics import *`。论文改指标只改这里。"""
import math
import numpy as np
from config import BOOTSTRAP_N, BOOTSTRAP_SEED, CI_LEVEL


# ========== Core metrics ==========

def gold_share(scores, gold):
    """gold 在 C(t) 内的软最大份额 ∈ [0,1]。scores = {candidate: log_prob}。"""
    cs = list(scores.values())
    m = max(cs)
    lse = m + math.log(sum(math.exp(x - m) for x in cs))
    glp = scores.get(gold)
    return math.exp(glp - lse) if glp is not None else float("nan")


def candidate_entropy(scores):
    """C(t) 内 softmax 分布的熵（nat）。高→模型不确定；低→模型确信。"""
    cs = list(scores.values())
    m = max(cs)
    probs = [math.exp(x - m) for x in cs]
    z = sum(probs)
    probs = [p / z for p in probs]
    return -sum(p * math.log(p) for p in probs if p > 0)


def gold_rank(scores, gold):
    """gold 在 C(t) 内的排名（1 = 最高分）。"""
    ranked = sorted(scores, key=scores.get, reverse=True)
    return (ranked.index(gold) + 1) if gold in scores else -1


# ========== Accuracy ==========

def constraint_gain(acc_hc, acc_uc):
    """约束增益 = HC − UC。行为层指标。"""
    return acc_hc - acc_uc


# ========== Bootstrap (term-level) ==========

def bootstrap_ci(values, n=BOOTSTRAP_N, seed=BOOTSTRAP_SEED, ci=CI_LEVEL):
    """术语级重采样 bootstrap 均值 CI。values: array-like。"""
    arr = np.asarray(values, dtype=float)
    rng = np.random.RandomState(seed)
    means = [arr[rng.randint(0, len(arr), len(arr))].mean() for _ in range(n)]
    lo = (1 - ci) / 2 * 100
    hi = 100 - lo
    return float(arr.mean()), float(np.percentile(means, lo)), float(np.percentile(means, hi))


def bootstrap_diff_paired(a, b, n=BOOTSTRAP_N, seed=BOOTSTRAP_SEED, ci=CI_LEVEL):
    """配对 bootstrap：对每对差值重采样。a,b: 等长 array-like。"""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, len(d), size=(n, len(d)))
    boots = d[idx].mean(axis=1)
    lo = (1 - ci) / 2 * 100
    hi = 100 - lo
    return float(d.mean()), float(np.percentile(boots, lo)), float(np.percentile(boots, hi))


def cliffs_delta(a, b):
    """Cliff's δ 效应量。a,b: array-like。∈ [-1,1]，|δ|>0.33=中，|δ|>0.47=大。"""
    a, b = np.asarray(a), np.asarray(b)
    na, nb = len(a), len(b)
    gt = sum((ai > bj) for ai in a for bj in b)
    lt = sum((ai < bj) for ai in a for bj in b)
    return (gt - lt) / (na * nb)


def pearson_r_ci(x, y, n=BOOTSTRAP_N, seed=BOOTSTRAP_SEED):
    """Pearson r + bootstrap CI。"""
    x, y = np.asarray(x, float), np.asarray(y, float)
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    r = float(np.corrcoef(x, y)[0, 1])
    rng = np.random.RandomState(seed)
    n_ok = len(x)
    rs = [float(np.corrcoef(x[rng.randint(0, n_ok, n_ok)],
                              y[rng.randint(0, n_ok, n_ok)])[0, 1])
          for _ in range(n)]
    lo, hi = np.percentile(rs, [2.5, 97.5])
    return r, lo, hi


# ========== Diagnostic ==========

def gold_share_to_norm_mass(gold_share, n_cand):
    """norm_mass = log(gold_share * |C|)。相对均匀基线 (1/|C|) 的对数优势。"""
    import math
    gs = max(gold_share, 1e-12)
    return math.log(gs * n_cand)


def mass_distribution_stats(masses):
    """
    mass 分布的动态范围统计。masses 可以是 gold_share 或 norm_mass。
    当输入为 gold_share 时，frac_low/high 用 [0,1] 阈值；
    当输入为 norm_mass 时，frac_low/high 用 log 尺度阈值（<0 = 低于均匀基线，>log(2)≈0.69 = 2×均匀）。
    """
    arr = np.asarray(masses, float)
    arr = arr[~np.isnan(arr)]
    p10, p50, p90 = np.percentile(arr, [10, 50, 90])
    # 自动检测量纲：norm_mass 有正有负，gold_share ∈ [0,1]
    is_norm = (arr.min() < -0.5) or (arr.max() > 2.0)
    if is_norm:
        low_thresh, high_thresh = 0.0, math.log(2)  # 0.693
    else:
        low_thresh, high_thresh = 0.2, 0.5
    return {
        "n": len(arr),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "p10": float(p10),
        "p50": float(p50),
        "p90": float(p90),
        "spread_p90_p10": float(p90 - p10),
        "frac_low": float((arr < low_thresh).mean()),
        "frac_high": float((arr > high_thresh).mean()),
        "is_norm_mass": is_norm,
        "low_thresh": low_thresh,
        "high_thresh": high_thresh,
    }
