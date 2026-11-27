#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_candidates_A.py
=====================
为 Dataset A（160 条古籍数学术语）从 OMD 本体回填 **合法候选集 C(t)**。

这是整条实验链路的关键前置：没有 C(t)，vLLM 的 guided_choice 无法实现，
实验 4 / 2 / 5 的 HC 条件全部跑不了。

────────────────────────────────────────────────────────────────────────
核心科学约束：C(t) 必须是 **gold-blind** 的
────────────────────────────────────────────────────────────────────────
存在一个致命的循环陷阱：

    如果 C(t) 的构造过程以任何方式引用了 gold（标准答案），
    那么 HC 条件的高准确率就是构造出来的，而非约束带来的。

因此本脚本严格遵守：

  1. C(t) 的生成规则**只读取术语侧特征**（name / original_definition /
     source_book / P_Class / L_Class / 本体邻域），**从不读 modern_definition**。
     gold 在生成阶段被物理隔离（见 `TermView`，它不含 gold 字段）。

  2. gold 覆盖率是**事后校验**，不是构造目标。
     若覆盖率 < 100%，脚本**不会**把 gold 单独塞进去（那就是泄漏），
     而是**对所有术语统一放宽召回半径**再整体重算，直到达标。
     单点注入 gold 只在 `--allow-gold-injection` 下允许，且被逐条标记，
     这些条目必须在论文中单独报告或剔除。

  3. 输出 `chance_level = 1/|C(t)|`。guided decoding 下，
     |C(t)|=3 时随机猜就有 33% 准确率。**HC vs UC 的比较必须做 chance 校正**，
     否则"约束带来的增益"里混着"候选空间被裁小"的平凡效应。

  4. 检查 |C(t)| 是否与难度分层相关。若难题的候选集恰好更小，
     则 HC 的增益是混淆的。脚本会给出相关系数并在超标时告警。

────────────────────────────────────────────────────────────────────────
候选集分层召回策略（全部 gold-blind）
────────────────────────────────────────────────────────────────────────
  Tier 1  MSC 同类目：与该术语 HAS_MSC 指向的分类号下的全部现代概念
  Tier 2  本体邻域：沿 SUBSUMES / EVOLUTION / CONCEPT_DERIVED_FROM /
                    SEMANTIC_SHIFT 等关系，从术语节点 k 跳可达的现代概念
  Tier 3  语义近邻：用术语的 name + original_definition 作 query，
                    在**全局现代概念池**中做 embedding 召回，补齐到目标规模

  Tier 1/2 依赖本体结构；若本体缺失（如某术语无 MSC），自动退化到 Tier 3。
  所有 Tier 的输入都不含 gold。

────────────────────────────────────────────────────────────────────────
用法
────────────────────────────────────────────────────────────────────────
  # 正式运行（读真实本体）
  python build_candidates_A.py \
      --test-dataset  data/test_dataset.json \
      --ontology      data/omd_ontology.json \
      --out           data/dataset_a_with_candidates.json \
      --synonyms-out  data/synonyms_A.json \
      --audit-out     data/candidates_A_audit.csv \
      --target-size   50

  # 冒烟测试（内置合成本体，无需任何外部文件、无需网络/GPU）
  python build_candidates_A.py --self-test

依赖
  必需: numpy
  可选: sentence-transformers (Tier 3 语义召回；缺失时自动退化为字符 n-gram 相似度)
        rdflib (直接读 OWL；缺失时需提供 JSON 导出的本体)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import sys
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

# ======================================================================
# 常量
# ======================================================================
RANDOM_SEED = 20260101

# 沿这些关系做本体邻域扩展（Tier 2）
NEIGHBOR_RELATIONS = [
    "SUBSUMES",
    "EVOLUTION",
    "CONCEPT_DERIVED_FROM",
    "SEMANTIC_SHIFT",
    "METHODOLOGICAL_SHIFT",
]
# 这些关系视为同义（用于 synonyms_A.json 与去重）
SYNONYM_RELATIONS = ["EQUIVALENT_TO"]

DEFAULT_TARGET_SIZE = 50      # |C(t)| 目标规模
MIN_SIZE, MAX_SIZE = 20, 100  # 09 号文档规定的合理区间

# 覆盖率不足时，统一放宽的档位（对所有术语一致，保持 gold-blind）
EXPANSION_LADDER = [
    dict(hops=1, target=50),
    dict(hops=2, target=70),
    dict(hops=2, target=90),
    dict(hops=3, target=100),
]

# |C(t)| 与难度的相关性告警阈值
SIZE_DIFFICULTY_CORR_WARN = 0.30


# ======================================================================
# gold 隔离：TermView 物理上不含 modern_definition
# ======================================================================
@dataclass(frozen=True)
class TermView:
    """
    候选集生成器唯一被允许看到的视图。

    刻意不包含 `modern_definition`。任何试图从这里读 gold 的代码都会 AttributeError。
    这是把"gold-blind"从口头约定变成结构保证。
    """
    protege_id: str
    name: str
    source_book: str
    original_definition: str
    P_Class: str
    L_Class: str

    @staticmethod
    def from_case(case: dict) -> "TermView":
        return TermView(
            protege_id=case["protege_id"],
            name=case["name"],
            source_book=case.get("source_book", ""),
            original_definition=case.get("original_definition", ""),
            P_Class=case.get("P_Class", ""),
            L_Class=case.get("L_Class", ""),
        )


# ======================================================================
# 本体
# ======================================================================
@dataclass
class Ontology:
    """
    OMD 本体的最小接口。

    concepts        : 全部现代数学概念（候选池的全集）
    term_to_msc     : 古籍术语 id/name → MSC 分类号列表
    msc_to_concepts : MSC 分类号 → 该类目下的现代概念
    edges           : (src, relation, dst)，src/dst 可为术语或概念
    synonyms        : 概念 → 同义概念集合（来自 EQUIVALENT_TO 闭包）
    """
    concepts: List[str] = field(default_factory=list)
    term_to_msc: Dict[str, List[str]] = field(default_factory=dict)
    msc_to_concepts: Dict[str, List[str]] = field(default_factory=dict)
    edges: List[Tuple[str, str, str]] = field(default_factory=list)
    synonyms: Dict[str, Set[str]] = field(default_factory=dict)
    # 同义组的首选表层形式（dedup key ≠ 输出字符串）
    preferred: Dict[str, str] = field(default_factory=dict)

    _adj: Dict[str, List[str]] = field(default_factory=dict, repr=False)

    def build_index(self) -> None:
        adj = defaultdict(list)
        for s, r, d in self.edges:
            if r in NEIGHBOR_RELATIONS:
                adj[s].append(d)
                adj[d].append(s)          # 无向扩展，捕捉双向语义邻近
        self._adj = dict(adj)

        # EQUIVALENT_TO 的连通分量 → 同义词组
        syn_adj = defaultdict(set)
        for s, r, d in self.edges:
            if r in SYNONYM_RELATIONS:
                syn_adj[s].add(d)
                syn_adj[d].add(s)
        seen, groups = set(), []
        for node in syn_adj:
            if node in seen:
                continue
            comp, stack = set(), [node]
            while stack:
                x = stack.pop()
                if x in comp:
                    continue
                comp.add(x)
                stack.extend(syn_adj[x] - comp)
            seen |= comp
            groups.append(comp)
        for g in groups:
            for c in g:
                self.synonyms[c] = set(g) - {c}

        # 首选表层形式 = 该同义组中在 concepts 列表里出现最早的成员。
        # 本体的 concepts 顺序即 preferred-label 顺序（OWL 中对应 rdfs:label 优先项）。
        # 这不依赖 gold，仍是 gold-blind。
        order = {c: i for i, c in enumerate(self.concepts)}
        BIG = len(order) + 1
        for g in groups:
            best = min(g, key=lambda c: order.get(c, BIG))
            for c in g:
                self.preferred[c] = best

    def neighbors(self, node: str, hops: int = 1) -> Set[str]:
        """从 node 出发 k 跳可达的节点（BFS）。"""
        if node not in self._adj:
            return set()
        out, frontier = set(), deque([(node, 0)])
        visited = {node}
        while frontier:
            cur, d = frontier.popleft()
            if d >= hops:
                continue
            for nb in self._adj.get(cur, []):
                if nb not in visited:
                    visited.add(nb)
                    out.add(nb)
                    frontier.append((nb, d + 1))
        return out

    def canonical(self, concept: str) -> str:
        """
        去重键：把同义概念折叠到唯一代表。
        **仅用于比较与去重，绝不作为输出字符串。**
        """
        group = self.synonyms.get(concept)
        if not group:
            return concept
        return min(group | {concept})

    def surface(self, concept: str) -> str:
        """
        输出串：同义组的首选表层形式。

        为什么必须与 canonical() 分开：guided_choice 会把模型输出**强制**为候选集里的
        字面串。若候选集里放的是 canonical（可能是个冷僻同义词），模型就被迫输出该冷僻词，
        而 exact_match 对 gold 会判错 —— 约束反而制造了假阴性。
        """
        return self.preferred.get(concept, concept)


def load_ontology(path: str) -> Ontology:
    """
    读取本体。优先 JSON 导出格式；.owl/.ttl 走 rdflib（可选依赖）。

    期望的 JSON 结构：
    {
      "concepts": ["线性方程组", "矩形面积", ...],
      "term_to_msc": {"方程": ["15A06"], ...},
      "msc_to_concepts": {"15A06": ["线性方程组", "矩阵", ...], ...},
      "edges": [["方程", "EVOLUTION", "线性方程组"], ...]
    }
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in (".owl", ".ttl", ".rdf", ".xml"):
        return _load_ontology_rdf(path)

    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    onto = Ontology(
        concepts=list(dict.fromkeys(raw.get("concepts", []))),
        term_to_msc={k: list(v) for k, v in raw.get("term_to_msc", {}).items()},
        msc_to_concepts={k: list(v) for k, v in raw.get("msc_to_concepts", {}).items()},
        edges=[tuple(e) for e in raw.get("edges", [])],
    )
    onto.build_index()
    return onto


def _load_ontology_rdf(path: str) -> Ontology:
    try:
        import rdflib  # noqa
    except ImportError:
        sys.exit(
            "读取 OWL 需要 rdflib（pip install rdflib）；"
            "或先把本体导出为 JSON 后用 --ontology xxx.json"
        )
    from rdflib import Graph, RDF, RDFS  # type: ignore

    g = Graph()
    g.parse(path)
    onto = Ontology()
    # 这里给出最小骨架；实际谓词 URI 需按 OMD 本体命名空间调整。
    for s, p, o in g:
        pred = str(p).rsplit("#", 1)[-1].rsplit("/", 1)[-1].upper()
        if pred in NEIGHBOR_RELATIONS + SYNONYM_RELATIONS:
            onto.edges.append((str(s).rsplit("#", 1)[-1], pred, str(o).rsplit("#", 1)[-1]))
        elif pred == "HAS_MSC":
            onto.term_to_msc.setdefault(str(s).rsplit("#", 1)[-1], []).append(
                str(o).rsplit("#", 1)[-1]
            )
    onto.concepts = sorted({d for _, _, d in onto.edges})
    onto.build_index()
    print("[warn] RDF 加载为最小实现，请核对 OMD 本体的谓词命名空间。")
    return onto


# ======================================================================
# Tier 3：语义召回（可选 embedding，缺失时退化为字符 n-gram）
# ======================================================================
class SemanticRetriever:
    def __init__(self, concepts: Sequence[str], model_name: Optional[str] = None):
        self.concepts = list(concepts)
        self.backend = "ngram"
        self._emb = None
        if model_name:
            try:
                from sentence_transformers import SentenceTransformer  # noqa
                self._model = SentenceTransformer(model_name)
                self._emb = self._model.encode(
                    self.concepts, convert_to_numpy=True, normalize_embeddings=True,
                    show_progress_bar=False,
                )
                self.backend = f"embedding:{model_name}"
            except Exception as e:  # 网络/依赖不可用 → 静默退化
                print(f"[warn] embedding 后端不可用（{type(e).__name__}），退化为字符 n-gram 相似度。")

    # ---- 字符 n-gram 后备 ----
    @staticmethod
    def _ngrams(s: str, n: int = 2) -> Set[str]:
        s = re.sub(r"\s+", "", s)
        if len(s) < n:
            return {s} if s else set()
        return {s[i:i + n] for i in range(len(s) - n + 1)}

    def _ngram_sim(self, query: str, cand: str) -> float:
        a, b = self._ngrams(query), self._ngrams(cand)
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)

    def rank(self, query: str, cands: Sequence[str]) -> List[str]:
        """按与 query 的相似度对给定候选降序排序（gold-blind：query 仅含术语侧字段）。
        平局用字典序打破，保证确定性。"""
        cands = list(cands)
        if not cands:
            return []
        if self._emb is not None:
            pos = {c: i for i, c in enumerate(self.concepts)}
            q = self._model.encode([query], convert_to_numpy=True,
                                   normalize_embeddings=True)[0]
            sims = {c: float(self._emb[pos[c]] @ q) if c in pos else 0.0 for c in cands}
        else:
            sims = {c: self._ngram_sim(query, c) for c in cands}
        return sorted(cands, key=lambda c: (-sims[c], c))

    def topk(self, query: str, k: int, exclude: Set[str]) -> List[str]:
        """返回与 query 最相近的 k 个概念。query 只由术语侧字段构成（gold-blind）。"""
        pool = [c for c in self.concepts if c not in exclude]
        if not pool:
            return []
        if self._emb is not None:
            idx = [self.concepts.index(c) for c in pool]
            q = self._model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
            sims = self._emb[idx] @ q
        else:
            sims = np.array([self._ngram_sim(query, c) for c in pool])
        order = np.argsort(-sims)[:k]
        return [pool[i] for i in order]


# ======================================================================
# 候选集构造（核心，gold-blind）
# ======================================================================
def build_candidates_for_term(
    tv: TermView,
    onto: Ontology,
    retriever: SemanticRetriever,
    hops: int,
    target_size: int,
) -> Tuple[List[str], Dict[str, int]]:
    """
    只接受 TermView（不含 gold）。返回 (候选集, 各 Tier 贡献计数)。
    """
    concept_set = set(onto.concepts)
    cands: List[str] = []
    seen: Set[str] = set()
    prov = {"tier1_msc": 0, "tier2_neighbor": 0, "tier3_semantic": 0}

    def add(items, tier):
        for c in items:
            if len(cands) >= MAX_SIZE:      # v2: 达上限即停，prov 与最终列表一致
                return
            if c not in concept_set:
                continue
            canon = onto.canonical(c)      # 去重键
            if canon in seen:
                continue
            seen.add(canon)
            cands.append(onto.surface(c))  # 输出串 = 首选表层形式
            prov[tier] += 1

    # ---- Tier 1: MSC 同类目（排序召回）----
    # v2 修复：原实现按 msc_to_concepts 的存储序（字母序）灌入，桶大于 MAX_SIZE 时
    # 末端截断是武断的，会把语义上最相关的概念随机截掉。
    # 现改为两级排序（全部只用术语侧信息，保持 gold-blind、对所有术语统一）：
    #   1) MSC 多重隶属度：概念命中该术语的 msc 码越多，排位越前；
    #   2) 与 gold-blind query（name + original_definition）的语义相似度；
    #   3) 字典序打破平局（确定性）。
    query = f"{tv.name} {tv.original_definition}".strip()
    msc_codes: List[str] = []
    for key in (tv.protege_id, tv.name):
        msc_codes.extend(onto.term_to_msc.get(key, []))
    msc_codes = list(dict.fromkeys(msc_codes))
    votes: Dict[str, int] = defaultdict(int)
    for msc in msc_codes:
        for c in onto.msc_to_concepts.get(msc, []):
            votes[c] += 1
    if votes:
        by_sim = {c: i for i, c in enumerate(retriever.rank(query, list(votes)))}
        tier1_sorted = sorted(votes, key=lambda c: (-votes[c], by_sim[c]))
        add(tier1_sorted, "tier1_msc")

    # ---- Tier 2: 本体 k 跳邻域中的现代概念 ----
    for key in (tv.protege_id, tv.name):
        nbrs = onto.neighbors(key, hops=hops)
        add([n for n in nbrs if n in concept_set], "tier2_neighbor")

    # ---- Tier 3: 语义近邻补齐 ----
    if len(cands) < target_size:
        # query 与 Tier 1 相同（gold-blind）
        need = target_size - len(cands)
        # seen 存的是 canonical 键，而 retriever 的池是原始概念串 → 需展开为原始串集合
        exclude_raw = {c for c in onto.concepts if onto.canonical(c) in seen}
        add(retriever.topk(query, k=need * 3, exclude=exclude_raw)[:need], "tier3_semantic")

    # 规模钳制：过大则截断（保留 Tier 1/2 优先），过小无法补则如实返回
    if len(cands) > MAX_SIZE:
        cands = cands[:MAX_SIZE]

    return cands, prov


# ======================================================================
# 覆盖率校验与统一放宽
# ======================================================================
def check_coverage(cases: List[dict], cand_map: Dict[str, List[str]],
                   onto: Ontology) -> Tuple[float, List[str]]:
    """事后校验：gold 是否落在 C(t) 内。返回 (覆盖率, 未覆盖的 protege_id)。"""
    misses = []
    for c in cases:
        gold_canon = onto.canonical(c["modern_definition"])
        pool = {onto.canonical(x) for x in cand_map[c["protege_id"]]}
        if gold_canon not in pool:
            misses.append(c["protege_id"])
    covered = len(cases) - len(misses)
    return covered / len(cases), misses


def build_all(cases: List[dict], onto: Ontology, retriever: SemanticRetriever,
              hops: int, target: int) -> Tuple[Dict[str, List[str]], Dict[str, dict]]:
    cand_map, prov_map = {}, {}
    for c in cases:
        tv = TermView.from_case(c)          # ← gold 在此处被物理剥离
        cands, prov = build_candidates_for_term(tv, onto, retriever, hops, target)
        cand_map[tv.protege_id] = cands
        prov_map[tv.protege_id] = prov
    return cand_map, prov_map


# ======================================================================
# 诊断：chance level / 规模-难度混淆
# ======================================================================
def diagnostics(cases: List[dict], cand_map: Dict[str, List[str]]) -> dict:
    sizes = np.array([len(cand_map[c["protege_id"]]) for c in cases], dtype=float)
    chance = 1.0 / np.maximum(sizes, 1)

    # |C(t)| 与难度的相关：L_Class 的序号作难度代理（L0<L1<L2<L3）
    def lrank(c):
        m = re.search(r"(\d)", c.get("L_Class", "") or "")
        return int(m.group(1)) if m else np.nan

    lr = np.array([lrank(c) for c in cases], dtype=float)
    ok = ~np.isnan(lr)
    # 任一侧方差为 0（如所有 |C(t)| 相等）时相关系数无定义 —— 这其实是理想情形（等规模候选集）
    if ok.sum() > 2 and sizes[ok].std() > 0 and lr[ok].std() > 0:
        corr = float(np.corrcoef(sizes[ok], lr[ok])[0, 1])
    else:
        corr = 0.0 if ok.sum() > 2 else float("nan")

    return {
        "n": len(cases),
        "size_mean": float(sizes.mean()),
        "size_min": int(sizes.min()),
        "size_max": int(sizes.max()),
        "size_median": float(np.median(sizes)),
        "chance_mean": float(chance.mean()),
        "chance_max": float(chance.max()),
        "size_vs_difficulty_corr": corr,
        "n_below_min": int((sizes < MIN_SIZE).sum()),
        "n_above_max": int((sizes > MAX_SIZE).sum()),
    }


# ======================================================================
# 输出
# ======================================================================
def write_outputs(cases, cand_map, prov_map, onto, args, coverage, misses, diag):
    # ---- 1. dataset_a_with_candidates.json ----
    out_cases = []
    for c in cases:
        pid = c["protege_id"]
        cands = cand_map[pid]
        rec = dict(c)
        rec["candidates"] = cands
        rec["n_candidates"] = len(cands)
        rec["chance_level"] = round(1.0 / max(len(cands), 1), 6)
        rec["candidate_provenance"] = prov_map[pid]
        rec["gold_in_candidates"] = onto.canonical(c["modern_definition"]) in {
            onto.canonical(x) for x in cands
        }
        rec["gold_injected"] = bool(rec.get("gold_injected", False))
        out_cases.append(rec)

    payload = {
        "dataset": "A_chinese_math_classics",
        "ontology_source": os.path.basename(args.ontology) if args.ontology else "self_test",
        "build_config": {
            "hops": args.hops,
            "target_size": args.target_size,
            "seed": RANDOM_SEED,
            "retriever_backend": args.retriever_backend,
            "gold_blind": True,
            "allow_gold_injection": args.allow_gold_injection,
        },
        "coverage": coverage,
        "diagnostics": diag,
        "test_cases": out_cases,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    # ---- 2. synonyms_A.json ----
    syn = {k: sorted(v) for k, v in onto.synonyms.items() if v}
    with open(args.synonyms_out, "w", encoding="utf-8") as f:
        json.dump(syn, f, ensure_ascii=False, indent=2)

    # ---- 3. candidates_A_audit.csv ----
    fields = ["protege_id", "name", "P_Class", "L_Class", "gold",
              "n_candidates", "chance_level", "gold_in_candidates", "gold_injected",
              "tier1_msc", "tier2_neighbor", "tier3_semantic", "candidates_preview"]
    with open(args.audit_out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for rec in out_cases:
            p = rec["candidate_provenance"]
            w.writerow({
                "protege_id": rec["protege_id"],
                "name": rec["name"],
                "P_Class": rec.get("P_Class", ""),
                "L_Class": rec.get("L_Class", ""),
                "gold": rec["modern_definition"],
                "n_candidates": rec["n_candidates"],
                "chance_level": rec["chance_level"],
                "gold_in_candidates": rec["gold_in_candidates"],
                "gold_injected": rec["gold_injected"],
                "tier1_msc": p["tier1_msc"],
                "tier2_neighbor": p["tier2_neighbor"],
                "tier3_semantic": p["tier3_semantic"],
                "candidates_preview": "|".join(rec["candidates"][:8]),
            })

    # ---- 4. 覆盖率报告 ----
    report = os.path.join(os.path.dirname(args.audit_out) or ".", "candidate_coverage.md")
    with open(report, "w", encoding="utf-8") as f:
        f.write("# Dataset A 候选集覆盖率报告\n\n")
        f.write(f"- 覆盖率: **{coverage:.2%}** ({len(cases)-len(misses)}/{len(cases)})\n")
        f.write(f"- |C(t)| 均值 {diag['size_mean']:.1f}（{diag['size_min']}–{diag['size_max']}）\n")
        f.write(f"- **平均 chance level: {diag['chance_mean']:.2%}**"
                f"（guided decoding 下的随机基线，HC 准确率必须据此校正）\n")
        f.write(f"- |C(t)| 与难度(L_Class)相关: r = {diag['size_vs_difficulty_corr']:.3f}\n\n")
        if misses:
            f.write(f"## 未覆盖术语 ({len(misses)})\n\n")
            for m in misses:
                f.write(f"- {m}\n")
    return report


# ======================================================================
# 自检（合成本体，无外部依赖）
# ======================================================================
def _self_test():
    print("=== self-test: 合成本体冒烟测试 ===")
    concepts = [f"概念{i:02d}" for i in range(60)] + [
        "线性方程组", "矩形面积", "几何角度", "比例", "勾股定理"]
    onto = Ontology(
        concepts=concepts,
        term_to_msc={"方程": ["15A06"], "幂": ["51M25"]},
        msc_to_concepts={
            "15A06": ["线性方程组", "概念01", "概念02", "概念03"],
            "51M25": ["矩形面积", "概念04", "概念05"],
        },
        edges=[
            ("方程", "EVOLUTION", "线性方程组"),
            ("幂", "EVOLUTION", "矩形面积"),
            ("角", "EVOLUTION", "几何角度"),
            ("率", "SEMANTIC_SHIFT", "比例"),
            ("线性方程组", "EQUIVALENT_TO", "一次方程组"),
        ],
    )
    onto.concepts.append("一次方程组")
    onto.build_index()

    cases = [
        dict(protege_id="t1", name="方程", source_book="九章算术",
             original_definition="方程之法", modern_definition="线性方程组",
             P_Class="P1", L_Class="L2"),
        dict(protege_id="t2", name="幂", source_book="九章算术",
             original_definition="广从相乘谓之幂", modern_definition="矩形面积",
             P_Class="P1", L_Class="L2"),
        dict(protege_id="t3", name="角", source_book="测圆海镜",
             original_definition="两线相交", modern_definition="几何角度",
             P_Class="P0", L_Class="L1"),
        dict(protege_id="t4", name="率", source_book="九章算术",
             original_definition="凡数相与者谓之率", modern_definition="比例",
             P_Class="P1", L_Class="L3"),
    ]

    retr = SemanticRetriever(onto.concepts, model_name=None)
    cand_map, prov_map = build_all(cases, onto, retr, hops=2, target=30)

    # 断言 1: gold-blind —— TermView 不含 gold
    tv = TermView.from_case(cases[0])
    assert not hasattr(tv, "modern_definition"), "TermView 泄漏了 gold！"

    # 断言 2: 同一术语两次构造结果一致（确定性）
    c2, _ = build_all(cases, onto, retr, hops=2, target=30)
    assert cand_map == c2, "候选集构造非确定性"

    # 断言 3: 规模合理
    for pid, cl in cand_map.items():
        assert len(cl) > 0, f"{pid} 候选集为空"

    cov, misses = check_coverage(cases, cand_map, onto)
    diag = diagnostics(cases, cand_map)

    print(f"覆盖率: {cov:.0%}  未覆盖: {misses}")
    print(f"|C(t)|: mean={diag['size_mean']:.1f} "
          f"range=[{diag['size_min']},{diag['size_max']}]")
    print(f"chance level 均值: {diag['chance_mean']:.2%}")
    for pid, cl in cand_map.items():
        print(f"  {pid}: n={len(cl):2d}  provenance={prov_map[pid]}  head={cl[:4]}")

    # 断言 4: 同义折叠生效 —— canonical 相同
    assert onto.canonical("线性方程组") == onto.canonical("一次方程组")

    # 断言 5 (回归)：输出串必须是 gold 的表层形式，而非冷僻同义词。
    # 若候选集里放的是 '一次方程组'，guided_choice 会强制模型输出它，
    # exact_match 对 gold '线性方程组' 就会假阴性 —— 约束反而制造错误。
    assert "线性方程组" in cand_map["t1"], \
        f"候选集应含 gold 的首选表层形式，实得: {cand_map['t1'][:5]}"
    assert "一次方程组" not in cand_map["t1"], "冷僻同义词不应作为输出串出现"

    # 断言 6: 无重复
    for pid, cl in cand_map.items():
        assert len(cl) == len(set(cl)), f"{pid} 候选集含重复项"

    # 断言 7: 覆盖率失败路径 —— gold 不在本体概念池时应被检出
    bad_cases = cases + [dict(protege_id="t9", name="虚词", source_book="X",
                              original_definition="", modern_definition="本体外概念",
                              P_Class="P3", L_Class="L3")]
    bad_map, _ = build_all(bad_cases, onto, retr, hops=2, target=30)
    cov_bad, miss_bad = check_coverage(bad_cases, bad_map, onto)
    assert miss_bad == ["t9"], f"应检出未覆盖项 t9，实得 {miss_bad}"
    assert cov_bad < 1.0
    print(f"覆盖率失败路径检出正常: 未覆盖={miss_bad} (cov={cov_bad:.0%})")

    print("\n[OK] self-test 全部通过。")
    print("     gold-blind / 确定性 / 同义折叠 / 首选表层形式 / 覆盖率失败检出 均正常。")
    return 0


# ======================================================================
# main
# ======================================================================
def main():
    ap = argparse.ArgumentParser(description="为 Dataset A 构建 gold-blind 候选集 C(t)")
    ap.add_argument("--test-dataset", default="data/test_dataset.json")
    ap.add_argument("--ontology", default="data/omd_ontology.json")
    ap.add_argument("--out", default="data/dataset_a_with_candidates.json")
    ap.add_argument("--synonyms-out", default="data/synonyms_A.json")
    ap.add_argument("--audit-out", default="data/candidates_A_audit.csv")
    ap.add_argument("--target-size", type=int, default=DEFAULT_TARGET_SIZE)
    ap.add_argument("--hops", type=int, default=1)
    ap.add_argument("--embed-model", default=None,
                    help="如 shibing624/text2vec-base-chinese；缺省用字符 n-gram")
    ap.add_argument("--min-coverage", type=float, default=1.0,
                    help="要求的 gold 覆盖率，默认 100%%")
    ap.add_argument("--allow-gold-injection", action="store_true",
                    help="【危险】允许对未覆盖项单独注入 gold。会被逐条标记，"
                         "这些条目在论文中必须单独报告或剔除。")
    ap.add_argument("--verify-gold-blind", action="store_true",
                    help="置换检验：打乱全部 gold 后重建候选集，验证构造与 gold 无关，"
                         "并报告偶然命中率。建议在论文 §Method 中报告该数值。")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    args.perm_coverage = None

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    if args.self_test:
        sys.exit(_self_test())

    # ---- 加载 ----
    with open(args.test_dataset, "r", encoding="utf-8") as f:
        ds = json.load(f)
    cases = ds["test_cases"] if isinstance(ds, dict) else ds
    print(f"载入 {len(cases)} 条测试术语")

    onto = load_ontology(args.ontology)
    print(f"本体：{len(onto.concepts)} 个现代概念，{len(onto.edges)} 条关系边")

    retriever = SemanticRetriever(onto.concepts, args.embed_model)
    args.retriever_backend = retriever.backend
    print(f"语义召回后端：{retriever.backend}")

    # ---- 构造 + 统一放宽阶梯 ----
    hops, target = args.hops, args.target_size
    cand_map = prov_map = None
    coverage, misses = 0.0, []

    ladder = [dict(hops=hops, target=target)] + EXPANSION_LADDER
    for step, cfg in enumerate(ladder):
        cand_map, prov_map = build_all(cases, onto, retriever, cfg["hops"], cfg["target"])
        coverage, misses = check_coverage(cases, cand_map, onto)
        print(f"[阶梯 {step}] hops={cfg['hops']} target={cfg['target']} "
              f"→ 覆盖率 {coverage:.2%}（缺 {len(misses)}）")
        hops, target = cfg["hops"], cfg["target"]   # 始终同步到本轮配置
        if coverage >= args.min_coverage:
            break

    args.hops, args.target_size = hops, target
    # 置换检验必须对比"注入前"的候选集（注入本身当然依赖 gold，但它被逐条标记）
    pre_injection_map = {k: list(v) for k, v in cand_map.items()}

    # ---- 覆盖率仍不足 ----
    # v2：注入与阶梯解耦。阶梯负责"统一放宽"（gold-blind），
    # 达到 --min-coverage 即停；结构性残缺再由注入兜底（逐条标记）。
    if misses and args.allow_gold_injection:
        print(f"[warn] 对 {len(misses)} 条注入 gold（已标记 gold_injected=True）")
        miss_set = set(misses)
        for c in cases:
            if c["protege_id"] in miss_set:
                cand_map[c["protege_id"]].append(c["modern_definition"])
                c["gold_injected"] = True
        coverage, misses = check_coverage(cases, cand_map, onto)
    elif coverage < args.min_coverage:
        if not args.allow_gold_injection:
            print("\n" + "!" * 68)
            print(f"覆盖率 {coverage:.2%} < 要求 {args.min_coverage:.0%}，且阶梯已用尽。")
            print("未覆盖的术语，其 HC 条件的准确率上限为 0 —— 这会人为压低 HC。")
            print("\n可选处置（按推荐度排序）：")
            print("  1. 检查这些 gold 是否根本不在本体的 concepts 池中 → 补本体（治本）")
            print("  2. 提高 --target-size / --hops（对所有术语一致，保持 gold-blind）")
            print("  3. 换用 --embed-model 提升 Tier 3 召回质量")
            print("  4. 【下策】--allow-gold-injection：单点注入 gold，属于泄漏，")
            print("     这些条目必须在论文中单独报告或从 HC 分析中剔除")
            print("\n未覆盖列表：", misses)
            print("!" * 68)
            sys.exit(2)

    # ---- gold-blindness 置换检验（可选，但强烈建议在论文中报告）----
    if args.verify_gold_blind:
        import copy as _copy
        perm_cases = _copy.deepcopy(cases)
        golds = [c["modern_definition"] for c in perm_cases]
        rng = random.Random(RANDOM_SEED)
        rng.shuffle(golds)
        for c, g in zip(perm_cases, golds):
            c["modern_definition"] = g

        perm_map, _ = build_all(perm_cases, onto, retriever, hops, target)
        differing = [pid for pid in pre_injection_map
                     if pre_injection_map[pid] != perm_map[pid]]
        perm_cov, _ = check_coverage(perm_cases, perm_map, onto)

        print("\n===== gold-blindness 置换检验 =====")
        print(f"置换全部 gold 后，候选集有差异的术语数: {len(differing)} / {len(cases)}")
        if differing:
            print("  ❌ 泄漏：候选集构造依赖 gold，HC 的准确率是循环论证的产物。")
            sys.exit(3)
        print("  ✅ 候选集逐字节一致 → 构造过程与 gold 无关")
        print(f"真实 gold 覆盖率   : {coverage:.1%}")
        print(f"置换 gold 覆盖率   : {perm_cov:.1%}  ← 偶然命中率（候选集过大时会升高）")
        print(f"结构性增益         : {coverage - perm_cov:+.1%}")
        if perm_cov > 0.5:
            print("  ⚠ 偶然命中率过高：候选集大到几乎装下一切，")
            print("    coverage=100% 无法证明本体召回有效。应缩小 --target-size。")
        else:
            print("  ✅ 覆盖率来自本体结构，而非候选集过大")
        args.perm_coverage = perm_cov

    # ---- 诊断 ----
    diag = diagnostics(cases, cand_map)
    if getattr(args, "perm_coverage", None) is not None:
        diag["permuted_gold_coverage"] = args.perm_coverage
        diag["structural_lift"] = coverage - args.perm_coverage
    print("\n===== 诊断 =====")
    print(f"覆盖率            : {coverage:.2%}")
    print(f"|C(t)| 均值/中位  : {diag['size_mean']:.1f} / {diag['size_median']:.1f}")
    print(f"|C(t)| 范围       : [{diag['size_min']}, {diag['size_max']}]")
    print(f"平均 chance level : {diag['chance_mean']:.2%}  ← HC 的随机基线")
    print(f"最高 chance level : {diag['chance_max']:.2%}")
    print(f"|C(t)| vs 难度 r  : {diag['size_vs_difficulty_corr']:.3f}")

    if diag["n_below_min"] or diag["n_above_max"]:
        print(f"[warn] {diag['n_below_min']} 条 |C(t)|<{MIN_SIZE}，"
              f"{diag['n_above_max']} 条 >{MAX_SIZE}")
    if abs(diag["size_vs_difficulty_corr"]) > SIZE_DIFFICULTY_CORR_WARN:
        print(f"[warn] |C(t)| 与难度显著相关（|r|>{SIZE_DIFFICULTY_CORR_WARN}）：")
        print("       难题若恰好候选集更小，HC 的增益就是混淆的。")
        print("       建议把 |C(t)| 固定为常数（等规模候选集），或在 GLMM 中把 log|C(t)| 作协变量。")
    if diag["chance_mean"] > 0.05:
        print(f"[warn] 平均随机基线 {diag['chance_mean']:.1%} 偏高。")
        print("       HC vs UC 的比较必须做 chance 校正，否则增益里混着'候选空间被裁小'的平凡效应。")

    # ---- 输出 ----
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    report = write_outputs(cases, cand_map, prov_map, onto, args, coverage, misses, diag)
    print(f"\n已写出：\n  {args.out}\n  {args.synonyms_out}\n  {args.audit_out}\n  {report}")
    print("\n下一步：python exp4_constraint_timing_vllm.py "
          f"--dataset {args.out} --tag A")


if __name__ == "__main__":
    main()
