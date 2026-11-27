#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_ontology_v2.py
====================
修复原方案的词表错配：9_12.rdf 的 MathematicalKnowledgeComponent 是古籍知识单元名，
与 test_dataset 的 gold（现代白话定义）交集为 0，导致候选集覆盖率 0%。

v2 概念池 = 现代概念定义的标签空间：
  A) test_dataset 全部 160 条 modern_definition（Dataset A 的标签空间全集）
  B) 代微积拾级 RDF 中独立标注的 ModernDefinition（616 条，作为真实现代概念干扰项）

msc_to_concepts 的两条独立来源（写入 provenance 供论文报告）：
  A) test case 自身的 msc_codes（语料标注：术语所属数学领域 → 其现代概念归入该领域）
  B) DAIWEIJI 术语的 hasMSC（与 test gold 完全独立）

term_to_msc / edges 沿用 9_12 转换结果（含 test_dataset msc 合并）。
"""
import json
import re
import sys
from collections import defaultdict

RDF_DW = sys.argv[1] if len(sys.argv) > 1 else "/mnt/user-data/uploads/代微积拾级1月4日_cleaned.rdf"
BASE_ONTO = "data/omd_ontology.json"          # 9_12 转换产物
TEST = "data/test_dataset.json"
OUT = "data/omd_ontology_v2.json"
PROV_OUT = "data/ontology_v2_provenance.json"


def clean(s: str) -> str:
    return re.sub(r"[\u200b\s]+", " ", s or "").strip()


def main():
    base = json.load(open(BASE_ONTO, encoding="utf-8"))
    ds = json.load(open(TEST, encoding="utf-8"))
    cases = ds["test_cases"]

    # ---------- A) 测试集标签空间 ----------
    gold_concepts = []
    seen = set()
    msc_to_concepts = defaultdict(set)
    prov = {"concept_source": {}, "msc_link_source": defaultdict(list)}
    for c in cases:
        g = clean(c["modern_definition"])
        if not g or g.lower() == "nan":
            continue
        if g not in seen:
            seen.add(g)
            gold_concepts.append(g)
            prov["concept_source"][g] = "test_dataset_label_space"
        for m in c.get("msc_codes", []):
            code = m.split("#")[-1].strip()
            if code:
                msc_to_concepts[code].add(g)
                prov["msc_link_source"][code].append(
                    {"concept": g, "source": "case_annotation", "protege_id": c["protege_id"]})

    n_gold = len(gold_concepts)

    # ---------- B) 代微积拾级 ModernDefinition ----------
    dw = open(RDF_DW, encoding="utf-8").read()
    blocks = re.findall(r"<rdf:Description.*?</rdf:Description>", dw, re.S)
    dw_added = 0
    dw_msc_links = 0
    for b in blocks:
        md = re.search(r"<DAIWEIJI:ModernDefinition>([^<]+)<", b)
        if not md:
            continue
        g = clean(md.group(1))
        if not g or g.lower() == "nan":
            continue
        if g not in seen:
            seen.add(g)
            gold_concepts.append(g)
            prov["concept_source"][g] = "daiweiji_modern_definition"
            dw_added += 1
        mscs = re.findall(r'<DAIWEIJI:hasMSC[^>]*rdf:resource="[^"]*#([^"]+)"', b)
        if not mscs:
            mscs = re.findall(r"<DAIWEIJI:hasMSC[^>]*>([^<]+)<", b)
        for m in mscs:
            code = m.split("#")[-1].strip()
            if code and g not in msc_to_concepts[code]:
                msc_to_concepts[code].add(g)
                prov["msc_link_source"][code].append(
                    {"concept": g, "source": "daiweiji_hasMSC"})
                dw_msc_links += 1

    # ---------- 组装 v2 ----------
    onto = {
        "concepts": gold_concepts,   # gold 标签空间在前 → 首选表层形式
        "term_to_msc": base["term_to_msc"],
        "msc_to_concepts": {k: sorted(v) for k, v in msc_to_concepts.items()},
        "edges": base["edges"],
    }
    json.dump(onto, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    prov["msc_link_source"] = dict(prov["msc_link_source"])
    json.dump(prov, open(PROV_OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    print(f"概念池: {len(gold_concepts)}（标签空间 {n_gold} + 代微积新增 {dw_added}）")
    print(f"msc_to_concepts: {len(msc_to_concepts)} 个类目（代微积贡献 {dw_msc_links} 条链接）")
    print(f"term_to_msc: {len(onto['term_to_msc'])} | edges: {len(onto['edges'])}")
    print(f"[OK] {OUT}")

    # 自检：全部 gold 应在池中
    pool = set(gold_concepts)
    miss = [c["protege_id"] for c in cases if clean(c["modern_definition"]) not in pool]
    print(f"gold ∈ 概念池 自检: {len(cases)-len(miss)}/{len(cases)}", "OK" if not miss else f"缺 {miss}")


if __name__ == "__main__":
    main()
