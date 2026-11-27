# -*- coding: utf-8 -*-
"""共享工具 v5.0 —— 数据加载 / 匹配 / 解析。所有脚本 `from utils import *`。"""
import json
import os
import csv

from config import FIELD


def load_dataset(path):
    """读取数据集，把真实字段名映射到内部标准名。返回 list[dict]。"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"数据集不存在: {path}")
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    records = raw["test_cases"] if isinstance(raw, dict) and "test_cases" in raw else raw
    out = []
    for r in records:
        out.append({k: r.get(src) for k, src in FIELD.items()})
    return out


def load_synonyms(path):
    """读取同义词表 {标准概念: [同义词列表]}。文件不存在返回空。"""
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def exact_match(pred, gold, synonyms=None):
    """精确匹配 + 同义词表。containment 已废弃（对术语不可靠）。"""
    if pred is None or gold is None:
        return False
    p, g = pred.strip(), gold.strip()
    if p == g:
        return True
    if synonyms:
        gold_syns = set(synonyms.get(g, [])) | {g}
        pred_syns = set(synonyms.get(p, [])) | {p}
        if gold_syns & pred_syns:
            return True
    return False


def parse_answer(text, prefix):
    """从模型输出提取 '映射结果: xxx' 后的内容。"""
    if prefix in text:
        return text.split(prefix, 1)[1].strip().split("\n")[0].strip()
    return text.strip().split("\n")[0].strip()


def save_json(obj, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_csv(rows, path, headers):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
