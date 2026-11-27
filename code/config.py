# -*- coding: utf-8 -*-
"""全局配置 v5.0 —— 所有路径/字段/种子/阈值的唯一来源。禁止在子脚本另立常量。"""
import os

# ========== 路径 ==========
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIG_DIR     = os.path.join(BASE_DIR, "figures")
for d in (DATA_DIR, RESULTS_DIR, FIG_DIR):
    os.makedirs(d, exist_ok=True)

# 输入数据
DATASET_A     = os.path.join(DATA_DIR, "dataset_a_with_candidates.json")
DATASET_B     = os.path.join(DATA_DIR, "dataset_b.json")
SYNONYMS_A    = os.path.join(DATA_DIR, "synonyms_A.json")
SYNONYMS_B    = os.path.join(DATA_DIR, "synonyms_B.json")
CONCEPT_MSC   = os.path.join(DATA_DIR, "concept_msc.json")
ALL_CONCEPTS  = os.path.join(DATA_DIR, "all_concepts.json")

# 原始来源（回填用）
OMD_ONTOLOGY_RAW = os.path.join(DATA_DIR, "omd_ontology.json")
CHIP_CDN_RAW     = os.path.join(DATA_DIR, "chip_cdn_raw.json")

# ========== 模型 ==========
MODEL_NAME   = "/root/autodl-tmp/models/Qwen3-32B"
DEVICE       = "cuda"
DTYPE        = "bfloat16"          # Qwen3-32B bf16, A800 80G
VLLM_BACKEND = "xgrammar"
TEMPERATURE  = 0.0
MAX_TOKENS   = 32                # 术语很短，避免生成跑飞

# ========== 实验参数 ==========
N_REPEATS      = 3
SEED_BASE      = 20260101
BOOTSTRAP_SEED = 42
BOOTSTRAP_N    = 10000
CI_LEVEL       = 0.95

# 数据集规模（当前实际值；计划值 260/200 待扩充）
N_DATASET_A = 150
N_DATASET_B = 150

# ========== 字段名（内部标准名 → 真实 JSON 字段）==========
# 若 JSON 字段不同，只改这里
FIELD = {
    "id":          "protege_id",
    "term":        "name",
    "source":      "source_book",
    "definition":  "original_definition",
    "gold":        "modern_definition",
    "p_class":     "P_Class",
    "l_class":     "L_Class",
    "candidates":  "candidates",
}

# ========== Pilot / 判定阈值（预注册前锁定，锁定后禁止事后改）==========
# v5.1: norm_mass = log(gold_share * |C|) 为主变量；gold_share 阈值保留向后兼容
THRESH = {
    "coverage_min":        1.00,
    "chance_hit_max":      0.50,
    "ro_pool_min":         19,
    "sc_pool_min":         19,
    "msc_same_min":        10,
    "common_support_iou":  0.15,
    # gold_share 阈值（旧版兼容）
    "dynrange_spread":     0.30,
    "dynrange_low_frac":   0.15,
    "dynrange_high_frac":  0.15,
    # norm_mass 阈值（v5.1，暂为描述性，预注册前锁定）
    "norm_dynrange_spread":     1.5,    # p90-p10 至少 1.5 log-odds 单位
    "norm_dynrange_low_frac":    0.10,  # 至少 10% 样本 < 0（弱于均匀基线）
    "norm_dynrange_high_frac":   0.10,  # 至少 10% 样本 > ln2≈0.69（gold 2×均匀基线）
}

# ========== Prompt（统一，所有条件共用）==========
SYSTEM_PROMPT = "你是一位中国古代数学文献专家。"

def build_user_prompt(term, source, definition):
    return (
        f"请给出古籍术语「{term}」对应的现代数学概念。\n"
        f"出处: {source or ''}\n"
        f"定义: {definition or ''}\n\n"
        f"请只输出映射结果，格式为：\n映射结果: "
    )

SYSTEM_PROMPT_B = "你是一位临床医学术语标准化专家。"

def build_user_prompt_b(raw_term, context):
    return (
        f"请给出临床诊断词「{raw_term}」对应的 ICD-10 标准诊断术语。\n"
        f"上下文: {context or ''}\n\n"
        f"请只输出标准术语，格式为：\n标准术语: "
    )

ANSWER_PREFIX   = "映射结果: "
ANSWER_PREFIX_B = "标准术语: "
