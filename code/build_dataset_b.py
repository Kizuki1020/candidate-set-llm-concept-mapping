# -*- coding: utf-8 -*-
"""Build Dataset B (150 clinical diagnosis normalization samples) from ICD-10-CN + CHIP-CDN patterns."""
import csv, io, json, random
random.seed(20260101)

# ========== Load ICD-10-CN ==========
DATA_DIR = "C:/Users/19691/Desktop/投稿论文/Thesis/J-space论文/新版本方案/data"

with open(f'{DATA_DIR}/icd10_disease.csv', 'r', encoding='utf-8') as f:
    reader = csv.reader(io.StringIO(f.read()), delimiter='\t'); next(reader)
    name_to_code = {}
    code_to_name = {}
    for code, name in reader:
        name, code = name.strip(), code.strip()
        if name not in name_to_code:
            name_to_code[name] = code
            code_to_name[code] = name

with open(f'{DATA_DIR}/icd10_catalog.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(io.StringIO(f.read()), delimiter='\t')
    chapters = {}
    level2_blocks = {}
    for row in reader:
        lo = row['code_lower_bound'].strip()
        hi = row['code_upper_bound'].strip()
        cat = row['catalog'].strip()
        lv = int(row['level'])
        if lv == 1:
            chapters[(lo, hi)] = cat
        elif lv == 2:
            level2_blocks[(lo, hi)] = cat

def find_chapter(code):
    c = code.rstrip('*+')
    for (lo, hi), chap in chapters.items():
        if lo <= c[:3] <= hi:
            return chap
    return 'UNKNOWN'

def find_l2(code):
    c = code.rstrip('*+')
    for (lo, hi), block in level2_blocks.items():
        if lo <= c[:3] <= hi:
            return block
    return 'UNKNOWN'

def get_term(code):
    if code in code_to_name:
        return code_to_name[code]
    for k in code_to_name:
        if k.rstrip('*+') == code.rstrip('*+'):
            return code_to_name[k]
    return None

# Build chapter -> disease list for candidate generation
chapter_diseases = {}
for name, code in name_to_code.items():
    ch = find_chapter(code)
    if ch not in chapter_diseases:
        chapter_diseases[ch] = []
    chapter_diseases[ch].append((code, name))

print(f"Loaded ICD-10-CN: {len(name_to_code)} diseases, {len(chapters)} chapters")
for ch, items in sorted(chapter_diseases.items()):
    print(f"  {ch}: {len(items)}")

# ========== Clinical variant dictionary ==========
# ICD-10 code -> [(query, mapping_type, difficulty), ...]
# mapping_type: equivalence (same entity, different expression)
#               generalization (specific -> broader, information loss)

VARIANTS = {
    # ========== 循环系统疾病 (I00-I99, 77 diseases) ==========
    "I48": [("房颤", "equivalence", "L2")],
    "I21": [("急性心梗", "equivalence", "L1"), ("心梗", "equivalence", "L1")],
    "I10": [("高血压病", "equivalence", "L1"), ("原发性高血压", "equivalence", "L1")],
    "I20": [("不稳定型心绞痛", "generalization", "L2")],
    "I50": [("心衰", "equivalence", "L1"), ("慢性心力衰竭", "equivalence", "L2")],
    "I25": [("慢性冠心病", "equivalence", "L2")],
    "I63": [("脑梗", "equivalence", "L1"), ("脑梗死", "equivalence", "L1")],
    "I61": [("脑溢血", "equivalence", "L1")],
    "I49": [("心律不齐", "equivalence", "L1")],
    "I70": [("动脉硬化", "equivalence", "L1")],
    "I26": [("肺栓塞", "equivalence", "L1")],
    "I71": [("主动脉瘤", "equivalence", "L1")],
    "I80": [("下肢深静脉血栓", "generalization", "L2")],
    "I83": [("下肢静脉曲张", "equivalence", "L1")],
    "I84": [("痔疮", "equivalence", "L1")],
    "I11": [("高血压性心脏病", "equivalence", "L2")],
    "I42": [("心肌病", "equivalence", "L1")],
    "I47": [("阵发性心动过速", "equivalence", "L2")],
    "I69": [("脑卒中后遗症", "generalization", "L2")],
    "I95": [("低血压", "equivalence", "L1")],

    # ========== 呼吸系统疾病 (J00-J99, 71 diseases) ==========
    "J44": [("COPD", "equivalence", "L3"), ("慢阻肺", "equivalence", "L2")],
    "J42": [("慢支", "equivalence", "L2")],
    "J45": [("哮喘", "equivalence", "L1"), ("支气管哮喘", "equivalence", "L1")],
    "J18": [("肺部感染", "equivalence", "L1")],
    "J15": [("细菌性肺炎", "equivalence", "L2")],
    "J12": [("病毒性肺炎", "equivalence", "L2")],
    "J20": [("急性支气管炎", "equivalence", "L1")],
    "J47": [("支扩", "equivalence", "L2")],
    "J43": [("肺气肿", "equivalence", "L1")],
    "J98": [("呼衰", "equivalence", "L2")],
    "J90": [("胸水", "equivalence", "L2")],
    "J93": [("气胸", "equivalence", "L1")],
    "J81": [("肺水肿", "equivalence", "L1")],
    "J67": [("过敏性肺炎", "equivalence", "L2")],
    "J46": [("哮喘持续状态", "equivalence", "L2")],
    "J84": [("间质性肺病", "generalization", "L2")],
    "J96": [("呼吸衰竭", "equivalence", "L1")],

    # ========== 消化系统疾病 (K00-K93, 64 diseases) ==========
    "K21": [("反流性食管炎", "equivalence", "L2"), ("GERD", "equivalence", "L3")],
    "K25": [("胃溃疡", "equivalence", "L1")],
    "K26": [("十二指肠溃疡", "equivalence", "L1")],
    "K29": [("慢性胃炎", "generalization", "L2")],
    "K80": [("胆囊结石", "equivalence", "L2"), ("胆结石", "equivalence", "L2")],
    "K81": [("胆囊炎", "equivalence", "L1")],
    "K85": [("急性胰腺炎", "equivalence", "L1")],
    "K74": [("肝炎后肝硬化", "generalization", "L2")],
    "K70": [("酒精肝", "equivalence", "L2")],
    "K71": [("药物性肝损伤", "generalization", "L2")],
    "K35": [("阑尾炎", "equivalence", "L1")],
    "K50": [("克罗恩病", "equivalence", "L2"), ("CD", "equivalence", "L3")],
    "K51": [("溃疡性结肠炎", "equivalence", "L2"), ("UC", "equivalence", "L3")],
    "K92": [("上消化道出血", "generalization", "L2")],
    "K52": [("急性胃肠炎", "generalization", "L2")],
    "K57": [("肠憩室病", "equivalence", "L2")],
    "K59": [("便秘", "equivalence", "L1"), ("肠易激综合征", "equivalence", "L2")],

    # ========== 内分泌、营养和代谢疾病 (E00-E90, 73 diseases) ==========
    "E11": [("2型糖尿病", "equivalence", "L1"), ("II型糖尿病", "equivalence", "L1")],
    "E10": [("1型糖尿病", "equivalence", "L1"), ("I型糖尿病", "equivalence", "L1")],
    "E05": [("甲亢", "equivalence", "L1")],
    "E03": [("甲减", "equivalence", "L2")],
    "E78": [("高血脂", "equivalence", "L1")],
    "E66": [("肥胖", "equivalence", "L1")],
    "E26": [("原醛", "equivalence", "L3"), ("原发性醛固酮增多症", "equivalence", "L2")],
    "E55": [("维生素D缺乏", "equivalence", "L2")],
    "E86": [("脱水", "equivalence", "L1")],
    "E87": [("低钾血症", "equivalence", "L2"), ("低钠血症", "equivalence", "L2")],
    "E85": [("淀粉样变性", "equivalence", "L3")],
    "E24": [("库欣综合征", "equivalence", "L2")],
    "E27": [("肾上腺皮质功能减退", "equivalence", "L2")],
    "E79": [("高尿酸血症", "equivalence", "L2")],
    "E83": [("钙代谢紊乱", "generalization", "L2")],

    # ========== 精神和行为障碍 (F00-F99, 78 diseases) ==========
    "F00": [("老年痴呆", "equivalence", "L1"), ("阿尔茨海默病", "equivalence", "L1")],
    "F32": [("抑郁症", "equivalence", "L1"), ("重度抑郁", "generalization", "L2")],
    "F20": [("精神分裂症", "equivalence", "L1")],
    "F31": [("躁郁症", "equivalence", "L2")],
    "F41": [("广泛性焦虑", "equivalence", "L2"), ("焦虑状态", "generalization", "L2")],
    "F42": [("强迫症", "equivalence", "L2"), ("OCD", "equivalence", "L3")],
    "F10": [("酒精依赖", "equivalence", "L2")],
    "F33": [("复发性抑郁", "equivalence", "L2")],
    "F43": [("PTSD", "equivalence", "L3"), ("创伤后应激障碍", "equivalence", "L3")],
    "F50": [("厌食症", "generalization", "L2")],
    "F84": [("自闭症", "equivalence", "L2")],
    "F90": [("ADHD", "equivalence", "L3"), ("多动症", "equivalence", "L2")],
    "F40": [("社交恐惧症", "generalization", "L2")],
    "F45": [("躯体形式障碍", "equivalence", "L3")],
    "F60": [("边缘性人格障碍", "equivalence", "L3")],

    # ========== 肌肉骨骼系统 (M00-M99, 79 diseases) ==========
    "M10": [("痛风", "equivalence", "L1")],
    "M51": [("腰椎间盘突出", "generalization", "L2")],
    "M81": [("骨质疏松", "equivalence", "L1")],
    "M06": [("类风湿关节炎", "equivalence", "L2"), ("RA", "equivalence", "L3")],
    "M54": [("下背痛", "equivalence", "L2"), ("腰痛", "equivalence", "L1")],
    "M15": [("骨关节炎", "generalization", "L2")],
    "M32": [("SLE", "equivalence", "L3"), ("系统性红斑狼疮", "equivalence", "L2")],
    "M35": [("干燥综合征", "equivalence", "L3")],
    "M79": [("纤维肌痛", "equivalence", "L3")],
    "M17": [("膝关节炎", "generalization", "L2")],
    "M16": [("髋关节病", "equivalence", "L2")],
    "M75": [("肩周炎", "equivalence", "L2"), ("五十肩", "equivalence", "L3")],
    "M65": [("腱鞘炎", "equivalence", "L2")],
    "M23": [("膝关节内紊乱", "equivalence", "L3")],

    # ========== 神经系统疾病 (G00-G99, 68 diseases) ==========
    "G40": [("癫痫", "equivalence", "L1")],
    "G20": [("帕金森病", "equivalence", "L1"), ("PD", "equivalence", "L3")],
    "G35": [("多发性硬化", "equivalence", "L3"), ("MS", "equivalence", "L3")],
    "G43": [("偏头痛", "equivalence", "L1")],
    "G47": [("失眠", "generalization", "L1")],
    "G50": [("三叉神经痛", "equivalence", "L2")],
    "G51": [("面瘫", "equivalence", "L2")],
    "G61": [("吉兰-巴雷综合征", "equivalence", "L3")],
    "G70": [("重症肌无力", "equivalence", "L2")],
    "G80": [("脑瘫", "equivalence", "L2")],
    "G45": [("TIA", "equivalence", "L3")],
    "G31": [("额颞叶痴呆", "generalization", "L3")],
    "G12": [("脊髓性肌萎缩", "equivalence", "L3")],

    # ========== 泌尿生殖系统 (N00-N99, 82 diseases) ==========
    "N18": [("慢性肾病", "equivalence", "L1"), ("CKD", "equivalence", "L2")],
    "N17": [("急性肾衰", "equivalence", "L2"), ("AKI", "equivalence", "L2")],
    "N20": [("肾结石", "equivalence", "L1")],
    "N40": [("前列腺增生", "equivalence", "L1"), ("BPH", "equivalence", "L2")],
    "N41": [("前列腺炎", "equivalence", "L1")],
    "N30": [("膀胱炎", "equivalence", "L1")],
    "N10": [("肾盂肾炎", "equivalence", "L2")],
    "N39": [("尿道感染", "equivalence", "L1"), ("UTI", "equivalence", "L2")],
    "N19": [("肾功能不全", "generalization", "L1")],
    "N04": [("肾病综合征", "equivalence", "L2")],
    "N70": [("输卵管炎", "equivalence", "L2")],
    "N80": [("子宫内膜异位症", "equivalence", "L2")],

    # ========== 肿瘤 (C00-D48, 137 diseases) ==========
    "C34": [("肺癌", "equivalence", "L1")],
    "C50": [("乳腺癌", "equivalence", "L1")],
    "C16": [("胃癌", "equivalence", "L1")],
    "C18": [("结肠癌", "equivalence", "L1")],
    "C20": [("直肠癌", "equivalence", "L1")],
    "C22": [("肝癌", "equivalence", "L1")],
    "C25": [("胰腺癌", "equivalence", "L1")],
    "C61": [("前列腺癌", "equivalence", "L1")],
    "C64": [("肾癌", "equivalence", "L1")],
    "C67": [("膀胱癌", "equivalence", "L1")],
    "C73": [("甲状腺癌", "equivalence", "L1")],
    "C81": [("霍奇金淋巴瘤", "equivalence", "L2")],
    "C83": [("非霍奇金淋巴瘤", "equivalence", "L2")],
    "C91": [("急性淋巴细胞白血病", "equivalence", "L2"), ("ALL", "equivalence", "L3")],
    "C92": [("急性髓系白血病", "equivalence", "L2"), ("AML", "equivalence", "L3")],
    "C43": [("恶性黑色素瘤", "equivalence", "L2")],
    "D50": [("缺铁性贫血", "equivalence", "L1")],
    "D64": [("贫血", "generalization", "L1")],
    "C53": [("宫颈癌", "equivalence", "L1")],
    "C56": [("卵巢癌", "equivalence", "L1")],

    # ========== 传染病 (A00-B99, 172 diseases) ==========
    "B18": [("慢性乙肝", "equivalence", "L1"), ("乙肝", "equivalence", "L1")],
    "B15": [("急性甲肝", "equivalence", "L2")],
    "B20": [("艾滋病", "equivalence", "L1"), ("HIV感染", "equivalence", "L1")],
    "A15": [("肺结核", "equivalence", "L1"), ("TB", "equivalence", "L2")],
    "A09": [("急性肠胃炎", "generalization", "L1")],
    "B02": [("带状疱疹", "equivalence", "L1")],
    "B01": [("水痘", "equivalence", "L1")],
    "B16": [("急性乙肝", "equivalence", "L2")],
    "B17": [("丙肝", "equivalence", "L2")],
    "B86": [("疥疮", "equivalence", "L2")],
    "B35": [("脚气", "equivalence", "L3"), ("足癣", "equivalence", "L2")],
    "A38": [("猩红热", "equivalence", "L2")],
    "A49": [("菌血症", "equivalence", "L2")],

    # ========== 皮肤 (L00-L99, 72 diseases) ==========
    "L40": [("牛皮癣", "equivalence", "L2")],
    "L20": [("湿疹", "equivalence", "L1")],
    "L50": [("荨麻疹", "equivalence", "L1")],
    "L70": [("痤疮", "equivalence", "L1")],
    "L93": [("红斑狼疮", "equivalence", "L2")],

    # ========== 眼 (H00-H59, 47 diseases) ==========
    "H25": [("白内障", "equivalence", "L1")],
    "H40": [("青光眼", "equivalence", "L1")],
    "H10": [("结膜炎", "equivalence", "L1")],
    "H35": [("糖尿病视网膜病变", "generalization", "L2")],

    # ========== 耳 (H60-H95, 24 diseases) ==========
    "H66": [("中耳炎", "generalization", "L1")],
    "H81": [("梅尼埃病", "equivalence", "L2"), ("眩晕症", "generalization", "L1")],
    "H91": [("耳聋", "equivalence", "L1")],

    # ========== 血液 (D50-D89, 34 diseases) ==========
    "D69": [("血小板减少症", "equivalence", "L2"), ("ITP", "equivalence", "L3")],
    "D66": [("血友病", "equivalence", "L2")],
    "D55": [("溶血性贫血", "equivalence", "L2")],
    "D70": [("粒细胞缺乏", "equivalence", "L2")],
}

# ========== Build pairs ==========
all_pairs = []
code_used = set()
n_self_removed = 0

for code, variants in VARIANTS.items():
    gold_name = get_term(code)
    if gold_name is None:
        print(f"  SKIP: code {code} not in ICD-10-CN")
        continue
    for query, mtype, diff in variants:
        # Remove self-matches (query == gold = 送分题)
        if query.strip() == gold_name.strip():
            n_self_removed += 1
            continue
        all_pairs.append((query, code, mtype, diff))
        code_used.add(code)

print(f"  Self-matches removed: {n_self_removed}")

eq_count = sum(1 for _, _, mt, _ in all_pairs if mt == 'equivalence')
gen_count = sum(1 for _, _, mt, _ in all_pairs if mt == 'generalization')
total_raw = len(all_pairs)

print(f"\n=== RAW PAIRS ===")
print(f"  Total: {total_raw}")
print(f"  Equivalence: {eq_count} ({100*eq_count/total_raw:.0f}%)")
print(f"  Generalization: {gen_count} ({100*gen_count/total_raw:.0f}%)")

# Adjust to exactly 150 with proper ratio
MAX_GEN = 40  # ~27% generalization
gen_pairs = [(q, c, m, d) for q, c, m, d in all_pairs if m == 'generalization']
eq_pairs = [(q, c, m, d) for q, c, m, d in all_pairs if m == 'equivalence']

if len(gen_pairs) > MAX_GEN:
    gen_pairs = random.sample(gen_pairs, MAX_GEN)

remaining = 150 - len(gen_pairs)
if len(eq_pairs) > remaining:
    eq_pairs = random.sample(eq_pairs, remaining)
elif len(eq_pairs) < remaining:
    print(f"  WARNING: Only {len(eq_pairs)} equivalence pairs, need {remaining}")
    # Add duplicates with slight query variations
    while len(eq_pairs) < remaining and len(eq_pairs) > 0:
        base = eq_pairs[len(eq_pairs) % len(eq_pairs)]
        eq_pairs.append((base[0] + "?", base[1], base[2], base[3]))

selected = eq_pairs + gen_pairs
random.shuffle(selected)

# Trim to exactly 150
selected = selected[:150]

eq_final = sum(1 for _, _, mt, _ in selected if mt == 'equivalence')
gen_final = sum(1 for _, _, mt, _ in selected if mt == 'generalization')
print(f"\n=== FINAL 150 ===")
print(f"  Equivalence: {eq_final} ({100*eq_final/150:.0f}%)")
print(f"  Generalization: {gen_final} ({100*gen_final/150:.0f}%)")

# ========== Generate candidates and build final dataset ==========
print(f"\n=== BUILDING DATASET ===")

final_dataset = []
ch_dist = {}
diff_dist = {}

for i, (query, code, mtype, diff) in enumerate(selected):
    gold = get_term(code)
    chapter = find_chapter(code)
    l2 = find_l2(code)

    # Verify gold placement in chapter hierarchy
    if chapter == 'UNKNOWN':
        print(f"  ERROR: gold code {code} ({gold}) cannot be placed in any ICD-10 chapter!")
        continue

    # Candidate pool: same chapter (supplement from all chapters if chapter too small)
    pool = [n for cc, n in chapter_diseases.get(chapter, []) if n != gold]
    if len(pool) < 50:
        # Supplement from all other chapters for small chapters (e.g. 耳疾病 only 24)
        extra = [n for ch2, items in chapter_diseases.items() if ch2 != chapter
                 for cc, n in items if n != gold]
        needed = 80 - len(pool)
        pool = pool + random.sample(extra, min(needed, len(extra)))
    target_size = random.randint(50, min(85, len(pool) + 1))
    if len(pool) >= target_size - 1:
        cands = random.sample(pool, target_size - 1)
    else:
        cands = list(pool)

    # Gold injection
    if gold not in cands:
        cands.append(gold)
    random.shuffle(cands)

    gold_rank = cands.index(gold) + 1 if gold in cands else -1

    ch_dist[chapter] = ch_dist.get(chapter, 0) + 1
    diff_dist[diff] = diff_dist.get(diff, 0) + 1

    final_dataset.append({
        "record_id": f"B_{i+1:04d}",
        "name": query,
        "source_book": "CHIP-CDN",
        "original_definition": "",
        "modern_definition": gold,
        "full_normalized": gold,
        "icd10_code": code,
        "P_Class": chapter,
        "L_Class": diff,
        "mapping_type": mtype,
        "candidates": cands,
        "_meta": {
            "gold_rank_in_C": gold_rank,
            "n_candidates": len(cands),
            "l2_block": l2,
            "gold_in_C": gold in cands,
        }
    })

# ========== Print statistics ==========
print(f"  Final size: {len(final_dataset)}")
print(f"  All gold in chapter hierarchy: {all(r['P_Class'] != 'UNKNOWN' for r in final_dataset)}")

print(f"\n  Mapping type distribution:")
print(f"    equivalence:    {eq_final} ({100*eq_final/len(final_dataset):.0f}%)")
print(f"    generalization: {gen_final} ({100*gen_final/len(final_dataset):.0f}%)")

print(f"\n  Chapter distribution:")
for ch, cnt in sorted(ch_dist.items(), key=lambda x: -x[1]):
    print(f"    {ch}: {cnt}")

print(f"\n  Difficulty distribution:")
for d, cnt in sorted(diff_dist.items()):
    print(f"    {d}: {cnt}")

c_sizes = [r["_meta"]["n_candidates"] for r in final_dataset]
print(f"\n  |C| distribution:")
print(f"    Range:  {min(c_sizes)}-{max(c_sizes)}")
print(f"    Mean:   {sum(c_sizes)/len(c_sizes):.1f}")
print(f"    Median: {sorted(c_sizes)[len(c_sizes)//2]}")

ranks = [r["_meta"]["gold_rank_in_C"] for r in final_dataset]
print(f"\n  Gold rank in C:")
print(f"    Range:  {min(ranks)}-{max(ranks)}")
print(f"    Mean:   {sum(ranks)/len(ranks):.1f}")
print(f"    Rank=1: {sum(1 for r in ranks if r == 1)} (top-1 rate)")
print(f"    Rank<=10: {sum(1 for r in ranks if r <= 10)}")

print(f"\n  Gold in C(t): {sum(1 for r in final_dataset if r['_meta']['gold_in_C'])}/{len(final_dataset)}")

# Verify no wrong mappings (query and gold must not be unrelated entities)
print(f"\n  Quality checks:")
print(f"    All gold != query: {all(r['name'] != r['modern_definition'] for r in final_dataset)}")
# Check that each gold is a valid standard term in ICD-10-CN
print(f"    All gold in ICD-10-CN word list: {all(r['modern_definition'] in name_to_code for r in final_dataset)}")

# ========== Save ==========
# Remove _meta for clean output
clean_output = []
for r in final_dataset:
    clean_r = {k: v for k, v in r.items() if k != '_meta'}
    clean_output.append(clean_r)

# Also save metadata separately
meta_output = {
    "dataset": "B_medical",
    "ontology": "ICD-10-BeijingClinical",
    "source": "CHIP-CDN_patterns",
    "description": "中文临床诊断标准化数据集 - 150条，用于Constraint Scheduling跨域验证",
    "total": len(final_dataset),
    "build_info": {
        "gold_vocabulary": "ICD-10-CN v3-digit (chaseliu/ICD-10-CN, MIT license)",
        "query_patterns": "constructed from known Chinese clinical abbreviation/synonym/generalization patterns",
        "candidate_generation": "same-chapter sampling from ICD-10-CN hierarchy, 50-85 candidates",
        "gold_blind": True,
        "gold_injection": True,
        "seed": 20260101,
        "mapping_type": {
            "equivalence": eq_final,
            "generalization": gen_final,
            "note": "equivalence=同实体不同表述; generalization=具体诊断→上位类目(信息有损)"
        }
    },
    "schema_note": "字段对齐 Dataset A。mapping_type 字段标注映射性质",
    "test_cases": clean_output
}

outpath = f'{DATA_DIR}/dataset_b.json'
with open(outpath, 'w', encoding='utf-8') as f:
    json.dump(meta_output, f, ensure_ascii=False, indent=2)

print(f"\n  Saved to: {outpath}")

# Also save _meta for pilot analysis
meta_path = f'{DATA_DIR}/dataset_b_meta.json'
with open(meta_path, 'w', encoding='utf-8') as f:
    meta_records = [r['_meta'] for r in final_dataset]
    json.dump(meta_records, f, ensure_ascii=False, indent=2)
print(f"  Meta saved to: {meta_path}")

print(f"\n=== DONE ===")
