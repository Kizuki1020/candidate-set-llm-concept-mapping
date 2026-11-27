"""Replay the 160-record construction audit.

Permuted-label coverage reported as current_replay is taken from the archived
pre-injection lists (construction_audit/pre_injection_lists_160.json) and does
not rebuild the candidate pool.
"""
import json
import random
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / 'DATA1'))
import build_candidates_A_v2 as builder

cases = json.loads((root / 'construction_audit/input_160.json').read_text())['test_cases']
archive = json.loads((root / 'construction_audit/pre_injection_lists_160.json').read_text())
assert [r['protege_id'] for r in cases] == [r['protege_id'] for r in archive['records']]
lists = {r['protege_id']: r['candidates'] for r in archive['records']}
onto = builder.load_ontology(str(root / 'DATA1/omd_ontology_v2.json'))
n = len(cases)
unpermuted, misses = builder.check_coverage(cases, lists, onto)
gold = [onto.canonical(r['modern_definition']) for r in cases]
pools = [{onto.canonical(s) for s in lists[r['protege_id']]} for r in cases]
order = list(range(n))
random.Random(builder.RANDOM_SEED).shuffle(order)
replay_hits = sum(gold[j] in pools[i] for i, j in enumerate(order))
assert unpermuted == 156 / n and len(misses) == 4 and replay_hits == 28

historical = json.loads((root / 'data/dataset_a_with_candidates.json').read_text())
report = {
    'historical': {k: historical['diagnostics'][k] for k in ['n', 'permuted_gold_coverage', 'structural_lift']},
    'current_replay': {
        'n': n,
        'permuted_gold_coverage': replay_hits / n,
        'structural_lift': 1 - replay_hits / n,
        'source': 'Archived pre-injection lists in construction_audit/pre_injection_lists_160.json',
    },
    'source': 'Recovered 160-record inputs with archived pre-injection lists; fixed archived ontology.',
    'lists_built_with_python': archive.get('build_environment', {}).get('python', '3.9.6'),
    'changed_pre_injection_lists': 0,
}
(root / 'construction_audit/replay_comparison.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
