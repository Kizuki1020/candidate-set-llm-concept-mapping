"""Conditional permutation distribution on 160 archived pre-injection lists.

The simulation loads construction_audit/pre_injection_lists_160.json and does
not rebuild the candidate pool. Canonical string matching uses the archived
ontology. A live rebuild is environment-dependent and is not used here.
"""
import json
import random
import sys
from pathlib import Path

import numpy as np

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / 'DATA1'))
import build_candidates_A_v2 as builder

N_DRAWS = 100_000
SEED = 20261127
out = root / 'construction_audit'
cases = json.loads((out / 'input_160.json').read_text())['test_cases']
archive = json.loads((out / 'pre_injection_lists_160.json').read_text())
n = len(cases)
assert n == len({r['protege_id'] for r in cases}) == archive['n'] == 160
assert [r['protege_id'] for r in cases] == [r['protege_id'] for r in archive['records']]
lists = {r['protege_id']: r['candidates'] for r in archive['records']}
assert all(len(lists[pid]) == rec['n_candidates'] for rec, pid in ((r, r['protege_id']) for r in archive['records']))

onto = builder.load_ontology(str(root / 'DATA1/omd_ontology_v2.json'))
unpermuted, misses = builder.check_coverage(cases, lists, onto)
assert unpermuted == 156 / n and len(misses) == 4
gold = [onto.canonical(r['modern_definition']) for r in cases]
pools = [{onto.canonical(s) for s in lists[r['protege_id']]} for r in cases]
# Column j represents one original record, so duplicate label strings retain their frequencies.
hits = np.asarray([[s in pool for s in gold] for pool in pools], dtype=np.uint8)
row = np.arange(n)
assert hits.trace() == 156
order = list(range(n))
random.Random(builder.RANDOM_SEED).shuffle(order)
replay_count = int(hits[row, order].sum())
assert replay_count == 28
historic = json.loads((root / 'data/dataset_a_with_candidates.json').read_text())['diagnostics']
assert historic['n'] == n and historic['permuted_gold_coverage'] == 36 / n
assert historic['structural_lift'] == 1 - 36 / n
rng = np.random.Generator(np.random.PCG64(SEED))
counts = np.fromiter((hits[row, rng.permutation(n)].sum() for _ in range(N_DRAWS)), dtype=np.int32, count=N_DRAWS)
quantiles = np.quantile(counts, [0.025, 0.5, 0.975], method='inverted_cdf').astype(int)
expected = float(hits.sum() / n)
centered = hits - hits.mean(axis=0) - hits.mean(axis=1)[:, None] + hits.mean()
sd = float(np.sqrt((centered ** 2).sum() / (n - 1)))
assert abs(counts.mean() - expected) < 6 * sd / np.sqrt(N_DRAWS)
import csv as _csv
with (out / 'permutation_frequencies.csv').open('w', newline='') as stream:
    writer = _csv.writer(stream)
    writer.writerow(['hits_of_160', 'number_of_permutations'])
    writer.writerows((int(k), int(v)) for k, v in zip(*np.unique(counts, return_counts=True)))
report = dict(
    n=n, draws=N_DRAWS, generator='numpy PCG64', simulation_seed=SEED,
    replay_seed=builder.RANDOM_SEED, replay_generator='Python random.Random.shuffle',
    list_source='construction_audit/pre_injection_lists_160.json',
    original_hits=int(hits.trace()), injected=4, archived_hits=36, replay_hits=replay_count,
    analytical_mean_hits=expected, analytical_sd_hits=sd, simulated_mean_hits=float(counts.mean()),
    simulated_sd_hits=float(counts.std(ddof=1)), discrete_quantiles_025_50_975=quantiles.tolist(),
    count_percentiles={str(k): float(np.mean(counts <= k)) for k in (28, 36)},
    probability_mass_at_observed_counts={str(k): float(np.mean(counts == k)) for k in (28, 36)},
)
(out / 'permutation_distribution.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
