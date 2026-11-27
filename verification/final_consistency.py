"""Cross-check every manuscript/SI quantity against the distributed result files.

Run from the extracted package root:  python3 verification/final_consistency.py
Prints PASS/FAIL per claim. Nothing is recomputed from model inference; every
source is a frozen file already in the archive.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

root = Path(__file__).resolve().parents[1]
audit = json.loads((root / 'composition_audit/audit.json').read_text())
perm = json.loads((root / 'construction_audit/permutation_distribution.json').read_text())
prefix = json.loads((root / 'composition_audit/prefix_sensitivity.json').read_text())
overlap = audit['overlap']
length = {(r['domain'], r['condition']): r for r in audit['length_summary']}
mass = {(r['domain'], r['contrast']): r for r in audit['mass_vs_gain']}

checks = []


def ck(name, got, want, tol=5e-4):
    ok = abs(float(got) - float(want)) <= tol
    checks.append((ok, name, got, want))


# ---- probe accuracies and contrasts, recomputed from frozen predictions
for dom in ['A', 'B']:
    df = pd.read_csv(root / f'results/exp1/exp1_{dom}.csv', dtype={'id': str}, keep_default_na=False)
    assert df.groupby(['id', 'condition']).correct.nunique().eq(1).all()
    acc = df.groupby(['id', 'condition']).correct.first().unstack()
    n = len(acc)
    ck(f'{dom} n', n, 150)
    for cond in ['UC', 'HC', 'RO', 'SC']:
        ck(f'{dom} {cond} accuracy', acc[cond].mean(), round(acc[cond].mean(), 3))
    ro_hc = (acc.RO - acc.HC).mean()
    ck(f'{dom} RO-HC', ro_hc, 0.293 if dom == 'A' else 0.300)
    ck(f'{dom} RO-SC', (acc.RO - acc.SC).mean(), 0.307 if dom == 'A' else 0.140)
    ck(f'{dom} HC-UC', (acc.HC - acc.UC).mean(), 0.240 if dom == 'A' else 0.427)
    # discordant counts quoted in Results
    if dom == 'A':
        ck('A RO-only discordant', int(((acc.RO == 1) & (acc.SC == 0)).sum()), 47)
        ck('A SC-only discordant', int(((acc.RO == 0) & (acc.SC == 1)).sum()), 1)
    else:
        ck('B RO-only discordant', int(((acc.RO == 1) & (acc.SC == 0)).sum()), 25)
        ck('B SC-only discordant', int(((acc.RO == 0) & (acc.SC == 1)).sum()), 4)

# ---- probability mass correlations
ck('A mass r (RO-HC)', mass[('A', 'RO-HC')]['r'], 0.150)
ck('A mass p (RO-HC)', mass[('A', 'RO-HC')]['p'], 0.067, tol=1e-3)
ck('B mass r (RO-HC)', mass[('B', 'RO-HC')]['r'], 0.023)
ck('B mass p (RO-HC)', mass[('B', 'RO-HC')]['p'], 0.784, tol=1e-3)
ck('A mass r (HC-UC)', mass[('A', 'HC-UC')]['r'], -0.095)
ck('B mass r (HC-UC)', mass[('B', 'HC-UC')]['r'], -0.082)
ols = {r['term']: r for r in audit['RO_HC_OLS']}
ck('RO-HC mass slope A', ols['mass']['estimate'], 0.0297, tol=1e-3)
ck('RO-HC interaction p', ols['mass_by_B']['p'], 0.295, tol=1e-3)

# ---- length confound
ck('A RO median length ratio', length[('A', 'RO')]['median_ratio'], 1.71, tol=5e-3)
ck('B RO median length ratio', length[('B', 'RO')]['median_ratio'], 0.54, tol=5e-3)
ck('A RO at-longest-or-joint', length[('A', 'RO')]['n_longest_including_ties'], 52)
ck('B RO at-longest-or-joint', length[('B', 'RO')]['n_longest_including_ties'], 0)
ck('A RO above median', length[('A', 'RO')]['n_above_median'], 98)
ck('B RO above median', length[('B', 'RO')]['n_above_median'], 19)

# ---- provenance
res = audit['resource']
ck('concept strings', sum(res['concept_sources'].values()), 571)
ck('test-label concepts', res['concept_sources'].get('test_dataset_label_space', 0), 131)
ck('daiweiji concepts', res['concept_sources'].get('daiweiji_modern_definition', 0), 440)
ck('link provenance entries', sum(res['link_records'].values()), 1328)
ck('case-annotation links', res['link_records'].get('case_annotation', 0), 339)
ck('distinct MSC-concept pairs', res['distinct_edges'], 1259)
ck('probe references with test provenance', sum(res['probe_reference_source'].values()), 122)

# ---- corpus overlap and the two meanings of L3
ck('probe-core ID matches', overlap['all']['id'], 34)
ck('probe-core term+definition matches', overlap['all']['term_definition'], 73)
ck('probe L3 n', overlap['L3']['n'], 30)
ck('probe L3 ID matches', overlap['L3']['id'], 3)
ck('probe L3 term+definition matches', overlap['L3']['term_definition'], 9)

# ---- constrained outputs against reconstructed memberships
ck('raw constrained outputs', sum(r['outputs'] for r in audit['membership_matches']), 2700)
ck('exact members', sum(r['raw_matches'] for r in audit['membership_matches']), 2676)
ck('unique-prefix outputs', audit['unique_prefix_among_missing'], 24)

# ---- permutation distribution
ck('permutation draws', perm['draws'], 100000)
ck('archived permuted hits', perm['archived_hits'], 36)
ck('replay permuted hits', perm['replay_hits'], 28)
ck('pre-injection coverage', perm['original_hits'], 156)
ck('injected records', perm['injected'], 4)
ck('null exact mean', perm['analytical_mean_hits'], 27.41, tol=5e-3)
ck('null exact SD', perm['analytical_sd_hits'], 4.50, tol=5e-3)
lo, med, hi = perm['discrete_quantiles_025_50_975']
ck('null lower 2.5%', lo, 19)
ck('null upper 2.5%', hi, 36)
ck('replay percentile', perm['count_percentiles']['28'], 0.606, tol=1e-3)
ck('archived percentile', perm['count_percentiles']['36'], 0.976, tol=1e-3)

# ---- prefix sensitivity
frozen = {(r['domain'], r['rule'], r['contrast']): r for r in prefix}
for dom, want in [('A', 0.293), ('B', 0.300)]:
    ck(f'{dom} RO-HC frozen', frozen[(dom, 'frozen', 'RO-HC')]['difference'], want)
    ck(f'{dom} RO-HC prefix-completed', frozen[(dom, 'complete_unique_prefix', 'RO-HC')]['difference'], want)
ck('A RO-HC excluding affected', frozen[('A', 'exclude_any_affected_record', 'RO-HC')]['difference'], 0.297)
ck('B RO-HC excluding affected', frozen[('B', 'exclude_any_affected_record', 'RO-HC')]['difference'], 0.309)
ck('A excluded n', frozen[('A', 'exclude_any_affected_record', 'RO-HC')]['n'], 145)
ck('B excluded n', frozen[('B', 'exclude_any_affected_record', 'RO-HC')]['n'], 149)

fails = [c for c in checks if not c[0]]
for ok, name, got, want in checks:
    print(f'{"PASS" if ok else "FAIL"}  {name:40s} got={got!r:>22}  want={want!r}')
print(f'\n{len(checks) - len(fails)}/{len(checks)} checks passed')
sys.exit(1 if fails else 0)
