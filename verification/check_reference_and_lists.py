"""Check diagnostic labels, matching keys and list sizes using archived inputs."""
from pathlib import Path
from collections import Counter
import json
import pandas as pd
from scipy.stats import binomtest
root = Path(__file__).resolve().parents[1]
records = [json.loads((root/'data'/name).read_text())['test_cases'] for name in ['dataset_a_with_candidates.json','dataset_b.json']]
for domain, batch, other in [('A', records[0], records[1]), ('B', records[1], records[0])]:
    other_refs = {r['modern_definition'] for r in other}
    same_strings = {s for r in batch for s in r['candidates']}
    for r in batch:
        k = len(r['candidates'])
        own = set(r['candidates'])
        reference = r['modern_definition']
        assert len(own | {reference}) == k and reference in own
        assert all(len(pool - own - {reference}) >= k-1 for pool in (other_refs, same_strings))
    print(domain, 'equal list lengths supported for',len(batch),'records')
core=pd.read_csv(root/'diagnostic/references/core_828.csv',keep_default_na=False)
sub=core.query("l_class=='L3'")
assert len(core)==828 and len(sub)==181 and sub.gold_label_ontology.eq('NONE').all()
print('Diagnostic L3', len(sub),'expert NONE',int(sub.gold_label.eq('NONE').sum()))
data=[]
keys=['chinese_term','l_class','gold_msc','gold_label','gold_label_ontology']
for path in sorted((root/'diagnostic/outputs').glob('commercial*.csv')):
    d=pd.read_csv(path,keep_default_na=False)
    error=d.selected_letter.replace('','NONE').ne(d.gold_label)
    q=d.query("l_class=='L3'").copy()
    q['within_key']=q.groupby(keys,sort=False).cumcount()
    q[path.stem]=q.selected_code.ne('')
    data.append(q[keys+['within_key',path.stem]])
    print(path.stem,'code selections on 181',int(q[path.stem].sum()),'core errors',int(error.sum()),'high-confidence errors',int(d.loc[error,'confidence'].astype(float).ge(80).sum()))
joint=data[0]
for table in data[1:]:joint=joint.merge(table,on=keys+['within_key'],validate='one_to_one')
flags=[t.columns[-1] for t in data]
assert len(joint)==181
print('Triple code / NONE with within-key order',int(joint[flags].all(axis=1).sum()),int((~joint[flags]).all(axis=1).sum()))
bounds=[0,0]
for _,group in joint.groupby(keys):
    n=len(group); marginal=group[flags].sum()
    bounds[0]+=int(max(0,marginal.sum()-2*n));bounds[1]+=int(marginal.min())
print('Triple-code bounds allowing within-key permutations',bounds)
a=pd.read_csv(root/'legacy_123/figB_A_conditions.csv')
base=pd.read_csv(root/'legacy_123/figB_baselines_wide.csv').query("domain=='A'")
pairs=a.merge(base,on='term',validate='one_to_one')
for c in ['bge_m3_top1','reranker_base_top1']:
    diff=pairs[c]-pairs.HC_majority
    print('Legacy',len(pairs),c,'McNemar p',binomtest(int((diff>0).sum()),int((diff!=0).sum())).pvalue)
