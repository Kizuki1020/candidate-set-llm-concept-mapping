"""Run from any directory with python3 path/to/verification/audit_candidate_composition.py.
Uses the archived inference code on CPU. Reconstructs a single uninterrupted pass in
JSON record order. CSV members are sorted for comparison, not in decoding order.
"""
from pathlib import Path
import csv
import json
import re
import sys
from collections import Counter
import numpy as np
import pandas as pd
from scipy import stats
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root/'code'))
from exp1_constraint_decoding import build_pools, build_sets, load_dataset_a, load_dataset_b, SEED_BASE
out = root/'composition_audit'
out.mkdir(exist_ok=True)
a,b = load_dataset_a(),load_dataset_b()
gold_a,gold_b,sc_a,sc_b = build_pools(a,b)
report = {}
length_rows = []
raw_checks = []
mass_rows = []
regression_inputs = []
with (out/'allowed_strings.csv').open('w',encoding='utf-8',newline='') as stream:
    writer=csv.writer(stream)
    writer.writerow(['domain','record_id','condition','allowed_string'])
    for dom,records,ro_pool,sc_pool in [('A',a,gold_b,sc_a),('B',b,gold_a,sc_b)]:
        rng=np.random.RandomState(SEED_BASE)
        df=pd.read_csv(root/f'results/exp1/exp1_{dom}.csv',dtype={'id':str},keep_default_na=False)
        reps=df.groupby(['id','condition']).correct
        assert reps.size().eq(3).all() and reps.nunique().eq(1).all()
        ref_scores=reps.first().unstack()
        mass=pd.read_csv(root/f'results/exp2/mass_{dom}.csv',dtype={'id':str}).set_index('id').loc[ref_scores.index].norm_mass
        for hi,lo in [('HC','UC'),('RO','HC')]:
            gain=ref_scores[hi]-ref_scores[lo]
            result=stats.pearsonr(mass,gain)
            mass_rows.append(dict(domain=dom,contrast=hi+'-'+lo,n=len(gain),r=float(result.statistic),p=float(result.pvalue)))
        regression_inputs.extend(zip(mass.to_numpy(), (ref_scores.RO-ref_scores.HC).to_numpy(), [int(dom=='B')]*len(mass)))
        report[dom]={'record_condition_cells':len(reps),'records':len(records)}
        for r in records:
            rid,g=r['id'],r['gold'];hc=list(dict.fromkeys(r['candidates']+[g]))
            ro,sc=build_sets(r,len(r['candidates']),ro_pool,sc_pool,rng)
            for cond,ls in [('HC',hc),('RO',ro),('SC',sc)]:
                assert len(set(ls))==len(hc) and g in ls
                for s in sorted(ls):writer.writerow([dom,rid,cond,s])
                distractors=[len(s) for s in ls if s!=g]
                length_rows.append(dict(domain=dom,record_id=rid,condition=cond,k=len(ls),reference_length=len(g),median_distractor_length=np.median(distractors),ratio=len(g)/np.median(distractors),longest_including_ties=int(len(g)>=max(distractors)),strictly_longest=int(len(g)>max(distractors))))
                for row in df[(df.id==rid)&(df.condition==cond)].itertuples():
                    raw_checks.append(dict(domain=dom,record_id=rid,condition=cond,rep=row.rep,raw_strict_prefix_matches=sum(z.startswith(row.raw_output) and z!=row.raw_output for z in ls) if row.raw_output not in ls and row.raw_output else 0,raw_in_set=row.raw_output in ls,parsed_in_set=row.pred in ls))
length=pd.DataFrame(length_rows);length.to_csv(out/'length_by_record.csv',index=False)
summary=length.groupby(['domain','condition']).agg(n=('record_id','count'),median_ratio=('ratio','median'),mean_ratio=('ratio','mean'),n_longest_including_ties=('longest_including_ties','sum'),n_strictly_longest=('strictly_longest','sum'),n_at_least_median=('ratio',lambda s:int(sum(s>=1))),n_above_median=('ratio',lambda s:int(sum(s>1)))).reset_index()
report['length_summary']=summary.to_dict('records')
c=pd.DataFrame(raw_checks)
c.to_csv(out/'outputs_vs_reconstructed_membership.csv',index=False)
report['membership_matches']=c.groupby(['domain','condition']).agg(outputs=('rep','count'),raw_matches=('raw_in_set','sum'),parsed_matches=('parsed_in_set','sum')).reset_index().to_dict('records')
report['mass_vs_gain']=mass_rows
report['exact_missing']=int((~c.raw_in_set).sum())
report['unique_prefix_among_missing']=int(((~c.raw_in_set)&c.raw_strict_prefix_matches.eq(1)).sum())
x,y,domain=np.array(regression_inputs).T
X=np.column_stack([np.ones(len(x)),x,domain,x*domain])
beta=np.linalg.lstsq(X,y,rcond=None)[0]
resid=y-X@beta
se=np.sqrt(np.diag(np.linalg.inv(X.T@X))*(resid@resid)/(len(x)-4))
report['RO_HC_OLS']=[dict(term=k,estimate=float(v),se=float(e),p=float(2*stats.t.sf(abs(v/e),len(x)-4))) for k,v,e in zip(['intercept','mass','domain_B','mass_by_B'],beta,se)]
ad=json.loads((root/'data/dataset_a_with_candidates.json').read_text())
prov=json.loads((root/'DATA1/ontology_v2_provenance.json').read_text())
links=[(m,l['concept'],l['source']) for m,ls in prov['msc_link_source'].items() for l in ls]
golds={re.sub(r'[\u200b\s]+',' ',r['gold']).strip() for r in a}
report['resource']=dict(concept_sources=Counter(prov['concept_source'].values()),link_records=Counter(s for m,c,s in links),distinct_edges=len({(m,c) for m,c,s in links}),probe_reference_source=Counter(prov['concept_source'].get(g,'unassigned') for g in golds),permutation_audit=ad['diagnostics'])
core=pd.read_csv(root/'diagnostic/references/core_828.csv',dtype=str,keep_default_na=False)
report['overlap']={}
for target,coredata,probedata in [('all',core,ad['test_cases']),('L3',core[core.l_class=='L3'],[r for r in ad['test_cases'] if r['L_Class']=='L3'])]:
    ids=set(coredata.term_id);terms=set(coredata.chinese_term);keys=set(zip(coredata.chinese_term,coredata.definition_original))
    report['overlap'][target]=dict(n=len(probedata),id=sum(r['protege_id'] in ids for r in probedata),term=sum(r['name'] in terms for r in probedata),term_definition=sum((r['name'],r['original_definition']) in keys for r in probedata))
(out/'audit.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
print(json.dumps(report,indent=2,ensure_ascii=False))
