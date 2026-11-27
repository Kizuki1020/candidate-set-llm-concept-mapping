"""Paired sensitivity to unique proper prefixes of reconstructed candidate strings."""
import csv
import json
import sys
from pathlib import Path

import pandas as pd
from scipy.stats import binomtest

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root/'code'))
from utils import exact_match, load_synonyms, parse_answer
from config import ANSWER_PREFIX, ANSWER_PREFIX_B

out = root/'composition_audit'
allowed = pd.read_csv(out/'allowed_strings.csv', dtype=str, keep_default_na=False)
members = allowed.groupby(['domain','record_id','condition']).allowed_string.agg(list)
exception_rows = []
summary = []
for domain in ('A','B'):
    df = pd.read_csv(root/f'results/exp1/exp1_{domain}.csv', dtype={'id':str}, keep_default_na=False)
    assert len(df)==1800 and not df.duplicated(['id','condition','rep']).any()
    synonyms = load_synonyms(root/'data/synonyms_A.json') if domain=='A' else {}
    prefix = ANSWER_PREFIX if domain=='A' else ANSWER_PREFIX_B
    df['completed_score'] = df.correct
    affected = set()
    for r in df.itertuples():
        assert int(exact_match(parse_answer(r.raw_output,prefix),r.gold,synonyms)) == r.correct
        if r.condition == 'UC':
            continue
        pool = members.loc[(domain,r.id,r.condition)]
        if r.raw_output in pool:
            continue
        matches = [s for s in pool if r.raw_output and s.startswith(r.raw_output)]
        assert len(matches)==1
        new = int(exact_match(parse_answer(matches[0],prefix),r.gold,synonyms))
        df.at[r.Index,'completed_score'] = new
        affected.add(r.id)
        exception_rows.append([domain,r.id,r.condition,r.rep,r.raw_output,matches[0],r.gold,r.correct,new])
    for rule,frame,col in [('frozen',df,'correct'),('complete_unique_prefix',df,'completed_score'),
                           ('exclude_any_affected_record',df[~df.id.isin(affected)],'correct')]:
        groups = frame.groupby(['id','condition'])[col]
        assert groups.size().eq(3).all()
        scores = groups.sum().ge(2).unstack().astype(int)
        assert scores.notna().all().all()
        for lo in ('HC','SC'):
            plus = int(((scores.RO==1)&(scores[lo]==0)).sum())
            minus = int(((scores.RO==0)&(scores[lo]==1)).sum())
            summary.append(dict(domain=domain,rule=rule,n=len(scores),affected_records=len(affected),
                                correct={k:int(v) for k,v in scores.sum().items()},contrast='RO-'+lo,
                                ro_only=plus,comparator_only=minus,difference=(plus-minus)/len(scores),
                                exact_p=binomtest(plus,plus+minus).pvalue if plus+minus else 1.0))
assert len(exception_rows)==24
with (out/'prefix_runs.csv').open('w',newline='') as f:
    w=csv.writer(f)
    w.writerow(['domain','id','condition','rep','raw_output','unique_completion','reference','frozen_score','completion_score'])
    w.writerows(exception_rows)
(out/'prefix_sensitivity.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
