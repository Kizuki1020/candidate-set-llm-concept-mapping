"""Diagnostic figure from released commercial records and explicitly labelled Qwen summaries."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import binomtest

root = Path(__file__).resolve().parents[1]
out = root/'verification'
archive = pd.read_csv(root/'diagnostic/archived_summaries/fig2_model_diagnostics.csv').set_index('model')
keys = ['chinese_term','l_class','gold_msc','gold_label','gold_label_ontology']
records, stats = [], []
for model, tag in [('DeepSeek-Chat','deepseek'),('GPT-4o-mini','gpt4omini'),('Claude Haiku 4.5','claude'),('Qwen3-32B','qwen')]:
    row = archive.loc[model]
    if tag == 'qwen':
        counts = [k for k in range(182) if round(100*k/181,1)==row.L3_overalignment_pct]
        assert len(counts)==1
        k = counts[0]
        conf_none,conf_code = row.L3_correct_NONE_mean_conf,row.L3_OA_mean_conf
        source = 'archived summary'
    else:
        data = pd.read_csv(root/f'diagnostic/outputs/commercial_{tag}_828.csv',keep_default_na=False)
        subset = data.loc[data.l_class.eq('L3')].copy()
        assert len(data)==828 and len(subset)==181 and subset.gold_label_ontology.eq('NONE').all()
        subset[tag] = subset.selected_code.ne('')
        k = int(subset[tag].sum())
        conf_none = subset.loc[~subset[tag],'confidence'].mean()
        conf_code = subset.loc[subset[tag],'confidence'].mean()
        subset['occurrence'] = subset.groupby(keys,sort=False).cumcount()
        records.append(subset[keys+['occurrence',tag]])
        source = 'record-level outputs'
    assert round(100*k/181,1)==row.L3_overalignment_pct
    ci = binomtest(k,181).proportion_ci(method='wilson')
    stats.append([model,source,k,181-k,k/181,ci.low,ci.high,conf_none,conf_code])
df = pd.DataFrame(stats,columns=['model','source','n_code','n_none','code_rate','ci_low','ci_high','mean_conf_none','mean_conf_code'])
df.to_csv(out/'figure2_source.csv',index=False)
joint = records[0]
for r in records[1:]:
    joint = joint.merge(r,on=keys+['occurrence'],validate='one_to_one')
assert len(joint)==181
order = ['deepseek','gpt4omini','claude']
counts = joint.groupby(order).size().sort_values(ascending=True)
counts.to_csv(out/'figure2_overlap.csv',header=['n'])
f,(a,b,c)=plt.subplots(1,3,figsize=(10.4,3.25),gridspec_kw={'width_ratios':[1.42,1.05,1.05]})
f.subplots_adjust(left=.13,right=.988,top=.81,bottom=.20,wspace=.66)
y=np.arange(len(df))
a.errorbar(df.code_rate,y,xerr=[df.code_rate-df.ci_low,df.ci_high-df.code_rate],fmt='o',color='#246276',capsize=3)
a.set_yticks(y,df.model);a.invert_yaxis();a.set_xlim(.75,1);a.set_xlabel('Proportion (95% Wilson interval)')
for i,(v,k) in enumerate(zip(df.code_rate,df.n_code)):
    a.annotate(f'{k}/181',(v,i),xytext=(0,7),textcoords='offset points',ha='center',fontsize=7)
for i,row in df.iterrows():b.plot([row.mean_conf_code,row.mean_conf_none],[i,i],color='.65',lw=1)
b.scatter(df.mean_conf_code,y,marker='o',color='#246276',label='Code');b.scatter(df.mean_conf_none,y,marker='s',color='#ad6322',label='NONE');b.set_yticks(y,[]);b.invert_yaxis();b.set_xlim(68,100);b.set_xlabel('Mean confidence (0–100)');b.legend(loc='lower center',bbox_to_anchor=(.5,-.39),ncol=2,frameon=False,fontsize=8)
patterns=[''.join('1' if x else '0' for x in k) for k in counts.index]
c.barh(np.arange(len(counts)),counts,color='#246276');c.set_yticks(np.arange(len(counts)),patterns);c.set_xlabel('Records');c.set_xlim(0,max(counts)*1.18)
for i,v in enumerate(counts):c.text(v+1,i,str(v),va='center',fontsize=8)
for ax,title in [(a,'a   Specific-code rate'),(b,'b   Confidence by outcome'),(c,'c   Commercial model overlap')]:
    ax.set_title(title,loc='left',fontsize=10,fontweight='bold',pad=12)
    ax.spines[['right','top']].set_visible(False);ax.tick_params(labelsize=8)
f.savefig(out/'Figure2.png',dpi=350,bbox_inches='tight',facecolor='white')
