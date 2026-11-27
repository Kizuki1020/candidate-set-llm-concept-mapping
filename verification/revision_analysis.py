"""Run with python revision_analysis.py PATH_TO_EXTRACTED_V2 OUTPUT_DIRECTORY.
Inputs are frozen predictions and resources. No model calls are made.
"""
from pathlib import Path
from itertools import product
import sys, json
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
root, out = map(Path, sys.argv[1:3])
out.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(42)
contrasts = [('RO','HC'),('HC','UC'),('RO','SC'),('SC','UC'),('RO','UC'),('SC','HC')]
rows, samples, domains, audit = [], {}, {}, {}
for domain, filename, idcol in [('A','dataset_a_with_candidates','protege_id'),('B','dataset_b','record_id')]:
    cases = pd.DataFrame(json.loads((root/'data'/f'{filename}.json').read_text())['test_cases']).set_index(idcol)
    df = pd.read_csv(root/f'results/exp1/exp1_{domain}.csv', keep_default_na=False)
    assert len(df)==1800 and not df.duplicated(['id','condition','rep']).any()
    by_record=df.groupby(['id','condition']).correct
    assert by_record.size().eq(3).all() and by_record.nunique().eq(1).all()
    scores=by_record.mean().unstack().astype(int).sort_index()
    assert len(scores)==150 and set(scores.index)==set(cases.index)
    m = pd.read_csv(root/f'results/exp2/mass_{domain}.csv').set_index('id').loc[scores.index]
    samples[domain]=scores
    for hi,lo in contrasts:
        diff=(scores[hi]-scores[lo]).to_numpy()
        plus=int(sum(diff==1)); minus=int(sum(diff==-1))
        ci=np.quantile(rng.choice(diff,(10000,len(diff))).mean(axis=1),[.025,.975])
        rows.append(dict(domain=domain,contrast=f'{hi} − {lo}',gain=float(diff.mean()),lo=ci[0],hi=ci[1],plus=plus,minus=minus,p=stats.binomtest(plus,plus+minus).pvalue))
    if domain=='A':
        idx=cases.modern_definition.str.contains('无现代|无对应'); labels=cases.loc[idx,'modern_definition']
        audit['marker_records']=scores.loc[labels.index].assign(label=labels).reset_index().to_dict('records')
    x=m.norm_mass.to_numpy(); y=(scores.HC-scores.UC).to_numpy()
    ii=rng.integers(0,len(x),(10000,len(x)))
    xp=x[ii];yp=y[ii];xc=xp-xp.mean(axis=1)[:,None];yc=yp-yp.mean(axis=1)[:,None]
    slopes=(xc*yc).sum(axis=1)/(xc**2).sum(axis=1)
    cor=(xc*yc).sum(axis=1)/np.sqrt((xc**2).sum(axis=1)*(yc**2).sum(axis=1))
    domains[domain]=(x,y,slopes)
    pear=stats.pearsonr(x,y); spe=stats.spearmanr(x,y)
    audit[domain]=dict(mean=float(x.mean()),sd=float(x.std(ddof=1)),median=float(np.median(x)),r=float(pear.statistic),p=float(pear.pvalue),rho=float(spe.statistic),corr_ci=np.quantile(cor,[.025,.975]).tolist(),slope_ci=np.quantile(slopes,[.025,.975]).tolist(),rank_desc=[float(q) for q in m.gold_rank_in_C.quantile([0,.25,.5,.75,1])])
ordered=sorted(rows,key=lambda r:r['p'])
prev=0
for i,row in enumerate(ordered):
    prev=max(prev,min(1.,row['p']*(len(rows)-i)));row['holm']=prev
audit['contrasts']=rows
audit['between']=[]
for hi,lo in [contrasts[1],contrasts[0]]:
    a,b=[(s[hi]-s[lo]).to_numpy() for s in samples.values()]
    test=stats.mannwhitneyu(b,a,alternative='two-sided')
    audit['between'].append([float(a.mean()),float(a.std(ddof=1)),float(b.mean()),float(b.std(ddof=1)),float(test.pvalue),float(2*test.statistic/(len(a)*len(b))-1)])
x=np.concatenate([d[0] for d in domains.values()]); y=np.concatenate([d[1] for d in domains.values()])
b=np.repeat([0,1],150); design=np.column_stack([np.ones(300),x,b,b*x])
beta=np.linalg.lstsq(design,y,rcond=None)[0];resid=y-design@beta
se=np.sqrt(np.diag(np.linalg.inv(design.T@design))*(resid@resid)/296)
audit['ols']=np.column_stack([beta,se,beta/se,2*stats.t.sf(abs(beta/se),296)]).tolist()
audit['r2']=float(1-resid@resid/np.sum((y-y.mean())**2))
(out/'revision_checks.json').write_text(json.dumps(audit,indent=2,ensure_ascii=False))
fig,axes=plt.subplots(1,2,figsize=(8.3,3.4),gridspec_kw={'width_ratios':[1.65,1]})
colors=['#166a69','#415b92']
for (domain,(x,y,slopes)),color in zip(domains.items(),colors):
    shift={'A':-.022,'B':.022}[domain];ax=axes[0]
    ax.scatter(x,y+shift,s=10,alpha=.38,color=color,label='Domain '+domain,linewidths=0)
    slope,intercept=np.polyfit(x,y,1);xx=np.array([x.min(),x.max()]);ax.plot(xx,slope*xx+intercept,color=color,lw=1.4)
axes[0].legend(frameon=False,fontsize=8);axes[0].set(xlabel='Normalized log mass',ylabel='HC − UC correctness',yticks=[-1,0,1],title='a   Record-level gains')
axes[1].axvline(0,color='#92969a',lw=.8,ls='--')
for i,(name,values) in enumerate([('Domain A',domains['A'][2]),('Domain B',domains['B'][2]),('B minus A',domains['B'][2]-domains['A'][2])]):
    lo,hi=np.quantile(values,[.025,.975]);point=[beta[1],beta[1]+beta[3],beta[3]][i]
    axes[1].plot([lo,hi],[2-i]*2,color='#31566c');axes[1].plot(point,2-i,'o',color='#31566c')
axes[1].set(yticks=[2,1,0],yticklabels=['Domain A','Domain B','B minus A'],ylim=(-.6,2.6),xlabel='Slope per log-mass unit',title='b   Slopes and 95% intervals')
for ax in axes:ax.spines[['top','right']].set_visible(False);ax.title.set_fontsize(10);ax.tick_params(labelsize=8)
fig.tight_layout();fig.savefig(out/'Figure5.png',dpi=450,bbox_inches='tight');plt.close(fig)
