"""Recompute manuscript numbers and paired bootstrap intervals from raw CSVs."""
from pathlib import Path
import csv,json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]

def main(root=ROOT):
    result=root/'results';rng=np.random.default_rng(91021)
    summary={'scientific_csv_rows':{p.name:len(pd.read_csv(p)) for p in sorted(result.glob('*.csv')) if not p.name.endswith('_summary.csv')}}
    outputs={}
    for name,keys,method in [
        ('finite_data_sgd',['angle_deg','label_noise','samples_per_source','budget'],'prepare_AB'),
        ('nonlinear_stress',['label_noise','steps'],'prepare_AB')]:
        df=pd.read_csv(result/f'{name}.csv');df=df[df.method==method];rows=[]
        for key,g in df.groupby(keys):
            values=g.gain_vs_hindsight_static.to_numpy()
            indices=rng.integers(0,len(values),size=(4000,len(values)))
            boots=np.median(values[indices],axis=1)
            key=key if isinstance(key,tuple) else (key,)
            rows.append(dict(zip(keys,key))|dict(seeds=len(values),median_gain=float(np.median(values)),
                ci_low=float(np.quantile(boots,.025)),ci_high=float(np.quantile(boots,.975)),wins=int(sum(values>1))))
        pd.DataFrame(rows).to_csv(result/f'{name}_summary.csv',index=False)
        outputs[name]=rows
        summary[name]={'runs':len(df),'wins':int(sum(df.gain_vs_hindsight_static>1)),'median_gain':float(df.gain_vs_hindsight_static.median())}
    pilot=pd.read_csv(result/'pilot_trials.csv')
    summary['pilots']={}
    for cost,g in pilot.groupby('planning_cost'):
        acc=g[g.accepted==1]
        summary['pilots'][str(cost)]={'trials':len(g),'coverage':int(g.coverage.sum()),'accepts':len(acc),
            'false_accepts':int(g.false_accept.sum()),'median_selected_gain':float(g.net_gain.median()),
            'accepted_median_gain':float(acc.net_gain.median()) if len(acc) else None,
            'accepted_min_gain':float(acc.net_gain.min()) if len(acc) else None,
            'accepted_max_gain':float(acc.net_gain.max()) if len(acc) else None}
    df=pd.read_csv(result/'capacity_compute.csv');summary['capacity_slopes']=[]
    for (alpha,beta),g in df.groupby(['spectral_exponent','mass_exponent']):
        g=g.sort_values('model_work').tail(10)
        summary['capacity_slopes'].append({'alpha':alpha,'beta':beta,'observed':float(np.polyfit(np.log(g.model_work),np.log(g.risk),1)[0]),'predicted':-beta/(alpha+1)})
    sp=pd.read_csv(result/'spectral_profiles.csv');summary['shared_spectral']={'max_candidate_gain':float(sp.gain.max()),'mean_candidate_gain':float(sp.gain.mean())}
    pert=pd.read_csv(result/'perturbed_geometry.csv');prows=[]
    for (budget,scale),g in pert.groupby(['budget','perturbation_scale']):
        prows.append(dict(budget=budget,scale=scale,median_gain=float(g.gain.median()),wins=int(sum(g.gain>1)),cases=len(g)))
    pd.DataFrame(prows).to_csv(result/'perturbation_summary.csv',index=False)
    (result/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (result/'statistical_protocol.md').write_text('Paired ratios use the best static grid separately on each test seed. This is a hindsight benchmark, not a deployable selection rule. Intervals are percentile bootstrap intervals for the median across independent training seeds, with 4,000 resamples, seed 91021. They are descriptive, unadjusted for multiple comparisons, and not confidence bounds for generalization to other tasks. Different methods share random initialization and data draws within a seed. Pilot cases are controlled simulations, not independent real-world deployment tests.\n')
    print(json.dumps(summary,indent=2))
    return summary
if __name__=='__main__':main()
