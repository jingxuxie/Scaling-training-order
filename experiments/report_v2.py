"""Summarize extension experiments and draw publication-style figures."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from figure_style import (apply_style, save_figure, legend_above,
                          reference_line, COLORS, EPSILON_COLORS, LINESTYLES, epsilon_label)

def summarize(results):
    results=Path(results);out={}
    x=pd.read_csv(results/'gaussian_sgd_moments.csv')
    for name,g in x.groupby('sampling'):
        out[name]={'cases':len(g),'gain_over_1_01':int((g.certified_gain>1.01).sum()),
          'median_candidate_gain':float(g.candidate_gain.median()),'max_candidate_gain':float(g.candidate_gain.max()),
          'median_preparation_gain':float(g.prep_gain.median()),'max_relative_static_bracket':float(g.relative_bracket_width.max())}
    group=x.groupby(['sampling','batch_size','budget']).agg(cases=('case','size'),
        median_candidate_gain=('candidate_gain','median'),median_preparation_gain=('prep_gain','median'),
        improvements_over_one_percent=('certified_gain',lambda a:int((a>1.01).sum()))).reset_index()
    group.to_csv(results/'moment_condition_summary.csv',index=False,float_format='%.17g')
    paper=results.parent/'paper'
    if paper.is_dir():
        lines=[r'\begin{table}[h]\centering\small',
          r'\caption{Complete aggregated Gaussian SGD conditions. Each row includes 48 settings. Median gains use the upper static bracket; $>1\%$ counts use the lower bracket.}',
          r'\begin{tabular}{llrrrr}\toprule Sampling & Batch & $B$ & Median candidate & Median flow rule & $>1\%$ gains\\\midrule']
        for _,row in group.iterrows():
            line=f"{row.sampling} & {int(row.batch_size)} & {row.budget:g} & {row.median_candidate_gain:.4f} & {row.median_preparation_gain:.4f} & {int(row.improvements_over_one_percent)} "
            lines.append(line+r'\\')
        lines.append(r'\bottomrule\end{tabular}\end{table}')
        (paper/'moment_table.tex').write_text('\n'.join(lines)+'\n')

    mc=pd.read_csv(results/'gaussian_sgd_monte_carlo.csv')
    out['monte_carlo']={'conditions':len(mc),'replicates_per_condition':int(mc.repetitions.min()),
        'within_three_se':int(mc.inside_three_se.sum()),'max_abs_z':float(mc.z_score.abs().max())}
    for stem in ['arbitrary_schedule_audit','boundary_high_precision','near_boundary_gain','diffusion_audit','global_boundary_curves','diffusion_curves']:
        a=pd.read_csv(results/(stem+'.csv'));out[stem]={'rows':len(a)}
        for col in ['violation','violates','prediction_matches','predicted_sign_matches','feasible','expected_sign']:
            if col in a:out[stem][col]=int(a[col].sum())
    (results/'extension_summary.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
    return out

def main(results,figures,figures_only=False):
    import matplotlib.pyplot as plt
    apply_style()
    results,figures=Path(results),Path(figures);figures.mkdir(parents=True,exist_ok=True)
    out=None if figures_only else summarize(results)
    def save(fig,name):
        save_figure(fig,figures,name)
    x=pd.read_csv(results/'global_boundary_curves.csv');fig,ax=plt.subplots(figsize=(6.0,3.1))
    for index,(eps,g) in enumerate(x[x.angle_deg==45].groupby('epsilon')):
        ax.plot(g.budget,g.gain,color=EPSILON_COLORS[eps],linestyle=LINESTYLES[index],label=epsilon_label(eps))
        if eps>0:
            stop=np.log(1/eps)/(2*np.cos(np.pi/4))
            ax.plot([stop],[1],color=EPSILON_COLORS[eps],marker='x',markersize=6,markeredgewidth=1.3,linestyle='none',zorder=4)
    ax.set(xlabel=r'Training work $B$',ylabel='Best-static loss / candidate loss',yscale='log',ylim=(.88,35))
    reference_line(ax);legend_above(ax,3);save(fig,'global_boundary')
    x=pd.read_csv(results/'near_boundary_gain.csv');print('Near-boundary columns:',x.columns.tolist())
    # Column names are kept explicit to fail rather than silently plot the wrong quantity.
    fig,ax=plt.subplots(figsize=(6.,2.7))
    for index,(angle,g) in enumerate(x[x.budget==4].groupby('angle_deg')):
        ax.semilogx(g.xi,g.witness_gain_coefficient,marker=('o','s','^')[index],linestyle=LINESTYLES[index],label=fr'$\phi={angle:g}^\circ$')
    reference_line(ax,.125,label=r'Predicted limit $1/8$')
    ax.set(xlabel=r'Relative distance below boundary $\xi$',ylabel=r'$(\mathrm{gain}-1)/\xi^2$')
    legend_above(ax,2);save(fig,'boundary_coefficient')
    x=pd.read_csv(results/'moment_condition_summary.csv');fig,ax=plt.subplots(figsize=(6.,2.9))
    for (sampling,batch),g in x.groupby(['sampling','batch_size']):
        if batch==1:continue
        ax.plot(g.budget,g.median_candidate_gain,color={8:COLORS[0],64:COLORS[1]}[batch],
                marker='o' if sampling=='example' else 's',linestyle='-' if sampling=='example' else '--',
                label=f'Per-{sampling}, batch {batch}')
    ax.set(xlabel=r'Training work $B$ = steps $\times$ step size',ylabel='Median best-static / candidate loss',xticks=[1,4,8])
    reference_line(ax);legend_above(ax,2);save(fig,'sgd_moment_comparison')
    x=pd.read_csv(results/'diffusion_curves.csv');print('Diffusion columns:',x.columns.tolist())
    fig,ax=plt.subplots(figsize=(6.,2.9))
    for index,(sigma,g) in enumerate(x.groupby('sigma')):
        ax.semilogy(g.budget,g.global_gain_ceiling,linestyle=LINESTYLES[index],label=fr'$\sigma={sigma:g}$')
    ax.set(xlabel=r'Training work $B$',ylabel='Upper bound on any schedule gain',ylim=(.9,100))
    reference_line(ax);legend_above(ax,4);save(fig,'diffusion_ceiling')
    print('Wrote four extension figures.' if figures_only else 'Wrote four extension figures and summary tables.')
    return out
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,default=Path('results'));p.add_argument('--figures',type=Path,default=Path('paper/figures'))
    p.add_argument('--figures-only',action='store_true',help='Read saved summaries without rewriting results or tables.')
    args=p.parse_args();main(args.results,args.figures,args.figures_only)
