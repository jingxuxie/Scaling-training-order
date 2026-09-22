"""Generate separate, reproducible matplotlib figures from saved scientific data."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from figure_style import (apply_style, save_figure, legend_above,
                          reference_line, EPSILON_COLORS, LINESTYLES, epsilon_label)

ROOT=Path(__file__).resolve().parents[1]

def main(results=ROOT/'results', figures=ROOT/'paper'/'figures'):
    apply_style()
    results=Path(results);figures=Path(figures);figures.mkdir(parents=True,exist_ok=True)
    def save(fig,name):
        save_figure(fig,figures,name)
    df=pd.read_csv(results/'family_curves.csv')
    df=df[(df.angle_deg==45)&(df.ridge==0)]
    fig,ax=plt.subplots(figsize=(6.0,3.35))
    for index,eps in enumerate([0,1e-6,1e-4,1e-2]):
        g=df[df.epsilon==eps].sort_values('budget')
        ax.semilogy(g.budget,g.gain,color=EPSILON_COLORS[eps],linestyle=LINESTYLES[index],label=epsilon_label(eps))
    reference_line(ax)
    plotted=df[df.epsilon.isin([0,1e-6,1e-4,1e-2])]
    ax.set(xlabel=r'Training work $B$',ylabel='Best-static loss / preparation loss',
           ylim=(plotted.gain.min()*.65,plotted.gain.max()*1.5))
    legend_above(ax,4);save(fig,'uncertainty_window')

    df=pd.read_csv(results/'uncertainty_peak.csv')
    fig,ax=plt.subplots(figsize=(6.0,3.2))
    for index,deg in enumerate([30,45,60,75]):
        g=df[df.angle_deg==deg].sort_values('epsilon')
        ax.loglog(g.epsilon,g.peak_gain,marker=('o','s','^','D')[index],markevery=3,
                  linestyle=LINESTYLES[index],label=fr'$\phi={deg}^\circ$')
    ax.set(xlabel=r'Orthogonal initial-error mass $\varepsilon$',ylabel='Peak gain of explicit preparation')
    legend_above(ax,4);save(fig,'peak_gain')

    df=pd.read_csv(results/'compute_to_target.csv');fig,ax=plt.subplots(figsize=(6.0,3.15))
    for index,(deg,g) in enumerate(df.groupby('angle_deg')):
        g=g.sort_values('target_loss');ax.semilogx(g.target_loss,g.work_ratio,linestyle=LINESTYLES[index],label=fr'$\phi={deg:g}^\circ$')
    ax.set(xlabel='Target loss',ylabel='Static work / preparation work')
    legend_above(ax,4);ax.invert_xaxis();save(fig,'work_savings')

    df=pd.read_csv(results/'capacity_compute.csv');fig,ax=plt.subplots(figsize=(6.0,3.2))
    for index,((alpha,beta),g) in enumerate(df.groupby(['spectral_exponent','mass_exponent'])):
        ax.loglog(g.model_work,g.risk,marker=('o','s','^','D')[index],markevery=2,
                  linestyle=LINESTYLES[index],label=fr'$\alpha={alpha:g},\ \beta={beta:g}$; slope ${-beta/(alpha+1):.3g}$')
    ax.set(xlabel=r'Total model work $C=JB$',ylabel='Optimized static risk, including tail')
    legend_above(ax,2);save(fig,'capacity_scaling')

    df=pd.read_csv(results/'finite_data_sgd_summary.csv')
    keys=['angle_deg','label_noise','samples_per_source']
    groups=list(df.groupby(keys));fig,ax=plt.subplots(figsize=(6.0,3.8))
    for T,shift,marker in [(4,-.1,'o'),(8,.1,'s')]:
        rows=pd.DataFrame([g[g.budget==T].iloc[0] for _,g in groups]);v=rows.median_gain.to_numpy()
        ax.errorbar(np.arange(len(groups))+shift,v,yerr=np.vstack([v-rows.ci_low.to_numpy(),rows.ci_high.to_numpy()-v]),
                    fmt=marker,capsize=3,elinewidth=1.2,label=fr'$B={T}$')
    reference_line(ax);ax.set_yscale('log')
    ax.set_xticks(np.arange(len(groups)),[f'{int(k[0])}° / {k[1]:g} / {int(k[2])}' for k,_ in groups],rotation=35,ha='right',fontsize=8.5)
    ax.set(xlabel='Source angle / label noise / samples per source',ylabel='Median gain vs. hindsight static grid')
    legend_above(ax);save(fig,'finite_sample_gain')

    df=pd.read_csv(results/'nonlinear_stress_summary.csv');v=df.median_gain.to_numpy()
    fig,ax=plt.subplots(figsize=(6.0,3.1))
    ax.errorbar(np.arange(len(df)),v,yerr=np.vstack([v-df.ci_low.to_numpy(),df.ci_high.to_numpy()-v]),fmt='o',capsize=4,elinewidth=1.2)
    reference_line(ax)
    ax.set_xticks(np.arange(len(df)),[f'Noise {r.label_noise:g}\n{int(r.steps)} steps' for r in df.itertuples()])
    ax.set(ylabel='Median gain vs. hindsight static grid',ylim=(.65,1.15));ax.margins(x=.2);save(fig,'nonlinear_gain')

    df=pd.read_csv(results/'perturbation_summary.csv');fig,ax=plt.subplots(figsize=(6.0,3.15))
    for index,(T,g) in enumerate(df[df.scale>0].groupby('budget')):
        ax.loglog(g.scale,g.median_gain,marker=('o','s')[index],linestyle=LINESTYLES[index],label=fr'$B={T:g}$')
    reference_line(ax)
    ax.set(xlabel='Angle perturbation standard deviation (rad)',ylabel='Median gain vs. numeric best static')
    legend_above(ax);save(fig,'geometry_robustness')
    print(f'Wrote 7 separate figures (PDF and PNG) to {figures}')

if __name__=='__main__':main()
