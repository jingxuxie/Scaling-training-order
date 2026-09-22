"""Deterministic CPU experiments. PYTHONPATH=src python experiments/run.py.
CSV files contain scientific results; runtime metadata is kept separately.
"""
from __future__ import annotations
import argparse,csv,json,sys,time,platform
from pathlib import Path
import numpy as np
import scipy
from scipy.optimize import minimize_scalar
from training_order.core import *
from training_order.certify import *
from training_order.spectral import *
ROOT=Path(__file__).resolve().parents[1]

def write_csv(out,name,rows):
    if not rows:raise RuntimeError(f'No rows for {name}')
    out.mkdir(parents=True,exist_ok=True)
    with (out/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(f'{name}: {len(rows)} rows',flush=True)

def exact(out):
    rows=[]
    for deg in [15,30,45,60,75]:
      angle=np.deg2rad(deg);k=family_constants(angle)
      for eps in [0.,1e-6,1e-4,1e-2,1.]:
       for ridge in [0.,.05]:
        for T in np.linspace(max(1,k['tau']),16,61):
          stat=float(family_static(T,angle,eps,ridge));cur=float(family_constructive(T,angle,eps,ridge))
          rows.append(dict(angle_deg=deg,epsilon=eps,ridge=ridge,budget=float(T),static=stat,
            constructive=cur,best_of_static_constructive=min(stat,cur),gain=stat/cur,
            epsilon_threshold=family_epsilon_threshold(T,angle)))
    write_csv(out,'family_curves.csv',rows)
    rows=[]
    for deg in [10,25,45,65,80]:
     angle=np.deg2rad(deg)
     for eps in [0.,.001,.1,1.]:
      for ridge in [0.,.07]:
       p=symmetric_problem(angle,eps,ridge)
       for T in [.1,1.,4.,8.,16.]:
        truth=float(family_static(T,angle,eps,ridge));vals=[risk(p,[(T,x)]) for x in np.linspace(0,1,101)]
        rows.append(dict(angle_deg=deg,epsilon=eps,ridge=ridge,budget=T,analytic=truth,
          grid_min=min(vals),minimum_margin=min(vals)-truth,grid_points=101))
    write_csv(out,'static_audit.csv',rows)
    rows=[];r=np.random.default_rng(4401)
    for case in range(160):
      d=[2,4,8,12][case%4];x=r.normal(size=(d,d));a=x@x.T/d;x=r.normal(size=(d,d));b=x@x.T/d
      p=Problem(a,b,np.eye(d));ts=r.uniform(.02,3,size=2);ps=r.uniform(size=2)
      cur=risk(p,zip(ts,ps));stat=risk(p,[(sum(ts),ts@ps/sum(ts))])
      rows.append(dict(case=case,dimension=d,budget=sum(ts),curriculum=cur,matched_static=stat,
        margin=cur-stat,commutator_norm=np.linalg.norm(a@b-b@a,2)))
    write_csv(out,'isotropic_audit.csv',rows)
    rows=[]
    for deg in [15,30,45,60,75]:
      angle=np.deg2rad(deg);k=family_constants(angle)
      for eps in np.logspace(-12,-2,11):
        objective=lambda T:-float(family_static(T,angle,eps)/family_constructive(T,angle,eps))
        grid=np.linspace(k['tau'],60,1001);ys=np.array([objective(t) for t in grid]);i=int(ys.argmin())
        opt=minimize_scalar(objective,bounds=(grid[max(0,i-1)],grid[min(i+1,1000)]),method='bounded')
        const=k['a']**k['a']*k['b']**k['b']*k['A']**(-k['c'])
        rows.append(dict(angle_deg=deg,epsilon=float(eps),peak_budget=float(opt.x),peak_gain=float(-opt.fun),
          predicted_gain=const*eps**(-k['b']),predicted_exponent=k['b']))
    write_csv(out,'uncertainty_peak.csv',rows)
    rows=[]
    for deg in [15,45,75]:
     k=family_constants(np.deg2rad(deg))
     for loss in np.logspace(-3,-12,19):
      stat=np.log(.5/loss)/(2*k['a']);cur=np.log(.5*k['A']/loss)/2
      rows.append(dict(angle_deg=deg,target_loss=float(loss),static_work=float(stat),constructive_work=float(cur),
        work_ratio=float(stat/cur),limiting_work_ratio=1/k['a']))
    write_csv(out,'compute_to_target.csv',rows)

def robustness(out):
    r=np.random.default_rng(4402);rows=[];p0=symmetric_problem(np.pi/4);k=family_constants(np.pi/4)
    for T in [4.,8.]:
     for scale in [0.,.001,.01,.05,.1]:
      for seed in range(24):
       angles=r.normal(scale=scale,size=2)
       va=np.array([np.cos(angles[0]),np.sin(angles[0])]);vb=np.array([np.cos(np.pi/4+angles[1]),np.sin(np.pi/4+angles[1])])
       initangle=np.pi/8+r.normal(scale=scale);e=np.array([np.cos(initangle),np.sin(initangle)])
       p=Problem(np.outer(va,va),np.outer(vb,vb),np.outer(e,e));stat,weight=static_grid(p,T,points=257)
       cur=risk(p,[(k['tau'],1),(T-k['tau'],0)])
       dg=max(np.linalg.norm(p.A-p0.A,2),np.linalg.norm(p.B-p0.B,2));ds=np.linalg.norm(p.S-p0.S,'nuc')
       radius=generic_risk_radius(T,dg,1.,ds);gap=float(family_static(T,np.pi/4)-family_constructive(T,np.pi/4))
       rows.append(dict(budget=T,perturbation_scale=scale,seed=seed,static_numeric=stat,static_weight=weight,
         constructive=cur,gain=stat/cur,generator_error=dg,covariance_error=ds,uniform_radius=radius,
         coarse_certificate=int(gap>2*radius)))
    write_csv(out,'perturbed_geometry.csv',rows)
    rows=[];r=np.random.default_rng(4403)
    for case in range(400):
      angle=r.uniform(.1,1.5);eps=10**r.uniform(-6,0);ridge=r.uniform(0,.2)
      T=r.uniform(.1,25);t=r.uniform(0,T);p1,p2=r.uniform(size=2);p=symmetric_problem(angle,eps,ridge)
      cur=risk(p,[(t,p1),(T-t,p2)]);lower=family_two_phase_lower(T,angle,eps,ridge)
      rows.append(dict(case=case,angle=float(angle),epsilon=float(eps),ridge=float(ridge),budget=T,
        phase1_time=t,p1=float(p1),p2=float(p2),risk=cur,lower_bound=lower,margin=cur-lower))
    write_csv(out,'two_phase_bound_audit.csv',rows)
    rows=[]
    for T in [2.,4.,8.]:
     p=symmetric_problem(ridge=.05)
     for h in [.2,.1,.05,.025,.0125]:
      K=round(T/h);switch=round(family_constants(np.pi/4)['tau']/h)
      phi=np.linalg.matrix_power(np.eye(2)-h*p.B,K-switch)@np.linalg.matrix_power(np.eye(2)-h*p.A,switch)
      exactphi=transition(p,[(switch*h,1),((K-switch)*h,0)])
      bound=T*h*max(np.linalg.norm(p.A,2),np.linalg.norm(p.B,2))**2/2
      rows.append(dict(budget=T,step_size=h,steps=K,operator_error=np.linalg.norm(phi-exactphi,2),error_bound=bound))
    write_csv(out,'discretization.csv',rows)

def spectral(out):
    rows=[]
    for mass in [.5,1.,2.]:
     for ridge in [0.,.1]:
      p=symmetric_problem(np.pi/4,0,ridge);scales,weights,tail=spectrum(1.,mass,2048)
      for T in [4.,16.,64.,256.]:
       st=block_risk(p,scales,weights,T,.5,.5,.5);cur,pars=schedule_candidate(p,scales,weights,T)
       rows.append(dict(mass_exponent=mass,spectral_exponent=1.,ridge=ridge,blocks=2048,budget=T,
         static=st+.5*tail,curriculum_candidate=cur+.5*tail,gain=(st+.5*tail)/(cur+.5*tail),
         p1=pars[0],p2=pars[1],fraction=pars[2],capacity_tail=.5*tail))
    write_csv(out,'spectral_profiles.csv',rows)
    rows=[]
    for alpha,beta in [(1.,1.),(1.,2.),(2.,1.)]:
      k=family_constants(np.pi/4)
      for C in np.logspace(2,7,26):
        opts=[]
        for J in np.unique(np.rint(np.geomspace(2,32768,97)).astype(int)):
          scales,w,tail=spectrum(alpha,beta,int(J));T=C/J
          val=.5*float(w@np.exp(-2*k['a']*T*scales))+.5*tail;opts.append((val,J,T))
        val,J,T=min(opts)
        rows.append(dict(spectral_exponent=alpha,mass_exponent=beta,model_work=float(C),capacity=int(J),
          training_work=float(T),risk=val,predicted_slope=-beta/(alpha+1)))
    write_csv(out,'capacity_compute.csv',rows)

def pilots(out):
    rows=[];r=np.random.default_rng(4404);angle=np.pi/4;base=symmetric_problem(angle);k=family_constants(angle)
    tasks=np.linspace(-.06,.06,9);vectors=[np.array([np.cos(angle/2+o),np.sin(angle/2+o)]) for o in tasks]
    stat={};env_full={T:StaticEnvelope.build(base,T,2048 if T==4. else 8192) for T in [4.,8.]}
    for T in [4.,8.]:
      for task,e in enumerate(vectors):stat[T,task]=static_grid(Problem(base.A,base.B,np.outer(e,e)),T,1025)[0]
    for T in [4.,8.]:
     for sigma in [.05,.2]:
      for n in [64,256,1024]:
       rad=gaussian_mean_radius(sigma,n,2,.05)
       for price in [1e-4,1e-3]:
        pilot_noise=r.normal(scale=sigma/np.sqrt(n),size=(90,2))
        for planning_cost in [0.,.02]:
         cost=2*n*price+planning_cost;remaining=T-cost;env_remaining=StaticEnvelope.build(base,remaining,512)
         matrices=[transition(base,[(k['tau'],1),(remaining-k['tau'],0)]),
                   transition(base,[(k['tau'],0),(remaining-k['tau'],1)])]
         for trial in range(90):
           task=trial%len(tasks);e=vectors[task];ehat=e+pilot_noise[trial]
           upper=[candidate_upper(phi,ehat,rad) for phi in matrices];which=int(np.argmin(upper))
           lower=env_full[T].lower(ehat,rad);accept=upper[which]<lower
           fallback_phi=flow(base.mixture(env_remaining.selected_p(ehat)),remaining)
           fallback=.5*float(np.linalg.norm(fallback_phi@e)**2);cur=.5*float(np.linalg.norm(matrices[which]@e)**2)
           chosen=cur if accept else fallback;oracle=stat[T,task]
           rows.append(dict(total_budget=T,sigma=sigma,n_per_coordinate=n,query_price=price,query_cost=2*n*price,planning_cost=planning_cost,pilot_cost=cost,
             remaining_budget=remaining,trial=trial,task=task,offset=float(tasks[task]),
             state_error=float(np.linalg.norm(ehat-e)),state_radius=rad,coverage=int(np.linalg.norm(ehat-e)<=rad),
             full_static_lower=lower,candidate_upper=upper[which],accepted=int(accept),
             false_accept=int(accept and cur>oracle+1e-11),candidate_risk=cur,fallback_risk=fallback,
             selected_risk=chosen,full_static_numeric=oracle,net_gain=oracle/chosen))
    write_csv(out,'pilot_trials.csv',rows)

def finite_data(out):
    rows=[];eta=.025;batch=32
    for deg in [45,70]:
     angle=np.deg2rad(deg);k=family_constants(angle);base=symmetric_problem(angle,ridge=.02)
     theta=-np.array([np.cos(angle/2),np.sin(angle/2)]);ca=np.linalg.cholesky(base.A);cb=np.linalg.cholesky(base.B)
     for noise in [0.,.1]:
      for n in [128,1024]:
       for T in [4.,8.]:
        steps=round(T/eta);switch=round(k['tau']/eta);fraction=switch/steps
        methods=[f'static_{p:.1f}' for p in np.linspace(0,1,11)]+['prepare_AB','prepare_BA','matched_static','greedy_oracle'];M=len(methods)
        for seed in range(20):
          rng=np.random.default_rng(100000+seed+1000*deg+10000*int(noise>0)+n)
          xa=rng.normal(size=(n,2))@ca.T;xb=rng.normal(size=(n,2))@cb.T
          ya=xa@theta+rng.normal(scale=noise,size=n);yb=xb@theta+rng.normal(scale=noise,size=n);th=np.zeros((M,2))
          for step in range(steps):
            ps=np.r_[np.linspace(0,1,11),float(step<switch),float(step>=switch),fraction,0.]
            err=th[-1]-theta;ps[-1]=float(err@base.A@err > err@base.B@err)
            idx=rng.integers(0,n,size=batch);coin=rng.random(batch);mask=coin[None,:]<ps[:,None]
            xx=np.where(mask[:,:,None],xa[idx][None,:,:],xb[idx][None,:,:]);yy=np.where(mask,ya[idx][None,:],yb[idx][None,:])
            er=np.einsum('mbd,md->mb',xx,th)-yy;th-=eta*np.einsum('mb,mbd->md',er,xx)/batch
          risks=np.sum((th-theta)**2,axis=1)/2;best=float(min(risks[:11]))
          for method,val in zip(methods,risks):
            rows.append(dict(angle_deg=deg,label_noise=noise,samples_per_source=n,budget=T,step_size=eta,
              batch_size=batch,steps=steps,seed=seed,method=method,target_risk=float(val),
              hindsight_static_grid=best,gain_vs_hindsight_static=best/val))
    write_csv(out,'finite_data_sgd.csv',rows)

def nonlinear(out):
    """Both hidden and output weights learn. Not a random-feature experiment."""
    rows=[];angle=np.pi/4;k=family_constants(angle);base=symmetric_problem(angle,ridge=.05)
    theta=-np.array([np.cos(angle/2),np.sin(angle/2)]);width=16;batch=32;eta=.02;n=512
    ca=np.linalg.cholesky(base.A);cb=np.linalg.cholesky(base.B)
    rng=np.random.default_rng(4405);xt=rng.normal(size=(4096,2));yt=xt@theta
    for noise in [0.,.1]:
     for steps,proxy_budget in [(200,4.),(800,8.)]:
      switch=round(steps*k['tau']/proxy_budget);fraction=switch/steps
      methods=['static_0.00','static_0.25','static_0.50','static_0.75','static_1.00','prepare_AB','prepare_BA','matched_static'];M=len(methods)
      for seed in range(12):
        rng=np.random.default_rng(200000+seed+1000*int(noise>0));xa=rng.normal(size=(n,2))@ca.T;xb=rng.normal(size=(n,2))@cb.T
        ya=xa@theta+rng.normal(scale=noise,size=n);yb=xb@theta+rng.normal(scale=noise,size=n)
        W=np.repeat(rng.normal(scale=.5,size=(width,2))[None,:,:],M,axis=0)
        b=np.zeros((M,width));v=np.zeros((M,width));c=np.zeros(M)
        for step in range(steps):
          ps=np.r_[np.linspace(0,1,5),float(step<switch),float(step>=switch),fraction]
          idx=rng.integers(0,n,batch);coin=rng.random(batch);mask=coin[None,:]<ps[:,None]
          xx=np.where(mask[:,:,None],xa[idx][None,:,:],xb[idx][None,:,:]);yy=np.where(mask,ya[idx][None,:],yb[idx][None,:])
          h=np.tanh(np.einsum('mbd,mhd->mbh',xx,W)+b[:,None,:]);err=np.einsum('mbh,mh->mb',h,v)+c[:,None]-yy
          dv=np.einsum('mb,mbh->mh',err,h)/batch;dc=err.mean(axis=1);dh=err[:,:,None]*v[:,None,:]*(1-h*h)
          dW=np.einsum('mbh,mbd->mhd',dh,xx)/batch;db=dh.mean(axis=1)
          W-=eta*dW;b-=eta*db;v-=eta*dv;c-=eta*dc
        testh=np.tanh(np.einsum('bd,mhd->mbh',xt,W)+b[:,None,:]);pred=np.einsum('mbh,mh->mb',testh,v)+c[:,None]
        rr=np.mean((pred-yt[None,:])**2,axis=1)/2;best=float(min(rr[:5]))
        for method,val in zip(methods,rr):
          rows.append(dict(label_noise=noise,steps=steps,examples_processed=steps*batch,width=width,
            step_size=eta,samples_per_source=n,seed=seed,method=method,test_risk=float(val),
            hindsight_static_grid=best,gain_vs_hindsight_static=best/val))
    write_csv(out,'nonlinear_stress.csv',rows)

SUITES={'exact':exact,'robustness':robustness,'spectral':spectral,'pilots':pilots,'finite_data':finite_data,'nonlinear':nonlinear}
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=ROOT/'results');ap.add_argument('--suite',choices=['all',*SUITES],default='all');args=ap.parse_args()
    suites=SUITES if args.suite=='all' else {args.suite:SUITES[args.suite]};elapsed={}
    for name,fn in suites.items():
        t=time.perf_counter();fn(args.out);elapsed[name]=time.perf_counter()-t;print(f'{name}: {elapsed[name]:.2f}s',flush=True)
    (args.out/f'environment_{args.suite}.json').write_text(json.dumps(dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,platform=platform.platform(),suite_seconds=elapsed),indent=2)+'\n')
if __name__=='__main__':main()
