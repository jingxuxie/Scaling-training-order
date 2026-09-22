"""New all-schedule, boundary, diffusion and exact mean-square SGD experiments.

Every condition grid below is fixed in code. No result is omitted based on its
performance. Monte Carlo conditions are fixed independently of the moment sweep.
"""
from __future__ import annotations
import argparse,csv,json,time,platform,sys
from pathlib import Path
import numpy as np
import mpmath as mp
from scipy.optimize import minimize_scalar
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from training_order.core import (symmetric_problem,family_constants,family_static,
    family_constructive,risk,transition,flow)
from training_order.global_bounds import *
from training_order.moments import *


def write(out,name,rows):
    out.mkdir(parents=True,exist_ok=True)
    with (out/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def arbitrary(out):
    rng=np.random.default_rng(921004)
    rows=[]
    for case in range(1200):
        angle=rng.uniform(.15,1.45);B=float(np.exp(rng.uniform(np.log(.1),np.log(15))))
        crit=global_epsilon_threshold(B,angle)
        eps=crit*float(np.exp(rng.uniform(np.log(.03),np.log(5))))
        rho=float(rng.choice([0.,.02,.1]));n=int(rng.choice([1,2,3,8,16,64]))
        p=symmetric_problem(angle,eps,rho)
        weights=rng.choice([0.,1.],n) if case%2==0 else rng.random(n)
        path=list(zip(B*rng.dirichlet(np.ones(n)),weights))
        phi=transition(p,path);rr=risk(p,path)
        v=np.array([-np.sin(angle/2),np.cos(angle/2)])
        slow=float(np.sum((phi@v)**2));sb=float(np.exp(-2*(rho+family_constants(angle)['b'])*B))
        lo=all_schedule_lower(B,angle,eps,rho);base=float(family_static(B,angle,eps,rho))
        rows.append(dict(case=case,angle=angle,budget=B,epsilon=eps,ridge=rho,stages=n,
            risk=rr,lower=lo,static=base,static_optimal=eps>=crit,slow_column=slow,
            slow_lower=sb,violation=bool(rr+1e-11*max(1,rr)<lo or slow+1e-11<sb)))
    write(out,'arbitrary_schedule_audit.csv',rows)


def _mp_problem(deg,B,eps,rho=0):
    angle=mp.mpf(deg)*mp.pi/180;c=mp.cos(angle);a=(1+c)/2;d=(1-c)/2;h=mp.sqrt(a*d)
    G=lambda z:mp.matrix([[a+rho,z],[z,d+rho]])
    S=mp.diag([1,eps]);base=(mp.exp(-2*(a+rho)*B)+eps*mp.exp(-2*(d+rho)*B))/2
    return c,a,d,h,G,S,base


def boundary(out):
    mp.mp.dps=80;rows=[]
    for deg in [15,30,45,65,80]:
      for B0 in [.1,.5,1.,4.,8.]:
       B=mp.mpf(str(B0))
       for factor0 in [.1,.5,.99,1.,1.01,2.]:
        factor=mp.mpf(str(factor0));c=mp.cos(mp.mpf(deg)*mp.pi/180);eps=factor*mp.exp(-2*c*B)
        c,a,d,h,G,S,base=_mp_problem(deg,B,eps)
        q=1/(2*mp.cosh(c*B/2)-1);eta=h*mp.mpf('0.00001')
        P=mp.expm(-B/2*G(-eta))*mp.expm(-B/2*G(q*eta))
        C=P*S*P.T;val=(C[0,0]+C[1,1])/2;gap=(base-val)/base
        expect=(gap>0) if factor<1 else (gap<=0)
        rows.append(dict(angle_deg=deg,budget=B0,epsilon_factor=factor0,epsilon=mp.nstr(eps,30),
            relative_improvement=mp.nstr(gap,40),expected_sign=bool(expect),precision_digits=80))
    write(out,'boundary_high_precision.csv',rows)


def near_boundary(out):
    mp.mp.dps=80;rows=[]
    for deg in [30,45,65]:
      for B0 in [1.,4.,8.]:
       B=mp.mpf(str(B0));c,a,d,h,G,S,_=_mp_problem(deg,B,mp.mpf(0))
       q=1/(2*mp.cosh(c*B/2)-1)
       def x1(t):
           if t<=B/2:
               integral=q*mp.expm1(c*t)/c
           else:
               integral=(q*mp.expm1(c*B/2)-(mp.exp(c*t)-mp.exp(c*B/2)))/c
           return -mp.exp(-a*t)*integral
       b=mp.exp(-2*d*B)
       K=2*x1(B)**2+2*c*b*mp.quad(lambda t:mp.exp(2*d*t)*x1(t)**2,[0,B/2,B])
       for xi0 in ['0.01','0.001','0.0001','0.00001','0.000001']:
        xi=mp.mpf(xi0);eps=mp.exp(-2*c*B)*(1-xi);eta=mp.sqrt(b*xi/(2*K))
        feasible=eta<=h
        S=mp.diag([1,eps]);base=(mp.exp(-2*a*B)+eps*mp.exp(-2*d*B))/2
        lower=mp.sqrt(eps)*mp.exp(-B)
        P=mp.expm(-B/2*G(-eta))*mp.expm(-B/2*G(q*eta))
        C=P*S*P.T;val=(C[0,0]+C[1,1])/2
        rows.append(dict(angle_deg=deg,budget=B0,xi=xi0,amplitude=float(eta),feasible=bool(feasible),
            witness_gain_coefficient=mp.nstr((base/val-1)/xi**2,32),
            global_upper_coefficient=mp.nstr((base/lower-1)/xi**2,32),
            prediction=.125))
    write(out,'near_boundary_gain.csv',rows)


def _ou_risk(prob,path,sigma):
    S=prob.S.copy();noise=np.zeros_like(S)
    for t,p in path:
        g=prob.mixture(p);lam,v=np.linalg.eigh(g);E=flow(g,t)
        integrals=np.empty_like(lam)
        pos=lam>1e-14
        integrals[pos]=-np.expm1(-2*lam[pos]*t)/(2*lam[pos]);integrals[~pos]=t
        q=sigma**2*(v*integrals)@v.T
        S=E@S@E.T+q;noise=E@noise@E.T+q
    return float(np.trace(S)/2),float(np.trace(noise)/2)


def diffusion(out):
    rng=np.random.default_rng(921005);rows=[]
    for case in range(480):
        angle=rng.uniform(.2,1.4);B=rng.uniform(.1,20);eps=float(rng.choice([0.,.001,.1,1.]))
        sigma=float(rng.choice([.01,.1,.3]));rho=.02;n=int(rng.choice([1,2,5,20]))
        p=symmetric_problem(angle,eps,rho);path=list(zip(B*rng.dirichlet(np.ones(n)),rng.random(n)))
        val,nval=_ou_risk(p,path,sigma);nl=balanced_noise_risk(B,angle,sigma,rho)
        rows.append(dict(case=case,angle=angle,budget=B,epsilon=eps,sigma=sigma,stages=n,
            risk=val,noise_risk=nval,noise_lower=nl,
            total_lower=all_schedule_lower(B,angle,eps,rho)+nl,
            violation=bool(nval+1e-11<nl or val+1e-11<all_schedule_lower(B,angle,eps,rho)+nl)))
    write(out,'diffusion_audit.csv',rows)
    curves=[]
    angle=np.pi/4;k=family_constants(angle)
    for sigma in [0.,.001,.01,.1]:
      for B in np.linspace(1.,24.,121):
        p=symmetric_problem(angle,0.,.02)
        path=[(k['tau'],1.),(B-k['tau'],0.)]
        val,_=_ou_risk(p,path,sigma)
        nl=balanced_noise_risk(B,angle,sigma,.02);base=float(family_static(B,angle,0.,.02))+nl
        curves.append(dict(sigma=sigma,budget=B,preparation_gain=base/val,
            global_gain_ceiling=noisy_gain_ceiling(B,angle,0.,sigma,.02)))
    write(out,'diffusion_curves.csv',curves)


def _moment_candidates(prob,oa,ob,K,B,angle,envelope,sampling="batch"):
    pure,purepath=best_pure_two_phase(oa,ob,prob.S,K)
    best=min(envelope['upper'],pure);path=[(K,envelope['p'])] if envelope['upper']<=pure else purepath
    method='static' if envelope['upper']<=pure else 'pure_two_phase'
    for amplitude in [.01,.05,.1,.25,.5,.75,1.]:
        continuous=boundary_witness(B,angle,amplitude)
        curr=[(K//2,continuous[0][1]),(K-K//2,continuous[1][1])]
        val=moment_risk(oa,ob,prob.S,curr,sampling)
        if val<best:best,path,method=val,curr,'mixed_two_phase'
    k=family_constants(angle);j=round(k['tau']/(B/K))
    prep=moment_risk(oa,ob,prob.S,[(j,1.),(K-j,0.)]) if j<=K else float('nan')
    return best,path,method,pure,prep


def moments(out):
    rows=[];case=0
    for sampling in ["example","batch"]:
     for deg in [30,45,65]:
      angle=np.deg2rad(deg)
      for B in [1.,4.,8.]:
       for step in [.01,.05]:
        K=round(B/step)
        for batch in [1,8,64]:
         for sigma in [0.,.1]:
          for eps in [0.,.0001,.01,.1]:
           p=symmetric_problem(angle,eps,.02)
           oa=gaussian_operator(p.A,step,batch,sigma);ob=gaussian_operator(p.B,step,batch,sigma)
           envelope=static_envelope(oa,ob,p.S,K,1025,sampling)
           best,path,method,pure,prep=_moment_candidates(p,oa,ob,K,B,angle,envelope,sampling)
           # Refine ambiguous or very wide brackets, without tuning the condition grid.
           for resolution in [4097,16385,65537]:
               if envelope['width'] <= 1e-3*envelope['upper']:
                   break
               envelope=static_envelope(oa,ob,p.S,K,resolution,sampling)
               best,path,method,pure,prep=_moment_candidates(p,oa,ob,K,B,angle,envelope,sampling)
           rows.append(dict(case=case,sampling=sampling,angle_deg=deg,budget=B,step_size=step,steps=K,batch_size=batch,
             label_sigma=sigma,epsilon=eps,ridge=.02,examples=K*batch,
             static_lower=envelope['lower'],static_upper=envelope['upper'],static_p=envelope['p'],
             grid_points=envelope['points'],relative_bracket_width=envelope['width']/envelope['upper'],
             candidate_risk=best,candidate_method=method,candidate_schedule=json.dumps(path),
             pure_two_phase_risk=pure,flow_preparation_risk=prep,
             certified_gain=envelope['lower']/best,candidate_gain=envelope['upper']/best,
             prep_gain=envelope['upper']/prep if np.isfinite(prep) else float('nan')))
           case+=1
    write(out,'gaussian_sgd_moments.csv',rows)


def monte_carlo(out):
    rows=[];n=8192;angle=np.pi/4;eps=.001;step=.05
    for sampling in ["example","batch"]:
     for B in [2.,4.]:
      for batch in [1,8]:
       for sigma in [0.,.1]:
        K=round(B/step);p=symmetric_problem(angle,eps,.02)
        oa=gaussian_operator(p.A,step,batch,sigma);ob=gaussian_operator(p.B,step,batch,sigma)
        env=static_envelope(oa,ob,p.S,K,4097,sampling)
        best,path,_,_,_=_moment_candidates(p,oa,ob,K,B,angle,env,sampling)
        methods={'static':[(K,env['p'])],'two_phase':path}
        for method,schedule in methods.items():
         # Independent replicate draws; same seed couples the two policies without
         # making their random source choices depend on the current error.
         seed=900000+1000*K+100*batch+int(sigma>0)
         rng=np.random.default_rng(seed)
         u=np.array([np.cos(angle/2),np.sin(angle/2)]);v=np.array([-u[1],u[0]])
         errors=u[None,:]+np.sqrt(eps)*rng.choice([-1.,1.],n)[:,None]*v
         ca=np.linalg.cholesky(p.A);cb=np.linalg.cholesky(p.B)
         probs=np.concatenate([np.repeat(frac,count) for count,frac in schedule])
         for frac in probs:
             mask=rng.random((n,batch))<frac if sampling=="example" else (rng.random(n)<frac)[:,None]
             standard=rng.normal(size=(n,batch,2))
             xa=standard@ca.T;xb=standard@cb.T
             x=np.where(mask[:,:,None],xa,xb)
             noise=rng.normal(0,sigma,(n,batch))
             residual=np.einsum('nbd,nd->nb',x,errors)+noise
             errors-=step*np.einsum('nbd,nb->nd',x,residual)/batch
         losses=np.sum(errors**2,axis=1)/2;mean=float(losses.mean());se=float(losses.std(ddof=1)/np.sqrt(n))
         exact=moment_risk(oa,ob,p.S,schedule,sampling)
         rows.append(dict(sampling=sampling,budget=B,batch_size=batch,label_sigma=sigma,method=method,
             repetitions=n,seed=seed,exact_risk=exact,monte_carlo_mean=mean,
             standard_error=se,z_score=(mean-exact)/se,inside_three_se=abs(mean-exact)<=3*se))
    write(out,'gaussian_sgd_monte_carlo.csv',rows)


def boundary_curves(out):
    rows=[]
    for deg in [30,45,65]:
     angle=np.deg2rad(deg)
     for eps in [0.,1e-4,.001,.01,.1]:
      for B in np.linspace(.2,12,100):
       p=symmetric_problem(angle,eps,.02);base=float(family_static(B,angle,eps,.02));k=family_constants(angle)
       def val(amplitude):return risk(p,boundary_witness(B,angle,max(amplitude,1e-12)))
       fit=minimize_scalar(val,bounds=(1e-12,1.),method='bounded',options={'xatol':1e-12})
       candidates=[base,val(1.),val(.01),fit.fun]
       if B>=k['tau']:candidates.append(float(family_constructive(B,angle,eps,.02)))
       best=min(candidates)
       lo=all_schedule_lower(B,angle,eps,.02)
       rows.append(dict(angle_deg=deg,epsilon=eps,budget=B,static=base,candidate=best,
           gain=base/best,global_gain_upper=base/lo,
           epsilon_critical=global_epsilon_threshold(B,angle)))
    write(out,'global_boundary_curves.csv',rows)

SUITES={'arbitrary':arbitrary,'boundary':boundary,'near_boundary':near_boundary,
        'diffusion':diffusion,'moments':moments,'monte_carlo':monte_carlo,'curves':boundary_curves}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=ROOT/'results')
    ap.add_argument('--suite',choices=['all',*SUITES],default='all');args=ap.parse_args()
    times={}
    for name,fn in (SUITES if args.suite=='all' else {args.suite:SUITES[args.suite]}).items():
        start=time.perf_counter();fn(args.out);times[name]=time.perf_counter()-start
        print(name,round(times[name],2),'seconds',flush=True)
    (args.out/f'environment_v2_{args.suite}.json').write_text(json.dumps(dict(suite_seconds=times,
        python=sys.version,numpy=np.__version__,platform=platform.platform()),indent=2)+'\n')
if __name__=='__main__':main()
