"""Block-spectrum risks with ONE shared schedule for all blocks."""
from __future__ import annotations
import numpy as np
from scipy.special import zeta
from scipy.optimize import minimize
from .core import Problem


def block_risk(problem: Problem, scales: np.ndarray, weights: np.ndarray,
               budget: float, p1: float, p2: float, fraction: float) -> float:
    if not 0 <= fraction <= 1 or budget < 0:
        raise ValueError("Invalid duration fraction or budget")
    t1 = budget*fraction*scales; t2 = budget*(1-fraction)*scales
    w1,v1 = np.linalg.eigh(problem.mixture(p1)); w2,v2 = np.linalg.eigh(problem.mixture(p2))
    if np.allclose(problem.Q, np.eye(problem.Q.shape[0])):
        # Eight scalar exponentials per mode in the two-dimensional setting.
        # This is algebraically identical to the matrix product, not interpolation.
        overlap=v2.T@v1; rotated=v1.T@problem.S@v1
        rates=[]; coefficients=[]
        for a in range(len(w2)):
            for b in range(len(w1)):
                for c in range(len(w1)):
                    rates.append(budget*(2*(1-fraction)*w2[a]+fraction*(w1[b]+w1[c])))
                    coefficients.append(.5*overlap[a,b]*overlap[a,c]*rotated[b,c])
        decays=np.exp(-np.asarray(rates)[:,None]*scales[None,:])@weights
        return max(0.,float(np.asarray(coefficients)@decays))
    f1 = np.einsum('ik,nk,jk->nij',v1,np.exp(-t1[:,None]*np.maximum(w1,0)),v1)
    f2 = np.einsum('ik,nk,jk->nij',v2,np.exp(-t2[:,None]*np.maximum(w2,0)),v2)
    phi = f2@f1
    r = .5*np.einsum('ij,njk,kl,nil->n',problem.Q,phi,problem.S,phi)
    return float(weights@r)


def spectrum(spectral_exponent: float, mass_exponent: float, blocks: int):
    if spectral_exponent <= 0 or mass_exponent <= 0 or blocks < 1:
        raise ValueError("Positive spectral/mass exponents and capacity required")
    j=np.arange(1,blocks+1,dtype=float)
    normalization=zeta(1+mass_exponent)
    return j**(-spectral_exponent), j**(-1-mass_exponent)/normalization, float(
        zeta(1+mass_exponent,blocks+1)/normalization)


def schedule_candidate(problem: Problem, scales, weights, budget: float):
    """Multi-start local candidate; explicitly includes the analytic static baseline.

This is an upper bound on optimized curriculum risk, not a global optimizer.
"""
    objective=lambda x: block_risk(problem,scales,weights,budget,*x)
    seeds=[(.5,.5,.5),(1.,0.,.15),(1.,0.,.5),(.9,.1,.3),(.8,.1,.7),(.9,.3,.5)]
    all_results=[]
    for x in seeds:
        f0=objective(x);all_results.append((f0,tuple(x)))
        sol=minimize(objective,x,method='L-BFGS-B',bounds=[(0,1)]*3,
                     options={'ftol':1e-11,'gtol':1e-10,'maxiter':80,'maxfun':500})
        all_results.append((float(sol.fun),tuple(float(y) for y in sol.x)))
    return min(all_results,key=lambda v:v[0])
