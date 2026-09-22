"""Exact expected minibatch-SGD risk for fresh Gaussian linear-regression data.

Both independent per-example source sampling and whole-minibatch source sampling
are supported. Source probabilities are precommitted and independent of data and
initial-error draws. Labels share one linear target with independent Gaussian
noise. The two sampling conventions have different fourth-moment operators.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .core import _psd

_BASIS = np.array([[[1.,0.],[0.,0.]], [[0.,2**-.5],[2**-.5,0.]], [[0.,0.],[0.,1.]]])
_TRACE = np.array([1.,0.,1.])


def vectorize(s):
    s = np.asarray(s, dtype=float)
    return np.stack([s[...,0,0], np.sqrt(2)*s[...,0,1], s[...,1,1]],axis=-1)


def unvectorize(v):
    return np.einsum('...k,kij->...ij', np.asarray(v), _BASIS)


def batch_power(a, n: int):
    if not isinstance(n, (int, np.integer)) or n < 0:
        raise ValueError('Exponent must be a nonnegative integer')
    a = np.asarray(a, dtype=float)
    out = np.broadcast_to(np.eye(a.shape[-1]),a.shape).copy()
    while n:
        if n & 1:
            out = out @ a
        a = a @ a
        n //= 2
    return out


@dataclass(frozen=True)
class MomentOperator:
    matrix: np.ndarray
    injection: np.ndarray
    trace_matrix: np.ndarray
    covariance: np.ndarray
    step: float
    batch: int

    @property
    def affine(self):
        a = np.eye(4)
        a[:3,:3] = self.matrix
        a[:3,3] = self.injection
        return a


def gaussian_operator(h, step: float, batch: int = 1, label_sigma: float = 0.) -> MomentOperator:
    h = _psd(h,'source covariance')
    if h.shape != (2,2) or step <= 0 or not isinstance(batch,(int,np.integer)) or batch < 1 or label_sigma < 0:
        raise ValueError('Require a 2x2 PSD covariance, positive step/batch, nonnegative noise')
    if not np.isfinite([step,label_sigma]).all():
        raise ValueError('Step and noise must be finite')
    def apply(s):
        return (s-step*(h@s+s@h)+step**2*((1+1/batch)*h@s@h+np.trace(h@s)*h/batch))
    t = np.column_stack([vectorize(apply(s)) for s in _BASIS])
    q = step**2*label_sigma**2*h/batch
    c = np.eye(2)-2*step*h+step**2*((1+1/batch)*h@h+np.trace(h)*h/batch)
    return MomentOperator(t, vectorize(q), c, h, step, batch)


def mixture_coefficients(oa: MomentOperator, ob: MomentOperator, sampling: str = 'batch'):
    """Affine moment map coefficients F(p)=F0+p*F1+p^2*F2.

    'example' means domains are sampled independently for every minibatch
    example. 'batch' means one domain is selected for the entire update.
    """
    if sampling not in ['batch','example']:
        raise ValueError('Sampling must be batch or example')
    if oa.step != ob.step or oa.batch != ob.batch:
        raise ValueError('Source step sizes and minibatch sizes must match')
    second = np.zeros((4,4))
    if sampling == 'example':
        dh=oa.covariance-ob.covariance
        second[:3,:3]=oa.step**2*(1-1/oa.batch)*np.column_stack(
            [vectorize(dh@s@dh) for s in _BASIS])
    return ob.affine,oa.affine-ob.affine-second,second


def mixture_affine(oa,ob,p,sampling='batch'):
    f0,f1,f2=mixture_coefficients(oa,ob,sampling)
    p=np.asarray(p)
    return f0+p[...,None,None]*f1+p[...,None,None]**2*f2


def moment_risk(oa: MomentOperator, ob: MomentOperator, second, schedule, sampling='batch'):
    state = np.r_[vectorize(_psd(second,'second moment')),1.]
    for steps,p in schedule:
        if not isinstance(steps,(int,np.integer)) or steps < 0 or not 0 <= p <= 1:
            raise ValueError('Invalid discrete schedule')
        state = batch_power(mixture_affine(oa,ob,p,sampling),steps) @ state
    return float(.5*(_TRACE @ state[:3]))


def static_envelope(oa: MomentOperator, ob: MomentOperator, second,
                    steps: int, points: int = 1025, sampling='batch'):
    """Global static bracket from a curvature-bounded grid (real-arithmetic theorem).

    Both batch-source and example-source sampling are supported. The numerical
    implementation is float64, not outward-rounded interval arithmetic.
    """
    if steps < 1 or points < 3:
        raise ValueError('Require positive steps and at least three grid points')
    s = _psd(second,'second moment'); tr_s = np.trace(s)
    ps = np.linspace(0.,1.,points)
    f0,f1,f2=mixture_coefficients(oa,ob,sampling)
    maps=f0+ps[:,None,None]*f1+ps[:,None,None]**2*f2
    state = np.r_[vectorize(s),1.]
    terminal = batch_power(maps,steps) @ state
    values = .5*(terminal[:,:3] @ _TRACE)
    derivatives=f1[None,:3,:3]+2*ps[:,None,None]*f2[None,:3,:3]
    dn=np.sqrt(2)*np.linalg.norm(derivatives,ord=2,axis=(-2,-1))
    delta=np.maximum(dn[:-1],dn[1:])
    delta2=2*np.sqrt(2)*np.linalg.norm(f2[:3,:3],2)
    trace_vectors=np.einsum('j,njk->nk',_TRACE,maps[:,:3,:3])
    rs=np.linalg.eigvalsh(unvectorize(trace_vectors))[:,-1]
    if rs.max() > 1+1e-10:
        raise ValueError('Curvature shortcut requires trace-nonexpansive mean-square maps')
    r=np.minimum(1.,np.maximum(rs[:-1],rs[1:]))
    curvature=.5*tr_s*(steps*(steps-1)*delta**2*r**max(steps-2,0)
                       +steps*delta2*r**(steps-1))
    qa,qb=unvectorize(oa.injection),unvectorize(ob.injection)
    qnorms=ps*np.trace(qa)+(1-ps)*np.trace(qb)
    qmax=np.maximum(qnorms[:-1],qnorms[1:]);dq=np.abs(np.linalg.eigvalsh(qa-qb)).sum()
    gap=np.maximum(1-r,np.finfo(float).tiny)
    with np.errstate(over='ignore',divide='ignore'):
        sum1=np.minimum(steps*(steps-1)/2,1/gap**2)
        sum2=np.minimum(steps*(steps-1)*(steps-2)/3,2/gap**3) if steps>=2 else np.zeros_like(r)
    curvature += .5*qmax*(delta**2*sum2+delta2*sum1)+delta*dq*sum1
    lower=max(0.,float(np.min(np.minimum(values[:-1],values[1:])-curvature/(8*(points-1)**2))))
    i=int(np.argmin(values))
    return dict(lower=lower,upper=float(values[i]),p=float(ps[i]),points=points,
                width=float(values[i]-lower),curvature_max=float(curvature.max()))


def best_pure_two_phase(oa: MomentOperator, ob: MomentOperator, second, steps: int):
    """Exact finite search over every integer switch time in both pure orders."""
    state = np.r_[vectorize(second),1.]
    pa = np.empty((steps+1,4,4));pb=pa.copy()
    pa[0]=np.eye(4);pb[0]=np.eye(4)
    aa,ab=oa.affine,ob.affine
    for k in range(steps):
        pa[k+1]=aa@pa[k];pb[k+1]=ab@pb[k]
    ab = (pb[::-1]@pa)@state
    ba = (pa[::-1]@pb)@state
    ra=.5*(ab[:,:3]@_TRACE);rb=.5*(ba[:,:3]@_TRACE)
    ia,ib=int(ra.argmin()),int(rb.argmin())
    if ra[ia] <= rb[ib]:
        return float(ra[ia]),[(ia,1.),(steps-ia,0.)]
    return float(rb[ib]),[(ib,0.),(steps-ib,1.)]
