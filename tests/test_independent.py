"""Independent high-precision, volume, and neural-gradient checks."""
import numpy as np
import mpmath as mp
import pytest
from training_order.core import (
    symmetric_problem, family_constants, family_static,
    family_constructive, risk, transition,
)

@pytest.mark.parametrize('deg,eps,T', [(20,0,5),(45,0,12),(70,0,20),
    (20,.01,5),(45,1e-5,12),(70,.1,20)])
def test_mpmath_closed_forms(deg,eps,T):
    """Matrix exponential checked independently at 70 decimal digits."""
    with mp.workdps(70):
        phi=mp.mpf(deg)*mp.pi/180; c=mp.cos(phi)
        a=(1+c)/2; b=(1-c)/2; tau=mp.log((1+c)/c)
        va=mp.matrix([1,0]); vb=mp.matrix([mp.cos(phi),mp.sin(phi)])
        u=mp.matrix([mp.cos(phi/2),mp.sin(phi/2)])
        v=mp.matrix([-mp.sin(phi/2),mp.cos(phi/2)])
        A=va*va.T; B=vb*vb.T; S=u*u.T+mp.mpf(eps)*(v*v.T)
        E=mp.expm(-T*(A+B)/2)
        F=mp.expm(-(T-tau)*B)*mp.expm(-tau*A)
        tr=lambda M: sum(M[i,i] for i in range(2))
        exact_static=tr(E*S*E.T)/2
        exact_cur=tr(F*S*F.T)/2
        formula_static=(mp.exp(-2*a*T)+mp.mpf(eps)*mp.exp(-2*b*T))/2
        formula_cur=((a/c**2+mp.mpf(eps)*b*(1+2*c)**2/c**2)*mp.exp(-2*T)+mp.mpf(eps)*c**2/a)/2
        assert abs(exact_static-formula_static) < mp.mpf('1e-60')
        assert abs(exact_cur-formula_cur) < mp.mpf('1e-60')
        assert np.isclose(float(exact_cur),float(family_constructive(T,float(phi),eps)),rtol=1e-11,atol=1e-25)

@pytest.mark.parametrize('seed',range(8))
def test_arbitrary_stage_volume_bound(seed):
    rng=np.random.default_rng(seed+710)
    angle=rng.uniform(.2,1.3);eps=10**rng.uniform(-4,0);rho=.05
    p=symmetric_problem(angle,eps,rho)
    ts=rng.uniform(.01,2,7);ps=rng.uniform(size=7);T=sum(ts)
    val=risk(p,zip(ts,ps))
    lower=np.sqrt(eps)*np.exp(-(1+2*rho)*T)
    assert val >= lower-1e-13
    Phi=transition(p,zip(ts,ps))
    assert np.isclose(np.linalg.det(Phi),np.exp(-(1+2*rho)*T),rtol=1e-5,atol=1e-14)

@pytest.mark.parametrize('deg',[25,45,70])
def test_peak_asymptotic_coefficient(deg):
    phi=np.deg2rad(deg);k=family_constants(phi);eps=1e-16
    ae=k['A']+eps*k['D']
    x=k['a']*eps*k['F']/(k['b']*ae);T=-np.log(x)/2
    observed=family_static(T,phi,eps)/family_constructive(T,phi,eps)
    constant=k['a']**k['a']*k['b']**k['b']*k['A']**(-k['c'])
    predicted=constant*eps**(-k['b'])
    # The smaller term is slow to vanish at small source angles.
    assert abs(observed/predicted-1) < .02

def test_manual_tanh_gradient():
    """Finite differences validate the backprop equations used by the stress test."""
    rng=np.random.default_rng(192);n,d,h=7,2,4
    X=rng.normal(size=(n,d));y=rng.normal(size=n)
    W=rng.normal(scale=.3,size=(h,d));b=rng.normal(scale=.1,size=h)
    v=rng.normal(scale=.2,size=h);c=.15
    z=np.tanh(X@W.T+b);err=z@v+c-y
    dz=err[:,None]*v[None,:]*(1-z*z)
    analytic=np.r_[((dz.T@X)/n).ravel(),dz.mean(0),z.T@err/n,err.mean()]
    params=np.r_[W.ravel(),b,v,c]
    def loss(q):
        ww=q[:h*d].reshape(h,d);bb=q[h*d:h*d+h];vv=q[h*d+h:-1]
        return np.mean((np.tanh(X@ww.T+bb)@vv+q[-1]-y)**2)/2
    numeric=np.empty_like(params);delta=1e-6
    for j in range(len(params)):
        qp=params.copy();qm=params.copy();qp[j]+=delta;qm[j]-=delta
        numeric[j]=(loss(qp)-loss(qm))/(2*delta)
    assert np.allclose(analytic,numeric,atol=1e-9,rtol=1e-6)
