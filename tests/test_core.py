import numpy as np
import pytest
from scipy.linalg import expm
from training_order.core import *
from training_order.certify import *
from training_order.spectral import block_risk,spectrum


@pytest.mark.parametrize('angle',np.deg2rad([10,25,45,65,80]))
@pytest.mark.parametrize('epsilon',[0.,.001,.1,1.])
@pytest.mark.parametrize('ridge',[0.,.07])
def test_family_formulas_and_global_static(angle,epsilon,ridge):
    p=symmetric_problem(angle,epsilon,ridge); k=family_constants(angle)
    for t in [k['tau'],2*k['tau'],5.]:
        exact=float(family_static(t,angle,epsilon,ridge))
        assert risk(p,[(t,.5)]) == pytest.approx(exact,abs=2e-14,rel=2e-12)
        for weight in np.linspace(0,1,21):
            assert risk(p,[(t,weight)]) >= exact-2e-13
        actual=risk(p,[(k['tau'],1),(t-k['tau'],0)])
        assert actual == pytest.approx(float(family_constructive(t,angle,epsilon,ridge)),abs=2e-14,rel=2e-11)


@pytest.mark.parametrize('seed',range(16))
def test_golden_thompson_and_condition_bound(seed):
    r=np.random.default_rng(seed);d=2+seed%5
    x=r.normal(size=(d,d));a=x@x.T/d
    x=r.normal(size=(d,d));b=x@x.T/d
    ts=r.uniform(.02,2,size=2);ps=r.uniform(size=2)
    p=Problem(a,b,np.eye(d));T=sum(ts);pbar=ts@ps/T
    phi=transition(p,list(zip(ts,ps)))
    matched=risk(p,[(T,pbar)])
    assert matched <= risk(p,list(zip(ts,ps)))+1e-11
    x=r.normal(size=(d,d));s=x@x.T+.3*np.eye(d)
    x=r.normal(size=(d,d));q=x@x.T+.3*np.eye(d)
    p=Problem(a,b,s,q)
    assert risk(p,[(T,pbar)]) <= np.linalg.cond(s)*np.linalg.cond(q)*risk(p,list(zip(ts,ps)))+1e-10


@pytest.mark.parametrize('seed',range(12))
def test_commuting_and_flow_expm(seed):
    r=np.random.default_rng(seed);d=6
    v,_=np.linalg.qr(r.normal(size=(d,d)))
    a=(v*r.uniform(0,2,d))@v.T;b=(v*r.uniform(0,2,d))@v.T
    p=Problem(a,b,np.eye(d));ts=r.uniform(size=5);ps=r.uniform(size=5)
    assert np.allclose(transition(p,zip(ts,ps)),flow(p.mixture(ts@ps/sum(ts)),sum(ts)),atol=2e-13)
    assert np.allclose(flow(a,1.2),expm(-1.2*a),atol=2e-13)


@pytest.mark.parametrize('seed',range(16))
def test_positive_uncertainty_two_phase_bound(seed):
    r=np.random.default_rng(seed);angle=r.uniform(.05,1.5);eps=10**r.uniform(-5,0)
    ridge=r.uniform(0,.2);p=symmetric_problem(angle,eps,ridge)
    for _ in range(20):
        ts=r.uniform(0,5,size=2);ps=r.uniform(size=2)
        lo=family_two_phase_lower(sum(ts),angle,eps,ridge)
        assert risk(p,zip(ts,ps)) >= lo-1e-13


@pytest.mark.parametrize('angle',np.deg2rad([15,30,45,60,75]))
def test_uncertainty_threshold(angle):
    for T in np.linspace(1,12,25):
        e=family_epsilon_threshold(T,angle)
        if e>1e-15:
            assert 0<e<1
            for factor in [.8,1.2]:
                diff=float(family_static(T,angle,e*factor)-family_constructive(T,angle,e*factor))
                assert diff>0 if factor<1 else diff<0


@pytest.mark.parametrize('seed',range(12))
def test_uniform_perturbation(seed):
    r=np.random.default_rng(seed);d=3
    x=r.normal(size=(d,d));a=x@x.T/d
    x=r.normal(size=(d,d));b=x@x.T/d
    da=.02*np.outer(r.normal(size=d),r.normal(size=d));da=da@da.T
    db=.02*np.outer(r.normal(size=d),r.normal(size=d));db=db@db.T
    x=r.normal(size=(d,d));s=x@x.T/d
    ds=.001*np.eye(d)
    p=Problem(a,b,s);q=Problem(a+da,b+db,s+ds)
    ts=r.uniform(size=4);ps=r.uniform(size=4);T=sum(ts)
    radius=generic_risk_radius(T,max(np.linalg.norm(da,2),np.linalg.norm(db,2)),np.trace(s),np.linalg.norm(ds,'nuc'))
    assert abs(risk(p,zip(ts,ps))-risk(q,zip(ts,ps))) <= radius+1e-12


@pytest.mark.parametrize('seed',range(12))
def test_static_uniform_ball_envelope(seed):
    r=np.random.default_rng(seed)
    angle=r.uniform(.3,1.3);p=symmetric_problem(angle,ridge=.03)
    T=r.uniform(1,10);ehat=r.normal(size=2);rad=.05
    env=StaticEnvelope.build(p,T,cells=64)
    lo=env.lower(ehat,rad)
    for _ in range(100):
        v=r.normal(size=2);v*=rad*r.uniform()/np.linalg.norm(v)
        e=ehat+v;weight=r.uniform();f=flow(p.mixture(weight),T)
        assert .5*np.linalg.norm(f@e)**2 >= lo-1e-12
    phi=transition(p,[(.4*T,.9),(.6*T,.1)])
    hi=candidate_upper(phi,ehat,rad)
    assert .5*np.linalg.norm(phi@(ehat+v))**2 <= hi+1e-12


@pytest.mark.parametrize('seed',range(8))
def test_block_vectorization(seed):
    r=np.random.default_rng(seed);p=symmetric_problem(r.uniform(.2,1.2),.01,.03)
    scales=r.uniform(.01,2,20);weights=r.uniform(size=20);weights/=sum(weights)
    T=3.;p1,p2,f=r.uniform(size=3)
    v=block_risk(p,scales,weights,T,p1,p2,f)
    explicit=sum(w*risk(p,[(T*f*s,p1),(T*(1-f)*s,p2)]) for s,w in zip(scales,weights))
    assert v==pytest.approx(explicit,abs=2e-14)


def test_spectral_tail_normalization():
    for exponent in [.3,1,2.5]:
        _,w,tail=spectrum(1,exponent,100)
        assert sum(w)+tail==pytest.approx(1.,abs=1e-12)


def test_invalid_inputs():
    for a in [np.array([[1.,2.],[0,1]]),np.diag([1.,-1.]),np.ones((2,3))]:
        with pytest.raises(ValueError):flow(a,1.)
    with pytest.raises(ValueError):flow(np.eye(2),-1.)
    with pytest.raises(ValueError):symmetric_problem(0.)
    with pytest.raises(ValueError):symmetric_problem(np.pi/2)
    with pytest.raises(ValueError):family_constructive(0.,np.pi/4)
    with pytest.raises(ValueError):gaussian_mean_radius(1.,0,2,.05)
    with pytest.raises(ValueError):gaussian_mean_radius(1.,10,2,1.)
    with pytest.raises(ValueError):Problem(np.eye(2),np.eye(3),np.eye(2))
