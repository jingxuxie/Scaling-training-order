import numpy as np
import pytest
from training_order.core import symmetric_problem, transition, risk, family_static
from training_order.global_bounds import (all_schedule_lower, global_epsilon_threshold,
                                        boundary_witness, balanced_noise_risk, noisy_gain_ceiling)

@pytest.mark.parametrize('angle',[.2,.6,1.1,1.4])
@pytest.mark.parametrize('budget',[.2,1.,4.,8.])
def test_arbitrary_schedule_bound(angle,budget):
    rng=np.random.default_rng(551)
    crit=global_epsilon_threshold(budget,angle)
    for eps in [0.,crit*.1,crit,crit*2,1.]:
        prob=symmetric_problem(angle,eps,.03)
        for n in [1,2,7,31]:
            ts=rng.dirichlet(np.ones(n))*budget
            path=list(zip(ts,rng.random(n)))
            value=risk(prob,path)
            assert value >= all_schedule_lower(budget,angle,eps,.03)-1e-12
            if eps >= crit:
                assert value >= family_static(budget,angle,eps,.03)-1e-12
            u=np.array([np.cos(angle/2),np.sin(angle/2)])
            v=np.array([-u[1],u[0]])
            phi=transition(prob,path)
            assert np.linalg.norm(phi@v)**2 >= np.exp(-2*(.03+(1-np.cos(angle))/2)*budget)-1e-12

@pytest.mark.parametrize('angle',[.3,.8,1.2])
@pytest.mark.parametrize('budget',[1.,3.,6.])
def test_local_necessity(angle,budget):
    eps=.8*global_epsilon_threshold(budget,angle)
    prob=symmetric_problem(angle,eps)
    value=risk(prob,boundary_witness(budget,angle,.005))
    assert value < family_static(budget,angle,eps)-1e-14

@pytest.mark.parametrize('budget',[.5,2,6,20])
def test_noise_ceiling(budget):
    assert noisy_gain_ceiling(budget,.7,0.,.1)>=1
    assert balanced_noise_risk(budget,.7,0.)==0
    assert np.isclose(noisy_gain_ceiling(budget,.7,1.,.1),1.)

@pytest.mark.parametrize('bad',[-1.,np.nan,np.inf])
def test_bad_values(bad):
    with pytest.raises(ValueError):global_epsilon_threshold(bad,.5)

@pytest.mark.parametrize('diagonal',[(1.,.1),(2.,.7),(4.,3.),(.3,.2)])
@pytest.mark.parametrize('fraction',[.03,.25,.9,1.])
def test_general_mirror_family(diagonal,fraction):
    """Exercise the theorem outside the normalized rank-one family."""
    from scipy.linalg import expm
    a,d=diagonal;h=fraction*np.sqrt(a*d);B=1.3;crit=np.exp(-2*(a-d)*B)
    rng=np.random.default_rng(923)
    for eps in [crit,2*crit,1.]:
        base=.5*(np.exp(-2*a*B)+eps*np.exp(-2*d*B))
        for stages in [1,3,16]:
            ts=rng.dirichlet(np.ones(stages))*B
            zs=rng.uniform(-h,h,stages);phi=np.eye(2)
            for t,z in zip(ts,zs):phi=expm(-t*np.array([[a,z],[z,d]]))@phi
            assert .5*np.trace(phi@np.diag([1,eps])@phi.T)>=base-2e-14
    q=1/(2*np.cosh((a-d)*B/2)-1);eta=.002*h
    phi=expm(-B/2*np.array([[a,-eta],[-eta,d]]))@expm(-B/2*np.array([[a,eta*q],[eta*q,d]]))
    eps=.5*crit;base=.5*(np.exp(-2*a*B)+eps*np.exp(-2*d*B))
    assert .5*np.trace(phi@np.diag([1,eps])@phi.T)<base
