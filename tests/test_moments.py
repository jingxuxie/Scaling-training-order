import numpy as np
import pytest
from training_order.moments import *
from training_order.core import symmetric_problem

@pytest.mark.parametrize('batch',[1,4,32])
@pytest.mark.parametrize('sigma',[0.,.2])
def test_operator_monte_carlo(batch,sigma):
    rng=np.random.default_rng(592)
    h=np.array([[1.,.2],[.2,.4]]);s=np.array([[.8,-.15],[-.15,.2]])
    n=60000
    e=rng.multivariate_normal([0.,0.],s,n)
    x=rng.multivariate_normal([0.,0.],h,(n,batch))
    noise=rng.normal(0,sigma,(n,batch))
    grad=np.einsum('nbd,nb->nd',x,np.einsum('nbd,nd->nb',x,e)+noise)/batch
    out=e-.04*grad
    observed=out.T@out/n
    op=gaussian_operator(h,.04,batch,sigma)
    expected=unvectorize(op.matrix@vectorize(s)+op.injection)
    assert np.max(abs(observed-expected))<.012
    assert np.allclose(unvectorize(op.matrix.T@np.array([1.,0.,1.])),op.trace_matrix)

@pytest.mark.parametrize('steps',[2,13,80])
def test_static_global_lower(steps):
    p=symmetric_problem(.8,.01,.02)
    oa=gaussian_operator(p.A,.03,4,.1);ob=gaussian_operator(p.B,.03,4,.1)
    bound=static_envelope(oa,ob,p.S,steps,129)
    vals=[moment_risk(oa,ob,p.S,[(steps,x)]) for x in np.linspace(0,1,1103)]
    assert min(vals)>=bound['lower']-1e-12
    assert bound['lower']<=bound['upper']

@pytest.mark.parametrize('n',[0,1,2,17])
def test_batch_power(n):
    rng=np.random.default_rng(34);a=rng.normal(size=(5,3,3))*.2
    actual=batch_power(a,n)
    for i in range(len(a)):
        assert np.allclose(actual[i],np.linalg.matrix_power(a[i],n))

@pytest.mark.parametrize('steps',[1,5,25])
def test_pure_search(steps):
    p=symmetric_problem(.8,.01)
    oa=gaussian_operator(p.A,.04);ob=gaussian_operator(p.B,.04)
    val,path=best_pure_two_phase(oa,ob,p.S,steps)
    brute=min(moment_risk(oa,ob,p.S,[(k,x),(steps-k,1-x)]) for k in range(steps+1) for x in [0.,1.])
    assert np.isclose(val,brute,atol=1e-12)

@pytest.mark.parametrize('batch',[1,8,64])
def test_per_example_mixture_mc(batch):
    rng=np.random.default_rng(925)
    problem=symmetric_problem(.8,.1,.03)
    oa=gaussian_operator(problem.A,.06,batch,.1);ob=gaussian_operator(problem.B,.06,batch,.1)
    n=45000;p=.3
    e=rng.multivariate_normal([0,0],problem.S,n)
    g=rng.normal(size=(n,batch,2));mask=rng.random((n,batch))<p
    x=np.where(mask[:,:,None],g@np.linalg.cholesky(problem.A).T,g@np.linalg.cholesky(problem.B).T)
    errs=np.einsum('nbd,nd->nb',x,e)+rng.normal(0,.1,(n,batch))
    after=e-.06*np.einsum('nbd,nb->nd',x,errs)/batch
    observed=after.T@after/n
    exact=mixture_affine(oa,ob,p,'example')@np.r_[vectorize(problem.S),1.]
    assert np.max(abs(observed-unvectorize(exact[:3])))<.016
    if batch==1:
        assert np.allclose(mixture_affine(oa,ob,p,'example'),mixture_affine(oa,ob,p,'batch'))

@pytest.mark.parametrize('steps',[2,13,80,300])
def test_example_static_bound(steps):
    p=symmetric_problem(.8,.01,.02)
    oa=gaussian_operator(p.A,.03,8,.1);ob=gaussian_operator(p.B,.03,8,.1)
    bound=static_envelope(oa,ob,p.S,steps,257,'example')
    vals=[moment_risk(oa,ob,p.S,[(steps,x)],'example') for x in np.linspace(0,1,901)]
    assert min(vals)>=bound['lower']-1e-12
