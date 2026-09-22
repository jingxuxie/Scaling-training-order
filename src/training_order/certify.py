"""Uniform finite-grid lower envelopes for every static mixture.

The envelopes are rigorous in real arithmetic. The implementation uses float64
and explicit outward numerical padding; it is not a directed-rounding proof
assistant. The paper's central family comparison is analytic and does not rely
on this numerical optimizer.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
from .core import Problem, flow

Array = NDArray[np.float64]


def gaussian_mean_radius(sigma: float, samples_per_coordinate: int,
                         dimension: int, delta: float) -> float:
    """For independent N(e_j, sigma^2) coordinate queries, n per coordinate."""
    if sigma < 0 or samples_per_coordinate < 1 or dimension < 1 or not 0 < delta < 1:
        raise ValueError("Invalid Gaussian confidence parameters")
    return float(sigma/np.sqrt(samples_per_coordinate) *
                 (np.sqrt(dimension) + np.sqrt(2*np.log(1/delta))))


@dataclass
class StaticEnvelope:
    budget: float
    matrices: Array
    operator_norms: Array
    deviations: Array
    grid: Array
    roundoff_pad: float = 1e-12

    @classmethod
    def build(cls, problem: Problem, budget: float, cells: int = 512,
              generator_radius: float = 0.) -> 'StaticEnvelope':
        if budget < 0 or cells < 1 or generator_radius < 0:
            raise ValueError("Invalid envelope parameters")
        ends = np.linspace(0., 1., cells+1)
        mids = (ends[:-1]+ends[1:])/2
        matrices = np.array([flow(problem.mixture(p), budget) for p in mids])
        norms = np.linalg.norm(matrices, ord=2, axis=(1,2))
        mins = np.array([max(0., np.linalg.eigvalsh(problem.mixture(p))[0]) for p in ends])
        # lambda_min is concave on each interval, so the endpoint minimum is valid.
        m = np.minimum(mins[:-1], mins[1:])
        radius = budget*np.linalg.norm(problem.A-problem.B, 2)*(.5/cells)*np.exp(-budget*m)
        radius += min(2., budget*generator_radius)
        return cls(budget, matrices, norms, radius, mids)

    def lower(self, ehat: Array, radius: float) -> float:
        if radius < 0:
            raise ValueError("Negative initial-state radius")
        ehat = np.asarray(ehat, dtype=float)
        y = np.einsum('nij,j->ni', self.matrices, ehat)
        norm = np.linalg.norm(y, axis=1)
        enorm=np.linalg.norm(ehat)
        low = norm-self.operator_norms*radius-self.deviations*(enorm+radius)
        amplitude=.5*np.maximum(low,0.)**2
        # A convex quadratic has a global first-order lower bound. Unlike a
        # triangle bound, its uncertainty penalty follows the remaining error.
        grad=np.einsum('nij,nj->ni',self.matrices,y)
        op_upper=np.minimum(1.,self.operator_norms+self.deviations)
        grad_upper=np.linalg.norm(grad,axis=1)+self.deviations*(norm+op_upper*enorm)
        value_lower=.5*np.maximum(norm-self.deviations*enorm,0.)**2
        directional=value_lower-radius*grad_upper
        return max(0.,float(np.min(np.maximum(amplitude,directional)))-self.roundoff_pad)

    def selected_p(self, ehat: Array) -> float:
        y = np.einsum('nij,j->ni', self.matrices, ehat)
        return float(self.grid[np.argmin(np.sum(y*y, axis=1))])


def candidate_upper(phi: Array, ehat: Array, state_radius: float,
                    budget: float = 0., generator_radius: float = 0.,
                    numerical_pad: float = 1e-12) -> float:
    if min(state_radius, budget, generator_radius, numerical_pad) < 0:
        raise ValueError("Negative radius or budget")
    phi = np.asarray(phi, dtype=float); ehat = np.asarray(ehat, dtype=float)
    normphi=np.linalg.norm(phi,2); y=phi@ehat
    delta=min(2.,budget*generator_radius)
    amplitude=.5*(np.linalg.norm(y)+normphi*state_radius+delta*(np.linalg.norm(ehat)+state_radius))**2
    directional=.5*float(y@y)+state_radius*np.linalg.norm(phi.T@y)+.5*state_radius**2*normphi**2
    directional+=.5*delta*(2*normphi+delta)*(np.linalg.norm(ehat)+state_radius)**2
    return float(min(amplitude,directional)+numerical_pad)
