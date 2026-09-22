"""Exact population dynamics; all budgets are normalized gradient-flow work.

A schedule is a sequence of (duration, fraction_of_source_A) pairs. Durations
sum to training work, not to total work when a pilot is used. All generators
must be symmetric positive semidefinite. No optimal-control solver is treated
as a certificate of global optimality.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable
import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import minimize_scalar

Array = NDArray[np.float64]


def _psd(a: ArrayLike, name: str) -> Array:
    a = np.asarray(a, dtype=float)
    if a.ndim != 2 or a.shape[0] != a.shape[1] or not np.isfinite(a).all():
        raise ValueError(f"{name} must be a finite square matrix")
    if not np.allclose(a, a.T, atol=1e-11, rtol=1e-11):
        raise ValueError(f"{name} must be symmetric")
    a = (a + a.T) / 2
    if np.linalg.eigvalsh(a)[0] < -1e-11 * max(1., np.linalg.norm(a, 2)):
        raise ValueError(f"{name} must be positive semidefinite")
    return a


@dataclass(frozen=True)
class Problem:
    A: Array
    B: Array
    S: Array
    Q: Array | None = None

    def __post_init__(self) -> None:
        for name in ("A", "B", "S"):
            object.__setattr__(self, name, _psd(getattr(self, name), name))
        q = np.eye(self.A.shape[0]) if self.Q is None else _psd(self.Q, "Q")
        object.__setattr__(self, "Q", q)
        if not (self.A.shape == self.B.shape == self.S.shape == q.shape):
            raise ValueError("All matrices must have the same shape")

    def mixture(self, p: float) -> Array:
        if not np.isfinite(p) or not 0 <= p <= 1:
            raise ValueError("Mixture fraction must lie in [0, 1]")
        return p * self.A + (1 - p) * self.B


def flow(g: ArrayLike, t: float) -> Array:
    if not np.isfinite(t) or t < 0:
        raise ValueError("Duration must be finite and nonnegative")
    g = _psd(g, "generator")
    w, v = np.linalg.eigh(g)
    return (v * np.exp(-t * np.maximum(w, 0))) @ v.T


def transition(problem: Problem, schedule: Iterable[tuple[float, float]]) -> Array:
    out = np.eye(problem.A.shape[0])
    for t, p in schedule:
        out = flow(problem.mixture(float(p)), float(t)) @ out
    return out


def risk(problem: Problem, schedule: Iterable[tuple[float, float]]) -> float:
    phi = transition(problem, schedule)
    return float(np.trace(problem.Q @ phi @ problem.S @ phi.T) / 2)


def static_grid(problem: Problem, budget: float, points: int = 1001) -> tuple[float, float]:
    """Numerical candidate, NOT a global-optimality certificate for general inputs."""
    if points < 3:
        raise ValueError("At least three grid points are required")
    ps = np.linspace(0., 1., points)
    fs = np.array([risk(problem, [(budget, p)]) for p in ps])
    i = int(fs.argmin())
    lo, hi = ps[max(0, i-1)], ps[min(points-1, i+1)]
    out = minimize_scalar(lambda p: risk(problem, [(budget, p)]), bounds=(lo, hi),
                          method="bounded", options={"xatol": 1e-13})
    candidates = [(float(fs[i]), float(ps[i])), (float(out.fun), float(out.x))]
    return min(candidates)


def symmetric_problem(angle: float = np.pi/4, epsilon: float = 0.,
                      ridge: float = 0.) -> Problem:
    """Unit-norm bisector initial error with orthogonal variance epsilon.

A = aa^T + ridge*I, B = bb^T + ridge*I; angle(a,b) = angle.
The same isotropic ridge on both sources multiplies every risk by exp(-2*ridge*T).
"""
    if not 0 < angle < np.pi/2 or epsilon < 0 or ridge < 0:
        raise ValueError("Require 0 < angle < pi/2, epsilon >= 0, ridge >= 0")
    a = np.array([1., 0.]); b = np.array([np.cos(angle), np.sin(angle)])
    u = np.array([np.cos(angle/2), np.sin(angle/2)])
    v = np.array([-u[1], u[0]])
    return Problem(np.outer(a,a)+ridge*np.eye(2), np.outer(b,b)+ridge*np.eye(2),
                   np.outer(u,u)+epsilon*np.outer(v,v))


def family_constants(angle: float) -> dict[str, float]:
    if not 0 < angle < np.pi/2:
        raise ValueError("Require an acute source angle")
    c = float(np.cos(angle)); a = (1+c)/2; b = (1-c)/2
    tau = float(np.log((1+c)/c))
    return dict(c=c, a=a, b=b, tau=tau, A=a/c**2,
                D=b*(1+2*c)**2/c**2, F=c**2/a)


def family_static(budget: ArrayLike, angle: float, epsilon: float = 0.,
                  ridge: float = 0.) -> Array:
    t = np.asarray(budget, dtype=float)
    if np.any(t < 0) or epsilon < 0 or ridge < 0:
        raise ValueError("Negative budget, variance or ridge")
    k = family_constants(angle)
    return .5*(np.exp(-2*(k['a']+ridge)*t)+epsilon*np.exp(-2*(k['b']+ridge)*t))


def family_constructive(budget: ArrayLike, angle: float, epsilon: float = 0.,
                        ridge: float = 0.) -> Array:
    t = np.asarray(budget, dtype=float); k = family_constants(angle)
    if np.any(t < k['tau']-1e-12) or epsilon < 0 or ridge < 0:
        raise ValueError("Budget must permit preparation; variance/ridge nonnegative")
    return .5*np.exp(-2*ridge*t)*((k['A']+epsilon*k['D'])*np.exp(-2*t)+epsilon*k['F'])


def family_epsilon_threshold(budget: float, angle: float) -> float:
    """Exact strict-improvement threshold for the explicit preparation schedule.

The result is restricted to 0 <= epsilon <= 1. Zero denotes no strict
improvement in that interval at this budget. This is NOT the threshold
for the entire class of two-phase schedules.
"""
    k = family_constants(angle)
    if budget < k['tau']:
        return 0.
    numerator = np.exp(-2*k['a']*budget)-k['A']*np.exp(-2*budget)
    if numerator <= 0:
        return 0.
    denominator = k['F']+k['D']*np.exp(-2*budget)-np.exp(-2*k['b']*budget)
    if denominator <= 0:
        raise ArithmeticError("Unexpected nonpositive uncertainty denominator")
    return float(numerator / denominator)


def family_two_phase_lower(budget: float, angle: float, epsilon: float,
                           ridge: float = 0.) -> float:
    """All two-phase schedules, 0 < epsilon <= 1, including mixed phases."""
    if not 0 < epsilon <= 1 or budget < 0 or ridge < 0:
        raise ValueError("Require positive uncertainty at most one")
    k = family_constants(angle)
    return .5*epsilon*k['c']**2*np.exp(-2*(ridge+k['b'])*budget)


def generic_risk_radius(budget: float, generator_radius: float,
                        trace_estimate: float, covariance_trace_radius: float) -> float:
    """Uniform Duhamel risk radius, Q=I; both estimated and true generators PSD.

The covariance error is measured in nuclear norm. Useful for theory; often
conservative numerically. Allows schedule selection using the calibration data.
"""
    if min(budget, generator_radius, trace_estimate, covariance_trace_radius) < 0:
        raise ValueError("All arguments must be nonnegative")
    return .5*covariance_trace_radius + trace_estimate*min(2., budget*generator_radius)
