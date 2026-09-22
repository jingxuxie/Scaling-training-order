"""All-schedule comparisons in the symmetric two-source family.

Every schedule is fixed before the initial-error draw (and before process noise).
The mathematical bounds cover arbitrary measurable source fractions, not merely
piecewise-constant schedules. Numerical functions use ordinary float64.
"""
from __future__ import annotations
import numpy as np
from .core import family_constants, family_static


def _validate(budget: float, epsilon: float = 0., ridge: float = 0.) -> None:
    if not np.isfinite([budget, epsilon, ridge]).all() or min(budget, epsilon, ridge) < 0:
        raise ValueError('Budget, epsilon, and ridge must be finite and nonnegative')


def global_epsilon_threshold(budget: float, angle: float) -> float:
    """Static mixing is globally optimal iff epsilon >= this value, for B>0.

    Below the threshold a two-phase schedule with mixed phases strictly improves
    risk. At B=0 all schedules coincide; the iff assertion excludes that case.
    """
    _validate(budget)
    return float(np.exp(-2*family_constants(angle)['c']*budget))


def all_schedule_lower(budget: float, angle: float, epsilon: float,
                       ridge: float = 0.) -> float:
    """Volume + slow-column lower envelope, combined with the energy bound.

    This is the exact optimum above the epsilon threshold. Below it the lower
    envelope need not be attainable. Valid for arbitrary epsilon >= 0.
    """
    _validate(budget, epsilon, ridge)
    k = family_constants(angle)
    threshold = global_epsilon_threshold(budget, angle)
    if epsilon >= threshold:
        return float(family_static(budget, angle, epsilon, ridge))
    volume = np.sqrt(epsilon)*np.exp(-(1+2*ridge)*budget)
    energy = .5*(1+epsilon)*np.exp(-2*(1+ridge)*budget)
    slow = .5*epsilon*np.exp(-2*(k['b']+ridge)*budget)
    return float(max(volume, energy, slow))


def boundary_witness(budget: float, angle: float, amplitude_fraction: float = .01):
    """A two-phase perturbation of balance used in the necessity proof.

    Off-diagonal z = h*amplitude_fraction*q in the first half, and
    z = -h*amplitude_fraction in the second. For each epsilon below the
    threshold, sufficiently small positive amplitude strictly improves risk.
    No fixed amplitude is promised to improve all instances below the boundary.
    """
    _validate(budget)
    if budget <= 0 or not 0 < amplitude_fraction <= 1:
        raise ValueError('Require positive budget and amplitude in (0,1]')
    k = family_constants(angle)
    x = .5*k['c']*budget
    # sinh(x)/(sinh(2x)-sinh(x)) = 1/(2*cosh(x)-1).
    q = 1/(2*np.cosh(x)-1)
    return [(budget/2, (1-amplitude_fraction*q)/2),
            (budget/2, (1+amplitude_fraction)/2)]


def balanced_noise_risk(budget: float, angle: float, sigma: float,
                        ridge: float = 0.) -> float:
    """Minimal additive isotropic diffusion contribution, for every schedule.

    For de=-G(p(t))e dt + sigma dW, this equals the balanced schedule's
    contribution and lower-bounds every precommitted schedule's noise risk.
    """
    _validate(budget, 0., ridge)
    if not np.isfinite(sigma) or sigma < 0:
        raise ValueError('sigma must be finite and nonnegative')
    k = family_constants(angle)
    rates = np.array([ridge+k['a'], ridge+k['b']])
    return float(sigma**2/4*np.sum(-np.expm1(-2*rates*budget)/rates))


def noisy_gain_ceiling(budget: float, angle: float, epsilon: float,
                       sigma: float, ridge: float = 0.) -> float:
    noise = balanced_noise_risk(budget, angle, sigma, ridge)
    den = all_schedule_lower(budget, angle, epsilon, ridge)+noise
    return float((family_static(budget, angle, epsilon, ridge)+noise)/den)
