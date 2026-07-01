"""Stochastic degradation and renovation progress models.

Both steps consume *externally supplied* noise rather than drawing from an rng.
This makes the stochastic transition a pure function of (state, noise), which is
what enables common random numbers (CRN) across candidate actions in rollouts:
callers generate the noise once (from an rng for the real environment, or from a
state/seed-derived key for rollouts) and reuse it across all candidates.

The Gamma increment uses the inverse-CDF transform `gammaincinv(alpha*dt, u)/beta`
so that a shared uniform `u` maps to the same quantile regardless of the (action-
dependent) shape parameter. This is exact CRN and, unlike rejection sampling,
consumes a fixed amount of randomness per asset.
"""
from __future__ import annotations

import numpy as np
from scipy.special import gammaincinv


def gamma_step(
    d: np.ndarray,
    alpha0: np.ndarray,
    beta: np.ndarray,
    ell: np.ndarray,
    dt: float,
    u: np.ndarray,
    restrict_degrad_multiplier: float = 0.5,
) -> np.ndarray:
    """
    Vectorized Gamma increment over N assets via inverse-CDF of given uniforms.

    u: (N,) uniforms in [0, 1) — the shared random quantiles.
    restrict_degrad_multiplier: fraction of alpha0 active under load restriction.
      0.5 = rate halved, 1.0 = no effect, 0.0 = degradation fully stopped.

    Returns updated d, shape (N,).
    """
    alpha = (1.0 - (1.0 - restrict_degrad_multiplier) * ell) * alpha0
    a = alpha * dt
    # gammaincinv(a, u) is the inverse regularized lower incomplete gamma:
    # a standard Gamma(shape=a, scale=1) variate at quantile u. Scale by 1/beta.
    delta = gammaincinv(np.maximum(a, 1e-12), u) / beta
    delta = np.where(a > 0.0, delta, 0.0)   # a==0 (degradation stopped) -> no increment
    return np.minimum(1.0, d + delta)


def wiener_step(
    h: np.ndarray,
    mu_h: float | np.ndarray,
    sigma_h: float | np.ndarray,
    dt: float,
    eps: np.ndarray,
) -> np.ndarray:
    """
    Vectorized Wiener step over N assets using given standard-normal draws.

    eps: (N,) standard normal draws — the shared random increments.
    Only advances assets where h > 0 (under active renovation).
    Returns updated h, shape (N,).
    """
    under_renovation = h > 0
    h_new = h - mu_h * dt + sigma_h * np.sqrt(dt) * eps
    return np.where(under_renovation, h_new, h)
