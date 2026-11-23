"""Single-barrier knock-in / knock-out options.

Monte Carlo on a time grid only checks the barrier at the grid dates, so it misses crossings
between them and overprices knock-outs. The Brownian-bridge correction fixes this: given the
log-prices at both ends of a step are on the safe side of H, the probability that GBM touched H
in between is

    p = exp(-2 ln(S_i / H) ln(S_{i+1} / H) / (sigma^2 dt)).

Each path is weighted by its survival probability prod(1 - p) instead of a 0/1 indicator.
"""

import math

import numpy as np
from scipy.stats import norm

from .monte_carlo import MCResult
from .paths import simulate_gbm_paths

KINDS = ("down-and-out", "down-and-in", "up-and-out", "up-and-in")


def price_barrier(S0, K, H, T, r, sigma, kind="down-and-out", option="call", n_steps=50,
                  n_paths=100_000, q=0.0, bridge=True, rng=None):
    """Monte Carlo price of a single-barrier option (no rebate), continuous barrier if bridge=True."""
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}")
    down = kind.startswith("down")
    if (down and S0 <= H) or (not down and S0 >= H):
        raise ValueError("S0 is already through the barrier")
    paths = simulate_gbm_paths(S0, T, r, sigma, n_steps, n_paths, q=q, rng=rng)
    ST = paths[:, -1]
    payoff = np.maximum(ST - K, 0.0) if option == "call" else np.maximum(K - ST, 0.0)

    log_dist = np.log(paths / H) if down else np.log(H / paths)  # > 0 on the safe side
    alive_on_grid = np.all(log_dist > 0, axis=1)
    if bridge:
        dt = T / n_steps
        a, b = log_dist[:, :-1], log_dist[:, 1:]
        p_cross = np.exp(-2.0 * np.clip(a, 0, None) * np.clip(b, 0, None) / (sigma**2 * dt))
        survival = np.where(alive_on_grid, np.prod(1.0 - p_cross, axis=1), 0.0)
    else:
        survival = alive_on_grid.astype(float)

    weight = survival if kind.endswith("out") else 1.0 - survival
    disc = math.exp(-r * T) * payoff * weight
    return MCResult(float(disc.mean()), float(disc.std(ddof=1) / np.sqrt(n_paths)), n_paths)


def down_and_out_call(S0, K, H, T, r, sigma, q=0.0):
    """Closed form (Merton / Reiner-Rubinstein) for a continuously monitored down-and-out call, H <= K."""
    if H > K:
        raise ValueError("closed form implemented for H <= K")
    st = sigma * math.sqrt(T)
    lam = (r - q + 0.5 * sigma**2) / sigma**2
    d1 = (math.log(S0 / K) + (r - q + 0.5 * sigma**2) * T) / st
    call = S0 * math.exp(-q * T) * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d1 - st)
    y = math.log(H**2 / (S0 * K)) / st + lam * st
    down_in = (S0 * math.exp(-q * T) * (H / S0) ** (2 * lam) * norm.cdf(y)
               - K * math.exp(-r * T) * (H / S0) ** (2 * lam - 2) * norm.cdf(y - st))
    return call - down_in
