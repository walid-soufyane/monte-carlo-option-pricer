"""Asian (average-price) options with discrete monitoring at t_i = i T / n, i = 1..n."""

import math

import numpy as np
from scipy.stats import norm

from .monte_carlo import MCResult
from .paths import simulate_gbm_paths


def _payoff(avg, K, option):
    if option == "call":
        return np.maximum(avg - K, 0.0)
    if option == "put":
        return np.maximum(K - avg, 0.0)
    raise ValueError("option must be 'call' or 'put'")


def geometric_asian_price(S0, K, T, r, sigma, n_steps, option="call", q=0.0):
    """Closed form for the discretely monitored geometric-average Asian option.

    ln G is normal with
        mean  m = ln S0 + (r - q - sigma^2/2) T (n+1) / (2n)
        var   v = sigma^2 T (n+1)(2n+1) / (6 n^2)
    so the price is a Black-Scholes-type formula on G.
    """
    n = n_steps
    m = math.log(S0) + (r - q - 0.5 * sigma**2) * T * (n + 1) / (2 * n)
    v = sigma**2 * T * (n + 1) * (2 * n + 1) / (6 * n**2)
    sd = math.sqrt(v)
    d2 = (m - math.log(K)) / sd
    d1 = d2 + sd
    fwd = math.exp(m + 0.5 * v)  # E[G]
    disc = math.exp(-r * T)
    if option == "call":
        return disc * (fwd * norm.cdf(d1) - K * norm.cdf(d2))
    if option == "put":
        return disc * (K * norm.cdf(-d2) - fwd * norm.cdf(-d1))
    raise ValueError("option must be 'call' or 'put'")


def price_asian(S0, K, T, r, sigma, n_steps=12, option="call", average="arithmetic",
                n_paths=100_000, q=0.0, rng=None):
    """Plain Monte Carlo price of an arithmetic- or geometric-average Asian option."""
    paths = simulate_gbm_paths(S0, T, r, sigma, n_steps, n_paths, q=q, rng=rng)[:, 1:]
    if average == "arithmetic":
        avg = paths.mean(axis=1)
    elif average == "geometric":
        avg = np.exp(np.log(paths).mean(axis=1))
    else:
        raise ValueError("average must be 'arithmetic' or 'geometric'")
    disc = math.exp(-r * T) * _payoff(avg, K, option)
    return MCResult(float(disc.mean()), float(disc.std(ddof=1) / np.sqrt(n_paths)), n_paths)
