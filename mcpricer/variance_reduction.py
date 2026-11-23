"""Variance reduction for European option Monte Carlo: antithetic variates and control variates."""

import numpy as np

from .monte_carlo import MCResult


def _payoff(ST, K, option):
    if option == "call":
        return np.maximum(ST - K, 0.0)
    if option == "put":
        return np.maximum(K - ST, 0.0)
    raise ValueError("option must be 'call' or 'put'")


def price_european_antithetic(S0, K, T, r, sigma, option="call", n_paths=100_000, q=0.0, rng=None):
    """Antithetic variates: pair each draw Z with -Z and average the two discounted payoffs.

    The pair average is negatively correlated within the pair for monotone payoffs, so its
    variance is below that of two independent draws. n_paths is the total number of
    simulated prices (n_paths // 2 pairs). The standard error uses the pair averages,
    which are i.i.d.
    """
    rng = np.random.default_rng(rng)
    n_pairs = n_paths // 2
    z = rng.standard_normal(n_pairs)
    drift = (r - q - 0.5 * sigma**2) * T
    vol = sigma * np.sqrt(T)
    up = S0 * np.exp(drift + vol * z)
    down = S0 * np.exp(drift - vol * z)
    disc = np.exp(-r * T)
    pair = 0.5 * disc * (_payoff(up, K, option) + _payoff(down, K, option))
    return MCResult(float(pair.mean()), float(pair.std(ddof=1) / np.sqrt(n_pairs)), 2 * n_pairs)


def price_european_control_variate(S0, K, T, r, sigma, option="call", n_paths=100_000, q=0.0, rng=None):
    """Control variate using the discounted terminal price, whose mean is known: E[e^{-rT} S_T] = S0 e^{-qT}.

    Estimator: Y - b (X - E[X]) with the variance-minimising coefficient b = Cov(X, Y) / Var(X),
    estimated from the same sample. The variance falls by a factor (1 - rho^2).
    """
    rng = np.random.default_rng(rng)
    z = rng.standard_normal(n_paths)
    ST = S0 * np.exp((r - q - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * z)
    disc = np.exp(-r * T)
    y = disc * _payoff(ST, K, option)
    x = disc * ST
    ex = S0 * np.exp(-q * T)
    cov = np.cov(x, y, ddof=1)
    b = cov[0, 1] / cov[0, 0]
    adj = y - b * (x - ex)
    return MCResult(float(adj.mean()), float(adj.std(ddof=1) / np.sqrt(n_paths)), n_paths)
