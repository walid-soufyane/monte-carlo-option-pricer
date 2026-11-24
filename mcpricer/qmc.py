"""Quasi-Monte Carlo with scrambled Sobol points.

Pseudo-random MC error falls like N^{-1/2}. Sobol points fill [0,1]^d far more evenly, and for
smooth enough integrands the error falls close to N^{-1}. Scrambling (Owen) randomises the points
while keeping that evenness, so we can run R independent scrambles and get an honest standard
error from their spread (randomised QMC).
"""

import numpy as np
from scipy.stats import norm, qmc

from .monte_carlo import MCResult


def price_european_rqmc(S0, K, T, r, sigma, option="call", m=14, n_scrambles=16, q=0.0, rng=None):
    """European option by randomised QMC: n_scrambles independent scrambled Sobol sets of 2^m points.

    Returns the mean of the scramble estimates and its standard error. Total points:
    n_scrambles * 2^m. Use a power of two per set to keep Sobol's balance properties.
    """
    seeds = np.random.default_rng(rng).integers(0, 2**31, n_scrambles)
    estimates = []
    for s in seeds:
        estimates.append(_single_scramble(S0, K, T, r, sigma, m, int(s), option, q))
    est = np.asarray(estimates)
    return MCResult(float(est.mean()), float(est.std(ddof=1) / np.sqrt(n_scrambles)), n_scrambles * 2**m)


def convergence_study(S0, K, T, r, sigma, ms=range(8, 17), n_reps=20, rng=0):
    """RMSE vs Black-Scholes for plain MC and scrambled Sobol at N = 2^m points.

    Returns a dict with N, MC RMSE, QMC RMSE and the fitted log-log slopes (theory: -0.5 vs ~ -1).
    """
    from .black_scholes import bs_price
    from .monte_carlo import price_european

    exact = bs_price(S0, K, T, r, sigma)
    gen = np.random.default_rng(rng)
    ns, mc_rmse, qmc_rmse = [], [], []
    for m in ms:
        n = 2**m
        mc = [price_european(S0, K, T, r, sigma, n_paths=n, rng=int(gen.integers(2**31))).price for _ in range(n_reps)]
        qm = [_single_scramble(S0, K, T, r, sigma, m, int(gen.integers(2**31))) for _ in range(n_reps)]
        ns.append(n)
        mc_rmse.append(float(np.sqrt(np.mean((np.array(mc) - exact) ** 2))))
        qmc_rmse.append(float(np.sqrt(np.mean((np.array(qm) - exact) ** 2))))
    slope = lambda e: float(np.polyfit(np.log(ns), np.log(e), 1)[0])
    return {"n": ns, "mc_rmse": mc_rmse, "qmc_rmse": qmc_rmse,
            "mc_slope": slope(mc_rmse), "qmc_slope": slope(qmc_rmse)}


def _single_scramble(S0, K, T, r, sigma, m, seed, option="call", q=0.0):
    """Price from one scrambled Sobol set of 2^m points."""
    u = qmc.Sobol(d=1, scramble=True, seed=seed).random_base2(m)[:, 0]
    ST = S0 * np.exp((r - q - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * norm.ppf(u))
    pay = np.maximum(ST - K, 0.0) if option == "call" else np.maximum(K - ST, 0.0)
    return float(np.exp(-r * T) * pay.mean())
