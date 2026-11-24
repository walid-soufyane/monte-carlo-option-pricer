"""Monte Carlo Greeks for European options.

Bump-and-reprice (finite differences) with common random numbers: the same normal draws are
reused for the base and bumped prices, so most of the simulation noise cancels in the
difference. Without that, the variance of (V(S+h) - V(S-h)) / 2h blows up like 1/h^2.
"""

import numpy as np

from .monte_carlo import price_european


def bump_greeks(S0, K, T, r, sigma, option="call", n_paths=200_000, q=0.0,
                h_rel=0.01, vol_bump=0.01, rng=0):
    """Delta, gamma and vega by central differences, all on one shared seed.

    rng must be a fixed seed (int) so every repricing uses identical random numbers.
    """
    if not isinstance(rng, (int, np.integer)):
        raise ValueError("pass an integer seed so the bumps share random numbers")
    h = h_rel * S0

    def v(s=S0, vol=sigma):
        return price_european(s, K, T, r, vol, option, n_paths, q, rng=rng).price

    up, mid, down = v(S0 + h), v(), v(S0 - h)
    return {
        "delta": (up - down) / (2 * h),
        "gamma": (up - 2 * mid + down) / h**2,
        "vega": (v(vol=sigma + vol_bump) - v(vol=sigma - vol_bump)) / (2 * vol_bump),
    }


def _draw(S0, T, r, sigma, n_paths, q, rng):
    z = np.random.default_rng(rng).standard_normal(n_paths)
    ST = S0 * np.exp((r - q - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * z)
    return z, ST


def _est(x):
    return float(x.mean()), float(x.std(ddof=1) / np.sqrt(len(x)))


def pathwise_greeks(S0, K, T, r, sigma, option="call", n_paths=200_000, q=0.0, rng=None):
    """Pathwise (IPA) delta and vega: differentiate the discounted payoff along each path.

    dS_T/dS0 = S_T / S0 and dS_T/dsigma = S_T (ln(S_T/S0) - (r - q + sigma^2/2) T) / sigma,
    so for a call  delta = E[e^{-rT} 1{S_T > K} S_T / S0].  Needs a payoff that is Lipschitz in
    S_T, so it does not work for digitals. Returns {greek: (estimate, std_error)}.
    """
    z, ST = _draw(S0, T, r, sigma, n_paths, q, rng)
    disc = np.exp(-r * T)
    if option == "call":
        dpay = (ST > K).astype(float)
    elif option == "put":
        dpay = -(ST < K).astype(float)
    else:
        raise ValueError("option must be 'call' or 'put'")
    dST_dsigma = ST * (np.log(ST / S0) - (r - q + 0.5 * sigma**2) * T) / sigma
    return {"delta": _est(disc * dpay * ST / S0), "vega": _est(disc * dpay * dST_dsigma)}


def likelihood_ratio_greeks(S0, K, T, r, sigma, payoff="call", n_paths=200_000, q=0.0, rng=None):
    """Likelihood-ratio (score function) delta, gamma and vega.

    Differentiate the density of S_T instead of the payoff: Greek = E[e^{-rT} payoff * score].
    With Z the normal driving S_T:
        delta score = Z / (S0 sigma sqrt T)
        gamma score = ((Z^2 - 1) / (sigma sqrt T) - Z) / (S0^2 sigma sqrt T)
        vega  score = (Z^2 - 1) / sigma - Z sqrt T
    Works for any payoff, including discontinuous ones (payoff="digital" pays 1 if S_T > K),
    at the price of higher variance than pathwise.
    """
    z, ST = _draw(S0, T, r, sigma, n_paths, q, rng)
    if payoff == "call":
        pay = np.maximum(ST - K, 0.0)
    elif payoff == "put":
        pay = np.maximum(K - ST, 0.0)
    elif payoff == "digital":
        pay = (ST > K).astype(float)
    else:
        raise ValueError("payoff must be 'call', 'put' or 'digital'")
    v = np.exp(-r * T) * pay
    sst = sigma * np.sqrt(T)
    return {
        "delta": _est(v * z / (S0 * sst)),
        "gamma": _est(v * ((z**2 - 1) / sst - z) / (S0**2 * sst)),
        "vega": _est(v * ((z**2 - 1) / sigma - z * np.sqrt(T))),
    }
