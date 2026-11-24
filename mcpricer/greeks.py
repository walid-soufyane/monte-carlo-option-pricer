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
