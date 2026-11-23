"""Simulation of full GBM price paths on a uniform time grid."""

import numpy as np


def simulate_gbm_paths(S0, T, r, sigma, n_steps, n_paths, q=0.0, antithetic=False, rng=None):
    """Exact GBM paths under the risk-neutral measure.

    Uses the exact log-increment  ln(S_{t+dt}/S_t) = (r - q - sigma^2/2) dt + sigma sqrt(dt) Z,
    so there is no time-discretisation bias at the grid dates.

    Returns an array of shape (n_paths, n_steps + 1) whose first column is S0.
    With antithetic=True, the second half of the paths uses the negated shocks of the first half
    (n_paths must then be even).
    """
    rng = np.random.default_rng(rng)
    dt = T / n_steps
    if antithetic:
        if n_paths % 2:
            raise ValueError("n_paths must be even for antithetic sampling")
        half = rng.standard_normal((n_paths // 2, n_steps))
        z = np.vstack([half, -half])
    else:
        z = rng.standard_normal((n_paths, n_steps))
    increments = (r - q - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
    log_paths = np.cumsum(increments, axis=1)
    return S0 * np.exp(np.hstack([np.zeros((n_paths, 1)), log_paths]))
