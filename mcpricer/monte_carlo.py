"""Monte Carlo pricing of European options under geometric Brownian motion."""

from dataclasses import dataclass

import numpy as np


@dataclass
class MCResult:
    price: float
    std_error: float
    n_paths: int

    @property
    def conf_int(self):
        """95% confidence interval for the price."""
        half = 1.96 * self.std_error
        return self.price - half, self.price + half


def simulate_terminal(S0, T, r, sigma, n_paths, q=0.0, rng=None):
    """Draw S_T exactly under risk-neutral GBM: S_T = S0 exp((r - q - sigma^2/2) T + sigma sqrt(T) Z)."""
    rng = np.random.default_rng(rng)
    z = rng.standard_normal(n_paths)
    return S0 * np.exp((r - q - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * z)


def price_european(S0, K, T, r, sigma, option="call", n_paths=100_000, q=0.0, rng=None):
    """Plain Monte Carlo price of a European call/put, with its standard error."""
    ST = simulate_terminal(S0, T, r, sigma, n_paths, q, rng)
    if option == "call":
        payoff = np.maximum(ST - K, 0.0)
    elif option == "put":
        payoff = np.maximum(K - ST, 0.0)
    else:
        raise ValueError("option must be 'call' or 'put'")
    disc = np.exp(-r * T) * payoff
    return MCResult(float(disc.mean()), float(disc.std(ddof=1) / np.sqrt(n_paths)), n_paths)
