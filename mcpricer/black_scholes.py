"""Closed-form Black-Scholes prices, used as a benchmark for the Monte Carlo engine."""

import math

from scipy.stats import norm


def _d1_d2(S, K, T, r, sigma, q=0.0):
    if T <= 0 or sigma <= 0:
        raise ValueError("T and sigma must be positive")
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    return d1, d1 - sigma * math.sqrt(T)


def bs_price(S, K, T, r, sigma, option="call", q=0.0):
    """Black-Scholes price of a European call or put with continuous dividend yield q."""
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    if option == "call":
        return S * math.exp(-q * T) * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    if option == "put":
        return K * math.exp(-r * T) * norm.cdf(-d2) - S * math.exp(-q * T) * norm.cdf(-d1)
    raise ValueError("option must be 'call' or 'put'")
