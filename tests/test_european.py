import math

import pytest

from mcpricer import bs_price, price_european

PARAMS = dict(S0=100.0, K=105.0, T=1.0, r=0.04, sigma=0.2)


@pytest.mark.parametrize("option", ["call", "put"])
def test_mc_matches_black_scholes(option):
    res = price_european(**PARAMS, option=option, n_paths=400_000, rng=42)
    exact = bs_price(PARAMS["S0"], PARAMS["K"], PARAMS["T"], PARAMS["r"], PARAMS["sigma"], option)
    # Within 4 standard errors of the closed form
    assert abs(res.price - exact) < 4 * res.std_error


def test_put_call_parity_closed_form():
    S, K, T, r, s = 100.0, 105.0, 1.0, 0.04, 0.2
    c, p = bs_price(S, K, T, r, s, "call"), bs_price(S, K, T, r, s, "put")
    assert math.isclose(c - p, S - K * math.exp(-r * T), rel_tol=1e-12)


def test_std_error_shrinks_like_sqrt_n():
    small = price_european(**PARAMS, n_paths=10_000, rng=1).std_error
    big = price_european(**PARAMS, n_paths=160_000, rng=1).std_error
    assert 3.0 < small / big < 5.0  # theory says 4
