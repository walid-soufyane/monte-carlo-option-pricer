import pytest

from mcpricer import bs_greeks, bump_greeks, price_european

S0, K, T, r, sigma = 100.0, 100.0, 1.0, 0.04, 0.2


@pytest.mark.parametrize("option", ["call", "put"])
def test_bump_greeks_match_black_scholes(option):
    exact = bs_greeks(S0, K, T, r, sigma, option)
    mc = bump_greeks(S0, K, T, r, sigma, option, n_paths=400_000, rng=5)
    assert mc["delta"] == pytest.approx(exact["delta"], abs=0.005)
    assert mc["vega"] == pytest.approx(exact["vega"], rel=0.02)
    assert mc["gamma"] == pytest.approx(exact["gamma"], rel=0.10)


def test_call_put_delta_parity():
    c, p = bs_greeks(S0, K, T, r, sigma, "call"), bs_greeks(S0, K, T, r, sigma, "put")
    assert c["delta"] - p["delta"] == pytest.approx(1.0)
    assert c["gamma"] == pytest.approx(p["gamma"]) and c["vega"] == pytest.approx(p["vega"])


def test_common_random_numbers_matter():
    # independent seeds for the two bumps give a far noisier delta than a shared seed
    h = 1.0
    noisy = [(price_european(S0 + h, K, T, r, sigma, n_paths=20_000, rng=2 * i).price
              - price_european(S0 - h, K, T, r, sigma, n_paths=20_000, rng=2 * i + 1).price) / (2 * h)
             for i in range(20)]
    crn = [bump_greeks(S0, K, T, r, sigma, n_paths=20_000, rng=i)["delta"] for i in range(20)]
    import numpy as np
    assert np.std(crn) < 0.1 * np.std(noisy)


def _within(est, exact, k=4):
    value, se = est
    return abs(value - exact) < k * se


@pytest.mark.parametrize("option", ["call", "put"])
def test_pathwise_matches_black_scholes(option):
    from mcpricer import pathwise_greeks

    exact = bs_greeks(S0, K, T, r, sigma, option)
    pw = pathwise_greeks(S0, K, T, r, sigma, option, n_paths=300_000, rng=8)
    assert _within(pw["delta"], exact["delta"]) and _within(pw["vega"], exact["vega"])


def test_likelihood_ratio_matches_black_scholes():
    from mcpricer import likelihood_ratio_greeks

    exact = bs_greeks(S0, K, T, r, sigma)
    lr = likelihood_ratio_greeks(S0, K, T, r, sigma, n_paths=400_000, rng=9)
    for g in ("delta", "gamma", "vega"):
        assert _within(lr[g], exact[g]), g


def test_likelihood_ratio_handles_digital():
    # digital call delta = e^{-rT} phi(d2) / (S0 sigma sqrt T): pathwise gives 0 here, LR works
    import math
    from scipy.stats import norm
    from mcpricer import likelihood_ratio_greeks

    d2 = (math.log(S0 / K) + (r - 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    exact = math.exp(-r * T) * norm.pdf(d2) / (S0 * sigma * math.sqrt(T))
    lr = likelihood_ratio_greeks(S0, K, T, r, sigma, payoff="digital", n_paths=400_000, rng=10)
    assert _within(lr["delta"], exact)


def test_pathwise_has_lower_variance_than_lr():
    from mcpricer import likelihood_ratio_greeks, pathwise_greeks

    pw = pathwise_greeks(S0, K, T, r, sigma, n_paths=100_000, rng=1)
    lr = likelihood_ratio_greeks(S0, K, T, r, sigma, n_paths=100_000, rng=1)
    assert pw["delta"][1] < lr["delta"][1] and pw["vega"][1] < lr["vega"][1]
