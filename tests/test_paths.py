import numpy as np
import pytest

from mcpricer import simulate_gbm_paths

S0, T, r, sigma = 100.0, 1.0, 0.04, 0.2


def test_shape_and_start():
    p = simulate_gbm_paths(S0, T, r, sigma, n_steps=50, n_paths=10, rng=0)
    assert p.shape == (10, 51)
    assert np.all(p[:, 0] == S0)


def test_discounted_price_is_martingale():
    p = simulate_gbm_paths(S0, T, r, sigma, n_steps=12, n_paths=200_000, rng=1)
    times = np.linspace(0, T, 13)
    disc_mean = (np.exp(-r * times) * p).mean(axis=0)
    se = (np.exp(-r * times) * p).std(axis=0, ddof=1) / np.sqrt(p.shape[0])
    assert np.all(np.abs(disc_mean - S0) < 4 * se + 1e-12)


def test_log_return_variance():
    p = simulate_gbm_paths(S0, T, r, sigma, n_steps=4, n_paths=200_000, rng=2)
    log_ret = np.log(p[:, -1] / S0)
    assert abs(log_ret.var() - sigma**2 * T) < 0.002


def test_antithetic_pairs():
    p = simulate_gbm_paths(S0, T, r, sigma, n_steps=5, n_paths=6, antithetic=True, rng=3)
    drift = (r - 0.5 * sigma**2) * np.linspace(0, T, 6)
    a = np.log(p[:3] / S0) - drift
    b = np.log(p[3:] / S0) - drift
    assert np.allclose(a, -b)
    with pytest.raises(ValueError):
        simulate_gbm_paths(S0, T, r, sigma, 5, 7, antithetic=True)
