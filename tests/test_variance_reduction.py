import pytest

from mcpricer import (
    bs_price,
    price_european,
    price_european_antithetic,
    price_european_control_variate,
)

S0, K, T, r, sigma = 100.0, 105.0, 1.0, 0.04, 0.2
N = 200_000


@pytest.mark.parametrize("option", ["call", "put"])
@pytest.mark.parametrize("method", [price_european_antithetic, price_european_control_variate])
def test_unbiased_vs_black_scholes(method, option):
    res = method(S0, K, T, r, sigma, option=option, n_paths=N, rng=7)
    exact = bs_price(S0, K, T, r, sigma, option)
    assert abs(res.price - exact) < 4 * res.std_error


@pytest.mark.parametrize("option", ["call", "put"])
def test_antithetic_reduces_std_error(option):
    plain = price_european(S0, K, T, r, sigma, option=option, n_paths=N, rng=3)
    anti = price_european_antithetic(S0, K, T, r, sigma, option=option, n_paths=N, rng=3)
    assert anti.std_error < plain.std_error


def test_control_variate_strong_for_itm_call():
    # Deep in-the-money call is almost linear in S_T, so the control is highly correlated
    plain = price_european(S0, 70.0, T, r, sigma, n_paths=N, rng=5)
    cv = price_european_control_variate(S0, 70.0, T, r, sigma, n_paths=N, rng=5)
    assert cv.std_error < 0.2 * plain.std_error
