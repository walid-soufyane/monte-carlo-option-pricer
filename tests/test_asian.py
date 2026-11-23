import pytest

from mcpricer import bs_price, geometric_asian_price, price_asian

S0, K, T, r, sigma = 100.0, 100.0, 1.0, 0.04, 0.25


@pytest.mark.parametrize("option", ["call", "put"])
def test_geometric_mc_matches_closed_form(option):
    exact = geometric_asian_price(S0, K, T, r, sigma, n_steps=12, option=option)
    res = price_asian(S0, K, T, r, sigma, 12, option, "geometric", n_paths=300_000, rng=11)
    assert abs(res.price - exact) < 4 * res.std_error


def test_single_date_geometric_is_black_scholes():
    # With one monitoring date the average is just S_T
    assert geometric_asian_price(S0, K, T, r, sigma, 1) == pytest.approx(bs_price(S0, K, T, r, sigma))


def test_ordering_geometric_arithmetic_european():
    # AM-GM: arithmetic >= geometric average, and averaging lowers volatility vs European
    geo = geometric_asian_price(S0, K, T, r, sigma, 12)
    arith = price_asian(S0, K, T, r, sigma, 12, n_paths=300_000, rng=12)
    assert geo < arith.price < bs_price(S0, K, T, r, sigma)
