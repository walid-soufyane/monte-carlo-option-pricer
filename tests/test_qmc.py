import pytest

from mcpricer import bs_price, convergence_study, price_european, price_european_rqmc

S0, K, T, r, sigma = 100.0, 105.0, 1.0, 0.04, 0.2


@pytest.mark.parametrize("option", ["call", "put"])
def test_rqmc_unbiased(option):
    res = price_european_rqmc(S0, K, T, r, sigma, option, m=12, n_scrambles=32, rng=3)
    assert abs(res.price - bs_price(S0, K, T, r, sigma, option)) < 4 * res.std_error


def test_rqmc_beats_mc_at_same_budget():
    rq = price_european_rqmc(S0, K, T, r, sigma, m=13, n_scrambles=16, rng=4)  # 131,072 points
    mc = price_european(S0, K, T, r, sigma, n_paths=rq.n_paths, rng=4)
    assert rq.std_error < 0.1 * mc.std_error


def test_convergence_rates():
    out = convergence_study(S0, K, T, r, sigma, ms=range(8, 15), n_reps=16, rng=1)
    assert -0.7 < out["mc_slope"] < -0.3   # theory: -1/2
    assert out["qmc_slope"] < -0.8         # close to -1
