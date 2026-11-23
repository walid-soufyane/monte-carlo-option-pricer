import pytest

from mcpricer import bs_price, down_and_out_call, price_barrier

S0, K, H, T, r, sigma = 100.0, 100.0, 90.0, 1.0, 0.04, 0.25
N = 100_000


def test_bridge_matches_continuous_closed_form():
    exact = down_and_out_call(S0, K, H, T, r, sigma)
    res = price_barrier(S0, K, H, T, r, sigma, n_steps=50, n_paths=N, rng=21)
    assert abs(res.price - exact) < 4 * res.std_error


def test_discrete_monitoring_overprices_knock_out():
    exact = down_and_out_call(S0, K, H, T, r, sigma)
    naive = price_barrier(S0, K, H, T, r, sigma, n_steps=50, n_paths=N, bridge=False, rng=21)
    assert naive.price - exact > 4 * naive.std_error


@pytest.mark.parametrize("direction,barrier", [("down", 90.0), ("up", 120.0)])
def test_in_out_parity(direction, barrier):
    # knock-in + knock-out = vanilla, path by path with the same seed
    kw = dict(n_steps=50, n_paths=N, rng=22)
    out = price_barrier(S0, K, barrier, T, r, sigma, kind=f"{direction}-and-out", **kw)
    kin = price_barrier(S0, K, barrier, T, r, sigma, kind=f"{direction}-and-in", **kw)
    vanilla = bs_price(S0, K, T, r, sigma)
    se = out.std_error + kin.std_error
    assert abs(out.price + kin.price - vanilla) < 4 * se


def test_rejects_start_through_barrier():
    with pytest.raises(ValueError):
        price_barrier(85.0, K, H, T, r, sigma)
