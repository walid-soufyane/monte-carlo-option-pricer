# Monte Carlo Option Pricer

[![tests](https://github.com/walid-soufyane/monte-carlo-option-pricer/actions/workflows/tests.yml/badge.svg)](https://github.com/walid-soufyane/monte-carlo-option-pricer/actions/workflows/tests.yml)

Pricing options by Monte Carlo simulation under risk-neutral geometric Brownian motion, checked against Black-Scholes closed forms.

## Current features
- Exact simulation of terminal prices under GBM (with dividend yield)
- European call/put pricing with standard error and 95% confidence interval
- Black-Scholes closed form as a benchmark
- Variance reduction: antithetic variates and control variates (discounted S_T as control, optimal b estimated from the sample)
- Exact GBM path simulation on a time grid (optionally antithetic)
- Asian options (arithmetic and geometric average): Monte Carlo, geometric closed form, and the geometric Asian as a control variate for the arithmetic one (about 30x lower standard error)
- Barrier options (up/down, in/out) with Brownian-bridge correction for continuous monitoring, checked against the closed-form down-and-out call
- Tests: MC vs closed forms, put-call parity, O(1/√N) convergence, martingale property, AM-GM ordering, knock-in + knock-out = vanilla. Run on every push by GitHub Actions

## Quick start
```python
from mcpricer import price_european, bs_price

res = price_european(S0=100, K=105, T=1, r=0.04, sigma=0.2, option="call", n_paths=200_000, rng=42)
print(res.price, res.conf_int, bs_price(100, 105, 1, 0.04, 0.2, "call"))
```

Run tests: `pip install -r requirements.txt && pytest`

## Variance reduction example
ATM-ish call (S0=100, K=105, T=1, r=4%, σ=20%), 200,000 simulated prices, Black-Scholes = 7.567:

| Method | Price | Std. error |
|---|---|---|
| Plain MC | 7.519 | 0.0287 |
| Antithetic | 7.529 | 0.0232 |
| Control variate | 7.550 | 0.0135 |

The control variate halves the standard error here, the same as running about 4x more paths.

## Exotic options example
Arithmetic Asian call (S0=K=100, T=1, r=4%, σ=25%, 12 monthly fixings), 100,000 paths:

| Method | Price | Std. error |
|---|---|---|
| Plain MC | 7.025 | 0.0329 |
| Geometric control variate | 7.063 | 0.0011 |

Down-and-out call (S0=K=100, H=90, T=1, r=4%, σ=25%), 50 monitoring steps, closed form = 8.702:

| Method | Price | Std. error |
|---|---|---|
| Grid check only | 9.634 | 0.057 |
| Brownian-bridge correction | 8.776 | 0.055 |

Checking the barrier only at grid dates misses crossings between them and overprices the knock-out by about 11%. The bridge correction removes that bias.

## Roadmap
- [x] European options + Black-Scholes benchmark
- [x] Variance reduction: antithetic variates, control variates
- [x] Path simulation + Asian options (arithmetic and geometric)
- [x] Barrier options (knock-in / knock-out) with Brownian-bridge correction
- [ ] Greeks: bump-and-reprice, pathwise and likelihood-ratio estimators
- [ ] Quasi-Monte Carlo (Sobol) vs pseudo-random convergence study
- [ ] Heston stochastic volatility model
- [ ] Convergence plots and a results notebook
