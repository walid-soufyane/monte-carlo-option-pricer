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
- Greeks three ways: bump-and-reprice with common random numbers, pathwise (IPA), and likelihood ratio (works for digitals), checked against analytic Black-Scholes Greeks
- Randomised quasi-Monte Carlo with scrambled Sobol points, plus a convergence study against plain MC
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

## Greeks example
ATM call (S0=K=100, T=1, r=4%, σ=20%), 200,000 paths, std. error in brackets:

| Greek | Black-Scholes | Bump (CRN) | Pathwise | Likelihood ratio |
|---|---|---|---|---|
| Delta | 0.6179 | 0.6169 | 0.6170 (0.0013) | 0.6137 (0.0032) |
| Gamma | 0.0191 | 0.0191 | n/a | 0.0189 (0.0003) |
| Vega | 38.14 | 37.86 | 37.86 (0.17) | 37.90 (0.61) |

Pathwise is the most precise but needs a payoff that is continuous in S_T. Likelihood ratio is noisier but works for any payoff, including digitals.

## Quasi-Monte Carlo convergence
European call (S0=100, K=105, T=1, r=4%, σ=20%), RMSE vs Black-Scholes over 20 repetitions:

| N | Plain MC | Scrambled Sobol |
|---|---|---|
| 256 | 0.717 | 0.0450 |
| 4,096 | 0.160 | 0.0025 |
| 65,536 | 0.050 | 0.00015 |

Fitted log-log slopes: **−0.51** for MC (theory −1/2) and **−1.05** for Sobol (near −1). At 65k points Sobol is about 300x more accurate.

## Roadmap
- [x] European options + Black-Scholes benchmark
- [x] Variance reduction: antithetic variates, control variates
- [x] Path simulation + Asian options (arithmetic and geometric)
- [x] Barrier options (knock-in / knock-out) with Brownian-bridge correction
- [x] Greeks: bump-and-reprice, pathwise and likelihood-ratio estimators
- [x] Quasi-Monte Carlo (Sobol) vs pseudo-random convergence study
- [ ] Heston stochastic volatility model
- [ ] Convergence plots and a results notebook
