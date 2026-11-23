# Monte Carlo Option Pricer

Pricing options by Monte Carlo simulation under risk-neutral geometric Brownian motion, checked against Black-Scholes closed forms.

## Current features
- Exact simulation of terminal prices under GBM (with dividend yield)
- European call/put pricing with standard error and 95% confidence interval
- Black-Scholes closed form as a benchmark
- Tests: MC vs closed form, put-call parity, O(1/√N) convergence of the standard error

## Quick start
```python
from mcpricer import price_european, bs_price

res = price_european(S0=100, K=105, T=1, r=0.04, sigma=0.2, option="call", n_paths=200_000, rng=42)
print(res.price, res.conf_int, bs_price(100, 105, 1, 0.04, 0.2, "call"))
```

Run tests: `pip install -r requirements.txt && pytest`

## Roadmap
- [x] European options + Black-Scholes benchmark
- [ ] Variance reduction: antithetic variates, control variates
- [ ] Path simulation + Asian options (arithmetic and geometric)
- [ ] Barrier options (knock-in / knock-out) with Brownian-bridge correction
- [ ] Greeks: bump-and-reprice, pathwise and likelihood-ratio estimators
- [ ] Quasi-Monte Carlo (Sobol) vs pseudo-random convergence study
- [ ] Heston stochastic volatility model
- [ ] Convergence plots and a results notebook
