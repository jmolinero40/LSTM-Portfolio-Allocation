# Method

This document describes exactly what the code does and why. The thesis
(Chapter 3 and Appendices B–C) motivates the design; here the emphasis is on
the protocol and on every point where this repository departs from the
original scripts.

## 1. Data and calendar

Eleven assets (seven US sector ETFs, four European blue chips), adjusted
closes, 2013–2025, frozen in [`data/snapshot/`](../data). Only days on which
all eleven markets traded are kept; daily log returns are computed on that
common calendar. See [`data/README.md`](../data/README.md).

## 2. The two networks

Same architecture for both, as in the thesis:

```
Input(lookback, 11) → LSTM(64, return sequences) → Dropout(0.2) → LSTM(32) → Dense(11)
L2(1e-5) on LSTM kernels · Adam(1e-3) · batch 32 · up to 200 epochs
EarlyStopping(patience 15, restore best) · ReduceLROnPlateau(×0.5, patience 7, min 1e-5)
```

| | Return network | Risk network |
|---|---|---|
| Lookback | 60 days | 240 days |
| Input scaling | `RobustScaler()` | `RobustScaler(with_centering=False)` |
| Target | `r_t − EMA_{t−1}` (span 20), scaled by its IQR, not centred | `r_t` |
| Loss | Huber(δ = 1) | Gaussian NLL, `v = softplus(output)` |
| Forecast | `μ̂_t = output (unscaled) + EMA_{t−1}` | `σ̂_t = sqrt(softplus(output))` |

Why these choices (from the thesis): the IQR scaling without centring and the
EMA baseline stop the return network collapsing to the training median; the
Gaussian NLL makes the risk network trade off over- and under-estimation of
variance, and softplus keeps the variance positive. Scalers live inside the
forecaster objects ([`models.py`](../src/lstm_portfolio/models.py)), so a
network can never be used without the scaling it was trained with.

**One change to training.** The risk network's output bias is initialised
at the training-sample variance of each asset (`softplus⁻¹(mean r²)`)
instead of zero. The output has to reach about `softplus⁻¹(10⁻⁴) ≈ −9.2`;
starting from 0 with Adam's step of 10⁻³ that takes ~9,000 updates, which is
more than 200 epochs provide in the early folds (with the thesis
initialisation, the 2017 fold, seed 0, was still improving at epoch 200 and
ended at a worse validation loss, −4.015, than the initialised network reaches
in four epochs, −4.043; 491 s of training against 49 s). It changes where optimisation starts, not the model.
Set `risk_model.init_output_bias: false` to train exactly as the thesis did.

The architecture and loss are checked against the saved thesis models in
[`tests/test_thesis_equivalence.py`](../tests/test_thesis_equivalence.py).

## 3. The meta-model

Unchanged from the thesis ([`portfolio.py`](../src/lstm_portfolio/portfolio.py),
tested against the original `Carteras.py`). For each day:

1. Signal `s_i = μ̂_i / (σ̂_i + ε)`.
2. Long-only: the `k = 5` largest positive signals. Long-short: the 5 largest
   `|s_i|`, split into a long and a short leg, each leg getting a share of gross
   exposure proportional to its total `|s|`.
3. Weights inside the selection (or leg): `softmax(τ·|s|)`, `τ = 6`.
4. Cap `|w_i| ≤ 0.3`. Whatever the cap removes stays in cash (no renormalising).
5. Covariance `Σ = D(σ̂) · C · D(σ̂)`, where `C` is an exponentially weighted
   (λ = 0.97) correlation of the last 60 days of returns, shrunk 30% towards
   the identity.
6. If the forecast annual volatility exceeds 15%, scale all weights down to
   hit it. Never up (no leverage).

## 4. Walk-forward protocol

One fold per calendar year, 2017 to 2025, expanding window:

```
2013 ─────────────────────── fit (85%) ──────────── │ val (15%) │ test year
     weights, scalers, winsorisation quantiles      │ early stop │ forecasts
```

- Both networks are retrained from scratch for every fold and every seed
  (0, 1, 2): 9 × 3 × 2 = 54 trainings.
- Everything estimated — network weights, scalers, winsorisation quantiles —
  comes from the fit rows. Validation rows only decide when to stop training
  and when to reduce the learning rate. Test rows are first touched at
  prediction time.
- Input windows for a forecast on day `t` end on day `t − 1`.
- 2017 is the first test year so that the first fold has three years of
  history for the 240-day risk network.

[`tests/test_no_lookahead.py`](../tests/test_no_lookahead.py) enforces this
by perturbing all data after a given day and checking that no forecast or
weight up to that day changes, including through a full retraining.

## 5. Accounting

Weights decided after the close of day `t − 1` earn the **raw** simple
returns of day `t`. Cash earns zero and shorting is free (as in the thesis).
After each day the weights drift with prices; the trade needed the next day,
and hence turnover, is measured from the drifted weights. Costs of 0, 5 and
10 bps per unit of turnover are deducted. A variant applies each day's
weights one day later, to bound the effect of the US/European close-time
mismatch.

## 6. Ablations and benchmark

The meta-model is held fixed and only the source of μ and σ changes:

| | μ from | σ from |
|---|---|---|
| Thesis system | return LSTM | risk LSTM |
| | return LSTM | EWMA (λ = 0.97) |
| | `EMA_{t−1}` (span 20) | risk LSTM |
| No neural network | `EMA_{t−1}` | EWMA |

and the benchmark is 1/N over the same 11 assets, rebalanced daily, with the
same accounting (DeMiguel, Garlappi and Uppal, 2009, *Review of Financial
Studies*, show how hard 1/N is to beat out of sample).

## 7. Inference

With nine years of daily data, the standard error of an annualised Sharpe
ratio is roughly 0.35. Differences between strategies are assessed with a paired moving-block
bootstrap (blocks of 20 days, 5,000 resamples) of the daily returns of the
two strategies, averaged over seeds. Forecast quality is measured directly:
directional accuracy against the "always up" rate, the daily cross-sectional
rank correlation between forecast and realised return (information
coefficient), QLIKE against an EWMA volatility with a Diebold–Mariano test,
and a 1% VaR backtest of the forecast portfolio volatility (Kupiec and
Christoffersen).

## 8. Differences from the thesis code

| | Thesis code | Here | Why |
|---|---|---|---|
| Evaluation | One 70/15/15 split, backtest over all of 2014–2025 | Annual walk-forward, test years only | Audit §1 |
| Return target | `r_t − EMA_t` | `r_t − EMA_{t−1}` | Audit §2; the thesis text describes `EMA_{t−1}` |
| Risk-network scaler | Fitted, not saved, not applied in backtest | Stored with the network | Audit §3 |
| Winsorisation | Full-sample quantiles, also applied to PnL | Fit-window quantiles, inputs only | Audit §4 |
| Costs | None | 0 / 5 / 10 bps on drift-adjusted turnover | Audit §5 |
| Seeds | One run | Three per fold | Audit §6 |
| Risk-network output bias | Zero | Training-sample variance | §2; convergence within 200 epochs |
| Data | Downloaded "until today" each run | Frozen snapshot, fixed end, hashed | Audit §6 |
| Portfolio volatility | Rounded to 3 decimals | Unrounded | Audit §7 |
| Christoffersen test | Double-counted transitions | Textbook formula | Audit §7 |
| Benchmark | S&P 500 / IBEX 35 price indices | 1/N on the same assets | Audit §7 |
| `k` | 5 in the scripts, 4 in the thesis text | 5 | Matches the thesis results |
| Correlation window | 60 days in the scripts, 240 in the thesis text | 60 | Matches the thesis results |
| EMA span | 20 in the scripts and Appendix G, 60 in B.1.4 | 20 | Matches the thesis results |

## 9. Known limitations

- **Small universe.** Eleven assets give the cross-sectional rule little to
  choose from; on most days the top-5 selection covers almost half the
  universe.
- **No currency conversion.** Returns are in local currency (USD, EUR, CHF),
  which amounts to a free currency hedge.
- **No borrowing cost for shorts, cash earns zero.** Both favour the
  long-short strategy slightly over 2023–2025, when cash yielded 4–5%.
- **Hyperparameters are the thesis ones**, chosen by the author on a single
  split that overlapped the later test years. They were not re-tuned here, to
  avoid tuning on the walk-forward test years; a nested tuning loop would be
  the rigorous alternative and was not run.
- **CPU training is deterministic per machine, not across machines.** Rerunning
  the walk-forward on different hardware or library versions changes each
  seed's results, by amounts comparable to the spread between seeds. The committed forecasts are what the tables are
  computed from.
