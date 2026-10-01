# Audit of the thesis backtest

This document records what I found when I re-examined my bachelor's thesis
code to publish it. The thesis reported, for the long-short strategy on the
known assets, a Sharpe ratio of 1.17, a CAGR of 14.8% and a maximum drawdown
of −18% over 2014–2025. Those figures should not be read as out-of-sample
performance, for the reasons below.

Every number in this document is re-derived from the files the thesis
actually produced (the saved networks, scalers, return files and backtest
spreadsheets in [`legacy/artifacts/`](../legacy/artifacts)) by

```bash
python scripts/audit/reproduce_audit.py      # writes results/audit/audit.md
```

and the headline claims are asserted in [`tests/test_audit.py`](../tests/test_audit.py)
on every commit. The corrected evaluation is in the [README](../README.md).

## Summary

| # | Finding | Effect on the reported results | Fixed in this repository by |
|---|---|---|---|
| 1 | The backtest ran over the training period | 84% of backtest days preceded the test set | Annual walk-forward re-training ([`folds.py`](../src/lstm_portfolio/folds.py)) |
| 2 | The return target leaked the return being predicted | 70% directional accuracy is an artefact; 52% without the leak | Target `r_t − EMA_{t−1}` ([`features.py`](../src/lstm_portfolio/features.py)) |
| 3 | The risk network received unscaled inputs in the backtest | σ̂ was effectively constant per asset | Scaler stored with the network ([`models.py`](../src/lstm_portfolio/models.py)) |
| 4 | Profit and loss computed on winsorised returns | Tail losses capped; quantiles from the full sample | Raw returns for PnL; quantiles per fold |
| 5 | No transaction costs at 0.40–0.51 daily turnover | ~5–6% a year at 5 bps | Costs at 0, 5, 10 bps on drift-adjusted turnover |
| 6 | One training run, not reproducible | Same configuration retrained: Sharpe 1.16 → 0.76 | Three seeds per fold, forecasts committed |
| 7 | Smaller issues: VaR test formula, rounded volatility, leverage rate, figures | See below | Fixed or documented |

None of these change the *idea* of the system: two networks for the first two
moments and a Sharpe-like rule to combine them. They change what can be
concluded from its backtest.

---

## 1. The backtest ran over the training period

The networks were trained on a single 70/15/15 chronological split:
training data up to 19 Jan 2022, validation (used for early stopping and
learning-rate decay) up to 30 Nov 2023, test afterwards. The backtest loop
then ran over **every** window from January 2014. Only the last ~470 of
~2,900 backtest days were out of sample.

Splitting the thesis's own saved daily returns at those dates:

| Saved run | Whole period | Train period | Validation | Test (≥ Dec 2023) |
|---|---|---|---|---|
| Long-only, known assets (thesis table 1) | 1.03 | 1.12 | 0.44 | 1.39 |
| Long-short, known assets (thesis table 3) | **1.16** | **1.44** | 0.43 | 0.86 |
| Long-short, same config, retrained Jan 2026 | 0.76 | 1.00 | 0.03 | 0.59 |
| Long-only, unseen assets | 0.91 | 0.84 | 0.60 | 1.65 |

*Sharpe ratios, before costs. 83.7% of the days of each run fall before the test set.*

The pattern of the long-short run — strongest where the network was fitted,
weakest on the validation years — is what in-sample evaluation looks like.
The test-period numbers are not evidence of skill either: they cover under two
years, and the long-only strategy on assets the network had **never seen**
did better there (1.65) than on the assets it was trained on. Over 2024–2025
a long-only basket of these sectors made money whatever selected it; an
equal-weighted portfolio of the same 11 assets had a Sharpe ratio of 1.75
over the same window (raw returns, from the snapshot in `data/`), above
both thesis strategies.

This reframes one of the thesis's conclusions. It read the weaker results on
unseen assets as a failure to generalise *across assets*. An equally consistent
reading is that on known assets the network could reproduce the period it had
been fitted on, and on unseen assets it could not.

## 2. The return target leaked the return being predicted

The thesis (Appendix B.1.4) describes the fix for the collapsing return
network as predicting `r_t − EMA_{t−1}`. The training script
(`copia_retornos.py`) used `r_t − EMA_t` instead, and added `EMA_t` back when
evaluating. With `pandas.ewm(span=20, adjust=False)`,

```
EMA_t = α·r_t + (1 − α)·EMA_{t−1},     α = 2/21 ≈ 0.095
```

so the reconstructed forecast `ŷ_t + EMA_t` contains `α·r_t`, a scaled copy of
the very return it is supposed to predict. The sign of that term is always
right, which is what moved directional accuracy from the ~53% the thesis
first observed to ~70%.

The saved return network on its own test set:

| Forecast | Directional accuracy |
|---|---|
| Network + `EMA_t` (as trained and as reported) | **69.9%** |
| Network + `EMA_{t−1}` (no leak) | 51.8% |
| `EMA_{t−1}` alone, no network | 50.4% |
| Always predict "up" | 55.0% |

The same evaluation applied to **pure noise**, where nothing is predictable by
construction, gives 79.6% for a linear model and 62.4% for a model that
always predicts zero. Without the leak: 50.1%.

The thesis backtest did use `EMA_{t−1}` (the backtest script lags the EMA by
one row), so this leak inflated the reported accuracy but not the backtest
directly. It did leave a mismatch: the network was trained to predict
`r_t − EMA_t` and used as if it predicted `r_t − EMA_{t−1}`.

*(The thesis text gives the EMA window as 60 days in B.1.4 and 20 in
Appendix G; the code used 20.)*

## 3. The risk network received unscaled inputs in the backtest

`riesgos.py` scaled the risk network's inputs with a `RobustScaler` (dividing
by the interquartile range, ~0.01–0.02) but never saved it, and the backtest fed
raw returns to the network. The inputs arrived 50–100 times smaller than
anything seen in training: to the network, every day looked perfectly calm.

| Risk network on its test set | Scaled inputs (as trained) | Raw inputs (as backtested) |
|---|---|---|
| QLIKE | −7.83 (the thesis figure) | −7.58 |
| Variation of σ̂ over time (coefficient of variation) | 0.336 | **0.001** |
| Correlation of σ̂ with \|r\| | 0.39 | 0.22 |

The network evaluated in the thesis and the network used in the backtest were
therefore different objects. In the backtest σ̂ was a constant per asset, so
the "risk network" contributed a fixed per-asset scaling to the signal and to
the covariance matrix, and nothing that reacted to market conditions. The
thesis attributed the strategy's behaviour around 2020 partly to the risk
mechanism; that attribution does not hold.

## 4. Profit and loss on winsorised returns

The return file read by every script was winsorised at the 0.5% and 99.5%
quantiles of each asset over the **whole sample**. The backtest computed the
strategy's daily profit from that same file. Extreme days (March 2020
included) were therefore capped before being booked, which understates
drawdowns and tail losses, and the clipping thresholds used information from
the test period.

Winsorising model *inputs* is a reasonable choice and is kept here, but the
quantiles are now estimated on each fold's training window, and the money is
always counted on raw returns.

## 5. Transaction costs

The thesis reported turnover but no costs. A daily turnover of 0.40
(long-only) and 0.51 (long-short) is high: at 5 bps per unit traded it costs
about 5.0% and 6.4% a year, at 10 bps about 10% and 13%. These are not
exotic cost assumptions for daily rebalancing of single European stocks.
Turnover here is also measured properly, from drift-adjusted weights (see
[`backtest.py`](../src/lstm_portfolio/backtest.py)).

## 6. One training run, not reproducible

The reported Sharpe of 1.17 cannot be regenerated by anyone, including me:

- The return network that produced it was overwritten on 20 Jan 2026 when the
  training script was re-run (after the thesis was submitted). The surviving
  model file is the January one.
- Re-running the identical backtest configuration with the retrained network
  gave a Sharpe of **0.76** instead of 1.16 (the two daily return series have a
  correlation of 0.83). Which run gets reported was, in effect, a draw.
- I re-ran the thesis backtest code with the surviving network and data and
  reproduced the January run exactly (correlation 1.0000 over 2,880 days,
  maximum difference 8·10⁻⁸), so the code path itself is understood.
- The performance tables printed in the thesis do not exactly match any saved
  run either (e.g. long-short Sharpe 1.166 and drawdown −17.96% in the thesis,
  1.157 and −16.85% in the closest saved spreadsheet), so they come from a run
  whose output was also overwritten.
- Data were downloaded "until today" on every run, and the scripts picked their
  input file as `parquet_files[2]` — the third file in alphabetical order in
  the data folder. After a new download on 20 Jan 2026, that index still
  pointed at the older October file, so the January backtest silently ran on
  October data.

Single-seed results from a network trained once are not a sound basis for a
Sharpe ratio to two decimals. This repository trains every fold with three
seeds and reports the spread.

## 7. Smaller issues

- **VaR independence test.** The Christoffersen statistic counted the `n01`
  and `n11` transitions twice in the restricted likelihood. On the thesis's
  own exception series the statistic is 19.83 (p = 8·10⁻⁶), not 41.16. The
  conclusion (exceptions cluster) stands. The thesis also printed p = 1.4·10⁻⁶
  where the code had computed 1.4·10⁻¹⁰.
- **Rounded volatility.** The meta-model returned the portfolio volatility
  rounded to three decimals. Over 2,895 days it took only 7 distinct values,
  which is why the thesis's volatility chart (figure 10) moves in steps of
  4.8, 6.3, 7.9, 9.5, 11.1, 12.7%. The VaR test used these rounded values.
- **Realised volatility chart.** Figure 11 is the rolling volatility of the
  strategy's own returns, not of the market. It shows the long-only strategy's
  realised volatility reaching 37% in March 2020 against a 15% target (and
  that is on winsorised returns): the volatility targeting failed
  ex post when it mattered most, consistent with point 3 and with the VaR
  exceptions (6% of days against an expected 1%), which the thesis did report.
- **Leverage cost.** The leveraged backtest (Appendix F) charged
  `rf = rl = 0.02/12` per **day**, i.e. a 52% annual rate instead of 2%.
  Figure 19's collapse is mostly that.
- **Predicted-versus-realised price charts.** Figures 8 and 16 were drawn
  after removing the forecast's average bias using realised returns over the
  whole period, and re-anchoring the predicted price to the actual price every
  month. A visually close fit is largely guaranteed by that procedure; the
  charts say little about forecasting ability.
- **Benchmarks.** The strategy (total return, dividends included, local
  currencies with no FX) was compared with the S&P 500 and IBEX 35 *price*
  indices, which exclude dividends. Here the benchmark is an equal-weighted
  portfolio of the same 11 assets, computed identically.
- **Close-time asynchrony.** A trading day's row contains the US close
  (22:00 CET), which is not known at the European close (17:30 CET). Deciding
  on European weights with that row uses about four hours of future US
  information. The corrected evaluation reports a variant with execution
  delayed by one day as a check.

## What was correct

The date alignment between the two networks and the realised return is right:
both input windows end the day before the return being traded. The covariance
matrix uses only past returns. Training scalers were fitted on training data
only, and the networks were trained without shuffling across the time split.
The idea and its implementation are sound; the evaluation protocol was not.
