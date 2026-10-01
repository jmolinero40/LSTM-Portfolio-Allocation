"""Daily backtest: forecasts -> weights -> profit and loss, with costs.

Timing convention (one row = one common trading day)::

    row t-1 close  : all returns up to t-1 are known
                     mu_t, sigma_t forecast from rows <= t-1
                     correlation estimated on rows t-60 .. t-1
                     weights w_t decided
    row t          : portfolio earns  sum_i w_t,i * R_t,i  on RAW returns

With ``exec_lag=1`` the weights decided after t-1 are only applied to row
t+1. That is the conservative check for the one asynchrony this calendar has:
row t-1 contains the US close (22:00 CET), which is not yet known at the
European close (17:30 CET).

Costs are charged on traded notional. After a day of returns the weights
drift (an asset that rose is now a larger share of the portfolio), so the
trade needed tomorrow is measured from the drifted weights, not from
yesterday's targets::

    w~_t = w_t * (1 + R_t) / (1 + R_p,t)
    turnover_{t+1} = sum_i |w_{t+1,i} - w~_t,i|
    net_t = gross_t - cost * turnover_t

The thesis reported a turnover of sum |w_t - w_{t-1}| and no costs.
Cash earns zero (as in the thesis, rf = 0); short positions pay no borrow fee.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from lstm_portfolio.config import PortfolioConfig
from lstm_portfolio.portfolio import sharpe_like_model

__all__ = ["WeightPath", "compute_weights", "PnL", "evaluate_weights", "equal_weight"]


@dataclass(frozen=True)
class WeightPath:
    dates: pd.DatetimeIndex
    weights: np.ndarray  # (T, N) target weights decided for each date
    vol_daily: np.ndarray  # (T,) forecast daily portfolio volatility after targeting


def compute_weights(
    mu: np.ndarray,
    sigma: np.ndarray,
    corr_source: np.ndarray,
    rows: np.ndarray,
    dates: pd.DatetimeIndex,
    cfg: PortfolioConfig,
    short_allowed: bool,
) -> WeightPath:
    """Run the meta-model on every decision day.

    Args:
        mu, sigma: (T, N) forecasts for the decision rows, aligned with ``rows``.
        corr_source: (T_all, N) return matrix; the correlation window for row t
            is ``corr_source[t - corr_window : t]``.
        rows: row index (in ``corr_source``) of each decision day.
    """
    rows = np.asarray(rows, dtype=int)
    if rows.min() < cfg.corr_window:
        raise ValueError("Not enough history for the correlation window.")
    w = np.zeros_like(mu, dtype=float)
    vol = np.zeros(len(rows))
    for j, t in enumerate(rows):
        a = sharpe_like_model(
            mu[j], sigma[j], corr_source[t - cfg.corr_window : t], cfg.k,
            short_allowed=short_allowed, target_annual_vol=cfg.target_annual_vol,
            leverage_allowed=False, tau=cfg.tau, w_max=cfg.w_max,
            corr_lambda=cfg.corr_lambda, corr_shrinkage=cfg.corr_shrinkage,
        )  # fmt: skip
        w[j], vol[j] = a.weights, a.vol_daily
    return WeightPath(pd.DatetimeIndex(dates), w, vol)


@dataclass(frozen=True)
class PnL:
    gross: pd.Series  # daily simple return before costs
    turnover: pd.Series  # traded notional as a fraction of capital
    exposure: pd.Series  # gross exposure sum |w|

    def net(self, cost_bps: float) -> pd.Series:
        return self.gross - self.turnover * cost_bps / 1e4


def evaluate_weights(path: WeightPath, simple_returns: pd.DataFrame, exec_lag: int = 0) -> PnL:
    """Apply the weight path to realised simple returns.

    ``simple_returns`` must be raw (not winsorised) and indexed by date with
    the same columns as the weights. With ``exec_lag=k`` the weights decided
    for date d are held over the k-th trading day after d.
    """
    all_dates = simple_returns.index
    pos = all_dates.get_indexer(path.dates)
    if (pos < 0).any():
        raise ValueError("Decision dates missing from the return panel.")
    hold = pos + exec_lag
    keep = hold < len(all_dates)
    w = path.weights[keep]
    hold = hold[keep]
    r = simple_returns.to_numpy()[hold]

    gross = (w * r).sum(axis=1)
    drifted = w * (1.0 + r) / (1.0 + gross)[:, None]
    prev = np.vstack([np.zeros((1, w.shape[1])), drifted[:-1]])
    turnover = np.abs(w - prev).sum(axis=1)
    idx = all_dates[hold]
    return PnL(
        gross=pd.Series(gross, idx, name="gross"),
        turnover=pd.Series(turnover, idx, name="turnover"),
        exposure=pd.Series(np.abs(w).sum(axis=1), idx, name="exposure"),
    )


def equal_weight(dates: pd.DatetimeIndex, n_assets: int) -> WeightPath:
    """1/N, rebalanced daily (DeMiguel, Garlappi and Uppal, 2009)."""
    return WeightPath(
        pd.DatetimeIndex(dates), np.full((len(dates), n_assets), 1.0 / n_assets), np.full(len(dates), np.nan)
    )
