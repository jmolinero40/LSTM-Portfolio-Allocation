"""Walk-forward fold schedule.

One fold per test year, expanding window::

    rows:  0 ............ val_start ........... pre_end | test year Y
           [  fit (85%)  ][ validation (15%)  ]         [  test   ]
           scalers, winsorisation,   early stopping        reported
           network weights           and LR schedule

Everything the model learns or estimates (weights, scalers, winsorisation
quantiles) comes from the *fit* rows. The validation rows only decide when to
stop training. Test rows are never seen until prediction. The next fold
re-trains from scratch with one more year of history.

This replaces the thesis protocol, where a single model trained on 2014-2022
was then backtested over 2014-2025: roughly 83% of the reported backtest
overlapped the training and validation data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from lstm_portfolio.config import FeatureConfig
from lstm_portfolio.features import Winsorizer, ema_lagged

__all__ = ["Fold", "make_folds", "FoldData", "prepare_fold"]


@dataclass(frozen=True)
class Fold:
    index: int
    test_year: int
    val_start: int  # first validation row
    pre_end: int  # one past the last pre-test row (= first test row)
    test_rows: np.ndarray

    def targets(self, lookback: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(train, val, test) target rows for a network with this lookback."""
        train = np.arange(lookback, self.val_start)
        val = np.arange(max(lookback, self.val_start), self.pre_end)
        return train, val, self.test_rows


def make_folds(dates: pd.DatetimeIndex, test_years: list[int], val_fraction: float) -> list[Fold]:
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be in (0, 1)")
    years = np.asarray(dates.year)
    folds = []
    for i, y in enumerate(test_years):
        test_rows = np.flatnonzero(years == y)
        if test_rows.size == 0:
            raise ValueError(f"No data in test year {y}.")
        pre_end = int(test_rows[0])
        if not np.all(np.diff(test_rows) == 1):
            raise ValueError("Dates are not sorted.")
        val_start = int(round(pre_end * (1.0 - val_fraction)))
        folds.append(Fold(i, int(y), val_start, pre_end, test_rows))
    return folds


@dataclass(frozen=True)
class FoldData:
    """Model inputs for one fold, all derived from fit-period statistics only."""

    returns_w: np.ndarray  # (T, N) log returns winsorised with fit-period quantiles
    ema_lag: np.ndarray  # (T, N) EMA_{t-1} of returns_w


def prepare_fold(log_returns: np.ndarray, fold: Fold, cfg: FeatureConfig) -> FoldData:
    w = Winsorizer(cfg.winsor_lower, cfg.winsor_upper).fit(log_returns[: fold.val_start])
    rw = w.transform(log_returns)
    return FoldData(returns_w=rw, ema_lag=ema_lagged(rw, span=cfg.ema_span))
