"""Feature transformations, all fitted on the training window only.

Three things the thesis code did globally are done per fold here:

* **Winsorisation.** The thesis clipped every return series at its 0.5% and
  99.5% quantiles computed over the *whole* sample, test period included, and
  then also used the clipped returns to compute the strategy's profit and loss.
  Here the quantiles come from the training window, and clipping only touches
  model inputs and targets; the PnL uses raw returns.
* **EMA baseline.** The return network predicts ``r_t - EMA_{t-1}``. The thesis
  trained on ``r_t - EMA_t``; because ``EMA_t = a*r_t + (1-a)*EMA_{t-1}``, that
  target contains the very return being predicted, and adding ``EMA_t`` back at
  evaluation leaks ``a*r_t`` into the forecast (this is what produced the 70%
  directional accuracy). :func:`ema_lagged` only ever uses information up to t-1.
* **Windows.** :func:`make_windows` builds the input for target day t from rows
  t-L .. t-1, never row t.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = ["Winsorizer", "ema_lagged", "ewma_sigma", "make_windows"]


@dataclass
class Winsorizer:
    """Column-wise clipping at quantiles estimated on a fitting sample."""

    lower: float = 0.005
    upper: float = 0.995
    lo_: np.ndarray | None = None
    hi_: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> Winsorizer:
        x = np.asarray(x, dtype=float)
        self.lo_ = np.quantile(x, self.lower, axis=0)
        self.hi_ = np.quantile(x, self.upper, axis=0)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.lo_ is None:
            raise RuntimeError("Winsorizer is not fitted.")
        return np.clip(np.asarray(x, dtype=float), self.lo_, self.hi_)


def ema_lagged(returns: np.ndarray, span: int = 20) -> np.ndarray:
    """``EMA_{t-1}`` aligned on row t (row 0 and the warm-up rows are 0).

    The EMA itself is the thesis one: ``pandas.ewm(span, adjust=False,
    min_periods=span)`` with missing warm-up values set to 0. It is shifted down
    one row so that row t only contains information up to t-1.
    """
    ema = pd.DataFrame(returns).ewm(span=span, adjust=False, min_periods=span).mean()
    ema = np.nan_to_num(ema.to_numpy(), nan=0.0)
    out = np.zeros_like(ema)
    out[1:] = ema[:-1]
    return out


def ewma_sigma(returns: np.ndarray, lam: float = 0.97, init_rows: int = 250) -> np.ndarray:
    """RiskMetrics-style EWMA volatility, row t uses returns up to t-1.

    ``var_t = lam * var_{t-1} + (1 - lam) * r_{t-1}^2``. During the first
    ``init_rows`` rows the recursion has not warmed up, so ``var_t`` is the
    mean of ``r^2`` over rows 0..t-1 instead (still past data only). Row 0 has
    no past and is NaN; it is never a decision day. This is the no-network
    volatility baseline (lambda = 0.97, the thesis's diagnostic baseline).
    """
    r = np.asarray(returns, dtype=float)
    var = np.full_like(r, np.nan)
    csum = np.cumsum(r**2, axis=0)
    for t in range(1, len(r)):
        if t <= init_rows:
            var[t] = csum[t - 1] / t
        else:
            var[t] = lam * var[t - 1] + (1.0 - lam) * r[t - 1] ** 2
    return np.sqrt(var)


def make_windows(values: np.ndarray, targets: np.ndarray, lookback: int) -> np.ndarray:
    """Input windows for the given target rows.

    Returns an array of shape (len(targets), lookback, N) where window j holds
    rows ``targets[j]-lookback .. targets[j]-1``. Row ``targets[j]`` itself is
    never included: it is the day being predicted.
    """
    targets = np.asarray(targets, dtype=int)
    if targets.size and targets.min() < lookback:
        raise ValueError(f"Target row {targets.min()} has fewer than {lookback} rows of history.")
    if targets.size and targets.max() >= len(values):
        raise ValueError("Target row beyond the end of the data.")
    v = np.asarray(values, dtype=np.float32)
    return (
        np.stack([v[t - lookback : t] for t in targets])
        if targets.size
        else np.empty((0, lookback, v.shape[1]), np.float32)
    )
