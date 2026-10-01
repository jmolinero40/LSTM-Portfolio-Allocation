"""Performance, forecast-quality and risk-model statistics.

Conventions: daily simple returns, 252 days per year, risk-free rate 0 (as in
the thesis). Sharpe is mean/sd * sqrt(252) with ddof=1.

Inference. With 9 years of daily data a Sharpe ratio has a standard error of
roughly 0.3, so point estimates alone say little. Differences between
strategies are tested with a moving-block bootstrap of the *paired* daily
returns (the same resampled days for both series), which keeps the
autocorrelation and the cross-correlation between them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

__all__ = [
    "performance",
    "sharpe",
    "max_drawdown",
    "block_bootstrap_sharpe_diff",
    "directional_accuracy",
    "information_coefficient",
    "qlike",
    "diebold_mariano",
    "var_backtest",
]

DAYS = 252


def sharpe(r) -> float:
    r = np.asarray(r, dtype=float)
    sd = r.std(ddof=1)
    return float(r.mean() / sd * np.sqrt(DAYS)) if sd > 0 else float("nan")


def max_drawdown(r) -> float:
    eq = np.cumprod(1.0 + np.asarray(r, dtype=float))
    return float((eq / np.maximum.accumulate(eq) - 1.0).min())


def performance(r: pd.Series, turnover: pd.Series | None = None) -> dict[str, float]:
    r = pd.Series(r).astype(float)
    n = len(r)
    growth = float(np.prod(1.0 + r.to_numpy()))
    cagr = growth ** (DAYS / n) - 1.0
    mdd = max_drawdown(r)
    out = {
        "days": n,
        "cagr": cagr,
        "vol": float(r.std(ddof=1) * np.sqrt(DAYS)),
        "sharpe": sharpe(r),
        "max_dd": mdd,
        "calmar": cagr / abs(mdd) if mdd < 0 else float("nan"),
        "hit_rate": float((r > 0).mean()),
    }
    if turnover is not None:
        out["turnover"] = float(turnover.mean())
    return out


def _block_indices(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    starts = rng.integers(0, n - block + 1, size=int(np.ceil(n / block)))
    return (starts[:, None] + np.arange(block)[None, :]).ravel()[:n]


def block_bootstrap_sharpe_diff(
    a, b, block: int = 20, samples: int = 5000, seed: int = 0
) -> dict[str, float]:
    """Sharpe(a) - Sharpe(b) with a paired moving-block bootstrap.

    Returns the point estimate, a 95% percentile interval and the two-sided
    bootstrap p-value of the null "difference = 0".
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape:
        raise ValueError("Series must be aligned.")
    rng = np.random.default_rng(seed)
    n = len(a)
    diffs = np.empty(samples)
    for i in range(samples):
        ix = _block_indices(n, block, rng)
        diffs[i] = sharpe(a[ix]) - sharpe(b[ix])
    est = sharpe(a) - sharpe(b)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p = 2.0 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return {"diff": est, "ci_low": float(lo), "ci_high": float(hi), "p_value": float(min(p, 1.0))}


# --------------------------------------------------------------------------
# Forecast quality
# --------------------------------------------------------------------------


def directional_accuracy(pred: np.ndarray, realised: np.ndarray) -> float:
    """Share of (day, asset) pairs where the forecast has the realised sign.

    Days with a realised return of exactly zero are excluded.
    """
    p, y = np.asarray(pred).ravel(), np.asarray(realised).ravel()
    m = y != 0
    return float((np.sign(p[m]) == np.sign(y[m])).mean())


def information_coefficient(pred: np.ndarray, realised: np.ndarray) -> dict[str, float]:
    """Daily cross-sectional Spearman correlation between forecast and outcome.

    This is what a ranking strategy actually uses: not whether each forecast
    has the right sign, but whether assets forecast higher do better than
    assets forecast lower *on the same day*. Reported as the mean daily IC and
    its t-statistic across days.
    """
    ics = []
    for p, y in zip(np.asarray(pred), np.asarray(realised), strict=True):
        if np.std(p) > 0 and np.std(y) > 0:
            ics.append(stats.spearmanr(p, y).statistic)
    ics = np.asarray(ics)
    return {"ic_mean": float(ics.mean()), "ic_t": float(ics.mean() / ics.std(ddof=1) * np.sqrt(len(ics)))}


def qlike(sigma: np.ndarray, realised: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """QLIKE loss log(v) + r^2/v per observation (lower is better)."""
    v = np.asarray(sigma, float) ** 2 + eps
    return np.log(v) + np.asarray(realised, float) ** 2 / v


def diebold_mariano(loss_a: np.ndarray, loss_b: np.ndarray, lags: int = 5) -> dict[str, float]:
    """DM test on the loss differential a - b (negative statistic: a is better).

    Uses a Newey-West (Bartlett) long-run variance with ``lags`` lags.
    Losses are first averaged across assets for each day.
    """
    d = np.asarray(loss_a, float) - np.asarray(loss_b, float)
    if d.ndim == 2:
        d = d.mean(axis=1)
    n = len(d)
    dc = d - d.mean()
    lrv = dc @ dc / n
    for k in range(1, lags + 1):
        lrv += 2.0 * (1.0 - k / (lags + 1)) * (dc[k:] @ dc[:-k]) / n
    stat = d.mean() / np.sqrt(lrv / n)
    return {"dm_stat": float(stat), "p_value": float(2.0 * stats.norm.sf(abs(stat)))}


# --------------------------------------------------------------------------
# VaR backtesting
# --------------------------------------------------------------------------


def var_backtest(returns: np.ndarray, vol_daily: np.ndarray, alpha: float = 0.01) -> dict[str, float]:
    """Parametric VaR backtest: Kupiec (coverage) and Christoffersen (independence).

    VaR_t = 1 - exp(z_alpha * sigma_t), a positive loss threshold (log-normal
    approximation, as in the thesis). An exception is a day with loss > VaR_t.

    The Christoffersen statistic here is the textbook one. The thesis version
    counted the n01 and n11 transitions twice in the restricted likelihood,
    which inflates the statistic.
    """
    r = np.asarray(returns, float)
    var = 1.0 - np.exp(stats.norm.ppf(alpha) * np.asarray(vol_daily, float))
    hits = (-r > var).astype(int)
    n, x = len(hits), int(hits.sum())
    out = {"T": n, "exceptions": x, "rate": x / n}

    def ll(p, k, m):  # log-likelihood of k successes in m Bernoulli(p) trials
        p = min(max(p, 1e-12), 1 - 1e-12)
        return k * np.log(p) + (m - k) * np.log(1 - p)

    lr_uc = -2.0 * (ll(alpha, x, n) - ll(x / n, x, n))
    out["kupiec_lr"] = float(lr_uc)
    out["kupiec_p"] = float(stats.chi2.sf(lr_uc, 1))

    prev, cur = hits[:-1], hits[1:]
    n00 = int(((prev == 0) & (cur == 0)).sum())
    n01 = int(((prev == 0) & (cur == 1)).sum())
    n10 = int(((prev == 1) & (cur == 0)).sum())
    n11 = int(((prev == 1) & (cur == 1)).sum())
    pi = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)
    pi0 = n01 / max(n00 + n01, 1)
    pi1 = n11 / max(n10 + n11, 1)
    l0 = ll(pi, n01 + n11, n00 + n01 + n10 + n11)
    l1 = ll(pi0, n01, n00 + n01) + ll(pi1, n11, n10 + n11)
    lr_ind = -2.0 * (l0 - l1)
    out["christoffersen_lr"] = float(lr_ind)
    out["christoffersen_p"] = float(stats.chi2.sf(lr_ind, 1))
    return out
