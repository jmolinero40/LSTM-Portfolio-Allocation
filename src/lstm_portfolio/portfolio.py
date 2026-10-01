"""The Sharpe-inspired meta-model: from (mu, sigma) forecasts to portfolio weights.

This module is a line-by-line port of the thesis file ``Carteras.py``
(``legacy/thesis_code/Carteras.py``). The logic is unchanged; the equivalence is
asserted against the original file in ``tests/test_thesis_equivalence.py``.

The only deliberate difference is in :func:`sharpe_like_model`: the original
rounded the returned portfolio volatility to three decimals
(``np.round(vol_daily, 3)``). That value fed the plotted volatility and the VaR
backtest, which is why the thesis volatility plot shows discrete steps at
4.8%, 7.9%, 9.5%, ... annualised. Here the volatility is returned unrounded.
The weights were rounded to four decimals; that is also dropped (it changes
weights by at most 5e-5).

Pipeline for one decision day::

    s_i   = mu_i / (sigma_i + eps)                  Sharpe-like signal
    top-k assets by s (long-only) or |s| (long-short)
    w     = softmax(tau * s) inside the selection    concentration set by tau
    w_i   = min(w_i, w_max)                          cap, NOT renormalised -> cash
    Sigma = D(sigma) . Corr_EWMA . D(sigma)          joint risk model
    if annual vol(w) > target: w *= target / vol     volatility targeting
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "ewma_corr",
    "shrink_to_identity",
    "covariance_from_sigma_corr",
    "build_joint_risk_model",
    "weights_topk_softmax",
    "portfolio_volatility",
    "target_annual_volatility",
    "sharpe_like_model",
    "Allocation",
]

DAYS_PER_YEAR = 252


# --------------------------------------------------------------------------
# Joint risk model
# --------------------------------------------------------------------------


def ewma_corr(returns_hist: np.ndarray, lam: float = 0.97, eps: float = 1e-12) -> np.ndarray:
    """Exponentially weighted correlation matrix.

    Args:
        returns_hist: (T, N) daily returns ordered oldest -> newest. Only past
            data may be passed here; the caller is responsible for that.
        lam: decay. The most recent row gets weight lam**0, the oldest lam**(T-1).
    """
    r = np.asarray(returns_hist, dtype=float)
    t_len, _ = r.shape
    w = lam ** np.arange(t_len - 1, -1, -1)
    w = w / w.sum()
    m = (w[:, None] * r).sum(axis=0, keepdims=True)
    x = r - m
    cov = (x.T * w) @ x
    vol = np.sqrt(np.clip(np.diag(cov), eps, None))
    d_inv = np.diag(1.0 / vol)
    corr = d_inv @ cov @ d_inv
    return np.clip(corr, -1.0, 1.0)


def shrink_to_identity(corr: np.ndarray, alpha: float = 0.3) -> np.ndarray:
    """(1 - alpha) * corr + alpha * I. Pulls noisy off-diagonals towards zero."""
    n = corr.shape[0]
    return (1.0 - alpha) * corr + alpha * np.eye(n)


def covariance_from_sigma_corr(sigma_hat: np.ndarray, corr: np.ndarray) -> np.ndarray:
    """Sigma = D(sigma_hat) . corr . D(sigma_hat)."""
    d = np.diag(np.asarray(sigma_hat, dtype=float).ravel())
    return d @ corr @ d


def build_joint_risk_model(
    sigma_hat_now: np.ndarray,
    returns_hist_window: np.ndarray,
    lam: float = 0.97,
    alpha: float = 0.3,
) -> tuple[np.ndarray, np.ndarray]:
    """Covariance from the forecast volatilities and a shrunk EWMA correlation."""
    corr = shrink_to_identity(ewma_corr(returns_hist_window, lam=lam), alpha=alpha)
    return covariance_from_sigma_corr(sigma_hat_now, corr), corr


# --------------------------------------------------------------------------
# Weights
# --------------------------------------------------------------------------


def weights_topk_softmax(
    mu: np.ndarray,
    sigma: np.ndarray,
    k: int,
    tau: float = 1.0,
    w_max: float | None = None,
    eps: float = 1e-12,
    short_allowed: bool = False,
):
    """Top-k selection on the Sharpe-like signal and a softmax with temperature.

    Long-only: keep assets with s > 0, take the k largest, softmax(tau * s).
    Long-short: take the k largest |s|, split them into a long and a short leg,
    give each leg a share of gross exposure proportional to its sum of |s|, and
    softmax(tau * |s|) inside each leg.

    The cap ``w_max`` applies to |w_i| and the weights are *not* renormalised
    afterwards: whatever the cap removes stays in cash.

    Returns:
        (w, top_idx, s, cash, long_pct), as in the original.
    """
    mu = np.asarray(mu, dtype=float).ravel()
    sigma = np.asarray(sigma, dtype=float).ravel()
    n = mu.size
    k = int(np.clip(k, 1, n))
    s = mu / (sigma + eps)

    s_rank = s.copy()
    bad = ~np.isfinite(s_rank)
    if bad.any():
        s_rank[bad] = -np.inf

    if not short_allowed:
        pos_mask = np.isfinite(s) & (s > 0)
        k_eff = min(k, int(pos_mask.sum()))
        if k_eff == 0:
            return np.zeros(n), [], s, 1.0, 0.0
        order = np.argsort(-s_rank)
        order = order[np.isin(order, np.where(pos_mask)[0])]
        top_idx = order[:k_eff]
        z = tau * s[top_idx]
        z -= np.max(z)
        w_sel = np.exp(z) / np.sum(np.exp(z))
        if w_max is not None:
            w_sel = np.minimum(w_sel, float(w_max))
        w = np.zeros(n)
        w[top_idx] = w_sel
        cash = max(0.0, 1.0 - np.sum(np.abs(w)))
        return w, top_idx, s, cash, (1.0 if w.any() else 0.0)

    order = np.argsort(-np.abs(s_rank))
    order = order[np.isin(order, np.where(np.isfinite(s))[0])]
    top_idx = order[:k]
    if top_idx.size == 0:
        return np.zeros(n), [], s, 1.0, 0.0

    pos_idx = [i for i in top_idx if s[i] > 0]
    neg_idx = [i for i in top_idx if s[i] < 0]
    if not pos_idx and not neg_idx:
        return np.zeros(n), top_idx.tolist(), s, 1.0, 0.0

    sum_pos = float(np.sum(np.abs(s[pos_idx]))) if pos_idx else 0.0
    sum_neg = float(np.sum(np.abs(s[neg_idx]))) if neg_idx else 0.0
    total = sum_pos + sum_neg
    if total <= 0.0:
        return np.zeros(n), top_idx.tolist(), s, 1.0, 0.0
    long_pct = sum_pos / total

    def side_softmax(idxs):
        if not idxs:
            return np.array([])
        zz = tau * np.abs(s[idxs])
        zz -= np.max(zz)
        ee = np.exp(zz)
        return ee / np.sum(ee)

    w = np.zeros(n)
    if pos_idx:
        w[pos_idx] = long_pct * side_softmax(pos_idx)
    if neg_idx:
        w[neg_idx] = -(1.0 - long_pct) * side_softmax(neg_idx)
    if w_max is not None:
        w = np.clip(w, -float(w_max), float(w_max))
    cash = max(0.0, 1.0 - float(np.sum(np.abs(w))))
    return w, top_idx.tolist(), s, cash, float(long_pct)


def portfolio_volatility(w: np.ndarray, cov: np.ndarray) -> float:
    """Daily portfolio volatility sqrt(w' Sigma w)."""
    w = np.asarray(w, dtype=float).ravel()
    return float(np.sqrt(max(w @ cov @ w, 0.0)))


def target_annual_volatility(w, vol_daily, target_annual_vol=0.15, days_per_year=DAYS_PER_YEAR):
    """Scale w to an annual volatility target. Returns (w_scaled, leverage)."""
    w = np.asarray(w, dtype=float).ravel()
    leverage_now = max(0.0, float(np.sum(np.abs(w))) - 1.0)
    if target_annual_vol is None or vol_daily <= 1e-12:
        return w, leverage_now
    vol_annual_now = float(vol_daily) * np.sqrt(days_per_year)
    if vol_annual_now <= 0.0:
        return w, leverage_now
    w_scaled = w * float(target_annual_vol / vol_annual_now)
    return w_scaled, max(0.0, float(np.sum(np.abs(w_scaled))) - 1.0)


@dataclass(frozen=True)
class Allocation:
    """One day's decision."""

    weights: np.ndarray
    vol_daily: float  # forecast daily volatility of the final portfolio (unrounded)
    signal: np.ndarray
    long_pct: float
    leverage: float


def sharpe_like_model(
    mu_pred: np.ndarray,
    sigma_pred: np.ndarray,
    returns_window: np.ndarray,
    k: int,
    short_allowed: bool = False,
    target_annual_vol: float | None = 0.15,
    leverage_allowed: bool = False,
    tau: float = 6.0,
    w_max: float | None = 0.3,
    corr_lambda: float = 0.97,
    corr_shrinkage: float = 0.3,
) -> Allocation:
    """Full meta-model for one day.

    ``returns_window`` is the (L, N) block of past returns used for the
    correlation estimate; it must end the day *before* the decision day.
    The thesis defaults (tau=6, w_max=0.3, lambda=0.97, alpha=0.3) were
    hard-coded inside the original function; here they are parameters with
    the same defaults.
    """
    w, _, s, _, long_pct = weights_topk_softmax(
        mu_pred, sigma_pred, k, tau=tau, w_max=w_max, short_allowed=short_allowed
    )
    cov, _ = build_joint_risk_model(sigma_pred, returns_window, lam=corr_lambda, alpha=corr_shrinkage)
    vol_daily = portfolio_volatility(w, cov)
    vol_annual = vol_daily * np.sqrt(DAYS_PER_YEAR)
    leverage = max(0.0, float(np.sum(np.abs(w))) - 1.0)

    if target_annual_vol is not None:
        # Without leverage the weights are only ever scaled *down*.
        need_scale = leverage_allowed or vol_annual > target_annual_vol
        if need_scale:
            w, leverage = target_annual_volatility(w, vol_daily, target_annual_vol)
            vol_daily = portfolio_volatility(w, cov)

    return Allocation(weights=w, vol_daily=vol_daily, signal=s, long_pct=long_pct, leverage=leverage)
