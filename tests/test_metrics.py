"""Statistics against hand computations and textbook values."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from lstm_portfolio.metrics import (
    block_bootstrap_sharpe_diff,
    diebold_mariano,
    directional_accuracy,
    max_drawdown,
    performance,
    var_backtest,
)


def test_drawdown_and_cagr_by_hand():
    r = np.array([0.10, -0.50, 0.20])
    assert max_drawdown(r) == pytest.approx(-0.5)
    perf = performance(r)
    assert perf["cagr"] == pytest.approx((1.1 * 0.5 * 1.2) ** (252 / 3) - 1)


def test_christoffersen_is_the_textbook_statistic():
    hits = np.array([0, 0, 1, 1, 0, 0, 0, 1, 0, 0] * 30)
    # Build returns and vols that produce exactly these hits at alpha = 5%.
    vol = np.full(len(hits), 0.01)
    var = 1 - np.exp(stats.norm.ppf(0.05) * 0.01)
    r = np.where(hits == 1, -2 * var, 0.0)
    res = var_backtest(r, vol, alpha=0.05)
    i0, i1 = hits[:-1], hits[1:]
    n = {(a, b): int(((i0 == a) & (i1 == b)).sum()) for a in (0, 1) for b in (0, 1)}
    pi = (n[0, 1] + n[1, 1]) / sum(n.values())
    p0, p1 = n[0, 1] / (n[0, 0] + n[0, 1]), n[1, 1] / (n[1, 0] + n[1, 1])
    l0 = (n[0, 0] + n[1, 0]) * np.log(1 - pi) + (n[0, 1] + n[1, 1]) * np.log(pi)
    l1 = n[0, 0] * np.log(1 - p0) + n[0, 1] * np.log(p0) + n[1, 0] * np.log(1 - p1) + n[1, 1] * np.log(p1)
    assert res["exceptions"] == hits.sum()
    assert res["christoffersen_lr"] == pytest.approx(-2 * (l0 - l1))


def test_kupiec_is_zero_at_exact_coverage():
    r = np.zeros(1000)
    r[::100] = -1.0
    res = var_backtest(r, np.full(1000, 0.01), alpha=0.01)
    assert res["exceptions"] == 10
    assert res["kupiec_lr"] == pytest.approx(0.0, abs=1e-9)


def test_bootstrap_detects_a_real_difference_and_not_a_fake_one():
    rng = np.random.default_rng(0)
    a = rng.normal(0.001, 0.01, 2000)
    same = block_bootstrap_sharpe_diff(a, a + rng.normal(0, 1e-6, 2000), samples=500)
    assert same["ci_low"] < 0 < same["ci_high"]
    better = block_bootstrap_sharpe_diff(a + 0.002, a, samples=500)
    assert better["ci_low"] > 0


def test_dm_sign_and_directional_accuracy():
    rng = np.random.default_rng(1)
    good, bad = rng.normal(0, 1, 500) ** 2, rng.normal(0, 2, 500) ** 2
    assert diebold_mariano(good, bad)["dm_stat"] < 0
    assert directional_accuracy(np.array([1, -1, 1]), np.array([2, -3, -1])) == pytest.approx(2 / 3)
