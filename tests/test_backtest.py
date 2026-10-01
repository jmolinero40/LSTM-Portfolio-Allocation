"""Profit and loss accounting: drift, turnover, costs and execution lag."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lstm_portfolio.backtest import WeightPath, equal_weight, evaluate_weights


def _panel(r):
    return pd.DataFrame(r, index=pd.bdate_range("2020-01-01", periods=len(r)), columns=["a", "b"])


def test_turnover_is_measured_from_drifted_weights():
    rets = _panel([[0.10, 0.00], [0.00, 0.00], [0.0, 0.0]])
    path = WeightPath(rets.index[:2], np.array([[0.5, 0.5], [0.5, 0.5]]), np.zeros(2))
    pnl = evaluate_weights(path, rets)
    assert pnl.gross.iloc[0] == pytest.approx(0.05)
    # After +10% on 'a' the book is 0.55/1.05 vs 0.5/1.05; back to 50/50:
    drifted = np.array([0.55, 0.5]) / 1.05
    assert pnl.turnover.iloc[0] == pytest.approx(1.0)  # building the position
    assert pnl.turnover.iloc[1] == pytest.approx(np.abs(np.array([0.5, 0.5]) - drifted).sum())


def test_costs_are_charged_on_turnover():
    rets = _panel([[0.01, -0.01], [0.02, 0.0]])
    path = WeightPath(rets.index, np.array([[1.0, 0.0], [0.0, 1.0]]), np.zeros(2))
    pnl = evaluate_weights(path, rets)
    np.testing.assert_allclose(pnl.net(10).to_numpy(), pnl.gross.to_numpy() - pnl.turnover.to_numpy() * 1e-3)


def test_execution_lag_applies_weights_to_the_next_day():
    rets = _panel([[0.01, 0.0], [0.03, 0.0], [0.05, 0.0]])
    path = WeightPath(rets.index[:2], np.array([[1.0, 0.0], [1.0, 0.0]]), np.zeros(2))
    lag0, lag1 = evaluate_weights(path, rets, 0), evaluate_weights(path, rets, 1)
    assert lag0.gross.tolist() == pytest.approx([0.01, 0.03])
    assert lag1.gross.tolist() == pytest.approx([0.03, 0.05])
    assert list(lag1.gross.index) == list(rets.index[1:])


def test_short_position_loses_when_asset_rises():
    rets = _panel([[0.10, 0.0]])
    path = WeightPath(rets.index, np.array([[-0.5, 0.0]]), np.zeros(1))
    assert evaluate_weights(path, rets).gross.iloc[0] == pytest.approx(-0.05)


def test_equal_weight_has_only_drift_turnover():
    rng = np.random.default_rng(0)
    rets = _panel(rng.normal(0, 0.01, (50, 2)))
    pnl = evaluate_weights(equal_weight(rets.index, 2), rets)
    assert pnl.turnover.iloc[0] == pytest.approx(1.0)
    assert pnl.turnover.iloc[1:].max() < 0.05
