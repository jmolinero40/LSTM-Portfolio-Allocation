"""The audit of the thesis backtest, re-derived from the thesis's own files.

Each assertion below corresponds to a claim in docs/AUDIT.md.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def audit():
    spec = importlib.util.spec_from_file_location("audit", ROOT / "scripts" / "audit" / "reproduce_audit.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_most_of_the_thesis_backtest_was_in_sample(audit):
    runs = audit.saved_backtests()
    ls = runs["known_long_short"]
    assert ls["whole"]["sharpe"] == pytest.approx(1.157, abs=1e-3)  # the reported figure's run
    assert ls["share_of_days_before_test"] > 0.83
    assert ls["test_period"]["sharpe"] < ls["train_period"]["sharpe"]


def test_rerunning_the_same_configuration_gave_a_different_sharpe(audit):
    runs = audit.saved_backtests()
    assert runs["known_long_short"]["params"] == runs["known_long_short_jan2026"]["params"]
    assert runs["known_long_short_jan2026"]["whole"]["sharpe"] == pytest.approx(0.759, abs=1e-3)


def test_christoffersen_and_rounding_and_leverage(audit):
    c = audit.christoffersen_as_in_thesis_vs_textbook()
    assert c["exceptions"] == 173
    assert c["lr_ind_thesis_formula"] == pytest.approx(41.16, abs=0.01)
    assert c["lr_ind_textbook"] < 25
    assert audit.volatility_rounding()["distinct_values"] <= 10
    assert audit.leverage_rate()["implied_annual_rate"] > 0.5


def test_ema_leak_creates_accuracy_from_noise(audit):
    n = audit.ema_leak_on_noise()
    assert n["linear_model_plus_ema_t"] > 0.75
    assert abs(n["linear_model_plus_ema_t_minus_1"] - 0.5) < 0.02


@pytest.mark.tf
@pytest.mark.slow
def test_networks(audit):
    ret, risk, xs, ys = audit._models()
    hr = audit.return_network_hit_ratio(ret, xs, ys)["rentabilidades_dim_11"]
    assert hr["hit_with_ema_t_as_trained"] == pytest.approx(0.6985, abs=0.003)
    assert hr["hit_with_ema_t_minus_1"] < hr["hit_always_up"]

    rk = audit.risk_network_scaling(risk)
    assert rk["qlike_test_scaled_inputs"] == pytest.approx(-7.83, abs=0.006)
    assert rk["sigma_time_cv_raw"] < 0.01 < 0.2 < rk["sigma_time_cv_scaled"]

    rep = audit.replicate_jan2026_backtest(ret, risk, xs, ys)
    assert rep["correlation_with_saved"] > 0.9999
