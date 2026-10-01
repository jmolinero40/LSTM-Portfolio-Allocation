"""Equivalence with the original thesis implementation.

The refactored meta-model is compared against the *original file*,
``legacy/thesis_code/Carteras.py``, imported as is. The networks are compared
against the *saved thesis models*. These tests are what justifies saying that
this repository implements the thesis system: the claim is checked on every
commit instead of being asserted in prose.

Where the code deliberately departs from the original (no rounding of the
returned volatility; the risk-network input scaler), the test pins the
difference explicitly.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from lstm_portfolio.portfolio import sharpe_like_model, weights_topk_softmax


def _legacy(root, name):
    spec = importlib.util.spec_from_file_location(
        f"legacy_{name}", root / "legacy" / "thesis_code" / f"{name}.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def carteras():
    from pathlib import Path

    return _legacy(Path(__file__).resolve().parents[1], "Carteras")


def _case(rng, n=11):
    mu = rng.normal(0, 0.002, n)
    sigma = np.abs(rng.normal(0.012, 0.004, n)) + 1e-4
    window = rng.normal(0, 0.012, (60, n))
    return mu, sigma, window


@pytest.mark.parametrize("short", [False, True])
def test_meta_model_matches_carteras_py(carteras, short):
    rng = np.random.default_rng(1)
    for _ in range(300):
        mu, sigma, window = _case(rng)
        k = int(rng.integers(1, 8))
        w0, vol0, *_ = carteras.sharpe_like_model(
            mu, sigma, window, k, short_allowed=short, target_annual_vol=0.15
        )
        a = sharpe_like_model(mu, sigma, window, k, short_allowed=short, target_annual_vol=0.15)
        # The original rounds weights to 4 decimals and volatility to 3.
        np.testing.assert_allclose(a.weights, w0, atol=5e-5 + 1e-12)
        assert abs(a.vol_daily - vol0) <= 5e-4 + 1e-12


def test_topk_softmax_matches_exactly(carteras):
    rng = np.random.default_rng(2)
    for short in (False, True):
        for _ in range(200):
            mu, sigma, _ = _case(rng)
            ref = carteras.weights_topk_softmax(mu, sigma, 5, tau=6.0, w_max=0.3, short_allowed=short)
            got = weights_topk_softmax(mu, sigma, 5, tau=6.0, w_max=0.3, short_allowed=short)
            np.testing.assert_array_equal(got[0], ref[0])


def test_volatility_is_no_longer_rounded(carteras):
    """Deliberate difference: the thesis returned round(vol, 3)."""
    rng = np.random.default_rng(3)
    mu, sigma, window = _case(rng)
    _, vol0, *_ = carteras.sharpe_like_model(mu, sigma, window, 5)
    a = sharpe_like_model(mu, sigma, window, 5)
    assert vol0 == round(vol0, 3)
    assert a.vol_daily != round(a.vol_daily, 3)


def test_ema_baseline_is_the_backtest_one(root):
    """ema_lagged reproduces ``ema_baseline`` of the thesis backtest script."""
    import pandas as pd

    from lstm_portfolio.features import ema_lagged

    rng = np.random.default_rng(4)
    v = rng.normal(0, 0.01, (300, 5))
    ema = pd.DataFrame(v).ewm(span=20, adjust=False, min_periods=20).mean().to_numpy()
    ema = np.nan_to_num(ema, nan=0.0)
    thesis = np.vstack([ema[0:1], ema[:-1]])  # verbatim from 'ejecucion adicional.py'
    np.testing.assert_allclose(ema_lagged(v, 20), thesis)


@pytest.mark.tf
@pytest.mark.parametrize(
    "fname, lookback",
    [("best_lstm_L60_h1_u64-32_bs32_assets11.keras", 60), ("best_lstmVOL_L240_h1_u64-32_bs32.keras", 240)],
)
def test_architecture_matches_saved_thesis_models(root, fname, lookback):
    import keras

    from lstm_portfolio.config import NetConfig
    from lstm_portfolio.models import build_lstm

    saved = keras.models.load_model(
        root / "legacy" / "artifacts" / "models" / fname, compile=False, safe_mode=False
    )
    mine = build_lstm(11, NetConfig(lookback=lookback), "mse", "x")
    assert [type(layer).__name__ for layer in mine.layers] == [type(layer).__name__ for layer in saved.layers]
    assert mine.count_params() == saved.count_params()
    assert mine.input_shape == saved.input_shape
    for a, b in zip(mine.layers, saved.layers, strict=True):
        for key in ("units", "return_sequences", "rate"):
            assert a.get_config().get(key) == b.get_config().get(key)


@pytest.mark.tf
def test_gaussian_nll_matches_thesis_loss():
    import tensorflow as tf

    from lstm_portfolio.models import gaussian_nll

    def thesis_nll(y_true, y_pred_logits, eps=1e-12):  # verbatim from riesgos.py
        var = tf.nn.softplus(y_pred_logits) + eps
        return 0.5 * tf.math.log(var) + 0.5 * tf.square(y_true) / var

    rng = np.random.default_rng(5)
    y = tf.constant(rng.normal(0, 0.01, (64, 11)), tf.float32)
    z = tf.constant(rng.normal(-9, 1, (64, 11)), tf.float32)
    np.testing.assert_allclose(
        float(tf.reduce_mean(gaussian_nll(y, z))), float(tf.reduce_mean(thesis_nll(y, z))), rtol=1e-6
    )
