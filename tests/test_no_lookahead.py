"""No decision may depend on data from its own day or later.

The method is the most direct one there is: change the data from some day d
onwards and check that nothing decided on or before d changes. If any step
peeked at the future (a scaler fitted on the whole sample, a centred moving
average, a window that includes the target row, a winsorisation quantile from
the test year...) one of these tests fails.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lstm_portfolio.backtest import compute_weights
from lstm_portfolio.config import ExperimentConfig, FeatureConfig, PortfolioConfig
from lstm_portfolio.data import log_returns
from lstm_portfolio.features import ema_lagged, ewma_sigma, make_windows
from lstm_portfolio.folds import make_folds, prepare_fold


def _perturb_from(r: np.ndarray, row: int, seed: int = 99) -> np.ndarray:
    out = r.copy()
    out[row:] = np.random.default_rng(seed).normal(0, 0.05, out[row:].shape)
    return out


def test_windows_never_contain_the_target_row():
    v = np.arange(100, dtype=float)[:, None] * np.ones((1, 3))
    x = make_windows(v, np.array([10, 50, 99]), lookback=5)
    assert x[:, -1, 0].tolist() == [9, 49, 98]


def test_ema_and_ewma_use_only_the_past():
    r = np.random.default_rng(0).normal(0, 0.01, (400, 3))
    for t in (30, 200, 399):
        p = _perturb_from(r, t)
        np.testing.assert_array_equal(ema_lagged(p, 20)[: t + 1], ema_lagged(r, 20)[: t + 1])
        np.testing.assert_array_equal(ewma_sigma(p)[: t + 1], ewma_sigma(r)[: t + 1])


def test_fold_preprocessing_ignores_test_year(prices):
    r = log_returns(prices).to_numpy()
    dates = log_returns(prices).index
    fold = make_folds(dates, [2017], 0.15)[0]
    a = prepare_fold(r, fold, FeatureConfig())
    # Changing everything from the first validation row on cannot change the
    # winsorisation fitted on the fit rows, nor anything before that row.
    b = prepare_fold(_perturb_from(r, fold.val_start), fold, FeatureConfig())
    np.testing.assert_array_equal(a.returns_w[: fold.val_start], b.returns_w[: fold.val_start])
    np.testing.assert_array_equal(a.ema_lag[: fold.val_start + 1], b.ema_lag[: fold.val_start + 1])


def test_portfolio_weights_ignore_the_future(prices):
    r = log_returns(prices).to_numpy()
    rows = np.arange(300, 600)
    dates = pd.RangeIndex(len(rows))
    mu = ema_lagged(r)[rows]
    sg = ewma_sigma(r)[rows]
    cut = 450
    base = compute_weights(mu, sg, r, rows, dates, PortfolioConfig(), short_allowed=True).weights
    rp = _perturb_from(r, cut)
    alt = compute_weights(
        ema_lagged(rp)[rows], ewma_sigma(rp)[rows], rp, rows, dates, PortfolioConfig(), True
    ).weights
    upto = rows <= cut
    np.testing.assert_array_equal(base[upto], alt[upto])
    assert not np.allclose(base[~upto], alt[~upto])  # the test can actually fail


@pytest.mark.tf
@pytest.mark.slow
def test_walkforward_forecasts_ignore_the_future(prices, tmp_path):
    """Full stage 1 with tiny networks: train, perturb mid-test-year, retrain.

    Every forecast up to and including the perturbation day must be identical,
    because the fit and validation data are before the test year and each
    input window ends the day before the forecast day.
    """
    from lstm_portfolio.data import write_snapshot
    from lstm_portfolio.walkforward import run_walkforward

    def cfg_for(path, out):
        cfg = ExperimentConfig()
        cfg.data.snapshot, cfg.data.start, cfg.data.end = str(path), "2013-01-01", "2018-12-31"
        cfg.walkforward.test_years, cfg.walkforward.seeds = [2017, 2018], [0]
        for net, lb in ((cfg.return_model, 10), (cfg.risk_model, 20)):
            net.lookback, net.units, net.epochs = lb, [4, 2], 2
        cfg.output.runs_dir, cfg.output.results_dir = str(out / "runs"), str(out / "res")
        return cfg

    a_dir, b_dir = tmp_path / "a", tmp_path / "b"
    write_snapshot(prices, a_dir / "p.csv")
    pa = run_walkforward(cfg_for(a_dir / "p.csv", a_dir), save_models=False)

    cut = pd.Timestamp("2018-06-15")
    px = prices.copy()
    later = px.index > cut
    px.loc[later] = px.loc[later] * np.exp(np.random.default_rng(7).normal(0, 0.05, px.loc[later].shape))
    write_snapshot(px, b_dir / "p.csv")
    pb = run_walkforward(cfg_for(b_dir / "p.csv", b_dir), save_models=False)

    key = ["seed", "date", "asset"]
    a = pa.set_index(key).sort_index()
    b = pb.set_index(key).sort_index()
    up_to = a.index.get_level_values("date") <= cut
    first_affected = a.index.get_level_values("date") > cut
    # The price on the first day after the cut changes the return of that day,
    # which only enters forecasts from the *following* day on.
    np.testing.assert_allclose(a[up_to].to_numpy(), b[up_to].to_numpy(), rtol=0, atol=1e-6)
    assert not np.allclose(a[first_affected].to_numpy(), b[first_affected].to_numpy())
