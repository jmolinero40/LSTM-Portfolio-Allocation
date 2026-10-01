"""Snapshot integrity and the calendar."""

from __future__ import annotations

import numpy as np
import pytest

from lstm_portfolio.config import ExperimentConfig, load_config
from lstm_portfolio.data import file_sha256, load_prices, log_returns, write_snapshot
from lstm_portfolio.folds import make_folds


def test_hash_ignores_windows_line_endings(tmp_path, prices):
    p = tmp_path / "p.csv"
    h = write_snapshot(prices, p)
    crlf = tmp_path / "crlf.csv"
    crlf.write_bytes(p.read_bytes().replace(b"\n", b"\r\n"))
    assert file_sha256(crlf) == h


def test_wrong_hash_is_refused(tmp_path, prices):
    p = tmp_path / "p.csv"
    write_snapshot(prices, p)
    with pytest.raises(ValueError, match="hash mismatch"):
        load_prices(p, expected_sha256="0" * 64)


def test_returns_use_only_days_when_every_market_traded(prices):
    r = log_returns(prices)
    assert not r.isna().any().any()
    common = prices.dropna(how="any").index
    assert r.index.equals(common[1:])


def test_folds_are_ordered_and_disjoint(prices):
    dates = log_returns(prices).index
    folds = make_folds(dates, [2016, 2017, 2018], 0.15)
    for f in folds:
        assert 0 < f.val_start < f.pre_end == f.test_rows[0]
        assert set(dates[f.test_rows].year) == {f.test_year}
        tr, va, te = f.targets(60)
        assert tr.max() < va.min() and va.max() < te.min()
    assert all(a.pre_end < b.pre_end for a, b in zip(folds, folds[1:], strict=False))


def test_shipped_config_loads_and_typos_are_rejected(root):
    cfg = load_config(root / "configs" / "walkforward.yaml")
    assert cfg.risk_model.lookback == 240 and cfg.return_model.lookback == 60
    with pytest.raises(ValueError, match="Unknown keys"):
        ExperimentConfig.from_dict({"portfolio": {"kk": 5}})


def test_committed_snapshot_matches_its_hash(root):
    cfg = load_config(root / "configs" / "walkforward.yaml")
    path = root / cfg.data.snapshot
    if not path.exists():
        pytest.skip("snapshot not present")
    assert file_sha256(path) == cfg.data.sha256
    px = load_prices(path, cfg.data.start, cfg.data.end, cfg.data.sha256)
    assert np.isfinite(log_returns(px).to_numpy()).all()
