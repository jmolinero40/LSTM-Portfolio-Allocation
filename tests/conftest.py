"""Shared fixtures. Everything here is synthetic and needs no download."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def root() -> Path:
    return ROOT


def synthetic_prices(
    years=(2013, 2018), n_assets: int = 4, seed: int = 0, holidays: bool = True
) -> pd.DataFrame:
    """Business-day random-walk prices; a few NaN days mimic market holidays."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(f"{years[0]}-01-01", f"{years[1]}-12-31", name="date")
    r = 0.01 * rng.standard_t(5, size=(len(dates), n_assets)) / np.sqrt(5 / 3)
    px = pd.DataFrame(
        100 * np.exp(np.cumsum(r, axis=0)), index=dates, columns=[f"A{i}" for i in range(n_assets)]
    )
    if holidays:
        holes = rng.choice(len(dates), size=len(dates) // 40, replace=False)
        px.iloc[holes, rng.integers(0, n_assets, size=len(holes))] = np.nan
    return px


@pytest.fixture
def prices() -> pd.DataFrame:
    return synthetic_prices()
