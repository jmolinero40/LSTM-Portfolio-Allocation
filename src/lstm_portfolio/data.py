"""Data: the frozen price snapshot and the returns derived from it.

Why a snapshot. The thesis downloaded prices on every run with ``end=None``
("up to today"). Two runs a week apart therefore trained and tested on
different samples, and Yahoo also revises adjusted closes retroactively. Here
the data is downloaded once, written as plain CSV and identified by its SHA-256;
every later step reads that file and refuses to run if the hash does not match.

Why CSV and not parquet. A CSV is readable, diffable in a pull request, and its
bytes do not depend on the version of the library that wrote it, so the hash is
stable across machines. Line endings are normalised before hashing so a Windows
checkout with ``core.autocrlf`` produces the same digest.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = [
    "TICKERS",
    "download_prices",
    "write_snapshot",
    "file_sha256",
    "load_prices",
    "log_returns",
]

#: Seven US sector ETFs and four European blue chips (thesis section 3.1).
TICKERS: tuple[str, ...] = (
    "XLK", "XLF", "XLY", "XLE", "XLV", "XLI", "XLB",
    "SAN.MC", "ENEL.MI", "SIE.DE", "NOVN.SW",
)  # fmt: skip


def download_prices(
    tickers: tuple[str, ...] | list[str] = TICKERS,
    start: str = "2013-01-01",
    end: str = "2025-12-31",
) -> pd.DataFrame:
    """Adjusted closes from Yahoo Finance, one column per ticker, sorted.

    ``end`` is inclusive here (yfinance's own ``end`` is exclusive, so one day
    is added). Days on which only some markets traded are kept with NaN; the
    alignment to common trading days happens in :func:`log_returns`.
    """
    import yfinance as yf  # optional dependency, only needed to (re)build the snapshot

    end_excl = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    raw = yf.download(
        list(tickers), start=start, end=end_excl, auto_adjust=False,
        actions=False, progress=False, threads=False,
    )  # fmt: skip
    px = raw["Adj Close"].reindex(columns=sorted(tickers))
    px.index = pd.to_datetime(px.index).tz_localize(None).normalize()
    px.index.name = "date"
    px = px.dropna(how="all")
    empty = [c for c in px.columns if px[c].notna().sum() == 0]
    if empty:
        raise RuntimeError(f"Yahoo returned no data for {empty}; retry in a minute.")
    return px


def write_snapshot(prices: pd.DataFrame, path: str | Path) -> str:
    """Write the canonical CSV and return its SHA-256."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(path, float_format="%.6f", lineterminator="\n", date_format="%Y-%m-%d")
    return file_sha256(path)


def file_sha256(path: str | Path) -> str:
    """SHA-256 of a text file with line endings normalised to LF."""
    data = Path(path).read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def load_prices(
    path: str | Path,
    start: str | None = None,
    end: str | None = None,
    expected_sha256: str | None = None,
) -> pd.DataFrame:
    """Read the snapshot, check its hash, and restrict it to [start, end]."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. See data/README.md for how to obtain it.")
    if expected_sha256 is not None:
        got = file_sha256(path)
        if got != expected_sha256:
            raise ValueError(
                f"Snapshot hash mismatch for {path}:\n  expected {expected_sha256}\n  got      {got}\n"
                "The numbers in this repository were produced from the expected file. "
                "A fresh download from Yahoo will usually differ slightly."
            )
    px = pd.read_csv(path, index_col="date", parse_dates=["date"]).sort_index()
    if start is not None:
        px = px.loc[pd.Timestamp(start) :]
    if end is not None:
        px = px.loc[: pd.Timestamp(end)]
    return px


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily log returns on the days on which *every* market traded.

    As in the thesis, rows with any missing price are dropped first, so the
    return on a given row spans from the previous common trading day. These
    returns are NOT winsorised: winsorisation is a feature transformation fitted
    inside each training window (see :mod:`lstm_portfolio.features`), and the
    profit and loss is always computed on these raw returns.
    """
    px = prices.dropna(how="any")
    if (px <= 0).any().any():
        raise ValueError("Non-positive prices in snapshot.")
    r = np.log(px / px.shift(1)).iloc[1:]
    return r.astype(np.float64)
