"""Stage 0: build (or rebuild) the frozen price snapshot.

    python -m lstm_portfolio.download --out data/snapshot/prices_adjclose.csv

You do not need this to reproduce the results: the snapshot is in the
repository. It exists so that the snapshot itself is produced by code in the
repository and can be audited, and so that the study can be extended to new
dates or tickers. A fresh download will almost never match the committed hash
exactly, because Yahoo revises adjusted prices; compare with
``--compare data/snapshot/prices_adjclose.csv`` to see by how much.
"""

from __future__ import annotations

import argparse

import numpy as np

from lstm_portfolio.data import TICKERS, download_prices, load_prices, log_returns, write_snapshot


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--start", default="2013-01-01")
    ap.add_argument("--end", default="2025-12-31", help="inclusive")
    ap.add_argument("--compare", help="existing snapshot to compare returns against")
    args = ap.parse_args(argv)

    px = download_prices(TICKERS, args.start, args.end)
    digest = write_snapshot(px, args.out)
    print(f"{len(px)} rows, {px.index[0].date()} to {px.index[-1].date()}")
    print(f"sha256: {digest}")
    if args.compare:
        a = log_returns(load_prices(args.compare))
        b = log_returns(load_prices(args.out))
        a, b = a.align(b, join="inner")
        print(
            f"max |difference| in daily log returns vs {args.compare}: {np.nanmax(np.abs(a - b).to_numpy()):.2e}"
        )


if __name__ == "__main__":
    main()
