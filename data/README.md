# Data

## What is here

`snapshot/prices_adjclose.csv` — daily adjusted closing prices (dividends and
splits) from Yahoo Finance for the 11 assets of the study, 2 Jan 2013 to
31 Dec 2025, in local currency. One column per ticker; empty cells are days
on which that market was closed.

| Ticker | Asset | Market | Currency |
|---|---|---|---|
| XLK, XLF, XLY, XLE, XLV, XLI, XLB | SPDR sector ETFs: technology, financials, consumer discretionary, energy, health care, industrials, materials | NYSE Arca | USD |
| SAN.MC | Banco Santander | Madrid | EUR |
| ENEL.MI | Enel | Milan | EUR |
| SIE.DE | Siemens | Xetra | EUR |
| NOVN.SW | Novartis | SIX Swiss | CHF |

Its SHA-256 is recorded in `configs/walkforward.yaml` (`data.sha256`). Every
stage of the pipeline checks it before running and refuses a different file,
and `tests/test_data.py` checks it on every commit.

## Why the data is frozen

The thesis downloaded prices on every run, up to the day of the run. Two runs
a few days apart therefore used different samples, and Yahoo also rescales
and occasionally revises adjusted prices after the fact. A frozen file with a
fixed end date and a recorded hash is what makes "run this command, get this
number" possible.

It is a CSV, not a binary format, because a CSV can be read and diffed, and
its bytes (hence its hash) do not depend on the version of the library that
wrote it. Line endings are normalised before hashing, so a Windows checkout
gives the same digest.

## How it was produced

```bash
pip install -e ".[data]"
python -m lstm_portfolio.download --out data/snapshot/prices_adjclose.csv
```

([`src/lstm_portfolio/download.py`](../src/lstm_portfolio/download.py)). A fresh
download today will usually *not* reproduce the hash exactly, for the reason
above; `--compare data/snapshot/prices_adjclose.csv` reports how large the
differences in daily returns are.

## Processing (in code, not in the file)

1. Keep only days on which **all** 11 markets traded (about 5% of rows are
   dropped), so every return spans the same interval for every asset.
2. Daily log returns `r_t = log(P_t / P_{t−1})`.
3. Model inputs are winsorised at the 0.5% / 99.5% quantiles estimated on each
   fold's training window. Profit and loss always uses the raw returns.

## Terms

Yahoo Finance data are provided for personal, non-commercial use. The
snapshot is included solely so that the academic results in this repository
can be reproduced and checked; it is not licensed under the repository's MIT
licence. If you are a rights holder and want it removed, open an issue.
