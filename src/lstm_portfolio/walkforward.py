"""Stage 1: train both networks fold by fold and store out-of-sample forecasts.

    python -m lstm_portfolio.walkforward --config configs/walkforward.yaml

For every test year and every seed this trains the return network and the
risk network from scratch on the data before that year, and writes their
forecasts for every day of that year. Nothing else: turning forecasts into
portfolios is stage 2 (:mod:`lstm_portfolio.report`), which needs no
TensorFlow and runs in seconds.

The split matters for reproducibility. Training is the expensive and slightly
machine-dependent part; the forecasts it produces are small and are committed
to the repository, so every table and figure can be regenerated exactly from
them by anyone.

Each (fold, seed) result is written to its own file as soon as it is done, so
an interrupted run resumes where it stopped.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from lstm_portfolio.config import ExperimentConfig, load_config
from lstm_portfolio.data import load_prices, log_returns
from lstm_portfolio.folds import make_folds, prepare_fold

__all__ = ["run_walkforward", "load_predictions", "main"]


def _part_path(runs_dir: Path, test_year: int, seed: int) -> Path:
    return runs_dir / "parts" / f"y{test_year}_s{seed}.parquet"


def run_walkforward(cfg: ExperimentConfig, save_models: bool = True, verbose: int = 0) -> pd.DataFrame:
    from lstm_portfolio.models import ReturnForecaster, RiskForecaster, set_global_seed

    prices = load_prices(cfg.data.snapshot, cfg.data.start, cfg.data.end, cfg.data.sha256)
    rets = log_returns(prices)
    r = rets.to_numpy()
    assets = list(rets.columns)
    folds = make_folds(rets.index, cfg.walkforward.test_years, cfg.walkforward.val_fraction)

    runs_dir = Path(cfg.output.runs_dir)
    (runs_dir / "parts").mkdir(parents=True, exist_ok=True)
    cfg.save(runs_dir / "config.yaml")
    log_path = runs_dir / "training_log.jsonl"

    for fold in folds:
        data = prepare_fold(r, fold, cfg.features)
        for seed in cfg.walkforward.seeds:
            part = _part_path(runs_dir, fold.test_year, seed)
            if part.exists():
                print(f"[skip] {fold.test_year} seed {seed} (done)")
                continue
            t0 = time.time()
            tr, va, te = fold.targets(cfg.return_model.lookback)
            set_global_seed(seed)
            ret = ReturnForecaster(cfg.return_model).fit(data.returns_w, data.ema_lag, tr, va, verbose)
            mu = ret.predict(data.returns_w, data.ema_lag, te)

            tr, va, te = fold.targets(cfg.risk_model.lookback)
            set_global_seed(seed)
            risk = RiskForecaster(cfg.risk_model).fit(data.returns_w, tr, va, verbose)
            sigma = risk.predict(data.returns_w, te)

            idx = pd.MultiIndex.from_product([rets.index[te], assets], names=["date", "asset"])
            out = pd.DataFrame({"mu": mu.ravel(), "sigma": sigma.ravel()}, index=idx).reset_index()
            out.insert(0, "seed", seed)
            out.insert(1, "test_year", fold.test_year)
            out.to_parquet(part, index=False)

            if save_models:
                mdir = runs_dir / "models" / f"y{fold.test_year}_s{seed}"
                mdir.mkdir(parents=True, exist_ok=True)
                ret.model.save(mdir / "return_lstm.keras")
                risk.model.save(mdir / "risk_lstm.keras")
            with open(log_path, "a", encoding="utf-8") as f:
                for net, log in (("return", ret.log), ("risk", risk.log)):
                    rec = {"test_year": fold.test_year, "seed": seed, "net": net, **log}
                    f.write(json.dumps(rec) + "\n")
            print(
                f"[done] {fold.test_year} seed {seed}: "
                f"return {ret.log['epochs_run']} ep, risk {risk.log['epochs_run']} ep, "
                f"{time.time() - t0:.0f}s",
                flush=True,
            )

    return collect(cfg)


def collect(cfg: ExperimentConfig) -> pd.DataFrame:
    """Merge the per-fold parts into the tracked predictions file."""
    runs_dir = Path(cfg.output.runs_dir)
    parts = []
    for y in cfg.walkforward.test_years:
        for s in cfg.walkforward.seeds:
            p = _part_path(runs_dir, y, s)
            if not p.exists():
                raise FileNotFoundError(f"Missing {p}; the walk-forward run is incomplete.")
            parts.append(pd.read_parquet(p))
    pred = pd.concat(parts, ignore_index=True).sort_values(["seed", "date", "asset"])
    out = Path(cfg.output.results_dir)
    out.mkdir(parents=True, exist_ok=True)
    pred.to_parquet(out / "predictions.parquet", index=False)
    if (runs_dir / "training_log.jsonl").exists():
        log = pd.read_json(runs_dir / "training_log.jsonl", lines=True)
        log = log.drop_duplicates(["test_year", "seed", "net"], keep="last")
        log.sort_values(["test_year", "seed", "net"]).to_csv(out / "training_log.csv", index=False)
    cfg.save(out / "config_used.yaml")
    return pred


def load_predictions(path: str | Path) -> dict[int, dict[str, pd.DataFrame]]:
    """{seed: {"mu": (dates x assets), "sigma": (dates x assets)}}."""
    pred = pd.read_parquet(path)
    out = {}
    for seed, g in pred.groupby("seed"):
        out[int(seed)] = {
            q: g.pivot(index="date", columns="asset", values=q).sort_index() for q in ("mu", "sigma")
        }
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="configs/walkforward.yaml")
    ap.add_argument("--no-save-models", action="store_true")
    ap.add_argument("--verbose", type=int, default=0)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    pred = run_walkforward(cfg, save_models=not args.no_save_models, verbose=args.verbose)
    n_days = pred["date"].nunique()
    print(f"\nWrote {len(pred)} rows ({n_days} test days) to {cfg.output.results_dir}/predictions.parquet")


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    main()
