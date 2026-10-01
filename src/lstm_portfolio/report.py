"""Stage 2: forecasts -> portfolios -> every table and figure.

    python -m lstm_portfolio.report --config configs/walkforward.yaml

Reads the committed forecasts (``results/walkforward/predictions.parquet``) and
the price snapshot, and needs no TensorFlow. Its output is deterministic: the
test suite re-runs it and checks that the committed metrics and the tables in
the README are exactly what the code produces.

Strategies (each one long-only and long-short, same meta-model, same k, tau,
w_max and volatility target; only the source of mu and sigma changes):

=================  ==================  =====================================
name               mu from             sigma from
=================  ==================  =====================================
lstm_lstm          return LSTM         risk LSTM          (the thesis system)
lstm_ewma          return LSTM         EWMA(0.97)
ema_lstm           EMA_{t-1}           risk LSTM
ema_ewma           EMA_{t-1}           EWMA(0.97)         (no neural network)
equal_weight       --                  --                 1/N, long-only
=================  ==================  =====================================

The ablations answer the question a reader will ask first: how much of the
result is due to the networks, and how much to the portfolio rules around them?
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from lstm_portfolio.backtest import PnL, compute_weights, equal_weight, evaluate_weights
from lstm_portfolio.config import ExperimentConfig, load_config
from lstm_portfolio.data import load_prices, log_returns
from lstm_portfolio.features import ewma_sigma
from lstm_portfolio.folds import make_folds, prepare_fold
from lstm_portfolio.metrics import (
    block_bootstrap_sharpe_diff,
    diebold_mariano,
    directional_accuracy,
    information_coefficient,
    performance,
    qlike,
    var_backtest,
)
from lstm_portfolio.walkforward import load_predictions

__all__ = ["run_report", "inject_readme", "main"]

STRATEGY_LABELS = {
    "lstm_lstm": "Thesis system: LSTM μ + LSTM σ",
    "lstm_lstm_lag1": "↳ same, executed one day later",
    "lstm_ewma": "LSTM μ + EWMA σ",
    "ema_lstm": "EMA μ + LSTM σ",
    "ema_ewma": "EMA μ + EWMA σ (no neural network)",
    "equal_weight": "1/N, rebalanced daily",
}
SIDES = {"long_only": False, "long_short": True}


# --------------------------------------------------------------------------
# Building the strategy panel
# --------------------------------------------------------------------------


def _strategy_paths(cfg: ExperimentConfig, rets: pd.DataFrame, preds: dict) -> dict:
    """Weight paths for every (strategy, side, seed). Seed is None if not used."""
    r = rets.to_numpy()
    assets = list(rets.columns)
    folds = make_folds(rets.index, cfg.walkforward.test_years, cfg.walkforward.val_fraction)
    seeds = sorted(preds)
    paths: dict[tuple, list] = {}
    sigma_inputs: dict[tuple, list] = {}

    for fold in folds:
        data = prepare_fold(r, fold, cfg.features)
        te = fold.test_rows
        dates = rets.index[te]
        mu_ema = data.ema_lag[te]
        sig_ewma = ewma_sigma(data.returns_w, cfg.portfolio.ewma_sigma_lambda)[te]
        sources = {}
        for s in seeds:
            mu_l = preds[s]["mu"].reindex(index=dates, columns=assets)
            sg_l = preds[s]["sigma"].reindex(index=dates, columns=assets)
            if mu_l.isna().any().any() or sg_l.isna().any().any():
                raise ValueError(f"Forecasts missing for test year {fold.test_year}, seed {s}.")
            sources[("lstm_lstm", s)] = (mu_l.to_numpy(), sg_l.to_numpy())
            sources[("lstm_ewma", s)] = (mu_l.to_numpy(), sig_ewma)
            sources[("ema_lstm", s)] = (mu_ema, sg_l.to_numpy())
            sigma_inputs.setdefault(("lstm", s), []).append(sg_l.to_numpy())
        sources[("ema_ewma", None)] = (mu_ema, sig_ewma)
        sigma_inputs.setdefault(("ewma", None), []).append(sig_ewma)

        for (name, s), (mu, sg) in sources.items():
            for side, short in SIDES.items():
                wp = compute_weights(mu, sg, data.returns_w, te, dates, cfg.portfolio, short)
                paths.setdefault((name, side, s), []).append(wp)

    merged = {}
    for key, parts in paths.items():
        merged[key] = type(parts[0])(
            pd.DatetimeIndex(np.concatenate([p.dates for p in parts])),
            np.vstack([p.weights for p in parts]),
            np.concatenate([p.vol_daily for p in parts]),
        )
    oos_dates = merged[("ema_ewma", "long_only", None)].dates
    merged[("equal_weight", "long_only", None)] = equal_weight(oos_dates, len(assets))
    return merged


def build_panel(cfg: ExperimentConfig) -> tuple[pd.DataFrame, dict]:
    prices = load_prices(cfg.data.snapshot, cfg.data.start, cfg.data.end, cfg.data.sha256)
    rets = log_returns(prices)
    simple = np.expm1(rets)  # RAW returns: the money is counted on these
    preds = load_predictions(Path(cfg.output.results_dir) / "predictions.parquet")
    paths = _strategy_paths(cfg, rets, preds)

    rows = []
    for (name, side, seed), wp in paths.items():
        lags = (0, 1) if name == "lstm_lstm" else (0,)
        for lag in lags:
            pnl = evaluate_weights(wp, simple, exec_lag=lag)
            strat = name if lag == 0 else f"{name}_lag{lag}"
            vol = pd.Series(wp.vol_daily[: len(pnl.gross)], index=pnl.gross.index)
            rows.append(
                pd.DataFrame(
                    {
                        "strategy": strat, "side": side, "seed": -1 if seed is None else seed,
                        "date": pnl.gross.index, "gross": pnl.gross.to_numpy(),
                        "turnover": pnl.turnover.to_numpy(), "exposure": pnl.exposure.to_numpy(),
                        "vol_forecast": vol.to_numpy(),
                    }
                )
            )  # fmt: skip
    panel = pd.concat(rows, ignore_index=True)
    return panel, {"rets": rets, "preds": preds}


# --------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------


def _metrics(panel: pd.DataFrame, cost_bps: list[float]) -> pd.DataFrame:
    out = []
    for (strat, side, seed), g in panel.groupby(["strategy", "side", "seed"], sort=False):
        pnl = PnL(
            g.set_index("date")["gross"], g.set_index("date")["turnover"], g.set_index("date")["exposure"]
        )
        for c in cost_bps:
            m = performance(pnl.net(c), pnl.turnover)
            out.append({"strategy": strat, "side": side, "seed": seed, "cost_bps": c, **m})
    return pd.DataFrame(out)


def _seed_avg(panel: pd.DataFrame, strat: str, side: str, cost_bps: float = 0.0) -> pd.Series:
    g = panel[(panel.strategy == strat) & (panel.side == side)]
    net = g["gross"] - g["turnover"] * cost_bps / 1e4
    return net.groupby(g["date"]).mean()


def _fmt_pct(x: float, nd: int = 1) -> str:
    return f"{100 * x:.{nd}f}%"


def _md_table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines) + "\n"


def main_table(metrics: pd.DataFrame) -> str:
    header = [
        "Strategy",
        "Sharpe",
        "Sharpe @5 bps",
        "Sharpe @10 bps",
        "CAGR",
        "Vol",
        "Max DD",
        "Turnover/day",
    ]
    rows = []
    for side, side_label in (("long_only", "**Long-only**"), ("long_short", "**Long-short**")):
        rows.append([side_label] + [""] * (len(header) - 1))
        for strat in STRATEGY_LABELS:
            m = metrics[(metrics.strategy == strat) & (metrics.side == side)]
            if m.empty:
                continue
            row = [STRATEGY_LABELS[strat]]
            for c in (0.0, 5.0, 10.0):
                s = m[m.cost_bps == c]["sharpe"]
                if len(s) > 1:
                    row.append(f"{s.mean():.2f} ({s.min():.2f} to {s.max():.2f})")
                else:
                    row.append(f"{s.iloc[0]:.2f}")
            g = m[m.cost_bps == 0.0]
            row += [
                _fmt_pct(g.cagr.mean()),
                _fmt_pct(g.vol.mean()),
                _fmt_pct(g.max_dd.mean()),
                f"{g.turnover.mean():.2f}",
            ]
            rows.append(row)
    return _md_table(header, rows)


#: Tables shown in README.md (the others are linked from it).
README_BLOCKS = ("headline", "main", "significance", "forecasts")

#: Sharpe ratios as printed in the thesis (tables 1 and 3), for the headline.
THESIS_REPORTED_SHARPE = {"long_only": 1.03, "long_short": 1.17}


def headline_table(metrics: pd.DataFrame) -> str:
    rows = []
    for side, label in (("long_only", "Long-only"), ("long_short", "Long-short")):
        m = metrics[(metrics.strategy == "lstm_lstm") & (metrics.side == side)]
        g, c5 = m[m.cost_bps == 0.0].sharpe, m[m.cost_bps == 5.0].sharpe
        rows.append([label, f"{THESIS_REPORTED_SHARPE[side]:.2f}", f"{g.mean():.2f}", f"{c5.mean():.2f}"])
    ew = metrics[(metrics.strategy == "equal_weight") & (metrics.cost_bps == 0.0)].sharpe.iloc[0]
    ew5 = metrics[(metrics.strategy == "equal_weight") & (metrics.cost_bps == 5.0)].sharpe.iloc[0]
    rows.append(["1/N on the same 11 assets", "", f"{ew:.2f}", f"{ew5:.2f}"])
    hdr = [
        "Sharpe ratio",
        "Thesis (mostly in-sample, no costs)",
        "Walk-forward, before costs",
        "Walk-forward, 5 bps costs",
    ]
    return _md_table(hdr, rows)


def significance_table(panel: pd.DataFrame, cfg: ExperimentConfig) -> tuple[str, pd.DataFrame]:
    ev = cfg.evaluation
    comps = [
        ("long_only", "lstm_lstm", "equal_weight", 0.0),
        ("long_only", "lstm_lstm", "equal_weight", 5.0),
        ("long_only", "lstm_lstm", "ema_ewma", 0.0),
        ("long_short", "lstm_lstm", "ema_ewma", 0.0),
        ("long_only", "ema_ewma", "equal_weight", 0.0),
    ]
    rows, recs = [], []
    for side, a, b, c in comps:
        sa = _seed_avg(panel, a, side, c)
        side_b = "long_only" if b == "equal_weight" else side
        sb = _seed_avg(panel, b, side_b, c).reindex(sa.index)
        res = block_bootstrap_sharpe_diff(
            sa.to_numpy(), sb.to_numpy(), ev.bootstrap_block, ev.bootstrap_samples, ev.bootstrap_seed
        )
        recs.append({"side": side, "a": a, "b": b, "cost_bps": c, **res})
        rows.append(
            [
                f"{STRATEGY_LABELS[a]} − {STRATEGY_LABELS[b]} ({side.replace('_', '-')}, {c:g} bps)",
                f"{res['diff']:+.2f}",
                f"[{res['ci_low']:+.2f}, {res['ci_high']:+.2f}]",
                f"{res['p_value']:.2f}",
            ]
        )
    md = _md_table(["Sharpe difference", "Estimate", "95% CI", "p-value"], rows)
    return md, pd.DataFrame(recs)


def forecast_table(cfg: ExperimentConfig, rets: pd.DataFrame, preds: dict) -> tuple[str, pd.DataFrame]:
    r = rets.to_numpy()
    folds = make_folds(rets.index, cfg.walkforward.test_years, cfg.walkforward.val_fraction)
    te = np.concatenate([f.test_rows for f in folds])
    dates = rets.index[te]
    realised = r[te]
    ema, ewma = [], []
    for f in folds:
        d = prepare_fold(r, f, cfg.features)
        ema.append(d.ema_lag[f.test_rows])
        ewma.append(ewma_sigma(d.returns_w, cfg.portfolio.ewma_sigma_lambda)[f.test_rows])
    ema, ewma = np.vstack(ema), np.vstack(ewma)
    assets = list(rets.columns)

    recs = []
    q_ewma = qlike(ewma, realised)
    for s, p in preds.items():
        mu = p["mu"].reindex(index=dates, columns=assets).to_numpy()
        sg = p["sigma"].reindex(index=dates, columns=assets).to_numpy()
        ic = information_coefficient(mu, realised)
        q = qlike(sg, realised)
        dm = diebold_mariano(q, q_ewma)
        recs.append(
            {
                "model": "lstm", "seed": s, "dir_acc": directional_accuracy(mu, realised),
                **ic, "qlike": float(q.mean()), "dm_vs_ewma": dm["dm_stat"], "dm_p": dm["p_value"],
                "calib_r2_over_v": float(np.mean(realised**2 / sg**2)),
            }
        )  # fmt: skip
    ic_e = information_coefficient(ema, realised)
    recs.append(
        {
            "model": "baseline", "seed": -1, "dir_acc": directional_accuracy(ema, realised), **ic_e,
            "qlike": float(q_ewma.mean()), "dm_vs_ewma": np.nan, "dm_p": np.nan,
            "calib_r2_over_v": float(np.mean(realised**2 / ewma**2)),
        }
    )  # fmt: skip
    df = pd.DataFrame(recs)
    lstm, base = df[df.model == "lstm"], df[df.model == "baseline"].iloc[0]
    always_up = directional_accuracy(np.ones_like(realised), realised)

    def rng(col, fmt):
        v = lstm[col]
        return f"{fmt(v.mean())} ({fmt(v.min())} to {fmt(v.max())})" if len(v) > 1 else fmt(v.iloc[0])

    pct = lambda x: f"{100 * x:.1f}%"  # noqa: E731
    f2 = lambda x: f"{x:.2f}"  # noqa: E731
    f3 = lambda x: f"{x:.3f}"  # noqa: E731
    rows = [
        ["Directional accuracy, all (day, asset) pairs", rng("dir_acc", pct), f"{pct(base.dir_acc)} (EMA)", pct(always_up) + " (always up)"],
        ["Mean daily cross-sectional IC (Spearman)", rng("ic_mean", f3), f"{base.ic_mean:.3f} (EMA)", "0"],
        ["t-statistic of the mean IC", rng("ic_t", f2), f"{base.ic_t:.2f} (EMA)", ""],
        ["QLIKE (lower is better)", rng("qlike", f3), f"{base.qlike:.3f} (EWMA)", ""],
        ["Diebold–Mariano vs EWMA (negative = LSTM better)", rng("dm_vs_ewma", f2), "", ""],
        ["Calibration E[r²/σ̂²] (1 = calibrated)", rng("calib_r2_over_v", f2), f"{base.calib_r2_over_v:.2f} (EWMA)", "1"],
    ]  # fmt: skip
    period = f"{min(cfg.walkforward.test_years)}–{max(cfg.walkforward.test_years)}"
    md = _md_table(
        [f"Out of sample, {period}", "LSTM (mean over seeds, range)", "Baseline", "Reference"], rows
    )
    return md, df


def var_table(panel: pd.DataFrame, alpha: float) -> tuple[str, pd.DataFrame]:
    recs, rows = [], []
    for side in SIDES:
        g = panel[(panel.strategy == "lstm_lstm") & (panel.side == side)]
        for seed, h in g.groupby("seed"):
            res = var_backtest(h["gross"].to_numpy(), h["vol_forecast"].to_numpy(), alpha)
            recs.append({"side": side, "seed": seed, **res})
            rows.append(
                [
                    side.replace("_", "-"), str(seed), f"{res['exceptions']} / {res['T']}",
                    _fmt_pct(res["rate"], 2), f"{res['kupiec_p']:.3g}", f"{res['christoffersen_p']:.3g}",
                ]
            )  # fmt: skip
    hdr = [
        "Side",
        "Seed",
        "Exceptions",
        f"Rate (expected {_fmt_pct(alpha, 0)})",
        "Kupiec p",
        "Christoffersen p",
    ]
    return _md_table(hdr, rows), pd.DataFrame(recs)


def yearly_table(panel: pd.DataFrame) -> str:
    cols = [
        ("lstm_lstm", "long_only", "Thesis LO"),
        ("lstm_lstm", "long_short", "Thesis LS"),
        ("ema_ewma", "long_only", "No-NN LO"),
        ("equal_weight", "long_only", "1/N"),
    ]
    series = {lab: _seed_avg(panel, s, side) for s, side, lab in cols}
    df = pd.DataFrame(series)
    yearly = df.groupby(df.index.year).apply(lambda x: (1 + x).prod() - 1)
    rows = [[str(y)] + [_fmt_pct(v) for v in yearly.loc[y]] for y in yearly.index]
    return _md_table(["Test year"] + [c[2] for c in cols], rows)


# --------------------------------------------------------------------------
# Figure
# --------------------------------------------------------------------------


def equity_figure(panel: pd.DataFrame, path: Path, period: str) -> None:
    """Growth of 1 for the thesis system, the no-network ablation and 1/N.

    Colours are the first four categorical slots of a palette validated for
    colour-vision deficiency; line style and a direct label at the right edge
    carry identity as well, so it never rests on colour alone.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    surface, ink, muted = "#fcfcfb", "#0b0b0b", "#52514e"
    spec = [
        ("lstm_lstm", "long_only", "Thesis system, long-only", "#2a78d6", "-"),
        ("lstm_lstm", "long_short", "Thesis system, long-short", "#eb6834", "-"),
        ("equal_weight", "long_only", "1/N", "#1baf7a", "--"),
        ("ema_ewma", "long_only", "No neural network, long-only", "#eda100", ":"),
    ]
    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=150, facecolor=surface)
    ax.set_facecolor(surface)
    for strat, side, label, color, ls in spec:
        g = panel[(panel.strategy == strat) & (panel.side == side)]
        for seed, h in g.groupby("seed"):
            if seed >= 0:
                eq = (1 + h.set_index("date")["gross"]).cumprod()
                ax.plot(eq.index, eq.values, color=color, lw=0.7, alpha=0.35)
        eq = (1 + _seed_avg(panel, strat, side)).cumprod()
        ax.plot(eq.index, eq.values, color=color, lw=2.0, ls=ls, label=label)
        ax.annotate(
            f"{eq.iloc[-1]:.2f}", (eq.index[-1], eq.iloc[-1]), xytext=(4, 0),
            textcoords="offset points", va="center", fontsize=8, color=ink,
        )  # fmt: skip
    ax.set_yscale("log")
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

    ax.yaxis.set_major_locator(FixedLocator([0.75, 1, 1.5, 2, 3, 4]))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_ylabel("Growth of 1, before costs (log scale)", color=muted)
    ax.set_title(
        f"Walk-forward, out of sample {period}. Thick: mean over seeds; thin: each seed",
        fontsize=10, color=ink, loc="left",
    )  # fmt: skip
    ax.grid(True, which="major", color="#d9d8d3", lw=0.6)
    ax.tick_params(colors=muted, labelsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#bdbcb6")
    ax.legend(frameon=False, fontsize=8, loc="upper left", labelcolor=ink)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=surface)
    plt.close(fig)


# --------------------------------------------------------------------------
# README injection
# --------------------------------------------------------------------------


def inject_readme(readme: Path, blocks: dict[str, str]) -> None:
    """Replace the text between ``<!-- results:NAME:start -->`` and ``:end -->``."""
    text = readme.read_text(encoding="utf-8")
    for name, body in blocks.items():
        pat = re.compile(rf"(<!-- results:{name}:start -->\n).*?(<!-- results:{name}:end -->)", re.S)
        if not pat.search(text):
            raise ValueError(f"README has no block '{name}'.")
        text = pat.sub(lambda m, b=body: m.group(1) + b + m.group(2), text)
    readme.write_text(text, encoding="utf-8", newline="\n")


def run_report(cfg: ExperimentConfig, figures: bool = True) -> dict[str, str]:
    out = Path(cfg.output.results_dir)
    panel, ctx = build_panel(cfg)
    metrics = _metrics(panel, cfg.evaluation.cost_bps)

    tables = {"headline": headline_table(metrics), "main": main_table(metrics)}
    tables["significance"], sig = significance_table(panel, cfg)
    tables["forecasts"], fc = forecast_table(cfg, ctx["rets"], ctx["preds"])
    tables["var"], var = var_table(panel, cfg.evaluation.var_alpha)
    tables["yearly"] = yearly_table(panel)

    (out / "tables").mkdir(parents=True, exist_ok=True)
    panel.to_parquet(out / "daily_returns.parquet", index=False)
    metrics.to_csv(out / "metrics.csv", index=False, float_format="%.10g", lineterminator="\n")
    sig.to_csv(out / "significance.csv", index=False, float_format="%.10g", lineterminator="\n")
    fc.to_csv(out / "forecast_quality.csv", index=False, float_format="%.10g", lineterminator="\n")
    var.to_csv(out / "var_backtest.csv", index=False, float_format="%.10g", lineterminator="\n")
    for name, md in tables.items():
        (out / "tables" / f"{name}.md").write_text(md, encoding="utf-8", newline="\n")
    if figures:
        equity_figure(
            panel,
            Path("docs/images/equity_oos.png"),
            f"{min(cfg.walkforward.test_years)}–{max(cfg.walkforward.test_years)}",
        )
    return tables


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="configs/walkforward.yaml")
    ap.add_argument("--update-readme", action="store_true", help="rewrite the result blocks in README.md")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    tables = run_report(cfg, figures=not args.no_figures)
    for name, md in tables.items():
        print(f"\n## {name}\n\n{md}")
    if args.update_readme:
        inject_readme(Path("README.md"), {k: tables[k] for k in README_BLOCKS})
        print("README.md updated.")


if __name__ == "__main__":
    main()
