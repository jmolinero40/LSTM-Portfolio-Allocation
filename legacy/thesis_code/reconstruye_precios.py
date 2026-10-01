# -*- coding: utf-8 -*-
"""
Utilities to transform LSTM return predictions (log-returns) into price-level predictions,
store a history, and plot predicted vs. realized prices.

Added:
- Optional bias calibration for μ (global or rolling)
- Optional periodic re-anchoring of predicted prices to real closes
"""

from dataclasses import dataclass
from typing import Optional, Union, Iterable, Literal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


CalibMode = Optional[Literal["none", "global", "rolling"]]


@dataclass
class PriceReconstructionConfig:
    span_ema: int = 20          # EMA span used in training for the baseline
    lag_baseline: int = 1       # lag to avoid look-ahead (use t-1 baseline to predict t)
    residual: bool = True       # True if mu_pred are residuals over EMA baseline
    output_dir: Union[str, Path] = "/mnt/data"  # where to save history files
    xlsx_name: str = "predicted_prices.xlsx"
    csv_name: str = "predicted_prices.csv"
    # New:
    calibrate_bias: CalibMode = "none"  # "none" | "global" | "rolling"
    rolling_window: int = 120           # window for rolling de-mean (days)
    reanchor_freq: Optional[str] = None # e.g., "W" (weekly), "M" (month end). None disables.


def _ema_baseline(log_returns: pd.DataFrame, span: int) -> pd.DataFrame:
    return log_returns.ewm(span=span, adjust=False).mean()


def reconstruct_mu_log(
    mu_pred: pd.DataFrame,
    realized_log_returns: Optional[pd.DataFrame] = None,
    config: PriceReconstructionConfig = PriceReconstructionConfig()
) -> pd.DataFrame:
    """Reconstruct the *log-return* mean μ_t from model predictions."""
    mu_pred = mu_pred.copy()
    if config.residual:
        if realized_log_returns is None:
            raise ValueError("realized_log_returns is required when residual=True.")
        base = _ema_baseline(realized_log_returns, span=config.span_ema)
        base = base.shift(config.lag_baseline)  # avoid look-ahead
        mu_log = mu_pred.add(base.reindex(mu_pred.index), fill_value=0.0)
    else:
        mu_log = mu_pred
    return mu_log


def calibrate_mu(
    mu_log: pd.DataFrame,
    realized_log_returns: Optional[pd.DataFrame],
    mode: CalibMode,
    rolling_window: int
) -> pd.DataFrame:
    """Debias μ to reduce cumulative drift."""
    if mode in (None, "none"):
        return mu_log

    if mode == "global":
        if realized_log_returns is None:
            raise ValueError("global calibration requires realized_log_returns.")
        real = realized_log_returns.reindex(mu_log.index)
        bias = (mu_log - real).mean()  # per-asset mean bias
        return mu_log.sub(bias, axis=1)

    if mode == "rolling":
        return mu_log - mu_log.rolling(rolling_window, min_periods=max(10, rolling_window//5)).mean()

    raise ValueError(f"Unknown calibrate_bias mode: {mode}")


def mu_log_to_price(
    mu_log: pd.DataFrame,
    close_prices: pd.DataFrame
) -> pd.DataFrame:
    """Transform μ (log-returns) into predicted price levels."""
    mu_log = mu_log.copy()
    mu_log = mu_log.loc[~mu_log.index.duplicated(keep="last")]

    # Ensure columns alignment / order
    common_cols = [c for c in mu_log.columns if c in close_prices.columns]
    mu_log = mu_log[common_cols]
    cp = close_prices[common_cols]

    # cumulative sum of log-returns => log price change
    cum_mu = mu_log.cumsum()
    growth = np.exp(cum_mu)

    # Anchor strictly at t0-1 (exclude t0)
    first_date = mu_log.index.min()
    prev_mask = cp.index < first_date
    if not prev_mask.any():
        raise ValueError("close_prices must include at least one day before first mu date.")
    anchor_series = cp.loc[prev_mask].iloc[-1]

    pred_prices = growth.multiply(anchor_series, axis=1)
    return pred_prices


def reanchor_prices(
    pred_prices: pd.DataFrame,
    close_prices: pd.DataFrame,
    freq: Optional[str]
) -> pd.DataFrame:
    """Periodically re-anchor predicted prices to last real close before each period."""
    if not freq:
        return pred_prices

    cols = [c for c in pred_prices.columns if c in close_prices.columns]
    pp = pred_prices[cols].copy()
    cp = close_prices[cols]

    periods = pp.resample(freq).first().index
    if len(periods) <= 1:
        return pp

    result = []
    for i, start in enumerate(periods):
        end = periods[i+1] if i+1 < len(periods) else pp.index.max() + pd.Timedelta(days=1)
        seg = pp.loc[(pp.index >= start) & (pp.index < end)]
        if seg.empty:
            continue
        prev_mask = cp.index < seg.index.min()
        if not prev_mask.any():
            result.append(seg)
            continue
        anchor = cp.loc[prev_mask].iloc[-1]
        factor = (anchor / seg.iloc[0])
        seg_rebased = seg.multiply(factor, axis=1)
        result.append(seg_rebased)

    return pd.concat(result).sort_index()


def build_price_predictions_history(
    mu_pred: pd.DataFrame,
    realized_log_returns: Optional[pd.DataFrame],
    close_prices: pd.DataFrame,
    config: PriceReconstructionConfig = PriceReconstructionConfig()
) -> pd.DataFrame:
    """High-level pipeline with optional calibration and re-anchoring."""
    mu_log = reconstruct_mu_log(mu_pred, realized_log_returns, config=config)
    mu_log = calibrate_mu(mu_log, realized_log_returns, config.calibrate_bias, config.rolling_window)
    pred_prices = mu_log_to_price(mu_log, close_prices)
    pred_prices = reanchor_prices(pred_prices, close_prices, config.reanchor_freq)

    outdir = Path(config.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    xlsx_path = outdir / config.xlsx_name
    csv_path = outdir / config.csv_name

    pred_prices.to_excel(xlsx_path)
    pred_prices.to_csv(csv_path)

    return pred_prices


def plot_pred_vs_real(
    pred_prices: pd.DataFrame,
    close_prices: pd.DataFrame,
    tickers: Optional[Iterable[str]] = None,
    title: Optional[str] = None
) -> None:
    """Plot predicted vs realized price for one or multiple tickers on a single axis."""
    common_cols = [c for c in pred_prices.columns if c in close_prices.columns]
    if tickers is None:
        tickers = common_cols[:1]
    else:
        tickers = [t for t in tickers if t in common_cols]

    fig = plt.figure()
    ax = fig.gca()

    for t in tickers:
        pred = pred_prices[t].dropna()
        real = close_prices[t].reindex(pred.index).ffill()
        ax.plot(pred.index, pred.values, label=f"Pred {t}")
        ax.plot(real.index, real.values, linestyle="--", label=f"Real {t}")

    ax.set_xlabel("Fecha")
    ax.set_ylabel("Precio")
    ax.set_title(title or "Precio predicho vs. precio real")
    ax.legend()
    plt.show()


def plot_grid_pred_vs_real(
    pred_prices: pd.DataFrame,
    close_prices: pd.DataFrame,
    tickers: Iterable[str],
    ncols: int = 3,
    figsize: tuple = (12, 12),
    suptitle: str = "Series temporales (predicho vs real)",
    mercado=1,
    short_allowed='NO',
    mercado_conocido='SI',
    out_dir: Optional[Union[str, Path]] = None  # <— NUEVO
) -> str:
    """
    Plot a grid (e.g., 3x3) of predicted vs realized prices for multiple tickers.
    - Uses matplotlib only, default colors, one axis per subplot.
    - Real series is dashed.
    Returns the path to the saved PNG image.
    """
    import math
    import matplotlib.pyplot as plt

    tickers = [t for t in tickers if (t in pred_prices.columns and t in close_prices.columns)]
    if not tickers:
        raise ValueError("No hay tickers comunes entre pred_prices y close_prices.")

    n = len(tickers)
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, sharex=False, sharey=False)

    # Normalizar el arreglo axes para poder indexarlo plano
    if nrows == 1 and ncols == 1:
        axes = [[axes]]
    elif nrows == 1:
        axes = [axes]
    elif ncols == 1:
        axes = [[ax] for ax in axes]

    axes_flat = [ax for row in axes for ax in row]

    for i, t in enumerate(tickers):
        ax = axes_flat[i]
        pred = pred_prices[t].dropna()
        real = close_prices[t].reindex(pred.index).ffill()

        ax.plot(pred.index, pred.values, label=f"Pred {t}")
        ax.plot(real.index, real.values, linestyle="--", label=f"Real {t}")

        ax.set_title(t)
        ax.set_xlabel("Fecha")
        ax.set_ylabel("Precio")

        if i == 0:
            ax.legend()

    # Ocultar ejes vacíos si sobran
    for j in range(n, nrows * ncols):
        axes_flat[j].axis("off")

    fig.suptitle(suptitle)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    if out_dir is None:
        out_path = f"C:/Users/jmoli/Desktop/TFG ECONOMIA/Códigos/3.1 Recogida de datos y procesamiento/gráficos/pred_vs_real_grid_{mercado}_short{short_allowed}_mercadoConocido{mercado_conocido}.png"
    else:
        out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        out_path = str(out_dir / f"pred_vs_real_grid_{mercado}_{short_allowed}.png")

    fig.savefig(out_path, dpi=144)
    plt.close(fig)

    return out_path