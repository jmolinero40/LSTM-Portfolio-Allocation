"""Reproduce every number in docs/AUDIT.md from the original thesis artefacts.

    python scripts/audit/reproduce_audit.py

Inputs are the files the thesis actually produced, kept unmodified under
``legacy/artifacts/``: the two trained networks, the input/target scalers of
the return network, the two return files the thesis scripts read, and the
spreadsheets its backtest wrote. Nothing is retrained.

Output: ``results/audit/audit.json`` (all numbers) and ``results/audit/audit.md``.
``tests/test_audit.py`` re-runs this and checks the headline numbers, so the
audit is verified on every commit, not just asserted in prose.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
LEG = ROOT / "legacy" / "artifacts"
OUT = ROOT / "results" / "audit"

# The thesis model was split 70/15/15 over its windows. These are the
# boundaries printed by the thesis scripts and quoted in the thesis (sec. 3.2).
THESIS_TRAIN_END = "2022-01-19"
THESIS_VAL_END = "2023-11-30"


def _load_legacy_module(name: str):
    """Import a file from legacy/thesis_code without executing anything else."""
    path = ROOT / "legacy" / "thesis_code" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"legacy_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _perf(r: pd.Series) -> dict:
    r = pd.Series(r, dtype=float)
    eq = (1 + r).cumprod()
    return {
        "days": int(len(r)),
        "sharpe": float(r.mean() / r.std() * np.sqrt(252)),
        "cagr": float(eq.iloc[-1] ** (252 / len(r)) - 1),
        "max_dd": float((eq / eq.cummax() - 1).min()),
    }


# --------------------------------------------------------------------------
# 1. The saved backtests, split by what the model had seen
# --------------------------------------------------------------------------


def saved_backtests() -> dict:
    """Split each saved daily return series at the thesis train/validation ends."""
    out = {}
    for run in sorted(p.name for p in (LEG / "backtests").iterdir() if p.is_dir()):
        book = LEG / "backtests" / run / "cuadernillo_metricas.xlsx"
        r = pd.read_excel(book, "retornos", index_col=0).iloc[:, 0]
        params = pd.read_excel(book, "params").iloc[0].to_dict()
        out[run] = {
            "params": {k: (v.item() if hasattr(v, "item") else v) for k, v in params.items()},
            "first_day": str(r.index[0].date()),
            "last_day": str(r.index[-1].date()),
            "whole": _perf(r),
            "train_period": _perf(r.loc[:THESIS_TRAIN_END]),
            "validation_period": _perf(
                r.loc[pd.Timestamp(THESIS_TRAIN_END) + pd.Timedelta(days=1) : THESIS_VAL_END]
            ),
            "test_period": _perf(r.loc[pd.Timestamp(THESIS_VAL_END) + pd.Timedelta(days=1) :]),
            "share_of_days_before_test": float((r.index <= pd.Timestamp(THESIS_VAL_END)).mean()),
        }
    return out


def volatility_rounding() -> dict:
    """The saved portfolio volatility takes only a handful of distinct values."""
    s = pd.read_excel(
        LEG / "backtests" / "known_long_only" / "cuadernillo_metricas.xlsx", "sigma_log", index_col=0
    )
    v = s.iloc[:, 0].to_numpy()
    vals = np.unique(np.round(v, 6))
    return {
        "days": int(len(v)),
        "distinct_values": int(len(vals)),
        "annualised_levels_pct": [round(float(x) * np.sqrt(252) * 100, 1) for x in vals],
    }


def christoffersen_as_in_thesis_vs_textbook() -> dict:
    """Recompute the independence test on the thesis's own exception series."""
    from scipy.stats import chi2, norm

    book = LEG / "backtests" / "known_long_only" / "cuadernillo_metricas.xlsx"
    r = pd.read_excel(book, "retornos", index_col=0).iloc[:, 0].to_numpy()
    sig = pd.read_excel(book, "sigma_log", index_col=0).iloc[:, 0].to_numpy()
    var = 1.0 - np.exp(norm.ppf(0.01) * sig)
    hits = (-r > var).astype(int)
    i0, i1 = hits[:-1], hits[1:]
    n00 = int(((i0 == 0) & (i1 == 0)).sum())
    n01 = int(((i0 == 0) & (i1 == 1)).sum())
    n10 = int(((i0 == 1) & (i1 == 0)).sum())
    n11 = int(((i0 == 1) & (i1 == 1)).sum())
    n = n00 + n01 + n10 + n11
    pi, pi01, pi11 = (n01 + n11) / n, n01 / (n00 + n01), n11 / (n10 + n11)
    lg = lambda x: np.log(np.clip(x, 1e-12, 1 - 1e-12))
    l1 = n00 * lg(1 - pi01) + n01 * lg(pi01) + n10 * lg(1 - pi11) + n11 * lg(pi11)
    l0_thesis = (n00 + n01) * lg(1 - pi) + n01 * lg(pi) + (n10 + n11) * lg(1 - pi) + n11 * lg(pi)
    l0_textbook = (n00 + n10) * lg(1 - pi) + (n01 + n11) * lg(pi)
    lr_t, lr_c = -2 * (l0_thesis - l1), -2 * (l0_textbook - l1)
    return {
        "exceptions": int(hits.sum()),
        "T": int(len(hits)),
        "lr_ind_thesis_formula": float(lr_t),
        "p_thesis_formula": float(chi2.sf(lr_t, 1)),
        "lr_ind_textbook": float(lr_c),
        "p_textbook": float(chi2.sf(lr_c, 1)),
    }


def leverage_rate() -> dict:
    """The leveraged run charged rf=rl=0.02/12 per *day*."""
    daily = 0.02 / 12
    return {"rate_per_day": daily, "implied_annual_rate": float((1 + daily) ** 252 - 1)}


# --------------------------------------------------------------------------
# 2. The networks (needs TensorFlow)
# --------------------------------------------------------------------------


def _ema(values: np.ndarray, span: int = 20) -> np.ndarray:
    e = pd.DataFrame(values).ewm(span=span, adjust=False, min_periods=span).mean()
    return np.nan_to_num(e.to_numpy(), nan=0.0)


def _windows(v: np.ndarray, lookback: int) -> tuple[np.ndarray, np.ndarray]:
    idx = np.arange(lookback, len(v))
    return np.stack([v[i - lookback : i] for i in idx]), idx


def _models():
    import keras
    import tensorflow as tf

    def nll_gauss(y_true, y_pred_logits, eps=1e-12):
        var = tf.nn.softplus(y_pred_logits) + eps
        return 0.5 * tf.math.log(var) + 0.5 * tf.square(y_true) / var

    def mae_abs_r_vs_sigma(y_true, y_pred_logits, eps=1e-12):
        return tf.reduce_mean(tf.abs(tf.abs(y_true) - tf.sqrt(tf.nn.softplus(y_pred_logits) + eps)))

    co = {"nll_gauss": nll_gauss, "mae_abs_r_vs_sigma": mae_abs_r_vs_sigma}
    m = LEG / "models"
    ret = keras.models.load_model(m / "best_lstm_L60_h1_u64-32_bs32_assets11.keras", compile=False)
    risk = keras.models.load_model(
        m / "best_lstmVOL_L240_h1_u64-32_bs32.keras", compile=False, custom_objects=co
    )
    import joblib

    return ret, risk, joblib.load(m / "X_scaler_L60.pkl"), joblib.load(m / "y_scaler_L60.pkl")


def _softplus(x):
    return np.logaddexp(0.0, x)


def return_network_hit_ratio(ret, xs, ys) -> dict:
    """The 70% directional accuracy and where it comes from.

    ``rentabilidades_dim_11.parquet`` is the file the current return model was
    trained on (20 Jan 2026); its 70/15/15 split puts the test set at the end.
    """
    out = {}
    for name in ("rentabilidades_dim_11", "rentabilidades"):
        df = pd.read_parquet(LEG / "data" / f"{name}.parquet")
        v = df.to_numpy().astype(np.float32)
        ema = _ema(v)
        ema_lag = np.vstack([ema[:1], ema[:-1]])
        x, idx = _windows(v, 60)
        n = len(x)
        te = slice(int(0.85 * n), None)
        xs_ = xs.transform(x.reshape(-1, v.shape[1])).reshape(x.shape)
        res = ys.inverse_transform(ret.predict(xs_, verbose=0, batch_size=512))
        y = v[idx]

        def hr(p, te=te, y=y):
            return float(np.mean(np.sign(p[te]) == np.sign(y[te])))

        out[name] = {
            "test_first_day": str(df.index[idx[int(0.85 * n)]].date()),
            "test_last_day": str(df.index[idx[-1]].date()),
            "hit_with_ema_t_as_trained": hr(res + ema[idx]),
            "hit_with_ema_t_minus_1": hr(res + ema_lag[idx]),
            "hit_ema_t_minus_1_alone": hr(ema_lag[idx]),
            "hit_always_up": hr(np.ones_like(y)),
        }
    return out


def risk_network_scaling(risk) -> dict:
    """Thesis QLIKE (scaled inputs) vs what the backtest fed the network (raw)."""
    from sklearn.preprocessing import RobustScaler

    df = pd.read_parquet(LEG / "data" / "rentabilidades.parquet")  # the file it was trained on
    v = df.to_numpy().astype(np.float32)
    x, idx = _windows(v, 240)
    y = v[idx]
    n = len(x)
    i_tr, te = int(0.7 * n), slice(int(0.85 * n), None)
    sc = RobustScaler(with_centering=False).fit(x[:i_tr].reshape(-1, v.shape[1]))
    xs = sc.transform(x.reshape(-1, v.shape[1])).reshape(x.shape)
    s_scaled = np.sqrt(_softplus(risk.predict(xs, verbose=0, batch_size=512)))
    s_raw = np.sqrt(_softplus(risk.predict(x, verbose=0, batch_size=512)))

    def q(s):
        return float(np.mean(np.log(s[te] ** 2) + y[te] ** 2 / s[te] ** 2))

    def cv(s):  # coefficient of variation over time, averaged over assets
        return float(np.mean(s.std(0) / s.mean(0)))

    def corr_abs(s):
        return float(np.mean([np.corrcoef(s[:, j], np.abs(y[:, j]))[0, 1] for j in range(v.shape[1])]))

    return {
        "qlike_test_scaled_inputs": q(s_scaled),
        "qlike_test_raw_inputs": q(s_raw),
        "sigma_time_cv_scaled": cv(s_scaled),
        "sigma_time_cv_raw": cv(s_raw),
        "corr_sigma_abs_return_scaled": corr_abs(s_scaled),
        "corr_sigma_abs_return_raw": corr_abs(s_raw),
    }


def replicate_jan2026_backtest(ret, risk, xs, ys) -> dict:
    """Re-run the thesis backtest loop with the thesis code and compare.

    The loop is ``ejecucion adicional.py`` vectorised (same windows, same
    lagged EMA, same raw risk inputs) and calls the *original* ``Carteras.py``.
    ``parquet_files[2]`` in that script resolves to ``rentabilidades.parquet``
    (alphabetical order), so that is the file used.
    """
    carteras = _load_legacy_module("Carteras")
    df = pd.read_parquet(LEG / "data" / "rentabilidades.parquet")
    v = df.to_numpy().astype(np.float32)
    ema = _ema(v)
    base = np.vstack([ema[:1], ema[:-1]])
    tgt = np.arange(240, len(v))
    xr = np.stack([v[i - 60 : i] for i in tgt])
    xk = np.stack([v[i - 240 : i] for i in tgt])
    xr_s = xs.transform(xr.reshape(-1, v.shape[1])).reshape(xr.shape)
    mu = ys.inverse_transform(ret.predict(xr_s, verbose=0, batch_size=512)) + base[tgt]
    sig = np.sqrt(_softplus(risk.predict(xk, verbose=0, batch_size=512)))
    rets, sig_out = [], []
    for j, t in enumerate(tgt):
        w, vol, *_ = carteras.sharpe_like_model(
            mu[j], sig[j], xr[j], 5, short_allowed=True, target_annual_vol=0.15
        )
        rets.append(float(np.dot(w, np.exp(v[t]) - 1.0)))
        sig_out.append(sig[j])
    mine = pd.Series(rets, index=df.index[tgt])
    saved = pd.read_excel(
        LEG / "backtests" / "known_long_short_jan2026" / "cuadernillo_metricas.xlsx", "retornos", index_col=0
    ).iloc[:, 0]
    a, b = mine.align(saved, join="inner")
    s = np.asarray(sig_out)
    return {
        "days_compared": int(len(a)),
        "correlation_with_saved": float(np.corrcoef(a, b)[0, 1]),
        "max_abs_difference": float(np.max(np.abs(a - b))),
        "sharpe_reproduced": _perf(mine)["sharpe"],
        "sharpe_saved": _perf(saved)["sharpe"],
        "sigma_time_cv_in_backtest": float(np.mean(s.std(0) / s.mean(0))),
    }


def ema_leak_on_noise(seed: int = 0) -> dict:
    """A model with zero skill, evaluated the thesis way, on pure noise."""
    rng = np.random.default_rng(seed)
    t_len, n, lb, a = 3128, 11, 60, 2 / 21
    r = 0.01 * rng.standard_t(4, size=(t_len, n)) / np.sqrt(2)
    ema = pd.DataFrame(r).ewm(span=20, adjust=False).mean().to_numpy()
    ema_lag = np.vstack([ema[:1], ema[:-1]])
    idx = np.arange(lb, t_len)
    x = np.stack([r[i - lb : i].T for i in idx])  # (windows, assets, lookback)
    y_target = r[idx] - ema[idx]
    m = len(idx)
    itr, ite = int(0.7 * m), int(0.85 * m)
    design = np.c_[x[:itr].reshape(-1, lb), np.ones(itr * n)]
    beta = np.linalg.lstsq(design, y_target[:itr].reshape(-1), rcond=None)[0]
    pred = (np.c_[x[ite:].reshape(-1, lb), np.ones((m - ite) * n)] @ beta).reshape(-1, n)
    y = r[idx][ite:]

    def hr(p):
        return float(np.mean(np.sign(p) == np.sign(y)))

    return {
        "alpha": a,
        "linear_model_plus_ema_t": hr(pred + ema[idx][ite:]),
        "zero_forecast_plus_ema_t": hr(ema[idx][ite:]),
        "linear_model_plus_ema_t_minus_1": hr(pred + ema_lag[idx][ite:]),
    }


# --------------------------------------------------------------------------


def run(with_networks: bool = True) -> dict:
    res = {
        "saved_backtests": saved_backtests(),
        "volatility_rounding": volatility_rounding(),
        "christoffersen": christoffersen_as_in_thesis_vs_textbook(),
        "leverage_rate": leverage_rate(),
        "ema_leak_on_noise": ema_leak_on_noise(),
    }
    if with_networks:
        ret, risk, xs, ys = _models()
        res["return_network"] = return_network_hit_ratio(ret, xs, ys)
        res["risk_network"] = risk_network_scaling(risk)
        res["replication_jan2026"] = replicate_jan2026_backtest(ret, risk, xs, ys)
    return res


def to_markdown(res: dict) -> str:
    p = lambda x: f"{100 * x:.1f}%"
    lines = ["# Audit numbers (generated by scripts/audit/reproduce_audit.py)\n"]
    lines.append("## Saved thesis backtests, split at the thesis train / validation ends\n")
    lines.append(
        "| Run | Whole period | Train period (≤ 2022-01-19) | Validation | Test (≥ 2023-12-01) | Days before test |"
    )
    lines.append("|---|---|---|---|---|---|")
    for run, d in res["saved_backtests"].items():
        f = lambda k, d=d: f"Sharpe {d[k]['sharpe']:.2f}, CAGR {p(d[k]['cagr'])}"
        lines.append(
            f"| {run} | {f('whole')} | {f('train_period')} | {f('validation_period')} | {f('test_period')} | {p(d['share_of_days_before_test'])} |"
        )
    if "return_network" in res:
        lines.append("\n## Return network: directional accuracy on its test split\n")
        lines.append(
            "| Data file | Test window | As trained (+EMA_t) | +EMA_{t-1} | EMA_{t-1} alone | Always up |"
        )
        lines.append("|---|---|---|---|---|---|")
        for k, d in res["return_network"].items():
            lines.append(
                f"| {k} | {d['test_first_day']} to {d['test_last_day']} | {p(d['hit_with_ema_t_as_trained'])} | "
                f"{p(d['hit_with_ema_t_minus_1'])} | {p(d['hit_ema_t_minus_1_alone'])} | {p(d['hit_always_up'])} |"
            )
        r = res["risk_network"]
        lines.append("\n## Risk network: scaled inputs (training) vs raw inputs (thesis backtest)\n")
        lines.append("| | Scaled | Raw |\n|---|---|---|")
        lines.append(
            f"| QLIKE on test | {r['qlike_test_scaled_inputs']:.2f} | {r['qlike_test_raw_inputs']:.2f} |"
        )
        lines.append(
            f"| Time variation of σ̂ (CV) | {r['sigma_time_cv_scaled']:.3f} | {r['sigma_time_cv_raw']:.3f} |"
        )
        lines.append(
            f"| corr(σ̂, abs r) | {r['corr_sigma_abs_return_scaled']:.2f} | {r['corr_sigma_abs_return_raw']:.2f} |"
        )
        j = res["replication_jan2026"]
        lines.append("\n## Re-running the thesis backtest with the thesis code\n")
        lines.append(
            f"Correlation with the saved January 2026 run: {j['correlation_with_saved']:.4f} over "
            f"{j['days_compared']} days (max abs difference {j['max_abs_difference']:.1e}); "
            f"Sharpe {j['sharpe_reproduced']:.3f} reproduced vs {j['sharpe_saved']:.3f} saved."
        )
    n = res["ema_leak_on_noise"]
    lines.append("\n## The EMA_t leak on pure noise (no predictability by construction)\n")
    lines.append(f"- linear model + EMA_t: {p(n['linear_model_plus_ema_t'])}")
    lines.append(f"- zero forecast + EMA_t: {p(n['zero_forecast_plus_ema_t'])}")
    lines.append(f"- linear model + EMA_(t-1): {p(n['linear_model_plus_ema_t_minus_1'])}")
    c = res["christoffersen"]
    lines.append("\n## Other checks\n")
    lines.append(
        f"- Christoffersen LR_ind on the long-only run: {c['lr_ind_thesis_formula']:.2f} with the thesis formula, "
        f"{c['lr_ind_textbook']:.2f} with the textbook one (p = {c['p_textbook']:.2g})."
    )
    v = res["volatility_rounding"]
    lines.append(
        f"- Saved portfolio volatility: {v['distinct_values']} distinct values over {v['days']} days "
        f"(annualised: {', '.join(str(x) for x in v['annualised_levels_pct'])}%)."
    )
    lr = res["leverage_rate"]
    lines.append(
        f"- Leveraged run with costs: {lr['rate_per_day']:.6f} per day = {p(lr['implied_annual_rate'])} a year."
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    with_networks = "--no-networks" not in sys.argv
    res = run(with_networks)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "audit.json").write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    md = to_markdown(res)
    (OUT / "audit.md").write_text(md, encoding="utf-8", newline="\n")
    print(md)


if __name__ == "__main__":
    main()
