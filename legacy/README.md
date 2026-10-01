# The original thesis code and artefacts

Everything in this folder is kept **exactly as it was** when the thesis was
written, so that the audit in [`docs/AUDIT.md`](../docs/AUDIT.md) can be checked
against the real thing rather than a description of it. Nothing here is
imported by the package except `Carteras.py`, which the equivalence tests load
to compare against the refactored meta-model.

The scripts will not run as they are: they contain absolute paths to the
author's Windows desktop, download data "until today" on import, and select
their input by position in a folder listing. That is part of the record.

## `thesis_code/`

The experiments were run by editing these files in place, so a script's
current contents correspond to its last use.

| File | Role | Status |
|---|---|---|
| `Descarga.py` | Download, log returns, full-sample winsorisation | **Final system** |
| `copia_retornos.py` | Trains the return LSTM (EMA baseline, saves scalers) | **Final system**, despite its name: the only script that writes the `_assets11` model and the scaler files the backtest loads |
| `riesgos.py` | Trains the risk LSTM (Gaussian NLL) | **Final system**. Its trailing calibration block fails at import (`res` is undefined outside `__main__`) |
| `Carteras.py` | Meta-model: signal, top-k softmax, covariance, volatility target | **Final system** |
| `ejecucion adicional.py` | Backtest, VaR tests, benchmarks, betas | **Final system**: produced the thesis tables |
| `Ejecución.py` | Earlier version of the backtest | Superseded (VaR exception sign error) |
| `reconstruye_precios.py` | Predicted vs realised price charts (thesis figs. 8, 16) | Figures only |
| `genera_grafico_windsor.py`, `grafico_set.py` | Thesis figures 5 and 7 | Figures only |
| `retornos.py` | Return LSTM before the fix for collapsing predictions | Dead end |
| `tuneo.py` | Keras Tuner search; selected models on *test* MAE | Dead end, unused |
| `LambdaMART.py` | Learning-to-rank alternative | Dead end, not in the thesis |
| `riesgos_corregido.py`, `prueba.py` | LSTM–GARCH hybrid ("attempted without success", thesis sec. 5.2) | Dead end |
| `pruebas_backtest_puntuales.py` | Single-day debugging | Scratch |

## `artifacts/`

| Path | What it is |
|---|---|
| `models/best_lstm_L60_h1_u64-32_bs32_assets11.keras` | Return network. Trained 20 Jan 2026, after the thesis; it overwrote the one behind the thesis tables |
| `models/X_scaler_L60.pkl`, `models/y_scaler_L60.pkl` | Its input and target scalers (same date) |
| `models/best_lstmVOL_L240_h1_u64-32_bs32.keras` | Risk network, trained 14 Nov 2025: the one in the thesis. Its input scaler was never saved |
| `data/rentabilidades.parquet` | Winsorised log returns to 24 Oct 2025 (read by every script via `parquet_files[2]`) |
| `data/rentabilidades_dim_11.parquet` | Same, downloaded 20 Jan 2026 (the file the current return network was trained on) |
| `backtests/*/cuadernillo_metricas.xlsx` | Daily returns, forecast volatility and parameters written by each backtest run |

The backtest folders were renamed (the originals encoded every parameter in a
name too long for Windows paths); the mapping and the date each was written:

| Folder | Original name | Written |
|---|---|---|
| `known_long_only` | `resultados_conocidoSI_activos11_shortFalse_leverageFalse_voltarget0.15_k5_rf0_rl0` | 2025-11-14 19:41 |
| `known_long_short` | `resultados_conocidoSI_activos11_shortTrue_leverageFalse_voltarget0.15_k5_rf0_rl0` | 2025-11-14 19:48 |
| `known_long_short_jan2026` | `..._shortTrue_leverageFalse_voltarget0.15_k5_rf0_rl0extra` | 2026-01-20 20:15 |
| `unknown_long_only` | `resultados_conocidoNO_..._shortFalse_leverageFalse_...` | 2025-11-13 20:14 |
| `unknown_long_short` | `resultados_conocidoNO_..._shortTrue_leverageFalse_...` | 2025-11-13 18:39 |
| `known_leveraged` | `..._shortTrue_leverageTrue_voltarget0.15_k5_rf0_rl0` | 2025-11-14 17:40 |
| `known_leveraged_cost` | `..._shortTrue_leverageTrue_..._rf0.0016666666666666668_rl0.0016666666666666668` | 2025-11-12 18:01 |

The return series inside these files are derived from Yahoo Finance data and
are included only so that the audit can be reproduced.
