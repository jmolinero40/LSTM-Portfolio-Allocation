# -*- coding: utf-8 -*-
"""
Created on Mon Oct 27 16:05:28 2025

@author: jmoli
"""
import numpy as np, pandas as pd
import tensorflow as tf
from pathlib import Path
import Carteras as c
import keras
from Descarga import recoge_datos
from retornos import make_sequence
import matplotlib.pyplot as plt
from joblib import load
from scipy.stats import norm, chi2
from reconstruye_precios import (
    PriceReconstructionConfig,
    build_price_predictions_history,
    plot_pred_vs_real,
    plot_grid_pred_vs_real,
)

print("\nCarga de modelos:")

# --- SOLO si el modelo de riesgo usó una loss/métrica custom ---
def nll_gauss(y_true, y_pred_logits, eps=1e-12):
    var = tf.nn.softplus(y_pred_logits) + eps
    return 0.5*tf.math.log(var) + 0.5*tf.square(y_true)/var

def mae_abs_r_vs_sigma(y_true, y_pred_logits, eps=1e-12):
    sigma = tf.sqrt(tf.nn.softplus(y_pred_logits) + eps)
    return tf.reduce_mean(tf.abs(tf.abs(y_true) - sigma))

def ema_baseline(values, span=20):
    import pandas as pd, numpy as np
    dff  = pd.DataFrame(values)
    ema  = dff.ewm(span=span, adjust=False, min_periods=span).mean().to_numpy()
    ema  = np.nan_to_num(ema, nan=0.0)
    # LAG 1: que la EMA de t represente sólo información hasta t-1
    ema_lag = np.vstack([ema[0:1], ema[:-1]])   # desplaza hacia abajo 1 fila
    return ema_lag


# Rutas (ajusta los nombres a los tuyos)
model_dir = Path(__file__).parent / "Modelos"
ret_path  = model_dir / "best_lstm_L60_h1_u64-32_bs32_assets11.keras"
risk_path = model_dir / "best_lstmVOL_L240_h1_u64-32_bs32.keras"

# Carga simple (sin compilar para evitar problemas de lambdas/objetos custom)
retornos_model = keras.models.load_model(ret_path, compile=False)
riesgos_model  = keras.models.load_model(
    risk_path,
    custom_objects={"nll_gauss": nll_gauss, "mae_abs_r_vs_sigma": mae_abs_r_vs_sigma},
    compile=False)

print("\nCargados los retornos:", retornos_model.input_shape, "->", retornos_model.summary())
print("\nCargados los riesgos :", riesgos_model.input_shape,  "->", riesgos_model.summary())

X_scaler = load(model_dir / "X_scaler_L60.pkl")   # el de entradas (ventanas)
y_scaler = load(model_dir / "y_scaler_L60.pkl")

print("\nCargados los escaladores")
print("\nCarga de datos")

#!!! Revisar en profundidad el acceso a datos porque hay errores seguro

# Llamamos a la función para que descargue los datos
close=recoge_datos()


# Localizamos el archivo que queremos
data_dir = Path(__file__).parent / "Datos"  # Vete a la carpeta Datos
parquet_files = list(data_dir.glob("*.parquet"))  # Coge todos los archivos

if not parquet_files:
    raise FileNotFoundError(f"\nNo se encontró ningún archivo .parquet en {data_dir}")

# Por lo pronto usaremos solo las rentabilidades
parquet_path = parquet_files[2]
print(f"\n📂 Cargando archivo: {parquet_path.name}")

# Lo devolvemos a un dataframe de pandas (las librerías exigen este formato)
df = pd.read_parquet(parquet_path)
tickers=list(df.columns)


tickers=df.columns

print("\n✅ Datos cargados correctamente")
print(df.head())

values = df.values.astype(np.float32)  # (T, n_features)
n_features = values.shape[1]   # Número de activos

span_ema = 20  #mismo span que en el entrenamiento de retornos
base = ema_baseline(values, span=span_ema)

retornos_X, retornos_y = make_sequence(values, lookback=60,  horizon=1)
riesgos_X,  riesgos_y  = make_sequence(values, lookback=240, horizon=1)

# ======== Parámetros del backtest ========
mercado_conocido='SI'
ret_lookback  = 60 
risk_lookback = 240 
horizon = 1
target_annual_vol=0.15
short_allowed=False
leverage_allowed=False
k = 5
rf=0
rl=0

# === Carpeta de resultados con el patrón solicitado ===
base_dir = Path(r"C:\Users\jmoli\Desktop\TFG ECONOMIA\Códigos\Backtest")
short_txt    = str(short_allowed)
leverage_txt = str(leverage_allowed)
run_dir_name = (
    f"resultados_conocido{mercado_conocido}"
    f"_activos{n_features}"
    f"_short{short_txt}"
    f"_leverage{leverage_txt}"
    f"_voltarget{target_annual_vol}"
    f"_k{k}"
    f"_rf{rf}"
    f"_rl{rl}extra"
)
out_dir = base_dir / run_dir_name
out_dir.mkdir(parents=True, exist_ok=True)
print("Guardando resultados en:", out_dir)

print("\nComienza el backtest.")

# Listas (mantenemos nombres)
rentabilidades = []
volatilidades = []
pesos_historico = []
cash_hist = []
fechas = []
long_pct_hist=[]
mu_pred_hist = []
leverage_hist=[]

# Alineación de ventanas (la corta se desplaza para terminar el mismo día que la larga)
offset  = risk_lookback - ret_lookback                  # 240-60 = 180
L       = min(len(riesgos_X), len(retornos_X) - offset) # longitud segura
start_df = risk_lookback                                # mapear t -> fecha en df

# Curva de capital (equity). Si prefieres empezar en 100, pon equity = 100.0
equity = 1.0
equity_curve = []

for t in range(L):
    i_risk = t
    i_ret  = t + offset

    # Ventanas que terminan el MISMO día
    ret_X = retornos_X[i_ret:i_ret+1]
    ret_y = retornos_y[i_ret:i_ret+1]    # realización del día t (log-ret por activo)
    risk_X = riesgos_X[i_risk:i_risk+1]

    # --- PREDICCIONES ALINEADAS (retornos y riesgos) ---
    # 1) RETORNOS: escala ret_X como en training (3D->2D->transform->3D)
    N = ret_X.shape[-1]
    ret_X_scaled = X_scaler.transform(ret_X.reshape(-1, N)).reshape(1, ret_lookback, N)

    # Predicción en la escala del modelo (residual escalado)
    y_pred_scaled = retornos_model.predict(ret_X_scaled, verbose=0)  # (1, N)

    # Invertimos el escalado de y (residual real)
    y_pred_res = y_scaler.inverse_transform(y_pred_scaled).ravel()   # (N,)

    # Reconstruimos μ̂ sumando el baseline (EMA) del día objetivo
    # target_idx = lookback + i_ret + (horizon - 1)  (coherente con make_sequence)
    target_idx = ret_lookback + i_ret + (horizon - 1)
    mu_pred = y_pred_res + base[target_idx, :]                       # (N,)
    mu_pred_hist.append(mu_pred.flatten())

    # 2) RIESGOS (σ̂): logits -> softplus -> sqrt
    risk_logits = riesgos_model.predict(risk_X, verbose=0).ravel()    # (N,)
    sigma_pred  = np.sqrt(tf.nn.softplus(risk_logits).numpy())        # (N,)
    
    # Construcción de cartera y PnL del MISMO día
    w, vol, top_idx, s, _cash_pre, long_pct, leverage = c.sharpe_like_model(
        mu_pred, sigma_pred, ret_X[0], k,
        short_allowed=short_allowed,
        target_annual_vol=target_annual_vol,
        leverage_allowed=leverage_allowed
    )
    
    # Chequeos y efectivo basado en pesos finales (coherente con leverage)
    gross = float(np.sum(np.abs(w)))
    if not leverage_allowed:
        assert gross <= 1.0 + 1e-3, f"Exposición bruta > 1 con leverage desactivado: {gross}"
    cash_t = max(0.0, 1.0 - gross)

    # Retorno aritmético de la cartera (a partir de log-retornos por activo)
    g_i = ret_y.ravel().astype(float)
    r_i = np.exp(g_i) - 1.0
    r_t = float(np.dot(w, r_i)) + cash_t*rf - leverage*rl

    # Guardar métricas diarias
    rentabilidades.append(r_t)
    volatilidades.append(float(vol))
    pesos_historico.append(w)
    cash_hist.append(float(cash_t))

    # Actualizar y guardar curva de capital y acumulada
    equity *= (1.0 + r_t)
    equity_curve.append(equity)
    long_pct_hist.append(long_pct)
    leverage_hist.append(leverage)
    
    # Fecha real
    fechas.append(df.index[start_df + t + horizon - 1])

    if (t + 1) % 240 == 0:
        print(f"Vamos por el día {str(fechas[-1].date())}")

# --- Series temporales ---
idx_fechas = pd.Index(fechas, name="fecha")
sr_r = pd.Series(rentabilidades, index=idx_fechas)
sr_equity = pd.Series(equity_curve, index=idx_fechas)

# --- Gráficos principales (guardados en out_dir) ---
plt.figure()
sr_equity.plot()
plt.title("Evolución histórica de la cartera (rentabilidad acumulada, %)")
plt.xlabel("Fecha"); plt.ylabel("% acumulado"); plt.grid(True); plt.tight_layout()
plt.savefig(out_dir / "equity_acumulada.png", dpi=150, bbox_inches="tight")

# Volatilidades: predicha (del modelo) y realizada (rolling 20)
sr_vol_pred = pd.Series(volatilidades, index=idx_fechas)
sr_vol_pred_ann = sr_vol_pred * np.sqrt(252) * 100.0  # anualizada en %

window = 20
sr_vol_real_ann = sr_r.rolling(window).std() * np.sqrt(252) * 100.0  # anualizada en %

plt.figure()
sr_vol_pred_ann.plot()
plt.title("Volatilidad predicha por el modelo (anualizada, %)")
plt.xlabel("Fecha"); plt.ylabel("Volatilidad %"); plt.grid(True); plt.tight_layout()
plt.savefig(out_dir / "vol_predicha_anualizada.png", dpi=150, bbox_inches="tight")

plt.figure()
sr_vol_real_ann.plot()
plt.title(f"Volatilidad realizada ({window} días, anualizada, %)")
plt.xlabel("Fecha"); plt.ylabel("Volatilidad %"); plt.grid(True); plt.tight_layout()
plt.savefig(out_dir / "vol_realizada_anualizada.png", dpi=150, bbox_inches="tight")

# Drawdown
sr_dd_pct = (sr_equity / sr_equity.cummax() - 1.0) * 100.0
plt.figure()
sr_dd_pct.plot()
plt.title("Drawdown (%)")
plt.xlabel("Fecha"); plt.ylabel("%"); plt.grid(True); plt.tight_layout()
plt.savefig(out_dir / "drawdown.png", dpi=150, bbox_inches="tight")
plt.close("all")

# ====== COMPARATIVO EN PREDICCIONES =========
mu_pred_df = pd.DataFrame(
    mu_pred_hist,
    index = fechas,
    columns = tickers
)

cfg = PriceReconstructionConfig(
    residual=False,          # ya pasas μ completo
    calibrate_bias="global",
    rolling_window=120,
    reanchor_freq="M",
    output_dir=str(out_dir),            # <-- guardar también predicciones aquí
    xlsx_name="predicted_prices.xlsx",
    csv_name="predicted_prices.csv"
)

pred_prices = build_price_predictions_history(
    mu_pred=mu_pred_df,
    realized_log_returns=df,   # necesario en 'global'
    close_prices=close,
    config=cfg
)

# Gráfico simple (no se guarda por defecto): lo guardamos nosotros
fig = plt.figure()
ax = fig.gca()
t0 = [t for t in tickers if t in pred_prices.columns and t in close.columns][:1]
if t0:
    t = t0[0]
    pred = pred_prices[t].dropna(); real = close[t].reindex(pred.index).ffill()
    ax.plot(pred.index, pred.values, label=f"Pred {t}")
    ax.plot(real.index, real.values, linestyle="--", label=f"Real {t}")
    ax.set_xlabel("Fecha"); ax.set_ylabel("Precio"); ax.set_title("Precio predicho vs. Precio real"); ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"pred_vs_real_{t}.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# Grid de predicciones: la función guarda en una ruta fija -> copiamos a out_dir
from shutil import copyfile
png_path_src = plot_grid_pred_vs_real(
    pred_prices, close, tickers, ncols=3, figsize=(14,14),
    mercado=1, short_allowed=('SI' if short_allowed else 'NO'), mercado_conocido=mercado_conocido
)
png_path_dst = out_dir / "pred_vs_real_grid.png"
try:
    copyfile(png_path_src, png_path_dst)
except Exception:
    pass
print("Grid guardado en:", png_path_dst)

# ====== RESUMEN DE MÉTRICAS ======
rets = np.asarray(rentabilidades, dtype=float)
equity_arr = np.asarray(equity_curve, dtype=float)
sigma_log_daily = np.asarray(volatilidades, dtype=float)  # σ̂_p diaria (log)

if len(rets) == 0:
    raise ValueError("No hay 'rentabilidades' calculadas.")
if len(equity_arr) != len(rets):
    # Por si equity tiene el punto inicial (len = len(rets)+1)
    if len(equity_arr) == len(rets) + 1:
        equity_arr = equity_arr[1:]
    else:
        raise ValueError("Longitudes inconsistentes entre equity_curve y rentabilidades.")

# Performance básicas
def _annualize_mean(r, periods=252): return (1 + np.mean(r))**periods - 1
def _annualize_vol (r, periods=252): return np.std(r, ddof=1) * np.sqrt(periods)
def _sharpe(r, rf_daily=0.0): 
    ex = r - rf_daily
    sd = np.std(ex, ddof=1)
    return (np.mean(ex)/sd) * np.sqrt(252) if sd>0 else np.nan

cagr = (equity_arr[-1]/equity_arr[0])**(252/len(equity_arr)) - 1
vol_ann = _annualize_vol(rets)
sr_ann  = _sharpe(rets, rf/252.0)
max_dd  = (pd.Series(equity_arr)/pd.Series(equity_arr).cummax()-1.0).min()

perf = pd.DataFrame([{
    "CAGR": cagr, "Vol_ann": vol_ann, "Sharpe": sr_ann, "MaxDD": max_dd,
    "Gross_mean": float(np.mean([np.sum(np.abs(w)) for w in pesos_historico])),
    "Net_mean": float(np.mean([np.sum(w) for w in pesos_historico])),
}])

# Backtest de VaR (paramétrico log-normal sencillo)
alpha = 0.05
VaR = - (np.exp(norm.ppf(alpha) * sigma_log_daily) - 1.0)  # < 0 (pérdida)
loss = -rets
I = (loss > -VaR).astype(int)
var_tests = pd.DataFrame([{
    "alpha": alpha,
    "Violations": int(I.sum()),
    "T": int(len(I)),
    "Uncond_p": I.mean(),
}])

# Prueba Kupiec (LR_uc)
p_hat = I.mean()
if p_hat>0 and p_hat<1:
    LR_uc = -2 * ( (len(I)-(I.sum()))*np.log(1-alpha) + (I.sum())*np.log(alpha) \
                   - ((len(I)-(I.sum()))*np.log(1-p_hat) + (I.sum())*np.log(p_hat)) )
    var_tests.loc[0,"LR_uc"] = LR_uc
    var_tests.loc[0,"p_LR_uc"] = 1 - chi2.cdf(LR_uc, df=1)

# Duration test y ES (igual que tenías)
from scipy.stats import kstest
try:
    idx_exc = np.where(I==1)[0]
    if len(idx_exc) >= 2:
        durations = np.diff(idx_exc)
        U_dur = 1.0 - (1.0 - alpha)**durations
        ks_stat, ks_p = kstest(U_dur, 'uniform')
        var_tests.loc[0, "Duration_mean"]  = float(np.mean(durations))
        var_tests.loc[0, "Duration_exp"]   = float(1.0/alpha)
        var_tests.loc[0, "Duration_KS"]    = float(ks_stat)
        var_tests.loc[0, "p_Duration_KS"]  = float(ks_p)
        var_tests.loc[0, "N_durations"]    = int(len(durations))
    else:
        var_tests.loc[0, "Duration_mean"]  = np.nan
        var_tests.loc[0, "Duration_exp"]   = float(1.0/alpha)
        var_tests.loc[0, "Duration_KS"]    = np.nan
        var_tests.loc[0, "p_Duration_KS"]  = np.nan
        var_tests.loc[0, "N_durations"]    = int(max(len(idx_exc)-1, 0))
except Exception:
    pass

def es_simple_from_sigma_log(sigma_log_series, alpha=0.01):
    sig = np.asarray(sigma_log_series, dtype=float)
    z = norm.ppf(alpha)
    Phi_z = max(norm.cdf(z), 1e-12)
    num = np.exp(0.5*sig**2) * norm.cdf(z - sig)
    cond_exp = num / Phi_z
    ES = 1.0 - cond_exp  # positivo
    return ES

try:
    ES_t = es_simple_from_sigma_log(sigma_log_daily, alpha=alpha)
    thr = (-VaR).astype(float)
    L   = loss.astype(float)
    denom = np.maximum(ES_t - thr, 1e-12)
    Z = I * ( (L - thr)/denom - 1.0 )
    n_eff = int(I.sum())
    if n_eff >= 5:
        Z_bar = float(np.mean(Z))
        Z_sd  = float(np.std(Z, ddof=1))
        zscore = Z_bar / (Z_sd/np.sqrt(max(n_eff,1)))
        p_as = float(2*(1 - norm.cdf(abs(zscore))))
    else:
        Z_bar, zscore, p_as = np.nan, np.nan, np.nan

    var_tests.loc[0, "ES_model_mean"] = float(np.mean(ES_t))
    var_tests.loc[0, "ES_realized"]   = float(L[I==1].mean()) if I.sum()>0 else np.nan
    var_tests.loc[0, "AS_meanZ"]      = Z_bar
    var_tests.loc[0, "AS_z"]          = zscore
    var_tests.loc[0, "p_AS"]          = p_as
except Exception:
    pass

# === Guardado de métricas (CSV/TXT) en out_dir ===
perf_path = out_dir / "metricas_performance.csv"
var_path  = out_dir / "metricas_var_tests.csv"
txt_path  = out_dir / "informe_resumen.txt"

perf.to_csv(perf_path, index=False)
var_tests.to_csv(var_path, index=False)

with open(txt_path, "w", encoding="utf-8") as f:
    f.write("=== RESUMEN DE PERFORMANCE ===\n")
    for k_, v_ in perf.iloc[0].items():
        f.write(f"{k_:>18}: {v_}\n")
    f.write("\n=== BACKTEST VaR ===\n")
    for k_, v_ in var_tests.iloc[0].items():
        f.write(f"{k_:>18}: {v_}\n")

print("\n✅ Métricas calculadas.")
print("• Performance  ->", perf_path)
print("• VaR tests    ->", var_path)
print("• Informe TXT  ->", txt_path)

# === Benchmarks (S&P500 e IBEX35) + gráficos y betas ===
try:
    import yfinance as yf
    bench = yf.download(['^GSPC','^IBEX'], start='2014-01-01', auto_adjust=False, progress=False)["Adj Close"]
    bench = bench.dropna()
    sp_base   = bench['^GSPC'] / bench['^GSPC'].iloc[0]
    ibex_base = bench['^IBEX'] / bench['^IBEX'].iloc[0]
    eq_base = (sr_equity / sr_equity.iloc[0]).rename('Equity')

    # Gráfico en fila
    fig, axes = plt.subplots(1, 3, figsize=(16, 4), sharex=False)
    eq_base.plot(ax=axes[0]); axes[0].set_title("Equity (base 1)")
    axes[0].set_xlabel("Fecha"); axes[0].set_ylabel("Índice base 1"); axes[0].grid(True)
    sp_base.plot(ax=axes[1]); axes[1].set_title("S&P 500 (base 1)")
    axes[1].set_xlabel("Fecha"); axes[1].set_ylabel(""); axes[1].grid(True)
    ibex_base.plot(ax=axes[2]); axes[2].set_title("IBEX 35 (base 1)")
    axes[2].set_xlabel("Fecha"); axes[2].set_ylabel(""); axes[2].grid(True)
    fig.tight_layout()
    fig.savefig(out_dir / "comparativa_en_fila.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Gráfico combinado
    plt.figure(figsize=(10,5))
    eq_base.plot(label="Equity")
    sp_base.plot(label="S&P 500")
    ibex_base.plot(label="IBEX 35")
    plt.title("Comparativa (base 1)")
    plt.xlabel("Fecha"); plt.ylabel("Índice base 1"); plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig(out_dir / "comparativa_combinado.png", dpi=150, bbox_inches="tight")
    plt.close()
except Exception as _e:
    pass

# === Cuadernillo Excel ===
xlsx_book = out_dir / "cuadernillo_metricas.xlsx"
with pd.ExcelWriter(xlsx_book, engine="xlsxwriter") as writer:
    perf.to_excel(writer, sheet_name="performance", index=False)
    var_tests.to_excel(writer, sheet_name="var_tests", index=False)
    params = pd.DataFrame([{
        "mercado_conocido": mercado_conocido,
        "n_features": n_features,
        "ret_lookback": ret_lookback,
        "risk_lookback": risk_lookback,
        "horizon": horizon,
        "target_annual_vol": target_annual_vol,
        "short_allowed": short_allowed,
        "leverage_allowed": leverage_allowed,
        "k": k,
        "rf": rf,
        "rl": rl
    }])
    params.to_excel(writer, sheet_name="params", index=False)
    pd.Series(rentabilidades, index=fechas, name="ret_diario").to_frame().to_excel(writer, sheet_name="retornos")
    pd.Series(equity_curve, index=fechas, name="equity").to_frame().to_excel(writer, sheet_name="equity")
    pd.Series(volatilidades, index=fechas, name="sigma_log_diaria").to_frame().to_excel(writer, sheet_name="sigma_log")

print("Cuadernillo Excel ->", xlsx_book)
print("\n✅ Backtest finalizado.")
