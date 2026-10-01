# -*- coding: utf-8 -*-
"""
Predicción de riesgo (volatilidad futura) con LSTM + rasgos GEW (GARCH/EGARCH/EWMA).

Cambios clave:
- Target RV_h = sqrt(sum_{j=1..h} r_{t+j}^2), opcional en log-espacio (por defecto True).
- Features dinámicos: varianza condicional EWMA/GARCH/EGARCH alineados en t (sin fuga).
- GARCH/EGARCH se calibran SOLO con tramo train y se proyectan OOS (rolling opcional).
- LSTM con recurrent_dropout; pérdida MSE sobre log-RV por defecto.
- Baseline EWMA y evaluación walk-forward opcional.
"""

from pathlib import Path
from typing import Optional
import pandas as pd, numpy as np

# Deep Learning / escalado
import tensorflow as tf
from sklearn.preprocessing import RobustScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l2

# Métricas
from sklearn.metrics import mean_squared_error, mean_absolute_error

# GARCH
try:
    from arch import arch_model
except Exception as e:
    raise ImportError("Falta el paquete 'arch'. Instala con: pip install arch") from e

# Helpers externos (datos)
from Descarga import recoge_datos  # tu pipeline de descarga/actualización

# Semillas
np.random.seed(42)
tf.random.set_seed(42)

# ================================
# 1. Carga de datos
# ================================
def carga_datos(parquet_idx: int = 2):
    """
    Actualiza y carga el parquet.
    """
    recoge_datos()  # asegura parquet actualizado

    data_dir = Path(__file__).parent / "Datos"
    parquet_files = sorted(list(data_dir.glob("*.parquet")))
    if not parquet_files:
        raise FileNotFoundError(f"No se encontró ningún archivo .parquet en {data_dir}")

    parquet_idx = min(max(0, parquet_idx), len(parquet_files)-1)
    parquet_path = parquet_files[parquet_idx]
    print(f"\n📂 Cargando archivo: {parquet_path.name}")

    df = pd.read_parquet(parquet_path)
    print("\n✅ Datos cargados correctamente")
    print(df.head())

    assert isinstance(df.index, pd.DatetimeIndex), "El índice no está en formato fecha"
    return df


# ===================================================
# 2. Helpers: secuencias, splits, escalado, métricas
# ===================================================

def make_sequence_risk(X_mat: np.ndarray, y_mat: np.ndarray, lookback=60, index: Optional[pd.DatetimeIndex]=None):
    """
    Convierte X (T,F) e y (T,K) en:
      X_seq: (N, lookback, F), y_seq: (N, K), opcionalmente devuelve índice de cada ventana.
    """
    if not isinstance(X_mat, np.ndarray) or not isinstance(y_mat, np.ndarray):
        raise TypeError("make_sequence_risk espera ndarrays para X e y (usa .values si tienes DataFrames).")

    if X_mat.shape[0] != y_mat.shape[0]:
        raise ValueError(f"X e y deben tener la misma longitud temporal. X={X_mat.shape}, y={y_mat.shape}")

    T = X_mat.shape[0]
    if T <= lookback:
        raise ValueError(
            f"No hay suficientes filas para crear ventanas: T={T}, lookback={lookback}. "
            "Reduce lookback o amplía datos."
        )

    X_list, y_list = [], []
    for i in range(lookback, T):
        X_list.append(X_mat[i - lookback:i, :])
        y_list.append(y_mat[i, :])

    X_seq = np.asarray(X_list, dtype=np.float32)
    y_seq = np.asarray(y_list, dtype=np.float32)

    seq_index = None
    if isinstance(index, pd.DatetimeIndex):
        seq_index = index[lookback:]

    return X_seq, y_seq, seq_index


def split_scale_simple(X, y, n_dyn_features, train_ratio=0.7, val_ratio=0.85):
    """
    X: (N, L, F_total), y: (N, K)
    n_dyn_features: nº de columnas dinámicas (retornos); el resto son estáticas (GEW condvar).
    """
    N, L, F = X.shape
    i_tr = max(1, int(N * train_ratio))
    i_va = max(i_tr + 1, int(N * val_ratio))
    if i_va >= N:
        i_va = N - 1  # deja algo de test

    # splits
    X_tr, y_tr = X[:i_tr], y[:i_tr]
    X_va, y_va = X[i_tr:i_va], y[i_tr:i_va]
    X_te, y_te = X[i_va:], y[i_va:]

    # separar dinámicas / estáticas
    X_tr_dyn, X_tr_stat = X_tr[:, :, :n_dyn_features], X_tr[:, :, n_dyn_features:]
    X_va_dyn, X_va_stat = X_va[:, :, :n_dyn_features], X_va[:, :, n_dyn_features:]
    X_te_dyn, X_te_stat = X_te[:, :, :n_dyn_features], X_te[:, :, n_dyn_features:]

    scaler = RobustScaler()

    def fit_trans(A):
        s = A.shape[-1]
        A2 = A.reshape(-1, s)
        A2 = scaler.fit_transform(A2)
        return A2.reshape(A.shape)

    def trans(A):
        s = A.shape[-1]
        A2 = A.reshape(-1, s)
        A2 = scaler.transform(A2)
        return A2.reshape(A.shape)

    if n_dyn_features > 0:
        X_tr_dyn = fit_trans(X_tr_dyn)
        X_va_dyn = trans(X_va_dyn)
        X_te_dyn = trans(X_te_dyn)

    # recomponer
    X_tr = np.concatenate([X_tr_dyn, X_tr_stat], axis=-1)
    X_va = np.concatenate([X_va_dyn, X_va_stat], axis=-1)
    X_te = np.concatenate([X_te_dyn, X_te_stat], axis=-1)

    return (X_tr, y_tr, X_va, y_va, X_te, y_te), scaler


def hmae(y_true, y_pred, eps=1e-12):
    # 1/T * sum |1 - v_hat / RV|
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    return np.mean(np.abs(1.0 - (y_pred / (y_true + eps))))


def hmse(y_true, y_pred, eps=1e-12):
    # 1/T * sum (1 - v_hat / RV)^2
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    return np.mean((1.0 - (y_pred / (y_true + eps)))**2)


# =======================================
# 3. Target de riesgo: RV futura por activo
# =======================================
def realized_vol_future_panel(df_returns: pd.DataFrame, horizon: int = 2,
                              use_sum_squares: bool = True, log_target: bool = True) -> pd.DataFrame:
    """
    RV_t(h) alineada en t.
    - use_sum_squares=True -> sqrt(sum_{j=1..h} r_{t+j}^2)
      else -> std(r_{t+1..t+h})
    - log_target=True -> log(RV)
    """
    out = {}
    for col in df_returns.columns:
        r = df_returns[col]
        if use_sum_squares:
            rv = np.sqrt(r.shift(-1).rolling(horizon).apply(lambda x: np.sum(x**2), raw=True))
        else:
            rv = r.shift(-1).rolling(horizon).std()
        out[col] = rv
    rv_df = pd.DataFrame(out, index=df_returns.index).dropna()
    if log_target:
        rv_df = np.log(rv_df.clip(lower=1e-12))
    return rv_df


# ===================================================
# 4. Rasgos GEW dinámicos (varianza condicional)
# ===================================================
def ewma_var_series(r: pd.Series, lam: float = 0.97) -> pd.Series:
    """Varianza EWMA_t = lam * Var_{t-1} + (1-lam) * r_{t-1}^2 (alineada en t)."""
    v = []
    prev = float(r.var()) if r.notna().any() else 0.0  # semilla
    first = True
    for ret in r.shift(1):  # usa info hasta t-1
        if pd.isna(ret):
            # Sólo el primer punto suele ser NaN por el shift; déjalo como semilla
            v.append(prev if first else prev)
            first = False
            continue
        prev = lam * prev + (1.0 - lam) * (ret**2)
        v.append(prev)
        first = False
    return pd.Series(v, index=r.index)


def garch_condvar_series(r: pd.Series, refit_idx: int | None, p=1, q=1, dist='normal',
                         rolling_refit: bool=False, window: int=1000) -> pd.Series:
    r_pct = (r * 100.0)
    v = pd.Series(index=r.index, dtype=float)
    # Si no hay suficiente histórico para ajustar con seguridad, vuelve a EWMA
    if refit_idx is None or refit_idx < 250 or r_pct.iloc[:int(refit_idx)].dropna().shape[0] < 250:
        return ewma_var_series(r, lam=0.97)

    try:
        if not rolling_refit:
            model = arch_model(r_pct.iloc[:int(refit_idx)].dropna(), vol='GARCH', p=p, q=q, dist=dist)
            res = model.fit(disp='off')
            cond = (res.conditional_volatility**2)  # en %^2
            v.loc[cond.index] = cond.values / (100.0**2)

            # proyección fuera de muestra recursiva
            h = float(cond.iloc[-1])
            alpha = float(res.params.get('alpha[1]', 0.0))
            beta  = float(res.params.get('beta[1]',  0.0))
            omega = float(res.params.get('omega',    0.0))
            for t in r_pct.index[len(cond):]:
                r_tm1 = r_pct.shift(1).loc[t]
                if pd.isna(r_tm1):
                    v.loc[t] = h / (100.0**2); continue
                h = omega + alpha * (r_tm1**2) + beta * h
                v.loc[t] = h / (100.0**2)
        else:
            for i, t in enumerate(r.index):
                if i == 0:
                    v.loc[t] = np.nan; continue
                start = max(0, i - window)
                r_fit = (r.iloc[start:i] * 100.0).dropna()
                if len(r_fit) < 250:
                    v.loc[t] = np.nan; continue
                res = arch_model(r_fit, vol='GARCH', p=p, q=q, dist=dist).fit(disp='off')
                v.loc[t] = (res.conditional_volatility.iloc[-1]**2) / (100.0**2)
    except Exception:
        # Fallback completo si algo falla en el ajuste
        return ewma_var_series(r, lam=0.97)

    return v.fillna(method='ffill').fillna(method='bfill')


def egarch_condvar_series(r: pd.Series, refit_idx: int | None, dist='normal') -> pd.Series:
    r_pct = (r * 100.0)
    # Guardas de tamaño; si no hay suficiente, vuelve a EWMA
    if refit_idx is None or refit_idx < 250 or r_pct.iloc[:int(refit_idx)].dropna().shape[0] < 250:
        return ewma_var_series(r, lam=0.97)

    try:
        model = arch_model(r_pct.iloc[:int(refit_idx)].dropna(), vol='EGARCH', p=1, o=1, q=1, dist=dist)
        res = model.fit(disp='off')
        cond = res.conditional_volatility  # sigma_t en %
        v = pd.Series(index=r.index, dtype=float)
        v.loc[cond.index] = (cond.values**2) / (100.0**2)
        # fuera de muestra: sostener último valor
        last_h = (cond.iloc[-1]**2) / (100.0**2)
        v.iloc[len(cond):] = last_h
        return v.fillna(method='ffill').fillna(method='bfill')
    except Exception:
        return ewma_var_series(r, lam=0.97)



# ======================================
# 5. Modelo
# ======================================
def build_model(n_features_total, lookback, n_out,
                neurons1=64, neurons2=32, learning_rate=1e-3, dropout=0.2):
    """
    Entrada: (lookback, n_features_total); salida: (n_out) = nº de activos.
    """
    n_features_total = int(n_features_total)
    lookback   = int(lookback)
    units1     = int(neurons1)
    units2     = int(neurons2)
    lr         = float(learning_rate)
    dr         = float(dropout)

    assert units1 > 0 and units2 > 0, "neurons deben ser enteros positivos"
    assert 0.0 <= dr < 1.0, "dropout debe estar en [0,1)"
    assert lookback >= 1 and n_features_total >= 1, "lookback/n_features inválidos"

    model = Sequential([
        tf.keras.Input(shape=(lookback, n_features_total)),
        LSTM(units1, return_sequences=True, kernel_regularizer=l2(1e-5), recurrent_dropout=0.1),
        Dropout(dr),
        LSTM(units2, kernel_regularizer=l2(1e-5), recurrent_dropout=0.1),
        Dense(n_out),  # salida lineal (adecuada para log-RV)
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  loss='mse',  # MSE en log-espacio
                  metrics=['mae'])
    return model


# ===========================
# 6. Baselines
# ===========================
def baseline_ewma_forecast(df_returns: pd.DataFrame, horizon: int, lam: float = 0.97, log_target: bool = True):
    """
    Forecast proxy: usa EWMA_t como varianza diaria y proyecta RV_h ≈ sqrt(h * var_t).
    Devuelve DataFrame alineado con RV_t(h). Si log_target, devuelve log(RV_hat).
    """
    out = {}
    for col in df_returns.columns:
        var_t = ewma_var_series(df_returns[col], lam=lam)
        rv_h  = np.sqrt(horizon * var_t).clip(lower=1e-12)
        out[col] = np.log(rv_h) if log_target else rv_h
    return pd.DataFrame(out, index=df_returns.index)


def compare_to_baselines(ret_df, rv_df, horizon, log_target=True):
    # alineación
    ewma_rvhat = baseline_ewma_forecast(ret_df.loc[rv_df.index], horizon=horizon, log_target=log_target)
    y_true = rv_df.values.reshape(-1)
    y_hat_ew = ewma_rvhat.values.reshape(-1)

    rmse = float(np.sqrt(mean_squared_error(y_true, y_hat_ew)))
    mae  = float(mean_absolute_error(y_true, y_hat_ew))
    # Si rv_df está en log, HMAE/HMSE no aplican directamente (definidas en nivel).
    return {"EWMA_RMSE": rmse, "EWMA_MAE": mae}


# ===========================
# 7. Entrenamiento end-to-end
# ===========================
def ejecuta_red_riesgo(
    df,
    lookback=22, horizon=3,
    train_number=0.7, val_number=0.85,
    neurons1=96, neurons2=48, learning_rate=1e-3, batch_size=64,
    dropout=0.25, verbose_fit=1,
    use_sum_squares=True, log_target=True,
    include_garch=True, include_egarch=True, include_ewma=True,
    rolling_garch=False, rolling_window=1000
):
    """
    LSTM para predecir RV_h con:
      - Target: log(RV_h) (por defecto)
      - Rasgos GEW dinámicos: EWMA/GARCH/EGARCH (sin fuga)
      - Extras: |r|, r^2, signo(r), lag de RV en nivel
      - Pérdida ponderada por activo
      - Métricas en log y en nivel (+ DM test frente a EWMA)
    """

    # ===== 1) Target en log =====
    rv_df = realized_vol_future_panel(
        df, horizon=horizon,
        use_sum_squares=use_sum_squares,
        log_target=log_target
    )

    # ===== 2) Alinear retornos con target =====
    ret_df = df.loc[rv_df.index]
    N_total = len(rv_df)
    i_train = int(N_total * train_number)

    # ===== 3) Rasgos GEW dinámicos =====
    gew_parts = []
    for col in ret_df.columns:
        feats = {}
        if include_ewma:
            feats[f'{col}_ewma_v'] = ewma_var_series(ret_df[col], lam=0.97)
        if include_garch:
            feats[f'{col}_garch_v'] = garch_condvar_series(
                ret_df[col], refit_idx=i_train,
                rolling_refit=rolling_garch, window=rolling_window
            )
        if include_egarch:
            feats[f'{col}_egarch_v'] = egarch_condvar_series(
                ret_df[col], refit_idx=i_train
            )
        if feats:
            gew_parts.append(pd.DataFrame(feats, index=ret_df.index))
    gew_df = pd.concat(gew_parts, axis=1) if len(gew_parts) else pd.DataFrame(index=ret_df.index)
    gew_df = gew_df.loc[rv_df.index]

    # ===== 4) Extras baratos que ayudan =====
    abs_r   = ret_df.abs().add_suffix('_abs')
    sq_r    = (ret_df**2).add_suffix('_sq')
    sign_r  = np.sign(ret_df).add_suffix('_sign')
    # lag de RV en nivel aunque el objetivo esté en log
    rv_level = np.exp(rv_df) if log_target else rv_df
    lag_rv1  = rv_level.shift(1).add_suffix('_rv_lag1')

    extra_df = pd.concat([abs_r, sq_r, sign_r, lag_rv1], axis=1).loc[rv_df.index]
    extra_df = extra_df.replace([np.inf, -np.inf], np.nan)\
                       .fillna(method='ffill').fillna(method='bfill')

    # ===== 5) LIMPIEZA GLOBAL anti-NaN/Inf =====
    rv_pref = rv_df.add_prefix('__RV__')
    all_df = pd.concat([ret_df, gew_df, extra_df, rv_pref], axis=1)
    all_df = all_df.replace([np.inf, -np.inf], np.nan).dropna(how='any')

    # Re-separar
    ret_cols = ret_df.columns.tolist()
    rv_cols  = rv_df.columns.tolist()
    rv_cols_pref = [f'__RV__{c}' for c in rv_cols]
    other_cols = [c for c in all_df.columns if c not in ret_cols + rv_cols_pref]

    ret_df = all_df[ret_cols]
    rv_df  = all_df[rv_cols_pref].rename(columns=lambda c: c.replace('__RV__', ''))
    aux_df = all_df[other_cols] if len(other_cols) else pd.DataFrame(index=all_df.index)

    # ===== 6) Construir X/y y ventanear =====
    blocks = [ret_df.values.astype(np.float32)]
    if gew_df.shape[1] > 0 or aux_df.shape[1] > 0:
        more = np.concatenate(
            [arr for arr in [
                gew_df.reindex(all_df.index).values.astype(np.float32) if gew_df.shape[1] else None,
                aux_df.values.astype(np.float32) if aux_df.shape[1] else None
            ] if arr is not None],
            axis=1
        ) if (gew_df.shape[1] + aux_df.shape[1]) > 0 else None
        if more is not None:
            blocks.append(more)
    X_base = np.concatenate(blocks, axis=1)
    y_base = rv_df.values.astype(np.float32)

    X, y, seq_index = make_sequence_risk(X_base, y_base, lookback=lookback, index=ret_df.index)

    # ===== 7) Split + escalado (solo dinámicas) =====
    n_dyn = ret_df.shape[1]  # nº de retornos
    (X_train, y_train, X_val, y_val, X_test, y_test), X_scaler = split_scale_simple(
        X, y, n_dyn_features=n_dyn, train_ratio=train_number, val_ratio=val_number
    )
    print("\n✅ Split y escalado completados")

    # ===== 8) Modelo (ligeramente más capaz) =====
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.regularizers import l2

    n_features_total = X_train.shape[-1]
    n_out = y_train.shape[-1]

    model = Sequential([
        tf.keras.Input(shape=(lookback, n_features_total)),
        LSTM(neurons1, return_sequences=True, kernel_regularizer=l2(1e-5), recurrent_dropout=0.1),
        Dropout(dropout),
        LSTM(neurons2, kernel_regularizer=l2(1e-5), recurrent_dropout=0.1),
        Dense(n_out),
    ])

    # ===== 9) Pérdida ponderada por activo =====
    col_std = np.std(y_train, axis=0) + 1e-6
    w = (1.0 / col_std)
    w = w / w.mean()
    w_tf = tf.constant(w.reshape(1, -1), dtype=tf.float32)

    def weighted_mse(y_true, y_pred):
        err2 = tf.square(y_true - y_pred)
        return tf.reduce_mean(tf.reduce_mean(err2 * w_tf, axis=-1))

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss=weighted_mse, metrics=['mae'])

    print("\n🧠 Arquitectura del modelo:\n")
    model.summary()

    # ===== 10) Entrenamiento =====
    es  = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    rlr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-5, verbose=1)
    ckp_path = Path(__file__).parent / "Modelos"; ckp_path.mkdir(exist_ok=True)
    ckp_name = ckp_path / f"best_lstm_risk_L{lookback}_h{horizon}_u{neurons1}-{neurons2}_bs{batch_size}.keras"
    ckp = tf.keras.callbacks.ModelCheckpoint(ckp_name, monitor='val_loss', save_best_only=True)

    hist = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=200,
        batch_size=batch_size,
        shuffle=False,
        callbacks=[es, rlr, ckp],
        verbose=verbose_fit
    )

    print("\n✅ Entrenamiento finalizado")

    # ===== 11) Métricas en log =====
    y_pred = model.predict(X_test, verbose=0)
    mse_log  = mean_squared_error(y_test.reshape(-1), y_pred.reshape(-1))
    rmse_log = float(np.sqrt(mse_log))
    mae_log  = mean_absolute_error(y_test.reshape(-1), y_pred.reshape(-1))
    print("\n🔎 Métricas (test) [LOG]:")
    print(f"RMSE(log): {rmse_log:.6e} | MAE(log): {mae_log:.6e}")

    # ===== 12) Métricas en NIVEL (interpretables) =====
    y_true_lvl = np.exp(y_test) if log_target else y_test
    y_pred_lvl = np.exp(y_pred) if log_target else y_pred
    rmse_lvl = float(np.sqrt(mean_squared_error(y_true_lvl.reshape(-1), y_pred_lvl.reshape(-1))))
    mae_lvl  = float(mean_absolute_error(y_true_lvl.reshape(-1), y_pred_lvl.reshape(-1)))
    hmae_lvl = float(np.mean(np.abs(1 - (y_pred_lvl.reshape(-1) / (y_true_lvl.reshape(-1) + 1e-12)))))
    hmse_lvl = float(np.mean((1 - (y_pred_lvl.reshape(-1) / (y_true_lvl.reshape(-1) + 1e-12)))**2))
    print(f"[NIVEL] RMSE={rmse_lvl:.6e} | MAE={mae_lvl:.6e} | HMAE={hmae_lvl:.6e} | HMSE={hmse_lvl:.6e}")

    # ===== 13) MAE por activo (en log) =====
    tickers = df.columns.to_list()
    per_asset_mae_log = {t: mean_absolute_error(y_test[:, i], y_pred[:, i]) for i, t in enumerate(tickers)}
    print("\nMAE(log) por activo:", per_asset_mae_log)

    # ===== 14) Baseline EWMA y Diebold–Mariano =====
    ewma_log_df = baseline_ewma_forecast(ret_df, horizon=horizon, log_target=True)
    ewma_log = ewma_log_df.loc[rv_df.index].values[-len(y_test):]  # alinea con test

    def dm_test(y_true_log, yhat_a_log, yhat_b_log, h=1, power=2):
        e_a = (y_true_log - yhat_a_log).reshape(-1)
        e_b = (y_true_log - yhat_b_log).reshape(-1)
        d_t = np.abs(e_a)**power - np.abs(e_b)**power
        d_bar = np.mean(d_t); T = len(d_t)
        lag = max(1, h-1)
        gamma0 = np.var(d_t, ddof=1); nw = gamma0
        for j in range(1, lag+1):
            if T - j <= 1: break
            gammaj = np.cov(d_t[j:], d_t[:-j], ddof=1)[0,1]
            wj = 1 - j/(lag+1)
            nw += 2 * wj * gammaj
        se = np.sqrt(max(nw, 1e-12) / T)
        dm_stat = d_bar / (se + 1e-12)
        try:
            from scipy.stats import norm
            pval = 2 * (1 - norm.cdf(np.abs(dm_stat)))
        except Exception:
            # aproximación mediante normal estándar si scipy no está
            pval = float(np.exp(-0.5*dm_stat**2) * np.sqrt(2/np.pi))  # aprox
        return float(dm_stat), float(pval)

    dm_stat, dm_p = dm_test(y_test, y_pred, ewma_log, h=horizon, power=2)
    print(f"\n🧪 Diebold–Mariano (LSTM vs EWMA) en log: DM={dm_stat:.3f}  p-value={dm_p:.4f}")

    baselines = {
        "EWMA_RMSE_log": float(np.sqrt(mean_squared_error(y_test.reshape(-1), ewma_log.reshape(-1)))),
        "EWMA_MAE_log":  float(mean_absolute_error(y_test.reshape(-1), ewma_log.reshape(-1)))
    }
    print("📏 Baselines (log):", baselines)

    # ===== 15) Resumen =====
    return {
        "lookback": lookback, "horizon": horizon,
        "train_number": train_number, "val_number": val_number,
        "neurons1": neurons1, "neurons2": neurons2,
        "learning_rate": learning_rate, "batch_size": batch_size, "dropout": dropout,
        "log_target": log_target, "use_sum_squares": use_sum_squares,
        "rmse_log": rmse_log, "mae_log": mae_log,
        "rmse_level": rmse_lvl, "mae_level": mae_lvl,
        "hmae_level": hmae_lvl, "hmse_level": hmse_lvl,
        "per_asset_mae_log": per_asset_mae_log,
        "dm_stat": dm_stat, "dm_pvalue": dm_p,
        "baselines_log": baselines
    }



# ===========================
# 8. Walk-forward opcional
# ===========================
def walk_forward_eval(df, lookback, horizon, splits=4,
                      train_ratio=0.6, val_ratio=0.8, **kwargs):
    """
    Divide el tiempo en 'splits' segmentos. En cada split:
    - usa [0 : k*len/splits] como 'histórico'
    - dentro, hace train/val/test con train_ratio/val_ratio relativos a ese histórico.
    Devuelve promedio de métricas.
    """
    T = len(df)
    cut_points = [int(T * (i+1)/splits) for i in range(splits)]
    metrics = []
    for k, cp in enumerate(cut_points, 1):
        df_k = df.iloc[:cp].copy()
        print(f"\n=== Walk-Forward {k}/{splits} (0:{cp}) ===")
        res = ejecuta_red_riesgo(df_k, lookback=lookback, horizon=horizon,
                                 train_number=train_ratio, val_number=val_ratio, **kwargs)
        metrics.append(res)
        print(f"[WF {k}/{splits}] RMSE={res['rmse']:.4e} | MAE={res['mae']:.4e}")
    mean_rmse = float(np.mean([m['rmse'] for m in metrics]))
    mean_mae  = float(np.mean([m['mae']  for m in metrics]))
    return {"mean_rmse": mean_rmse, "mean_mae": mean_mae, "details": metrics}


# ==============================
# 9. Ejecución de ejemplo
# ==============================
if __name__ == "__main__":
    # Hiperparámetros base
    LOOKBACK = 22
    HORIZON  = 3
    NEU1, NEU2 = 64, 32
    LR = 1e-3
    BS = 64
    DR = 0.2

    # Carga
    df = carga_datos(parquet_idx=2)

    # Entrenamiento estándar (una partición)
    resultados = ejecuta_red_riesgo(
        df,
        lookback=LOOKBACK,
        horizon=HORIZON,
        train_number=0.7,
        val_number=0.85,
        neurons1=NEU1,
        neurons2=NEU2,
        learning_rate=LR,
        batch_size=BS,
        dropout=DR,
        verbose_fit=1,
        use_sum_squares=True,
        log_target=True,          # entrenar en log-RV
        include_garch=True,
        include_egarch=True,
        include_ewma=True,
        rolling_garch=False       # pon True para re-ajustes rolling de GARCH
    )
    print("\nResumen (split único):", resultados)

    # Evaluación walk-forward (opcional; costosa)
    # wf = walk_forward_eval(
    #     df, lookback=LOOKBACK, horizon=HORIZON, splits=4,
    #     train_ratio=0.6, val_ratio=0.8,
    #     neurons1=NEU1, neurons2=NEU2, learning_rate=LR, batch_size=BS, dropout=DR,
    #     use_sum_squares=True, log_target=True,
    #     include_garch=True, include_egarch=True, include_ewma=True,
    #     rolling_garch=False
    # )
    # print("\nResumen walk-forward:", wf)
