# -*- coding: utf-8 -*-
"""
Corrección de 'riesgos.py' manteniendo el estilo original en la medida de lo posible.
- Se corrigen errores de tipos/índices y acoplamientos globales.
- Se asegura la ausencia de NaN en features.
- Se activan los toggles include_garch/egarch/ewma.
- Se desacopla la dimensión de salida del modelo del global n_features.
- Se respetan nombres y comentarios en español, y la estructura de funciones.
"""

import numpy as np
import pandas as pd
from pathlib import Path

import tensorflow as tf
from sklearn.preprocessing import RobustScaler

# Si usas TensorFlow/Keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# Si usas 'arch' para GARCH/EGARCH
try:
    from arch import arch_model
    _HAS_ARCH = True
    
except Exception:
    _HAS_ARCH = False

# Helpers externos (datos)
from Descarga import recoge_datos  # usa tu mismo pipeline de descarga/guardado  :contentReference[oaicite:1]{index=1}

# Semilla para reproducibilidad (igual que en retornos.py)
np.random.seed(42)
tf.random.set_seed(42)

# ================================
# 1. Carga de datos (idéntico flujo)
# ================================
recoge_datos()  # asegura parquet actualizado

data_dir = Path(__file__).parent / "Datos"
parquet_files = list(data_dir.glob("*.parquet"))
if not parquet_files:
    raise FileNotFoundError(f"\nNo se encontró ningún archivo .parquet en {data_dir}")

# En tu 'retornos.py' coges parquet_files[2] para rentabilidades; replico la lógica.
parquet_path = parquet_files[2]
print(f"\n📂 Cargando archivo: {parquet_path.name}")

df = pd.read_parquet(parquet_path)
print("\n✅ Datos cargados correctamente")
print(df.head())

assert isinstance(df.index, pd.DatetimeIndex), "El índice no está en formato fecha"

# Adaptamos a float32 por memoria, exactamente como en retornos.py  :contentReference[oaicite:2]{index=2}
values = df.values.astype(np.float32)  # (T, n_features)
n_features = values.shape[1]
print("\n✅ Último procesado de datos realizado")

# =======================================
# 1. Ventanizado
# =======================================
def make_sequence_risk(X_df: pd.DataFrame, y_ser: pd.Series, lookback=60):
    y_ser = y_ser.loc[X_df.index]

    if not isinstance(X_df, pd.DataFrame):
        X_df = pd.DataFrame(X_df, index=y_ser.index if hasattr(y_ser, 'index') else None)
    if isinstance(y_ser, pd.Series):
        y_df = y_ser.to_frame()
    else:
        y_df = pd.DataFrame(y_ser, index=X_df.index)

    if len(X_df) <= lookback:
        raise ValueError(f"Datos insuficientes para lookback={lookback}: len={len(X_df)}")

    Xv, yv, idx = [], [], []
    L = int(lookback)
    vals = X_df.values.astype(np.float32)
    tvals = y_df.values.astype(np.float32)
    dates = X_df.index

    for i in range(L, len(X_df)):
        Xi = vals[i-L:i, :]
        yi = tvals[i, :]
        # saltar si hay NaN en X o en y
        if np.isnan(Xi).any() or np.isnan(yi).any():
            continue
        Xv.append(Xi)
        yv.append(yi)
        idx.append(dates[i])

    if len(Xv) == 0:
        raise ValueError(
            "No se han podido construir ventanas: todas las muestras contienen NaN "
            "tras el ventanizado. Revisa 'horizon', 'lookback' o calidad de datos."
        )

    Xv = np.asarray(Xv, dtype=np.float32)
    yv = np.asarray(yv, dtype=np.float32)
    idx = pd.Index(idx, name='date')
    return Xv, yv, idx



# =======================================
# 2. Split + Escalado (dinámicas)
# =======================================
def _flatten_time(X):
    N, L, F = X.shape
    return X.reshape(N * L, F), (N, L, F)


def split_scale_simple(X, y, n_dyn_features, train_ratio=0.7, val_ratio=0.85):
    """
    X: np.ndarray (N, lookback, F_total)
    y: np.ndarray (N, n_out)
    n_dyn_features: nº de columnas dinámicas (p.ej., retornos)
    train_ratio, val_ratio: cortes temporales (0<train<val<1)
    Devuelve: (X_tr, y_tr, X_va, y_va, X_te, y_te), scaler
    """
    N, L, F = X.shape
    assert 0 < train_ratio < val_ratio < 1, "Ratios deben cumplir 0 < train < val < 1"
    i_tr = max(1, int(N * train_ratio))
    i_va = max(i_tr + 1, int(N * val_ratio))

    X_tr, y_tr = X[:i_tr], y[:i_tr]
    X_va, y_va = X[i_tr:i_va], y[i_tr:i_va]
    X_te, y_te = X[i_va:], y[i_va:]

    # Separamos dinámicas/estáticas
    X_tr_dyn, X_tr_stat = X_tr[:, :, :n_dyn_features], X_tr[:, :, n_dyn_features:]
    X_va_dyn, X_va_stat = X_va[:, :, :n_dyn_features], X_va[:, :, n_dyn_features:]
    X_te_dyn, X_te_stat = X_te[:, :, :n_dyn_features], X_te[:, :, n_dyn_features:]

    # Ajuste scaler en TRAIN (aplanando el eje temporal)
    scaler = RobustScaler()
    X_tr_dyn_flat, shape_tr = _flatten_time(X_tr_dyn)
    scaler.fit(X_tr_dyn_flat)

    def trans(block):
        flat, shp = _flatten_time(block)
        flat = scaler.transform(flat)
        return flat.reshape(shp)

    X_tr_dyn = trans(X_tr_dyn)
    X_va_dyn = trans(X_va_dyn)
    X_te_dyn = trans(X_te_dyn)

    # recomponer
    X_tr = np.concatenate([X_tr_dyn, X_tr_stat], axis=-1)
    X_va = np.concatenate([X_va_dyn, X_va_stat], axis=-1)
    X_te = np.concatenate([X_te_dyn, X_te_stat], axis=-1)

    return (X_tr, y_tr, X_va, y_va, X_te, y_te), scaler


# =======================================
# 3. Métricas
# =======================================
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
# 4. Target de riesgo: RV futura por activo
# =======================================
def realized_vol_future_panel(df_returns: pd.DataFrame, horizon: int = 2) -> pd.DataFrame:
    """
    RV_t(h) por columna = sqrt( sum_{k=1..h} r_{t+k}^2 ), alineada en t.
    Con h=1 => |r_{t+1}|.
    """
    out = {}
    for col in df_returns.columns:
        r = df_returns[col]
        # suma de cuadrados a futuro y alineada en t
        rv = (r.pow(2).rolling(window=horizon, min_periods=horizon).sum()
              .shift(-horizon)).pow(0.5)
        out[col] = rv
    return pd.DataFrame(out, index=df_returns.index)



# =======================================
# 5. Estimación de parámetros GARCH/EGARCH + EWMA (constantes en el tiempo)
# =======================================
def estimate_garch_params_per_asset(df_returns: pd.DataFrame, train_stop_idx: int, ewma_rho: float = 0.97):
    """
    Estima GARCH(1,1) y EGARCH(1,1,1) en el tramo de entrenamiento (0:train_stop_idx)
    para cada activo. Devuelve un DataFrame de parámetros por activo.
    """
    params = {}
    for col in df_returns.columns:
        r_train = (df_returns[col].iloc[:int(train_stop_idx)] * 100.0).dropna()
        if len(r_train) < 250 or not _HAS_ARCH:
            params[col] = {
                'garch_alpha1': np.nan, 'garch_beta1': np.nan, 'garch_omega': np.nan,
                'egarch_omega': np.nan, 'egarch_beta': np.nan, 'egarch_gamma': np.nan, 'egarch_alpha': np.nan,
                'ewma_rho': ewma_rho
            }
            continue

        try:
            garch_res = arch_model(r_train, vol='Garch', p=1, q=1, dist='normal').fit(disp='off')
            g_params = garch_res.params
            g_alpha = float(g_params.get('alpha[1]', np.nan))
            g_beta  = float(g_params.get('beta[1]',  np.nan))
            g_omega = float(g_params.get('omega',    np.nan))
        except Exception:
            g_alpha = g_beta = g_omega = np.nan

        try:
            egarch_res = arch_model(r_train, vol='EGARCH', p=1, o=1, q=1, dist='normal').fit(disp='off')
            e_params = egarch_res.params
            e_omega = float(e_params.get('omega',    np.nan))
            e_beta  = float(e_params.get('beta[1]',  np.nan))
            e_gamma = float(e_params.get('gamma[1]', np.nan))
            e_alpha = float(e_params.get('alpha[1]', np.nan))
        except Exception:
            e_omega = e_beta = e_gamma = e_alpha = np.nan

        params[col] = {
            'garch_alpha1': g_alpha,
            'garch_beta1':  g_beta,
            'garch_omega':  g_omega,
            'egarch_omega': e_omega,
            'egarch_beta':  e_beta,
            'egarch_gamma': e_gamma,
            'egarch_alpha': e_alpha,
            'ewma_rho':     ewma_rho
        }

    return pd.DataFrame(params).T  # index=tickers, cols=param names


def build_param_feature_matrix(params_df: pd.DataFrame, T: int):
    """
    A partir del DataFrame de parámetros (index=activos, cols=params), crea
    una matriz (T, n_activos*len(params)) repitiendo por cada fecha.
    Devuelve: np.ndarray de shape (T, n_features * n_params)
    """
    # Orden estable para columnas
    cols = list(params_df.columns)
    by_assets = []
    for asset in params_df.index:
        by_assets.append(params_df.loc[asset, cols].values.astype(np.float32))
    per_t = np.concatenate(by_assets, axis=0)  # (n_activos * n_params,)
    P = np.repeat(per_t.reshape(1, -1), T, axis=0)  # (T, n_activos * n_params)
    # Evita NaN en features (imputación simple a 0)
    P = np.nan_to_num(P, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return P, cols


# =======================================
# 6. Modelo
# =======================================
def build_model(n_features_total, lookback, n_outputs, neurons1=64, neurons2=32, learning_rate=1e-3, dropout=0.2):
    """
    Estructura LSTM sencilla con dos capas y salida densa del tamaño del target.
    """
    n_features_total = int(n_features_total)
    lookback   = int(lookback)
    units1     = int(neurons1)
    units2     = int(neurons2)
    lr         = float(learning_rate)
    dr         = float(dropout)

    model = Sequential([
        LSTM(units1, input_shape=(lookback, n_features_total),
             kernel_regularizer=l2(1e-5), return_sequences=True),
        Dropout(dr),
        LSTM(units2, kernel_regularizer=l2(1e-5), return_sequences=False),
        Dropout(dr),
        Dense(n_outputs)
    ])
    model.compile(optimizer=Adam(learning_rate=lr), loss='mse', metrics=['mae'])
    return model


# =======================================
# 7. Pipeline principal
# =======================================
def ejecuta_red_riesgo(values, df, lookback=22, horizon=1, train_number=0.7, test_number=0.85,
                       neurons1=10, neurons2=4, learning_rate=1e-4, batch_size=64,
                       dropout=0.3, verbose_fit=1, include_garch=True, include_egarch=True, include_ewma=True):
    """
    Entrena la LSTM de riesgo con parámetros GARCH fijos como entradas.
    - values: np.ndarray (T, n_features) con retornos (float32).
    - df: DataFrame de retornos con índice de fechas y mismas columnas.
    - horizon: h de la RV futura.
    - include_*: toggles para seleccionar qué parámetros añadir a X.
    """
    # Asegura tipos/índices coherentes
    ret_df = pd.DataFrame(values, index=df.index, columns=df.columns).astype(np.float32)

    # Target: RV futura por activo
    rv_df = realized_vol_future_panel(ret_df, horizon=horizon)

    # Alineación por intersección de fechas
    common_idx = ret_df.index.intersection(rv_df.index)
    ret_df = ret_df.loc[common_idx]
    rv_df  = rv_df.loc[common_idx]

    T = len(ret_df)
    n_assets = ret_df.shape[1]

    # Cálculo del índice de corte de entrenamiento para estimar parámetros econométricos
    train_stop_idx = max(1, int(T * float(train_number)))

    # Estimar parámetros por activo en el tramo de entrenamiento
    params_df = estimate_garch_params_per_asset(ret_df, train_stop_idx=train_stop_idx, ewma_rho=0.97)

    # Activar toggles: filtra columnas de parámetros
    keep_cols = []
    if include_garch:
        keep_cols += [c for c in params_df.columns if c.startswith('garch_')]
    if include_egarch:
        keep_cols += [c for c in params_df.columns if c.startswith('egarch_')]
    if include_ewma:
        keep_cols += [c for c in params_df.columns if c.startswith('ewma_')]
    if keep_cols:
        params_df = params_df[keep_cols]
    else:
        # si ningún bloque está activo, crea matriz vacía
        params_df = pd.DataFrame(index=params_df.index)

    # Matriz de parámetros repetidos por fecha
    P, param_cols = build_param_feature_matrix(params_df, T=T)  # (T, n_assets * n_params_sel)

    # Construir X base: [retornos dinámicos | parámetros estáticos]
    X_base = np.concatenate([ret_df.values.astype(np.float32), P.astype(np.float32)], axis=1)

    # y base: objetivo (RV futura)
    y_base = rv_df.values.astype(np.float32)

    # Envolver en DataFrames para mantener índices
    X_df = pd.DataFrame(X_base, index=ret_df.index)
    y_df = pd.DataFrame(y_base, index=rv_df.index)

    # Ventanizado
    X, y, seq_idx = make_sequence_risk(X_df, y_df, lookback=lookback)

    # Split + scale (dinámicas = primeras n_assets columnas)
    (X_tr, y_tr, X_va, y_va, X_te, y_te), scaler = split_scale_simple(
        X, y, n_dyn_features=n_assets, train_ratio=float(train_number), val_ratio=float(test_number)
    )

    # Modelo
    n_features_total = X_tr.shape[-1]
    n_outputs = y_tr.shape[1]
    model = build_model(n_features_total, lookback=lookback, n_outputs=n_outputs,
                        neurons1=neurons1, neurons2=neurons2, learning_rate=learning_rate, dropout=dropout)


    # === Callbacks ===
    es = EarlyStopping(
    monitor='val_loss',
    patience=10,          # <– menos paciencia
    min_delta=1e-5,       # <– ignora mejoras microscópicas
    restore_best_weights=True
    )
    
    rlr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,           # <– deja actuar antes a la bajada de LR
        cooldown=2,           # <– evita bajar LR en cada micro-rizo
        min_lr=1e-6,          # <– algo más bajo por si acaso
        verbose=1
    )
    
    # Carpeta para guardar el mejor modelo
    ckp_path = Path(__file__).parent / "Modelos_Riesgo"
    ckp_path.mkdir(exist_ok=True)
    
    ckp_name = ckp_path / f"best_RISK_L{lookback}_h{horizon}_u{neurons1}-{neurons2}_bs{batch_size}.keras"
    
    ckp = ModelCheckpoint(
        ckp_name,
        monitor='val_loss',
        save_best_only=True
    )

    # Entrenamiento
    history = model.fit(
    X_tr, y_tr,
    validation_data=(X_va, y_va),
    epochs=200,
    batch_size=int(batch_size),
    verbose=int(verbose_fit),
    shuffle=False,                 # <– muy importante en time series
    callbacks=[es, rlr, ckp]
    )


    # Predicciones
    yhat_tr = model.predict(X_tr, verbose=0)
    yhat_va = model.predict(X_va, verbose=0)
    yhat_te = model.predict(X_te, verbose=0)

    # Métricas
    out = {
        'train_mae': float(np.mean(np.abs(y_tr - yhat_tr))),
        'val_mae':   float(np.mean(np.abs(y_va - yhat_va))),
        'test_mae':  float(np.mean(np.abs(y_te - yhat_te))),
        'train_mse': float(np.mean((y_tr - yhat_tr)**2)),
        'val_mse':   float(np.mean((y_va - yhat_va)**2)),
        'test_mse':  float(np.mean((y_te - yhat_te)**2)),
        'train_hmae': float(hmae(y_tr, yhat_tr)),
        'val_hmae':   float(hmae(y_va, yhat_va)),
        'test_hmae':  float(hmae(y_te, yhat_te)),
        'train_hmse': float(hmse(y_tr, yhat_tr)),
        'val_hmse':   float(hmse(y_va, yhat_va)),
        'test_hmse':  float(hmse(y_te, yhat_te)),
        'n_features_total': int(n_features_total),
        'n_outputs': int(n_outputs),
        'lookback': int(lookback),
        'n_assets': int(n_assets),
        'params_columns': param_cols,
        'scaler': scaler,
        'model': model,
        'history': history.history,
        'seq_index': seq_idx
    }
    return out


# ====== LLAMADA A LA RED ======
out = ejecuta_red_riesgo(
    values,
    df,
    lookback=22,
    horizon=1,
    include_garch=True,
    include_egarch=True,
    include_ewma=True
)

print(out['test_mae'], out['test_mse'])
