# -*- coding: utf-8 -*-
"""
Created on Sun Oct 26 11:27:03 2025

@author: jmoli

LSTM para predecir volatilidad (desviación típica) a 1 día
- Misma estructura de carga/ventaneo que tu red de retornos
- Pérdida: NLL gaussiana (modelo predice varianza condicional por activo)
- Salida: varianza/volatilidad por activo (n_features)

Notas importantes:
- El DataFrame de entrada `df` son RETORNOS LOGARÍTMICOS DIARIOS (en decimales)
- Escalamos SOLO X. NO escalamos y (retornos verdaderos) para la NLL.
- La red produce una "logit" de varianza por activo. En la pérdida se transforma con
  softplus para garantizar varianza positiva.

Requisitos: tensorflow, numpy, pandas, scikit-learn
Opcional para métricas: scipy (no imprescindible)
"""
from pathlib import Path
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l2

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.metrics import mean_absolute_error

# Si usas tu función de descarga, impórtala (ajusta el import si cambia el path)
from Descarga import recoge_datos


#Semilla para reproducibilidad

np.random.seed(42)
tf.random.set_seed(42)

"""
Lo primero es llamar a la función que descarga y procesa todos los datos.
Posteriormente los recuperaremos y haremos el split preciso
"""

#Llamamos a la función para que descargue los datos
recoge_datos()

#Localizamos el archivo que queremos
data_dir = Path(__file__).parent / "Datos" #Vete a la carpeta Datos
parquet_files = list(data_dir.glob("*.parquet")) #Coge todos los archivos

if not parquet_files:
    
    raise FileNotFoundError(f"\nNo se encontró ningún archivo .parquet en {data_dir}")

#Por lo pronto usaremos solo las rentabilidades
parquet_path = parquet_files[2]
print(f"\n📂 Cargando archivo: {parquet_path.name}")

#Lo devolvemos a un dataframe de pandas (las librerías exigen este formato)
df = pd.read_parquet(parquet_path)

print("\n✅ Datos cargados correctamente")
print(df.head())

#Aseguramos que el índice es una fecha, si no, detenemos

assert isinstance(df.index, pd.DatetimeIndex), "El índice no está en formato fecha"

#Adaptamos el formato del dataframe a lo exigido por las bibliotecas
#Además el formato de coma flotante de 32 bits ahorra memoria
values=df.values.astype(np.float32) #(T, n_features)
n_features=values.shape[1] #Número de activos

print("\n✅ Último procesado de datos realizado")


def make_sequence(data: np.ndarray, lookback: int = 240, horizon: int = 1):
    """
    data: ndarray (T, n_features) con retornos log (decimales)
    Devuelve X: (N, lookback, n_features)
            y: (N, n_features) -> retorno en t+h-1 (verdad para pérdida NLL)
    """
    X, y = [], []
    for i in range(lookback, len(data) - horizon + 1):
        X.append(data[i - lookback:i, :])
        y.append(data[i + horizon - 1, :])
    return np.array(X), np.array(y)


def _split_X_scaled_Y_raw(values: np.ndarray, lookback: int, horizon: int,
                          train_ratio: float = 0.7, val_stop: float = 0.85, df_index=None):
    """
    Genera ventanas (X, y). Escala SOLO X. y queda en escala real (retorno en decimales).
    Devuelve datasets escalados + scaler de X + info de fechas.
    """
    X, y = make_sequence(values, lookback, horizon)
    N = X.shape[0]
    i_train = int(N * train_ratio)
    i_val = int(N * val_stop)

    X_train, y_train = X[:i_train], y[:i_train]
    X_val, y_val     = X[i_train:i_val], y[i_train:i_val]
    X_test, y_test   = X[i_val:], y[i_val:]

    n_features = X.shape[-1]
    X_train_2d = X_train.reshape(-1, n_features)
    X_val_2d   = X_val.reshape(-1, n_features)
    X_test_2d  = X_test.reshape(-1, n_features)

    X_scaler = RobustScaler(with_centering=False)
    X_train_scaled = X_scaler.fit_transform(X_train_2d).reshape(X_train.shape)
    X_val_scaled   = X_scaler.transform(X_val_2d).reshape(X_val.shape)
    X_test_scaled  = X_scaler.transform(X_test_2d).reshape(X_test.shape)

    info = {
        "i_train": i_train,
        "i_val": i_val,
        "train_day": str(df_index[i_train]) if df_index is not None else None,
        "val_day": str(df_index[i_val]) if df_index is not None else None,
    }
    datasets = (X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test)
    return datasets, X_scaler, info


# --------------------------
# Modelo: LSTM -> logits de varianza por activo
# --------------------------

def build_vol_model(n_features: int, lookback: int,
                    neurons1: int = 64, neurons2: int = 32,
                    learning_rate: float = 1e-3, dropout: float = 0.2) -> tf.keras.Model:
    """
    Salida del modelo: y_pred_raw (logits de varianza) de dimensión (batch, n_features)
    La pérdida convertirá estos logits a varianza positiva con softplus.
    """
    n_features = int(n_features)
    lookback   = int(lookback)
    units1     = int(neurons1)
    units2     = int(neurons2)
    lr         = float(learning_rate)
    dr         = float(dropout)

    model = Sequential([
        tf.keras.Input(shape=(lookback, n_features)),
        LSTM(units1, return_sequences=True, kernel_regularizer=l2(1e-5)),
        Dropout(dr),
        LSTM(units2, kernel_regularizer=l2(1e-5)),
        Dense(n_features, name="var_logits")  # sin activación: logits de varianza
    ])

    # Pérdida NLL gaussiana donde y_true son retornos reales en decimales
    # y_pred son logits de varianza -> var = softplus(logits) + eps
    eps = tf.constant(1e-12, dtype=tf.float32)

    def nll_gauss(y_true, y_pred_logits):
        var = tf.nn.softplus(y_pred_logits) + eps      # garantiza var > 0
        return 0.5*tf.math.log(var) + 0.5*tf.square(y_true)/var
        # Keras hace el mean automáticamente sobre el batch si lo especificas en metrics;
        # aquí devolvemos el tensor para que reduzca a media por batch.

    def mae_abs_r_vs_sigma(y_true, y_pred_logits):
    
       sigma = tf.sqrt(tf.nn.softplus(y_pred_logits) + eps)
       return tf.reduce_mean(tf.abs(tf.abs(y_true) - sigma))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=nll_gauss,
        metrics=[mae_abs_r_vs_sigma]
    )
    
    return model


# --------------------------
# Entrenamiento / evaluación
# --------------------------

def ejecuta_red_vol(df: pd.DataFrame,
                    lookback: int = 240, horizon: int = 1,
                    train_number: float = 0.7, test_number: float = 0.85,
                    neurons1: int = 64, neurons2: int = 32,
                    learning_rate: float = 1e-3, batch_size: int = 32,
                    dropout: float = 0.2, verbose_fit: int = 1):
    """
    df: DataFrame de retornos log (decimales), columnas = activos
    Devuelve dict con métricas y objetos útiles
    """
    assert isinstance(df.index, pd.DatetimeIndex), "El índice no está en formato fecha"

    values = df.values.astype(np.float32)
    (X_train, y_train,
     X_val,   y_val,
     X_test,  y_test), X_scaler, info = _split_X_scaled_Y_raw(
        values, lookback, horizon, train_ratio=train_number, val_stop=test_number, df_index=df.index
    )

    print("\n✅ Split listo. Sin escalado en y (NLL usa retornos reales).")
    if info is not None:
        print(f"Train hasta: {info['train_day']} | Val hasta: {info['val_day']}")

    n_features = X_train.shape[-1]
    model = build_vol_model(n_features, lookback, neurons1, neurons2, learning_rate, dropout)
    print("\nEstructura del modelo de volatilidad:\n")
    model.summary()

    # Callbacks
    es  = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-5, verbose=1)

    ckp_path = Path(__file__).parent / "Modelos"
    ckp_path.mkdir(exist_ok=True)
    ckp_name = ckp_path / f"best_lstmVOL_L{lookback}_h{horizon}_u{neurons1}-{neurons2}_bs{batch_size}.keras"
    ckp = ModelCheckpoint(ckp_name, monitor='val_loss', save_best_only=True)

    print("\n🏋️ Entrenando modelo de volatilidad (NLL gaussiana) ...\n")
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
    val_hist = hist.history.get('val_loss', [])
    val_loss_min = float(np.min(val_hist)) if len(val_hist) else float('inf')

    # Predicciones en test → var/sigma pronosticadas t+1
    y_pred_logits = model.predict(X_test, verbose=0)
    var_pred = tf.nn.softplus(y_pred_logits).numpy()  # softplus
    sigma_pred = np.sqrt(var_pred)

    # Métricas sencillas de sanity-check
    y_true = y_test  # retornos reales t+1 en decimales
    mae_abs = mean_absolute_error(np.abs(y_true).ravel(), sigma_pred.ravel())

    # QLIKE promedio por activo
    eps = 1e-12
    qlike = np.log(var_pred + eps) + (y_true**2) / (var_pred + eps)
    qlike_mean = float(np.mean(qlike))

    # MAE por activo entre |r| y sigma_pred
    tickers = df.columns.to_list()
    per_asset_mae = {t: float(mean_absolute_error(np.abs(y_true[:, i]), sigma_pred[:, i])) for i, t in enumerate(tickers)}

    print("\n📊 Métricas en test:")
    print(f"QLIKE medio: {qlike_mean:.6f} | MAE(|r|, sigma_hat): {mae_abs:.6f}")
    print("MAE por activo:", {k: round(v, 6) for k, v in per_asset_mae.items()})

    out = {
        "lookback": lookback,
        "horizon": horizon,
        "train_number": train_number,
        "test_number": test_number,
        "neurons1": neurons1,
        "neurons2": neurons2,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "dropout": dropout,
        "val_loss_min": val_loss_min,
        "QLIKE": qlike_mean,
        "MAE_abs_vs_sigma": mae_abs,
        "per_asset_mae": per_asset_mae,
        "model": model,
        "X_scaler": X_scaler,
        "info": info,
        "y_true_test": y_true,
        "sigma_pred_test": sigma_pred,
        "var_pred_test": var_pred,
    }
    return out


# --------------------------
# Script principal (opcional)
# --------------------------
if __name__ == "__main__":
    np.random.seed(42)
    tf.random.set_seed(42)

    # 1) Descarga/carga igual que en tu script
    recoge_datos()
    data_dir = Path(__file__).parent / "Datos"
    parquet_files = list(data_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No se encontró ningún archivo .parquet en {data_dir}")

    parquet_path = parquet_files[2]  # mismo criterio que tu script
    print(f"\n📂 Cargando archivo: {parquet_path.name}")
    df = pd.read_parquet(parquet_path)

    assert isinstance(df.index, pd.DatetimeIndex), "El índice no está en formato fecha"
    print("\n✅ Datos cargados correctamente: retornos log diarios (decimales)")
    print(df.head())

    # 2) Ejecuta red de volatilidad
    res = ejecuta_red_vol(
        df,
        lookback=240,  # puedes usar 60 si quieres probar rápido; 240 suele ir mejor para vol
        horizon=1,
        neurons1=64,
        neurons2=32,
        learning_rate=1e-3,
        batch_size=32,
        dropout=0.2,
        verbose_fit=1
    )

    print("\n✅ Listo. Puedes acceder a res['sigma_pred_test'] y res['var_pred_test'] para análisis.")
    

# === CALIBRACIÓN ADICIONAL (pegar al final de riesgos.py) ===
import numpy as np

# 0) Prepara inputs desde lo que ya tienes en tu script
# y_test: retornos reales de test (T,N)
# res['sigma_pred_test'] o res['var_pred_test']: predicciones del modelo
r_true = res['y_true_test']
sigma_pred = res.get('sigma_pred_test', None)
if sigma_pred is None:
    sigma_pred = np.sqrt(res['var_pred_test'])

eps = 1e-12

def ensure_2d(a):
    a = np.asarray(a)
    return a if a.ndim == 2 else a.reshape(-1, 1)

def qlike(r, sigma, eps=eps):
    var = np.square(sigma)
    return np.log(var + eps) + np.square(r) / (var + eps)

# Asegura forma (T,N)
r_true = ensure_2d(r_true)
sigma_pred = ensure_2d(sigma_pred)

# 1) Checks de calibración
ratio = (np.square(r_true)) / (np.square(sigma_pred) + eps)
mean_ratio = float(ratio.mean())
exceed_prob = float((np.abs(r_true) > sigma_pred).mean())   # ≈ 31.7% si Gauss calibrado
qlike_model = float(qlike(r_true, sigma_pred).mean())

print(f"\n🔎 Calibración (test)")
print(f"Mean(r^2/σ̂^2): {mean_ratio:.3f}")
print(f"P(|r| > σ̂):    {exceed_prob:.3%}")
print(f"QLIKE modelo:   {qlike_model:.3f}")

# 2) Baseline EWMA(0.97) para comparar
def ewma_sigma_test(r_test, lam=0.97, eps=eps):
    r = ensure_2d(r_test)
    T, N = r.shape
    var = np.zeros((T, N), dtype=float)
    var[0] = np.square(r[0]) + eps   # init
    for t in range(1, T):
        var[t] = lam * var[t-1] + (1-lam) * np.square(r[t-1])  # solo pasado
    return np.sqrt(var)

sigma_ewma = ewma_sigma_test(r_true, lam=0.97)
qlike_ewma = float(qlike(r_true, sigma_ewma).mean())
print(f"QLIKE EWMA(0.97): {qlike_ewma:.3f}")
# === FIN CALIBRACIÓN ===
