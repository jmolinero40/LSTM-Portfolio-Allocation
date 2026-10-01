# -*- coding: utf-8 -*-
"""
Created on Wed Nov  5 20:43:02 2025

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

print("\nCarga de modelos:")

# --- SOLO si el modelo de riesgo usó una loss/métrica custom ---
def nll_gauss(y_true, y_pred_logits, eps=1e-12):
    var = tf.nn.softplus(y_pred_logits) + eps
    return 0.5*tf.math.log(var) + 0.5*tf.square(y_true)/var

def mae_abs_r_vs_sigma(y_true, y_pred_logits, eps=1e-12):
    sigma = tf.sqrt(tf.nn.softplus(y_pred_logits) + eps)
    return tf.reduce_mean(tf.abs(tf.abs(y_true) - sigma))

# Rutas (ajusta los nombres a los tuyos)
model_dir = Path(__file__).parent / "Modelos"
ret_path  = model_dir / "best_lstm_L60_h1_u64-32_bs32.keras"
risk_path = model_dir / "best_lstmVOL_L240_h1_u64-32_bs32.keras"

# Carga simple (sin compilar para evitar problemas de lambdas/objetos custom)
retornos_model = keras.models.load_model(ret_path, compile=False)
riesgos_model  = keras.models.load_model(
    risk_path,
    custom_objects={"nll_gauss": nll_gauss, "mae_abs_r_vs_sigma": mae_abs_r_vs_sigma},
    compile=False)

print("\nCargados los retornos:", retornos_model.input_shape, "->", retornos_model.summary())
print("\nCargados los riesgos :", riesgos_model.input_shape,  "->", riesgos_model.summary())

print("\nCarga de datos")

#!!! Revisar en profundidad el acceso a datos porque hay errores seguro
tickers=[]
# Llamamos a la función para que descargue los datos
recoge_datos()

# Localizamos el archivo que queremos
data_dir = Path(__file__).parent / "Datos"  # Vete a la carpeta Datos
parquet_files = list(data_dir.glob("*.parquet"))  # Coge todos los archivos

if not parquet_files:
    raise FileNotFoundError(f"\nNo se encontró ningún archivo .parquet en {data_dir}")

# Por lo pronto usaremos solo las rentabilidades
parquet_path = parquet_files[3]
print(f"\n📂 Cargando archivo: {parquet_path.name}")

# Lo devolvemos a un dataframe de pandas (las librerías exigen este formato)
df = pd.read_parquet(parquet_path)

print("\n✅ Datos cargados correctamente")
print(df.head())

values = df.values.astype(np.float32)  # (T, n_features)
n_features = values.shape[1]           # Número de activos

retornos_X, retornos_y = make_sequence(values, lookback=60,  horizon=1)
riesgos_X,  riesgos_y  = make_sequence(values, lookback=240, horizon=1)

import numpy as np, pandas as pd, tensorflow as tf
from pathlib import Path
from joblib import load

# ----- Configuración de la prueba -----
ret_lookback  = 60
risk_lookback = 240
horizon       = 1
span_ema      = 20         # el mismo que usaste al entrenar
t             = 1000     # índice "día t" sobre la serie base del riesgo

# Alineación entre redes (la ventana corta debe terminar el mismo día que la larga)
offset = risk_lookback - ret_lookback   # 240-60 = 180
i_risk = t
i_ret  = t + offset

# ----- Datos base -----
values = df.values.astype(np.float32)   # (T, N)
N = values.shape[1]

# Ventanas que TERMINAN en el mismo día
ret_X  = values[i_ret+ret_lookback - ret_lookback : i_ret+ret_lookback]   # (60, N)
risk_X = values[i_risk+risk_lookback - risk_lookback : i_risk+risk_lookback]  # (240, N)

# Añadimos dimensión batch para Keras
ret_X  = ret_X[None, ...]   # (1, 60,  N)
risk_X = risk_X[None, ...]  # (1, 240, N)

# ----- Cargar scalers usados en entrenamiento -----
scaler_dir = Path(__file__).parent / "Modelos"
X_scaler = load(scaler_dir / f"X_scaler_L{ret_lookback}.pkl")  # de entradas (ventanas)
y_scaler = load(scaler_dir / f"y_scaler_L{ret_lookback}.pkl")  # de residuales (with_centering=False)

# Escalar ret_X exactamente como en training (3D -> 2D -> 3D)
ret_X_scaled = X_scaler.transform(ret_X.reshape(-1, N)).reshape(1, ret_lookback, N)

# ----- Predicción de retornos (residuales escalados) -----
y_pred_scaled = retornos_model.predict(ret_X_scaled, verbose=0)      # (1, N)
y_pred_res    = y_scaler.inverse_transform(y_pred_scaled).ravel()    # (N,) residuales reales

# ----- Reconstrucción con EMA (baseline) -----
def ema_baseline(vals, span=20):
    dff = pd.DataFrame(vals)
    base = dff.ewm(span=span, adjust=False, min_periods=span).mean().to_numpy()
    return np.nan_to_num(base, nan=0.0)

base = ema_baseline(values, span=span_ema)

# Índice del target consistente con make_sequence: target_idx = lookback + j + (horizon-1)
# Aquí j = i_ret porque i_ret es el índice de ventana para la red de retornos
target_idx = ret_lookback + i_ret + (horizon - 1)

mu_pred = y_pred_res + base[target_idx, :]   # (N,) retornos esperados diarios reconstruidos

# ----- Predicción de riesgos (σ̂ diaria) -----
risk_logits = riesgos_model.predict(risk_X, verbose=0).ravel()    # (N,) logits de var
sigma_pred  = np.sqrt(tf.nn.softplus(risk_logits).numpy())        # σ̂ = sqrt(softplus(var_logit))

# ----- Resultado de la prueba -----
tickers = df.columns.to_list()
print("\nFecha objetivo:", df.index[target_idx])
print("μ̂ (primeros 5):", np.round(mu_pred[:5], 6))
print("σ̂ (primeros 5):", np.round(sigma_pred[:5], 6))

# Opcional: ranking por señal Sharpe-like s = μ̂ / (σ̂ + 1e-12)
s = mu_pred / (sigma_pred + 1e-12)
order = np.argsort(-s)
print("\nTop-5 por señal:")
for idx in order[:5]:
    print(f"{tickers[idx]:<8}  μ̂={mu_pred[idx]:+.6f}  σ̂={sigma_pred[idx]:.6f}  s={s[idx]:+.3f}")