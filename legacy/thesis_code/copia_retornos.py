# -*- coding: utf-8 -*-
"""
Created on Wed Oct 29 21:03:56 2025

@author: jmoli
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Oct  5 19:34:44 2025

@author: Javier Molinero Araguas

Este código aborda la predicción de los retornos de los activos mediante
LSTM. 

Primero: Obtenemos los datos a partir del primer código y hacemos un split 
train-test para el modelo.

Segundo: Establecemos las bases en las que se desarrolla la red y aplicamos. 
Guardamos estos datos para tuning de hiperparámetros.

Tercero: Optimización de las variables del modelo (hiperparámetros)
"""

# Librerías
from pathlib import Path
import pandas as pd, numpy as np

# Escalado
import tensorflow as tf
from sklearn.preprocessing import RobustScaler
from joblib import dump

# Redes
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l2

# Métricas
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Helpers externos
from Descarga import recoge_datos


# Semillas
np.random.seed(42)
tf.random.set_seed(42)

# ---------------- Carga de datos ----------------
df=recoge_datos([])

"""
data_dir = Path(__file__).parent / "Datos"
parquet_files = list(data_dir.glob("*.parquet"))
if not parquet_files:
    raise FileNotFoundError(f"\nNo se encontró ningún archivo .parquet en {data_dir}")

parquet_path = parquet_files[2]
print(f"\n📂 Cargando archivo: {parquet_path.name}")
df = pd.read_parquet(parquet_path)
print("\n✅ Datos cargados correctamente")
print(df.head())

assert isinstance(df.index, pd.DatetimeIndex), "El índice no está en formato fecha"
"""
values = df.values.astype(np.float32)  # (T, N)
n_features = values.shape[1]
print("\n✅ Último procesado de datos realizado")


# ---------------- Baseline (EMA) ----------------
def ema_baseline(values, span=20):
    """
    Baseline suave por activo: EMA (span días) de los retornos.
    Devuelve un ndarray (T, N) alineado con 'values'.
    """
    dff = pd.DataFrame(values)
    base = dff.ewm(span=span, adjust=False, min_periods=span).mean().to_numpy()
    return np.nan_to_num(base, nan=0.0)


# ---------------- Ventanas ----------------
def make_sequence(values, lookback, horizon, baseline=None):
    """
    Genera X (ventanas) e y (targets). Si baseline no es None,
    y se convierte en residual: y_raw - baseline[target_idx].
    """
    X, y = [], []
    for i in range(lookback, len(values) - horizon + 1):
        X.append(values[i - lookback:i, :])
        target_idx = i + horizon - 1
        y_raw = values[target_idx, :]
        if baseline is not None:
            y_raw = y_raw - baseline[target_idx, :]
        y.append(y_raw)
    return np.array(X), np.array(y)


# ---------------- Modelo ----------------
def build_model(n_features, lookback, neurons1=64, neurons2=32, learning_rate=1e-3, dropout=0.2):
    n_features = int(n_features)
    lookback   = int(lookback)
    units1     = int(neurons1)
    units2     = int(neurons2)
    lr         = float(learning_rate)
    dr         = float(dropout)

    assert units1 > 0 and units2 > 0, "neurons deben ser enteros positivos"
    assert 0.0 <= dr < 1.0, "dropout debe estar en [0,1)"
    assert lookback >= 1 and n_features >= 1, "lookback/n_features inválidos"

    model = Sequential([
        tf.keras.Input(shape=(lookback, n_features)),
        LSTM(units1, return_sequences=True, kernel_regularizer=l2(1e-5)),
        Dropout(dr),
        LSTM(units2, kernel_regularizer=l2(1e-5)),
        Dense(n_features),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=tf.keras.losses.Huber(delta=1.0),
        metrics=['mae']
    )
    return model


# ---------------- Split + escalado ----------------
def _split_yX_scaled(values, lookback, horizon, train_ratio=0.7, val_stop=0.85, df_index=None):
    """
    Genera ventanas (residuales con baseline EMA), divide y escala sin fugas.
    Devuelve (datasets, scalers, info).
    """
    base = ema_baseline(values, span=20)
    X, y = make_sequence(values, lookback, horizon, baseline=base)  # y = residuales
    N = X.shape[0]                      # nº total de ventanas
    i_train = int(N * train_ratio)      # corte train (sobre ventanas)
    i_val   = int(N * val_stop)         # corte val   (sobre ventanas)
    i_test  = N                         # fin exclusivo (sobre ventanas)

    # Splits (sobre ventanas)
    X_train, y_train = X[:i_train],          y[:i_train]
    X_val,   y_val   = X[i_train:i_val],     y[i_train:i_val]
    X_test,  y_test  = X[i_val:i_test],      y[i_val:i_test]

    # Escalado de X (3D -> 2D -> 3D)
    nf = X.shape[-1]
    X_train_2d = X_train.reshape(-1, nf)
    X_val_2d   = X_val.reshape(-1, nf)
    X_test_2d  = X_test.reshape(-1, nf)

    X_scaler = RobustScaler()
    X_train_scaled = X_scaler.fit_transform(X_train_2d).reshape(X_train.shape)
    X_val_scaled   = X_scaler.transform(X_val_2d).reshape(X_val.shape)
    X_test_scaled  = X_scaler.transform(X_test_2d).reshape(X_test.shape)

    # Escalado de y (solo escala, sin centrar)
    y_scaler = RobustScaler(with_centering=False)
    y_train_scaled = y_scaler.fit_transform(y_train)
    y_val_scaled   = y_scaler.transform(y_val)
    y_test_scaled  = y_scaler.transform(y_test)

    # Mapeo de índices de ventana -> índice en 'values' del target
    # target_idx(j) = lookback + j + (horizon - 1)
    train_day_idx = lookback + i_train + (horizon - 1)
    val_day_idx   = lookback + i_val   + (horizon - 1)

    # Tramo de TEST en el eje de 'values' para la y (start inclusive, end exclusive)
    start_values_test = lookback + i_val  + (horizon - 1)
    end_values_test   = lookback + i_test + (horizon - 1)

    info = {
        "i_train": i_train,
        "i_val": i_val,
        "i_test": i_test,
        "train_day": str(df_index[train_day_idx]) if df_index is not None else None,
        "val_day":   str(df_index[val_day_idx])   if df_index is not None else None,
        "start_values_test": start_values_test,
        "end_values_test":   end_values_test
    }

    datasets = (X_train_scaled, y_train_scaled,
                X_val_scaled,   y_val_scaled,
                X_test_scaled,  y_test_scaled)
    scalers = (X_scaler, y_scaler)
    return datasets, scalers, info


# ---------------- Ejecución ----------------
def ejecuta_red(values, lookback=60, horizon=1, train_number=0.7, test_number=0.85,
                neurons1=64, neurons2=32, learning_rate=1e-3, batch_size=32, 
                dropout=0.2, verbose_fit=1):

    # Baseline global (la misma que se resta al crear y residuales)
    base = ema_baseline(values, span=20)

    (X_train_scaled, y_train_scaled,
     X_val_scaled,   y_val_scaled,
     X_test_scaled,  y_test_scaled), (X_scaler, y_scaler), info = _split_yX_scaled(
        values, lookback, horizon, train_ratio=train_number, val_stop=test_number, df_index=df.index
    )

    ckp_path = Path(__file__).parent / "Modelos"
    ckp_path.mkdir(exist_ok=True)
    
    dump(X_scaler, ckp_path / f"X_scaler_L{lookback}.pkl")
    dump(y_scaler, ckp_path / f"y_scaler_L{lookback}.pkl")
    print("✅ Scalers guardados:",
          ckp_path / f"X_scaler_L{lookback}.pkl",
          ckp_path / f"y_scaler_L{lookback}.pkl")
         
    print("\n ✅​ El split y escalado de ventanas se ha completado")
    if info is not None:
        print(f"\nEl conjunto de train llega hasta el día {info['train_day']}")
        print(f"y el conjunto de validación llega hasta el día {info['val_day']}.\n")

    # Modelo
    n_features = X_train_scaled.shape[-1]
    model = build_model(n_features, lookback, neurons1, neurons2, learning_rate, dropout)
    print("\nEl modelo se ha definido correctamente con las siguientes características:\n")
    model.summary()

    # Callbacks
    print("\nEmpieza el entrenamiento del modelo:\n")
    es  = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-5, verbose=1)
    ckp_path = Path(__file__).parent / "Modelos"
    ckp_path.mkdir(exist_ok=True)
    ckp_name = ckp_path / f"best_lstm_L{lookback}_h{horizon}_u{neurons1}-{neurons2}_bs{batch_size}_assets{n_features}.keras"
    ckp = ModelCheckpoint(ckp_name, monitor='val_loss', save_best_only=True)

    tickers = df.columns.to_list()
    print("X_train:", X_train_scaled.shape, "y_train:", y_train_scaled.shape)
    print("Orden columnas:", tickers[:5])

    # Fit
    hist = model.fit(
        X_train_scaled, y_train_scaled,
        validation_data=(X_val_scaled, y_val_scaled),
        epochs=200,
        batch_size=batch_size,
        shuffle=False,
        callbacks=[es, rlr, ckp],
        verbose=verbose_fit
    )
    print("\n✅​ El modelo ha terminado de entrenarse")

    val_hist = hist.history.get('val_loss', [])
    val_loss_min = float(np.min(val_hist)) if (val_hist is not None and len(val_hist) > 0) else float('inf')

    # -------- Evaluación --------
    print("\nSe hacen las predicciones con la red")
    y_pred_scaled = model.predict(X_test_scaled, verbose=0)
    print("y_pred_scaled mean/std (test):", y_pred_scaled.mean(), y_pred_scaled.std())

    # Residuales a escala original
    y_pred_res = y_scaler.inverse_transform(y_pred_scaled)   # residuales predichos
    y_true_res = y_scaler.inverse_transform(y_test_scaled)   # residuales reales

    # Baseline del tramo TEST (alineado al eje 'values')
    start_v = info["start_values_test"]
    end_v   = info["end_values_test"]
    base_test = base[start_v:end_v]                          # (T_test, N)

    # Reconstrucción a retornos originales
    y_pred = y_pred_res + base_test
    y_true = y_true_res + base_test

    # -------- Diagnósticos --------
    y_train = y_scaler.inverse_transform(y_train_scaled)     # train original (porque with_centering=False)
    med_train = np.median(y_train, axis=0)
    med_pred  = np.median(y_pred,  axis=0)
    print("sign(median_train):", np.sign(med_train))
    print("sign(median_pred) :", np.sign(med_pred))

    var_pred = np.var(y_pred, axis=0)
    var_true = np.var(y_true, axis=0)
    print("ratio var_pred/var_true:", np.round(var_pred / (var_true + 1e-12), 3))

    pct_pos  = (y_pred > 0).mean(axis=0)
    hit_sign = (np.sign(y_pred) == np.sign(y_true)).mean(axis=0)
    print("pct_pred_pos:", np.round(pct_pos, 2))
    print("hit_sign    :", np.round(hit_sign, 2))

    baseline = np.tile(med_train, (y_true.shape[0], 1))
    mse_model = np.mean((y_pred - y_true)**2, axis=0)
    mse_base  = np.mean((baseline - y_true)**2, axis=0)
    print("MSE model:", np.round(mse_model, 6))
    print("MSE base :", np.round(mse_base,  6))
    print("¿Mejoras al baseline? (model<base):", mse_model < mse_base)

    # Métricas agregadas
    mse  = mean_squared_error(y_true.reshape(-1), y_pred.reshape(-1))
    rmse = float(np.sqrt(mse))
    mae  = mean_absolute_error(y_true.reshape(-1), y_pred.reshape(-1))

    sign_true = np.sign(y_true)
    sign_pred = np.sign(y_pred)
    mask = (sign_true != 0)
    hit_ratio = float((sign_true[mask] == sign_pred[mask]).mean())
    hit_ratio_pct = round(hit_ratio * 100, 2)

    print("\nLa métricas del modelo son de:")
    print(f"\nRMSE: {rmse:.6e} | MAE: {mae:.6e} | Hit ratio: {hit_ratio_pct}%")

    per_asset_mae = {t: round(mean_absolute_error(y_true[:, i], y_pred[:, i]), 3)
                     for i, t in enumerate(tickers)}
    print("\nMAE por activo:", per_asset_mae)

    return {
        "lookback": lookback,
        "horizon": horizon,
        "train_number": train_number,
        "test_number": test_number,
        "neurons1": neurons1,
        "neurons2": neurons2,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "dropout": dropout,
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "hit_ratio": hit_ratio_pct,
        "val_loss_min": val_loss_min
    }


if __name__ == "__main__":
    ejecuta_red(values, lookback=60, horizon=1)
