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

#Librerías usuales de datos
from pathlib import Path
import pandas as pd, numpy as np

#Escalado
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler

#Librerías de redes
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l2

#Métricas
from sklearn.metrics import mean_squared_error, mean_absolute_error

#Funciones auxiliares (helpers) externas
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

"""
Establecemos varios de los hiperparámetros que influyen en el aprendizaje del
modelo. Estos son:
    
    LOOKBACK : ¿Cuántos días atrás deberías estar mirando para predecir ahora?
    HORIZON: ¿Cuántos días hacia adelante quiero que me predigas?
    
    -> Construcción de las ventanas deslizantes
    
    %TRAIN: ¿Qué porcentaje de datos se usa para entrenar? Habitualmente 70-80
    %VAL: ¿Qué porcentaje validará los hiperparámetros del modelo?
    %TEST: ¿Qué porcentaje de datos pondrá a prueba al modelo?
    
"""



#Función auxiliar make_sequence para las  ventanas al final
#!!!POR AHORA TIENE QUE IR AQUÍ
def make_sequence(data, lookback=60, horizon=1):
    
    """
    Esta función recibe un conjunto de datos en formato numpy.ndarray, la 
    ventana temporal pasada (lookback) y el horizonte de predicción(horizon).
    
    Devuelve dos conjuntos en formato np.array:
        
        X -> Lookback (matriz de datos con la ventana pasada)
        y -> Horizon (vector/matriz con la predicción)
        
    """
    
    X, y = [], [] #Aquí almacenamos las ventanas
   
    #REC: Nº de ventanas = datos-L-H+1
    for i in range(lookback, len(data) - horizon + 1):
        
        X.append(data[i - lookback:i, :])
        y.append(data[i + horizon - 1, :])  # vector de todos los activos
        
    return np.array(X), np.array(y)

def ejecuta_red(values, lookback=60, horizon=1, train_number=0.7, test_number=0.85,
                neurons1=64, neurons2=32, learning_rate=1e-3, batch_size=32, 
                dropout=0.2, verbose_fit=1):
    
    (X_train_scaled, y_train_scaled,
     X_val_scaled,   y_val_scaled,
     X_test_scaled,  y_test_scaled), (X_scaler, y_scaler), info = _split_yX_scaled(
        values, lookback, horizon, train_ratio=train_number, val_stop=test_number, df_index=df.index
    )

    
    print("\n ✅​ El split y escalado de ventanas se ha completado")
    
    if info is not None:
        print(f"\nEl conjunto de train llega hasta el día {info['train_day']}")
        print(f"y el conjunto de validación llega hasta el día {info['val_day']}.\n")
    
    """
    Construimos el modelo
    
    Flujo de formas (shapes) batch (argot de entrenamiento de ML) -> ventanas
    
    Entrada → (batch, lookback, n_features)
    LSTM(64, return_sequences=True) → (batch, lookback, 64)
    Dropout(0.2) → (batch, lookback, 64)
    LSTM(32) → (batch, 32)
    Dense(n_features) → (batch, n_features)
    """
    
    n_features = X_train_scaled.shape[-1]
    model=build_model(n_features, lookback, neurons1, neurons2, learning_rate, dropout)
    print("\nEl modelo se ha definido correctamente con las siguientes características:\n")
    model.summary()  
    
    """
    Callbacks y entrenamiento del modelo
    
    Callbacks: Proceso típico
    
    1.El entrenamiento progresa.
    2.Si val_loss se estanca 7 épocas → RLR reduce LR.
    3.Si aun así no mejora en 15 → ES detiene y, gracias a CKP guardamos el mejor modelo.
    """
    
    print("\nEmpieza el entrenamiento del modelo:\n")
    
    #Early stop para vigiliar que el modelo "no memorice" o sea, 
    #que la validation loss no se estanque después de 15 épocas
    es = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    
    #Otro callback para reducir la learning rate a la mitad después de 7 épocas
    rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-5, verbose=1)
    
    #Guarda en disco el mejor modelo visto (por val_loss) en el fichero best_lstm.keras
    #Hace las veces de backup
    ckp_path = Path(__file__).parent / "Modelos"  # carpeta Modelos junto al script
    ckp_path.mkdir(exist_ok=True)                 # crea la carpeta si no existe
    ckp_name = ckp_path / f"best_lstm_L{lookback}_h{horizon}_u{neurons1}-{neurons2}_bs{batch_size}.keras"
    ckp = ModelCheckpoint(ckp_name, monitor='val_loss', save_best_only=True)

    tickers = df.columns.to_list()
#8) ¿Las ventanas/datos están bien alineados y el y es multiactivo?
    print("X_train:", X_train_scaled.shape, "y_train:", y_train_scaled.shape)  # (n_samples, lookback, N) vs (n_samples, N)
    print("Orden columnas:", tickers[:5])  # asegúrate que coincide en todo el pipeline

    
    #Ajustamos el modelo
    hist = model.fit(
        X_train_scaled, y_train_scaled,
        validation_data=(X_val_scaled, y_val_scaled),
        epochs=200, #normalmente no se usará con el early stop
        batch_size=batch_size, #tamaño del minilote usado en el algoritmo de descenso
        shuffle=False,     # Fundamental en sstt (la muestra no es independiente)
        callbacks=[es, rlr, ckp],
        verbose=verbose_fit #muestra barra de progreso
    )
    
    print("\n✅​ El modelo ha terminado de entrenarse")
    
    # Aseguramos val_loss_min aunque por cualquier motivo no existiera el history
    val_hist = hist.history.get('val_loss', [])
    if val_hist is None or len(val_hist) == 0:
        val_loss_min = float('inf')   # fallback si no hay validación
    else:
        import numpy as np
        val_loss_min = float(np.min(val_hist))
    
    """
    Evaluación y métricas
    """
    print("\nSe hacen las predicciones con la red")
    #Hacemos la predicción
    y_pred_scaled=model.predict(X_test_scaled, verbose=0)
    
#1) ¿El modelo está colapsando a “casi cero” en el espacio escalado?
    print("y_pred_scaled mean/std (test):", y_pred_scaled.mean(), y_pred_scaled.std())
    


    
    #Deshacemos el escalado de datos
    y_pred=y_scaler.inverse_transform((y_pred_scaled))
    y_true = y_scaler.inverse_transform(y_test_scaled)

    
#----------------- DIAGNÓSTICOS METODOLÓGICOS-----------------------------

# 2) ¿El “cero escalado” vuelve a la mediana de train?
    y_train = y_scaler.inverse_transform(y_train_scaled)   # <- con with_centering=False esto te devuelve y_train original
    med_train = np.median(y_train, axis=0)                 # (N,)
    
    med_pred  = np.median(y_pred, axis=0)
    print("sign(median_train):", np.sign(med_train))
    print("sign(median_pred) :", np.sign(med_pred))
    

#3) ¿Las predicciones tienen muy poca varianza?
    var_pred = np.var(y_pred, axis=0)
    var_true = np.var(y_true, axis=0)
    print("ratio var_pred/var_true:", np.round(var_pred / (var_true + 1e-12), 3))


#4) ¿El modelo acierta el signo o siempre repite el mismo?
    pct_pos = (y_pred > 0).mean(axis=0)
    hit_sign = (np.sign(y_pred) == np.sign(y_true)).mean(axis=0)
    print("pct_pred_pos:", np.round(pct_pos, 2))
    print("hit_sign    :", np.round(hit_sign, 2))

#5) ¿Tu modelo mejora a un baseline “constante = mediana de train”?
    baseline = np.tile(med_train, (y_true.shape[0], 1))  # sigue siendo "constante = mediana(train)"
    mse_model = np.mean((y_pred - y_true)**2, axis=0)
    mse_base  = np.mean((baseline - y_true)**2, axis=0)
    print("MSE model:", np.round(mse_model,6))
    print("MSE base :", np.round(mse_base ,6))
    print("¿Mejoras al baseline? (model<base):", mse_model < mse_base)
    """
#6) Escalador de y: ¿está recentrando a una mediana ≠ 0?
    print("RobustScaler y — center_ (mediana train) primera fila:", y_scaler.center_[:5])
    print("RobustScaler y — scale_  (IQR) primera fila:", y_scaler.scale_[:5])
    
    
    y_pred_scaled mean/std (test): -0.029 / 0.055 → las predicciones, en el espacio 
    escalado, están muy cerca de 0 y con varianza bajísima → tendencia a colapso a constante.
    
    sign(median_train) = todo +1 → la mediana de train por activo es positiva (ligeramente).
    
    sign(median_pred) = casi todo +1 (salvo un activo) → al desescalar, ese 
    “casi 0” vuelve a la mediana de train de cada activo (que es >0), por eso 
    la predicción tiende a positiva.
    
    ratio var_pred/var_true ≈ 0.001–0.007 → las predicciones tienen muy poca 
    dispersión (casi constantes) frente a la realidad.
    
    pct_pred_pos muy alta en varios activos (0.86–0.99) → sesgo a positivo.
    
    hit_sign ≈ 0.48–0.58 → signo casi aleatorio (ligeramente >50% en algunos).
    
    MSE model vs MSE base (mediana train) → empatas o pierdes en varios activos: 
    el modelo está, en la práctica, devolviendo algo muy cercano a la mediana de train.
    
    RobustScaler y — center_ ~ 0.0003–0.0009 → justo confirma que el 0 escalado 
    mapea a la mediana de train (positiva).
    
    👉 Conclusión: la causa principal del patrón “SAN.MC siempre positivo y otros 
    negativos/positivos fijos” no es la serie en sí, sino el escalado de y con 
    centrado (RobustScaler), combinado con una red que colapsa hacia 0 en el 
    espacio escalado. Al desescalar, ese 0 se convierte en la mediana de cada 
    activo (positiva), por eso sale “siempre >0” para muchos.
    """


    #Calculamos las métricas aplanando las predicciones en un vector
    mse=(mean_squared_error(y_true.reshape(-1), y_pred.reshape(-1)))
    rmse=np.sqrt(mse)
    mae=mean_absolute_error(y_true.reshape(-1), y_pred.reshape(-1))
    
    
    sign_true = np.sign(y_true)
    sign_pred = np.sign(y_pred)
    mask = (sign_true != 0)
    hit_ratio = (sign_true[mask] == sign_pred[mask]).mean()
    hit_ratio_pct = round(hit_ratio * 100, 2)

    
    #.6e es en formato científico
    print("\nLa métricas del modelo son de:")
    print(f"\nRMSE: {rmse:.6e} | MAE: {mae:.6e} | Hit ratio: {hit_ratio_pct}%")
    
    #Podemos ofrecer métricas por cada tipo de activo
    
    tickers = df.columns.to_list()
    per_asset_mae = {t: round(mean_absolute_error(y_true[:, i], y_pred[:, i]),3) for i, t in enumerate(tickers)}
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


def build_model(n_features, lookback, neurons1=64, neurons2=32, learning_rate=1e-3, dropout=0.2):
    
    #Garantizamos los formatos de las variables
    
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
        loss=tf.keras.losses.Huber(delta=1.0))
    
    return model


def _split_yX_scaled(values, lookback, horizon, train_ratio=0.7, val_stop=0.85, df_index=None):
    
    """Helper: genera ventanas, divide y escala sin fugas. Devuelve (datasets, scalers, info)."""
    
    X, y = make_sequence(values, lookback, horizon)
    N = X.shape[0]
    i_train = int(N * train_ratio)
    i_val = int(N * val_stop)

    X_train, y_train = X[:i_train], y[:i_train]
    X_val, y_val = X[i_train:i_val], y[i_train:i_val]
    X_test, y_test = X[i_val:], y[i_val:]

    n_features = X.shape[-1]
    X_train_2d = X_train.reshape(-1, n_features)
    X_val_2d   = X_val.reshape(-1, n_features)
    X_test_2d  = X_test.reshape(-1, n_features)

    X_scaler = RobustScaler()
    X_train_scaled = X_scaler.fit_transform(X_train_2d).reshape(X_train.shape)
    X_val_scaled   = X_scaler.transform(X_val_2d).reshape(X_val.shape)
    X_test_scaled  = X_scaler.transform(X_test_2d).reshape(X_test.shape)

    y_scaler = RobustScaler(with_centering=False)
    y_train_scaled = y_scaler.fit_transform(y_train)
    y_val_scaled   = y_scaler.transform(y_val)
    y_test_scaled  = y_scaler.transform(y_test)

    info = {
        "i_train": i_train,
        "i_val": i_val,
        "train_day": str(df_index[i_train]), #aqui antes ponia df.index
        "val_day": str(df_index[i_val])
    }
    datasets = (X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled, X_test_scaled, y_test_scaled)
    scalers = (X_scaler, y_scaler)
    
    return datasets, scalers, info

if __name__ == "__main__":
    # ... entrenamiento, prints, etc.

    ejecuta_red(values, lookback=60, horizon=1)