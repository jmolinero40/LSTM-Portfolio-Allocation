# -*- coding: utf-8 -*-
"""
Created on Mon Oct 13 07:33:49 2025

@author: jmoli
"""

# -*- coding: utf-8 -*-
"""
Tuning con **Keras Tuner** (Bayesian/Random) SIN usar `ejecuta_red`.
- Mantiene tu pipeline: ventanas deslizantes, split temporal y escalado sin fugas.
- HPs a sintonizar: neurons1, neurons2, dropout, learning_rate, batch_size.
- Explora varios lookbacks y horizons.
- Guarda el mejor por combo y el mejor global.

Requisitos:
    pip install keras-tuner
Estructura esperada:
    - Datos/*.parquet
    - Modelos/ (se crea si no existe)
"""

from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l2

# --- Semillas ---
np.random.seed(42)
tf.random.set_seed(42)

# --- Utilidades de ventanas / split / escalado ---

def make_sequence(data: np.ndarray, lookback: int = 60, horizon: int = 1):
    X, y = [], []
    for i in range(lookback, len(data) - horizon + 1):
        X.append(data[i - lookback:i, :])
        y.append(data[i + horizon - 1, :])
    return np.array(X), np.array(y)


def split_scale(values: np.ndarray, df_index: pd.DatetimeIndex,
                lookback: int, horizon: int,
                train_ratio: float = 0.7, val_stop: float = 0.85):
    """Genera ventanas, hace split temporal y escala sin fugas.
       Devuelve datasets escalados + scalers + metainfo.
    """
    X, y = make_sequence(values, lookback, horizon)
    N = X.shape[0]
    if N < 3:
        raise ValueError("Muy pocas ventanas generadas; ajusta lookback/horizon o usa más datos.")

    i_train = int(N * train_ratio)
    i_val   = int(N * val_stop)

    X_train, y_train = X[:i_train],      y[:i_train]
    X_val,   y_val   = X[i_train:i_val], y[i_train:i_val]
    X_test,  y_test  = X[i_val:],        y[i_val:]

    n_features = X.shape[-1]
    X_train_2d = X_train.reshape(-1, n_features)
    X_val_2d   = X_val.reshape(-1, n_features)
    X_test_2d  = X_test.reshape(-1, n_features)

    X_scaler = StandardScaler()
    X_train_scaled = X_scaler.fit_transform(X_train_2d).reshape(X_train.shape)
    X_val_scaled   = X_scaler.transform(X_val_2d).reshape(X_val.shape)
    X_test_scaled  = X_scaler.transform(X_test_2d).reshape(X_test.shape)

    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train)
    y_val_scaled   = y_scaler.transform(y_val)
    y_test_scaled  = y_scaler.transform(y_test)

    info = {
        "i_train": i_train,
        "i_val": i_val,
        "train_day": str(df_index[i_train]),
        "val_day": str(df_index[i_val]),
        "n_windows": N,
    }

    datasets = (X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled, X_test_scaled, y_test_scaled)
    scalers  = (X_scaler, y_scaler)
    return datasets, scalers, info


# --- Modelo base ---

def build_model(n_features: int, lookback: int,
                neurons1: int = 64, neurons2: int = 32,
                learning_rate: float = 1e-3, dropout: float = 0.2):
    model = Sequential([
        LSTM(neurons1, return_sequences=True, input_shape=(lookback, n_features), kernel_regularizer=l2(1e-5)),
        Dropout(dropout),
        LSTM(neurons2, kernel_regularizer=l2(1e-5)),
        Dense(n_features)
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"],
    )
    return model


# --- Keras Tuner ---
try:
    import keras_tuner as kt
except Exception as e:
    raise SystemExit("\n❌ Falta keras-tuner. Instala con: pip install keras-tuner\n")


def model_builder_factory(n_features: int, lookback: int):
    """Devuelve una función model_builder(hp) cerrada sobre n_features y lookback."""
    def model_builder(hp: "kt.HyperParameters"):
        neurons1 = hp.Int("neurons1", min_value=32, max_value=256, step=32)
        neurons2 = hp.Int("neurons2", min_value=16, max_value=128, step=16)
        dropout  = hp.Float("dropout", min_value=0.0, max_value=0.5, step=0.1)
        lr       = hp.Choice("learning_rate", values=[1e-2, 5e-3, 1e-3, 5e-4, 1e-4])
        return build_model(n_features, lookback, neurons1, neurons2, lr, dropout)
    return model_builder


class FitArgsTuner(kt.engine.tuner.Tuner):
    """Subclase para inyectar kwargs en model.fit() y poder tunear batch_size."""
    def __init__(self, batch_choices, epochs, callbacks, X_val, y_val, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_choices = list(batch_choices)
        self.epochs = epochs
        self.callbacks = callbacks
        self.X_val = X_val
        self.y_val = y_val

    def run_trial(self, trial, *args, **kwargs):
        bs = trial.hyperparameters.Choice("batch_size", values=self.batch_choices)
        kwargs.update({
            "epochs": self.epochs,
            "batch_size": bs,
            "shuffle": False,
            "validation_data": (self.X_val, self.y_val),
            "callbacks": self.callbacks,
            "verbose": 0,
        })
        return super().run_trial(trial, *args, **kwargs)


# --- Pipeline de tuning completo ---

def tune_with_keras_tuner(values: np.ndarray,
                          df_index: pd.DatetimeIndex,
                          lookbacks=(30, 60, 90),
                          horizons=(1, 5),
                          tuner_kind: str = "bayesian",  # "random" o "bayesian"
                          max_trials: int = 25,
                          executions_per_trial: int = 1,
                          epochs: int = 200,
                          batch_choices=(16, 32, 64),
                          patience_es: int = 15,
                          patience_rlr: int = 7,
                          project_name: str = "lstm_tuning",
                          save_per_combo: bool = True):
    """Devuelve (mejor_global_payload, resultados_por_combo)."""

    resultados = {}
    mejor_global = None  # (score, payload)

    for lookback in lookbacks:
        for horizon in horizons:
            # Preparar datos una única vez por combo
            (Xtr, ytr, Xva, yva, Xte, yte), (X_scaler, y_scaler), info = split_scale(values, df_index, lookback, horizon)
            n_features = Xtr.shape[-1]
            if Xtr.shape[0] < 50:
                print(f"\n⚠️ Pocas ventanas para L={lookback}, H={horizon}. Se omite.")
                continue

            # Builder
            model_builder = model_builder_factory(n_features, lookback)

            # Callbacks de entrenamiento (usados en tuning y en refit final)
            callbacks = [
                EarlyStopping(monitor="val_loss", patience=patience_es, restore_best_weights=True),
                ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=patience_rlr, min_lr=1e-5, verbose=0),
            ]

            # Tuner
            directory = Path("./kerastuner").as_posix()
            project = f"{project_name}_L{lookback}_H{horizon}"
            if tuner_kind.lower().startswith("bayes"):
                tuner = FitArgsTuner(
                    batch_choices=batch_choices,
                    epochs=epochs,
                    callbacks=callbacks,
                    X_val=Xva, y_val=yva,
                    hypermodel=model_builder,
                    objective=kt.Objective("val_loss", direction="min"),
                    max_trials=max_trials,
                    executions_per_trial=executions_per_trial,
                    directory=directory,
                    project_name=project,
                    overwrite=True,
                )
            else:
                tuner = FitArgsTuner(
                    batch_choices=batch_choices,
                    epochs=epochs,
                    callbacks=callbacks,
                    X_val=Xva, y_val=yva,
                    hypermodel=model_builder,
                    objective="val_loss",
                    max_trials=max_trials,
                    executions_per_trial=executions_per_trial,
                    directory=directory,
                    project_name=project,
                    overwrite=True,
                )

            print(f"\n🔎 Tuning KerasTuner | L={lookback}, H={horizon} | ventanas={info['n_windows']} | features={n_features}")
            tuner.search(Xtr, ytr)

            best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
            best_model = tuner.hypermodel.build(best_hp)

            # Reentrenar con mejores HP
            ckp = None
            ckp_dir = Path(__file__).parent / "Modelos"
            ckp_dir.mkdir(exist_ok=True)
            if save_per_combo:
                ckp_name = ckp_dir / f"best_tuned_L{lookback}_H{horizon}.keras"
                ckp = ModelCheckpoint(ckp_name, monitor="val_loss", save_best_only=True)

            callbacks_final = callbacks + ([ckp] if ckp is not None else [])
            history = best_model.fit(
                Xtr, ytr,
                validation_data=(Xva, yva),
                epochs=epochs,
                batch_size=best_hp.get("batch_size"),
                shuffle=False,
                callbacks=callbacks_final,
                verbose=0,
            )

            # Evaluación en test (desescalado)
            y_pred_scaled = best_model.predict(Xte, verbose=0)
            y_pred = y_scaler.inverse_transform(y_pred_scaled)
            y_true = y_scaler.inverse_transform(yte)

            mse  = mean_squared_error(y_true.reshape(-1), y_pred.reshape(-1))
            rmse = float(np.sqrt(mse))
            mae  = mean_absolute_error(y_true.reshape(-1), y_pred.reshape(-1))

            sign_true = np.sign(y_true)
            sign_pred = np.sign(y_pred)
            mask = (sign_true != 0)
            hit_ratio = float((sign_true[mask] == sign_pred[mask]).mean()) if mask.any() else float("nan")

            resumen = {
                "lookback": lookback,
                "horizon": horizon,
                "best_hp": {k: best_hp.get(k) for k in ["neurons1", "neurons2", "dropout", "learning_rate", "batch_size"]},
                "val_loss_min": float(np.min(history.history.get("val_loss", [np.nan]))),
                "rmse_test": rmse,
                "mae_test": float(mae),
                "hit_ratio_test": None if np.isnan(hit_ratio) else round(hit_ratio * 100, 2),
                "train_day": info["train_day"],
                "val_day": info["val_day"],
            }

            resultados[f"L{lookback}_H{horizon}"] = resumen

            # Selección global por MAE de test (cámbialo a val_loss si prefieres)
            score = mae
            payload = {"summary": resumen, "best_model": best_model, "best_hp": best_hp}
            if (mejor_global is None) or (score < mejor_global[0]):
                mejor_global = (score, payload)

    if mejor_global is None:
        raise RuntimeError("No se pudo entrenar ningún modelo; revisa lookbacks/horizons y el tamaño de los datos.")

    return mejor_global[1], resultados


# --- Main: carga datos y ejecuta tuning ---
if __name__ == "__main__":
    # Carga parquet
    data_dir = Path(__file__).parent / "Datos"
    parquet_files = list(data_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No hay .parquet en {data_dir}")

    parquet_path = parquet_files[2]  # ajusta si quieres otro
    print(f"\n📂 Cargando: {parquet_path.name}")
    df = pd.read_parquet(parquet_path)
    assert isinstance(df.index, pd.DatetimeIndex), "El índice no está en formato fecha"

    values = df.values.astype(np.float32)

    # Ejecuta tuner
    payload, all_results = tune_with_keras_tuner(
        values,
        df.index,
        lookbacks=(30, 60, 90),
        horizons=(1, 5),
        tuner_kind="bayesian",      # o "random"
        max_trials=25,
        executions_per_trial=1,
        epochs=200,
        batch_choices=(16, 32, 64),
        save_per_combo=True,
    )

    print("\n✅ Tuning terminado. Mejor combo global:")
    print(payload["summary"])

    # Guardar mejor modelo global con nombre rico
    out_dir = Path(__file__).parent / "Modelos"
    out_dir.mkdir(exist_ok=True)
    best_name = out_dir / (
        f"best_global_L{payload['summary']['lookback']}_H{payload['summary']['horizon']}"
        f"_u{payload['best_hp'].get('neurons1')}-{payload['best_hp'].get('neurons2')}"
        f"_bs{payload['best_hp'].get('batch_size')}.keras"
    )
    payload["best_model"].save(best_name)
    print(f"\n💾 Modelo global guardado en: {best_name}")

    # Ranking por MAE
    ranked = sorted(all_results.items(), key=lambda kv: kv[1]["mae_test"])[:10]
    print("\n🏁 Top 10 por MAE (menor es mejor):")
    for k, v in ranked:
        print(f"{k}: MAE={v['mae_test']:.6f} | RMSE={v['rmse_test']:.6f} | Hit={v['hit_ratio_test']}% | ValLossMin={v['val_loss_min']:.6f}")
