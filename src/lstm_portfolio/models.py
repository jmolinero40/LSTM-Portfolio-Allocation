"""The two LSTM forecasters.

Both networks have the thesis architecture (Appendix B, tables 5 and 9)::

    Input(lookback, N) -> LSTM(64, return_sequences) -> Dropout(0.2)
                       -> LSTM(32) -> Dense(N)

with L2(1e-5) on the LSTM kernels, Adam(1e-3), early stopping on validation
loss and ReduceLROnPlateau. ``tests/test_thesis_equivalence.py`` checks that
the layers and parameter counts match the saved thesis models.

**Return network.** Input: 60 days of (winsorised) log returns, RobustScaler.
Target: ``r_t - EMA_{t-1}``, scaled by its interquartile range without
centring (the thesis fix for predictions collapsing to the training median).
Loss: Huber(1). Forecast: ``mu_t = inverse_scale(output) + EMA_{t-1}``.

**Risk network.** Input: 240 days of log returns, RobustScaler without
centring. Output: one variance logit per asset; ``var = softplus(logit)``.
Loss: Gaussian negative log-likelihood of the realised return. Forecast:
``sigma_t = sqrt(softplus(output))``.

The input scaler of the risk network is part of the forecaster and is applied
at prediction time. The thesis fitted it during training but never saved it,
and its backtest fed raw returns to a network trained on scaled ones; the
network then produced an almost constant sigma per asset (see docs/AUDIT.md).
Keeping scaler and network in one object makes that mismatch impossible.
"""

from __future__ import annotations

import contextlib
import os
import time
from dataclasses import dataclass, field

import numpy as np
from sklearn.preprocessing import RobustScaler

from lstm_portfolio.config import NetConfig
from lstm_portfolio.features import make_windows

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

__all__ = [
    "set_global_seed",
    "gaussian_nll",
    "build_lstm",
    "ReturnForecaster",
    "RiskForecaster",
]

NLL_EPS = 1e-12


def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy and TensorFlow, and request deterministic kernels.

    On CPU this makes a training run bit-for-bit repeatable on the same machine
    and library versions. Across machines or versions results still differ in
    the last digits, which is why every result is reported over several seeds.
    """
    import keras
    import tensorflow as tf

    keras.utils.set_random_seed(int(seed))
    with contextlib.suppress(Exception):  # older TF versions lack the switch
        tf.config.experimental.enable_op_determinism()


def gaussian_nll(y_true, logits):
    """0.5*log(v) + 0.5*r^2/v with v = softplus(logit) + eps (thesis eq. in 3.3.1)."""
    import tensorflow as tf

    var = tf.nn.softplus(logits) + NLL_EPS
    return tf.reduce_mean(0.5 * tf.math.log(var) + 0.5 * tf.square(y_true) / var, axis=-1)


def build_lstm(n_assets: int, cfg: NetConfig, loss, name: str):
    import keras
    from keras import layers, regularizers

    u1, u2 = cfg.units
    model = keras.Sequential(
        [
            keras.Input(shape=(cfg.lookback, n_assets)),
            layers.LSTM(u1, return_sequences=True, kernel_regularizer=regularizers.l2(cfg.l2)),
            layers.Dropout(cfg.dropout),
            layers.LSTM(u2, kernel_regularizer=regularizers.l2(cfg.l2)),
            layers.Dense(n_assets),
        ],
        name=name,
    )
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=cfg.learning_rate), loss=loss)
    return model


def _fit(model, x_tr, y_tr, x_va, y_va, cfg: NetConfig, verbose: int) -> dict:
    import keras

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=cfg.early_stopping_patience, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=cfg.reduce_lr_factor,
            patience=cfg.reduce_lr_patience,
            min_lr=cfg.min_lr,
        ),  # fmt: skip
    ]
    t0 = time.time()
    hist = model.fit(
        x_tr, y_tr, validation_data=(x_va, y_va), epochs=cfg.epochs,
        batch_size=cfg.batch_size, shuffle=cfg.shuffle, callbacks=callbacks, verbose=verbose,
    )  # fmt: skip
    val = np.asarray(hist.history["val_loss"], dtype=float)
    return {
        "epochs_run": int(len(val)),
        "best_epoch": int(np.argmin(val)) + 1,
        "best_val_loss": float(val.min()),
        "seconds": round(time.time() - t0, 1),
        "n_train": int(len(x_tr)),
        "n_val": int(len(x_va)),
    }


def _scale3d(scaler: RobustScaler, x: np.ndarray) -> np.ndarray:
    return scaler.transform(x.reshape(-1, x.shape[-1])).reshape(x.shape).astype(np.float32)


@dataclass
class ReturnForecaster:
    """Return LSTM with its scalers. Predicts mu_t (daily log return)."""

    cfg: NetConfig
    model: object = None
    x_scaler: RobustScaler = field(default_factory=RobustScaler)
    y_scaler: RobustScaler = field(default_factory=lambda: RobustScaler(with_centering=False))
    log: dict = field(default_factory=dict)

    def fit(self, returns_w, ema_lag, train_rows, val_rows, verbose: int = 0) -> ReturnForecaster:
        import keras

        lb = self.cfg.lookback
        x_tr, x_va = make_windows(returns_w, train_rows, lb), make_windows(returns_w, val_rows, lb)
        y_tr = returns_w[train_rows] - ema_lag[train_rows]
        y_va = returns_w[val_rows] - ema_lag[val_rows]
        self.x_scaler.fit(x_tr.reshape(-1, x_tr.shape[-1]))
        self.y_scaler.fit(y_tr)
        loss = keras.losses.Huber(delta=self.cfg.huber_delta)
        self.model = build_lstm(returns_w.shape[1], self.cfg, loss, "return_lstm")
        self.log = _fit(
            self.model, _scale3d(self.x_scaler, x_tr), self.y_scaler.transform(y_tr).astype(np.float32),
            _scale3d(self.x_scaler, x_va), self.y_scaler.transform(y_va).astype(np.float32),
            self.cfg, verbose,
        )  # fmt: skip
        return self

    def predict(self, returns_w, ema_lag, rows) -> np.ndarray:
        x = _scale3d(self.x_scaler, make_windows(returns_w, rows, self.cfg.lookback))
        res = self.y_scaler.inverse_transform(self.model.predict(x, verbose=0, batch_size=512))
        return res + ema_lag[rows]


@dataclass
class RiskForecaster:
    """Risk LSTM with its input scaler. Predicts sigma_t (daily volatility)."""

    cfg: NetConfig
    model: object = None
    x_scaler: RobustScaler = field(default_factory=lambda: RobustScaler(with_centering=False))
    log: dict = field(default_factory=dict)

    def fit(self, returns_w, train_rows, val_rows, verbose: int = 0) -> RiskForecaster:
        lb = self.cfg.lookback
        x_tr, x_va = make_windows(returns_w, train_rows, lb), make_windows(returns_w, val_rows, lb)
        self.x_scaler.fit(x_tr.reshape(-1, x_tr.shape[-1]))
        self.model = build_lstm(returns_w.shape[1], self.cfg, gaussian_nll, "risk_lstm")
        if self.cfg.init_output_bias:
            # Start the variance at its training-sample level instead of
            # softplus(0) = 0.69. The output has to reach softplus^-1(1e-4) ~ -9.2;
            # at Adam's step size of 1e-3 that alone takes ~9,000 updates, more
            # than 200 epochs provide in the early, shorter folds.
            var0 = np.mean(returns_w[train_rows] ** 2, axis=0)
            dense = self.model.layers[-1]
            kernel, _ = dense.get_weights()
            dense.set_weights([kernel, np.log(np.expm1(var0)).astype(np.float32)])
        self.log = _fit(
            self.model, _scale3d(self.x_scaler, x_tr), returns_w[train_rows].astype(np.float32),
            _scale3d(self.x_scaler, x_va), returns_w[val_rows].astype(np.float32),
            self.cfg, verbose,
        )  # fmt: skip
        return self

    def predict(self, returns_w, rows) -> np.ndarray:
        x = _scale3d(self.x_scaler, make_windows(returns_w, rows, self.cfg.lookback))
        logits = self.model.predict(x, verbose=0, batch_size=512).astype(np.float64)
        return np.sqrt(np.logaddexp(0.0, logits))  # softplus, numerically stable
