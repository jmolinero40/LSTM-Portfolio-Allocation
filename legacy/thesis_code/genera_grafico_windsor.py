# -*- coding: utf-8 -*-
"""
SAN.MC: retornos log sin winsorizar (con líneas de quantiles 0.5% y 99.5%)
y serie winsorizada. Gráficos lado a lado.
Robusto a distintas formas de salida de yfinance (Series/DataFrame/MultiIndex).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

# -----------------------------
# Parámetros
# -----------------------------
ticker = "SAN.MC"
start  = "2013-01-01"
end    = None
q_low, q_high = 0.005, 0.995  # 0.5% y 99.5%

# -----------------------------
# Descarga precios
# -----------------------------
raw = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)

# Función auxiliar para extraer una Serie de precios ajustados haga lo que haga yfinance
def extract_adj_close_series(df, ticker):
    """
    Devuelve una Serie 1D de Adj Close.
    Soporta:
      - DataFrame clásico con columnas ['Open','High','Low','Close','Adj Close','Volume']
      - DataFrame con MultiIndex en columnas (p.ej., (campo, ticker) o (ticker, campo))
      - Serie directa (ya buena)
    """
    if isinstance(df, pd.Series):
        # Si ya es Serie, asumimos que son precios (y dejamos tal cual)
        return df.dropna()

    # Caso DataFrame con columnas simples
    if not isinstance(df.columns, pd.MultiIndex):
        if "Adj Close" in df.columns:
            s = df["Adj Close"]
        elif "Close" in df.columns:
            s = df["Close"]
        else:
            # Toma la primera columna como fallback
            s = df.iloc[:, 0]
        return pd.to_numeric(s, errors="coerce").dropna()

    # Caso DataFrame con MultiIndex en columnas
    # Intento 1: nivel más interno es el campo
    try:
        tmp = df.xs("Adj Close", axis=1, level=-1)
        # tmp puede ser Serie (si 1 solo ticker) o DataFrame (varios)
        if isinstance(tmp, pd.Series):
            return pd.to_numeric(tmp, errors="coerce").dropna()
        # Si hay varias columnas (varios tickers), intenta seleccionar la del ticker
        if ticker in tmp.columns:
            return pd.to_numeric(tmp[ticker], errors="coerce").dropna()
        # Fallback: toma la primera columna disponible
        return pd.to_numeric(tmp.iloc[:, 0], errors="coerce").dropna()
    except Exception:
        pass

    # Intento 2: nivel más externo es el campo
    try:
        tmp = df["Adj Close"]
        if isinstance(tmp, pd.Series):
            return pd.to_numeric(tmp, errors="coerce").dropna()
        if ticker in tmp.columns:
            return pd.to_numeric(tmp[ticker], errors="coerce").dropna()
        return pd.to_numeric(tmp.iloc[:, 0], errors="coerce").dropna()
    except Exception:
        pass

    # Último recurso: usa Close
    try:
        tmp = df.xs("Close", axis=1, level=-1)
        if isinstance(tmp, pd.Series):
            return pd.to_numeric(tmp, errors="coerce").dropna()
        if ticker in tmp.columns:
            return pd.to_numeric(tmp[ticker], errors="coerce").dropna()
        return pd.to_numeric(tmp.iloc[:, 0], errors="coerce").dropna()
    except Exception:
        # Como ultimísimo fallback, coge la primera columna que pille
        s = df.iloc[:, 0]
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]
        return pd.to_numeric(s, errors="coerce").dropna()

# Extrae Serie 1D de precios ajustados
px = extract_adj_close_series(raw, ticker)

# -----------------------------
# Retornos logarítmicos
# -----------------------------
rets = np.log(px / px.shift(1)).dropna()
rets.name = "ret_log_SAN"

# Cuantiles ESCALARES
q_lo_val = float(rets.quantile(q_low))
q_hi_val = float(rets.quantile(q_high))

# Serie winsorizada (recorte en los cuantiles)
rets_win = rets.clip(lower=q_lo_val, upper=q_hi_val)

# Métricas informativas (opcionales)
n_total = rets.size
n_low   = int((rets < q_lo_val).sum())
n_high  = int((rets > q_hi_val).sum())

# -----------------------------
# Gráficos lado a lado
# -----------------------------
plt.figure(figsize=(14, 5))

# (1) Sin winsorizar + líneas de cuantiles
ax1 = plt.subplot(1, 2, 1)
ax1.plot(rets.index, rets.values, linewidth=0.8)
ax1.axhline(q_lo_val, linestyle="--", linewidth=1, label=f"q{q_low*100:.1f}% = {q_lo_val:.4f}")
ax1.axhline(q_hi_val, linestyle="--", linewidth=1, label=f"q{q_high*100:.1f}% = {q_hi_val:.4f}")
ax1.set_title(f"{ticker} · Retornos log (sin winsorizar)")
ax1.set_xlabel("Fecha")
ax1.set_ylabel("Retorno logarítmico")
ax1.grid(True, alpha=0.3)
ax1.legend(loc="upper right")

# (2) Winsorizada
ax2 = plt.subplot(1, 2, 2, sharey=ax1)
ax2.plot(rets_win.index, rets_win.values, linewidth=0.8)
ax2.set_title(f"{ticker} · Retornos log winsorizados\n"
              f"(recorte en {q_low*100:.1f}% y {q_high*100:.1f}%)")
ax2.set_xlabel("Fecha")
ax2.grid(True, alpha=0.3)

plt.suptitle(
    f"Winsorización por cuantiles · recortados: abajo={n_low}, arriba={n_high} de {n_total} obs.",
    y=1.03, fontsize=11
)
plt.tight_layout()
plt.show()
