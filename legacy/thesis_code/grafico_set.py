# -*- coding: utf-8 -*-
"""
Created on Sat Nov  8 12:48:19 2025

@author: jmoli
"""

"""
Gráfico base 100 de SAN.MC con líneas verticales y etiquetas de tramos.
Requisitos: yfinance, pandas, matplotlib
pip install yfinance pandas matplotlib
"""

import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# --- Parámetros ---
ticker = "SAN.MC"
fecha_1 = pd.to_datetime("19/01/2022", dayfirst=True)
fecha_2 = pd.to_datetime("30/11/2023", dayfirst=True)
# Opcional: fija un inicio para el histórico (o deja None para máximo disponible)
inicio = "2013-01-01"   # cambia si quieres otro rango
fin = None

# --- Descarga y base 100 ---
df = yf.download(ticker, start=inicio, end=fin, auto_adjust=False, progress=False)
if df.empty:
    raise RuntimeError("No se han obtenido datos de Yahoo Finance para SAN.MC en el rango indicado.")

serie = df["Adj Close"].dropna().sort_index()
base100 = serie / serie.iloc[0] * 100  # índice en base 100

# --- Preparar figura ---
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(base100.index, base100, linewidth=1.8)
ax.set_title(f"{ticker} — Base 100", pad=12)
ax.set_xlabel("Fecha")
ax.set_ylabel("Índice (Base 100)")
ax.grid(True, alpha=0.3)

# --- Líneas verticales ---
ax.axvline(fecha_1, linestyle="--", linewidth=1.5)  # 1ª partición
ax.axvline(fecha_2, linestyle="--", linewidth=1.5)  # 2ª partición

# --- Etiquetas de tramos en la parte inferior del gráfico ---
xmin, xmax = base100.index.min(), base100.index.max()
ymin, ymax = ax.get_ylim()

# Puntos medios de cada tramo para colocar el texto centrado
mid_1 = xmin + (fecha_1 - xmin) / 2
mid_2 = fecha_1 + (fecha_2 - fecha_1) / 2
mid_3 = fecha_2 + (xmax - fecha_2) / 2

# Colocamos el texto un pelín por encima del borde inferior del eje
y_text = ymin + 0.02 * (ymax - ymin)

ax.text(mid_1, y_text, "Train Set", ha="center", va="bottom")
ax.text(mid_2, y_text, "Validation Set", ha="center", va="bottom")
ax.text(mid_3, y_text, "Test Set", ha="center", va="bottom")

# Mejoras visuales
ax.set_xlim(xmin, xmax)
plt.tight_layout()
plt.show()
