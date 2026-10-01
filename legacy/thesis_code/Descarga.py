# -*- coding: utf-8 -*-
"""
Created on Sat Oct  4 12:19:57 2025

@author: Javier Molinero Araguas

Este código pretende hacer la recogida de datos, almacenamiento eficiente de
los mismos y posteriormente procesarlos de la manera adecuada para entrenar
al modelo.
"""

import yfinance as yf, pandas as pd, pyarrow, duckdb, math, numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from pathlib import Path
from difflib import get_close_matches

"""
Paso 1: Definir qué empresas/fondos nos interesan para construir la base
de datos y posteriormente entrenar al modelo.

Sector	    ETF (Ticker)	    Empresa(s)
Tecnología	XLK	                Microsoft, Apple, NVIDIA…
Finanzas	XLF	                bancos y aseguradoras USA
Energía	    XLE	                Exxon, Chevron…
Salud	    XLV	                pharma, biotecnología, seguros de salud
IndustriA   XLI	                Boeing, GE, 3M
Real Estate	XLRE	            REITs

Finanzas    SAN.MC              Santander (España)
Energía     ENEL.MI             ENEL (Italia)
Industria   SIE.DE              Siemens (Alemania)
Salud       NOVN.SW             Novartis (Suiza)


"""
def recoge_datos(tickers=[]):

    #Definimos los tickers de EEUU y de Europa
    if tickers==[]:
        
        tic_eeuu=["XLK", "XLF", "XLY", "XLE", "XLV", "XLI", "XLB"]
        tic_eu=["SAN.MC", "ENEL.MI", "SIE.DE", "NOVN.SW"]
        #tic_eu=["XLK"]
        
        tickers=tic_eeuu+tic_eu
        #tickers=tic_eu
    
    #Definimos la fecha de inicio de los datos y el directorio
    
    mi_directorio = Path("C:/Users/jmoli/Desktop/TFG ECONOMIA/Códigos/3.1 Recogida de datos y procesamiento/Datos")
    mi_directorio.mkdir(parents=True, exist_ok=True) #asegura la existencia de la carpeta
    
    start, end= "2013-01-01", None
    
    #Llamamos a Yahoo Finance para descargar los datos (progress es la barra de progreso en la descarga)
    df=yf.download(tickers, start=start, end=end, auto_adjust=False, progress=False)["Adj Close"]
    df=df.sort_index() #aseguramos el orden cronológico
    print("\n✅ La descarga se ha hecho correctamente", flush=True)
    
    #Gestión de missings
    dimension=df.shape
    df=df.dropna()
    missing=dimension[0] - df.shape[0]
    
    print("\nEl dataframe tiene este aspecto:")
    print(df.head())
    print(
        f"\nEl dataframe original tenía {dimension[0]} observaciones, "
        f"mientras que ahora tiene {df.shape[0]}, "
        f"o sea, se han eliminado {missing} valores faltantes."
        f"Esto representa un {round(missing/dimension[0]*100,2)}% del total."
        )
    
    
    #Calculamos los retorno con la función pct.change()
    
    data = np.log(df.replace(0, np.nan) / df.shift(1))
    data=data.dropna() #quitamos la primera fila porque van a ser Nan

   
    #Opcional: Winsoriza
    print ("\n📈 El dataframe después de calcular los retornos tiene este aspecto antes de winsorizar: ")
    
    print("\nAntes:")
    #print(data.describe())

    # Aplicar winsorización al 0.5% y 99.5%
    data = winsoriza_retorno(data, 0.005, 0.995)

    # Comparar estadísticos antes/después
    
    print("\nDespués:")
    print(data.describe())
    
    #print(data.head())
 
    
    #Guardamos el archivo en el directorio en formato parquet
    dim=len(tickers)
    ruta_parquet = mi_directorio / f"rentabilidades_dim_{dim}.parquet"
    data.to_parquet(ruta_parquet)
    
    print(f"\n💾 Archivo guardado correctamente en:\n{ruta_parquet}")
    
    #duckdb usa sentencias SQL
    
    #print("\nVisualizamos cómo es el archivo .parquet;")
    #duckdb.query(f"SELECT * FROM '{(mi_directorio / 'rentabilidades.parquet')}' LIMIT 100").show()
    
    #Pasamos a coger las mátricas de riesgo (matriz de varianza-covarianza)
    
    matriz_covarianza=data.cov()
    ruta_parquet = mi_directorio / "covarianzas.parquet"
    matriz_covarianza.to_parquet(ruta_parquet)
    print(f"\n💾 Matriz de covarianza guardada correctamente en:\n{ruta_parquet}")
    
    #Y ahora las correlaciones entre los activos por si fuera de utilidad
    
    matriz_correlaciones=data.corr()
    ruta_parquet = mi_directorio / "correlaciones.parquet"
    matriz_correlaciones.to_parquet(ruta_parquet)
    print(f"\n💾 Matriz de correlaciones correctamente en:\n{ruta_parquet}")
    
    #representa_serie(df)
    #print("\n📈​ Los gráficos de las series temporales se han guardado correctamente")

    return data

def representa_serie(data):
    
    carpeta=Path("C:/Users/jmoli/Desktop/TFG ECONOMIA/Códigos/3.1 Recogida de datos y procesamiento/gráficos")    
    carpeta.mkdir(parents=True, exist_ok=True)  # asegura la carpeta
    
    #Cogemos índice base para comparar
    
    data=data/data.iloc[0]*100 #100 es la base del índice
    
    #Coge los tickers
    tickers=list(data.columns)
    print("\nLos tickers que se van a representar y guardar son:", tickers)
    
    for ticker in tickers:
     
    #Representamos    
        plt.figure(figsize=(9, 4))
        plt.plot(data.index, data[ticker])
        plt.title(f"Serie temporal: {ticker}")
        plt.xlabel("Fecha")
        plt.ylabel("Valor")
        plt.grid(True, alpha=0.3)    
        
     # Nombre del archivo
        try:
            f_ini = data.index.min().strftime("%Y%m%d")
            f_fin = data.index.max().strftime("%Y%m%d")
            nombre = f"{ticker}_{f_ini}_{f_fin}.png"
        except Exception:
            nombre = f"{ticker}.png"

        ruta = Path(carpeta) / nombre

        # Guardar y cerrar
        plt.tight_layout()
        plt.savefig(ruta, dpi=150, bbox_inches="tight")
        plt.close()

        print("\n✅ Guardado:", ruta)
     
    #Creamos también una imagen con todos los gráficos para comparativa
    n = len(tickers)
    if n == 0:
        return
    ncols = 3
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 3*nrows))
    # --- aplanar SIEMPRE a lista de Axes ---
    if isinstance(axes, np.ndarray):
        axes = axes.ravel().tolist()
    else:
        axes = [axes]

    for i, ticker in enumerate(tickers):
        ax = axes[i]
        ax.plot(data.index, data[ticker])
        ax.set_title(ticker)
        ax.grid(True, alpha=0.3)

    # ocultar huecos sobrantes
    for j in range(len(tickers), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Series temporales (todos los tickers)", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])

    try:
        f_ini = data.index.min().strftime("%Y%m%d")
        f_fin = data.index.max().strftime("%Y%m%d")
        nombre_all = f"ALL_{f_ini}_{f_fin}.png"
    except Exception:
        nombre_all = "ALL.png"

    ruta_all = carpeta / nombre_all
    fig.savefig(str(ruta_all), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("\n🖼️ Imagen combinada guardada:", ruta_all)


"""REVISAR MÁS ADELANTE
def grafico_interactivo(df):
    ""
    Muestra un gráfico interactivo (hover) del ticker indicado.
    df: DataFrame con índice de fechas y columnas=tickers.
    ticker: string, por ejemplo 'XLK'.
    ""
    
    tickers = [str(c) for c in df.columns]
    ticker=input(f"Elegir un ticker de entre{tickers}: ").strip()
    
    # Validación simple + sugerencia si hay error de tecleo
    if ticker not in df.columns:
        sugerencia = get_close_matches(ticker, tickers, n=1)
        msg = f"Ticker '{ticker}' no encontrado."
        if sugerencia:
            msg += f" ¿Quisiste decir '{sugerencia[0]}'?"
        raise ValueError(msg)

    # Graficar (modo wide: y es el nombre de la columna)
    fig = px.line(
        df,
        x=df.index,
        y=ticker,
        title=f"Serie temporal: {ticker}",
        labels={"x": "Fecha", "y": "Valor"}
    )
    fig.update_layout(hovermode="x unified", template="plotly_white")
    fig.show()  # Si no se ve en Spyder: fig.show(renderer="browser")
    
"""

def winsoriza_retorno(data, limite_inf = 0.005, limite_sup  = 0.995):
    """
    Winsoriza los retornos de un DataFrame, limitando los valores extremos a los percentiles especificados.
    
    Parámetros
    ----------
    data : pd.DataFrame
        DataFrame con retornos (logarítmicos o simples).
    limite_inf : float
        Percentil inferior (por defecto 0.01 → 1%).
    limite_sup : float
        Percentil superior (por defecto 0.99 → 99%).

    Retorna
    -------
    pd.DataFrame
        DataFrame con los valores winsorizados.
    """

    # Copiamos para no modificar el original
    data_w = data.copy()

    # Iteramos por columnas (por activo)
    for col in data_w.columns:
        
        q_inf = data_w[col].quantile(limite_inf)
        q_sup = data_w[col].quantile(limite_sup)

        # Limita valores por debajo o por encima de los cuantiles
        data_w[col] = np.clip(data_w[col], q_inf, q_sup)

    print(f"\n✅ Winsorización aplicada entre los percentiles {limite_inf*100:.1f}% y {limite_sup*100:.1f}%")
    return data_w

