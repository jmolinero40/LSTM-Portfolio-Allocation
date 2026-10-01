# -*- coding: utf-8 -*-
"""
Created on Tue Oct 14 15:51:51 2025

@author: Javier Molinero Araguas
"""

from __future__ import annotations

from pathlib import Path
from Descarga import recoge_datos

import math
import json
import pickle
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional
import os

import numpy as np
import pandas as pd
from lightgbm import LGBMRanker

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
returns_df = pd.read_parquet(parquet_path) #NOTA IMPORTANTE: lambdamart usa pd.DataFrame

print("\n✅ Datos cargados correctamente")
print(returns_df.head())

#Aseguramos que el índice es una fecha, si no, detenemos

assert isinstance(returns_df.index, pd.DatetimeIndex), "El índice no está en formato fecha"

n_features=returns_df.shape[1] #Número de activos

print("\n✅ Último procesado de datos realizado")


""""
Esta primera fase es la de preprocesamiento. 

El preprocesamiento (o feature engeneering) en algoritmos de clasificación es 
más compleja, pues se deben definir las métricas, hacer los scoring,  hacer 
el split del conjunto de datos y organizarlo en los formatos correctos
para el algoritmo del ranker. 
"""


def _rolling_feats(s: pd.Series):

    """
    Esta función toma la serie temporal de un activo y devuelve varias de 
    sus features de interés para hacer el scoring correspondiente como:
        
        w21-> Ventana de datos a 21 días
        vol-> Volatilidad de la serie
        mean,min,max -> media, minimo, maximo
        skew -> asimetría
        kurt -> curtosis
        
    Además añadiremos unas métricas de rentabilidad ajustada al riesgo, algo
    así como un Ratio de Sharpe con el momentum o inercia del activo.  
    """
    
    w21 = s.tail(21)
    w63 = s.tail(63)
    
    out = {
        "mom_21": w21.sum(),
        "mom_63": w63.sum(),
        "vol_look": s.std(ddof=0), #ddof usa N y no N-1 en dt
        "ret_mean": s.mean(),
        "ret_min": s.min(),
        "ret_max": s.max(),
        "skew": (s.skew() if s.std(ddof=0) > 0 else 0.0), #evitar Nan si la serie tiene desviación 0
        "kurt": s.kurt(),
    }
    
    #Ratios de momentum "inercia" ajustados por volatilidad
    #Sumamos 1e-8 para no dividir por 0 nunca
    out["mom21_vol"] = out["mom_21"] / (out["vol_look"] + 1e-8)
    out["mom63_vol"] = out["mom_63"] / (out["vol_look"] + 1e-8)
    
    return out


"""
Creamos la clase estructurada DatasetLTR, que precisa el algoritmo.
En esencia es lo mismo que se hace en redes convencionales, pero estructurando
todo dentro de un bolsillo donde cada elemento está definido y tipado

-> Es mucho más limpio y estructurado

Matriz X : Variables predictoras (filas ejemplos; columnas variables)
Vector y : Vector de 
Lista g: En Learning-to-Rank, los datos están agrupados por "queries" (aquí: cada fecha).
g_train dice al modelo cuántas filas corresponden a cada query en X_train.

"""
@dataclass
class DatasetLTR:
    
    #Vectores de variables
    X_train: np.ndarray; y_train: np.ndarray; g_train: List[int]
    X_val:   np.ndarray; y_val:   np.ndarray; g_val:   List[int]
    X_test:  np.ndarray; y_test:  np.ndarray; g_test:  List[int]
    
    #Guarda información para reconstruir el ranking
    meta_test: pd.DataFrame       # ['date','ticker','y_cont','qid']
    
    #Lista de variables predictoras
    feat_cols: List[str]
    
    #Guardamos aquí los límites temporales de cada split que se haga
    splits_info: Dict[str, List[pd.Timestamp]]


"""
Transformamos nuestro df de  retornos (fecha x activo) en filas (fecha, activo) 
con features, label y query_id.
Label continua: suma de retornos en [t, t+horizon). Relevancia: decilas por fecha.
"""

def construye_dataset_ltr(
    returns: pd.DataFrame,
    lookback: int = 60,
    horizon: int = 1,
    train_frac: float = 0.70,
    val_frac: float = 0.15):

    
    cols = list(returns.columns)
    dates = list(returns.index)
    
    #Construimos las ventanas
    rows = []
    
    for t in range(lookback, len(dates) - horizon + 1):
        
        d = dates[t]                         # fecha t   (query_id)
        
        window = returns.iloc[t - lookback:t] #datos de los activos como tal
        y_future = returns.iloc[t:t + horizon].sum()  # retorno futuro acumulado por activo (con h=1, solo coge una fila)
        
        for ticker in cols:
            
            s = window[ticker]
            feats = _rolling_feats(s) #damos las features de ese activo en la ventana temporal
            
            #esta fila guarda en esa fecha y para esa empresa lo que le va a ocurrir
            row = {"date": d, "ticker": ticker, "y_cont": float(y_future[ticker])}
            row.update(feats) #añade todo el resto de variables decisoras para ese ticker
            rows.append(row) #hacemos este proceso para cada ticker y para cada ventana temporal

    Xy = pd.DataFrame(rows)
    
    # Relevancia por deciles (0..9) por fecha
    def _decilas(s: pd.Series) -> pd.Series : 
        
        """
        Xy es un DataFrame con una fila por (fecha, activo), 
        y una columna y_cont que contiene el retorno futuro de ese activo 
        (por ejemplo, el retorno de los próximos 5 días).

        Queremos transformar esos valores continuos (y_cont) 
        en deciles (0–9) dentro de cada fecha. Así, el modelo sabrá que el 
        activo con y_decile=9 tuvo mejor retorno futuro que el de y_decile=0.
        
        Funcionamiento de rank.
        
        s = pd.Series([0.03, -0.01, 0.02, 0.05, 0.05])
        
        r = s.rank(method="first")
        print(r)
        0    3.0
        1    1.0
        2    2.0
        3    4.0
        4    5.0
        dtype: float64
        
        
        Funcionamiento de qcut.
        
        q = pd.qcut(r, 10, labels=False, duplicates="drop")
        print(q)
        0    5
        1    0
        2    3
        3    8
        4    9
        dtype: int64
        
        NOTA: Devuelve los deciles (0–9) calculados sobre los rangos r.
        Si usáramos pd.qcut(s, 10, ...) directamente, y hay valores repetidos o 
        empates en s, qcut puede dar errores o grupos desiguales.

        s.rank() convierte esos valores a una secuencia ordenada continua (1, 2, 3, …, N),
        que garantiza que qcut tendrá una distribución perfectamente creciente 
        para dividir en cuantiles.
        """
        r = s.rank(method="first")
        
        #qcut rompe los datos en cuantiles definidos
        q = pd.qcut(r, 10, labels=False, duplicates="drop")
        
        return q.astype(int)
    
    #Forma compacta para coger el vector de retornos por fecha, aplicar el ranking y guardar
    Xy["y_decile"] = Xy.groupby("date")["y_cont"].transform(_decilas)

    # El ranker necesita saber cuántos elementos hay en cada query
    
    #Damos las fechas únicas que hay (nuestro rango temporal)
    uniq_dates = sorted(Xy["date"].unique().tolist())
    
    # Esta forma compacta nos devuelve un diccionario de fechas enumeradas
    date2qid = {d: i for i, d in enumerate(uniq_dates)}
    
    #Esto nos va a devolver las Query ID (como su puesto de enumeración)
    Xy["qid"] = Xy["date"].map(date2qid)
    
    #Features que ya teníamos definidas
    feat_cols = [
        "mom_21", "mom_63", "vol_look", "ret_mean", "ret_min", "ret_max",
        "skew", "kurt", "mom21_vol", "mom63_vol"
    ]


    # Splits temporales por fecha
    n = len(uniq_dates)
    i_train = int(n * train_frac)
    i_val = int(n * (train_frac + val_frac))
    
    #Fechas del split que hacemos
    train_dates = set(uniq_dates[:i_train])
    val_dates   = set(uniq_dates[i_train:i_val])
    test_dates  = set(uniq_dates[i_val:])

    """
    Vamos a hacer para cada conjunto del split un corte en las variables que
    nos interesan en cada uno
    """
    def _pack(sub: pd.DataFrame):
        
        # Para la matriz X las variables predictoras
        X = sub[feat_cols].astype(np.float32).values #al hacer .values perdemos indice
        
        #Para el vector y los retornos a predecir (LAS CLASES)
        y = sub["y_decile"].astype(int).values
        
        #Para el vector g el tamaño de las queries que estamos pasando
        #le hacemos el tolist() porque es lo que el ranker espera
        g = sub.groupby("qid").size().tolist()
        
        return X, y, g
    
    #Adaptamos los conjuntos 
    tr = Xy[Xy["date"].isin(train_dates)].copy()
    va = Xy[Xy["date"].isin(val_dates)].copy()
    te = Xy[Xy["date"].isin(test_dates)].copy()

    X_train, y_train, g_train = _pack(tr)
    X_val,   y_val,   g_val   = _pack(va)
    X_test,  y_test,  g_test  = _pack(te)

    meta_test = te[["date", "ticker", "y_cont", "qid"]].reset_index(drop=True)

    return DatasetLTR(
        X_train, y_train, g_train,
        X_val,   y_val,   g_val,
        X_test,  y_test,  g_test,
        meta_test, feat_cols,
        splits_info={
            "train_dates": sorted(list(train_dates)),
            "val_dates":   sorted(list(val_dates)),
            "test_dates":  sorted(list(test_dates)),
        }
    )


"""
A continuación pasamos a definir las métricas que van a guiar a nuestro modelo
tanto en evaluación como en backtesting. Son métricas financieras, ya que este
modelo es una especie de scoring, más que uno de clasificación al uso.

Su desempeño por lo tanto depende de lo bien que le vaya haciendo trading
no clasificando cosas
"""

"""
El máximo drawdown (MDD) mide la mayor pérdida acumulada desde un pico 
hasta un valle en la curva de capital (equity curve).

Interpretación:
Si MDD = -0.25, la estrategia llegó a perder un 25% desde su máximo histórico 
antes de recuperarse.
Es una medida de riesgo de pérdida extrema o dolor de inversor.


Unidades:
Se expresa como porcentaje o proporción negativa.
Ejemplo: -0.15 → −15%.


Importancia:
Complementa la volatilidad: una estrategia puede tener Sharpe alto pero 
drawdowns profundos.
El drawdown mide la “profundidad del agujero” al que caes antes de volver a máximos.
Usarlo junto con rentabilidad permite calcular ratios de tipo 
“retorno/riesgo máximo” (como el Calmar ratio).


Ejemplo:
equity = [1.00, 1.10, 1.05, 1.20, 1.10]
cummax = [1.00, 1.10, 1.10, 1.20, 1.20]
dd = [0, 0, 1.05/1.10-1=-0.045, 0, 1.10/1.20-1=-0.0833]
max_drawdown = -0.0833 (-8.33%).
"""
def max_drawdown(equity: pd.Series) -> float:
    cummax = equity.cummax()
    dd = equity / cummax - 1.0
    return dd.min()


"""
El Sharpe ratio es la rentabilidad ajustada por riesgo

Unidades:
Adimensional (sin unidades)

Interpretación:
0 → sin exceso de rentabilidad.
1 → aceptable (una desviación de retorno por unidad de riesgo).
2 → muy bueno.
3 o más → excepcional (raro y difícil de mantener).

Importancia:
Es el ratio más usado en gestión cuantitativa.
Permite comparar estrategias con diferente volatilidad.
En nuestro modelo, el Sharpe de la estrategia resultante mide si el modelo 
ordena activos de forma útil para ganar más por unidad de riesgo.
"""
def annualize_sharpe(ret: pd.Series, rf: float = 0.0, periods: int = 252) -> float:
    ex = ret - rf / periods
    mu = ex.mean() * periods
    sigma = ex.std(ddof=0) * math.sqrt(periods)
    return 0.0 if sigma == 0 else mu / sigma


"""
El Calmar ratio mide la rentabilidad anual compuesta (CAGR) dividida entre el 
máximo drawdown absoluto.

Unidades:
Adimensional, como el Sharpe.
Ejemplo: Calmar = 0.5 → la rentabilidad anual es la mitad del peor drawdown sufrido.

Interpretación:
Cuanto mayor, mejor: más rendimiento relativo al dolor máximo.

CAGR = 15%, MDD = 10% → Calmar = 1.5 ✅
CAGR = 15%, MDD = 30% → Calmar = 0.5 ❌

Importancia:
Mide la eficiencia de recuperación: cuánto ganas por cada unidad de caída máxima.
Muy usado en hedge funds y managed futures porque penaliza estrategias con drawdowns 
grandes aunque tengan buena media.
En nuestro caso, sirve para evaluar si los rankings del modelo generan una 
curva de equity estable y sin caídas dolorosas.
"""
def calmar_ratio(ret: pd.Series) -> float:
    eq = np.exp(ret.cumsum())#por ser retornos LOGARITMICOS
    mdd = abs(max_drawdown(eq))
    if mdd == 0:
        return float("inf")
    cagr = eq.iloc[-1] ** (252 / len(eq)) - 1.0
    return cagr / mdd


"""
Esta función ajusta los retornos para que la volatilidad anualizada sea 
aproximadamente igual al valor objetivo (target_annual_vol).
Es una técnica de control de riesgo (risk targeting).

Ejemplo simple: si la vol rolling diaria ≈ 1% y daily_tgt≈0.94%, escala ≈ 0.94 
y reduces un 6%
"""
def apply_vol_target(ret: pd.Series, target_annual_vol: float = 0.15, lookback: int = 21) -> pd.Series:

    if ret.empty:
        return ret
    
    daily_tgt = target_annual_vol / math.sqrt(252)
    
    rolling_vol = ret.rolling(lookback).std(ddof=0).replace(0, np.nan).bfill()
    
    scale = daily_tgt / rolling_vol
    
    return ret * scale


"""
Toma los scores del modelo por (fecha, activo), rankea dentro de cada fecha, 
marca +1 a los top (p.ej., 10%), −1 a los bottom, 0 al resto, calcula el 
retorno diario medio de ese long–short del equity o cartera, aplica vol targeting,
mecanismos de ajuste para mantener la volatilidad de la cartera siempre constante, y 
devuelve la serie de PnL (pérdidas y ganancias) más una tabla con picks/posiciones.

Una cartera long–short es aquella en la que:

Tomas posiciones largas (long) en activos que crees que subirán (los compras),

Tomas posiciones cortas (short) en activos que crees que bajarán (los vendes prestados),

Y equilibras ambas partes para que la cartera tenga riesgo de mercado neutralizado (por ejemplo, inviertes +100% long y −100% short).
"""

def backtest_from_scores(
    meta: pd.DataFrame,
    scores: np.ndarray,
    top_q: float = 0.10,#de aquí sale la cantidad de activos en cartera
    bottom_q: float = 0.10,
    vol_target: Optional[float] = 0.15) -> Tuple[pd.Series, pd.DataFrame]:
    
    """
    Construye long–short por fecha a partir de scores:
      +1 en top_q, -1 en bottom_q, igual-ponderado por lado.
    """
    m = meta.copy()
    m["score"] = scores
    #revisar en caso de fallo por aqui
    m["rank"] = m.groupby("date")["score"].rank(ascending=False, method="first")
    counts = m.groupby("date")["ticker"].transform("count")
    
    #clip(lower=1) es para que al menos haya un activo considerado
    k_top = (counts * top_q).clip(lower=1).astype(int)
    k_bot = (counts * bottom_q).clip(lower=1).astype(int)

    m["pos"] = 0
    #estas lineas unsan la sintaxis de numpy de (condicion, valor si true, valor si false)
    m["pos"] = np.where(m["rank"] <= k_top, 1, m["pos"])
    m["pos"] = np.where(m["rank"] > (counts - k_bot), -1, m["pos"])


    #El método .apply() de pandas aplica una función a cada grupo por separado.
    #lambda en Python es una forma abreviada de definir funciones anónimas, es decir, sin usar def.
    pnl = m.groupby("date").apply(lambda df: (df["pos"] * df["y_cont"]).mean()).rename("ret")

    if vol_target is not None:
        #!!!Si aplicas vol targeting a 15% anual, el retorno del día se escala por el factor daily_tgt / vol_rolling.
        pnl = apply_vol_target(pnl, target_annual_vol=vol_target, lookback=21)

    return pnl, m[["date", "ticker", "y_cont", "score", "rank", "pos"]].copy()


# ===================== Entrenamiento del LambdaMART =====================

@dataclass
class LTRConfig:
    lookback: int = 60
    horizon: int = 1
    n_estimators: int = 800
    learning_rate: float = 0.05
    num_leaves: int = 63
    max_depth: int = -1
    feature_fraction: float = 0.9
    min_data_in_leaf: int = 50
    top_q: float = 0.10
    bottom_q: float = 0.10
    vol_target: float = 0.15    # anualizada


def entrena_lambdamart(returns_df: pd.DataFrame, cfg: LTRConfig):
    
    """
    Punto de entrada principal: recibe returns_df (log diarios, sin NA),
    entrena LGBMRanker (LambdaMART), hace backtest y devuelve resultados.
    """
    ds = construye_dataset_ltr(returns_df, cfg.lookback, cfg.horizon)

    model = LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=cfg.n_estimators,
        learning_rate=cfg.learning_rate,
        num_leaves=cfg.num_leaves,
        max_depth=cfg.max_depth,
        feature_fraction=cfg.feature_fraction,
        min_data_in_leaf=cfg.min_data_in_leaf,
        subsample=1.0,
        subsample_freq=0,
        random_state=42,
        reg_alpha=0.0,
        reg_lambda=0.0,
        n_jobs=-1,
    )

    model.fit(
        ds.X_train, ds.y_train, group=ds.g_train,
        eval_set=[(ds.X_val, ds.y_val)],
        eval_group=[ds.g_val],
        eval_at=[5, 10, 20],
        #En LightGBM, ese parámetro le dice al modelo en qué posiciones del 
        #ranking evaluar la calidad NDCG durante el entrenamiento y validación.
    )

    # Predicción + backtest en test
    scores_test = model.predict(ds.X_test)
    pnl, picks = backtest_from_scores(
        ds.meta_test, scores_test,
        top_q=cfg.top_q, bottom_q=cfg.bottom_q,
        vol_target=cfg.vol_target
    )

    # Métricas
    sharpe = annualize_sharpe(pnl)
    calmar = calmar_ratio(pnl)
    eq = (1 + pnl).cumprod()
    mdd = abs(max_drawdown(eq))
    hit = (pnl > 0).mean()

    metrics = {
        "Sharpe": float(sharpe),
        "Calmar": float(calmar),
        "MaxDD": float(mdd),
        "HitRatio": float(hit),
        "N_test_days": int(pnl.shape[0]),
        "Lookback": cfg.lookback,
        "Horizon": cfg.horizon,
        "TopQ": cfg.top_q,
        "BottomQ": cfg.bottom_q,
        "VolTarget": cfg.vol_target,
    }

    return model, pnl, picks, metrics



cfg = LTRConfig(lookback=60, horizon=1, top_q=0.10, bottom_q=0.10, vol_target=0.15)
model, pnl, picks, metrics = entrena_lambdamart(returns_df, cfg)

#Guardamos el modelo

data_dir = Path(__file__).parent / "Modelos"  # puede tener acentos
os.makedirs(data_dir, exist_ok=True)
model_path = data_dir / "lambdamart_model.txt"

txt = model.booster_.model_to_string()  # ← obtiene el modelo como texto
model_path.write_text(txt, encoding="utf-8")  # ← Python lo escribe (Unicode OK)

print(f"✅ Guardado en: {model_path}")

print("\n",metrics)
pnl.cumsum().plot(title="Equity curve (vol-targeted)")
