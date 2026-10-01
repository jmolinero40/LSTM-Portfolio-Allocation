# -*- coding: utf-8 -*-
"""
Created on Sun Oct 26 12:25:47 2025

@author: jmoli
"""

from pathlib import Path
import keras, tensorflow as tf
import numpy as np


"""
Pasamos a definir el modelo de riesgo conjunto de la cartera

Queremos obtener ls matrices de correlación y de covarianza en base a los
datos muestrales y las predicciones de riesgo
"""

# ========= 1) Correlaciones EWMA =========
def ewma_corr(returns_hist, lam=0.97, eps=1e-12):
    """
    returns_hist: array (T, N) de retornos diarios (decimales), en orden **antiguo→reciente**.
    lam: lambda de decaimiento EWMA (0.94–0.99 típico).
    Devuelve: matriz de correlaciones (N, N).

    Nota: Con el orden antiguo→reciente, esta implementación da más peso al día T-1 que al 0.
    """
    R = np.asarray(returns_hist, dtype=float)
    T, N = R.shape

    # Pesos EWMA, más peso a observaciones más recientes (última fila)
    w = lam ** np.arange(T-1, -1, -1)     # [lam^(T-1), ..., lam^0]
    w = w / w.sum()

    # Centrado ponderado
    m = (w[:, None] * R).sum(axis=0, keepdims=True)
    X = R - m

    # Covarianza EWMA
    cov = (X.T * w) @ X                    # (N,N)

    # Volatilidad marginal y correlación
    vol = np.sqrt(np.clip(np.diag(cov), eps, None))
    Dinv = np.diag(1.0 / vol)
    corr = Dinv @ cov @ Dinv
    
    # Recorte numérico
    corr = np.clip(corr, -1.0, 1.0)
    
    return corr

# ========= 2) Shrinkage a Identidad (Robustez) =========
def shrink_to_identity(corr, alpha=0.3):
    
    """
    corr_shrunk = (1-alpha)*corr + alpha*I
    alpha en [0,1]. Más alpha => más robusto/menos sobreajuste.
    
    Por qué es importante:

    Ruido y sobreajuste. Las correlaciones muestrales (incluso con EWMA) son ruidosas, 
    sobre todo con muchos activos o ventanas cortas. Mezclarlas con la identidad 
    “tira” de la matriz hacia “sin correlaciones”, reduciendo el ruido.
    
    Estabilidad numérica. Aumenta la condición de la matriz y reduce el riesgo de 
    que se vuelva casi singular (crítico si luego vas a invertirla en Media–Varianza).
    
    Robustez fuera de muestra. Disminuye el error de estimación típico de 
    las off-diagonales, evitando pesos extremos.
    
    Cómo elegir alpha:
    
    0.2–0.6 funciona bien en práctica.
    
    Más alpha ⇒ más robusto (pero menos reactivo a estructura real).
    
    Puedes validarlo (grid simple) con una métrica de cartera 
    (p. ej., estabilidad de pesos o Sharpe en validación).
    """
    N = corr.shape[0]
    return (1.0 - alpha) * corr + alpha * np.eye(N)

# ========= 3) Σ = D(σ̂) · Corr · D(σ̂) =========
def covariance_from_sigma_corr(sigma_hat, corr):
    """
    sigma_hat: vector (N,) de volatilidades diarias (decimales).
    corr: (N,N) matriz de correlaciones.
    """
    s = np.asarray(sigma_hat, dtype=float).ravel()
    D = np.diag(s)
    return D @ corr @ D

# ========= 4) Pipeline sencillo =========
def build_joint_risk_model(sigma_hat_now, returns_hist_window, lam=0.97, alpha=0.3):
    """
    Construye Σ_t usando σ̂_t de tu red + Corr_t (EWMA con shrinkage).
    - sigma_hat_now: (N,) σ̂ diario a fecha t (en el MISMO orden de activos que returns_hist_window).
    - returns_hist_window: (T,N) retornos históricos para estimar correlaciones (p.ej., últimos 125–250 días).
    """
    corr = ewma_corr(returns_hist_window, lam=lam)
    corr = shrink_to_identity(corr, alpha=alpha)
    Sigma = covariance_from_sigma_corr(sigma_hat_now, corr)
    return Sigma, corr


"""
Propuesta de unificación 1: Retorno ajustado al riesgo "Sharpe-like"
"""

def weights_topk_softmax(mu, sigma, k, tau=1.0, w_max=None, eps=1e-12, short_allowed=False):
    """
    Top-K por señal s = mu/(sigma+eps).

    - short_allowed=False: solo largos con s>0. Softmax sobre Top-K por s (desc.).
    - short_allowed=True : se permite corto. Se selecciona Top-K por |s|, se separa en
      positivos (largos) y negativos (cortos). La EXPOSICIÓN BRUTA se reparte entre
      largos y cortos en proporción a la suma de |s| de cada pata dentro del Top-K.
      Dentro de cada pata, los pesos relativos se obtienen con softmax(|s|*tau).
      El cap w_max aplica a la magnitud |w_i| y NO se renormaliza tras capar.
      'efectivo' = 1 - sum(|w|).

    Parámetros:
      mu:    (N,) retornos esperados diarios
      sigma: (N,) volatilidades diarias
      k:     nº total de activos a mantener (largos + cortos)
      tau:   "temperatura" de la softmax (↑tau => más concentración)
      w_max: cap por activo en magnitud (p.ej. 0.12). Si None, sin cap.
      eps:   estabilidad numérica
      short_allowed: si True, permite w<0

    Devuelve:
      w:        (N,) pesos finales (sum(w)=long_pct - (1-long_pct); sum(|w|)≤1 si hay cap)
      top_idx:  índices Top-K usados (orden por |s| si short_allowed, si no por s)
      s:        señales completas (mu/(sigma+eps))
      efectivo: 1 - sum(|w|)  (remanente tras cap)
      long_pct: fracción de exposición bruta asignada a LARGOS (∈[0,1])
    """
    mu = np.asarray(mu, dtype=float).ravel()
    sigma = np.asarray(sigma, dtype=float).ravel()
    N = mu.size
    k = int(np.clip(k, 1, N))

    # Señal Sharpe-like
    s = mu / (sigma + eps)

    # invalidar señales no finitas para el ranking
    s_rank = s.copy()
    bad = ~np.isfinite(s_rank)
    if bad.any():
        s_rank[bad] = -np.inf

    # --- Caso SOLO LARGOS (como el original) ---
    if not short_allowed:
        # Solo s>0
        pos_mask = np.isfinite(s) & (s > 0)
        num_pos = int(np.sum(pos_mask))
        k_eff = min(k, num_pos)

        if k_eff == 0:
            w = np.zeros(N)
            return w, [], s, 1.0, 0.0  # todo en efectivo, long_pct=0

        # Top-K por s (descendente) entre positivos
        order = np.argsort(-s_rank)  # ya tiene -inf para no finitos
        order = order[np.isin(order, np.where(pos_mask)[0])]
        top_idx = order[:k_eff]

        # Softmax en seleccionados (positivos)
        z = tau * s[top_idx]
        z -= np.max(z)
        expz = np.exp(z)
        w_sel = expz / np.sum(expz)

        # Aplicar cap (magnitud) y no renormalizar
        if w_max is not None:
            w_sel = np.minimum(w_sel, float(w_max))

        w = np.zeros(N, dtype=float)
        w[top_idx] = w_sel

        efectivo = max(0.0, 1.0 - np.sum(np.abs(w)))  # en este caso = 0 si no hay cap
        long_pct = 1.0 if w.any() else 0.0
        return w, top_idx, s, efectivo, long_pct

    # --- Caso LARGOS + CORTOS ---
    # Selección por |s| (descendente)
    abs_s = np.abs(s_rank)
    order = np.argsort(-abs_s)
    # Excluir no finitos (eran -inf en s_rank)
    finite_idx = np.where(np.isfinite(s))[0]
    order = order[np.isin(order, finite_idx)]
    top_idx = order[:k]

    if top_idx.size == 0:
        w = np.zeros(N)
        return w, [], s, 1.0, 0.0

    # Separar en largos (s>0) y cortos (s<0) dentro del Top-K
    pos_idx = [i for i in top_idx if s[i] > 0]
    neg_idx = [i for i in top_idx if s[i] < 0]

    # Si todas las señales ~0 entre Top-K -> sin exposición
    if (len(pos_idx) + len(neg_idx)) == 0:
        w = np.zeros(N)
        return w, top_idx.tolist(), s, 1.0, 0.0

    # Reparto bruto según fuerza relativa: sum(|s|) por pata
    sum_abs_pos = float(np.sum(np.abs(s[pos_idx]))) if pos_idx else 0.0
    sum_abs_neg = float(np.sum(np.abs(s[neg_idx]))) if neg_idx else 0.0
    total_abs = sum_abs_pos + sum_abs_neg

    if total_abs <= 0.0:
        w = np.zeros(N)
        return w, top_idx.tolist(), s, 1.0, 0.0

    long_pct = sum_abs_pos / total_abs  # porcentaje de GROSS en largos

    # Softmax dentro de cada pata usando |s| (más robusto con signos)
    def side_softmax(idxs):
        if not idxs:
            return idxs, np.array([])
        zz = tau * np.abs(s[idxs])
        zz -= np.max(zz)
        ee = np.exp(zz)
        return idxs, ee / np.sum(ee)

    pos_idx, w_pos_rel = side_softmax(pos_idx)
    neg_idx, w_neg_rel = side_softmax(neg_idx)

    # Construcción de pesos con escala por reparto bruto
    w = np.zeros(N, dtype=float)
    if len(pos_idx):
        w[pos_idx] = long_pct * w_pos_rel
    if len(neg_idx):
        w[neg_idx] = -(1.0 - long_pct) * w_neg_rel

    # Cap por magnitud y NO renormalizar
    if w_max is not None:
        cap = float(w_max)
        w = np.clip(w, -cap, cap)

    # Efectivo = 1 - sum(|w|) (remanente de gross tras cap)
    efectivo = max(0.0, 1.0 - float(np.sum(np.abs(w))))

    return w, top_idx.tolist(), s, efectivo, float(long_pct)


def portfolio_volatility(w, Sigma):
    """Volatilidad diaria de la cartera."""
    w = np.asarray(w, dtype=float).ravel() # ravel aplana la dimensión por si las dudas
    return float(np.sqrt(max(w @ Sigma @ w, 0.0)))


def target_annual_volatility(w, vol_daily, target_annual_vol=0.15, days_per_year=252):
    """
    Escala el vector de pesos para alcanzar una volatilidad **anual** objetivo.
    Devuelve: (w_escalado, leverage), donde leverage = max(0, sum(|w_escalado|) - 1).

    Nota:
      - Si target_annual_vol es None o vol_daily es ~0, no escala y calcula leverage sobre w actual.
      - 'leverage' es robusto tanto para estrategias long-only como long-short (usa exposición bruta).
    """
    w = np.asarray(w, dtype=float).ravel()

    # Leverage actual (por si no escalamos)
    gross_now = float(np.sum(np.abs(w)))
    leverage_now = max(0.0, gross_now - 1.0)

    if target_annual_vol is None or vol_daily <= 1e-12:
        return w, leverage_now

    vol_annual_now = float(vol_daily) * np.sqrt(days_per_year)
    if vol_annual_now <= 0.0:
        return w, leverage_now

    # Ratio de escalado en anual (coherente de unidades)
    scale = float(target_annual_vol / vol_annual_now)

    w_scaled = w * scale
    gross_after = float(np.sum(np.abs(w_scaled)))
    leverage_after = max(0.0, gross_after - 1.0)

    return w_scaled, leverage_after


def sharpe_like_model(mu_pred, sigma_pred, returns, k, short_allowed=False, target_annual_vol=0.15, leverage_allowed=False):
    # ====== Ejemplo de uso con tus variables ======
    # mu_pred:  (N,)  -> de la red de retornos
    # sigma_hat:(N,)  -> de la red de riesgo
    # Sigma:    (N,N) -> de build_joint_risk_model
    
    # 1) Pesos SR-softmax
    w_sr, top_idx, s, efectivo, long_pct = weights_topk_softmax(
        mu_pred,
        sigma_pred,
        k,
        tau=6.0,        # más alto = más concentrado
        w_max=0.3,
        short_allowed=short_allowed
    )
    
    Sigma, corr = build_joint_risk_model(sigma_pred, returns)
    
    # 2) Medir riesgo de cartera (diario y anual)
    vol_daily = portfolio_volatility(w_sr, Sigma)
    vol_annual = vol_daily * np.sqrt(252)

    # Leverage por defecto (aunque no escalemos)
    leverage = max(0.0, float(np.sum(np.abs(w_sr))) - 1.0)

    if target_annual_vol is not None:
        # Si NO se permite apalancamiento: solo reducir si te pasas de la vol objetivo.
        # Si SÍ se permite apalancamiento: escalar siempre a objetivo (subir o bajar).
        need_scale = (not leverage_allowed and vol_annual > target_annual_vol) or (leverage_allowed is True)
        if need_scale:
            w_sr, leverage = target_annual_volatility(
                w_sr, vol_daily, target_annual_vol=target_annual_vol, days_per_year=252
            )
            vol_daily = portfolio_volatility(w_sr, Sigma)
            vol_annual = vol_daily * np.sqrt(252)

    return np.round(w_sr, 4), np.round(vol_daily, 3), top_idx, s, efectivo, long_pct, leverage
