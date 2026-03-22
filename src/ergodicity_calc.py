"""
ergodicity_calc.py
==================
Calcolo delle metriche di ergodicità sul mercato (SPX).

L'ergodicità misura se la media temporale (rolling) di un asset converge
alla sua media di lungo periodo (expanding). Un mercato non ergodico è in
uno stato strutturalmente divergente dal suo comportamento storico medio.

Metodo: Standard Error of the Mean (SEM)
  threshold = k × σ / √N

  σ = deviazione standard globale dei log-return
  N = finestra rolling (252 trading days = 1 anno)
  k = moltiplicatore di confidenza (default 1.75 ≈ 92% CI)

  Il mercato è NON ERGODICO quando |rolling_mean - expanding_mean| > threshold

Riferimenti:
  Peters, O. (2019). The ergodicity problem in economics. Nature Physics 15.
  Peters & Gell-Mann (2016). Evaluating gambles using dynamics. Chaos 26.

Adattato da: ergodicity-market/src/calculations.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ================================================================
# PARAMETRI DEFAULT
# ================================================================

ROLLING_WINDOW:    int   = 252    # 1 anno di trading days
THRESHOLD_K:       float = 1.75   # moltiplicatore SEM (≈ 92% CI)
MIN_PERIODS_RATIO: int   = 1      # min_periods = ROLLING_WINDOW × ratio


# ================================================================
# FUNZIONI PRINCIPALI
# ================================================================

def compute_ergodicity_features(
    spx_df:        pd.DataFrame,
    rolling_window: int   = ROLLING_WINDOW,
    threshold_k:   float = THRESHOLD_K,
) -> pd.DataFrame:
    """
    Calcola le feature di ergodicità sull'indice SPX.

    Pipeline:
      1. Log-return giornalieri
      2. Rolling mean (media temporale, finestra 252g)
      3. Expanding mean (media spaziale, intera storia)
      4. diff = rolling_mean - expanding_mean
      5. Soglia SEM = threshold_k × σ_globale / √N
      6. Flag is_non_ergodic = |diff| > soglia

    Args:
        spx_df:         DataFrame SPX con 'adjusted_close' o 'close'
        rolling_window: Finestra rolling (default 252 = 1 anno)
        threshold_k:    Moltiplicatore k per la soglia SEM (default 1.75)

    Returns:
        DataFrame con colonne aggiuntive:
          - log_ret         : log-return giornaliero
          - rolling_mean    : media temporale rolling
          - expanding_mean  : media spaziale expanding
          - diff            : rolling_mean - expanding_mean
          - is_non_ergodic  : bool, True se mercato non ergodico
        Righe con NaN in rolling_mean/expanding_mean rimossi.
    """
    price_col = "adjusted_close" if "adjusted_close" in spx_df.columns else "close"
    price     = spx_df[price_col].dropna()

    # 1. Log-return
    log_ret = np.log(price / price.shift(1))

    # 2. Rolling mean (media temporale — stima locale dell'aspettativa)
    rolling_mean = (
        log_ret
        .rolling(window=rolling_window, min_periods=rolling_window)
        .mean()
    )

    # 3. Expanding mean (media spaziale — stima globale di lungo periodo)
    expanding_mean = (
        log_ret
        .expanding(min_periods=rolling_window)
        .mean()
    )

    # 4. Differenza rolling - expanding
    diff = rolling_mean - expanding_mean

    # 5. Soglia SEM
    sigma_global = float(log_ret.dropna().std())
    sem          = sigma_global / np.sqrt(rolling_window)
    threshold    = threshold_k * sem

    # 6. Classificazione giorni
    is_non_ergodic = diff.abs() > threshold

    feat = pd.DataFrame({
        "close":          price,
        "log_ret":        log_ret,
        "rolling_mean":   rolling_mean,
        "expanding_mean": expanding_mean,
        "diff":           diff,
        "is_non_ergodic": is_non_ergodic,
    })

    # Rimuovi righe senza dati sufficienti (warm-up rolling)
    feat = feat.dropna(subset=["rolling_mean", "expanding_mean", "diff"])

    # Salva i parametri come attributi del DataFrame per uso downstream
    feat.attrs["threshold"]    = threshold
    feat.attrs["sigma_global"] = sigma_global
    feat.attrs["sem"]          = sem
    feat.attrs["k_mult"]       = threshold_k

    return feat


def compute_ergodicity_thresholds(erg_feat: pd.DataFrame) -> dict:
    """
    Restituisce le soglie di ergodicità utilizzate nell'analisi.

    Args:
        erg_feat: DataFrame da compute_ergodicity_features()

    Returns:
        Dizionario con:
          - threshold    (float): soglia SEM = k × σ / √N
          - sigma_global (float): σ globale dei log-return
          - sem          (float): Standard Error of the Mean puro (σ / √N)
          - k_mult       (float): moltiplicatore k usato
          - pct_non_erg  (float): percentuale storica di giorni non ergodici
    """
    n_total       = len(erg_feat)
    n_non_erg     = int(erg_feat["is_non_ergodic"].sum())
    pct_non_erg   = 100.0 * n_non_erg / n_total if n_total > 0 else 0.0

    return {
        "threshold":    erg_feat.attrs.get("threshold",    0.0),
        "sigma_global": erg_feat.attrs.get("sigma_global", 0.0),
        "sem":          erg_feat.attrs.get("sem",          0.0),
        "k_mult":       erg_feat.attrs.get("k_mult",       THRESHOLD_K),
        "pct_non_erg":  pct_non_erg,
    }


def assign_ergodicity_state(diff_val: float, threshold: float) -> str:
    """
    Assegna lo stato di ergodicità a un singolo valore di differenza.

    Args:
        diff_val:  Valore corrente di (rolling_mean - expanding_mean)
        threshold: Soglia SEM calcolata da compute_ergodicity_features()

    Returns:
        'Ergodico' se |diff_val| <= threshold, 'Non Ergodico' altrimenti
    """
    if np.isnan(diff_val):
        return "N/D"
    return "Ergodico" if abs(diff_val) <= threshold else "Non Ergodico"


def assign_ergodicity_state_series(erg_feat: pd.DataFrame) -> pd.Series:
    """
    Assegna lo stato di ergodicità all'intera serie storica.

    Args:
        erg_feat: DataFrame da compute_ergodicity_features()

    Returns:
        pd.Series con valori 'Ergodico' o 'Non Ergodico'
    """
    threshold = erg_feat.attrs.get("threshold", 0.0)
    return erg_feat["is_non_ergodic"].map(
        {True: "Non Ergodico", False: "Ergodico"}
    )
