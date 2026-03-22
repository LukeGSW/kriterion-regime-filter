"""
entropy_calc.py
===============
Calcolo delle metriche di entropia sul mercato (SPX).

Implementa due misure complementari:
  1. Shannon Entropy rolling sui log-return (finestra 63 giorni ≈ 1 trimestre)
  2. Permutation Entropy normalizzata rolling (Bandt & Pompe 2002)

Queste misure catturano due aspetti distinti del "disordine" di mercato:
  - Shannon Entropy → dispersione statistica delle distribuzioni dei rendimenti
  - Permutation Entropy → complessità strutturale delle sequenze temporali

Riferimenti:
  Shannon, C.E. (1948). A Mathematical Theory of Communication.
  Bandt & Pompe (2002). Permutation Entropy. PRL 88, 174102.

Adattato da: entropy-market/src/calculations.py
"""

from __future__ import annotations

import itertools
from math import factorial, log2

import numpy as np
import pandas as pd
from scipy.stats import entropy as scipy_entropy


# ================================================================
# PARAMETRI DEFAULT
# ================================================================

SHANNON_WINDOW: int = 63    # finestra rolling ≈ 1 trimestre (63 trading days)
SHANNON_BINS:   int = 10    # bin per discretizzazione
PE_ORDER:       int = 3     # embedding dimension Permutation Entropy
PE_WINDOW:      int = 63    # finestra rolling PE (coerente con Shannon)

# Soglie per classificazione regime (tertili)
REGIME_P_LOW:  float = 1 / 3
REGIME_P_HIGH: float = 2 / 3


# ================================================================
# CORE — MISURE DI ENTROPIA
# ================================================================

def _shannon_entropy_window(arr: np.ndarray, bins: int = SHANNON_BINS) -> float:
    """
    Shannon Entropy H = -Σ p_i · log2(p_i) su un array discretizzato in `bins` bin.

    Un'alta Shannon Entropy indica che i rendimenti sono distribuiti uniformemente
    tra i bin → alta imprevedibilità / bassa struttura.
    Una bassa Shannon Entropy indica concentrazione in pochi bin → maggiore struttura.

    Args:
        arr:  Array di valori (es. log-return)
        bins: Numero di bin per la discretizzazione

    Returns:
        Valore di Shannon Entropy in [0, log2(bins)], o np.nan se l'array è vuoto
    """
    counts, _ = np.histogram(arr, bins=bins)
    total = counts.sum()
    if total == 0:
        return np.nan
    probs = counts[counts > 0] / total
    return float(scipy_entropy(probs, base=2))


def shannon_entropy_series(
    series: pd.Series,
    window: int = SHANNON_WINDOW,
    bins:   int = SHANNON_BINS,
) -> pd.Series:
    """
    Shannon Entropy rolling su una serie temporale.

    Args:
        series: Serie temporale (es. log-return)
        window: Lunghezza finestra rolling
        bins:   Numero di bin per la discretizzazione

    Returns:
        Serie con Shannon Entropy rolling (NaN per i primi window-1 valori)
    """
    return series.rolling(window).apply(
        lambda x: _shannon_entropy_window(x, bins=bins),
        raw=True,
    )


def _permutation_entropy_window(arr: np.ndarray, order: int = PE_ORDER) -> float:
    """
    Permutation Entropy normalizzata (Bandt & Pompe, 2002).

    Conta la frequenza dei pattern ordinali di lunghezza `order` e calcola:
      H_perm = -Σ p_i · log2(p_i) / log2(order!)   [normalizzata 0-1]

    PE ≈ 1 → alta complessità (mercato vicino al random walk)
    PE << 1 → struttura temporale forte (potenziale prevedibilità)

    Args:
        arr:   Array di valori
        order: Embedding dimension (consigliato 3-6)

    Returns:
        Permutation Entropy normalizzata in [0, 1]
    """
    n = len(arr)
    if n < order:
        return np.nan

    all_patterns = list(itertools.permutations(range(order)))
    counts: dict = {p: 0 for p in all_patterns}

    for i in range(n - order + 1):
        pattern = tuple(np.argsort(arr[i: i + order], kind="stable"))
        counts[pattern] = counts.get(pattern, 0) + 1

    total = sum(counts.values())
    if total == 0:
        return np.nan

    probs = np.array([c / total for c in counts.values() if c > 0])
    h     = -np.sum(probs * np.log2(probs))
    h_max = log2(factorial(order))

    return float(h / h_max) if h_max > 0 else np.nan


def permutation_entropy_series(
    series: pd.Series,
    order:  int = PE_ORDER,
    window: int = PE_WINDOW,
) -> pd.Series:
    """
    Permutation Entropy normalizzata rolling su una serie temporale.

    Args:
        series: Serie temporale (tipicamente log-return)
        order:  Embedding dimension m
        window: Lunghezza finestra rolling

    Returns:
        Serie con PE normalizzata in [0, 1]
    """
    return series.rolling(window).apply(
        lambda x: _permutation_entropy_window(x, order=order),
        raw=True,
    )


# ================================================================
# BUILD — funzione principale per uso nell'app
# ================================================================

def compute_entropy_features(spx_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcola tutte le feature di entropia sull'indice SPX.

    Pipeline:
      1. Log-return giornalieri da adjusted_close
      2. Shannon Entropy rolling (63g) sui log-return
      3. Permutation Entropy normalizzata rolling (63g) sui log-return
      4. Percentile expanding di Shannon Entropy (Studio 5 storico)
      5. Soglie regime (tertili storici di shannon_ret)

    Args:
        spx_df: DataFrame SPX con colonna 'adjusted_close'

    Returns:
        DataFrame con colonne aggiuntive:
          - log_ret        : log-return giornaliero
          - shannon_ret    : Shannon Entropy rolling 63g
          - perm_entropy   : Permutation Entropy rolling 63g (normalizzata)
          - shannon_pctile : percentile expanding di shannon_ret (0-100)
        Righe con NaN nelle colonne entropia rimossi (warm-up iniziale).
    """
    price_col = "adjusted_close" if "adjusted_close" in spx_df.columns else "close"
    close     = spx_df[price_col].dropna()

    # 1. Log-return
    log_ret = np.log(close / close.shift(1))

    # 2. Shannon Entropy rolling
    sh_ret = shannon_entropy_series(log_ret, window=SHANNON_WINDOW, bins=SHANNON_BINS)

    # 3. Permutation Entropy rolling
    pe = permutation_entropy_series(log_ret, order=PE_ORDER, window=PE_WINDOW)

    # 4. Percentile expanding (per la dashboard storica)
    sh_pctile = sh_ret.expanding().rank(pct=True) * 100

    feat = pd.DataFrame({
        "close":         close,
        "log_ret":       log_ret,
        "shannon_ret":   sh_ret,
        "perm_entropy":  pe,
        "shannon_pctile": sh_pctile,
    })

    # Rimuovi il warm-up iniziale (finestre incomplete)
    feat = feat.dropna(subset=["shannon_ret", "perm_entropy"])

    return feat


def compute_entropy_thresholds(entropy_feat: pd.DataFrame) -> dict:
    """
    Calcola le soglie di regime per l'entropia (tertili storici).

    I tertili suddividono la distribuzione storica di shannon_ret in:
      - Regime BASSA   : shannon_ret <= p33
      - Regime MEDIA   : p33 < shannon_ret <= p67
      - Regime ALTA    : shannon_ret > p67

    Args:
        entropy_feat: DataFrame da compute_entropy_features()

    Returns:
        Dizionario con:
          - p33_shannon  (float): soglia bassa/media
          - p67_shannon  (float): soglia media/alta
          - p33_pe       (float): soglia PE bassa/media
          - p67_pe       (float): soglia PE media/alta
    """
    return {
        "p33_shannon": float(entropy_feat["shannon_ret"].quantile(REGIME_P_LOW)),
        "p67_shannon": float(entropy_feat["shannon_ret"].quantile(REGIME_P_HIGH)),
        "p33_pe":      float(entropy_feat["perm_entropy"].quantile(REGIME_P_LOW)),
        "p67_pe":      float(entropy_feat["perm_entropy"].quantile(REGIME_P_HIGH)),
    }


def assign_entropy_regime(shannon_val: float, thresholds: dict) -> str:
    """
    Assegna il regime di entropia a un singolo valore di Shannon Entropy.

    Args:
        shannon_val: Valore corrente di Shannon Entropy
        thresholds:  Dizionario da compute_entropy_thresholds()

    Returns:
        Stringa: 'Bassa', 'Media' o 'Alta'
    """
    if np.isnan(shannon_val):
        return "N/D"
    if shannon_val <= thresholds["p33_shannon"]:
        return "Bassa"
    if shannon_val <= thresholds["p67_shannon"]:
        return "Media"
    return "Alta"


def assign_entropy_regime_series(
    entropy_feat:  pd.DataFrame,
    thresholds:    dict,
) -> pd.Series:
    """
    Assegna il regime di entropia a tutta la serie storica.

    Args:
        entropy_feat: DataFrame da compute_entropy_features()
        thresholds:   Dizionario da compute_entropy_thresholds()

    Returns:
        pd.Series con valori 'Bassa', 'Media', 'Alta' e lo stesso indice di entropy_feat
    """
    def _classify(val: float) -> str:
        if np.isnan(val):
            return "N/D"
        if val <= thresholds["p33_shannon"]:
            return "Bassa"
        if val <= thresholds["p67_shannon"]:
            return "Media"
        return "Alta"

    return entropy_feat["shannon_ret"].apply(_classify)
