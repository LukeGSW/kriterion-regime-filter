"""
optimizer.py
============
Ottimizzazione automatica delle regole di esposizione per ogni Trading System.

Algoritmo:
  Per ogni TS, per ogni regime (6 combinazioni entropia × ergodicità):
    1. Filtra i trade entrati in quel regime
    2. Calcola mean_pnl e sharpe_like ratio
    3. Confronta con la performance media globale del TS
    4. Assegna un moltiplicatore di esposizione (0 / 0.5 / 1.0 / 1.5)

Logica di assegnazione moltiplicatori:
  - Se mean_pnl_regime >= BOOST_RATIO × mean_pnl_overall   → BOOST  (×1.5)
  - Se mean_pnl_regime >= STANDARD_RATIO × mean_pnl_overall → STANDARD (×1.0)
  - Se mean_pnl_regime >= 0                                  → REDUCE (×0.5)
  - Se mean_pnl_regime < 0                                   → INHIBIT (×0.0)

Per regimi con meno di MIN_TRADES trade, il moltiplicatore default è 1.0 (STANDARD)
per evitare regole basate su campioni troppo piccoli.

Nota per i sistemi short: la logica è identica. Il PnL è già il netto del sistema,
quindi un PnL positivo in un regime è già un segnale buono indipendentemente dalla
direzione del sistema.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .regime_engine import ALL_REGIMES, map_trades_to_regimes
from .equity_loader import (
    MIN_TRADES_PER_REGIME,
    EXPOSURE_LABELS,
    EXPOSURE_COLORS,
    EXPOSURE_EMOJIS,
)


# ================================================================
# COSTANTI OTTIMIZZAZIONE
# ================================================================

# Rapporto mean_regime / mean_overall per assegnare BOOST
BOOST_RATIO:    float = 1.40  # performance regime ≥ 140% della media → BOOST

# Rapporto per demarcazione STANDARD (sopra) vs REDUCE (sotto)
STANDARD_RATIO: float = 0.30  # performance regime ≥ 30% della media → STANDARD

# Soglia assoluta: se mean_overall ≤ 0, usiamo valori assoluti
ABSOLUTE_BOOST_THRESHOLD:    float = 50.0   # USD/trade per regime → BOOST
ABSOLUTE_STANDARD_THRESHOLD: float = 0.0    # USD/trade per regime → STANDARD


# ================================================================
# CALCOLO STATISTICHE PER REGIME
# ================================================================

def compute_regime_stats(
    trades:        pd.DataFrame,
    regime_series: pd.Series,
) -> pd.DataFrame:
    """
    Calcola le statistiche di performance per ogni regime (6 combinazioni).

    Args:
        trades:        DataFrame dei trade con colonne 'entry_date', 'pnl'
        regime_series: Serie storica dei regimi da build_regime_series()

    Returns:
        DataFrame con colonne:
          - regime       (str):   etichetta regime (es. 'Bassa|Ergodico')
          - n_trades     (int):   numero di trade in quel regime
          - mean_pnl     (float): PnL medio per trade
          - total_pnl    (float): PnL totale cumulato
          - win_rate     (float): % trade vincenti (0-100)
          - sharpe_like  (float): mean_pnl / std_pnl × √n_trades
                                  (misura di qualità risk-adjusted)
          - std_pnl      (float): deviazione standard del PnL
        Index: regime
    """
    # Associa ogni trade al regime SPX all'entry_date
    trade_regimes = map_trades_to_regimes(trades, regime_series)
    trades_with_regime = trades.copy()
    trades_with_regime["regime"] = trade_regimes

    rows = []
    for regime in ALL_REGIMES:
        mask   = trades_with_regime["regime"] == regime
        subset = trades_with_regime.loc[mask, "pnl"]
        n      = len(subset)

        if n == 0:
            rows.append({
                "regime":      regime,
                "n_trades":    0,
                "mean_pnl":    np.nan,
                "total_pnl":   0.0,
                "win_rate":    np.nan,
                "sharpe_like": np.nan,
                "std_pnl":     np.nan,
            })
            continue

        mean_pnl  = float(subset.mean())
        total_pnl = float(subset.sum())
        win_rate  = float((subset > 0).sum() / n * 100)
        std_pnl   = float(subset.std()) if n > 1 else 0.0

        # Sharpe-like = mean / std × √n (misura di significatività statistica)
        sharpe_like = (mean_pnl / std_pnl * np.sqrt(n)) if std_pnl > 0 else np.nan

        rows.append({
            "regime":      regime,
            "n_trades":    n,
            "mean_pnl":    mean_pnl,
            "total_pnl":   total_pnl,
            "win_rate":    win_rate,
            "sharpe_like": sharpe_like,
            "std_pnl":     std_pnl,
        })

    df = pd.DataFrame(rows).set_index("regime")
    return df


# ================================================================
# ASSEGNAZIONE MOLTIPLICATORE DI ESPOSIZIONE
# ================================================================

def assign_exposure_multiplier(
    mean_pnl_regime:  float,
    mean_pnl_overall: float,
    n_trades_regime:  int,
    min_trades:       int = MIN_TRADES_PER_REGIME,
    boost_ratio:      float = BOOST_RATIO,
    standard_ratio:   float = STANDARD_RATIO,
) -> float:
    """
    Assegna il moltiplicatore di esposizione per un singolo regime.

    Logica (in ordine di priorità):
      1. Se n_trades_regime < min_trades → STANDARD (1.0) per prudenza
      2. Se mean_pnl_regime è NaN → STANDARD (1.0)
      3. Se mean_pnl_overall > 0 (sistema overall profittevole):
           - Regime ≥ boost_ratio × overall   → BOOST (1.5)
           - Regime ≥ standard_ratio × overall → STANDARD (1.0)
           - Regime ≥ 0                         → REDUCE (0.5)
           - Regime < 0                          → INHIBIT (0.0)
      4. Se mean_pnl_overall ≤ 0 (sistema overall in perdita):
           Usiamo soglie assolute per selezionare i regimi migliori:
           - Regime ≥ ABSOLUTE_BOOST_THRESHOLD    → BOOST (1.5)
           - Regime ≥ ABSOLUTE_STANDARD_THRESHOLD  → STANDARD (1.0)
           - Regime < ABSOLUTE_STANDARD_THRESHOLD  → REDUCE (0.5)
           - Regime < 0                             → INHIBIT (0.0)

    Args:
        mean_pnl_regime:  PnL medio del TS in questo regime
        mean_pnl_overall: PnL medio del TS su tutti i trade
        n_trades_regime:  Numero di trade in questo regime
        min_trades:       Soglia minima di trade per regole non-standard
        boost_ratio:      Rapporto soglia BOOST/overall
        standard_ratio:   Rapporto soglia STANDARD/overall

    Returns:
        Moltiplicatore: 0.0 / 0.5 / 1.0 / 1.5
    """
    # Insufficienza di dati → default STANDARD
    if n_trades_regime < min_trades:
        return 1.0

    if np.isnan(mean_pnl_regime):
        return 1.0

    if mean_pnl_overall > 0:
        # Sistema mediamente profittevole: soglie relative
        if mean_pnl_regime >= boost_ratio * mean_pnl_overall:
            return 1.5   # BOOST
        if mean_pnl_regime >= standard_ratio * mean_pnl_overall:
            return 1.0   # STANDARD
        if mean_pnl_regime >= 0:
            return 0.5   # REDUCE
        return 0.0       # INHIBIT
    else:
        # Sistema mediamente in perdita: soglie assolute
        if mean_pnl_regime >= ABSOLUTE_BOOST_THRESHOLD:
            return 1.5   # BOOST
        if mean_pnl_regime >= ABSOLUTE_STANDARD_THRESHOLD:
            return 1.0   # STANDARD
        if mean_pnl_regime >= 0:
            return 0.5   # REDUCE
        return 0.0       # INHIBIT


# ================================================================
# OTTIMIZZAZIONE COMPLETA PER UN TS
# ================================================================

def optimize_ts_exposure(
    ts_data:       dict,
    regime_series: pd.Series,
    min_trades:    int   = MIN_TRADES_PER_REGIME,
    boost_ratio:   float = BOOST_RATIO,
    standard_ratio: float = STANDARD_RATIO,
) -> dict:
    """
    Esegue l'ottimizzazione completa per un singolo Trading System.

    Args:
        ts_data:       Dizionario del TS da load_all_trading_systems()
                       (contiene 'trades', 'is_short', etc.)
        regime_series: Serie storica dei regimi da build_regime_series()
        min_trades:    Trade minimi per assegnare regole non-standard
        boost_ratio:   Soglia BOOST relativa
        standard_ratio: Soglia STANDARD relativa

    Returns:
        Dizionario con:
          - regime_stats     (pd.DataFrame): statistiche per regime
          - exposure_rules   (dict):         {regime: multiplier}
          - overall_mean_pnl (float):        PnL medio globale del TS
          - n_trades_total   (int):          trade totali
          - optimization_info (dict):        parametri usati nell'ottimizzazione
    """
    trades = ts_data["trades"]

    if trades.empty:
        return {
            "regime_stats":     pd.DataFrame(),
            "exposure_rules":   {r: 1.0 for r in ALL_REGIMES},
            "overall_mean_pnl": 0.0,
            "n_trades_total":   0,
            "optimization_info": {},
        }

    # Calcola statistiche per regime
    regime_stats     = compute_regime_stats(trades, regime_series)
    overall_mean_pnl = float(trades["pnl"].mean())
    n_trades_total   = len(trades)

    # Assegna moltiplicatore per ogni regime
    exposure_rules: dict[str, float] = {}
    for regime in ALL_REGIMES:
        if regime not in regime_stats.index:
            exposure_rules[regime] = 1.0
            continue

        row = regime_stats.loc[regime]
        mult = assign_exposure_multiplier(
            mean_pnl_regime  = float(row["mean_pnl"]) if not np.isnan(row["mean_pnl"]) else np.nan,
            mean_pnl_overall = overall_mean_pnl,
            n_trades_regime  = int(row["n_trades"]),
            min_trades       = min_trades,
            boost_ratio      = boost_ratio,
            standard_ratio   = standard_ratio,
        )
        exposure_rules[regime] = mult

    return {
        "regime_stats":     regime_stats,
        "exposure_rules":   exposure_rules,
        "overall_mean_pnl": overall_mean_pnl,
        "n_trades_total":   n_trades_total,
        "optimization_info": {
            "min_trades":    min_trades,
            "boost_ratio":   boost_ratio,
            "standard_ratio": standard_ratio,
            "is_short":      ts_data.get("is_short", False),
        },
    }


def optimize_all_ts(
    trading_systems: dict,
    regime_series:   pd.Series,
    min_trades:      int   = MIN_TRADES_PER_REGIME,
    boost_ratio:     float = BOOST_RATIO,
    standard_ratio:  float = STANDARD_RATIO,
) -> dict[str, dict]:
    """
    Esegue l'ottimizzazione per tutti i Trading System.

    Args:
        trading_systems: Dizionario da load_all_trading_systems()
        regime_series:   Serie storica dei regimi da build_regime_series()
        min_trades:      Trade minimi per regole non-standard
        boost_ratio:     Soglia BOOST relativa
        standard_ratio:  Soglia STANDARD relativa

    Returns:
        Dizionario {ts_name: optimization_result} per ogni TS
    """
    results: dict[str, dict] = {}
    for ts_name, ts_data in trading_systems.items():
        results[ts_name] = optimize_ts_exposure(
            ts_data        = ts_data,
            regime_series  = regime_series,
            min_trades     = min_trades,
            boost_ratio    = boost_ratio,
            standard_ratio = standard_ratio,
        )
    return results


def get_current_exposure(
    ts_name:       str,
    opt_results:   dict[str, dict],
    current_regime: str,
) -> dict:
    """
    Restituisce lo stato di esposizione corrente per un Trading System.

    Args:
        ts_name:        Nome del TS
        opt_results:    Risultati ottimizzazione da optimize_all_ts()
        current_regime: Regime corrente da build_regime_series()

    Returns:
        Dizionario con:
          - multiplier (float): 0.0 / 0.5 / 1.0 / 1.5
          - label      (str):   'INIBITO' / 'RIDOTTO' / 'STANDARD' / 'BOOST'
          - color      (str):   hex color
          - emoji      (str):   emoji
          - regime     (str):   regime corrente
    """
    if ts_name not in opt_results:
        mult = 1.0
    else:
        rules = opt_results[ts_name].get("exposure_rules", {})
        mult  = rules.get(current_regime, 1.0)

    return {
        "multiplier": mult,
        "label":      EXPOSURE_LABELS.get(mult, "STANDARD"),
        "color":      EXPOSURE_COLORS.get(mult, "#2196F3"),
        "emoji":      EXPOSURE_EMOJIS.get(mult, "🟡"),
        "regime":     current_regime,
    }
