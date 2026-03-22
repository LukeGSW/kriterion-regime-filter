"""
exposure_engine.py
==================
Applica le regole di esposizione ottimizzate per costruire equity curve adjusted.

Per ogni Trade System, ricostruisce due equity curve parallele:
  1. Equity BASELINE  : PnL cumulato con moltiplicatore 1.0 (nessun filtro)
  2. Equity ADJUSTED  : PnL cumulato con il moltiplicatore ottimale per regime

Il confronto diretto tra le due curve mostra l'impatto del filtro regime.

Le curve sono costruite a frequenza giornaliera (una barra per ogni giorno
di calendario) assegnando il PnL di ogni trade alla sua data di uscita (exit_date).
Trade multipli nella stessa giornata vengono sommati.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .regime_engine import map_trades_to_regimes, ALL_REGIMES
from .equity_loader import EXPOSURE_LABELS, EXPOSURE_COLORS


# ================================================================
# EQUITY CURVE
# ================================================================

def build_equity_curves(
    ts_data:       dict,
    regime_series: pd.Series,
    exposure_rules: dict[str, float],
) -> pd.DataFrame:
    """
    Costruisce le equity curve baseline e adjusted per un Trading System.

    Logica:
      - La moltiplicazione viene applicata al PnL di ogni trade in base al
        regime SPX alla DATA DI INGRESSO del trade (entry_date), per rispettare
        la causalità: il filtro regime decide SE e QUANTO entrare nel trade.
      - Il PnL adjusted del trade = pnl × multiplier(regime_at_entry)
      - Le curve sono cumulate per exit_date

    Args:
        ts_data:        Dizionario del TS (contiene 'trades')
        regime_series:  Serie storica dei regimi
        exposure_rules: {regime: multiplier} da optimize_ts_exposure()

    Returns:
        DataFrame con colonne:
          - baseline_pnl_daily   : PnL giornaliero baseline (per visualizzazione bar)
          - adjusted_pnl_daily   : PnL giornaliero adjusted
          - baseline_equity      : equity cumulata baseline
          - adjusted_equity      : equity cumulata adjusted
          - exposure_level       : moltiplicatore applicato quel giorno
          - regime               : etichetta regime quel giorno
        Index: DatetimeIndex (daily, solo giorni con trade in uscita)
    """
    trades = ts_data["trades"].copy()
    if trades.empty:
        return pd.DataFrame()

    # 1. Associa ogni trade al regime all'entry_date
    trades["regime"]     = map_trades_to_regimes(trades, regime_series)
    trades["multiplier"] = trades["regime"].map(
        lambda r: exposure_rules.get(r, 1.0)
    )
    trades["pnl_baseline"] = trades["pnl"]
    trades["pnl_adjusted"] = trades["pnl"] * trades["multiplier"]

    # 2. Aggrega per exit_date (più trade in uscita lo stesso giorno vengono sommati)
    daily = (
        trades
        .groupby("exit_date")
        .agg(
            baseline_pnl_daily=("pnl_baseline", "sum"),
            adjusted_pnl_daily=("pnl_adjusted", "sum"),
            # Per l'esposizione del giorno usiamo la media ponderata dei moltiplicatori
            # (handle multipli trade con moltiplicatori diversi nella stessa giornata)
            exposure_level=("multiplier", "mean"),
            regime=("regime", "first"),  # regime del primo trade del giorno
        )
        .sort_index()
    )

    # 3. Equity cumulata
    daily["baseline_equity"] = daily["baseline_pnl_daily"].cumsum()
    daily["adjusted_equity"] = daily["adjusted_pnl_daily"].cumsum()

    return daily


def build_all_equity_curves(
    trading_systems: dict,
    regime_series:   pd.Series,
    opt_results:     dict[str, dict],
) -> dict[str, pd.DataFrame]:
    """
    Costruisce le equity curve per tutti i Trading System.

    Args:
        trading_systems: Dizionario da load_all_trading_systems()
        regime_series:   Serie storica dei regimi
        opt_results:     Dizionario risultati ottimizzazione da optimize_all_ts()

    Returns:
        Dizionario {ts_name: equity_curve_df} per ogni TS
    """
    curves: dict[str, pd.DataFrame] = {}
    for ts_name, ts_data in trading_systems.items():
        if ts_name not in opt_results:
            continue
        exposure_rules = opt_results[ts_name].get("exposure_rules", {})
        curve = build_equity_curves(ts_data, regime_series, exposure_rules)
        if not curve.empty:
            curves[ts_name] = curve
    return curves


# ================================================================
# STATISTICHE COMPARATE
# ================================================================

def compute_performance_comparison(equity_df: pd.DataFrame) -> dict:
    """
    Calcola le metriche di performance comparative baseline vs adjusted.

    Args:
        equity_df: DataFrame da build_equity_curves()

    Returns:
        Dizionario con metriche per 'baseline' e 'adjusted':
          - total_pnl    (float): PnL totale cumulato
          - n_active_days (int):  giorni con almeno un trade in uscita
          - max_drawdown (float): massimo drawdown (valore negativo)
          - sharpe_like  (float): mean_daily / std_daily × √n
          - improvement  (float): % miglioramento adjusted vs baseline
    """
    if equity_df.empty:
        return {}

    def _max_drawdown(equity: pd.Series) -> float:
        """Calcola il max drawdown della serie equity."""
        roll_max = equity.cummax()
        drawdown = equity - roll_max
        return float(drawdown.min())

    def _sharpe_like(daily_pnl: pd.Series) -> float:
        """Sharpe-like ratio annualizzato (basato su trading days)."""
        if daily_pnl.std() == 0 or len(daily_pnl) < 2:
            return np.nan
        return float(daily_pnl.mean() / daily_pnl.std() * np.sqrt(252))

    baseline = {
        "total_pnl":     float(equity_df["baseline_equity"].iloc[-1]),
        "n_active_days": len(equity_df),
        "max_drawdown":  _max_drawdown(equity_df["baseline_equity"]),
        "sharpe_like":   _sharpe_like(equity_df["baseline_pnl_daily"]),
    }

    adjusted = {
        "total_pnl":     float(equity_df["adjusted_equity"].iloc[-1]),
        "n_active_days": len(equity_df),
        "max_drawdown":  _max_drawdown(equity_df["adjusted_equity"]),
        "sharpe_like":   _sharpe_like(equity_df["adjusted_pnl_daily"]),
    }

    # Miglioramento percentuale del PnL totale
    if baseline["total_pnl"] != 0:
        improvement = (adjusted["total_pnl"] - baseline["total_pnl"]) / abs(baseline["total_pnl"]) * 100
    else:
        improvement = 0.0

    return {
        "baseline":    baseline,
        "adjusted":    adjusted,
        "improvement": improvement,
    }


# ================================================================
# SERIE STORICA ESPOSIZIONE (per il grafico lineare)
# ================================================================

def build_exposure_history(
    equity_df:     pd.DataFrame,
    regime_series: pd.Series,
    start_date:    pd.Timestamp,
    end_date:      pd.Timestamp,
) -> pd.Series:
    """
    Costruisce la serie storica dell'esposizione su base GIORNALIERA.

    Riempe i giorni senza trade usando il regime SPX del giorno stesso
    e il moltiplicatore associato (forward-fill per weekend/festivi).
    Questo permette di mostrare sul grafico anche i periodi in cui il TS
    non aveva trade aperti ma il filtro avrebbe comunque inhibito/boostato.

    Args:
        equity_df:     DataFrame da build_equity_curves()
        regime_series: Serie storica dei regimi
        start_date:    Prima data della serie
        end_date:      Ultima data della serie

    Returns:
        pd.Series con l'exposure_level per ogni giorno del calendario,
        indice DatetimeIndex, valori float (0.0 / 0.5 / 1.0 / 1.5)
    """
    if equity_df.empty:
        return pd.Series(dtype=float)

    # Serie giornaliera dall'intera finestra del regime
    daily_index = pd.date_range(start=start_date, end=end_date, freq="B")

    # Usa l'exposure_level dall'equity_df dove disponibile
    exposure = equity_df["exposure_level"].reindex(daily_index)

    # Per i giorni senza trade, usa il regime SPX del giorno stesso
    # (forward-fill per gestire weekend/festivi)
    regime_reindexed = regime_series.reindex(daily_index, method="ffill")

    # Nota: qui lasciamo i giorni senza trade come NaN nell'exposure
    # (il grafico li mostrerà come linea piatta all'ultimo valore)
    exposure = exposure.fillna(method="ffill")

    return exposure
