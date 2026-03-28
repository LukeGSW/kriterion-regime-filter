"""
wf_optimizer.py
===============
Walk-Forward Optimizer per la validazione out-of-sample delle regole di esposizione.

PROBLEMA CHE RISOLVE
--------------------
Il modulo optimizer.py calibra i moltiplicatori di esposizione (0×/0.5×/1×/1.5×)
sull'INTERO storico dei trade → i moltiplicatori sono ottimizzati in-sample su dati
che il filtro ha già "visto". Il risultato è un'equity adjusted artificialmente gonfiata.

SOLUZIONE: WALK-FORWARD ANCHORED (finestra espandente)
------------------------------------------------------
Per ogni trade i, i moltiplicatori applicati sono calibrati SOLO sui trade 0..i-1.
Questo replica esattamente la condizione operativa reale: al momento dell'entrata
nel trade i, si conoscono solo i trade precedenti.

Dettagli algoritmici:
  - Finestra espandente (anchored): si parte dal trade 0 e si aggiunge un trade
    alla volta. Questo è più stabile di una finestra rolling perché non "dimentica"
    la storia.
  - Ricalibrazione ogni WF_RETRAIN_EVERY trade: invece di ricalcolare ad ogni
    singolo trade (computazionalmente ridondante), si ricalibrino le regole ogni
    N trade. Le regole restano invariate tra una ricalibrazione e la successiva.
  - Cold start: per i primi WF_TRAIN_MIN_TRADES trade non si dispone di abbastanza
    storia per calibrare → moltiplicatore default = 1.0 (STANDARD, nessun filtro).

COMPATIBILITÀ CON IL CODICE ESISTENTE
--------------------------------------
Questo modulo riutilizza INTEGRALMENTE:
  - optimizer.optimize_ts_exposure() per la calibrazione su sottoinsieme
  - vix_modulator.optimize_ts_vix_exposure() per le regole VIX
  - vix_modulator.compute_final_multiplier() per il moltiplicatore combinato
  - regime_engine.map_trades_to_regimes() per l'assegnazione regime
  - vix_modulator.map_trades_to_vix_states() per l'assegnazione stato VIX

Non modifica nessun modulo esistente.

OUTPUT
------
Per ogni TS vengono prodotte 3 equity curve confrontabili:
  1. Baseline     : PnL grezzo senza alcun filtro (moltiplicatore fisso 1×)
  2. WF Regime    : filtro entropia+ergodicità walk-forward (out-of-sample)
  3. WF Combinato : filtro regime × VIX walk-forward (se VIX disponibile)

Usato da pages/2_Walk_Forward.py per la visualizzazione Streamlit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .optimizer import optimize_ts_exposure, BOOST_RATIO, STANDARD_RATIO
from .equity_loader import MIN_TRADES_PER_REGIME
from .regime_engine import map_trades_to_regimes


# ================================================================
# COSTANTI WALK-FORWARD
# ================================================================

# Numero minimo di trade storici necessari prima di iniziare la calibrazione.
# Con meno trade, il filtro non ha abbastanza campioni per stimare i regimi
# in modo affidabile → si usa moltiplicatore neutro 1.0.
WF_TRAIN_MIN_TRADES: int = 30

# Frequenza di ricalibrazione: si ricalcolano le regole ogni N trade.
# Valore basso → adattamento rapido ai cambiamenti di regime strutturale.
# Valore alto → regole più stabili ma meno reattive.
# 15 trade è un buon compromesso per sistemi con 150-600 trade totali.
WF_RETRAIN_EVERY: int = 15


# ================================================================
# FUNZIONE PRINCIPALE PER UN SINGOLO TS
# ================================================================

def run_wf_optimization(
    ts_data: dict,
    regime_series: pd.Series,
    vix_features: "pd.DataFrame | None" = None,
    min_trades: int = MIN_TRADES_PER_REGIME,
    boost_ratio: float = BOOST_RATIO,
    standard_ratio: float = STANDARD_RATIO,
    train_min_trades: int = WF_TRAIN_MIN_TRADES,
    retrain_every: int = WF_RETRAIN_EVERY,
) -> dict:
    """
    Esegue il walk-forward optimization per un singolo Trading System.

    Per ogni trade i:
      1. Se i < train_min_trades → moltiplicatore = 1.0 (cold start)
      2. Se i == train_min_trades oppure i % retrain_every == 0:
           ricalibra le regole sui trade 0..i-1 (anchored)
      3. Applica le regole correnti al trade i → PnL walk-forward

    Args:
        ts_data: Dizionario del TS da load_all_trading_systems().
                 Deve contenere: 'trades' (DataFrame), 'is_short' (bool).
        regime_series: Serie storica dei regimi da build_regime_series().
        vix_features: DataFrame da compute_vix_features() (opzionale).
                      Se None, viene prodotta solo la curva WF Regime.
        min_trades: Soglia minima trade per regime per assegnare un moltiplicatore
                    non-default. Passato a optimize_ts_exposure().
        boost_ratio: Soglia BOOST (default 1.40×). Passato a optimize_ts_exposure().
        standard_ratio: Soglia STANDARD (default 0.30×). Passato a optimize_ts_exposure().
        train_min_trades: Numero minimo trade per avviare la calibrazione.
        retrain_every: Frequenza di ricalibrazione in numero di trade.

    Returns:
        Dizionario con:
        - wf_equity_df (pd.DataFrame): equity curve giornaliera con colonne:
            baseline_pnl, wf_regime_pnl, [wf_combined_pnl],
            baseline_equity, wf_regime_equity, [wf_combined_equity],
            wf_regime_mult, [wf_combined_mult]
          Index: DatetimeIndex (exit_date)
        - performance (dict): metriche comparative per baseline / wf_regime / [wf_combined]
        - train_min_trades (int): parametro usato
        - retrain_every (int): parametro usato
        - n_retraining_points (int): numero di ricalibrazione effettuate
        - has_vix (bool): se il layer VIX è stato incluso
        - final_regime_rules (dict): regole regime calibrate su TUTTO lo storico
          (da usare come regole operative correnti)
        - final_vix_rules (dict): regole VIX calibrate su tutto lo storico
        - error (str): presente solo se il calcolo non è stato possibile
    """
    # Import condizionale per evitare dipendenza circolare se vix non è presente
    try:
        from .vix_modulator import (
            optimize_ts_vix_exposure,
            map_trades_to_vix_states,
            compute_final_multiplier,
        )
        _vix_available = True
    except ImportError:
        _vix_available = False

    trades = ts_data["trades"].copy().reset_index(drop=True)
    n = len(trades)

    if n < train_min_trades:
        return {
            "error": (
                f"Trade insufficienti per il walk-forward ({n} < {train_min_trades} minimo). "
                "Aumenta lo storico del backtest o riduci WF_TRAIN_MIN_TRADES."
            ),
            "has_vix": False,
        }

    # ── Determina se il layer VIX è utilizzabile ────────────────
    has_vix = (
        _vix_available
        and vix_features is not None
        and not vix_features.empty
    )

    # ── Pre-calcola regime e stato VIX per TUTTI i trade ────────
    # Questo è causale: regime(t) è calcolato da dati SPX fino a t
    # (la regime_series stessa è una finestra rolling sul passato).
    trades["_regime"] = map_trades_to_regimes(trades, regime_series)
    if has_vix:
        trades["_vix_state"] = map_trades_to_vix_states(trades, vix_features)

    # ── Array risultati ─────────────────────────────────────────
    wf_regime_mult = np.ones(n, dtype=float)
    wf_vix_mult    = np.ones(n, dtype=float)

    current_regime_rules: dict[str, float] = {}
    current_vix_rules:    dict[str, float] = {}

    # ── Loop walk-forward ────────────────────────────────────────
    for i in range(n):

        # Punto di ricalibrazione: primo punto utile o ogni retrain_every
        is_retrain_point = (
            i == train_min_trades
            or (i > train_min_trades and (i - train_min_trades) % retrain_every == 0)
        )

        if i >= train_min_trades and is_retrain_point:
            # Sottoinsieme di training: solo i trade 0..i-1 (out-of-sample puro)
            train_subset: dict = {
                "trades":   trades.iloc[:i].copy(),
                "is_short": ts_data["is_short"],
            }

            # ── Calibra regole regime ────────────────────────────
            regime_result = optimize_ts_exposure(
                ts_data       = train_subset,
                regime_series = regime_series,
                min_trades    = min_trades,
                boost_ratio   = boost_ratio,
                standard_ratio= standard_ratio,
            )
            current_regime_rules = regime_result.get("exposure_rules", {})

            # ── Calibra regole VIX ───────────────────────────────
            if has_vix:
                vix_result = optimize_ts_vix_exposure(
                    ts_data        = train_subset,
                    vix_features   = vix_features,
                    min_trades     = min_trades,
                    boost_ratio    = boost_ratio,
                    standard_ratio = standard_ratio,
                )
                current_vix_rules = vix_result.get("vix_rules", {})

        # ── Applica regole correnti al trade i ───────────────────
        if i < train_min_trades or not current_regime_rules:
            # Cold start: nessun filtro
            wf_regime_mult[i] = 1.0
        else:
            regime_label = trades.iloc[i]["_regime"]
            wf_regime_mult[i] = current_regime_rules.get(regime_label, 1.0)

        if has_vix:
            if i < train_min_trades or not current_vix_rules:
                wf_vix_mult[i] = 1.0
            else:
                vix_state = trades.iloc[i]["_vix_state"]
                wf_vix_mult[i] = current_vix_rules.get(vix_state, 1.0)

    # ── Calcola PnL walk-forward ─────────────────────────────────
    trades["wf_regime_mult"]    = wf_regime_mult
    trades["wf_vix_mult"]       = wf_vix_mult
    trades["pnl_baseline"]      = trades["pnl"]
    trades["pnl_wf_regime"]     = trades["pnl"] * trades["wf_regime_mult"]

    if has_vix:
        trades["wf_combined_mult"] = trades.apply(
            lambda r: compute_final_multiplier(r["wf_regime_mult"], r["wf_vix_mult"]),
            axis=1,
        )
        trades["pnl_wf_combined"] = trades["pnl"] * trades["wf_combined_mult"]

    # ── Aggrega per exit_date (più trade nello stesso giorno) ────
    agg_dict: dict = {
        "baseline_pnl":   ("pnl_baseline",   "sum"),
        "wf_regime_pnl":  ("pnl_wf_regime",  "sum"),
        "wf_regime_mult": ("wf_regime_mult",  "mean"),
    }
    if has_vix:
        agg_dict["wf_combined_pnl"]  = ("pnl_wf_combined",  "sum")
        agg_dict["wf_combined_mult"] = ("wf_combined_mult",  "mean")

    daily = (
        trades
        .groupby("exit_date")
        .agg(**agg_dict)
        .sort_index()
    )

    # ── Equity cumulate ──────────────────────────────────────────
    daily["baseline_equity"]  = daily["baseline_pnl"].cumsum()
    daily["wf_regime_equity"] = daily["wf_regime_pnl"].cumsum()
    if has_vix:
        daily["wf_combined_equity"] = daily["wf_combined_pnl"].cumsum()

    # ── Metriche di performance ──────────────────────────────────
    perf = _compute_wf_performance(daily, has_vix)

    # ── Regole finali (calibrate su TUTTO lo storico) ────────────
    # Sono le regole da usare operativamente. Corrispondono all'output
    # del modulo optimizer.py standard (in-sample completo).
    final_regime_result = optimize_ts_exposure(
        ts_data        = ts_data,
        regime_series  = regime_series,
        min_trades     = min_trades,
        boost_ratio    = boost_ratio,
        standard_ratio = standard_ratio,
    )
    final_regime_rules = final_regime_result.get("exposure_rules", {})

    final_vix_rules: dict = {}
    if has_vix:
        final_vix_result = optimize_ts_vix_exposure(
            ts_data        = ts_data,
            vix_features   = vix_features,
            min_trades     = min_trades,
            boost_ratio    = boost_ratio,
            standard_ratio = standard_ratio,
        )
        final_vix_rules = final_vix_result.get("vix_rules", {})

    n_retraining = max(0, (n - train_min_trades) // retrain_every)

    return {
        "wf_equity_df":       daily,
        "performance":        perf,
        "train_min_trades":   train_min_trades,
        "retrain_every":      retrain_every,
        "n_retraining_points": n_retraining,
        "has_vix":            has_vix,
        "final_regime_rules": final_regime_rules,
        "final_vix_rules":    final_vix_rules,
        "n_trades_total":     n,
    }


# ================================================================
# BATCH SU TUTTI I TRADING SYSTEM
# ================================================================

def run_all_ts_wf(
    trading_systems: dict,
    regime_series: pd.Series,
    vix_features: "pd.DataFrame | None" = None,
    min_trades: int = MIN_TRADES_PER_REGIME,
    boost_ratio: float = BOOST_RATIO,
    standard_ratio: float = STANDARD_RATIO,
    train_min_trades: int = WF_TRAIN_MIN_TRADES,
    retrain_every: int = WF_RETRAIN_EVERY,
) -> dict[str, dict]:
    """
    Esegue il walk-forward optimization su tutti i Trading System.

    Wrapper di run_wf_optimization() per uso batch.

    Args:
        trading_systems: Dizionario da load_all_trading_systems().
        regime_series: Serie storica dei regimi da build_regime_series().
        vix_features: DataFrame da compute_vix_features() (opzionale).
        min_trades, boost_ratio, standard_ratio: Parametri ottimizzatore.
        train_min_trades: Soglia cold start.
        retrain_every: Frequenza di ricalibrazione.

    Returns:
        Dizionario {ts_name: risultato_run_wf_optimization()} per ogni TS.
        I TS con errori hanno la chiave 'error' nel risultato.
    """
    results: dict[str, dict] = {}

    for ts_name, ts_data in trading_systems.items():
        results[ts_name] = run_wf_optimization(
            ts_data        = ts_data,
            regime_series  = regime_series,
            vix_features   = vix_features,
            min_trades     = min_trades,
            boost_ratio    = boost_ratio,
            standard_ratio = standard_ratio,
            train_min_trades = train_min_trades,
            retrain_every  = retrain_every,
        )

    return results


# ================================================================
# HELPER PRIVATO: METRICHE DI PERFORMANCE
# ================================================================

def _compute_wf_performance(daily: pd.DataFrame, has_vix: bool) -> dict:
    """
    Calcola le metriche di performance comparative per le 3 equity curve.

    Args:
        daily: DataFrame da run_wf_optimization() aggregato per exit_date.
        has_vix: Se True, include anche le metriche della curva WF Combined.

    Returns:
        Dizionario con sezioni 'baseline', 'wf_regime', ['wf_combined'],
        ognuna con: total_pnl, max_drawdown, sharpe, improvement_pct.
    """

    def _max_drawdown(equity: pd.Series) -> float:
        """Massimo drawdown (valore negativo, in USD)."""
        dd = equity - equity.cummax()
        return float(dd.min())

    def _sharpe_like(pnl: pd.Series) -> float:
        """Sharpe-like annualizzato (trading days)."""
        if len(pnl) < 2 or pnl.std() == 0.0:
            return float("nan")
        return float(pnl.mean() / pnl.std() * np.sqrt(252))

    base_total = float(daily["baseline_equity"].iloc[-1])

    def _improvement(total: float) -> float:
        if base_total == 0.0:
            return 0.0
        return (total - base_total) / abs(base_total) * 100.0

    baseline_stats = {
        "total_pnl":     base_total,
        "max_drawdown":  _max_drawdown(daily["baseline_equity"]),
        "sharpe":        _sharpe_like(daily["baseline_pnl"]),
        "improvement_pct": 0.0,
    }

    wf_regime_total = float(daily["wf_regime_equity"].iloc[-1])
    wf_regime_stats = {
        "total_pnl":      wf_regime_total,
        "max_drawdown":   _max_drawdown(daily["wf_regime_equity"]),
        "sharpe":         _sharpe_like(daily["wf_regime_pnl"]),
        "improvement_pct": _improvement(wf_regime_total),
    }

    result = {
        "baseline":  baseline_stats,
        "wf_regime": wf_regime_stats,
    }

    if has_vix:
        wf_comb_total = float(daily["wf_combined_equity"].iloc[-1])
        result["wf_combined"] = {
            "total_pnl":      wf_comb_total,
            "max_drawdown":   _max_drawdown(daily["wf_combined_equity"]),
            "sharpe":         _sharpe_like(daily["wf_combined_pnl"]),
            "improvement_pct": _improvement(wf_comb_total),
        }

    return result
