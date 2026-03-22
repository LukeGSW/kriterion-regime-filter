"""
vix_modulator.py
================
VIX come terzo layer di regime — indipendente da Entropia ed Ergodicità.

Il VIX (CBOE Volatility Index) misura la volatilità implicita delle opzioni sull'S&P 500
a 30 giorni. A differenza di Shannon Entropy (storica/statistica) ed Ergodicità (strutturale),
il VIX cattura la "paura forward-looking" del mercato delle opzioni.

Architettura:
  1. Fetch dati VIX storici da EODHD (ticker VIX.INDX)
  2. Calcolo rolling percentile (finestra 252g) → posizione relativa 0–100
  3. Applicazione isteresi → stato discreto LOW_VIX / NORMAL_VIX / HIGH_VIX
  4. Ottimizzazione per TS: stesso meccanismo dell'ottimizzazione di regime
     → vix_rules: {vix_state: multiplier} per ogni TS
  5. Moltiplicatore finale combinato:
     combined = snap(regime_mult × vix_mult)  ∈ {0.0, 0.5, 1.0, 1.5}

Isteresi anti-whipsaw:
  - HIGH_VIX : si entra quando percentile > 80, si esce quando < 65
  - LOW_VIX  : si entra quando percentile < 20, si esce quando > 35
  - NORMAL_VIX: zona intermedia / default

La transizione LOW → HIGH o HIGH → LOW passa obbligatoriamente per NORMAL,
evitando jump diretti che creerebbero segnali instabili.

Nota sui sistemi trend following azionari:
  I sistemi trend following su azioni non mostrano sensibilità a entropia/ergodicità
  ma possono rispondere significativamente al VIX, che misura la "fear" e la
  direzionalità implicita del mercato azionario.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import requests
from datetime import date

from .equity_loader import MIN_TRADES_PER_REGIME, EXPOSURE_LABELS, EXPOSURE_COLORS, EXPOSURE_EMOJIS


# ================================================================
# COSTANTI
# ================================================================

VIX_TICKER             = "VIX.INDX"
VIX_PERCENTILE_WINDOW  = 252    # finestra rolling per il percentile (1 anno trading)

# Soglie isteresi per HIGH_VIX
HIGH_VIX_ENTRY_PCT = 80.0   # entra in HIGH_VIX quando percentile > 80
HIGH_VIX_EXIT_PCT  = 65.0   # esce da HIGH_VIX quando percentile < 65

# Soglie isteresi per LOW_VIX
LOW_VIX_ENTRY_PCT  = 20.0   # entra in LOW_VIX quando percentile < 20
LOW_VIX_EXIT_PCT   = 35.0   # esce da LOW_VIX quando percentile > 35

# Nomi stati VIX (usati come chiavi nei dict di esposizione)
VIX_STATES = ["LOW_VIX", "NORMAL_VIX", "HIGH_VIX"]

VIX_STATE_LABELS: dict[str, str] = {
    "LOW_VIX":    "Bassa Vol. (VIX)",
    "NORMAL_VIX": "Vol. Normale (VIX)",
    "HIGH_VIX":   "Alta Vol. (VIX)",
}

VIX_STATE_COLORS: dict[str, str] = {
    "LOW_VIX":    "#1565C0",   # blu — bassa fear / complacency
    "NORMAL_VIX": "#2E7D32",   # verde — condizioni normali
    "HIGH_VIX":   "#B71C1C",   # rosso — alta fear / stress
}

VIX_STATE_EMOJIS: dict[str, str] = {
    "LOW_VIX":    "🔵",
    "NORMAL_VIX": "🟢",
    "HIGH_VIX":   "🔴",
}

# Soglie snap per moltiplicatore combinato finale
# midpoint tra i 4 valori standard: 0 / 0.5 / 1.0 / 1.5
_SNAP_THRESHOLDS = [0.25, 0.75, 1.25]
_SNAP_VALUES     = [0.0,  0.5,  1.0,  1.5]


# ================================================================
# FETCH VIX DA EODHD
# ================================================================

def fetch_vix(api_key: str, to_date: str | None = None) -> pd.DataFrame:
    """
    Scarica l'intera storia disponibile del VIX da EODHD.

    Usa il ticker VIX.INDX. La storia disponibile copre tipicamente
    dal 1990 a oggi. Nessun `from_date` per ottenere la serie completa.

    Args:
        api_key: Chiave API EODHD
        to_date: Data finale 'YYYY-MM-DD' (default: oggi)

    Returns:
        DataFrame con colonne: open, high, low, close, adjusted_close
        Index: DatetimeIndex ordinato dal più vecchio al più recente

    Raises:
        requests.HTTPError: Se la chiave API non è valida
        ValueError: Se il JSON restituito è vuoto
    """
    if to_date is None:
        to_date = str(date.today())

    url = (
        f"https://eodhd.com/api/eod/{VIX_TICKER}"
        f"?to={to_date}"
        f"&period=d&api_token={api_key}&fmt=json"
    )

    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    df.sort_index(inplace=True)

    cols = ["open", "high", "low", "close", "adjusted_close"]
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    available_cols = [c for c in cols if c in df.columns]
    return df[available_cols]


# ================================================================
# PERCENTILE ROLLING
# ================================================================

def compute_vix_percentile_series(
    vix_df: pd.DataFrame,
    window: int = VIX_PERCENTILE_WINDOW,
) -> pd.Series:
    """
    Calcola il rolling percentile del VIX close rispetto alla finestra di `window` giorni.

    Il percentile misura DOVE si trova il VIX attuale rispetto al suo range storico
    recente: 0 = minimo storico recente, 100 = massimo storico recente.

    Implementazione: per ogni barra, conta quante osservazioni nella finestra
    sono INFERIORI al valore corrente → percentile in [0, 100].

    Args:
        vix_df: DataFrame con colonna 'adjusted_close' (da fetch_vix)
        window: Finestra rolling in giorni di trading (default: 252 = 1 anno)

    Returns:
        pd.Series con percentile rolling 0–100, stesso indice di vix_df
    """
    price_col = "adjusted_close" if "adjusted_close" in vix_df.columns else "close"
    vix_close = vix_df[price_col].dropna()

    # Percentile basato su rank all'interno della finestra
    # raw=True: più veloce, x è array numpy con l'intera finestra
    pct = vix_close.rolling(
        window=window,
        min_periods=max(20, window // 4),
    ).apply(
        lambda x: float((x[:-1] < x[-1]).sum()) / (len(x) - 1) * 100
        if len(x) > 1 else 50.0,
        raw=True,
    )

    pct.name = "vix_pct"
    return pct


# ================================================================
# ISTERESI ANTI-WHIPSAW
# ================================================================

def apply_vix_hysteresis(
    pct_series:       pd.Series,
    high_entry:       float = HIGH_VIX_ENTRY_PCT,
    high_exit:        float = HIGH_VIX_EXIT_PCT,
    low_entry:        float = LOW_VIX_ENTRY_PCT,
    low_exit:         float = LOW_VIX_EXIT_PCT,
    initial_state:    str   = "NORMAL_VIX",
) -> pd.Series:
    """
    Applica l'isteresi alla serie di percentili VIX per ottenere stati discreti
    stabili (LOW_VIX / NORMAL_VIX / HIGH_VIX) senza transizioni rapide sul confine.

    Logica di transizione:
      - Da NORMAL_VIX:
          Se pct > high_entry  → HIGH_VIX
          Se pct < low_entry   → LOW_VIX
      - Da HIGH_VIX:
          Se pct < high_exit   → NORMAL_VIX  (non può andare direttamente a LOW)
      - Da LOW_VIX:
          Se pct > low_exit    → NORMAL_VIX  (non può andare direttamente a HIGH)

    Il vincolo di passare per NORMAL_VIX previene jump HIGH↔LOW che creerebbero
    esposizioni instabili.

    Args:
        pct_series:    Serie rolling percentile da compute_vix_percentile_series()
        high_entry:    Percentile entrata HIGH_VIX (default: 80)
        high_exit:     Percentile uscita HIGH_VIX (default: 65)
        low_entry:     Percentile entrata LOW_VIX (default: 20)
        low_exit:      Percentile uscita LOW_VIX (default: 35)
        initial_state: Stato di partenza (default: 'NORMAL_VIX')

    Returns:
        pd.Series con stati 'LOW_VIX' / 'NORMAL_VIX' / 'HIGH_VIX',
        stesso indice di pct_series
    """
    states: list[str | float] = []
    current = initial_state

    for pct in pct_series:
        if pd.isna(pct):
            states.append(np.nan)
            continue

        # Transizioni con isteresi
        if current == "NORMAL_VIX":
            if pct > high_entry:
                current = "HIGH_VIX"
            elif pct < low_entry:
                current = "LOW_VIX"
        elif current == "HIGH_VIX":
            if pct < high_exit:
                current = "NORMAL_VIX"
            # Non può saltare direttamente a LOW_VIX
        elif current == "LOW_VIX":
            if pct > low_exit:
                current = "NORMAL_VIX"
            # Non può saltare direttamente a HIGH_VIX

        states.append(current)

    result = pd.Series(states, index=pct_series.index, name="vix_state")
    return result


# ================================================================
# FUNZIONE PRINCIPALE — CALCOLO FEATURE VIX
# ================================================================

def compute_vix_features(
    vix_df: pd.DataFrame,
    window: int = VIX_PERCENTILE_WINDOW,
) -> pd.DataFrame:
    """
    Calcola le feature complete del VIX: close, percentile rolling e stato isteresi.

    Args:
        vix_df: DataFrame VIX da fetch_vix()
        window: Finestra rolling per il percentile

    Returns:
        DataFrame con colonne:
          - vix_close (float): prezzo close VIX
          - vix_pct   (float): rolling percentile 0–100
          - vix_state (str):   stato con isteresi ('LOW_VIX' / 'NORMAL_VIX' / 'HIGH_VIX')
        Index: DatetimeIndex (solo righe con vix_pct non NaN)
    """
    if vix_df.empty:
        return pd.DataFrame(columns=["vix_close", "vix_pct", "vix_state"])

    price_col = "adjusted_close" if "adjusted_close" in vix_df.columns else "close"
    vix_close = vix_df[price_col].rename("vix_close")

    vix_pct   = compute_vix_percentile_series(vix_df, window=window)

    # Applica isteresi solo sulle righe con percentile valido
    vix_pct_valid = vix_pct.dropna()
    vix_state = apply_vix_hysteresis(vix_pct_valid)

    feat = pd.DataFrame({
        "vix_close": vix_close,
        "vix_pct":   vix_pct,
        "vix_state": vix_state,
    })

    # Mantieni solo le righe dove il percentile è disponibile
    feat = feat.dropna(subset=["vix_pct"])
    return feat


def get_current_vix_info(vix_features: pd.DataFrame) -> dict:
    """
    Restituisce le informazioni VIX correnti (ultima riga disponibile).

    Args:
        vix_features: DataFrame da compute_vix_features()

    Returns:
        Dizionario con:
          - state     (str):   'LOW_VIX' / 'NORMAL_VIX' / 'HIGH_VIX'
          - vix_close (float): VIX close corrente
          - vix_pct   (float): percentile rolling corrente
          - label     (str):   etichetta leggibile
          - color     (str):   hex color
          - emoji     (str):   emoji
    """
    if vix_features.empty:
        return {
            "state":     "NORMAL_VIX",
            "vix_close": np.nan,
            "vix_pct":   50.0,
            "label":     VIX_STATE_LABELS["NORMAL_VIX"],
            "color":     VIX_STATE_COLORS["NORMAL_VIX"],
            "emoji":     VIX_STATE_EMOJIS["NORMAL_VIX"],
        }

    last = vix_features.iloc[-1]
    state = str(last["vix_state"]) if not pd.isna(last["vix_state"]) else "NORMAL_VIX"

    return {
        "state":     state,
        "vix_close": float(last["vix_close"]) if not pd.isna(last["vix_close"]) else np.nan,
        "vix_pct":   float(last["vix_pct"])   if not pd.isna(last["vix_pct"])   else 50.0,
        "label":     VIX_STATE_LABELS.get(state, "N/D"),
        "color":     VIX_STATE_COLORS.get(state, "#9E9E9E"),
        "emoji":     VIX_STATE_EMOJIS.get(state, "⚪"),
    }


# ================================================================
# LOOKUP STATO VIX PER DATA
# ================================================================

def get_vix_state_for_date(
    vix_features: pd.DataFrame,
    target_date:  pd.Timestamp,
) -> str:
    """
    Restituisce lo stato VIX per una data specifica.

    Usa il valore più recente non posteriore a target_date
    (forward-fill fino a 5 giorni per gestire weekend/festivi).

    Args:
        vix_features: DataFrame da compute_vix_features()
        target_date:  Data di riferimento

    Returns:
        Stato VIX ('LOW_VIX' / 'NORMAL_VIX' / 'HIGH_VIX') o 'NORMAL_VIX' se non disponibile
    """
    if vix_features.empty:
        return "NORMAL_VIX"

    available = vix_features[vix_features.index <= target_date]
    if available.empty:
        return "NORMAL_VIX"

    state = available["vix_state"].iloc[-1]
    if pd.isna(state):
        return "NORMAL_VIX"

    return str(state)


def map_trades_to_vix_states(
    trades:       pd.DataFrame,
    vix_features: pd.DataFrame,
) -> pd.Series:
    """
    Associa ogni trade allo stato VIX alla sua data di ingresso.

    Usa la causalità: lo stato VIX all'entry_date determina la regola di esposizione.

    Args:
        trades:       DataFrame con colonna 'entry_date'
        vix_features: DataFrame da compute_vix_features()

    Returns:
        pd.Series con lo stato VIX per ogni trade (stesso indice di trades)
    """
    return trades["entry_date"].apply(
        lambda d: get_vix_state_for_date(vix_features, d)
    )


# ================================================================
# STATISTICHE PER STATO VIX (per ottimizzazione per TS)
# ================================================================

def compute_vix_stats(
    trades:       pd.DataFrame,
    vix_features: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calcola le statistiche di performance per ogni stato VIX (LOW/NORMAL/HIGH).

    Identica logica a compute_regime_stats() in optimizer.py, ma per VIX state.

    Args:
        trades:       DataFrame con colonne 'entry_date', 'pnl'
        vix_features: DataFrame da compute_vix_features()

    Returns:
        DataFrame con colonne n_trades, mean_pnl, total_pnl, win_rate, sharpe_like, std_pnl
        Index: vix_state
    """
    if trades.empty or vix_features.empty:
        rows = [
            {
                "vix_state":   s,
                "n_trades":    0,
                "mean_pnl":    np.nan,
                "total_pnl":   0.0,
                "win_rate":    np.nan,
                "sharpe_like": np.nan,
                "std_pnl":     np.nan,
            }
            for s in VIX_STATES
        ]
        return pd.DataFrame(rows).set_index("vix_state")

    t = trades.copy()
    t["vix_state"] = map_trades_to_vix_states(t, vix_features)

    rows = []
    for state in VIX_STATES:
        subset = t.loc[t["vix_state"] == state, "pnl"]
        n = len(subset)

        if n == 0:
            rows.append({
                "vix_state":   state,
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
        sharpe_like = (mean_pnl / std_pnl * np.sqrt(n)) if std_pnl > 0 else np.nan

        rows.append({
            "vix_state":   state,
            "n_trades":    n,
            "mean_pnl":    mean_pnl,
            "total_pnl":   total_pnl,
            "win_rate":    win_rate,
            "sharpe_like": sharpe_like,
            "std_pnl":     std_pnl,
        })

    return pd.DataFrame(rows).set_index("vix_state")


# ================================================================
# OTTIMIZZAZIONE VIX PER TRADING SYSTEM
# ================================================================

def assign_vix_exposure(
    mean_pnl_vix:     float,
    mean_pnl_overall: float,
    n_trades_vix:     int,
    min_trades:       int   = MIN_TRADES_PER_REGIME,
    boost_ratio:      float = 1.40,
    standard_ratio:   float = 0.30,
) -> float:
    """
    Assegna il moltiplicatore VIX per un singolo stato (identica logica a
    assign_exposure_multiplier in optimizer.py).

    Args:
        mean_pnl_vix:     PnL medio del TS in questo stato VIX
        mean_pnl_overall: PnL medio globale del TS
        n_trades_vix:     Numero di trade in questo stato VIX
        min_trades:       Soglia minima trade per regole non-STANDARD
        boost_ratio:      Soglia BOOST
        standard_ratio:   Soglia STANDARD

    Returns:
        Moltiplicatore: 0.0 / 0.5 / 1.0 / 1.5
    """
    if n_trades_vix < min_trades or np.isnan(mean_pnl_vix):
        return 1.0  # STANDARD per dati insufficienti

    if mean_pnl_overall > 0:
        if mean_pnl_vix >= boost_ratio * mean_pnl_overall:
            return 1.5
        if mean_pnl_vix >= standard_ratio * mean_pnl_overall:
            return 1.0
        if mean_pnl_vix >= 0:
            return 0.5
        return 0.0
    else:
        # Sistema overall in perdita: soglie assolute
        if mean_pnl_vix >= 50.0:
            return 1.5
        if mean_pnl_vix >= 0.0:
            return 1.0
        if mean_pnl_vix >= 0:
            return 0.5
        return 0.0


def optimize_ts_vix_exposure(
    ts_data:       dict,
    vix_features:  pd.DataFrame,
    min_trades:    int   = MIN_TRADES_PER_REGIME,
    boost_ratio:   float = 1.40,
    standard_ratio: float = 0.30,
) -> dict:
    """
    Ottimizzazione VIX per un singolo Trading System.

    Args:
        ts_data:       Dizionario del TS da load_all_trading_systems()
        vix_features:  DataFrame da compute_vix_features()
        min_trades:    Trade minimi per regole non-STANDARD
        boost_ratio:   Soglia BOOST
        standard_ratio: Soglia STANDARD

    Returns:
        Dizionario con:
          - vix_stats     (pd.DataFrame): statistiche per stato VIX
          - vix_rules     (dict):         {vix_state: multiplier}
          - overall_mean_pnl (float):     PnL medio globale
    """
    trades = ts_data.get("trades", pd.DataFrame())

    if trades.empty or vix_features.empty:
        return {
            "vix_stats":       pd.DataFrame(),
            "vix_rules":       {s: 1.0 for s in VIX_STATES},
            "overall_mean_pnl": 0.0,
        }

    vix_stats        = compute_vix_stats(trades, vix_features)
    overall_mean_pnl = float(trades["pnl"].mean())

    vix_rules: dict[str, float] = {}
    for state in VIX_STATES:
        if state not in vix_stats.index:
            vix_rules[state] = 1.0
            continue

        row = vix_stats.loc[state]
        mult = assign_vix_exposure(
            mean_pnl_vix     = float(row["mean_pnl"]) if not np.isnan(row["mean_pnl"]) else np.nan,
            mean_pnl_overall = overall_mean_pnl,
            n_trades_vix     = int(row["n_trades"]),
            min_trades       = min_trades,
            boost_ratio      = boost_ratio,
            standard_ratio   = standard_ratio,
        )
        vix_rules[state] = mult

    return {
        "vix_stats":        vix_stats,
        "vix_rules":        vix_rules,
        "overall_mean_pnl": overall_mean_pnl,
    }


def optimize_all_ts_vix(
    trading_systems: dict,
    vix_features:    pd.DataFrame,
    min_trades:      int   = MIN_TRADES_PER_REGIME,
    boost_ratio:     float = 1.40,
    standard_ratio:  float = 0.30,
) -> dict[str, dict]:
    """
    Ottimizzazione VIX per tutti i Trading System.

    Args:
        trading_systems: da load_all_trading_systems()
        vix_features:    da compute_vix_features()
        min_trades:      Trade minimi per regole non-STANDARD
        boost_ratio:     Soglia BOOST
        standard_ratio:  Soglia STANDARD

    Returns:
        Dizionario {ts_name: {vix_stats, vix_rules, overall_mean_pnl}}
    """
    results: dict[str, dict] = {}
    for ts_name, ts_data in trading_systems.items():
        results[ts_name] = optimize_ts_vix_exposure(
            ts_data        = ts_data,
            vix_features   = vix_features,
            min_trades     = min_trades,
            boost_ratio    = boost_ratio,
            standard_ratio = standard_ratio,
        )
    return results


# ================================================================
# MOLTIPLICATORE FINALE COMBINATO
# ================================================================

def snap_to_standard_level(value: float) -> float:
    """
    Arrotonda il moltiplicatore combinato al livello standard più vicino.

    Livelli standard: 0.0 / 0.5 / 1.0 / 1.5
    Soglie di snap (midpoint tra livelli adiacenti): 0.25 / 0.75 / 1.25

    Esempi:
      - regime=BOOST(1.5) × vix=REDUCE(0.5) = 0.75 → STANDARD(1.0)
      - regime=STANDARD(1.0) × vix=REDUCE(0.5) = 0.50 → REDUCE(0.5)
      - regime=BOOST(1.5) × vix=INHIBIT(0.0) = 0.0 → INHIBIT(0.0)
      - regime=BOOST(1.5) × vix=BOOST(1.5) = 2.25 → clamp → BOOST(1.5)

    Args:
        value: Prodotto raw (regime_mult × vix_mult)

    Returns:
        Moltiplicatore standard: 0.0 / 0.5 / 1.0 / 1.5
    """
    # Clamp a [0, 1.5] prima dello snap
    v = float(np.clip(value, 0.0, 1.5))

    if v < _SNAP_THRESHOLDS[0]:
        return _SNAP_VALUES[0]   # 0.0 — INHIBIT
    elif v < _SNAP_THRESHOLDS[1]:
        return _SNAP_VALUES[1]   # 0.5 — REDUCE
    elif v < _SNAP_THRESHOLDS[2]:
        return _SNAP_VALUES[2]   # 1.0 — STANDARD
    else:
        return _SNAP_VALUES[3]   # 1.5 — BOOST


def compute_final_multiplier(
    regime_mult: float,
    vix_mult:    float,
) -> float:
    """
    Calcola il moltiplicatore finale combinato (Regime × VIX), snappato al livello standard.

    Args:
        regime_mult: Moltiplicatore da ottimizzazione entropia+ergodicità (0/0.5/1.0/1.5)
        vix_mult:    Moltiplicatore da ottimizzazione VIX (0/0.5/1.0/1.5)

    Returns:
        Moltiplicatore finale standard: 0.0 / 0.5 / 1.0 / 1.5
    """
    return snap_to_standard_level(regime_mult * vix_mult)


def get_combined_exposure(
    ts_name:         str,
    opt_results:     dict[str, dict],
    vix_opt_results: dict[str, dict],
    current_regime:  str,
    current_vix_state: str,
) -> dict:
    """
    Restituisce lo stato di esposizione COMBINATO (regime + VIX) per un TS.

    Args:
        ts_name:           Nome del TS
        opt_results:       Risultati ottimizzazione regime da optimize_all_ts()
        vix_opt_results:   Risultati ottimizzazione VIX da optimize_all_ts_vix()
        current_regime:    Regime corrente (es. 'Bassa|Ergodico')
        current_vix_state: Stato VIX corrente (es. 'HIGH_VIX')

    Returns:
        Dizionario con:
          - regime_mult   (float): moltiplicatore dal regime
          - vix_mult      (float): moltiplicatore dal VIX
          - final_mult    (float): moltiplicatore finale combinato
          - label         (str):   etichetta finale
          - color         (str):   hex color
          - emoji         (str):   emoji
          - regime        (str):   regime corrente
          - vix_state     (str):   stato VIX corrente
    """
    # Moltiplicatore regime (entropia + ergodicità)
    regime_mult = 1.0
    if ts_name in opt_results:
        rules = opt_results[ts_name].get("exposure_rules", {})
        regime_mult = rules.get(current_regime, 1.0)

    # Moltiplicatore VIX
    vix_mult = 1.0
    if ts_name in vix_opt_results:
        vix_rules = vix_opt_results[ts_name].get("vix_rules", {})
        vix_mult = vix_rules.get(current_vix_state, 1.0)

    final_mult = compute_final_multiplier(regime_mult, vix_mult)

    return {
        "regime_mult":  regime_mult,
        "vix_mult":     vix_mult,
        "final_mult":   final_mult,
        "multiplier":   final_mult,                              # alias per backward compat
        "label":        EXPOSURE_LABELS.get(final_mult, "STANDARD"),
        "color":        EXPOSURE_COLORS.get(final_mult, "#2196F3"),
        "emoji":        EXPOSURE_EMOJIS.get(final_mult, "🟡"),
        "regime":       current_regime,
        "vix_state":    current_vix_state,
    }
