"""
regime_engine.py
================
Combina le metriche di entropia ed ergodicità in etichette di regime di mercato.

Il regime di mercato è una combinazione bidimensionale:
  - Asse Entropia (3 stati): Bassa / Media / Alta
  - Asse Ergodicità (2 stati): Ergodico / Non Ergodico
  → 6 regimi totali (es. "Bassa|Ergodico", "Alta|Non Ergodico", ...)

Questo modulo produce la serie storica dei regimi sull'SPX e il regime corrente,
usato poi dall'optimizer.py per identificare in quale contesto ogni TS ha
meglio o peggio performato storicamente.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .entropy_calc import (
    compute_entropy_features,
    compute_entropy_thresholds,
    assign_entropy_regime_series,
    assign_entropy_regime,
)
from .ergodicity_calc import (
    compute_ergodicity_features,
    compute_ergodicity_thresholds,
    assign_ergodicity_state_series,
    assign_ergodicity_state,
)


# ================================================================
# COSTANTI
# ================================================================

ENTROPY_STATES   = ["Bassa", "Media", "Alta"]
ERGODICITY_STATES = ["Ergodico", "Non Ergodico"]

# Tutti i 6 regimi possibili (ordinati per uso nelle tabelle)
ALL_REGIMES: list[str] = [
    f"{e}|{g}"
    for e in ENTROPY_STATES
    for g in ERGODICITY_STATES
]

# Colori per ogni regime (per la dashboard)
REGIME_COLORS: dict[str, str] = {
    "Bassa|Ergodico":       "#1B5E20",  # verde scuro — condizioni ideali
    "Bassa|Non Ergodico":   "#388E3C",  # verde chiaro
    "Media|Ergodico":       "#1565C0",  # blu scuro — condizioni neutre
    "Media|Non Ergodico":   "#1976D2",  # blu chiaro
    "Alta|Ergodico":        "#E65100",  # arancio scuro — attenzione
    "Alta|Non Ergodico":    "#B71C1C",  # rosso — alta incertezza
}

REGIME_DESCRIPTIONS: dict[str, str] = {
    "Bassa|Ergodico":
        "Mercato ordinato e prevedibile. Entropia bassa indica struttura "
        "nei rendimenti; ergodicità confermata indica che il regime storico è stabile.",
    "Bassa|Non Ergodico":
        "Struttura nei rendimenti ma regime storico in transizione. "
        "Il mercato mostra pattern ma la media temporale diverge da quella storica.",
    "Media|Ergodico":
        "Condizioni di mercato nella norma. Entropia nella fascia centrale, "
        "ergodicità rispettata. Regime neutro.",
    "Media|Non Ergodico":
        "Entropia nella norma ma segnale di non ergodicità. Possibile "
        "inizio di fase di transizione strutturale.",
    "Alta|Ergodico":
        "Alta imprevedibilità nei rendimenti ma media storica stabile. "
        "Volatilità elevata senza breakout strutturale.",
    "Alta|Non Ergodico":
        "Regime di massima incertezza. Alta entropia + non ergodicità "
        "indicano un mercato strutturalmente anomalo rispetto alla storia.",
}


# ================================================================
# FUNZIONE PRINCIPALE
# ================================================================

def build_regime_series(spx_df: pd.DataFrame) -> dict:
    """
    Costruisce la serie storica dei regimi di mercato sull'SPX.

    Esegue in sequenza:
      1. Calcolo feature entropia (Shannon + Permutation Entropy)
      2. Calcolo feature ergodicità (SEM)
      3. Allineamento temporale dei due DataFrame (inner join sulle date)
      4. Assegnazione etichette regime per ogni giorno
      5. Calcolo del regime corrente (ultima candela disponibile)

    Args:
        spx_df: DataFrame SPX da fetch_spx() con 'adjusted_close'

    Returns:
        Dizionario con:
          - regime_series (pd.Series):  serie storica etichette regime
                                         indice DatetimeIndex, valori "Ent|Erg"
          - entropy_feat  (pd.DataFrame): feature entropia complete
          - erg_feat      (pd.DataFrame): feature ergodicità complete
          - entropy_thresh (dict):        soglie entropia (p33, p67)
          - erg_thresh     (dict):        soglie ergodicità (threshold, sigma)
          - current_regime (str):         regime dell'ultima candela disponibile
          - current_entropy_state (str):  'Bassa' / 'Media' / 'Alta'
          - current_erg_state     (str):  'Ergodico' / 'Non Ergodico'
          - last_date      (pd.Timestamp): data ultima candela
    """
    # 1. Feature entropia
    entropy_feat   = compute_entropy_features(spx_df)
    entropy_thresh = compute_entropy_thresholds(entropy_feat)

    # 2. Feature ergodicità
    erg_feat    = compute_ergodicity_features(spx_df)
    erg_thresh  = compute_ergodicity_thresholds(erg_feat)
    threshold   = erg_thresh["threshold"]

    # 3. Allineamento temporale (inner join sulle date comuni)
    entropy_regimes = assign_entropy_regime_series(entropy_feat, entropy_thresh)
    erg_states      = assign_ergodicity_state_series(erg_feat)

    # Crea un DataFrame allineato per data
    aligned = pd.DataFrame({
        "entropy_regime": entropy_regimes,
        "erg_state":      erg_states,
    }).dropna()

    # 4. Etichetta regime composita: "Bassa|Ergodico"
    regime_series = (
        aligned["entropy_regime"].astype(str)
        + "|"
        + aligned["erg_state"].astype(str)
    )

    # 5. Regime corrente (ultima data disponibile nell'allineamento)
    last_date = aligned.index[-1]
    last_row  = aligned.iloc[-1]

    current_entropy_state = str(last_row["entropy_regime"])
    current_erg_state     = str(last_row["erg_state"])
    current_regime        = f"{current_entropy_state}|{current_erg_state}"

    return {
        "regime_series":          regime_series,
        "entropy_feat":           entropy_feat,
        "erg_feat":               erg_feat,
        "entropy_thresh":         entropy_thresh,
        "erg_thresh":             erg_thresh,
        "current_regime":         current_regime,
        "current_entropy_state":  current_entropy_state,
        "current_erg_state":      current_erg_state,
        "last_date":              last_date,
    }


def get_regime_for_date(
    regime_series: pd.Series,
    target_date:   pd.Timestamp,
) -> str:
    """
    Restituisce il regime per una data specifica.

    Usa la data più recente disponibile non superiore a target_date
    (forward-fill con lookback max 5 giorni per gestire weekend/festivi).

    Args:
        regime_series: Serie storica dei regimi da build_regime_series()
        target_date:   Data di riferimento

    Returns:
        Etichetta regime (es. 'Bassa|Ergodico') o 'N/D' se non disponibile
    """
    if regime_series.empty:
        return "N/D"

    # Cerca la data più vicina non posteriore a target_date
    available = regime_series[regime_series.index <= target_date]

    if available.empty:
        return "N/D"

    return str(available.iloc[-1])


def map_trades_to_regimes(
    trades:        pd.DataFrame,
    regime_series: pd.Series,
) -> pd.Series:
    """
    Associa ogni trade al regime SPX vigente alla sua data di ingresso.

    Usa il regime alla data di entry del trade (o il giorno precedente più
    recente disponibile) per rispettare la causalità: la decisione di entrare
    in un trade è presa sulla base delle condizioni di mercato al momento
    dell'ingresso, non dell'uscita.

    Args:
        trades:        DataFrame dei trade con colonna 'entry_date'
        regime_series: Serie storica dei regimi da build_regime_series()

    Returns:
        pd.Series con il regime per ogni trade (stesso indice di trades)
    """
    return trades["entry_date"].apply(
        lambda d: get_regime_for_date(regime_series, d)
    )
