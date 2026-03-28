"""
regime_state_store.py
=====================
Persistenza dello stato di regime per il rilevamento dei cambiamenti tra report.

PROBLEMA CHE RISOLVE
--------------------
notify.py invia ogni giorno lo stesso formato di report indipendentemente da
se il regime è cambiato o meno. Dal punto di vista operativo, ciò che conta
è QUANDO il regime cambia — quel cambiamento triggera una modifica all'esposizione
dei sistemi live.

SOLUZIONE
---------
Questo modulo salva su disco (file JSON) lo stato dell'ultimo report inviato:
  - current_regime   : etichetta regime composita (es. "Media|Ergodico")
  - current_vix_state: stato VIX (es. "NORMAL_VIX", "HIGH_VIX", "LOW_VIX")
  - report_date      : data ISO del report

Al successivo avvio di notify.py:
  1. Si carica lo stato precedente dal file JSON
  2. Si confronta con lo stato corrente appena calcolato
  3. Se c'è un cambio, il report Telegram lo evidenzia con un blocco dedicato

PERCORSO FILE DI STATO
-----------------------
Il file .regime_state.json viene creato nella root del progetto
(stessa directory di notify.py). Aggiungerlo a .gitignore è consigliato
perché è un artefatto runtime, non parte del codice sorgente.

COMPATIBILITÀ
-------------
Questo è un modulo completamente nuovo. Non modifica nessun modulo esistente.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Optional


# ================================================================
# PERCORSO FILE DI STATO
# ================================================================

# Il file viene posizionato nella root del progetto (un livello sopra src/)
_MODULE_DIR  = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_MODULE_DIR)
STATE_FILEPATH = os.path.join(_PROJECT_DIR, ".regime_state.json")


# ================================================================
# LETTURA STATO PRECEDENTE
# ================================================================

def load_previous_state() -> Optional[dict]:
    """
    Carica lo stato del regime dall'ultimo report inviato.

    Returns:
        Dizionario con le chiavi:
          - current_regime (str)  : etichetta regime precedente
          - current_vix_state (str): stato VIX precedente
          - entropy_state (str)   : stato entropia precedente
          - erg_state (str)       : stato ergodicità precedente
          - report_date (str)     : data ISO del report precedente
        oppure None se il file non esiste o non è leggibile
        (prima esecuzione o file corrotto).
    """
    if not os.path.exists(STATE_FILEPATH):
        return None

    try:
        with open(STATE_FILEPATH, "r", encoding="utf-8") as f:
            state = json.load(f)
        # Validazione minima: le chiavi essenziali devono essere presenti
        if "current_regime" in state and "current_vix_state" in state:
            return state
        return None
    except (json.JSONDecodeError, OSError):
        return None


# ================================================================
# SALVATAGGIO STATO CORRENTE
# ================================================================

def save_current_state(
    current_regime: str,
    current_vix_state: str,
    entropy_state: str = "",
    erg_state: str = "",
    report_date: Optional[str] = None,
) -> bool:
    """
    Salva lo stato corrente del regime sul file JSON.

    Da chiamare DOPO l'invio del report Telegram, in modo che lo stato
    salvato corrisponda all'ultimo report effettivamente inviato.

    Args:
        current_regime:    Etichetta regime composita (es. "Alta|Non Ergodico")
        current_vix_state: Stato VIX (es. "NORMAL_VIX", "HIGH_VIX", "LOW_VIX")
        entropy_state:     Stato entropia (es. "Alta") — opzionale, per log
        erg_state:         Stato ergodicità (es. "Non Ergodico") — opzionale
        report_date:       Data ISO del report. Default: ora corrente UTC.

    Returns:
        True se il salvataggio è avvenuto con successo, False altrimenti.
    """
    if report_date is None:
        report_date = datetime.now(tz=timezone.utc).isoformat()

    state = {
        "current_regime":    current_regime,
        "current_vix_state": current_vix_state,
        "entropy_state":     entropy_state,
        "erg_state":         erg_state,
        "report_date":       report_date,
    }

    try:
        with open(STATE_FILEPATH, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
        return True
    except OSError as exc:
        print(f"[regime_state_store] Impossibile salvare lo stato: {exc}")
        return False


# ================================================================
# RILEVAMENTO CAMBIAMENTI
# ================================================================

def detect_changes(
    current_regime: str,
    current_vix_state: str,
    previous_state: Optional[dict],
) -> dict:
    """
    Confronta lo stato corrente con lo stato precedente e rileva i cambiamenti.

    Args:
        current_regime:    Regime corrente (es. "Alta|Non Ergodico")
        current_vix_state: Stato VIX corrente (es. "HIGH_VIX")
        previous_state:    Dizionario da load_previous_state(), o None se
                           è la prima esecuzione.

    Returns:
        Dizionario con:
          - regime_changed (bool)    : True se il regime composito è cambiato
          - vix_changed (bool)       : True se lo stato VIX è cambiato
          - any_changed (bool)       : True se almeno uno dei due è cambiato
          - previous_regime (str)    : regime precedente ("N/D" se prima esecuzione)
          - previous_vix_state (str) : stato VIX precedente ("N/D" se prima esecuzione)
          - previous_date (str)      : data del report precedente ("N/D" se assente)
          - is_first_run (bool)      : True se non esiste uno stato precedente
    """
    if previous_state is None:
        return {
            "regime_changed":     False,
            "vix_changed":        False,
            "any_changed":        False,
            "previous_regime":    "N/D",
            "previous_vix_state": "N/D",
            "previous_date":      "N/D",
            "is_first_run":       True,
        }

    prev_regime    = previous_state.get("current_regime",    "N/D")
    prev_vix_state = previous_state.get("current_vix_state", "N/D")
    prev_date      = previous_state.get("report_date",       "N/D")

    regime_changed = (prev_regime    != current_regime)
    vix_changed    = (prev_vix_state != current_vix_state)

    return {
        "regime_changed":     regime_changed,
        "vix_changed":        vix_changed,
        "any_changed":        regime_changed or vix_changed,
        "previous_regime":    prev_regime,
        "previous_vix_state": prev_vix_state,
        "previous_date":      prev_date,
        "is_first_run":       False,
    }
