"""
equity_loader.py
================
Scarica e parsa i file delle equity dei Trading System da Google Drive pubblica.

Formato file TradeStation/MultiCharts (pipe/semicolon delimited):
  ENTRY_DATE|TIME ; EXIT_DATE|TIME ; direction ; contracts ; pnl_total ; ...

Formato data TradeStation: YYYMMDD dove YYY = anni dal 1900
  Es: 1230501 → 1900+123=2023, mese=05, giorno=01 → 2023-05-01
      1060123 → 1900+106=2006, mese=01, giorno=23 → 2006-01-23
"""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

# ================================================================
# COSTANTI
# ================================================================

DRIVE_FOLDER_ID = "1kc0fu8UB8rZOZrSsfUnNtUcbOLyLx7ps"
DRIVE_FOLDER_URL = f"https://drive.google.com/drive/folders/{DRIVE_FOLDER_ID}"

# Moltiplicatori di esposizione e relative etichette/colori
EXPOSURE_LEVELS: list[float] = [0.0, 0.5, 1.0, 1.5]

EXPOSURE_LABELS: dict[float, str] = {
    0.0: "INIBITO",
    0.5: "RIDOTTO",
    1.0: "STANDARD",
    1.5: "BOOST",
}

EXPOSURE_COLORS: dict[float, str] = {
    0.0: "#F44336",   # rosso — sistema inibito
    0.5: "#FF9800",   # arancio — capitale ridotto del 50%
    1.0: "#2196F3",   # blu — capitale standard
    1.5: "#4CAF50",   # verde — capitale boostato del 50%
}

EXPOSURE_EMOJIS: dict[float, str] = {
    0.0: "🔴",
    0.5: "🟠",
    1.0: "🟡",
    1.5: "🟢",
}

# Numero minimo di trade in un regime per assegnare una regola non-standard
MIN_TRADES_PER_REGIME: int = 8


# ================================================================
# DOWNLOAD DA GOOGLE DRIVE (gdown)
# ================================================================

def download_equity_files(
    folder_id: str = DRIVE_FOLDER_ID,
    dest_dir: Optional[str] = None,
    force_redownload: bool = False,
) -> str:
    """
    Scarica tutti i file .txt delle equity dalla cartella Google Drive pubblica.

    Usa gdown.download_folder() che supporta cartelle pubbliche (link condivisibile).
    La funzione è idempotente: se i file esistono già non riscarica (a meno che
    force_redownload=True).

    Args:
        folder_id:        ID della cartella Google Drive
                          (dall'URL: drive.google.com/drive/folders/{ID})
        dest_dir:         Directory locale dove salvare i file
                          (default: /tmp/kriterion_equities)
        force_redownload: Se True, cancella i file esistenti e riscarica

    Returns:
        Percorso assoluto della directory con i file .txt scaricati

    Raises:
        RuntimeError: Se il download fallisce o gdown non è installato
    """
    try:
        import gdown
    except ImportError:
        raise RuntimeError(
            "gdown non è installato. Esegui: pip install gdown"
        )

    if dest_dir is None:
        dest_dir = os.path.join(tempfile.gettempdir(), "kriterion_equities")

    os.makedirs(dest_dir, exist_ok=True)

    # Controlla se ci sono già file .txt scaricati (skip se esistono e non forzato)
    existing_txts = list(Path(dest_dir).rglob("*.txt"))
    if existing_txts and not force_redownload:
        return dest_dir

    # Cancella i file precedenti se si forza il redownload
    if force_redownload:
        for f in Path(dest_dir).rglob("*"):
            if f.is_file():
                f.unlink()

    url = f"https://drive.google.com/drive/folders/{folder_id}"
    try:
        gdown.download_folder(
            url,
            output=dest_dir,
            quiet=False,
            use_cookies=False,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Errore nel download da Google Drive: {exc}\n"
            f"Verifica che la cartella '{url}' sia pubblica e "
            f"che gdown >= 4.6 sia installato."
        )

    return dest_dir


# ================================================================
# PARSING DATE TRADESTATION
# ================================================================

def parse_ts_date(ts_str: str) -> Optional[pd.Timestamp]:
    """
    Parsa una data nel formato proprietario TradeStation YYYMMDD.

    TradeStation usa un offset di 1900 anni sul campo YYY (3 cifre):
      - '1230501' → YYY=123, MM=05, DD=01 → anno 1900+123=2023 → 2023-05-01
      - '1060123' → YYY=106, MM=01, DD=23 → anno 1900+106=2006 → 2006-01-23

    Gestisce anche il formato standard YYYYMMDD (8 cifre) per compatibilità.

    Args:
        ts_str: Stringa data (7 cifre TradeStation o 8 cifre standard)

    Returns:
        pd.Timestamp o None se il parsing fallisce
    """
    s = str(ts_str).strip()
    try:
        if len(s) == 7:   # Formato TradeStation YYYMMDD (offset 1900)
            year  = 1900 + int(s[:3])
            month = int(s[3:5])
            day   = int(s[5:7])
        elif len(s) == 8:  # Formato standard YYYYMMDD
            year  = int(s[:4])
            month = int(s[4:6])
            day   = int(s[6:8])
        else:
            return None

        return pd.Timestamp(year=year, month=month, day=day)
    except (ValueError, TypeError):
        return None


def parse_ts_datetime_field(field: str) -> Optional[pd.Timestamp]:
    """
    Parsa un campo data+ora TradeStation nel formato 'DATE|TIME'.

    Esempi:
      '1230501|1600' → 2023-05-01 16:00
      '1060123|800'  → 2006-01-23 08:00

    Args:
        field: Stringa 'DATE|TIME' o solo 'DATE'

    Returns:
        pd.Timestamp con data e orario, o None se fallisce
    """
    field = field.strip()
    if "|" in field:
        date_part, time_part = field.split("|", 1)
    else:
        date_part = field
        time_part = "0"

    ts_date = parse_ts_date(date_part)
    if ts_date is None:
        return None

    try:
        time_val = int(time_part.strip())
        hour     = time_val // 100
        minute   = time_val % 100
        return ts_date.replace(hour=hour, minute=minute)
    except (ValueError, TypeError):
        return ts_date   # restituisce solo la data se l'orario non è parsabile


# ================================================================
# PARSING FILE EQUITY
# ================================================================

def parse_equity_file(filepath: str) -> pd.DataFrame:
    """
    Parsa un singolo file equity TradeStation e restituisce il DataFrame dei trade.

    Struttura riga (campi semicolon-separati):
      [0] ENTRY_DATE|TIME
      [1] EXIT_DATE|TIME
      [2] direction (buy/sell)
      [3] contracts (int)
      [4] pnl_total (float, PnL netto del trade in USD)
      [5+] dettagli intratrade opzionali (ignorati)

    Args:
        filepath: Percorso assoluto al file .txt

    Returns:
        DataFrame con colonne:
          - entry_date (pd.Timestamp): data di ingresso normalizzata (solo data)
          - exit_date  (pd.Timestamp): data di uscita normalizzata (solo data)
          - direction  (str):          'buy' o 'sell'
          - contracts  (int):          numero di contratti
          - pnl        (float):        PnL netto del trade in USD
        Ordinato per exit_date ascending.
    """
    trades = []

    with open(filepath, "r", encoding="utf-8", errors="replace") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line:
                continue

            # I campi sono separati da ';'
            parts = line.split(";")
            if len(parts) < 5:
                continue

            entry_field   = parts[0].strip()
            exit_field    = parts[1].strip()
            direction     = parts[2].strip().lower()
            contracts_str = parts[3].strip()
            pnl_str       = parts[4].strip()

            # Parsa le date (normalizzate a mezzanotte, solo data)
            entry_dt = parse_ts_datetime_field(entry_field)
            exit_dt  = parse_ts_datetime_field(exit_field)
            if entry_dt is None or exit_dt is None:
                continue

            try:
                contracts = int(contracts_str)
            except ValueError:
                contracts = 1

            # Parsa PnL — ignora righe con valori non numerici
            try:
                pnl = float(pnl_str)
            except ValueError:
                continue

            trades.append({
                "entry_date": entry_dt.normalize(),
                "exit_date":  exit_dt.normalize(),
                "direction":  direction,
                "contracts":  contracts,
                "pnl":        pnl,
            })

    if not trades:
        return pd.DataFrame(
            columns=["entry_date", "exit_date", "direction", "contracts", "pnl"]
        )

    df = pd.DataFrame(trades)
    df = df.sort_values("exit_date").reset_index(drop=True)
    return df


def extract_ts_name(filename: str) -> str:
    """
    Estrae il nome leggibile del Trading System dal nome del file equity.

    Esempi:
      'PF_BiasIntraweekAggregata_USD.txt' → 'PF BiasIntraweek'
      'MNQ_ShortCopertura-MNQ_USD.txt'   → 'MNQ ShortCopertura'
      'MGC_BRKPREZZI-GC_USD.txt'         → 'MGC BRKPREZZI'
      'MES_ZScoreMES_USD.txt'            → 'MES ZScoreMES'

    Args:
        filename: Nome del file (con o senza path)

    Returns:
        Nome leggibile del Trading System
    """
    stem = Path(filename).stem                    # rimuovi estensione
    stem = re.sub(r"_USD$", "", stem)             # rimuovi suffisso _USD
    stem = re.sub(r"\s*\([^)]*\)\s*$", "", stem) # rimuovi "(MNQ)" finale
    stem = re.sub(r"\s+", " ", stem)             # normalizza spazi
    stem = stem.replace("_", " ")                # underscore → spazio
    # Rimuovi il suffisso "-STRUMENTO" (es. "-MNQ", "-GC", "-MES", "-M2K")
    stem = re.sub(r"\s*-[A-Z0-9]{2,4}$", "", stem)
    return stem.strip()


def is_short_system(filename: str) -> bool:
    """
    Determina se un Trading System è una copertura short dal nome del file.

    Args:
        filename: Nome del file (con o senza path)

    Returns:
        True se il sistema è classificabile come copertura short
    """
    name_lower = Path(filename).stem.lower()
    return "short" in name_lower or "copertura" in name_lower


# ================================================================
# LOAD TUTTI I TRADING SYSTEM
# ================================================================

def load_all_trading_systems(equities_dir: str) -> dict[str, dict]:
    """
    Carica e parsa tutti i file equity .txt presenti nella directory.

    Cerca ricorsivamente i file .txt nella directory specificata,
    includendo le sottocartelle (gdown scarica in sottocartelle).

    Args:
        equities_dir: Directory radice contenente i file .txt

    Returns:
        Dizionario con struttura:
          {
            ts_name (str): {
              'filename':   str           — nome file originale
              'filepath':   str           — path assoluto
              'trades':     pd.DataFrame  — trade parsati
              'is_short':   bool          — True se sistema short/copertura
              'start_date': pd.Timestamp  — prima data di ingresso
              'end_date':   pd.Timestamp  — ultima data di uscita
              'n_trades':   int           — numero di trade
            }
          }
    """
    result: dict[str, dict] = {}

    # Cerca ricorsivamente tutti i .txt (gdown può creare sottocartelle)
    txt_files = sorted(Path(equities_dir).rglob("*.txt"))

    for filepath in txt_files:
        filename = filepath.name
        ts_name  = extract_ts_name(filename)

        trades = parse_equity_file(str(filepath))
        if trades.empty:
            continue

        # Se esistono nomi duplicati, aggiungi un suffisso numerico
        base_name = ts_name
        i = 2
        while ts_name in result:
            ts_name = f"{base_name} ({i})"
            i += 1

        result[ts_name] = {
            "filename":   filename,
            "filepath":   str(filepath),
            "trades":     trades,
            "is_short":   is_short_system(filename),
            "start_date": trades["entry_date"].min(),
            "end_date":   trades["exit_date"].max(),
            "n_trades":   len(trades),
        }

    return result
