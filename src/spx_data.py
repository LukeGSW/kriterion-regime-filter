"""
spx_data.py
===========
Fetch e caching dei dati OHLCV dell'S&P 500 (GSPC.INDX) da EODHD.

Questo modulo è il punto di ingresso unico per i dati di mercato SPX.
L'indice SPX è usato come fonte per il calcolo dei regimi di entropia
ed ergodicità che governano l'esposizione dei Trading System.

Note sul ticker EODHD:
  - Ticker: GSPC.INDX
  - Non viene passato from_date per scaricare tutta la storia disponibile:
    EODHD limita a ~19.000 record per chiamata, quindi partendo dall'inizio
    si ottiene la serie più aggiornata possibile.
"""

from __future__ import annotations

import requests
import pandas as pd
import streamlit as st
from datetime import date, datetime


# ================================================================
# COSTANTI
# ================================================================

SPX_TICKER         = "GSPC.INDX"
SPX_LABEL          = "S&P 500 (SPX)"
EODHD_BASE_URL     = "https://eodhd.com/api/eod"

# Orario chiusura mercato US in UTC (16:00 ET = 21:00 UTC estate, 22:00 UTC inverno)
# Usiamo 21:00 UTC come riferimento (estate = DST attivo)
MARKET_CLOSE_UTC   = "21:00"


# ================================================================
# FETCH DATI EODHD
# ================================================================

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_spx(
    api_key: str,
    to_date: str | None = None,
) -> pd.DataFrame:
    """
    Scarica l'intera storia disponibile dell'S&P 500 da EODHD.

    La chiave cache include to_date (default = oggi) per garantire che
    la cache si invalidi ogni giorno dopo la chiusura del mercato.

    Args:
        api_key: Chiave API EODHD (letta da st.secrets)
        to_date: Data di fine 'YYYY-MM-DD' (default: oggi)

    Returns:
        DataFrame con colonne: open, high, low, close, adjusted_close, volume
        Index: DatetimeIndex ordinato dal più vecchio al più recente

    Raises:
        requests.HTTPError: Se la chiave API non è valida o il ticker non esiste
        ValueError:          Se il JSON restituito è vuoto o malformato
    """
    if to_date is None:
        to_date = str(date.today())

    url = (
        f"{EODHD_BASE_URL}/{SPX_TICKER}"
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

    # Coercizione numerica sicura per tutte le colonne OHLCV
    cols = ["open", "high", "low", "close", "adjusted_close", "volume"]
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    available_cols = [c for c in cols if c in df.columns]
    return df[available_cols]


def fetch_spx_no_cache(api_key: str) -> pd.DataFrame:
    """
    Versione senza cache per uso nello script notify.py (esecuzione batch).

    Scarica sempre dati freschi, ignora la cache Streamlit.

    Args:
        api_key: Chiave API EODHD

    Returns:
        DataFrame SPX identico a fetch_spx()
    """
    to_date = str(date.today())
    url = (
        f"{EODHD_BASE_URL}/{SPX_TICKER}"
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

    cols = ["open", "high", "low", "close", "adjusted_close", "volume"]
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    available_cols = [c for c in cols if c in df.columns]
    return df[available_cols]


def get_latest_spx_info(spx_df: pd.DataFrame) -> dict:
    """
    Estrae le informazioni sull'ultima candela disponibile.

    Args:
        spx_df: DataFrame SPX da fetch_spx()

    Returns:
        Dizionario con:
          - last_date  (pd.Timestamp): data ultima candela
          - last_close (float):        prezzo di chiusura
          - prev_close (float):        chiusura precedente
          - day_return (float):        rendimento giornaliero %
          - ytd_return (float):        rendimento YTD %
    """
    if spx_df.empty or len(spx_df) < 2:
        return {}

    price_col = "adjusted_close" if "adjusted_close" in spx_df.columns else "close"
    price = spx_df[price_col].dropna()

    last_date  = price.index[-1]
    last_close = float(price.iloc[-1])
    prev_close = float(price.iloc[-2])
    day_return = (last_close / prev_close - 1) * 100

    # YTD: primo giorno dell'anno corrente
    year_start_price = float(
        price[price.index.year == last_date.year].iloc[0]
        if (price.index.year == last_date.year).any()
        else price.iloc[0]
    )
    ytd_return = (last_close / year_start_price - 1) * 100

    return {
        "last_date":  last_date,
        "last_close": last_close,
        "prev_close": prev_close,
        "day_return": day_return,
        "ytd_return": ytd_return,
    }
