"""
notify.py
=========
Script standalone per l'invio del report Telegram giornaliero.

Esegui questo script ogni giorno di mercato (lun-ven) alle 16:30 ET
(22:30 CEST estate / 21:30 CET inverno) tramite cron o Task Scheduler.

Funzionamento:
  1. Legge le configurazioni da .env o variabili d'ambiente
  2. Scarica i dati SPX aggiornati da EODHD
  3. Calcola i regimi di entropia ed ergodicità
  4. Carica i file equity da Google Drive
  5. Ottimizza le regole di esposizione
  6. Formatta e invia il messaggio Telegram

Setup cron (Linux/Mac) — esegui alle 21:30 UTC (= 16:30 ET + 30min buffer):
  30 21 * * 1-5  cd /path/to/kriterion-regime-filter && python notify.py

Setup alternativo con orario estivo (CEST = UTC+2):
  30 23 * * 1-5  cd /path/to/kriterion-regime-filter && python notify.py

Setup Windows Task Scheduler:
  Vedi README.md per le istruzioni dettagliate.

Variabili d'ambiente richieste (o nel file .env):
  EODHD_API_KEY       — chiave API EODHD
  TELEGRAM_BOT_TOKEN  — token bot Telegram (da @BotFather)
  TELEGRAM_CHAT_ID    — ID chat/gruppo destinatario
  DRIVE_FOLDER_ID     — (opzionale) ID cartella Google Drive [default: hardcoded]
"""

from __future__ import annotations

import os
import sys
import tempfile
from datetime import date, datetime, timezone

# Carica .env se presente (per esecuzione locale)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass   # python-dotenv opzionale

# Aggiunge la directory del progetto al path
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

# ================================================================
# IMPORT MODULI
# ================================================================
from src.equity_loader  import download_equity_files, load_all_trading_systems
from src.spx_data       import fetch_spx_no_cache, get_latest_spx_info
from src.regime_engine  import build_regime_series
from src.optimizer      import optimize_all_ts, get_current_exposure
from src.telegram_bot   import send_telegram_message, format_daily_report
from src.optimizer      import BOOST_RATIO, STANDARD_RATIO
from src.equity_loader  import MIN_TRADES_PER_REGIME


# ================================================================
# CONFIGURAZIONE
# ================================================================

EODHD_API_KEY      = os.environ.get("EODHD_API_KEY",      "")
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID   = os.environ.get("TELEGRAM_CHAT_ID",   "")
DRIVE_FOLDER_ID    = os.environ.get("DRIVE_FOLDER_ID",    "1kc0fu8UB8rZOZrSsfUnNtUcbOLyLx7ps")


def validate_config() -> bool:
    """Valida che le variabili d'ambiente siano configurate correttamente."""
    missing = []
    if not EODHD_API_KEY:
        missing.append("EODHD_API_KEY")
    if not TELEGRAM_BOT_TOKEN:
        missing.append("TELEGRAM_BOT_TOKEN")
    if not TELEGRAM_CHAT_ID:
        missing.append("TELEGRAM_CHAT_ID")

    if missing:
        print(f"[ERRORE] Variabili d'ambiente mancanti: {', '.join(missing)}")
        print("Imposta le variabili nel file .env oppure come variabili di ambiente.")
        return False
    return True


# ================================================================
# MAIN
# ================================================================

def main() -> None:
    """Pipeline principale del notificatore giornaliero."""

    now = datetime.now(tz=timezone.utc)
    print(f"[{now.strftime('%Y-%m-%d %H:%M UTC')}] Avvio notify.py")

    # ── 0. Validazione configurazione ────────────────────────────
    if not validate_config():
        sys.exit(1)

    # ── 1. Download equity da Google Drive ──────────────────────
    equities_dir = os.path.join(tempfile.gettempdir(), "kriterion_equities")
    print("[1/5] Download equity da Google Drive...")
    try:
        equities_dir = download_equity_files(
            folder_id        = DRIVE_FOLDER_ID,
            dest_dir         = equities_dir,
            force_redownload = False,   # usa cache se disponibile (aggiornata ogni ~24h)
        )
    except RuntimeError as exc:
        print(f"[ERRORE] Download equity fallito: {exc}")
        sys.exit(1)

    trading_systems = load_all_trading_systems(equities_dir)
    print(f"  → {len(trading_systems)} Trading System caricati")

    if not trading_systems:
        print("[ERRORE] Nessun Trading System disponibile. Controlla la cartella Drive.")
        sys.exit(1)

    # ── 2. Fetch SPX ─────────────────────────────────────────────
    print("[2/5] Download dati SPX da EODHD...")
    try:
        spx_df = fetch_spx_no_cache(EODHD_API_KEY)
    except Exception as exc:
        print(f"[ERRORE] Fetch SPX fallito: {exc}")
        sys.exit(1)

    if spx_df.empty:
        print("[ERRORE] Dati SPX vuoti. Verifica la chiave API EODHD.")
        sys.exit(1)

    spx_info = get_latest_spx_info(spx_df)
    print(f"  → Ultima candela SPX: {spx_info.get('last_date', 'N/D')} | "
          f"Close: ${spx_info.get('last_close', 0):,.2f} "
          f"({spx_info.get('day_return', 0):+.2f}%)")

    # ── 3. Calcolo regime ─────────────────────────────────────────
    print("[3/5] Calcolo regime (Entropia + Ergodicità)...")
    regime_data = build_regime_series(spx_df)

    regime_series         = regime_data["regime_series"]
    entropy_feat          = regime_data["entropy_feat"]
    erg_feat              = regime_data["erg_feat"]
    erg_thresh            = regime_data["erg_thresh"]
    current_regime        = regime_data["current_regime"]
    current_entropy_state = regime_data["current_entropy_state"]
    current_erg_state     = regime_data["current_erg_state"]

    last_sh   = float(entropy_feat["shannon_ret"].iloc[-1])
    last_pe   = float(entropy_feat["perm_entropy"].iloc[-1])
    last_diff = float(erg_feat["diff"].iloc[-1])
    threshold = erg_thresh["threshold"]

    print(f"  → Regime corrente: {current_regime}")
    print(f"  → Shannon Entropy: {last_sh:.4f} ({current_entropy_state})")
    print(f"  → Ergodicità diff: {last_diff:+.6f} | soglia: ±{threshold:.6f} ({current_erg_state})")

    # ── 4. Ottimizzazione esposizione ─────────────────────────────
    print("[4/5] Ottimizzazione regole di esposizione...")
    opt_results = optimize_all_ts(
        trading_systems = trading_systems,
        regime_series   = regime_series,
        min_trades      = MIN_TRADES_PER_REGIME,
        boost_ratio     = BOOST_RATIO,
        standard_ratio  = STANDARD_RATIO,
    )

    ts_exposures = {
        ts_name: get_current_exposure(ts_name, opt_results, current_regime)
        for ts_name in trading_systems
    }

    # Stampa riepilogo in console
    print("\n  === ESPOSIZIONI CORRENTI ===")
    for ts_name, exp in ts_exposures.items():
        print(f"  {exp['emoji']} {ts_name:50s} → {exp['label']:8s} (×{exp['multiplier']})")
    print()

    # ── 5. Formatta e invia messaggio Telegram ────────────────────
    print("[5/5] Invio report Telegram...")

    msg = format_daily_report(
        spx_info         = spx_info,
        current_regime   = current_regime,
        entropy_state    = current_entropy_state,
        erg_state        = current_erg_state,
        entropy_val      = last_sh,
        perm_entropy_val = last_pe,
        erg_diff_val     = last_diff,
        erg_threshold    = threshold,
        ts_exposures     = ts_exposures,
        report_time      = now,
    )

    ok = send_telegram_message(msg, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)

    if ok:
        print(f"  ✅ Report Telegram inviato con successo.")
    else:
        print(f"  ❌ Invio fallito. Controlla token e chat_id.")
        sys.exit(1)

    print(f"[DONE] {now.strftime('%Y-%m-%d %H:%M UTC')}")


if __name__ == "__main__":
    main()
