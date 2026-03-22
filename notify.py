"""
notify.py
=========
Script standalone per l'invio del report Telegram giornaliero.

Esegui questo script ogni giorno di mercato (lun-ven) alle 16:30 ET
(22:30 CEST estate / 21:30 CET inverno) tramite cron o Task Scheduler.

Funzionamento:
  1. Legge le configurazioni da .env o variabili d'ambiente
  2. Scarica i dati SPX e VIX aggiornati da EODHD
  3. Calcola i regimi di entropia ed ergodicità (SPX)
  4. Calcola VIX percentile + isteresi → stato VIX corrente
  5. Carica i file equity da Google Drive
  6. Ottimizza le regole di esposizione (regime + VIX) per ogni TS
  7. Calcola il moltiplicatore finale combinato per ogni TS
  8. Formatta e invia il messaggio Telegram con stato completo

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
from datetime import datetime, timezone

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
from src.optimizer      import optimize_all_ts, BOOST_RATIO, STANDARD_RATIO
from src.equity_loader  import MIN_TRADES_PER_REGIME
from src.vix_modulator  import (
    fetch_vix, compute_vix_features, get_current_vix_info,
    optimize_all_ts_vix, get_combined_exposure,
)
from src.telegram_bot   import send_telegram_message, format_daily_report


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
    print("[1/6] Download equity da Google Drive...")
    try:
        equities_dir = download_equity_files(
            folder_id        = DRIVE_FOLDER_ID,
            dest_dir         = equities_dir,
            force_redownload = False,
        )
    except RuntimeError as exc:
        print(f"[ERRORE] Download equity fallito: {exc}")
        sys.exit(1)

    trading_systems = load_all_trading_systems(equities_dir)
    print(f"  → {len(trading_systems)} Trading System caricati")

    if not trading_systems:
        print("[ERRORE] Nessun Trading System disponibile.")
        sys.exit(1)

    # ── 2. Fetch SPX ─────────────────────────────────────────────
    print("[2/6] Download dati SPX da EODHD...")
    try:
        spx_df = fetch_spx_no_cache(EODHD_API_KEY)
    except Exception as exc:
        print(f"[ERRORE] Fetch SPX fallito: {exc}")
        sys.exit(1)

    if spx_df.empty:
        print("[ERRORE] Dati SPX vuoti. Verifica la chiave API EODHD.")
        sys.exit(1)

    spx_info = get_latest_spx_info(spx_df)
    print(f"  → SPX: {spx_info.get('last_date', 'N/D')} | "
          f"${spx_info.get('last_close', 0):,.2f} "
          f"({spx_info.get('day_return', 0):+.2f}%)")

    # ── 3. Fetch VIX ─────────────────────────────────────────────
    print("[3/6] Download dati VIX da EODHD...")
    vix_features     = None
    current_vix_info = None
    current_vix_state = "NORMAL_VIX"

    try:
        vix_df_raw = fetch_vix(EODHD_API_KEY)
        if not vix_df_raw.empty:
            vix_features      = compute_vix_features(vix_df_raw)
            current_vix_info  = get_current_vix_info(vix_features)
            current_vix_state = current_vix_info["state"]
            print(f"  → VIX: {current_vix_info.get('vix_close', 0):.2f} | "
                  f"Percentile: {current_vix_info.get('vix_pct', 50):.0f}° | "
                  f"Stato: {current_vix_info.get('label', 'N/D')}")
        else:
            print("  ⚠️ VIX vuoto, si procede senza layer VIX (moltiplicatori VIX = 1.0)")
    except Exception as exc:
        print(f"  ⚠️ Fetch VIX fallito ({exc}), si procede senza layer VIX.")

    # ── 4. Calcolo regime SPX (Entropia + Ergodicità) ─────────────
    print("[4/6] Calcolo regime (Entropia + Ergodicità)...")
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

    print(f"  → Regime: {current_regime} | Entropy: {last_sh:.4f} ({current_entropy_state})")
    print(f"  → Ergodicità: diff={last_diff:+.6f} | soglia=±{threshold:.6f} ({current_erg_state})")

    # ── 5. Ottimizzazione esposizione (Regime + VIX) ──────────────
    print("[5/6] Ottimizzazione regole di esposizione...")
    opt_results = optimize_all_ts(
        trading_systems = trading_systems,
        regime_series   = regime_series,
        min_trades      = MIN_TRADES_PER_REGIME,
        boost_ratio     = BOOST_RATIO,
        standard_ratio  = STANDARD_RATIO,
    )

    vix_opt_results: dict = {}
    if vix_features is not None and not vix_features.empty:
        vix_opt_results = optimize_all_ts_vix(
            trading_systems = trading_systems,
            vix_features    = vix_features,
            min_trades      = MIN_TRADES_PER_REGIME,
            boost_ratio     = BOOST_RATIO,
            standard_ratio  = STANDARD_RATIO,
        )

    # Calcola esposizione COMBINATA per ogni TS
    ts_exposures = {
        ts_name: get_combined_exposure(
            ts_name           = ts_name,
            opt_results       = opt_results,
            vix_opt_results   = vix_opt_results,
            current_regime    = current_regime,
            current_vix_state = current_vix_state,
        )
        for ts_name in trading_systems
    }

    # Stampa riepilogo in console
    print("\n  === ESPOSIZIONI CORRENTI ===")
    for ts_name, exp in ts_exposures.items():
        print(
            f"  {exp['emoji']} {ts_name:50s} → {exp['label']:8s} "
            f"(R:×{exp['regime_mult']:.1f} × V:×{exp['vix_mult']:.1f} = ×{exp['final_mult']:.1f})"
        )
    print()

    # ── 6. Formatta e invia messaggio Telegram ────────────────────
    print("[6/6] Invio report Telegram...")

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
        vix_info         = current_vix_info,
    )

    ok = send_telegram_message(msg, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)

    if ok:
        print("  ✅ Report Telegram inviato con successo.")
    else:
        print("  ❌ Invio fallito. Controlla token e chat_id.")
        sys.exit(1)

    print(f"[DONE] {now.strftime('%Y-%m-%d %H:%M UTC')}")


if __name__ == "__main__":
    main()
