"""
telegram_bot.py
===============
Formattazione e invio dei messaggi Telegram giornalieri.

Il messaggio viene inviato ogni giorno alle 16:30 ET (22:30 CET estate
/ 21:30 CET inverno) dopo la chiusura del mercato US.

Struttura del messaggio:
  🤖 Header con data e condizioni SPX
  📊 Shannon Entropy + Permutation Entropy (valore e regime)
  ⚙️ Stato ergodicità (valore diff e soglia)
  💼 Lista di tutti i Trading System con esposizione corrente
  📝 Footer con regime composito e orario di aggiornamento

Requisiti:
  - TELEGRAM_BOT_TOKEN: token del bot (da @BotFather)
  - TELEGRAM_CHAT_ID:   chat/gruppo dove inviare i messaggi
"""

from __future__ import annotations

import requests
from datetime import datetime, timezone
import pandas as pd


# ================================================================
# INVIO MESSAGGIO
# ================================================================

def send_telegram_message(
    message:    str,
    bot_token:  str,
    chat_id:    str,
    parse_mode: str = "HTML",
) -> bool:
    """
    Invia un messaggio Telegram tramite l'API del bot.

    Usa parse_mode='HTML' per supportare tag <b>, <i>, <code>.

    Args:
        message:    Testo del messaggio (supporta HTML)
        bot_token:  Token del bot Telegram (da @BotFather)
        chat_id:    ID della chat/gruppo destinatario
        parse_mode: Modalità di parsing ('HTML' o 'MarkdownV2')

    Returns:
        True se l'invio ha avuto successo, False altrimenti
    """
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        "chat_id":    chat_id,
        "text":       message,
        "parse_mode": parse_mode,
    }

    try:
        resp = requests.post(url, json=payload, timeout=15)
        resp.raise_for_status()
        return True
    except requests.exceptions.RequestException as exc:
        print(f"[Telegram] Errore invio messaggio: {exc}")
        return False


# ================================================================
# FORMATTAZIONE MESSAGGIO
# ================================================================

def format_daily_report(
    spx_info:         dict,
    current_regime:   str,
    entropy_state:    str,
    erg_state:        str,
    entropy_val:      float,
    perm_entropy_val: float,
    erg_diff_val:     float,
    erg_threshold:    float,
    ts_exposures:     dict[str, dict],
    report_time:      "datetime | None" = None,
    vix_info:         "dict | None" = None,
    regime_changes:   "dict | None" = None,    # ← NUOVO parametro
) -> str:
    """
    Formatta il messaggio Telegram giornaliero completo.

    Args:
        spx_info:         Dizionario da get_latest_spx_info()
        current_regime:   Etichetta regime corrente (es. 'Bassa|Ergodico')
        entropy_state:    Stato entropia ('Bassa' / 'Media' / 'Alta')
        erg_state:        Stato ergodicità ('Ergodico' / 'Non Ergodico')
        entropy_val:      Valore Shannon Entropy corrente
        perm_entropy_val: Valore Permutation Entropy corrente
        erg_diff_val:     Valore diff ergodicità corrente
        erg_threshold:    Soglia SEM ergodicità
        ts_exposures:     {ts_name: exposure_dict} da get_combined_exposure()
                          Gli exposure_dict devono avere:
                            multiplier (float), label (str), emoji (str),
                            regime_mult (float), vix_mult (float), final_mult (float) [opzionali]
        report_time:      Timestamp del report (default: ora corrente UTC)
        vix_info:         Dizionario da get_current_vix_info() (opzionale)
        regime_changes:   Dizionario da detect_changes() (opzionale)

    Returns:
        Stringa HTML formattata per Telegram
    """
    if report_time is None:
        report_time = datetime.now(tz=timezone.utc)

    # Formatta la data in italiano
    giorni = ["Lunedì", "Martedì", "Mercoledì", "Giovedì",
              "Venerdì", "Sabato", "Domenica"]
    mesi   = ["Gennaio", "Febbraio", "Marzo", "Aprile", "Maggio", "Giugno",
              "Luglio", "Agosto", "Settembre", "Ottobre", "Novembre", "Dicembre"]

    last_date  = spx_info.get("last_date", report_time)
    giorno_str = f"{giorni[last_date.weekday()]} {last_date.day} {mesi[last_date.month - 1]} {last_date.year}"

    spx_close  = spx_info.get("last_close", 0.0)
    day_ret    = spx_info.get("day_return", 0.0)
    ytd_ret    = spx_info.get("ytd_return", 0.0)

    # Icone dinamiche
    ret_icon   = "🟢" if day_ret >= 0 else "🔴"
    ytd_icon   = "🟢" if ytd_ret >= 0 else "🔴"

    ent_icons  = {"Bassa": "🟢", "Media": "🟡", "Alta": "🔴"}
    erg_icons  = {"Ergodico": "✅", "Non Ergodico": "⚠️"}
    ent_icon   = ent_icons.get(entropy_state, "⚪")
    erg_icon   = erg_icons.get(erg_state, "⚪")

    # ── Sezione VIX (se disponibile) ─────────────────────────────
    vix_block = ""
    if vix_info:
        vix_state  = vix_info.get("state",     "NORMAL_VIX")
        vix_close  = vix_info.get("vix_close", 0.0)
        vix_pct    = vix_info.get("vix_pct",   50.0)
        vix_emoji  = vix_info.get("emoji",     "🟢")
        vix_label  = vix_info.get("label",     "Vol. Normale")

        vix_block = (
            f"  {vix_emoji} VIX: <code>{vix_close:.2f}</code> "
            f"| Percentile: <code>{vix_pct:.0f}°</code> "
            f"→ <b>{vix_label}</b>\n"
        )

    # ── Sezione cambio regime (se presente) ──────────────────────
    change_block = ""
    if regime_changes and regime_changes.get("any_changed") and not regime_changes.get("is_first_run"):
        change_lines = ["🔄 <b>CAMBIO DI REGIME RILEVATO</b>"]

        if regime_changes.get("regime_changed"):
            prev_r = regime_changes.get("previous_regime", "N/D")
            change_lines.append(
                f"  📊 Regime: <code>{prev_r}</code> → <code>{current_regime}</code>"
            )

        if regime_changes.get("vix_changed"):
            _vix_labels = {
                "HIGH_VIX":   "Alta Volatilità",
                "LOW_VIX":    "Bassa Volatilità",
                "NORMAL_VIX": "Vol. Normale",
            }
            prev_v       = regime_changes.get("previous_vix_state", "N/D")
            prev_v_label = _vix_labels.get(prev_v, prev_v)
            curr_v_label = vix_info.get("label", "N/D") if vix_info else "N/D"
            change_lines.append(
                f"  📉 VIX: <code>{prev_v_label}</code> → <code>{curr_v_label}</code>"
            )

        prev_date = regime_changes.get("previous_date", "N/D")
        if prev_date != "N/D":
            try:
                _prev_dt      = datetime.fromisoformat(prev_date)
                prev_date_str = _prev_dt.strftime("%d/%m/%Y")
            except Exception:
                prev_date_str = prev_date[:10]
            change_lines.append(f"  ⏱ Precedente report: <code>{prev_date_str}</code>")

        change_block = "\n".join(change_lines) + "\n\n"

    # ── Sezione Trading Systems ───────────────────────────────────
    ts_lines = []
    for ts_name, exp in ts_exposures.items():
        emoji = exp.get("emoji",      "🟡")
        label = exp.get("label",      "STANDARD")
        mult  = exp.get("multiplier", 1.0)

        # Mostra dettaglio regime × VIX se disponibile
        regime_mult = exp.get("regime_mult")
        vix_mult    = exp.get("vix_mult")

        if regime_mult is not None and vix_mult is not None:
            detail = f"R:{regime_mult:.1f}× × V:{vix_mult:.1f}× = {mult:.1f}×"
        elif mult == 1.5:
            detail = "+50% capitale"
        elif mult == 0.5:
            detail = "-50% capitale"
        elif mult == 0.0:
            detail = "sistema inibito"
        else:
            detail = "capitale standard"

        ts_lines.append(
            f"{emoji} <b>{ts_name}</b> → {label} ({detail})"
        )

    ts_block = "\n".join(ts_lines) if ts_lines else "Nessun TS disponibile"

    # Regime summary
    regime_description = _get_regime_description(entropy_state, erg_state)

    # Costruisci il messaggio completo
    msg = (
        f"🤖 <b>Kriterion Quant — Report Giornaliero</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📅 <b>{giorno_str}</b> | 16:30 ET\n\n"
        f"📊 <b>S&P 500 (SPX)</b>\n"
        f"  Chiusura: <code>${spx_close:,.2f}</code> "
        f"{ret_icon} <code>{day_ret:+.2f}%</code> (oggi)\n"
        f"  YTD: {ytd_icon} <code>{ytd_ret:+.2f}%</code>\n\n"
        f"{change_block}"                        # ← blocco cambio regime
        f"🧪 <b>Metriche di Mercato SPX</b>\n"
        f"  {ent_icon} Shannon Entropy: <code>{entropy_val:.3f}</code> "
        f"→ Regime <b>{entropy_state}</b>\n"
        f"  📐 Perm. Entropy: <code>{perm_entropy_val:.3f}</code>\n"
        f"  {erg_icon} Ergodicità: <b>{erg_state}</b> "
        f"| diff=<code>{erg_diff_val:+.5f}</code> "
        f"| soglia=<code>±{erg_threshold:.5f}</code>\n"
        f"{vix_block}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"💼 <b>Stato Trading Systems</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"{ts_block}\n\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🎯 <b>Regime Attuale:</b> <code>{current_regime}</code>\n"
        f"<i>{regime_description}</i>\n\n"
        f"⏰ Report generato: {report_time.strftime('%d/%m/%Y %H:%M UTC')}\n"
        f"<i>Fonte dati: EODHD | Kriterion Quant</i>"
    )

    return msg


def _get_regime_description(entropy_state: str, erg_state: str) -> str:
    """
    Restituisce una descrizione breve del regime corrente per il messaggio Telegram.

    Args:
        entropy_state: 'Bassa', 'Media' o 'Alta'
        erg_state:     'Ergodico' o 'Non Ergodico'

    Returns:
        Stringa descrittiva del regime
    """
    descriptions = {
        ("Bassa",  "Ergodico"):     "Mercato ordinato e prevedibile. Condizioni ideali per sistemi trend-following.",
        ("Bassa",  "Non Ergodico"): "Struttura nei rendimenti, media storica in transizione. Cautela nelle strategie direzionali.",
        ("Media",  "Ergodico"):     "Condizioni nella norma. Nessuna anomalia strutturale rilevata.",
        ("Media",  "Non Ergodico"): "Entropia nella norma ma segnale di non ergodicità. Possibile inizio di transizione.",
        ("Alta",   "Ergodico"):     "Alta imprevedibilità ma media storica stabile. Volatilità elevata senza breakout strutturale.",
        ("Alta",   "Non Ergodico"): "Massima incertezza. Alta entropia + non ergodicità: regime storicamente anomalo.",
    }
    return descriptions.get(
        (entropy_state, erg_state),
        "Regime in analisi."
    )


# ================================================================
# MESSAGGIO TEST
# ================================================================

def send_test_message(bot_token: str, chat_id: str) -> bool:
    """
    Invia un messaggio di test per verificare la configurazione Telegram.

    Args:
        bot_token: Token del bot Telegram
        chat_id:   ID della chat/gruppo

    Returns:
        True se il test ha avuto successo
    """
    now = datetime.now(tz=timezone.utc)
    msg = (
        f"✅ <b>Test Kriterion Quant</b>\n"
        f"Connessione Telegram operativa.\n"
        f"⏰ {now.strftime('%d/%m/%Y %H:%M UTC')}"
    )
    return send_telegram_message(msg, bot_token, chat_id)
