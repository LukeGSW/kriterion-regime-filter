"""
app.py
======
Kriterion Regime Filter — Dashboard Streamlit principale.

Questa dashboard applica gli studi di Entropia ed Ergodicità sull'indice S&P 500,
con un terzo layer basato sul VIX (volatilità implicita), per determinare il regime
di mercato ottimale per ogni Trading System algoritmico.

Per ogni TS, il sistema:
  1. Scarica i file equity da Google Drive (pubblica)
  2. Calcola Shannon Entropy e Permutation Entropy sull'SPX (finestra 63g)
  3. Calcola l'Ergodicità SPX tramite il metodo SEM (finestra 252g)
  4. Scarica e analizza il VIX: rolling percentile + isteresi (LOW/NORMAL/HIGH)
  5. Identifica le 6 combinazioni di regime (3 Entropia × 2 Ergodicità)
  6. Ottimizza le regole di esposizione per ogni TS su regime E su VIX state
  7. Calcola il moltiplicatore finale COMBINATO = snap(regime × vix) ∈ {0/0.5/1/1.5}
  8. Mostra le 3 equity curve (Baseline / Regime-Adj / Combined) e lo stato corrente

Deploy: Streamlit Cloud — imposta EODHD_API_KEY, TELEGRAM_BOT_TOKEN,
        TELEGRAM_CHAT_ID nei Secrets di Streamlit Cloud.
"""

from __future__ import annotations

import os
import tempfile
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd
import streamlit as st

# ================================================================
# CONFIGURAZIONE PAGINA (deve essere la prima chiamata Streamlit)
# ================================================================
st.set_page_config(
    page_title="Kriterion Regime Filter | Kriterion Quant",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ================================================================
# IMPORT MODULI INTERNI
# ================================================================
from src.equity_loader  import (
    download_equity_files, load_all_trading_systems,
    EXPOSURE_LABELS, EXPOSURE_COLORS, EXPOSURE_EMOJIS,
)
from src.spx_data       import fetch_spx, get_latest_spx_info
from src.regime_engine  import build_regime_series, REGIME_COLORS, REGIME_DESCRIPTIONS
from src.optimizer      import optimize_all_ts, BOOST_RATIO, STANDARD_RATIO
from src.exposure_engine import build_all_equity_curves, compute_performance_comparison
from src.vix_modulator  import (
    fetch_vix, compute_vix_features, get_current_vix_info,
    optimize_all_ts_vix, get_combined_exposure,
    VIX_STATE_LABELS, VIX_STATE_COLORS, VIX_STATE_EMOJIS, VIX_STATES,
    HIGH_VIX_ENTRY_PCT, HIGH_VIX_EXIT_PCT, LOW_VIX_ENTRY_PCT, LOW_VIX_EXIT_PCT,
)
from src.charts         import (
    build_equity_comparison_chart,
    build_regime_heatmap,
    build_spx_regime_chart,
    build_exposure_gauge,
    build_pnl_distribution_chart,
    build_overview_table,
    build_vix_chart,
)
from src.telegram_bot   import send_telegram_message, format_daily_report, send_test_message


# ================================================================
# SECRETS & PARAMETRI GLOBALI
# ================================================================
try:
    EODHD_API_KEY      = st.secrets["EODHD_API_KEY"]
    TELEGRAM_BOT_TOKEN = st.secrets.get("TELEGRAM_BOT_TOKEN", "")
    TELEGRAM_CHAT_ID   = st.secrets.get("TELEGRAM_CHAT_ID",   "")
except KeyError as missing_key:
    st.error(
        f"❌ **Secret mancante: `{missing_key}`**\n\n"
        "Configura nei Secrets di Streamlit Cloud (o in `.streamlit/secrets.toml`):\n"
        "- `EODHD_API_KEY` — chiave EODHD Historical Data\n"
        "- `TELEGRAM_BOT_TOKEN` — token bot Telegram\n"
        "- `TELEGRAM_CHAT_ID` — ID chat Telegram"
    )
    st.stop()

DRIVE_FOLDER_ID = "1kc0fu8UB8rZOZrSsfUnNtUcbOLyLx7ps"


# ================================================================
# FUNZIONI CACHED
# ================================================================

@st.cache_data(ttl=3600, show_spinner=False)
def cached_fetch_spx(api_key: str, to_date: str) -> pd.DataFrame:
    return fetch_spx(api_key=api_key, to_date=to_date)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_fetch_vix(api_key: str, to_date: str) -> pd.DataFrame:
    """Cache del fetch VIX: stesso TTL di SPX (1 ora)."""
    return fetch_vix(api_key=api_key, to_date=to_date)


@st.cache_data(ttl=86400, show_spinner=False)
def cached_build_regime_series(spx_hash: str, _spx_df: pd.DataFrame) -> dict:
    """Cache del calcolo regime: si invalida giornalmente."""
    return build_regime_series(_spx_df)


@st.cache_data(ttl=86400, show_spinner=False)
def cached_compute_vix_features(vix_hash: str, _vix_df: pd.DataFrame) -> pd.DataFrame:
    """Cache del calcolo VIX features: si invalida giornalmente."""
    return compute_vix_features(_vix_df)


@st.cache_resource(show_spinner=False)
def cached_load_trading_systems(equities_dir: str) -> dict:
    return load_all_trading_systems(equities_dir)


# ================================================================
# SIDEBAR — PARAMETRI E CONTROLLI
# ================================================================
with st.sidebar:
    st.image(
        "https://via.placeholder.com/200x60/1E1E2E/2196F3?text=Kriterion+Quant",
        use_column_width=True,
    )
    st.title("⚙️ Parametri")
    st.divider()

    # ── Parametri ottimizzazione ───────────────────────────────
    st.subheader("🔧 Ottimizzazione")

    min_trades = st.slider(
        "Trade minimi per regime",
        min_value=3, max_value=30, value=8, step=1,
        help="Numero minimo di trade in un regime per assegnare una regola non-STANDARD. "
             "Vale sia per i regimi Entropia+Ergodicità sia per gli stati VIX."
    )

    boost_ratio = st.slider(
        "Soglia BOOST (× media globale)",
        min_value=1.1, max_value=3.0, value=1.4, step=0.1,
        format="%.1f×",
        help="Rapporto mean_pnl_regime / mean_pnl_overall sopra il quale si applica BOOST (+50%)."
    )

    standard_ratio = st.slider(
        "Soglia STANDARD (× media globale)",
        min_value=0.0, max_value=1.0, value=0.3, step=0.05,
        format="%.2f×",
        help="Rapporto sopra il quale il regime è STANDARD. Sotto si applica REDUCE (-50%)."
    )

    st.divider()

    # ── VIX Isteresi ──────────────────────────────────────────
    st.subheader("📊 VIX — Isteresi")

    st.caption(
        "Le bande di isteresi prevengono transizioni rapide sul confine "
        "(anti-whipsaw). Entry > Exit in entrambe le direzioni."
    )

    col_h1, col_h2 = st.columns(2)
    with col_h1:
        st.caption("🔴 HIGH VIX")
        st.caption(f"Entry: > {HIGH_VIX_ENTRY_PCT:.0f}°")
        st.caption(f"Exit: < {HIGH_VIX_EXIT_PCT:.0f}°")
    with col_h2:
        st.caption("🔵 LOW VIX")
        st.caption(f"Entry: < {LOW_VIX_ENTRY_PCT:.0f}°")
        st.caption(f"Exit: > {LOW_VIX_EXIT_PCT:.0f}°")

    st.divider()

    # ── Dati SPX ──────────────────────────────────────────────
    st.subheader("📡 Dati & Cache")
    force_reload = st.button(
        "🔄 Ricarica Equity da Drive",
        help="Forza il re-download dei file equity da Google Drive e ricalcola tutto."
    )

    send_test_tg = st.button(
        "📲 Test Telegram",
        help="Invia un messaggio di test al bot Telegram configurato."
    )

    send_report_now = st.button(
        "📤 Invia Report Ora",
        help="Invia subito il report Telegram senza aspettare la chiusura del mercato."
    )

    st.divider()
    st.caption(f"📡 Dati: EODHD Historical Data API")
    st.caption(f"🗓️ Aggiornato: {datetime.now().strftime('%d/%m/%Y %H:%M')}")
    st.caption("🔗 [Kriterion Quant](https://kriterionquant.com)")


# ================================================================
# HEADER PRINCIPALE
# ================================================================
st.title("🎯 Kriterion Regime Filter")
st.markdown("""
Questo sistema applica tre studi quantitativi sull'**S&P 500** per classificare
il contesto di mercato e regolare l'esposizione dei Trading System algoritmici:

- **Entropia** (Shannon + Permutation Entropy): complessità statistica dei log-return → 3 stati
- **Ergodicità** (Standard Error of the Mean): stabilità strutturale della distribuzione → 2 stati
- **VIX Percentile + Isteresi**: paura forward-looking del mercato delle opzioni → 3 stati

Il **moltiplicatore finale** per ogni TS = **snap(Regime × VIX)** ∈ {×0, ×0.5, ×1.0, ×1.5}:

- 🔴 **INIBITO** (×0): sistema disabilitato nel regime
- 🟠 **RIDOTTO** (×0.5): capitale ridotto del 50%
- 🟡 **STANDARD** (×1.0): esposizione normale
- 🟢 **BOOST** (×1.5): capitale aumentato del 50%
""")
st.divider()


# ================================================================
# STEP 1 — DOWNLOAD EQUITY DA GOOGLE DRIVE
# ================================================================
equities_dir = os.path.join(tempfile.gettempdir(), "kriterion_equities")

with st.spinner("⏳ Caricamento file equity da Google Drive..."):
    try:
        equities_dir = download_equity_files(
            folder_id        = DRIVE_FOLDER_ID,
            dest_dir         = equities_dir,
            force_redownload = force_reload,
        )
    except RuntimeError as e:
        st.error(f"❌ Errore download Google Drive: {e}")
        st.info(
            "💡 **Suggerimento**: verifica che la cartella Drive sia condivisa come "
            "**'Chiunque abbia il link può visualizzare'** "
            "(tasto destro sulla cartella → Condividi → cambia permesso)."
        )
        st.stop()

trading_systems = cached_load_trading_systems(equities_dir)

if not trading_systems:
    st.error(
        "❌ Nessun file equity trovato nella cartella Google Drive. "
        "Verifica che la cartella contenga file .txt in formato TradeStation."
    )
    st.stop()

# ================================================================
# STEP 2 — FETCH SPX E VIX DA EODHD
# ================================================================
today_str = str(date.today())

with st.spinner("⏳ Download dati SPX e VIX da EODHD..."):
    try:
        spx_df = cached_fetch_spx(EODHD_API_KEY, today_str)
    except Exception as e:
        st.error(f"❌ Errore fetch SPX: {e}. Verifica la chiave EODHD_API_KEY.")
        st.stop()

    try:
        vix_df_raw = cached_fetch_vix(EODHD_API_KEY, today_str)
    except Exception as e:
        st.warning(f"⚠️ Fetch VIX fallito: {e}. Il sistema continuerà senza il layer VIX.")
        vix_df_raw = pd.DataFrame()

if spx_df.empty:
    st.error("❌ Dati SPX non disponibili. Verifica la chiave API EODHD.")
    st.stop()

spx_info = get_latest_spx_info(spx_df)

# ================================================================
# STEP 3 — CALCOLO REGIMI (Entropia + Ergodicità)
# ================================================================
with st.spinner("⏳ Calcolo regimi di mercato (Entropia + Ergodicità)..."):
    regime_data = cached_build_regime_series(today_str, spx_df)

regime_series         = regime_data["regime_series"]
entropy_feat          = regime_data["entropy_feat"]
erg_feat              = regime_data["erg_feat"]
entropy_thresh        = regime_data["entropy_thresh"]
erg_thresh            = regime_data["erg_thresh"]
current_regime        = regime_data["current_regime"]
current_entropy_state = regime_data["current_entropy_state"]
current_erg_state     = regime_data["current_erg_state"]
last_date             = regime_data["last_date"]

# ================================================================
# STEP 4 — CALCOLO VIX FEATURES + STATO CORRENTE
# ================================================================
vix_features      = pd.DataFrame()
current_vix_info  = {
    "state":     "NORMAL_VIX",
    "vix_close": float("nan"),
    "vix_pct":   50.0,
    "label":     VIX_STATE_LABELS["NORMAL_VIX"],
    "color":     VIX_STATE_COLORS["NORMAL_VIX"],
    "emoji":     VIX_STATE_EMOJIS["NORMAL_VIX"],
}
vix_available = False

if not vix_df_raw.empty:
    with st.spinner("⏳ Calcolo VIX percentile e stati isteresi..."):
        vix_features     = cached_compute_vix_features(today_str, vix_df_raw)
        current_vix_info = get_current_vix_info(vix_features)
        vix_available    = not vix_features.empty

current_vix_state = current_vix_info["state"]

# ================================================================
# STEP 5 — OTTIMIZZAZIONE ESPOSIZIONE PER OGNI TS (Regime)
# ================================================================
with st.spinner("⏳ Ottimizzazione regole di esposizione (Regime)..."):
    opt_results = optimize_all_ts(
        trading_systems = trading_systems,
        regime_series   = regime_series,
        min_trades      = min_trades,
        boost_ratio     = boost_ratio,
        standard_ratio  = standard_ratio,
    )

# ================================================================
# STEP 6 — OTTIMIZZAZIONE VIX PER OGNI TS
# ================================================================
vix_opt_results: dict = {}
if vix_available:
    with st.spinner("⏳ Ottimizzazione regole di esposizione (VIX)..."):
        vix_opt_results = optimize_all_ts_vix(
            trading_systems = trading_systems,
            vix_features    = vix_features,
            min_trades      = min_trades,
            boost_ratio     = boost_ratio,
            standard_ratio  = standard_ratio,
        )

# ================================================================
# STEP 7 — EQUITY CURVES (Baseline / Regime / Combined)
# ================================================================
with st.spinner("⏳ Costruzione equity curve (3 scenari)..."):
    equity_curves = build_all_equity_curves(
        trading_systems  = trading_systems,
        regime_series    = regime_series,
        opt_results      = opt_results,
        vix_features     = vix_features if vix_available else None,
        vix_opt_results  = vix_opt_results if vix_available else None,
    )

# ================================================================
# ESPOSIZIONI CORRENTI COMBINED (usate in più sezioni)
# ================================================================
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


# ================================================================
# HANDLE PULSANTI SIDEBAR
# ================================================================
if send_test_tg:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        st.sidebar.error("❌ Configura TELEGRAM_BOT_TOKEN e TELEGRAM_CHAT_ID nei Secrets.")
    else:
        ok = send_test_message(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
        if ok:
            st.sidebar.success("✅ Messaggio di test inviato!")
        else:
            st.sidebar.error("❌ Invio fallito. Verifica token e chat_id.")

if send_report_now:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        st.sidebar.error("❌ Configura TELEGRAM_BOT_TOKEN e TELEGRAM_CHAT_ID nei Secrets.")
    else:
        last_sh   = float(entropy_feat["shannon_ret"].iloc[-1])
        last_pe   = float(entropy_feat["perm_entropy"].iloc[-1])
        last_diff = float(erg_feat["diff"].iloc[-1])
        threshold = erg_thresh["threshold"]

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
            vix_info         = current_vix_info if vix_available else None,
        )
        ok = send_telegram_message(msg, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
        if ok:
            st.sidebar.success("✅ Report inviato su Telegram!")
        else:
            st.sidebar.error("❌ Invio fallito.")


# ================================================================
# HELPER: Descrizione stato VIX (definita prima dei tab che la usano)
# ================================================================
def _vix_state_description(state: str) -> str:
    descs = {
        "LOW_VIX":    "Complacency di mercato. Volatilità implicita bassa: le opzioni prezzano "
                      "pochi rischi. Regime di quiete, tendenzialmente favorevole a mean reversion.",
        "NORMAL_VIX": "Condizioni di volatilità implicita nella norma storica. "
                      "Nessuna anomalia nel pricing delle opzioni.",
        "HIGH_VIX":   "Fear elevata. Il mercato delle opzioni prezza alta incertezza forward. "
                      "Potenziale per forti movimenti direzionali ma anche alta dispersione.",
    }
    return descs.get(state, "Stato VIX in analisi.")


# ================================================================
# TAB NAVIGATION
# ================================================================
tab_overview, tab_spx_regime, tab_vix, tab_ts_detail, tab_methodology = st.tabs([
    "📊 Overview & Stato Attuale",
    "📈 SPX — Regime Storico",
    "📉 VIX — Volatilità Implicita",
    "🔍 Analisi per Trading System",
    "📚 Metodologia",
])


# ================================================================
# TAB 1 — OVERVIEW & STATO ATTUALE
# ================================================================
with tab_overview:

    # ── KPI SPX attuali ──────────────────────────────────────────
    st.subheader("📡 Condizioni di Mercato — Ultima Candela")
    st.markdown(
        f"Dati al **{last_date.strftime('%d/%m/%Y')}** "
        f"(chiusura mercato US 16:00 ET)"
    )

    last_sh   = float(entropy_feat["shannon_ret"].iloc[-1])
    last_pe   = float(entropy_feat["perm_entropy"].iloc[-1])
    last_diff = float(erg_feat["diff"].iloc[-1])
    threshold = erg_thresh["threshold"]
    sh_pct    = float(entropy_feat["shannon_pctile"].iloc[-1])

    # 6 KPI: SPX, Entropy, Perm.Ent, Ergodicità, VIX, VIX Percentile
    c1, c2, c3, c4, c5, c6 = st.columns(6)

    c1.metric(
        "SPX Close",
        f"${spx_info.get('last_close', 0):,.2f}",
        f"{spx_info.get('day_return', 0):+.2f}%",
    )
    c2.metric(
        "Shannon Entropy",
        f"{last_sh:.3f}",
        f"Regime: {current_entropy_state}",
        delta_color="off",
    )
    c3.metric(
        "Perm. Entropy",
        f"{last_pe:.3f}",
        delta_color="off",
    )
    c4.metric(
        "Ergodicità",
        current_erg_state,
        f"diff: {last_diff:+.5f}",
        delta_color="off",
    )

    if vix_available and not np.isnan(current_vix_info.get("vix_close", float("nan"))):
        c5.metric(
            "VIX Close",
            f"{current_vix_info['vix_close']:.2f}",
            current_vix_info["label"],
            delta_color="off",
        )
        c6.metric(
            "VIX Percentile",
            f"{current_vix_info['vix_pct']:.0f}°",
            f"{current_vix_info['emoji']} {current_vix_info['state'].replace('_',' ')}",
            delta_color="off",
        )
    else:
        c5.metric("VIX", "N/D", "Dati non disponibili", delta_color="off")
        c6.metric("VIX Percentile", "N/D", delta_color="off")

    # Regime composito evidenziato
    regime_color = REGIME_COLORS.get(current_regime, "#2196F3")
    vix_color    = current_vix_info.get("color", "#2E7D32")
    vix_label_ui = current_vix_info.get("label", "N/D")
    vix_emoji_ui = current_vix_info.get("emoji", "⚪")

    col_r, col_v = st.columns(2)
    with col_r:
        st.markdown(
            f"""
            <div style="
                background-color: {regime_color}22;
                border-left: 4px solid {regime_color};
                border-radius: 6px;
                padding: 14px 18px;
                margin: 8px 0;
            ">
                <b style="font-size: 16px;">🎯 Regime Entropia+Erg.: {current_regime}</b><br>
                <span style="color: #CCCCCC; font-size: 13px;">
                    {REGIME_DESCRIPTIONS.get(current_regime, '')}
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_v:
        st.markdown(
            f"""
            <div style="
                background-color: {vix_color}22;
                border-left: 4px solid {vix_color};
                border-radius: 6px;
                padding: 14px 18px;
                margin: 8px 0;
            ">
                <b style="font-size: 16px;">{vix_emoji_ui} VIX State: {vix_label_ui}</b><br>
                <span style="color: #CCCCCC; font-size: 13px;">
                    {_vix_state_description(current_vix_state)}
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()

    # ── Tabella stato corrente tutti i TS ────────────────────────
    st.subheader("💼 Stato Corrente — Tutti i Trading System")
    st.markdown(
        f"Esposizione **COMBINATA** (Regime × VIX) raccomandata per ogni TS. "
        f"Regime: **{current_regime}** | VIX: **{vix_label_ui}**"
    )

    fig_table = build_overview_table(ts_exposures)
    st.plotly_chart(fig_table, width="stretch")

    st.divider()

    # ── Gauge per ogni TS ────────────────────────────────────────
    st.subheader("🎯 Esposizione Combinata per Trading System")

    ts_names = list(trading_systems.keys())
    n_cols   = min(3, len(ts_names))
    cols     = st.columns(n_cols)

    for i, ts_name in enumerate(ts_names):
        exp  = ts_exposures[ts_name]
        with cols[i % n_cols]:
            fig_gauge = build_exposure_gauge(
                ts_name    = ts_name,
                multiplier = exp["final_mult"],
                regime     = f"{current_regime} | {current_vix_state.replace('_',' ')}",
            )
            st.plotly_chart(fig_gauge, width="stretch")

    st.divider()

    # ── Riepilogo dati caricati ───────────────────────────────────
    with st.expander("📋 Riepilogo Trading System caricati"):
        summary_rows = []
        for ts_name, ts_data in trading_systems.items():
            exp = ts_exposures[ts_name]
            summary_rows.append({
                "Trading System":  ts_name,
                "File":            ts_data["filename"],
                "Tipo":            "Short/Copertura" if ts_data["is_short"] else "Long",
                "Dal":             ts_data["start_date"].strftime("%Y-%m-%d"),
                "Al":              ts_data["end_date"].strftime("%Y-%m-%d"),
                "N° Trade":        ts_data["n_trades"],
                "Molt. Regime":    f"×{exp['regime_mult']:.1f}",
                "Molt. VIX":       f"×{exp['vix_mult']:.1f}",
                "Molt. Combined":  f"×{exp['final_mult']:.1f}",
                "Esposizione":     f"{exp['emoji']} {exp['label']}",
            })
        st.dataframe(
            pd.DataFrame(summary_rows).set_index("Trading System"),
            width="stretch",
        )


# ================================================================
# TAB 2 — SPX REGIME STORICO
# ================================================================
with tab_spx_regime:

    st.subheader("📈 Analisi Storica Regime S&P 500")
    st.markdown("""
    Il grafico mostra l'andamento storico delle metriche di **Entropia** ed **Ergodicità**
    calcolate sui prezzi dell'S&P 500.
    """)

    fig_spx = build_spx_regime_chart(entropy_feat, erg_feat, regime_series)
    st.plotly_chart(fig_spx, width="stretch")

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📐 Soglie Entropia")
        st.markdown(f"""
        | Soglia       | Valore |
        |--------------|--------|
        | P33 (Bassa/Media) | `{entropy_thresh['p33_shannon']:.4f}` |
        | P67 (Media/Alta)  | `{entropy_thresh['p67_shannon']:.4f}` |
        | Valore attuale    | `{last_sh:.4f}` |
        | Regime            | **{current_entropy_state}** |
        """)

    with col2:
        st.subheader("⚙️ Soglie Ergodicità (SEM)")
        st.markdown(f"""
        | Parametro         | Valore |
        |-------------------|--------|
        | σ globale         | `{erg_thresh['sigma_global']:.6f}` |
        | SEM (σ/√N)        | `{erg_thresh['sem']:.6f}` |
        | k moltiplicatore  | `{erg_thresh['k_mult']:.2f}` |
        | Soglia (k × SEM)  | `±{erg_thresh['threshold']:.6f}` |
        | % giorni non erg. | `{erg_thresh['pct_non_erg']:.1f}%` |
        | Valore attuale    | `{last_diff:+.6f}` |
        | Stato             | **{current_erg_state}** |
        """)

    st.divider()

    st.subheader("📊 Distribuzione Storica dei Regimi")
    regime_counts = regime_series.value_counts().reset_index()
    regime_counts.columns = ["Regime", "N° Giorni"]
    regime_counts["% Giorni"] = (regime_counts["N° Giorni"] / regime_counts["N° Giorni"].sum() * 100).round(1)
    st.dataframe(
        regime_counts[["Regime", "N° Giorni", "% Giorni"]].set_index("Regime"),
        width="stretch",
    )


# ================================================================
# TAB 3 — VIX — VOLATILITÀ IMPLICITA
# ================================================================
with tab_vix:

    st.subheader("📉 Analisi VIX — Volatilità Implicita (Terzo Layer)")
    st.markdown("""
    Il **VIX** (CBOE Volatility Index) misura la volatilità implicita 30-day delle opzioni
    sull'S&P 500. A differenza di Shannon Entropy (storica) ed Ergodicità (strutturale),
    il VIX cattura la **paura forward-looking** del mercato delle opzioni.

    - Il **rolling percentile** (finestra 252g) normalizza il VIX rispetto alla sua storia
      recente, rendendolo confrontabile in periodi diversi.
    - L'**isteresi** previene transizioni rapide sul confine delle soglie (anti-whipsaw):
      per passare da NORMAL a HIGH serve percentile > 80°, per uscire da HIGH basta < 65°.
    - Il **moltiplicatore VIX** per ogni TS è ottimizzato storicamente allo stesso modo
      del moltiplicatore regime: BOOST / STANDARD / REDUCE / INHIBIT.
    """)

    if vix_available:
        fig_vix = build_vix_chart(vix_features)
        st.plotly_chart(fig_vix, width="stretch")

        st.divider()

        # KPI VIX correnti
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("VIX Close", f"{current_vix_info['vix_close']:.2f}")
        col2.metric(
            "Percentile Rolling",
            f"{current_vix_info['vix_pct']:.1f}°",
            help=f"Posizione del VIX corrente rispetto agli ultimi 252 giorni"
        )
        col3.metric("Stato (isteresi)", current_vix_info["label"])
        col4.metric("Emoji Stato", current_vix_info["emoji"])

        st.divider()

        # Distribuzione storica degli stati VIX
        if "vix_state" in vix_features.columns:
            st.subheader("📊 Distribuzione Storica degli Stati VIX")
            vix_state_counts = vix_features["vix_state"].value_counts().reset_index()
            vix_state_counts.columns = ["Stato VIX", "N° Giorni"]
            vix_state_counts["% Giorni"] = (
                vix_state_counts["N° Giorni"] / vix_state_counts["N° Giorni"].sum() * 100
            ).round(1)
            vix_state_counts["Descrizione"] = vix_state_counts["Stato VIX"].map(VIX_STATE_LABELS)
            st.dataframe(
                vix_state_counts.set_index("Stato VIX"),
                width="stretch",
            )
    else:
        st.warning(
            "⚠️ Dati VIX non disponibili. Verifica la chiave EODHD_API_KEY "
            "e che il ticker VIX.INDX sia accessibile con il tuo piano."
        )


# ================================================================
# TAB 4 — ANALISI PER TRADING SYSTEM
# ================================================================
with tab_ts_detail:

    st.subheader("🔍 Analisi Dettagliata per Trading System")

    if not trading_systems:
        st.warning("Nessun Trading System disponibile.")
    else:
        selected_ts = st.selectbox(
            "Seleziona Trading System",
            options=list(trading_systems.keys()),
            format_func=lambda x: f"{'⚠️' if trading_systems[x]['is_short'] else '📈'} {x}",
        )

        ts_data      = trading_systems[selected_ts]
        opt_result   = opt_results.get(selected_ts, {})
        vix_opt_res  = vix_opt_results.get(selected_ts, {})
        exp_curr     = ts_exposures[selected_ts]

        # Info TS
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Tipo",            "Short/Copertura" if ts_data["is_short"] else "Long")
        col2.metric("N° Trade",        f"{ts_data['n_trades']:,}")
        col3.metric("Periodo",         f"{ts_data['start_date'].strftime('%Y')} – {ts_data['end_date'].strftime('%Y')}")
        col4.metric("Molt. Regime",    f"×{exp_curr['regime_mult']:.1f}")
        col5.metric(
            "Molt. Finale (R×V)",
            f"{exp_curr['emoji']} ×{exp_curr['final_mult']:.1f}",
            f"VIX: ×{exp_curr['vix_mult']:.1f}",
            delta_color="off",
        )

        if ts_data["is_short"]:
            st.info(
                "⚠️ **Sistema Short/Copertura**: la logica di esposizione si applica invariata. "
                "Il PnL è già il netto del sistema, indipendentemente dalla direzione."
            )

        st.divider()

        # ── Grafico Equity Curve (3 scenari) ─────────────────────
        st.subheader("📈 Equity Curve — 3 Scenari")
        st.markdown(
            "**Arancio tratteggiato** = Baseline (×1.0) | "
            "**Blu tratteggiato** = Regime-Adjusted | "
            "**Verde** = Combined (Regime × VIX)"
        )

        if selected_ts in equity_curves:
            eq_df  = equity_curves[selected_ts]
            fig_eq = build_equity_comparison_chart(eq_df, selected_ts)
            st.plotly_chart(fig_eq, width="stretch")

            # Metriche comparative
            perf = compute_performance_comparison(eq_df)
            if perf:
                st.subheader("📊 Performance Comparata")
                has_vix_col = "vix_adjusted_equity" in eq_df.columns

                if has_vix_col:
                    c1, c2, c3 = st.columns(3)
                    baseline = perf["baseline"]
                    adjusted = perf["adjusted"]

                    c1.metric("PnL Baseline",         f"${baseline['total_pnl']:,.0f}")
                    c2.metric(
                        "PnL Regime-Adj.",
                        f"${adjusted['total_pnl']:,.0f}",
                        f"{perf['improvement']:+.1f}% vs Baseline",
                        delta_color="normal" if perf["improvement"] >= 0 else "inverse",
                    )

                    # Metriche VIX-combined
                    vix_total = float(eq_df["vix_adjusted_equity"].iloc[-1])
                    vix_impr  = (
                        (vix_total - baseline["total_pnl"]) / abs(baseline["total_pnl"]) * 100
                        if baseline["total_pnl"] != 0 else 0.0
                    )
                    c3.metric(
                        "PnL Combined (R+V)",
                        f"${vix_total:,.0f}",
                        f"{vix_impr:+.1f}% vs Baseline",
                        delta_color="normal" if vix_impr >= 0 else "inverse",
                    )
                else:
                    c1, c2 = st.columns(2)
                    baseline = perf["baseline"]
                    adjusted = perf["adjusted"]
                    c1.metric("PnL Baseline",    f"${baseline['total_pnl']:,.0f}")
                    c2.metric(
                        "PnL Regime-Adj.",
                        f"${adjusted['total_pnl']:,.0f}",
                        f"{perf['improvement']:+.1f}%",
                        delta_color="normal" if perf["improvement"] >= 0 else "inverse",
                    )
        else:
            st.warning("Equity curve non disponibile per questo TS.")

        st.divider()

        # ── Heatmap PnL per Regime ────────────────────────────────
        st.subheader("🔥 Mean PnL per Regime Entropia+Ergodicità")
        regime_stats = opt_result.get("regime_stats", pd.DataFrame())
        if not regime_stats.empty:
            fig_hm = build_regime_heatmap(regime_stats, selected_ts)
            st.plotly_chart(fig_hm, width="stretch")

        # ── Statistiche VIX per TS ────────────────────────────────
        if vix_available and vix_opt_res:
            st.divider()
            st.subheader("📊 Performance per Stato VIX")
            vix_stats = vix_opt_res.get("vix_stats", pd.DataFrame())

            if not vix_stats.empty:
                vix_rows = []
                vix_rules = vix_opt_res.get("vix_rules", {})
                for state in VIX_STATES:
                    if state not in vix_stats.index:
                        continue
                    row  = vix_stats.loc[state]
                    mult = vix_rules.get(state, 1.0)
                    n    = int(row.get("n_trades", 0))
                    mpnl = row.get("mean_pnl", float("nan"))
                    wr   = row.get("win_rate", float("nan"))

                    vix_rows.append({
                        "Stato VIX":    VIX_STATE_LABELS.get(state, state),
                        "N° Trade":     n,
                        "Mean PnL ($)": f"${mpnl:,.0f}" if not pd.isna(mpnl) else "N/D",
                        "Win Rate (%)": f"{wr:.1f}%" if not pd.isna(wr) else "N/D",
                        "Molt. VIX":    f"×{mult:.1f}",
                        "Esposizione":  f"{EXPOSURE_EMOJIS.get(mult,'🟡')} {EXPOSURE_LABELS.get(mult,'STANDARD')}",
                        "Attivo?":      "✅" if state == current_vix_state else "",
                    })

                if vix_rows:
                    st.dataframe(
                        pd.DataFrame(vix_rows).set_index("Stato VIX"),
                        width="stretch",
                    )

        st.divider()

        # ── Distribuzione PnL per Regime ──────────────────────────
        st.subheader("📦 Distribuzione PnL per Regime (Box Plot)")
        fig_box = build_pnl_distribution_chart(
            trades        = ts_data["trades"],
            regime_series = regime_series,
            ts_name       = selected_ts,
        )
        st.plotly_chart(fig_box, width="stretch")

        st.divider()

        # ── Tabella regole di esposizione regime ──────────────────
        st.subheader("⚖️ Regole di Esposizione — Regime Entropia+Ergodicità")

        rules = opt_result.get("exposure_rules", {})
        if not regime_stats.empty and rules:
            rule_rows = []
            for regime in list(regime_stats.index):
                row  = regime_stats.loc[regime]
                mult = rules.get(regime, 1.0)
                n    = int(row.get("n_trades", 0)) if hasattr(row, "get") else 0
                mpnl = row.get("mean_pnl", float("nan")) if hasattr(row, "get") else float("nan")
                wpnl = row.get("win_rate", float("nan")) if hasattr(row, "get") else float("nan")

                rule_rows.append({
                    "Regime":          regime,
                    "N° Trade":        n,
                    "Mean PnL ($)":    f"${mpnl:,.0f}" if not pd.isna(mpnl) else "N/D",
                    "Win Rate (%)":    f"{wpnl:.1f}%" if not pd.isna(wpnl) else "N/D",
                    "Molt. Regime":    f"×{mult:.1f}",
                    "Esposizione":     f"{EXPOSURE_EMOJIS.get(mult,'🟡')} {EXPOSURE_LABELS.get(mult,'STANDARD')}",
                    "Attivo?":         "✅" if regime == current_regime else "",
                })

            if rule_rows:
                st.dataframe(
                    pd.DataFrame(rule_rows).set_index("Regime"),
                    width="stretch",
                )

        with st.expander("ℹ️ Come sono calcolate le regole?"):
            overall = opt_result.get("overall_mean_pnl", 0.0)
            n_total = opt_result.get("n_trades_total", 0)
            st.markdown(f"""
            **Trading System:** `{selected_ts}`
            - PnL medio globale: **${overall:,.2f}** su {n_total} trade
            - Trade minimi per regime: **{min_trades}** (impostabile nella sidebar)
            - Soglia BOOST: **{boost_ratio:.1f}×** la media globale
            - Soglia STANDARD: **{standard_ratio:.2f}×** la media globale

            **Logica di assegnazione (identica per Regime e VIX):**
            1. < {min_trades} trade nel regime → **STANDARD** (dati insufficienti)
            2. Mean PnL ≥ {boost_ratio:.1f}× media globale → **BOOST** (×1.5)
            3. Mean PnL ≥ {standard_ratio:.2f}× media globale → **STANDARD** (×1.0)
            4. Mean PnL ≥ $0 → **REDUCE** (×0.5)
            5. Mean PnL < $0 → **INHIBIT** (×0.0)

            **Moltiplicatore finale:** snap(Regime × VIX) ∈ {{0, 0.5, 1.0, 1.5}}
            """)


# ================================================================
# TAB 5 — METODOLOGIA
# ================================================================
with tab_methodology:

    st.subheader("📚 Metodologia — Basi Teoriche")

    with st.expander("📐 Studio 1 — Shannon Entropy (Regime Entropia)", expanded=True):
        st.markdown(f"""
        La **Shannon Entropy** misura l'imprevedibilità statistica della distribuzione
        dei log-return:

        $$H = -\\sum_i p_i \\cdot \\log_2(p_i)$$

        Calcolata su una finestra rolling di **63 giorni** (≈ 1 trimestre) sui log-return
        giornalieri dell'S&P 500. La serie viene discretizzata in **10 bin**.

        **Soglie regime:** tertili della distribuzione storica di H
        - Regime **BASSA**: H ≤ P33 = `{entropy_thresh['p33_shannon']:.4f}`
        - Regime **MEDIA**: P33 < H ≤ P67 = `{entropy_thresh['p67_shannon']:.4f}`
        - Regime **ALTA**:  H > P67

        *Riferimento: Shannon, C.E. (1948). A Mathematical Theory of Communication.*
        """)

    with st.expander("🔀 Studio 2 — Permutation Entropy"):
        st.markdown(f"""
        La **Permutation Entropy** (Bandt & Pompe, 2002) misura la complessità
        strutturale della sequenza temporale analizzando i pattern ordinali.
        Normalizzata in [0, 1], calcolata con embedding dimension m = 3,
        finestra rolling di 63 giorni.

        *Riferimento: Bandt & Pompe (2002). Permutation Entropy. PRL 88, 174102.*
        """)

    with st.expander("⚙️ Studio 3 — Ergodicità (Standard Error of the Mean)"):
        st.markdown(f"""
        L'**Ergodicità** misura se la media temporale (rolling) converge alla media
        di lungo periodo (expanding). Soglia = k × σ/√N.

        Con: σ = `{erg_thresh['sigma_global']:.6f}`, N = 252, k = `{erg_thresh['k_mult']:.2f}`,
        Soglia = `±{erg_thresh['threshold']:.6f}`.

        *Riferimento: Peters, O. (2019). The ergodicity problem in economics. Nature Physics 15.*
        """)

    with st.expander("📉 Studio 4 — VIX Percentile + Isteresi (NUOVO)", expanded=True):
        st.markdown(f"""
        Il **VIX** (CBOE Volatility Index) misura la volatilità implicita delle opzioni
        sull'S&P 500 a 30 giorni. A differenza dei primi due studi (storici/statistici),
        il VIX cattura la paura **forward-looking** del mercato delle opzioni.

        **Rolling Percentile (finestra 252g):** normalizza il VIX rispetto alla sua storia
        recente. Un VIX a 25 in un regime tranquillo (pct 85°) è molto diverso da un VIX a 25
        durante un periodo volatile (pct 40°).

        **Isteresi anti-whipsaw:** previene transizioni rapide vicino alla soglia usando
        due threshold per ogni confine:
        - HIGH_VIX: entra quando percentile > **{HIGH_VIX_ENTRY_PCT:.0f}°**, esce quando < **{HIGH_VIX_EXIT_PCT:.0f}°**
        - LOW_VIX:  entra quando percentile < **{LOW_VIX_ENTRY_PCT:.0f}°**, esce quando > **{LOW_VIX_EXIT_PCT:.0f}°**
        - Le transizioni LOW↔HIGH passano obbligatoriamente per NORMAL

        **Ottimizzazione per TS:** identica logica al regime entropia+ergodicità.
        Per ogni TS, il sistema calcola il mean PnL in ciascuno dei 3 stati VIX
        e assegna BOOST/STANDARD/REDUCE/INHIBIT.

        **Moltiplicatore finale:** snap(regime_mult × vix_mult)
        Snap-points: 0.25→INHIBIT, 0.75→REDUCE, 1.25→STANDARD/BOOST
        """)

    with st.expander("🎯 Ottimizzazione Esposizione per Trading System"):
        st.markdown(f"""
        Per ogni TS, il sistema calcola la performance in ognuno dei **6 regimi**
        (3 Entropia × 2 Ergodicità) e nei **3 stati VIX** separatamente, poi combina:

        | Condizione                                  | Moltiplicatore | Esposizione |
        |---------------------------------------------|---------------|-------------|
        | Mean PnL ≥ {boost_ratio:.1f}× media globale | **×1.5**      | 🟢 BOOST    |
        | Mean PnL ≥ {standard_ratio:.2f}× media globale | **×1.0**  | 🟡 STANDARD |
        | Mean PnL ≥ $0                               | **×0.5**      | 🟠 RIDOTTO  |
        | Mean PnL < $0                               | **×0.0**      | 🔴 INIBITO  |

        Il **moltiplicatore finale** = snap(molt_regime × molt_vix):
        - BOOST(1.5) × REDUCE(0.5) = 0.75 → STANDARD(1.0)
        - BOOST(1.5) × INHIBIT(0.0) = 0 → INHIBIT(0.0)
        - STANDARD(1.0) × BOOST(1.5) = 1.5 → BOOST(1.5)
        """)

    with st.expander("📲 Notifica Telegram Giornaliera"):
        st.markdown("""
        Il report Telegram viene inviato automaticamente tramite `notify.py`
        ogni giorno di mercato (lunedì-venerdì) alle **16:30 ET** (22:30 CEST estate).

        Il messaggio include: SPX close, Shannon/Perm Entropy, Ergodicità, **VIX state**,
        e per ogni TS il dettaglio Regime×VIX = Combined.

        **Setup cron (Linux/Mac):**
        ```bash
        30 21 * * 1-5 cd /path/to/kriterion-regime-filter && python notify.py
        ```
        **Invio manuale:** usa il pulsante "📤 Invia Report Ora" nella sidebar.
        """)
