"""
app.py
======
Kriterion Regime Filter — Dashboard Streamlit principale.

Questa dashboard applica gli studi di Entropia ed Ergodicità sull'indice S&P 500
per determinare il regime di mercato ottimale per ogni Trading System algoritmico.

Per ogni TS, il sistema:
  1. Scarica i file equity da Google Drive (pubblica)
  2. Calcola Shannon Entropy e Permutation Entropy sull'SPX (finestra 63g)
  3. Calcola l'Ergodicità SPX tramite il metodo SEM (finestra 252g)
  4. Identifica le 6 combinazioni di regime (3 Entropia × 2 Ergodicità)
  5. Ottimizza automaticamente le regole di esposizione per ogni TS
  6. Mostra le equity curve baseline vs adjusted e lo stato corrente

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
from src.optimizer      import optimize_all_ts, get_current_exposure, BOOST_RATIO, STANDARD_RATIO
from src.exposure_engine import build_all_equity_curves, compute_performance_comparison
from src.charts         import (
    build_equity_comparison_chart,
    build_regime_heatmap,
    build_spx_regime_chart,
    build_exposure_gauge,
    build_pnl_distribution_chart,
    build_overview_table,
)
from src.telegram_bot   import send_telegram_message, format_daily_report, send_test_message


# ================================================================
# SECRETS & PARAMETRI GLOBALI
# ================================================================
try:
    EODHD_API_KEY      = st.secrets["EODHD_API_KEY"]
    TELEGRAM_BOT_TOKEN = st.secrets.get("TELEGRAM_BOT_TOKEN", "")
    TELEGRAM_CHAT_ID   = st.secrets.get("TELEGRAM_CHAT_ID", "")
except Exception:
    st.error(
        "❌ **Secrets mancanti.** Configura `EODHD_API_KEY`, "
        "`TELEGRAM_BOT_TOKEN` e `TELEGRAM_CHAT_ID` nei Secrets di Streamlit Cloud "
        "o nel file `.streamlit/secrets.toml` in locale."
    )
    st.stop()

DRIVE_FOLDER_ID = "1kc0fu8UB8rZOZrSsfUnNtUcbOLyLx7ps"


# ================================================================
# FUNZIONI CACHED
# ================================================================

@st.cache_data(ttl=3600, show_spinner=False)
def cached_fetch_spx(api_key: str, to_date: str) -> pd.DataFrame:
    return fetch_spx(api_key=api_key, to_date=to_date)


@st.cache_data(ttl=86400, show_spinner=False)
def cached_build_regime_series(spx_hash: str, _spx_df: pd.DataFrame) -> dict:
    """Cache del calcolo regime: si invalida giornalmente (spx_hash include la data)."""
    return build_regime_series(_spx_df)


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
             "Valori più bassi → più regole personalizzate ma con meno dati."
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
Questo sistema applica gli studi di **Entropia** (Shannon + Permutation Entropy)
ed **Ergodicità** (Standard Error of the Mean) sull'indice **S&P 500** per classificare
il contesto di mercato in 6 regimi. Per ogni Trading System algoritmico, il backtest
storico individua i regimi in cui il sistema ha performance significativamente migliori
o peggiori, e regola di conseguenza l'esposizione al capitale:

- 🔴 **INIBITO** (×0): sistema disabilitato nel regime
- 🟠 **RIDOTTO** (×0.5): capitale ridotto del 50%
- 🟡 **STANDARD** (×1.0): esposizione normale
- 🟢 **BOOST** (×1.5): capitale aumentato del 50%

> **Come si usa:** il sistema si aggiorna automaticamente ogni giorno dopo la chiusura
> dell'S&P 500. La colonna "Stato Attuale" mostra l'esposizione raccomandata per oggi.
""")
st.divider()


# ================================================================
# STEP 1 — DOWNLOAD EQUITY DA GOOGLE DRIVE
# ================================================================
equities_dir = os.path.join(tempfile.gettempdir(), "kriterion_equities")

with st.spinner("⏳ Caricamento file equity da Google Drive..."):
    try:
        equities_dir = download_equity_files(
            folder_id       = DRIVE_FOLDER_ID,
            dest_dir        = equities_dir,
            force_redownload = force_reload,
        )
    except RuntimeError as e:
        st.error(f"❌ Errore download Google Drive: {e}")
        st.stop()

trading_systems = cached_load_trading_systems(equities_dir)

if not trading_systems:
    st.error(
        "❌ Nessun file equity trovato nella cartella Google Drive. "
        "Verifica che la cartella contenga file .txt in formato TradeStation."
    )
    st.stop()

# ================================================================
# STEP 2 — FETCH SPX DA EODHD
# ================================================================
today_str = str(date.today())

with st.spinner("⏳ Download dati SPX da EODHD..."):
    try:
        spx_df = cached_fetch_spx(EODHD_API_KEY, today_str)
    except Exception as e:
        st.error(f"❌ Errore fetch SPX: {e}. Verifica la chiave EODHD_API_KEY.")
        st.stop()

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
# STEP 4 — OTTIMIZZAZIONE ESPOSIZIONE PER OGNI TS
# ================================================================
with st.spinner("⏳ Ottimizzazione regole di esposizione per ogni Trading System..."):
    opt_results = optimize_all_ts(
        trading_systems = trading_systems,
        regime_series   = regime_series,
        min_trades      = min_trades,
        boost_ratio     = boost_ratio,
        standard_ratio  = standard_ratio,
    )

# ================================================================
# STEP 5 — EQUITY CURVES ADJUSTED
# ================================================================
with st.spinner("⏳ Costruzione equity curve adjusted..."):
    equity_curves = build_all_equity_curves(
        trading_systems = trading_systems,
        regime_series   = regime_series,
        opt_results     = opt_results,
    )

# ================================================================
# ESPOSIZIONI CORRENTI (usate in più sezioni)
# ================================================================
ts_exposures = {
    ts_name: get_current_exposure(ts_name, opt_results, current_regime)
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
        # Recupera valori correnti per il messaggio
        last_sh  = float(entropy_feat["shannon_ret"].iloc[-1])
        last_pe  = float(entropy_feat["perm_entropy"].iloc[-1])
        last_diff= float(erg_feat["diff"].iloc[-1])
        threshold= erg_thresh["threshold"]

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
        )
        ok = send_telegram_message(msg, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
        if ok:
            st.sidebar.success("✅ Report inviato su Telegram!")
        else:
            st.sidebar.error("❌ Invio fallito.")


# ================================================================
# TAB NAVIGATION
# ================================================================
tab_overview, tab_spx_regime, tab_ts_detail, tab_methodology = st.tabs([
    "📊 Overview & Stato Attuale",
    "📈 SPX — Regime Storico",
    "🔍 Analisi per Trading System",
    "📚 Metodologia",
])


# ================================================================
# TAB 1 — OVERVIEW & STATO ATTUALE
# ================================================================
with tab_overview:

    # ── KPI SPX attuali ──────────────────────────────────────────
    st.subheader("📡 Condizioni di Mercato SPX — Ultima Candela")
    st.markdown(
        f"Dati al **{last_date.strftime('%d/%m/%Y')}** "
        f"(chiusura mercato US 16:00 ET)"
    )

    last_sh  = float(entropy_feat["shannon_ret"].iloc[-1])
    last_pe  = float(entropy_feat["perm_entropy"].iloc[-1])
    last_diff= float(erg_feat["diff"].iloc[-1])
    threshold= erg_thresh["threshold"]
    sh_pct   = float(entropy_feat["shannon_pctile"].iloc[-1])

    c1, c2, c3, c4, c5 = st.columns(5)

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
    c5.metric(
        "Percentile Entropia",
        f"{sh_pct:.0f}°",
        help="Percentile corrente della Shannon Entropy rispetto all'intera storia",
        delta_color="off",
    )

    # Regime attuale evidenziato
    regime_color = REGIME_COLORS.get(current_regime, "#2196F3")
    st.markdown(
        f"""
        <div style="
            background-color: {regime_color}22;
            border-left: 4px solid {regime_color};
            border-radius: 6px;
            padding: 14px 18px;
            margin: 12px 0;
        ">
            <b style="font-size: 18px;">Regime Attuale: {current_regime}</b><br>
            <span style="color: #CCCCCC; font-size: 14px;">
                {REGIME_DESCRIPTIONS.get(current_regime, '')}
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    # ── Tabella stato corrente tutti i TS ────────────────────────
    st.subheader("💼 Stato Corrente — Tutti i Trading System")
    st.markdown(
        "Esposizione raccomandata per ogni Trading System in base al regime attuale "
        f"(**{current_regime}**) e alle regole ottimizzate dal backtest storico."
    )

    fig_table = build_overview_table(ts_exposures)
    st.plotly_chart(fig_table, use_container_width=True)

    st.divider()

    # ── Gauge per ogni TS ────────────────────────────────────────
    st.subheader("🎯 Esposizione per Trading System")

    ts_names = list(trading_systems.keys())
    n_cols   = min(3, len(ts_names))
    cols     = st.columns(n_cols)

    for i, ts_name in enumerate(ts_names):
        exp  = ts_exposures[ts_name]
        with cols[i % n_cols]:
            fig_gauge = build_exposure_gauge(
                ts_name    = ts_name,
                multiplier = exp["multiplier"],
                regime     = current_regime,
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

    st.divider()

    # ── Riepilogo dati caricati ───────────────────────────────────
    with st.expander("📋 Riepilogo Trading System caricati"):
        summary_rows = []
        for ts_name, ts_data in trading_systems.items():
            exp = ts_exposures[ts_name]
            summary_rows.append({
                "Trading System": ts_name,
                "File":           ts_data["filename"],
                "Tipo":           "Short/Copertura" if ts_data["is_short"] else "Long",
                "Dal":            ts_data["start_date"].strftime("%Y-%m-%d"),
                "Al":             ts_data["end_date"].strftime("%Y-%m-%d"),
                "N° Trade":       ts_data["n_trades"],
                "Esposizione":    f"{exp['emoji']} {exp['label']}",
            })
        st.dataframe(
            pd.DataFrame(summary_rows).set_index("Trading System"),
            use_container_width=True,
        )


# ================================================================
# TAB 2 — SPX REGIME STORICO
# ================================================================
with tab_spx_regime:

    st.subheader("📈 Analisi Storica Regime S&P 500")
    st.markdown("""
    Il grafico mostra l'andamento storico delle metriche di **Entropia** ed **Ergodicità**
    calcolate sui prezzi dell'S&P 500:

    - **Shannon Entropy** (pannello centrale): misura la complessità statistica dei
      log-return. Alta entropia = alta imprevedibilità. La soglia p33/p67 divide in
      tre regimi: Bassa / Media / Alta.
    - **Diff Ergodicità** (pannello inferiore): differenza tra media temporale (rolling 252g)
      e media spaziale (expanding). Valori oltre la banda tratteggiata indicano **non ergodicità**
      (divergenza strutturale dal comportamento storico medio).
    """)

    fig_spx = build_spx_regime_chart(entropy_feat, erg_feat, regime_series)
    st.plotly_chart(fig_spx, use_container_width=True)

    st.divider()

    # ── Statistiche soglie ────────────────────────────────────────
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

    # ── Distribuzione regime storica ──────────────────────────────
    st.subheader("📊 Distribuzione Storica dei Regimi")
    regime_counts = regime_series.value_counts().reset_index()
    regime_counts.columns = ["Regime", "N° Giorni"]
    regime_counts["% Giorni"] = (regime_counts["N° Giorni"] / regime_counts["N° Giorni"].sum() * 100).round(1)
    regime_counts["Colore"] = regime_counts["Regime"].map(REGIME_COLORS)
    st.dataframe(
        regime_counts[["Regime", "N° Giorni", "% Giorni"]].set_index("Regime"),
        use_container_width=True,
    )


# ================================================================
# TAB 3 — ANALISI PER TRADING SYSTEM
# ================================================================
with tab_ts_detail:

    st.subheader("🔍 Analisi Dettagliata per Trading System")

    if not trading_systems:
        st.warning("Nessun Trading System disponibile.")
    else:
        # Selezione TS
        selected_ts = st.selectbox(
            "Seleziona Trading System",
            options=list(trading_systems.keys()),
            format_func=lambda x: f"{'⚠️' if trading_systems[x]['is_short'] else '📈'} {x}",
        )

        ts_data    = trading_systems[selected_ts]
        opt_result = opt_results.get(selected_ts, {})
        exp_curr   = ts_exposures[selected_ts]

        # Info TS
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Tipo",         "Short/Copertura" if ts_data["is_short"] else "Long Direzionale")
        col2.metric("N° Trade",     f"{ts_data['n_trades']:,}")
        col3.metric("Periodo",      f"{ts_data['start_date'].strftime('%Y')} – {ts_data['end_date'].strftime('%Y')}")
        col4.metric("Stato Attuale", f"{exp_curr['emoji']} {exp_curr['label']} (×{exp_curr['multiplier']})")

        if ts_data["is_short"]:
            st.info(
                "⚠️ **Sistema Short/Copertura**: questo sistema opera in direzione opposta "
                "al mercato. I regimi ottimali tipicamente divergono da quelli dei sistemi long."
            )

        st.divider()

        # ── Grafico Equity Curve ──────────────────────────────────
        st.subheader("📈 Equity Curve Baseline vs Adjusted")
        st.markdown(
            "Il grafico mostra l'equity cumulata **senza filtro** (arancio tratteggiato) "
            "confrontata con l'equity **adjusted** (blu) dove il PnL di ogni trade è "
            "moltiplicato per il coefficiente del regime SPX all'ingresso."
        )

        if selected_ts in equity_curves:
            eq_df = equity_curves[selected_ts]
            fig_eq = build_equity_comparison_chart(eq_df, selected_ts)
            st.plotly_chart(fig_eq, use_container_width=True)

            # Metriche comparative
            perf = compute_performance_comparison(eq_df)
            if perf:
                st.subheader("📊 Performance Comparata")
                c1, c2, c3, c4 = st.columns(4)
                baseline = perf["baseline"]
                adjusted = perf["adjusted"]

                c1.metric(
                    "PnL Totale Baseline",
                    f"${baseline['total_pnl']:,.0f}",
                )
                c2.metric(
                    "PnL Totale Adjusted",
                    f"${adjusted['total_pnl']:,.0f}",
                    f"{perf['improvement']:+.1f}%",
                    delta_color="normal" if perf["improvement"] >= 0 else "inverse",
                )
                c3.metric(
                    "Max Drawdown Baseline",
                    f"${baseline['max_drawdown']:,.0f}",
                )
                c4.metric(
                    "Max Drawdown Adjusted",
                    f"${adjusted['max_drawdown']:,.0f}",
                    f"{(adjusted['max_drawdown'] - baseline['max_drawdown']) / abs(baseline['max_drawdown']) * 100:+.1f}%"
                    if baseline["max_drawdown"] != 0 else "N/D",
                    delta_color="inverse",  # drawdown: meno negativo = meglio
                )
        else:
            st.warning("Equity curve non disponibile per questo TS.")

        st.divider()

        # ── Heatmap PnL per Regime ────────────────────────────────
        st.subheader("🔥 Mean PnL per Regime (3 Entropia × 2 Ergodicità)")
        st.markdown(
            "La heatmap mostra il PnL medio per trade in ciascuno dei 6 regimi. "
            "Verde = regime favorevole, Rosso = regime sfavorevole. "
            "Il numero in parentesi indica quanti trade storici sono disponibili nel campione."
        )

        regime_stats = opt_result.get("regime_stats", pd.DataFrame())
        if not regime_stats.empty:
            fig_hm = build_regime_heatmap(regime_stats, selected_ts)
            st.plotly_chart(fig_hm, use_container_width=True)

        st.divider()

        # ── Distribuzione PnL per Regime ──────────────────────────
        st.subheader("📦 Distribuzione PnL per Regime (Box Plot)")
        st.markdown(
            "Box plot del PnL per ogni regime: mostra mediana, range interquartile "
            "e outlier. Il punto interno è la media. Permette di identificare non solo "
            "la performance media ma anche la dispersione e i tail risk."
        )
        fig_box = build_pnl_distribution_chart(
            trades        = ts_data["trades"],
            regime_series = regime_series,
            ts_name       = selected_ts,
        )
        st.plotly_chart(fig_box, use_container_width=True)

        st.divider()

        # ── Tabella regole di esposizione ────────────────────────
        st.subheader("⚖️ Regole di Esposizione Ottimizzate")

        rules = opt_result.get("exposure_rules", {})
        if regime_stats.empty or rules:
            rule_rows = []
            for regime in list(regime_stats.index if not regime_stats.empty else []):
                row = regime_stats.loc[regime] if not regime_stats.empty else {}
                mult  = rules.get(regime, 1.0)
                n     = int(row.get("n_trades", 0)) if hasattr(row, "get") else 0
                mpnl  = row.get("mean_pnl", np.nan) if hasattr(row, "get") else np.nan
                wpnl  = row.get("win_rate", np.nan) if hasattr(row, "get") else np.nan

                rule_rows.append({
                    "Regime":           regime,
                    "N° Trade":         n,
                    "Mean PnL ($)":     f"${mpnl:,.0f}" if not pd.isna(mpnl) else "N/D",
                    "Win Rate (%)":     f"{wpnl:.1f}%" if not pd.isna(wpnl) else "N/D",
                    "Moltiplicatore":   f"{mult:.1f}×",
                    "Esposizione":      f"{EXPOSURE_EMOJIS.get(mult,'🟡')} {EXPOSURE_LABELS.get(mult,'STANDARD')}",
                    "Regime Attuale?":  "✅" if regime == current_regime else "",
                })

            if rule_rows:
                st.dataframe(
                    pd.DataFrame(rule_rows).set_index("Regime"),
                    use_container_width=True,
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

            **Logica di assegnazione:**
            1. < {min_trades} trade nel regime → **STANDARD** (dati insufficienti)
            2. Mean PnL ≥ {boost_ratio:.1f}× media globale → **BOOST** (×1.5)
            3. Mean PnL ≥ {standard_ratio:.2f}× media globale → **STANDARD** (×1.0)
            4. Mean PnL ≥ $0 → **REDUCE** (×0.5)
            5. Mean PnL < $0 → **INHIBIT** (×0.0)
            """)


# ================================================================
# TAB 4 — METODOLOGIA
# ================================================================
with tab_methodology:

    st.subheader("📚 Metodologia — Basi Teoriche")

    with st.expander("📐 Studio 1 — Shannon Entropy (Regime Entropia)", expanded=True):
        st.markdown(f"""
        La **Shannon Entropy** misura l'imprevedibilità statistica della distribuzione
        dei log-return:

        $$H = -\\sum_i p_i \\cdot \\log_2(p_i)$$

        Calcolata su una finestra rolling di **63 giorni** (≈ 1 trimestre) sui log-return
        giornalieri dell'S&P 500. La serie temporale viene discretizzata in **10 bin**.

        **Soglie regime:** tertili della distribuzione storica di H
        - Regime **BASSA**: H ≤ P33 = `{entropy_thresh['p33_shannon']:.4f}`
        - Regime **MEDIA**: P33 < H ≤ P67 = `{entropy_thresh['p67_shannon']:.4f}`
        - Regime **ALTA**:  H > P67

        **Interpretazione:** Alta entropia → alta imprevedibilità → mercato più casuale.
        Bassa entropia → struttura nei rendimenti → possibile prevedibilità.

        *Riferimento: Shannon, C.E. (1948). A Mathematical Theory of Communication.*
        """)

    with st.expander("🔀 Studio 2 — Permutation Entropy"):
        st.markdown(f"""
        La **Permutation Entropy** (Bandt & Pompe, 2002) misura la complessità
        strutturale della sequenza temporale analizzando i pattern ordinali:

        $$H_{{perm}} = -\\sum_i p_i \\cdot \\log_2(p_i) \\; / \\; \\log_2(m!)$$

        Normalizzata in [0, 1]. Calcolata con embedding dimension **m = {3}**
        su finestra rolling di **63 giorni**.

        - PE ≈ 1 → alta complessità (serie vicina al random walk)
        - PE << 1 → struttura temporale forte (potenziale prevedibilità)

        *Riferimento: Bandt & Pompe (2002). Permutation Entropy. PRL 88, 174102.*
        """)

    with st.expander("⚙️ Studio 3 — Ergodicità (Standard Error of the Mean)"):
        st.markdown(f"""
        L'**Ergodicità** misura se la media temporale (rolling) converge alla media
        di lungo periodo (expanding). La soglia è il **SEM (Standard Error of the Mean)**:

        $$\\text{{threshold}} = k \\times \\frac{{\\sigma}}{{\\sqrt{{N}}}}$$

        Con: σ = `{erg_thresh['sigma_global']:.6f}` (volatilità globale SPX),
        N = 252 (1 anno trading), k = `{erg_thresh['k_mult']:.2f}` (≈ 92% CI),
        Soglia = `±{erg_thresh['threshold']:.6f}`.

        Il mercato è **NON ERGODICO** quando `|rolling_mean − expanding_mean|` > soglia:
        la stima locale si discosta significativamente dal comportamento storico medio.

        *Riferimento: Peters, O. (2019). The ergodicity problem in economics. Nature Physics 15.*
        """)

    with st.expander("🎯 Ottimizzazione Esposizione per Trading System"):
        st.markdown(f"""
        Per ogni TS, il sistema calcola statisticamente la performance in ognuno dei
        **6 regimi** (3 Entropia × 2 Ergodicità) e assegna un moltiplicatore di esposizione:

        | Condizione                                  | Moltiplicatore | Esposizione |
        |---------------------------------------------|---------------|-------------|
        | Mean PnL ≥ {boost_ratio:.1f}× media globale | **×1.5**      | 🟢 BOOST    |
        | Mean PnL ≥ {standard_ratio:.2f}× media globale | **×1.0**  | 🟡 STANDARD |
        | Mean PnL ≥ $0                               | **×0.5**      | 🟠 RIDOTTO  |
        | Mean PnL < $0                               | **×0.0**      | 🔴 INIBITO  |

        Regimi con meno di **{min_trades} trade** ricevono automaticamente **STANDARD**
        per evitare regole basate su campioni statisticamente insufficienti.

        I sistemi short/copertura applicano la stessa logica: il PnL è già il netto
        del sistema, quindi un PnL positivo in un regime è un segnale favorevole
        indipendentemente dalla direzione di mercato.
        """)

    with st.expander("📲 Notifica Telegram Giornaliera"):
        st.markdown("""
        Il report Telegram viene inviato automaticamente tramite `notify.py`
        ogni giorno di mercato (lunedì-venerdì) alle **16:30 ET** (22:30 CEST estate,
        21:30 CET inverno), dopo la chiusura definitiva dell'S&P 500 e con buffer
        di 30 minuti per garantire che i dati siano aggiornati su EODHD.

        **Setup cron (Linux/Mac):**
        ```bash
        30 21 * * 1-5 cd /path/to/kriterion-regime-filter && python notify.py
        ```
        **Setup Windows Task Scheduler:**
        Vedi il README per le istruzioni complete.

        **Invio manuale:** usa il pulsante "📤 Invia Report Ora" nella sidebar.
        """)
