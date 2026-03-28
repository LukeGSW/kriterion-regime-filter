"""
pages/2_Walk_Forward.py
=======================
Pagina Streamlit per la validazione Walk-Forward dei Trading System.

COME FUNZIONA LA PAGINA
-----------------------
Streamlit scopre automaticamente i file in pages/ e li aggiunge alla
sidebar come pagine separate. Zero modifiche ad app.py richieste.

La pagina:
  1. Carica i dati SPX, VIX ed equity dalla stessa pipeline di app.py
     (usa @st.cache_data per evitare re-fetch inutili)
  2. Esegue il walk-forward optimization con run_all_ts_wf()
  3. Mostra per ogni TS le 3 equity curve confrontabili:
       - Baseline (nessun filtro)
       - WF Regime (filtro entropia+ergodicità out-of-sample)
       - WF Combinato (filtro regime×VIX out-of-sample, se VIX disponibile)
  4. Mostra una tabella comparativa delle metriche di performance
  5. Mostra il numero di punti di ricalibrazione (con tooltip esplicativo)

CONFIGURAZIONE
--------------
La pagina legge le credenziali da st.secrets (Streamlit Cloud) o da
variabili d'ambiente / file .env (esecuzione locale), nello stesso modo
del resto dell'app.

DIPENDENZE NUOVE
----------------
Solo src/wf_optimizer.py (nuovo modulo, nessuna modifica all'esistente).
"""

from __future__ import annotations

import os
import sys
import tempfile

import streamlit as st
import plotly.graph_objects as go
import pandas as pd

# ── Aggiunge la root del progetto al sys.path ────────────────────
# Necessario perché pages/ è una subdirectory: senza questo, gli import
# da src/ fallirebbero con ModuleNotFoundError.
_PAGE_DIR    = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_PAGE_DIR)
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

# ── Import moduli del progetto ───────────────────────────────────
from src.equity_loader import (
    download_equity_files,
    load_all_trading_systems,
    MIN_TRADES_PER_REGIME,
)
from src.spx_data import fetch_spx_no_cache
from src.regime_engine import build_regime_series
from src.optimizer import BOOST_RATIO, STANDARD_RATIO
from src.wf_optimizer import run_all_ts_wf, WF_TRAIN_MIN_TRADES, WF_RETRAIN_EVERY

try:
    from src.vix_modulator import fetch_vix, compute_vix_features
    _VIX_MODULE_AVAILABLE = True
except ImportError:
    _VIX_MODULE_AVAILABLE = False


# ================================================================
# CONFIGURAZIONE PAGINA
# ================================================================

st.set_page_config(
    page_title="Walk-Forward | Kriterion Quant",
    page_icon="🔄",
    layout="wide",
)

# ── Colori tema dark (coerenti con app.py) ───────────────────────
COL_BASELINE = "#607D8B"   # grigio-blu  → baseline
COL_WF_REGIME = "#42A5F5"  # blu chiaro  → WF regime
COL_WF_COMB   = "#66BB6A"  # verde       → WF combinato
COL_BG        = "#0E1117"
COL_PAPER     = "#1A1D23"


# ================================================================
# HELPER: LETTURA CREDENZIALI
# ================================================================

def _get_secret(key: str, default: str = "") -> str:
    """
    Legge una credenziale da st.secrets (Streamlit Cloud) o da
    variabili d'ambiente (locale / cron). Fallback su stringa vuota.
    """
    # st.secrets
    try:
        val = st.secrets.get(key, "")
        if val:
            return str(val)
    except Exception:
        pass
    # Variabile d'ambiente / .env
    return os.environ.get(key, default)


# ================================================================
# CARICAMENTO DATI (cached)
# ================================================================

@st.cache_data(ttl=3600, show_spinner=False)
def _load_trading_systems(drive_folder_id: str) -> dict:
    """Scarica e carica i Trading System da Google Drive."""
    equities_dir = os.path.join(tempfile.gettempdir(), "kriterion_equities_wf")
    equities_dir = download_equity_files(
        folder_id      = drive_folder_id,
        dest_dir       = equities_dir,
        force_redownload = False,
    )
    return load_all_trading_systems(equities_dir)


@st.cache_data(ttl=3600, show_spinner=False)
def _load_spx(api_key: str) -> pd.DataFrame:
    """Fetch dati SPX da EODHD."""
    return fetch_spx_no_cache(api_key)


@st.cache_data(ttl=3600, show_spinner=False)
def _load_vix(api_key: str) -> "pd.DataFrame | None":
    """Fetch e calcolo feature VIX. Restituisce None se non disponibile."""
    if not _VIX_MODULE_AVAILABLE:
        return None
    try:
        vix_raw = fetch_vix(api_key)
        if vix_raw.empty:
            return None
        return compute_vix_features(vix_raw)
    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def _build_regime(_spx_df_hash: str, spx_df: pd.DataFrame) -> dict:
    """Costruisce la serie storica dei regimi SPX."""
    # _spx_df_hash è usato solo per il caching (st.cache_data non hash i DataFrame)
    return build_regime_series(spx_df)


@st.cache_data(ttl=3600, show_spinner=False)
def _run_wf(
    ts_names_key: str,  # usato per hash del cache
    trading_systems: dict,
    regime_series: pd.Series,
    vix_features: "pd.DataFrame | None",
    train_min_trades: int,
    retrain_every: int,
) -> dict:
    """Esegue il walk-forward su tutti i TS (operazione costosa, cachata)."""
    return run_all_ts_wf(
        trading_systems  = trading_systems,
        regime_series    = regime_series,
        vix_features     = vix_features,
        min_trades       = MIN_TRADES_PER_REGIME,
        boost_ratio      = BOOST_RATIO,
        standard_ratio   = STANDARD_RATIO,
        train_min_trades = train_min_trades,
        retrain_every    = retrain_every,
    )


# ================================================================
# COMPONENTI DI VISUALIZZAZIONE
# ================================================================

def _plot_equity_curves(ts_name: str, wf_result: dict) -> go.Figure:
    """
    Crea il grafico Plotly con le 3 equity curve per un TS.

    Mostra:
      - Baseline (grigio): nessun filtro, riferimento puro
      - WF Regime (blu): filtro regime entropia+ergodicità out-of-sample
      - WF Combinato (verde): filtro regime×VIX out-of-sample (se disponibile)
    """
    daily    = wf_result["wf_equity_df"]
    has_vix  = wf_result.get("has_vix", False)

    fig = go.Figure()

    # ── Baseline ────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x    = daily.index,
        y    = daily["baseline_equity"],
        name = "Baseline (no filtro)",
        line = dict(color=COL_BASELINE, width=1.5, dash="dot"),
        hovertemplate = "<b>Baseline</b><br>Data: %{x|%d/%m/%Y}<br>Equity: $%{y:,.0f}<extra></extra>",
    ))

    # ── WF Regime ────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x    = daily.index,
        y    = daily["wf_regime_equity"],
        name = "WF Regime (out-of-sample)",
        line = dict(color=COL_WF_REGIME, width=2.0),
        hovertemplate = "<b>WF Regime</b><br>Data: %{x|%d/%m/%Y}<br>Equity: $%{y:,.0f}<extra></extra>",
    ))

    # ── WF Combinato (Regime × VIX) ──────────────────────────────
    if has_vix and "wf_combined_equity" in daily.columns:
        fig.add_trace(go.Scatter(
            x    = daily.index,
            y    = daily["wf_combined_equity"],
            name = "WF Regime × VIX (out-of-sample)",
            line = dict(color=COL_WF_COMB, width=2.0),
            hovertemplate = "<b>WF Regime×VIX</b><br>Data: %{x|%d/%m/%Y}<br>Equity: $%{y:,.0f}<extra></extra>",
        ))

    # ── Linea zero ───────────────────────────────────────────────
    fig.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.15)", line_width=1)

    fig.update_layout(
        title        = dict(
            text     = f"<b>{ts_name}</b> — Equity Curve Walk-Forward",
            font     = dict(size=16, color="#FFFFFF"),
            x        = 0.0,
        ),
        plot_bgcolor  = COL_BG,
        paper_bgcolor = COL_PAPER,
        font          = dict(color="#CCCCCC", size=12),
        legend        = dict(
            orientation = "h",
            yanchor     = "bottom",
            y           = 1.02,
            xanchor     = "left",
            x           = 0,
            bgcolor      = "rgba(0,0,0,0)",
        ),
        xaxis = dict(
            gridcolor    = "rgba(255,255,255,0.07)",
            showgrid     = True,
            zeroline     = False,
        ),
        yaxis = dict(
            title        = "PnL cumulato (USD)",
            gridcolor    = "rgba(255,255,255,0.07)",
            showgrid     = True,
            zeroline     = False,
            tickprefix   = "$",
            tickformat   = ",.0f",
        ),
        hovermode = "x unified",
        margin    = dict(l=60, r=20, t=60, b=40),
        height    = 420,
    )

    return fig


def _build_performance_table(wf_result: dict) -> pd.DataFrame:
    """
    Costruisce la tabella comparativa delle metriche di performance.

    Righe: Baseline / WF Regime / WF Combinato (se disponibile)
    Colonne: PnL Totale / Max Drawdown / Sharpe / Miglioramento %
    """
    perf     = wf_result.get("performance", {})
    has_vix  = wf_result.get("has_vix", False)

    rows = []

    def _fmt_pnl(v):
        return f"${v:,.0f}" if v == v else "N/D"  # N/D se NaN

    def _fmt_dd(v):
        return f"${v:,.0f}" if v == v else "N/D"

    def _fmt_sharpe(v):
        return f"{v:.2f}" if v == v else "N/D"

    def _fmt_impr(v):
        sign = "+" if v >= 0 else ""
        return f"{sign}{v:.1f}%"

    if "baseline" in perf:
        b = perf["baseline"]
        rows.append({
            "Curva":          "Baseline (no filtro)",
            "PnL Totale":     _fmt_pnl(b["total_pnl"]),
            "Max Drawdown":   _fmt_dd(b["max_drawdown"]),
            "Sharpe":         _fmt_sharpe(b["sharpe"]),
            "Miglioramento":  "—",
        })

    if "wf_regime" in perf:
        r = perf["wf_regime"]
        rows.append({
            "Curva":          "WF Regime (out-of-sample)",
            "PnL Totale":     _fmt_pnl(r["total_pnl"]),
            "Max Drawdown":   _fmt_dd(r["max_drawdown"]),
            "Sharpe":         _fmt_sharpe(r["sharpe"]),
            "Miglioramento":  _fmt_impr(r.get("improvement_pct", 0.0)),
        })

    if has_vix and "wf_combined" in perf:
        c = perf["wf_combined"]
        rows.append({
            "Curva":          "WF Regime × VIX (out-of-sample)",
            "PnL Totale":     _fmt_pnl(c["total_pnl"]),
            "Max Drawdown":   _fmt_dd(c["max_drawdown"]),
            "Sharpe":         _fmt_sharpe(c["sharpe"]),
            "Miglioramento":  _fmt_impr(c.get("improvement_pct", 0.0)),
        })

    return pd.DataFrame(rows)


# ================================================================
# MAIN PAGE
# ================================================================

def main() -> None:
    """Entry point della pagina Streamlit Walk-Forward."""

    # ── Header ──────────────────────────────────────────────────
    st.title("🔄 Walk-Forward Analysis")
    st.markdown(
        """
        Valutazione **out-of-sample** dei filtri di regime.
        A differenza dell'analisi in-sample di app.py, qui i moltiplicatori
        di esposizione sono calibrati **solo sui trade passati** rispetto a
        ciascun punto della curva — eliminando il look-ahead bias.
        """
    )

    with st.expander("ℹ️ Come funziona il Walk-Forward", expanded=False):
        st.markdown(
            """
            **Finestra espandente (anchored):**
            Per il trade *i*, le regole di esposizione vengono calibrate usando
            esclusivamente i trade *0 … i−1*. I primi `Train Min Trades` trade
            usano moltiplicatore neutro 1× (cold start).

            **Ricalibrazione:**
            Le regole vengono aggiornate ogni `Retrain Every` trade.
            Tra una ricalibrazione e la successiva, le regole restano invariate.

            **3 curve confrontabili:**
            - 🔘 **Baseline** — nessun filtro (riferimento)
            - 🔵 **WF Regime** — filtro Entropia+Ergodicità out-of-sample
            - 🟢 **WF Combinato** — filtro Regime×VIX out-of-sample (se VIX attivo)

            Una curva WF significativamente migliore della baseline conferma che
            il filtro ha **edge reale** e non è un artefatto di overfitting.
            """
        )

    st.divider()

    # ── Lettura credenziali ──────────────────────────────────────
    eodhd_key       = _get_secret("EODHD_API_KEY")
    drive_folder_id = _get_secret(
        "DRIVE_FOLDER_ID",
        default="1kc0fu8UB8rZOZrSsfUnNtUcbOLyLx7ps",
    )

    if not eodhd_key:
        st.error(
            "❌ **EODHD_API_KEY** non configurata. "
            "Aggiungila in `.streamlit/secrets.toml` o come variabile d'ambiente."
        )
        st.stop()

    # ── Parametri WF nella sidebar ───────────────────────────────
    st.sidebar.header("⚙️ Parametri Walk-Forward")
    train_min = st.sidebar.slider(
        "Train Min Trades",
        min_value = 20,
        max_value = 60,
        value     = WF_TRAIN_MIN_TRADES,
        step      = 5,
        help      = "Numero minimo di trade storici prima di avviare la calibrazione.",
    )
    retrain_every = st.sidebar.slider(
        "Retrain Every (trade)",
        min_value = 5,
        max_value = 40,
        value     = WF_RETRAIN_EVERY,
        step      = 5,
        help      = "Ogni quanti trade si ricalibrano le regole di esposizione.",
    )

    # ── Caricamento dati ─────────────────────────────────────────
    with st.spinner("📥 Caricamento dati in corso…"):
        try:
            trading_systems = _load_trading_systems(drive_folder_id)
        except Exception as exc:
            st.error(f"❌ Errore download equity: {exc}")
            st.stop()

        if not trading_systems:
            st.warning("⚠️ Nessun Trading System trovato. Controlla la cartella Google Drive.")
            st.stop()

        try:
            spx_df = _load_spx(eodhd_key)
        except Exception as exc:
            st.error(f"❌ Errore fetch SPX: {exc}")
            st.stop()

        if spx_df.empty:
            st.error("❌ Dati SPX vuoti. Verifica la chiave EODHD.")
            st.stop()

        # VIX (non bloccante: se fallisce, la pagina funziona senza)
        vix_features = _load_vix(eodhd_key)
        if vix_features is None:
            st.sidebar.warning("⚠️ VIX non disponibile. Solo curva WF Regime.")

        # Regime series
        spx_hash   = str(spx_df.index[-1])  # chiave per cache
        regime_data = _build_regime(spx_hash, spx_df)
        regime_series = regime_data["regime_series"]

    # ── Walk-Forward Optimization ─────────────────────────────────
    ts_names_key = ",".join(sorted(trading_systems.keys()))
    with st.spinner("⚙️ Calcolo Walk-Forward in corso (può richiedere qualche secondo)…"):
        wf_results = _run_wf(
            ts_names_key     = ts_names_key,
            trading_systems  = trading_systems,
            regime_series    = regime_series,
            vix_features     = vix_features,
            train_min_trades = train_min,
            retrain_every    = retrain_every,
        )

    # ── Selezione TS ─────────────────────────────────────────────
    valid_ts = [
        name for name, res in wf_results.items()
        if "error" not in res and not res["wf_equity_df"].empty
    ]
    error_ts = [
        name for name, res in wf_results.items()
        if "error" in res
    ]

    if not valid_ts:
        st.error("❌ Nessun TS ha abbastanza trade per il walk-forward.")
        if error_ts:
            for name in error_ts:
                st.warning(f"**{name}**: {wf_results[name]['error']}")
        st.stop()

    if error_ts:
        with st.expander(f"⚠️ {len(error_ts)} TS esclusi (trade insufficienti)", expanded=False):
            for name in error_ts:
                st.write(f"- **{name}**: {wf_results[name]['error']}")

    selected_ts = st.selectbox(
        "Seleziona Trading System",
        options = valid_ts,
        index   = 0,
    )

    wf_res = wf_results[selected_ts]

    # ── Metriche riepilogo ────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Trade totali",         wf_res["n_trades_total"])
    col2.metric("Punti ricalibrazione", wf_res["n_retraining_points"])
    col3.metric("Cold start (trade)",   wf_res["train_min_trades"])
    col4.metric("Retrain ogni",         f"{wf_res['retrain_every']} trade")

    st.divider()

    # ── Grafico equity curve ──────────────────────────────────────
    fig = _plot_equity_curves(selected_ts, wf_res)
    st.plotly_chart(fig, use_container_width=True)

    # ── Tabella performance ───────────────────────────────────────
    st.subheader("📊 Metriche Comparative")

    perf_df = _build_performance_table(wf_res)

    # Stile colore sulla colonna Miglioramento
    def _color_improvement(val: str) -> str:
        if val == "—":
            return ""
        try:
            num = float(val.replace("%", "").replace("+", ""))
            if num > 0:
                return "color: #66BB6A; font-weight: bold"
            elif num < 0:
                return "color: #EF5350; font-weight: bold"
        except ValueError:
            pass
        return ""

    styled = perf_df.style.applymap(
        _color_improvement,
        subset=["Miglioramento"],
    ).set_properties(**{
        "background-color": COL_PAPER,
        "color": "#EEEEEE",
        "border-color": "#333333",
    })

    st.dataframe(styled, use_container_width=True, hide_index=True)

    # ── Interpretazione ───────────────────────────────────────────
    perf = wf_res.get("performance", {})
    wf_impr = perf.get("wf_regime", {}).get("improvement_pct", None)

    if wf_impr is not None:
        if wf_impr > 5:
            st.success(
                f"✅ Il filtro WF Regime mostra un **miglioramento reale** del {wf_impr:+.1f}% "
                "rispetto alla baseline out-of-sample. Il filtro ha edge statistico."
            )
        elif wf_impr > 0:
            st.info(
                f"🔵 Il filtro WF Regime mostra un miglioramento marginale del {wf_impr:+.1f}%. "
                "Verifica con una finestra temporale più lunga."
            )
        else:
            st.warning(
                f"⚠️ Il filtro WF Regime non migliora la baseline out-of-sample ({wf_impr:+.1f}%). "
                "Possibile overfitting nei parametri. Considera di aumentare `Train Min Trades`."
            )

    # ── Tutti i TS: tabella riepilogativa ────────────────────────
    st.divider()
    st.subheader("📋 Riepilogo Tutti i Trading System")

    summary_rows = []
    for ts_name in valid_ts:
        r     = wf_results[ts_name]
        perf  = r.get("performance", {})
        b_pnl = perf.get("baseline",  {}).get("total_pnl", float("nan"))
        r_pnl = perf.get("wf_regime", {}).get("total_pnl", float("nan"))
        r_impr= perf.get("wf_regime", {}).get("improvement_pct", float("nan"))
        c_pnl = perf.get("wf_combined", {}).get("total_pnl", float("nan")) if r["has_vix"] else None
        c_impr= perf.get("wf_combined", {}).get("improvement_pct", float("nan")) if r["has_vix"] else None

        row = {
            "Trading System":   ts_name,
            "Baseline PnL":     f"${b_pnl:,.0f}" if b_pnl == b_pnl else "N/D",
            "WF Regime PnL":    f"${r_pnl:,.0f}" if r_pnl == r_pnl else "N/D",
            "WF Regime Δ%":     f"{r_impr:+.1f}%" if r_impr == r_impr else "N/D",
            "WF Comb. PnL":     f"${c_pnl:,.0f}" if (c_pnl is not None and c_pnl == c_pnl) else "—",
            "WF Comb. Δ%":      f"{c_impr:+.1f}%" if (c_impr is not None and c_impr == c_impr) else "—",
            "Ricalibr.":        r["n_retraining_points"],
        }
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)


# ================================================================
# ENTRY POINT
# ================================================================

if __name__ == "__main__":
    main()
else:
    # Streamlit esegue il modulo direttamente al caricamento della pagina
    main()
