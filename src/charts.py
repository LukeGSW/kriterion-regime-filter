"""
charts.py
=========
Funzioni per la creazione di grafici Plotly professionali (dark theme).

Tutti i grafici condividono la palette colori e il layout base definiti
in COLORS / _base_layout().
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .equity_loader import EXPOSURE_COLORS, EXPOSURE_LABELS
from .regime_engine  import REGIME_COLORS, ALL_REGIMES


# ================================================================
# PALETTE COLORI E LAYOUT BASE
# ================================================================

COLORS = {
    "primary":    "#2196F3",   # blu — linee principali
    "secondary":  "#FF9800",   # arancio — linee secondarie / baseline
    "positive":   "#4CAF50",   # verde — profitti / boost
    "negative":   "#F44336",   # rosso — perdite / inibito
    "neutral":    "#9E9E9E",   # grigio — riferimenti
    "background": "#1E1E2E",   # sfondo scuro
    "surface":    "#2A2A3E",   # pannelli / card
    "text":       "#E0E0E0",   # testo principale
    "accent":     "#AB47BC",   # viola — indicatori speciali
    "inhibit":    "#F44336",
    "reduce":     "#FF9800",
    "standard":   "#2196F3",
    "boost":      "#4CAF50",
}


def _base_layout(
    title:    str = "",
    x_title:  str = "",
    y_title:  str = "",
    height:   int = 450,
) -> dict:
    """Layout Plotly condiviso da tutti i grafici."""
    return dict(
        title=dict(
            text=title,
            font=dict(size=16, color=COLORS["text"]),
        ),
        paper_bgcolor=COLORS["background"],
        plot_bgcolor=COLORS["surface"],
        font=dict(color=COLORS["text"], family="Inter, Arial, sans-serif"),
        xaxis=dict(
            title=x_title,
            showgrid=True,
            gridcolor="#333355",
            zeroline=False,
            color=COLORS["text"],
        ),
        yaxis=dict(
            title=y_title,
            showgrid=True,
            gridcolor="#333355",
            zeroline=False,
            color=COLORS["text"],
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            bordercolor="#444466",
            font=dict(size=12),
        ),
        hovermode="x unified",
        margin=dict(l=60, r=20, t=60, b=60),
        height=height,
    )


# ================================================================
# GRAFICO 1 — Equity Curve Baseline vs Adjusted
# ================================================================

def build_equity_comparison_chart(
    equity_df: pd.DataFrame,
    ts_name:   str,
) -> go.Figure:
    """
    Grafico a doppio pannello: equity curve + esposizione storica.

    Pannello superiore (70%): Equity curve baseline (arancio) vs adjusted (blu).
    Pannello inferiore (30%): Serie storica del moltiplicatore di esposizione
                              (step chart colorato per livello).

    Args:
        equity_df: DataFrame da build_equity_curves()
        ts_name:   Nome del Trading System (per il titolo)

    Returns:
        go.Figure con doppio pannello
    """
    if equity_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="Nessun dato disponibile",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color=COLORS["neutral"]),
        )
        fig.update_layout(**_base_layout(f"{ts_name} — Nessun dato"))
        return fig

    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.70, 0.30],
        shared_xaxes=True,
        vertical_spacing=0.04,
        subplot_titles=["Equity Curve (USD)", "Esposizione per Regime"],
    )

    # ── Pannello 1: Equity Curves ────────────────────────────────
    fig.add_trace(
        go.Scatter(
            x=equity_df.index,
            y=equity_df["baseline_equity"],
            name="Baseline (×1.0)",
            line=dict(color=COLORS["secondary"], width=1.8, dash="dot"),
            hovertemplate="<b>Baseline</b>: $%{y:,.0f}<extra></extra>",
        ),
        row=1, col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=equity_df.index,
            y=equity_df["adjusted_equity"],
            name="Adjusted (regime-filtered)",
            line=dict(color=COLORS["primary"], width=2.2),
            hovertemplate="<b>Adjusted</b>: $%{y:,.0f}<extra></extra>",
        ),
        row=1, col=1,
    )

    # Linea dello zero
    fig.add_hline(y=0, line_dash="solid", line_color=COLORS["neutral"],
                  line_width=0.8, row=1, col=1)

    # ── Pannello 2: Esposizione ──────────────────────────────────
    exposure = equity_df["exposure_level"]
    colors_map = {
        0.0: COLORS["inhibit"],
        0.5: COLORS["reduce"],
        1.0: COLORS["standard"],
        1.5: COLORS["boost"],
    }

    # Raggruppa per run-length encoding per visualizzare bande colorate
    for mult, color in colors_map.items():
        mask = exposure.round(2) == round(mult, 2)
        if not mask.any():
            continue
        y_vals = exposure.copy()
        y_vals[~mask] = np.nan

        fig.add_trace(
            go.Scatter(
                x=y_vals.index,
                y=y_vals,
                name=EXPOSURE_LABELS.get(mult, str(mult)),
                mode="markers",
                marker=dict(color=color, size=6, symbol="square"),
                hovertemplate=f"<b>{EXPOSURE_LABELS.get(mult, str(mult))}</b> (×{mult})<extra></extra>",
                showlegend=True,
            ),
            row=2, col=1,
        )

    fig.update_layout(
        **_base_layout(
            title=f"📈 {ts_name} — Equity Baseline vs Adjusted",
            height=550,
        )
    )
    fig.update_yaxes(title_text="Equity (USD)", row=1, col=1,
                     color=COLORS["text"])
    fig.update_yaxes(title_text="Moltiplicatore", row=2, col=1,
                     range=[-0.1, 1.8], color=COLORS["text"])
    fig.update_xaxes(color=COLORS["text"])

    # Formatta il subplot title
    for ann in fig.layout.annotations:
        ann.update(font=dict(color=COLORS["text"], size=13))

    return fig


# ================================================================
# GRAFICO 2 — Heatmap Performance per Regime
# ================================================================

def build_regime_heatmap(
    regime_stats: pd.DataFrame,
    ts_name:      str,
) -> go.Figure:
    """
    Heatmap del mean_pnl per regime (3 entropia × 2 ergodicità).

    Args:
        regime_stats: DataFrame da compute_regime_stats(), indice=regime
        ts_name:      Nome del TS (per il titolo)

    Returns:
        go.Figure heatmap
    """
    from .regime_engine import ENTROPY_STATES, ERGODICITY_STATES

    if regime_stats.empty:
        fig = go.Figure()
        fig.update_layout(**_base_layout(f"{ts_name} — Nessun dato"))
        return fig

    # Riorganizza in matrice 3×2
    matrix = np.full((len(ENTROPY_STATES), len(ERGODICITY_STATES)), np.nan)
    n_matrix = np.zeros((len(ENTROPY_STATES), len(ERGODICITY_STATES)), dtype=int)

    for i, ent in enumerate(ENTROPY_STATES):
        for j, erg in enumerate(ERGODICITY_STATES):
            regime_key = f"{ent}|{erg}"
            if regime_key in regime_stats.index:
                val = regime_stats.loc[regime_key, "mean_pnl"]
                n   = regime_stats.loc[regime_key, "n_trades"]
                if not np.isnan(val):
                    matrix[i, j]   = val
                    n_matrix[i, j] = int(n)

    # Testo delle celle: PnL + n_trades
    text_matrix = [
        [
            f"${matrix[i, j]:,.0f}<br>({n_matrix[i, j]} trades)"
            if not np.isnan(matrix[i, j])
            else "N/D<br>(0 trades)"
            for j in range(len(ERGODICITY_STATES))
        ]
        for i in range(len(ENTROPY_STATES))
    ]

    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=ERGODICITY_STATES,
        y=ENTROPY_STATES,
        text=text_matrix,
        texttemplate="%{text}",
        textfont=dict(size=13, color="white"),
        colorscale="RdYlGn",
        zmid=0,
        colorbar=dict(
            title=dict(
                text="Mean PnL (USD)",
                font=dict(color=COLORS["text"]),
            ),
            tickfont=dict(color=COLORS["text"]),
        ),
        hovertemplate="Entropia: %{y}<br>Ergodicità: %{x}<br>Mean PnL: $%{z:,.0f}<extra></extra>",
    ))

    fig.update_layout(
        **_base_layout(
            title=f"🔥 {ts_name} — Mean PnL per Regime",
            x_title="Stato Ergodicità",
            y_title="Regime Entropia",
            height=350,
        )
    )

    return fig


# ================================================================
# GRAFICO 3 — SPX con Regime Overlay
# ================================================================

def build_spx_regime_chart(
    entropy_feat:  pd.DataFrame,
    erg_feat:      pd.DataFrame,
    regime_series: pd.Series,
) -> go.Figure:
    """
    Grafico SPX con overlay del regime (bande colorate di sfondo).

    Mostra l'andamento della Shannon Entropy e dell'Ergodicità (diff)
    con bande colorate che indicano il regime corrente.

    Args:
        entropy_feat:  DataFrame da compute_entropy_features()
        erg_feat:      DataFrame da compute_ergodicity_features()
        regime_series: Serie storica dei regimi

    Returns:
        go.Figure con 3 pannelli: Prezzi SPX, Entropy, Ergodicità diff
    """
    fig = make_subplots(
        rows=3, cols=1,
        row_heights=[0.50, 0.25, 0.25],
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=["S&P 500 (SPX) — Prezzi", "Shannon Entropy (63g)", "Ergodicità — Diff (rolling − expanding)"],
    )

    # Pannello 1: SPX Price
    if "close" in entropy_feat.columns:
        fig.add_trace(
            go.Scatter(
                x=entropy_feat.index,
                y=entropy_feat["close"],
                name="SPX Close",
                line=dict(color=COLORS["primary"], width=1.5),
                hovertemplate="SPX: %{y:,.0f}<extra></extra>",
            ),
            row=1, col=1,
        )

    # Pannello 2: Shannon Entropy
    if "shannon_ret" in entropy_feat.columns:
        fig.add_trace(
            go.Scatter(
                x=entropy_feat.index,
                y=entropy_feat["shannon_ret"],
                name="Shannon Entropy",
                line=dict(color=COLORS["accent"], width=1.5),
                hovertemplate="Entropy: %{y:.3f}<extra></extra>",
            ),
            row=2, col=1,
        )

    # Pannello 3: Ergodicity diff
    if "diff" in erg_feat.columns:
        threshold = erg_feat.attrs.get("threshold", 0.001)
        fig.add_trace(
            go.Scatter(
                x=erg_feat.index,
                y=erg_feat["diff"],
                name="Erg. Diff",
                line=dict(color=COLORS["secondary"], width=1.5),
                hovertemplate="Diff: %{y:.5f}<extra></extra>",
            ),
            row=3, col=1,
        )
        # Bande threshold
        fig.add_hline(y= threshold, line_dash="dash", line_color=COLORS["negative"],
                      line_width=0.8, row=3, col=1)
        fig.add_hline(y=-threshold, line_dash="dash", line_color=COLORS["negative"],
                      line_width=0.8, row=3, col=1)
        fig.add_hline(y=0, line_color=COLORS["neutral"], line_width=0.5, row=3, col=1)

    fig.update_layout(
        **_base_layout(
            title="📊 Analisi Regime SPX — Entropia & Ergodicità",
            height=650,
        )
    )
    for ann in fig.layout.annotations:
        ann.update(font=dict(color=COLORS["text"], size=12))
    fig.update_xaxes(color=COLORS["text"])
    fig.update_yaxes(color=COLORS["text"])

    return fig


# ================================================================
# GRAFICO 4 — Esposizione corrente (gauge/bar orizzontale)
# ================================================================

def build_exposure_gauge(
    ts_name:     str,
    multiplier:  float,
    regime:      str,
) -> go.Figure:
    """
    Indicatore di esposizione corrente per un TS (grafico a gauge).

    Args:
        ts_name:    Nome del TS
        multiplier: Moltiplicatore corrente (0.0 / 0.5 / 1.0 / 1.5)
        regime:     Regime corrente (es. 'Bassa|Ergodico')

    Returns:
        go.Figure con indicatore gauge
    """
    label = EXPOSURE_LABELS.get(multiplier, "STANDARD")
    color = EXPOSURE_COLORS.get(multiplier, "#2196F3")

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=multiplier,
        title=dict(
            text=f"<b>{ts_name}</b><br><sub>{label} — {regime}</sub>",
            font=dict(size=13, color=COLORS["text"]),
        ),
        number=dict(
            suffix="×",
            font=dict(size=28, color=color),
        ),
        gauge=dict(
            axis=dict(
                range=[0, 1.5],
                tickvals=[0, 0.5, 1.0, 1.5],
                ticktext=["INIBITO", "RIDOTTO", "STANDARD", "BOOST"],
                tickfont=dict(size=10, color=COLORS["text"]),
            ),
            bar=dict(color=color, thickness=0.6),
            bgcolor=COLORS["surface"],
            bordercolor=COLORS["background"],
            steps=[
                dict(range=[0, 0.5],   color="#3E1515"),
                dict(range=[0.5, 1.0], color="#3E2E10"),
                dict(range=[1.0, 1.5], color="#102E1E"),
            ],
            threshold=dict(
                line=dict(color=color, width=3),
                thickness=0.8,
                value=multiplier,
            ),
        ),
        domain=dict(x=[0, 1], y=[0, 1]),
    ))

    fig.update_layout(
        paper_bgcolor=COLORS["background"],
        font=dict(color=COLORS["text"]),
        height=220,
        margin=dict(l=20, r=20, t=60, b=20),
    )

    return fig


# ================================================================
# GRAFICO 5 — Distribuzione PnL per Regime
# ================================================================

def build_pnl_distribution_chart(
    trades:        pd.DataFrame,
    regime_series: pd.Series,
    ts_name:       str,
) -> go.Figure:
    """
    Box plot del PnL per regime (6 box plot affiancati).

    Args:
        trades:        DataFrame dei trade con 'entry_date', 'pnl'
        regime_series: Serie storica dei regimi
        ts_name:       Nome del TS

    Returns:
        go.Figure con box plot per ogni regime
    """
    from .regime_engine import map_trades_to_regimes

    if trades.empty:
        fig = go.Figure()
        fig.update_layout(**_base_layout(f"{ts_name} — Nessun dato"))
        return fig

    t = trades.copy()
    t["regime"] = map_trades_to_regimes(t, regime_series)

    fig = go.Figure()
    for regime in ALL_REGIMES:
        subset = t[t["regime"] == regime]["pnl"]
        if subset.empty:
            continue
        color = REGIME_COLORS.get(regime, COLORS["neutral"])
        fig.add_trace(go.Box(
            y=subset,
            name=regime.replace("|", "\n"),
            boxmean=True,
            marker_color=color,
            line_color=color,
            hovertemplate=f"<b>{regime}</b><br>PnL: %{{y:,.0f}}<extra></extra>",
        ))

    fig.add_hline(y=0, line_color=COLORS["neutral"], line_width=1, line_dash="dash")

    fig.update_layout(
        **_base_layout(
            title=f"📦 {ts_name} — Distribuzione PnL per Regime",
            y_title="PnL (USD)",
            height=400,
        )
    )

    return fig


# ================================================================
# GRAFICO 6 — Tabella riepilogo esposizione corrente (tutti i TS)
# ================================================================

def build_overview_table(
    ts_exposures: dict[str, dict],
) -> go.Figure:
    """
    Tabella Plotly con lo stato di esposizione corrente di tutti i TS.

    Args:
        ts_exposures: Dizionario {ts_name: exposure_dict}
                      dove exposure_dict è da get_current_exposure()

    Returns:
        go.Figure con tabella
    """
    rows = []
    for ts_name, exp in ts_exposures.items():
        rows.append({
            "Trading System": ts_name,
            "Regime":         exp.get("regime", "N/D"),
            "Esposizione":    exp.get("label", "N/D"),
            "Moltiplicatore": f"{exp.get('multiplier', 1.0):.1f}×",
            "Stato":          exp.get("emoji", "🟡"),
        })

    if not rows:
        fig = go.Figure()
        fig.update_layout(**_base_layout("Nessun TS disponibile"))
        return fig

    df = pd.DataFrame(rows)

    # Colori celle per la colonna Esposizione
    cell_colors_exposure = [
        EXPOSURE_COLORS.get(
            ts_exposures[r["Trading System"]].get("multiplier", 1.0),
            "#2196F3",
        )
        for r in rows
    ]

    fill_colors = [
        [COLORS["surface"]] * len(rows),     # col Trading System
        [COLORS["surface"]] * len(rows),     # col Regime
        cell_colors_exposure,                # col Esposizione (colorata)
        [COLORS["surface"]] * len(rows),     # col Moltiplicatore
        [COLORS["surface"]] * len(rows),     # col Stato
    ]

    fig = go.Figure(go.Table(
        header=dict(
            values=["<b>Trading System</b>", "<b>Regime</b>",
                    "<b>Esposizione</b>", "<b>Moltiplicatore</b>", "<b>Stato</b>"],
            fill_color=COLORS["background"],
            font=dict(color=COLORS["text"], size=13),
            align="left",
            line_color="#444466",
            height=35,
        ),
        cells=dict(
            values=[
                df["Trading System"].tolist(),
                df["Regime"].tolist(),
                df["Esposizione"].tolist(),
                df["Moltiplicatore"].tolist(),
                df["Stato"].tolist(),
            ],
            fill_color=fill_colors,
            font=dict(color=COLORS["text"], size=12),
            align="left",
            line_color="#333355",
            height=30,
        ),
    ))

    fig.update_layout(
        paper_bgcolor=COLORS["background"],
        font=dict(color=COLORS["text"]),
        margin=dict(l=0, r=0, t=30, b=0),
        height=max(200, 40 * len(rows) + 60),
    )

    return fig
