"""Country profile page — static render from URL parameters, printable from browser."""

import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Input, Output, clientside_callback, dcc, html, register_page

from .. import data
from ..features.country_utils import get_un_membership_years

P5 = ["USA", "GBR", "FRA", "RUS", "CHN"]
MIN_JOINT_VOTES = 100

register_page(__name__, path="/profile")

# ── shared style constants ────────────────────────────────────────────────────
_H3 = {"fontSize": "1rem", "fontWeight": "600", "color": "#343a40", "margin": "0 0 10px 0"}
_TH = {
    "padding": "6px 8px",
    "borderBottom": "1px solid #dee2e6",
    "textAlign": "left",
    "fontSize": "0.8em",
    "fontWeight": "600",
    "color": "#495057",
}


def _section(title, *children):
    return html.Div(
        [html.H3(title, style=_H3), *children],
        style={"marginBottom": "28px"},
    )


def _stat_tile(label, value, mb="14px"):
    return html.Div([
        html.Div(label, style={"fontSize": "0.8rem", "color": "#6c757d", "marginBottom": "2px"}),
        html.Div(value, style={"fontSize": "1.5rem", "fontWeight": "700", "color": "#212529"}),
    ], style={"marginBottom": mb})


def _res_table(df, title):
    _TD = {"padding": "4px 8px", "fontSize": "0.82em", "verticalAlign": "top"}
    rows = [html.Tr([html.Th(title, colSpan=3, style=_TH)])]
    for _, row in df.iterrows():
        date_str    = row["date"].strftime("%Y") if pd.notnull(row.get("date")) else ""
        raw_title   = row["title"] or ""
        short_title = raw_title[:200] + "…" if len(raw_title) > 200 else raw_title
        title_cell  = (
            html.A(
                short_title,
                href=row["undl_link"],
                target="_blank",
                style={"color": "#333", "textDecoration": "none"},
            )
            if pd.notnull(row.get("undl_link")) else short_title
        )
        rows.append(html.Tr([
            html.Td(date_str,   style={**_TD, "color": "#999", "whiteSpace": "nowrap"}),
            html.Td(title_cell, style=_TD),
            html.Td(
                f"{row['consensus_score']:.2f}",
                style={**_TD, "color": "#666", "textAlign": "right", "whiteSpace": "nowrap"},
            ),
        ]))
    return html.Table(rows, style={"borderCollapse": "collapse", "width": "100%"})


def _rank_table(series, title, footnote=None):
    _TD = {"padding": "4px 8px", "fontSize": "0.85em"}
    rows = [html.Tr([html.Th(title, colSpan=2, style=_TH)])]
    for iso3, score in series.items():
        rows.append(html.Tr([
            html.Td(data.get_country_name(iso3), style=_TD),
            html.Td(f"{score:.3f}", style={**_TD, "color": "#666", "textAlign": "right"}),
        ]))
    children = [html.Table(rows, style={"borderCollapse": "collapse", "width": "100%"})]
    if footnote:
        children.append(
            html.P(footnote, style={"fontSize": "0.75em", "color": "#999", "margin": "4px 0 0 0"})
        )
    return html.Div(children)


# ── layout ────────────────────────────────────────────────────────────────────

def layout(
    country1=None,
    start_date=None,
    end_date=None,
    country2=None,
    **kwargs,
):
    if not country1:
        return html.Div(
            "No country selected. Please return to the main page and select a country.",
            style={"padding": "40px", "color": "#666"},
        )

    # ── clamp year range to UN membership period ──────────────────────────────
    membership = get_un_membership_years(country1)
    if membership:
        mem_start = f"{membership[0]}-01-01"
        mem_end   = f"{membership[1]}-12-31"
        if start_date is None or start_date < mem_start:
            start_date = mem_start
        if end_date is None or end_date > mem_end:
            end_date = mem_end

    comparison_raw = [
        c for c in (country2.split(",") if country2 else [])
        if c in data.available_countries and c != country1
    ]
    comparison    = comparison_raw or P5
    country1_name = data.get_country_name(country1)

    # ── query resolutions ─────────────────────────────────────────────────────
    res_df = data.query_engine.query_resolutions(
        start_date=start_date,
        end_date=end_date,
    )

    if res_df.empty:
        return html.Div(
            "No resolutions found for the current filters.",
            style={"padding": "40px", "color": "#666"},
        )

    res_ids = res_df["undl_id"].tolist()

    # ── vote data for country1 ────────────────────────────────────────────────
    res_table = data.query_engine.resolution_table
    cols    = [c for c in ["undl_id", "date", "consensus_score", country1] if c in res_table.columns]
    vote_df = res_table[res_table["undl_id"].isin(res_ids)][cols].copy()
    vote_df["date"] = pd.to_datetime(vote_df["date"])
    vote_df["year"] = vote_df["date"].dt.year

    # ── multilateral alignment score + rank ───────────────────────────────────
    ml_stats = data.query_engine.query_multilateral_stats(res_ids)
    multilateral_rank_str = None
    if not ml_stats.empty:
        ml_valid = ml_stats.dropna(subset=["multilateral_alignment"]).copy()
        ml_valid["_rank"] = (
            ml_valid["multilateral_alignment"].rank(ascending=False, method="min").astype(int)
        )
        ml_row = ml_valid[ml_valid["country"] == country1]
        if not ml_row.empty:
            score = float(ml_row["multilateral_alignment"].iloc[0])
            rank  = int(ml_row["_rank"].iloc[0])
            multilateral_rank_str = f"{score:.3f} ({rank} of {len(ml_valid)})"

    # ── vote breakdown counts + segment bar ───────────────────────────────────
    total    = len(vote_df)
    n_yes    = int((vote_df[country1] == "Y").sum())
    n_no     = int((vote_df[country1] == "N").sum())
    n_abs    = int((vote_df[country1] == "A").sum())
    n_absent = total - n_yes - n_no - n_abs

    def _pct(n):
        return n / total * 100 if total else 0

    y_pct, n_pct, a_pct, x_pct = _pct(n_yes), _pct(n_no), _pct(n_abs), _pct(n_absent)

    vote_bar = html.Div([
        html.Div(style={"width": f"{y_pct:.2f}%", "backgroundColor": "#74bb88", "height": "100%"}),
        html.Div(style={"width": f"{n_pct:.2f}%", "backgroundColor": "#cc575f", "height": "100%"}),
        html.Div(style={"width": f"{a_pct:.2f}%", "backgroundColor": "#c19f5b", "height": "100%"}),
        html.Div(style={"width": f"{x_pct:.2f}%", "backgroundColor": "#ced4da", "height": "100%"}),
    ], style={
        "display": "flex",
        "height": "18px",
        "borderRadius": "999px",
        "overflow": "hidden",
        "backgroundColor": "#e5e7eb",
    })

    vote_legend = html.Div([
        html.Span([html.Span("■ ", style={"color": "#74bb88"}), f"Yes: {y_pct:.1f}%"]),
        html.Span([html.Span("■ ", style={"color": "#cc575f"}), f"No: {n_pct:.1f}%"]),
        html.Span([html.Span("■ ", style={"color": "#c19f5b"}), f"Abstain: {a_pct:.1f}%"]),
        html.Span([html.Span("■ ", style={"color": "#adb5bd"}), f"Did not vote: {x_pct:.1f}%"]),
    ], style={
        "display": "flex",
        "flexDirection": "column",
        "gap": "4px",
        "marginTop": "8px",
        "fontSize": "0.8rem",
        "color": "#495057",
    })

    # ── bilateral agreement (ranking tables + alignment over time) ────────────
    scores_df = data.query_engine.query_agreement_between_countries(
        country1, resolution_ids=res_ids, average=False
    )
    score_country_cols = [c for c in scores_df.columns if c != "undl_id"]
    joint_counts = scores_df[score_country_cols].notna().sum()
    qualified    = joint_counts[joint_counts >= MIN_JOINT_VOTES].index

    avg_by_country = scores_df[score_country_cols].mean().dropna().sort_values(ascending=False)
    avg_qualified  = avg_by_country[avg_by_country.index.isin(qualified)]
    top_aligned    = avg_qualified.head(10)
    bottom_aligned = avg_qualified.tail(10).sort_values()

    valid_comp = [c for c in comparison if c in scores_df.columns]
    yearly = (
        scores_df.merge(vote_df[["undl_id", "year"]], on="undl_id", how="left")
                 .groupby("year")[valid_comp].mean()
        if valid_comp else pd.DataFrame()
    )

    colors    = px.colors.qualitative.D3
    align_fig = go.Figure()
    for i, c in enumerate(valid_comp):
        align_fig.add_trace(go.Scatter(
            x=yearly.index,
            y=yearly[c],
            mode="lines+markers",
            name=data.get_country_name(c),
            line=dict(color=colors[i % len(colors)]),
            marker=dict(size=5),
        ))
    align_fig.update_layout(
        height=300,
        margin=dict(l=10, r=10, t=30, b=10),
        template="plotly_white",
        yaxis=dict(range=[0, 1.05], title="Agreement"),
        xaxis=dict(title="Year"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )

    # ── notable votes ─────────────────────────────────────────────────────────
    merged_df = vote_df.merge(
        res_df[["undl_id", "title", "resolution", "undl_link"]],
        on="undl_id", how="left"
    )
    detail_df = merged_df.dropna(subset=["consensus_score", country1]
    )
    opposed_hi = (
        detail_df[detail_df[country1].isin(["N", "A"])]
        .sort_values("consensus_score", ascending=False)
        .head(5)
    )
    supported_lo = (
        detail_df[detail_df[country1] == "Y"]
        .sort_values("consensus_score", ascending=True)
        .head(5)
    )

    # ── header metadata ───────────────────────────────────────────────────────
    year_start = (start_date or "")[:4]
    year_end   = (end_date   or "")[:4]
    selected_years_val = f"{year_start}–{year_end}" if (year_start and year_end) else "all"

    if membership:
        latest_year   = datetime.date.today().year
        mem_end_label = "present" if membership[1] >= latest_year - 1 else str(membership[1])
        membership_str = f"UN Member: {membership[0]}–{mem_end_label}"
    else:
        membership_str = ""

    align_title = (
        "Bilateral Agreement Over Time (vs. P5)"
        if not comparison_raw else
        "Bilateral Agreement Over Time (vs. Selected Countries)"
    )
    subregion = data.get_country_subregion(country1)

    # ── assemble page ─────────────────────────────────────────────────────────
    return html.Div([
        html.Div(id="profile-print-dummy", style={"display": "none"}),

        # print button
        html.Div(
            html.Button(
                "Print / Save as PDF",
                id="profile-print-btn",
                n_clicks=0,
                style={
                    "border": "none",
                    "borderRadius": "4px",
                    "padding": "8px 18px",
                    "backgroundColor": "#1a73e8",
                    "color": "white",
                    "cursor": "pointer",
                    "fontSize": "14px",
                    "fontWeight": "600",
                },
            ),
            className="no-print",
            style={"textAlign": "right", "marginBottom": "16px"},
        ),

        # header
        html.Div([
            html.H1(
                country1_name,
                style={"margin": "0 0 8px 0", "fontSize": "2rem", "color": "#212529"}
            ),
            *(
                [html.Div(
                    subregion,
                    style={"color": "#6c757d", "marginBottom": "4px"}
                )]
                if subregion else []
            ),
            html.Div(
                membership_str,
                style={"color": "#6c757d", "marginRight": "18px"}
                ),
        ], style={"borderBottom": "2px solid #dee2e6", "paddingBottom": "16px", "marginBottom": "24px"}
        ),

        # top row: stats block (left) + ranking tables (right)
        html.Div([
            html.Div([
                _stat_tile("Selected Years",       selected_years_val, mb="16px"),
                _stat_tile("Accepted Resolutions", f"{total:,}"),
                *(
                    [_stat_tile("Multilateral Alignment", multilateral_rank_str)]
                    if multilateral_rank_str is not None else []
                ),
                vote_bar,
                vote_legend,
            ], style={"display": "flex", "flexDirection": "column", "flex": "0 0 280px", "minWidth": "280px"}),

            html.Div(
                _section(
                    "Most & Least Aligned Countries",
                    html.P(
                        f"Shows only countries having at least {MIN_JOINT_VOTES} shared votes with {country1_name}",
                        style={"fontSize": "0.75em", "color": "#999", "margin": "6px 0 12px 0"},
                    ),
                    html.Div([
                        _rank_table(top_aligned,    "Most aligned"),
                        _rank_table(bottom_aligned, "Least aligned"),
                    ], style={"display": "flex", "gap": "60px"}),
                ),
                style={"flex": "1", "minWidth": "0"},
            ),
        ], style={"display": "flex", "gap": "32px", "alignItems": "flex-start", "marginBottom": "28px"}),

        # bilateral alignment over time
        _section(
            align_title,
            dcc.Graph(figure=align_fig, config={"displayModeBar": False}),
        ),

        # notable votes
        _section(
            "Notable Votes",
            html.Div([
                html.Div([_res_table(opposed_hi, "✗  Opposed resolutions — highest consensus first")],
                         style={"flex": "1", "minWidth": "0"}),
                html.Div([_res_table(supported_lo, "✓  Supported resolutions — lowest consensus first")],
                         style={"flex": "1", "minWidth": "0"}),
            ], style={"display": "flex", "gap": "40px"}),
        ),

    ], style={"maxWidth": "1100px", "margin": "0 auto", "padding": "32px 24px", "fontFamily": "inherit"})


# trigger window.print() when the button is clicked
clientside_callback(
    "function(n) { if (n) { window.print(); } return ''; }",
    Output("profile-print-dummy", "children"),
    Input("profile-print-btn", "n_clicks"),
    prevent_initial_call=True,
)
