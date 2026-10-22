"""Country profile page — static render from URL parameters, printable from browser."""

import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Input, Output, clientside_callback, dcc, html, register_page

from .. import data
from ..features.resolution_list import VOTE_MAP
from ..un_data_stream.analysis import profile
from ..un_data_stream.analysis.profile import MIN_JOINT_VOTES

register_page(__name__, path="/profile")

# ── vote colours (sourced from VOTE_MAP so they stay in sync with the rest of the UI) ──
_COLOR_YES = VOTE_MAP["Y"]["color"]
_COLOR_NO = VOTE_MAP["N"]["color"]
_COLOR_ABS = VOTE_MAP["A"]["color"]
_COLOR_ABSENT = VOTE_MAP["X"]["color"]

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
    return html.Div(
        [
            html.Div(
                label, style={"fontSize": "0.8rem", "color": "#6c757d", "marginBottom": "2px"}
            ),
            html.Div(value, style={"fontSize": "1.5rem", "fontWeight": "700", "color": "#212529"}),
        ],
        style={"marginBottom": mb},
    )


def _res_table(df, title):
    _TD = {"padding": "4px 8px", "fontSize": "0.82em", "verticalAlign": "top"}
    rows = [html.Tr([html.Th(title, colSpan=3, style=_TH)])]
    for _, row in df.iterrows():
        date_str = row["date"].strftime("%Y") if pd.notnull(row.get("date")) else ""
        raw_title = row["title"] or ""
        short_title = raw_title[:200] + "…" if len(raw_title) > 200 else raw_title
        title_cell = (
            html.A(
                short_title,
                href=row["undl_link"],
                target="_blank",
                style={"color": "#333", "textDecoration": "none"},
            )
            if pd.notnull(row.get("undl_link"))
            else short_title
        )
        rows.append(
            html.Tr(
                [
                    html.Td(date_str, style={**_TD, "color": "#999", "whiteSpace": "nowrap"}),
                    html.Td(title_cell, style=_TD),
                    html.Td(
                        f"{row['consensus_score']:.2f}",
                        style={
                            **_TD,
                            "color": "#666",
                            "textAlign": "right",
                            "whiteSpace": "nowrap",
                        },
                    ),
                ]
            )
        )
    return html.Table(rows, style={"borderCollapse": "collapse", "width": "100%"})


def _rank_table(series, title, footnote=None):
    _TD = {"padding": "4px 8px", "fontSize": "0.85em"}
    rows = [html.Tr([html.Th(title, colSpan=2, style=_TH)])]
    for iso3, score in series.items():
        rows.append(
            html.Tr(
                [
                    html.Td(data.get_country_name(iso3), style=_TD),
                    html.Td(f"{score:.3f}", style={**_TD, "color": "#666", "textAlign": "right"}),
                ]
            )
        )
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

    membership = data.get_participation_year_range(country1)
    country1_name = data.get_country_name(country1)

    prof = profile.country_profile(
        data.query_engine,
        country1,
        start_date=start_date,
        end_date=end_date,
        comparison=country2.split(",") if country2 else None,
        participation=membership,
    )
    if prof is None:
        return html.Div(
            "No resolutions found for the current filters.",
            style={"padding": "40px", "color": "#666"},
        )
    start_date, end_date = prof.start_date, prof.end_date

    multilateral_rank_str = None
    if prof.multilateral_alignment is not None:
        multilateral_rank_str = (
            f"{prof.multilateral_alignment:.3f} "
            f"({prof.multilateral_rank} of {prof.multilateral_ranked})"
        )

    # ── vote breakdown counts + segment bar ───────────────────────────────────
    total = prof.total_resolutions
    n_yes, n_no, n_abs, n_absent = (prof.votes[k] for k in ("Y", "N", "A", "absent"))

    def _pct(n):
        return n / total * 100 if total else 0

    y_pct, n_pct, a_pct, x_pct = _pct(n_yes), _pct(n_no), _pct(n_abs), _pct(n_absent)

    vote_bar = html.Div(
        [
            html.Div(
                style={"width": f"{y_pct:.2f}%", "backgroundColor": _COLOR_YES, "height": "100%"}
            ),
            html.Div(
                style={"width": f"{n_pct:.2f}%", "backgroundColor": _COLOR_NO, "height": "100%"}
            ),
            html.Div(
                style={"width": f"{a_pct:.2f}%", "backgroundColor": _COLOR_ABS, "height": "100%"}
            ),
            html.Div(
                style={"width": f"{x_pct:.2f}%", "backgroundColor": _COLOR_ABSENT, "height": "100%"}
            ),
        ],
        style={
            "display": "flex",
            "height": "18px",
            "borderRadius": "999px",
            "overflow": "hidden",
            "backgroundColor": "#e5e7eb",
        },
    )

    vote_legend = html.Div(
        [
            html.Span([html.Span("■ ", style={"color": _COLOR_YES}), f"Yes: {y_pct:.1f}%"]),
            html.Span([html.Span("■ ", style={"color": _COLOR_NO}), f"No: {n_pct:.1f}%"]),
            html.Span([html.Span("■ ", style={"color": _COLOR_ABS}), f"Abstain: {a_pct:.1f}%"]),
            html.Span([html.Span("■ ", style={"color": "#adb5bd"}), f"Did not vote: {x_pct:.1f}%"]),
        ],
        style={
            "display": "flex",
            "flexDirection": "column",
            "gap": "4px",
            "marginTop": "8px",
            "fontSize": "0.8rem",
            "color": "#495057",
        },
    )

    # ── bilateral agreement (ranking tables + alignment over time) ────────────
    top_aligned = prof.most_aligned
    bottom_aligned = prof.least_aligned
    yearly = prof.yearly_alignment
    valid_comp = list(yearly.columns)

    colors = px.colors.qualitative.D3
    align_fig = go.Figure()
    for i, c in enumerate(valid_comp):
        align_fig.add_trace(
            go.Scatter(
                x=yearly.index,
                y=yearly[c],
                mode="lines+markers",
                name=data.get_country_name(c),
                line=dict(color=colors[i % len(colors)]),
                marker=dict(size=5),
            )
        )
    align_fig.update_layout(
        height=300,
        margin=dict(l=10, r=10, t=30, b=10),
        template="plotly_white",
        yaxis=dict(range=[0, 1.05], title="Agreement"),
        xaxis=dict(title="Year"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )

    # ── notable votes ─────────────────────────────────────────────────────────
    opposed_hi = prof.opposed_high_consensus
    supported_lo = prof.supported_low_consensus

    # ── header metadata ───────────────────────────────────────────────────────
    year_start = (start_date or "")[:4]
    year_end = (end_date or "")[:4]
    selected_years_val = f"{year_start}–{year_end}" if (year_start and year_end) else "all"

    if membership:
        latest_year = datetime.date.today().year
        mem_end_label = "present" if membership[1] >= latest_year - 1 else str(membership[1])
        membership_str = f"UN Member: {membership[0]}–{mem_end_label}"
    else:
        membership_str = ""

    align_title = (
        "Bilateral Agreement Over Time (vs. P5)"
        if prof.comparison_is_default
        else "Bilateral Agreement Over Time (vs. Selected Countries)"
    )
    subregion = data.get_country_subregion(country1)

    # ── assemble page ─────────────────────────────────────────────────────────
    return html.Div(
        [
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
            html.Div(
                [
                    html.H1(
                        country1_name,
                        style={"margin": "0 0 8px 0", "fontSize": "2rem", "color": "#212529"},
                    ),
                    *(
                        [html.Div(subregion, style={"color": "#6c757d", "marginBottom": "4px"})]
                        if subregion
                        else []
                    ),
                    html.Div(membership_str, style={"color": "#6c757d", "marginRight": "18px"}),
                ],
                style={
                    "borderBottom": "2px solid #dee2e6",
                    "paddingBottom": "16px",
                    "marginBottom": "24px",
                },
            ),
            # top row: stats block (left) + ranking tables (right)
            html.Div(
                [
                    html.Div(
                        [
                            _stat_tile("Selected Years", selected_years_val, mb="16px"),
                            _stat_tile("Accepted Resolutions", f"{total:,}"),
                            *(
                                [_stat_tile("Multilateral Alignment", multilateral_rank_str)]
                                if multilateral_rank_str is not None
                                else []
                            ),
                            vote_bar,
                            vote_legend,
                        ],
                        style={
                            "display": "flex",
                            "flexDirection": "column",
                            "flex": "0 0 280px",
                            "minWidth": "280px",
                        },
                    ),
                    html.Div(
                        _section(
                            "Most & Least Aligned Countries",
                            html.P(
                                f"Shows only countries having at least {MIN_JOINT_VOTES} shared votes with {country1_name}",
                                style={
                                    "fontSize": "0.75em",
                                    "color": "#999",
                                    "margin": "6px 0 12px 0",
                                },
                            ),
                            html.Div(
                                [
                                    _rank_table(top_aligned, "Most aligned"),
                                    _rank_table(bottom_aligned, "Least aligned"),
                                ],
                                style={"display": "flex", "gap": "60px"},
                            ),
                        ),
                        style={"flex": "1", "minWidth": "0"},
                    ),
                ],
                style={
                    "display": "flex",
                    "gap": "32px",
                    "alignItems": "flex-start",
                    "marginBottom": "28px",
                },
            ),
            # bilateral alignment over time
            _section(
                align_title,
                dcc.Graph(figure=align_fig, config={"displayModeBar": False}),
            ),
            # notable votes
            _section(
                "Notable Votes",
                html.Div(
                    [
                        html.Div(
                            [
                                _res_table(
                                    opposed_hi, "✗  Opposed resolutions — highest consensus first"
                                )
                            ],
                            style={"flex": "1", "minWidth": "0"},
                        ),
                        html.Div(
                            [
                                _res_table(
                                    supported_lo,
                                    "✓  Supported resolutions — lowest consensus first",
                                )
                            ],
                            style={"flex": "1", "minWidth": "0"},
                        ),
                    ],
                    style={"display": "flex", "gap": "40px"},
                ),
            ),
        ],
        style={
            "maxWidth": "1100px",
            "margin": "0 auto",
            "padding": "32px 24px",
            "fontFamily": "inherit",
        },
    )


# trigger window.print() when the button is clicked
clientside_callback(
    "function(n) { if (n) { window.print(); } return ''; }",
    Output("profile-print-dummy", "children"),
    Input("profile-print-btn", "n_clicks"),
    prevent_initial_call=True,
)
