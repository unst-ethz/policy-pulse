import functools

from dash import Input, Output, callback, html
import pandas as pd

from .. import data


def _safe_count(value):
    if pd.isna(value):
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _safe_pct(part, whole):
    if whole <= 0:
        return 0.0
    return (part / whole) * 100.0


def _stats_card(icon, title, value, subtitle):
    return html.Div(
        [
            html.Div(icon, style={"fontSize": "1.5rem", "lineHeight": "1"}),
            html.Div(title, style={"color": "#4b5563", "fontSize": "0.9rem", "marginTop": "4px"}),
            html.Div(value, style={"fontSize": "1.5rem", "fontWeight": "700", "color": "#1b3357", "marginTop": "2px"}),
            html.Div(subtitle, style={"fontSize": "0.8rem", "color": "#6b7280", "marginTop": "2px"}),
        ],
        style={
            "flex": "1",
            "minWidth": "180px",
            "background": "linear-gradient(135deg, #f8fbff 0%, #eef4ff 100%)",
            "border": "1px solid #dbe7ff",
            "borderRadius": "12px",
            "padding": "14px",
            "boxShadow": "0 3px 8px rgba(27,51,87,0.08)",
        },
    )


def _build_stats_component():
    df = data.query_engine.query_resolutions().copy()
    if df.empty:
        return html.Div("No data available.", style={"color": "#666"})

    df["date"] = pd.to_datetime(df.get("date"), errors="coerce")
    valid_dates = df["date"].dropna()

    total_resolutions = len(df)
    year_min = int(valid_dates.dt.year.min()) if not valid_dates.empty else None
    year_max = int(valid_dates.dt.year.max()) if not valid_dates.empty else None
    year_span = (year_max - year_min + 1) if year_min and year_max else 0

    num_countries = len(getattr(data.query_engine, "country_columns", []) or [])
    total_subject_links = len(getattr(data.query_engine, "resolution_subject_table", pd.DataFrame()))
    unique_subjects = (
        getattr(data.query_engine, "resolution_subject_table", pd.DataFrame()).get("subject_id", pd.Series(dtype="object")).nunique()
    )
    avg_participation = df["total_ms"].fillna(0).mean() if "total_ms" in df.columns else 0

    y_total = _safe_count(df["total_yes"].fillna(0).sum()) if "total_yes" in df.columns else 0
    n_total = _safe_count(df["total_no"].fillna(0).sum()) if "total_no" in df.columns else 0
    a_total = _safe_count(df["total_abstentions"].fillna(0).sum()) if "total_abstentions" in df.columns else 0
    x_total = _safe_count(df["total_non_voting"].fillna(0).sum()) if "total_non_voting" in df.columns else 0
    vote_total = max(1, y_total + n_total + a_total + x_total)

    y_pct = _safe_pct(y_total, vote_total)
    n_pct = _safe_pct(n_total, vote_total)
    a_pct = _safe_pct(a_total, vote_total)
    x_pct = _safe_pct(x_total, vote_total)

    return html.Div(
        [
            _stats_card("📄", "Resolutions", f"{total_resolutions:,}", "Total records"),
            _stats_card(
                "📅",
                "Year Span",
                f"{year_min}-{year_max}" if year_min and year_max else "N/A",
                f"{year_span} years covered" if year_span else "No valid date range",
            ),
            _stats_card("🌍", "Countries", f"{num_countries:,}", "Voting columns available"),
            _stats_card("🏷️", "Subjects", f"{unique_subjects:,}", f"{total_subject_links:,} resolution-subject links"),
            _stats_card("🗳️", "Avg Participation", f"{avg_participation:.1f}", "Member states per resolution"),
            html.Div("Vote Composition Across All Resolutions", style={"fontWeight": "600", "color": "#1f2937", "marginBottom": "8px"}),
            html.Div(
                [
                    html.Div(style={"width": f"{y_pct:.2f}%", "backgroundColor": "#1a7f37", "height": "100%"}),
                    html.Div(style={"width": f"{n_pct:.2f}%", "backgroundColor": "#cf222e", "height": "100%"}),
                    html.Div(style={"width": f"{a_pct:.2f}%", "backgroundColor": "#9a6700", "height": "100%"}),
                    html.Div(style={"width": f"{x_pct:.2f}%", "backgroundColor": "#0969da", "height": "100%"}),
                ],
                style={"display": "flex", "height": "14px", "borderRadius": "999px", "overflow": "hidden", "backgroundColor": "#e5e7eb"},
            ),
            html.Div(
                [
                    html.Span(f"Y(Yes): {y_total:,} ({y_pct:.1f}%)", style={"color": "#1a7f37", "fontWeight": "600"}),
                    html.Span(f"N(No): {n_total:,} ({n_pct:.1f}%)", style={"color": "#cf222e", "fontWeight": "600"}),
                    html.Span(f"A(Abstain): {a_total:,} ({a_pct:.1f}%)", style={"color": "#9a6700", "fontWeight": "600"}),
                    html.Span(f"X(Not Voting): {x_total:,} ({x_pct:.1f}%)", style={"color": "#0969da", "fontWeight": "600"}),
                ],
                style={"display": "flex", "flexWrap": "wrap", "gap": "12px", "marginTop": "8px", "fontSize": "0.9rem"},
            ),
        ],
        style={
            "display": "flex",
            "flexDirection": "column",
            "flexWrap": "wrap",
            "gap": "12px",
            "marginTop": "14px",
            "backgroundColor": "#ffffff",
            "border": "1px solid #e5e7eb",
            "borderRadius": "12px",
            "padding": "12px",
        },
    )


@functools.lru_cache(maxsize=1)
def _get_stats_component_cached():
    return _build_stats_component()


layout = html.Div(
    [
        html.Div(id="index-general-stats-content"),
    ],
    style={
        "marginTop": "8px",
        "padding": "14px",
        "background": "linear-gradient(180deg, #f8fbff 0%, #ffffff 100%)",
        "border": "1px solid #e6edf8",
        "borderRadius": "14px",
    },
)


def register_callbacks():
    @callback(
        Output("index-general-stats-content", "children"),
        Input("index-general-stats-content", "id"),
    )
    def update_general_stats(_):
        try:
            return _get_stats_component_cached()
        except Exception as e:
            return html.Div(f"Error loading statistics: {e}", style={"color": "red"})
