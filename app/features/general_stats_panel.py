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


def _stats_card(icon, title, value, subtitle, subtitle_style=None):
    merged_subtitle_style = {"fontSize": "0.8rem", "color": "#6b7280"}
    if subtitle_style:
        merged_subtitle_style.update(subtitle_style)
    return html.Div(
        [
            html.Div(icon, style={"fontSize": "1.5rem", "lineHeight": "1"}),
            html.Div([
                html.Div(title, style={"color": "#4b5563", "fontSize": "0.9rem"}),
                html.Div(value, style={"fontSize": "1.5rem", "fontWeight": "700", "color": "#1b3357"}),
                html.Div(subtitle, style=merged_subtitle_style),
            ], style={
                "display": "flex",
                "flexDirection": "column",
                "gap": "0.25rem",
            })
        ],
        className="resolution-card",
        style={
            "display": "flex",
            "flexDirection": "row",
            "gap": "0.5rem",
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
            _stats_card("📄", "Accepted Resolutions", f"{total_resolutions:,}", "Total records"),
            _stats_card(
                "📅",
                "Year Span",
                f"{year_min}-{year_max}" if year_min and year_max else "N/A",
                f"{year_span} years covered" if year_span else "No valid date range",
            ),
            _stats_card(
                "🌍",
                "Countries",
                f"{num_countries:,}",
                f"UN member entities appearing in dataset",
                subtitle_style={"whiteSpace": "nowrap"},
            ),
            # _stats_card("🏷️", "Subjects", f"{unique_subjects:,}", f""),
            _stats_card("🏷️", "Subjects", f"{unique_subjects:,}", f"{total_subject_links:,} resolutions with assigned subjects"),
            html.Div("Vote Composition Across Accepted Resolutions", style={"fontWeight": "600", "color": "#1f2937", "marginBottom": "8px"}),
            html.Div(
                [
                    html.Div(style={"width": f"{y_pct:.2f}%", "backgroundColor": "#74bb88", "height": "100%"}),
                    html.Div(style={"width": f"{n_pct:.2f}%", "backgroundColor": "#cc575f", "height": "100%"}),
                    html.Div(style={"width": f"{a_pct:.2f}%", "backgroundColor": "#c19f5b", "height": "100%"}),
                    html.Div(style={"width": f"{x_pct:.2f}%", "backgroundColor": "#6ea3e0", "height": "100%"}),
                ],
                style={"display": "flex", "height": "14px", "borderRadius": "999px", "overflow": "hidden", "backgroundColor": "#e5e7eb"},
            ),
            html.Div(
                [
                    html.Span(f"■ Yes: {y_total:,} ({y_pct:.1f}%)", style={"color": "#1a7f37"}),
                    html.Span(f"■ No: {n_total:,} ({n_pct:.1f}%)", style={"color": "#cf222e"}),
                    html.Span(f"■ Abstain: {a_total:,} ({a_pct:.1f}%)", style={"color": "#9a6700"}),
                    html.Span(f"■ Not Voting: {x_total:,} ({x_pct:.1f}%)", style={"color": "#0969da"}),
                ],
                style={"display": "flex", "flexDirection": "column", "gap": "0.5rem", "marginTop": "8px", "fontSize": "0.75rem"},
            ),
        ],
        style={
            "display": "flex",
            "flexDirection": "column",
            "flexWrap": "wrap",
            "minWidth": "300px",
        },
    )


@functools.lru_cache(maxsize=1)
def _get_stats_component_cached():
    return _build_stats_component()


layout = html.Div(id="index-general-stats-content")


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
