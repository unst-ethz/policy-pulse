import functools

import pandas as pd
from dash import Input, Output, callback, html

from app.features.resolution_list import create_vote_summary

from .. import data
from ..un_data_stream.analysis import overview

INITIAL_VISIBLE = 3
STEP_VISIBLE = 5

_LIST_CONTAINER_STYLE = {
    "backgroundColor": "white",
}

_BTN_VISIBLE_STYLE = {
    "marginTop": "1rem",
}

_BTN_HIDDEN_STYLE = {"display": "none"}


def _extract_category_tag(row):
    raw_category = row.get("category_tag")
    if pd.isna(raw_category) or raw_category is None or str(raw_category).strip() == "":
        return "No Category"
    return str(raw_category).strip() or "No Category"


def _build_resolution_card(row):
    res_id = row.get("resolution", "N/A")
    link = row.get("undl_link", "#")
    date_val = row.get("date")
    date_str = date_val.strftime("%Y-%m-%d") if pd.notnull(date_val) else "Unknown"
    title = row.get("title", "Untitled")
    category_tag = _extract_category_tag(row)
    session_val = row.get("session", "")
    session_str = f"Session {session_val}" if session_val else ""

    consensus_score = row.get("consensus_score")
    consensus_display = f"{consensus_score:.2f}" if pd.notna(consensus_score) else "N/A"

    vote_summary = create_vote_summary(
        row.get("total_yes"),
        row.get("total_no"),
        row.get("total_abstentions"),
    )

    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.A(
                                html.Span(
                                    res_id,
                                    style={"color": "#007bff", "fontWeight": "bold"},
                                ),
                                href=link,
                                target="_blank",
                                style={"textDecoration": "none"},
                            ),
                            html.Span(
                                date_str,
                                style={
                                    "color": "#666",
                                    "fontSize": "0.9em",
                                    "marginLeft": "12px",
                                },
                            ),
                            html.Span(
                                session_str,
                                style={
                                    "color": "#666",
                                    "fontSize": "0.9em",
                                    "marginLeft": "12px",
                                },
                            )
                            if session_str
                            else None,
                            html.Span(
                                f"Consensus score: {consensus_display}",
                                style={
                                    "color": "#666" if pd.notna(consensus_score) else "#999",
                                    "fontSize": "0.9em",
                                    "marginLeft": "12px",
                                },
                            ),
                            vote_summary,
                        ],
                        style={
                            "marginBottom": "0.5rem",
                            "display": "flex",
                            "alignItems": "center",
                            "flexWrap": "wrap",
                        },
                    ),
                    html.Div(title),
                    html.Div(
                        category_tag,
                        style={
                            "color": "#666",
                            "fontSize": "0.75em",
                            "marginTop": "0.5rem",
                            "padding": "2px 6px",
                            "borderRadius": "8px",
                            "backgroundColor": "#dbebff",
                            "width": "fit-content",
                        },
                    ),
                ],
                className="resolution-card-main",
            ),
        ],
        className="resolution-card",
    )


def _query_sorted_resolutions():
    return overview.recent_resolutions(data.query_engine)


@functools.lru_cache(maxsize=1)
def _get_recent_resolutions_cached():
    return _query_sorted_resolutions()


def invalidate() -> None:
    """Drop the cached resolution list so the next render re-queries current data.

    Called by `app.data.reload_if_stale()`: this list is derived from the repository, and it is
    exactly where stale data is most visible — a reload exists to surface new resolutions.
    """
    _get_recent_resolutions_cached.cache_clear()


layout = html.Div(
    [
        html.Div(
            id="index-recent-resolutions-summary",
            style={"color": "#666", "marginBottom": "8px"},
        ),
        html.Div(id="index-recent-resolutions-list", style=_LIST_CONTAINER_STYLE),
        html.Button(
            f"Show {STEP_VISIBLE} more",
            id="index-recent-resolutions-more-btn",
            n_clicks=0,
            className="cta-button",
            style=_BTN_VISIBLE_STYLE,
        ),
    ],
    style={"marginTop": "10px"},
)


def register_callbacks():
    @callback(
        Output("index-recent-resolutions-list", "children"),
        Output("index-recent-resolutions-summary", "children"),
        Output("index-recent-resolutions-more-btn", "style"),
        Input("index-recent-resolutions-more-btn", "n_clicks"),
    )
    def update_recent_resolutions(n_clicks):
        try:
            # Copy to avoid mutating cached dataframe downstream.
            df = _get_recent_resolutions_cached().copy()
        except Exception as e:
            return (
                html.Div(f"Error loading recent resolutions: {e}", style={"color": "red"}),
                "",
                _BTN_HIDDEN_STYLE,
            )

        if df.empty:
            return (
                html.Div("No resolutions found.", style={"color": "#666"}),
                "0 resolutions",
                _BTN_HIDDEN_STYLE,
            )

        visible_count = INITIAL_VISIBLE + ((n_clicks or 0) * STEP_VISIBLE)
        display_df = df.head(visible_count)
        cards = [_build_resolution_card(row) for _, row in display_df.iterrows()]

        total = len(df)
        shown = len(display_df)
        summary = f"Here are the {shown} most recent resolutions voted on (of {total})"
        btn_style = _BTN_VISIBLE_STYLE if shown < total else _BTN_HIDDEN_STYLE
        return cards, summary, btn_style
