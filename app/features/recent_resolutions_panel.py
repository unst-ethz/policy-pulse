import functools

from dash import Input, Output, callback, html
import pandas as pd

from app.features.resolution_list import create_vote_summary

from .. import data

INITIAL_VISIBLE = 3
STEP_VISIBLE = 5

_LIST_CONTAINER_STYLE = {
    "backgroundColor": "white",
}

_BTN_VISIBLE_STYLE = {
    "marginTop": "1rem",
}

_BTN_HIDDEN_STYLE = {"display": "none"}


def _safe_count(value):
    if pd.isna(value):
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _safe_pct(count, total):
    if total <= 0:
        return 0.0
    return (count / total) * 100.0


def _extract_category_tag(row):
    raw_category = row.get("category_tag")
    if pd.isna(raw_category) or raw_category is None or str(raw_category).strip() == "":
        return "No Category"
    return str(raw_category).strip() or "No Category"


def _build_category_tag_map():
    resolution_subject_df = data.query_engine.resolution_subject_table
    subject_df = data.query_engine.subject_table

    if resolution_subject_df.empty or subject_df.empty:
        return {}

    label_col = "label_en" if "label_en" in subject_df.columns else None
    if label_col is None:
        return {}

    subject_labels = (
        subject_df[["subject_id", label_col]]
        .dropna(subset=["subject_id", label_col])
        .copy()
    )
    merged = resolution_subject_df.merge(subject_labels, on="subject_id", how="left")
    merged = merged.dropna(subset=[label_col])
    if merged.empty:
        return {}

    merged["undl_id"] = merged["undl_id"].astype(str)
    merged[label_col] = merged[label_col].astype(str).str.strip()
    merged = merged[merged[label_col] != ""]
    if merged.empty:
        return {}

    first_labels = merged.groupby("undl_id", as_index=False)[label_col].first()
    return dict(zip(first_labels["undl_id"], first_labels[label_col]))


def _build_resolution_card(row):
    res_id = row.get("resolution", "N/A")
    link = row.get("undl_link", "#")
    date_val = row.get("date")
    date_str = date_val.strftime("%Y-%m-%d") if pd.notnull(date_val) else "Unknown"
    title = row.get("title", "Untitled")
    category_tag = _extract_category_tag(row)
    yes_count = _safe_count(row.get("total_yes", 0))
    no_count = _safe_count(row.get("total_no", 0))
    abstain_count = _safe_count(row.get("total_abstentions", 0))
    not_voting_count = _safe_count(row.get("total_non_voting", 0))
    total_ms = _safe_count(row.get("total_ms", 0))
    session_val = row.get("session", "")
    session_str = f"Session {session_val}" if session_val else ""

    consensus_score = row.get("consensus_score")
    consensus_display = f"{consensus_score:.2f}" if pd.notna(consensus_score) else "N/A"

    vote_summary = create_vote_summary(
        row.get("total_yes"),
        row.get("total_no"),
        row.get("total_abstentions"),
    )

    y_pct = _safe_pct(yes_count, total_ms)
    n_pct = _safe_pct(no_count, total_ms)
    a_pct = _safe_pct(abstain_count, total_ms)
    x_pct = _safe_pct(not_voting_count, total_ms)

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
                                    "color": "#666"
                                    if pd.notna(consensus_score)
                                    else "#999",
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
    df = data.query_engine.query_resolutions()
    if df.empty:
        return df

    category_map = _build_category_tag_map()
    if "undl_id" in df.columns:
        df["category_tag"] = (
            df["undl_id"].astype(str).map(category_map).fillna("No Category")
        )
    else:
        df["category_tag"] = "No Category"

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.sort_values(by=["date"], ascending=False, na_position="last")
    return df


@functools.lru_cache(maxsize=1)
def _get_recent_resolutions_cached():
    return _query_sorted_resolutions()


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
                html.Div(
                    f"Error loading recent resolutions: {e}", style={"color": "red"}
                ),
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
