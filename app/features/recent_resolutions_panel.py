import functools

from dash import Input, Output, callback, html
import pandas as pd

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

    y_pct = _safe_pct(yes_count, total_ms)
    n_pct = _safe_pct(no_count, total_ms)
    a_pct = _safe_pct(abstain_count, total_ms)
    x_pct = _safe_pct(not_voting_count, total_ms)

    return html.Div(
        [
            html.Div(
                [
                    html.Span(date_str, style={"color": "#666", "fontSize": "0.9em"}),
                    html.A(
                        html.Span(
                            f"{res_id}",
                            style={
                                "color": "#007bff",
                                "fontWeight": "600",
                                "fontSize": "1.05em",
                            },
                        ),
                        href=link,
                        target="_blank",
                        style={"textDecoration": "none"},
                    ),
                ],
                style={
                    "display": "flex",
                    "justifyContent": "space-between",
                    "alignItems": "center",
                    "gap": "12px",
                },
            ),
            html.Div(
                title,
                style={
                    "marginTop": "6px",
                    "marginBottom": "8px",
                    "fontSize": "1.05em",
                },
            ),
            html.Span(
                category_tag,
                style={
                    "display": "inline-block",
                    "marginBottom": "8px",
                    "padding": "2px 8px",
                    "fontSize": "0.8em",
                    "fontWeight": "600",
                    "color": "#1b3357",
                    "backgroundColor": "#eaf1ff",
                    "borderRadius": "999px",
                },
            ),
            html.Div(
                [
                    html.Span(
                        f"Yes: {yes_count} ({y_pct:.1f}%)",
                        style={"color": "#1a7f37", "fontWeight": "500"},
                    ),
                    html.Span(
                        f"No: {no_count} ({n_pct:.1f}%)",
                        style={"color": "#cf222e", "fontWeight": "500"},
                    ),
                    html.Span(
                        f"Abstain: {abstain_count} ({a_pct:.1f}%)",
                        style={"color": "#9a6700", "fontWeight": "500"},
                    ),
                    html.Span(
                        f"Not Voting: {not_voting_count} ({x_pct:.1f}%)",
                        style={"color": "#0969da", "fontWeight": "500"},
                    ),
                ],
                style={
                    "display": "flex",
                    "flexWrap": "wrap",
                    "gap": "12px",
                    "paddingTop": "8px",
                    "borderTop": "1px solid #eee",
                    "fontSize": "0.92em",
                },
            ),
        ],
        className="resolution-card",
        style={
            "backgroundColor": "white",
            "border": "1px solid #e0e0e0",
            "borderRadius": "8px",
            "padding": "15px",
            "marginBottom": "12px",
            "boxShadow": "0 2px 4px rgba(0,0,0,0.05)",
        },
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
