from dash import callback, Input, Output, State, html, dcc
import pandas as pd
from .. import data


_PAGE_SIZE = 10
_LOAD_MORE_SIZE = 50
_NO_FILTER = "NO_FILTER"

VOTE_MAP = {
    "Y": {"symbol": "✓", "color": "#2ecc71", "label": "Yes"},           # green checkmark
    "N": {"symbol": "✗", "color": "#e74c3c", "label": "No"},            # red 'x'
    "A": {"symbol": "●", "color": "#f39c12", "label": "Abstain"},       # yellow dot
    "X": {"symbol": "–", "color": "#999",    "label": "Did not vote"},  # grey hyphen
}
_VOTE_NA = {"symbol": "·", "color": "#ccc", "label": "Non-member / no data"}  # tiny grey dot


def create_vote_summary(yes_count, no_count, abstain_count):
    def fmt(n):
        return str(int(n)) if pd.notna(n) else "–"
    items = [("Yes", yes_count), ("No", no_count), ("Abstain", abstain_count)]
    inner = " · ".join(f"{label}: {fmt(count)}" for label, count in items)
    return html.Span(
        f"({inner})",
        style={"fontSize": "0.9em", "marginLeft": "12px", "color": "#666"},
    )


def create_vote_indicator(country_name, vote):
    if pd.isna(vote) or vote not in VOTE_MAP:
        config = _VOTE_NA
        return html.Span(
            [
                html.Span(config["symbol"], style={"marginRight": "4px", "color": config["color"]}),
                html.Span(country_name),
            ],
            style={"color": "#bbb", "marginRight": "15px", "fontSize": "0.9em"},
        )
    config = VOTE_MAP[vote]
    return html.Span(
        [
            html.Span(config["symbol"], style={"color": config["color"], "marginRight": "4px", "fontWeight": "bold"}),
            html.Span(country_name),
        ],
        style={"fontWeight": "500", "marginRight": "15px", "fontSize": "0.9em"},
    )


layout = [
    html.Div(
        [
            # --- Local Controls ---
            html.Div(
                [
                    # Controls box (always visible)
                    html.Div(
                        id="rl-agreement-filter-container",
                        style={
                            "display": "flex",
                            "alignItems": "center",
                            "flexWrap": "wrap",
                            "gap": "20px",
                            "padding": "15px",
                            "backgroundColor": "#f1f3f4",
                            "borderRadius": "8px",
                        },
                        children=[
                            # Sort by dropdown
                            html.Div(
                                [
                                    html.Label(
                                        "Sort by:",
                                        style={"fontWeight": "bold", "marginRight": "10px"},
                                    ),
                                    dcc.Dropdown(
                                        id="rl-sort-dropdown",
                                        options=[
                                            {"label": "Date (Newest First)", "value": "date_desc"},
                                            {"label": "Date (Oldest First)", "value": "date_asc"},
                                            {"label": "Consensus Score (Highest First)", "value": "consensus_desc"},
                                            {"label": "Consensus Score (Lowest First)", "value": "consensus_asc"},
                                        ],
                                        value="date_desc",
                                        clearable=False,
                                        style={
                                            "width": "300px",
                                            "display": "inline-block",
                                            "verticalAlign": "middle",
                                        },
                                    ),
                                ],
                                style={"display": "flex", "alignItems": "center"},
                            ),
                            # Vote filter (only when main country selected, no comparison countries)
                            html.Div(
                                id="rl-vote-filter-wrapper",
                                style={"display": "none"},
                                children=[
                                    html.Label(
                                        "Filter by Vote:",
                                        style={"fontWeight": "bold", "marginRight": "10px"},
                                    ),
                                    dcc.Dropdown(
                                        id="rl-vote-filter",
                                        options=[
                                            {"label": "Show All", "value": _NO_FILTER},
                                            {"label": "Yes", "value": "Y"},
                                            {"label": "No", "value": "N"},
                                            {"label": "Abstain", "value": "A"},
                                            {"label": "Did not vote", "value": "X"},
                                        ],
                                        value=_NO_FILTER,
                                        clearable=False,
                                        style={
                                            "width": "180px",
                                            "display": "inline-block",
                                            "verticalAlign": "middle",
                                        },
                                    ),
                                ],
                                className="fade-in",
                            ),
                            # Agreement filter (only when exactly 1 comparison country)
                            html.Div(
                                id="rl-agreement-dropdown-wrapper",
                                style={"display": "none"},
                                children=[
                                    html.Label(
                                        "Filter by Agreement:",
                                        style={"fontWeight": "bold", "marginRight": "10px"},
                                    ),
                                    dcc.Dropdown(
                                        id="rl-agreement-dropdown",
                                        options=[
                                            {"label": "Show All", "value": _NO_FILTER},
                                            {"label": "Agreed (Voted Same)", "value": "AGREED"},
                                            {
                                                "label": "Disagreed (Voted Differently)",
                                                "value": "DISAGREED",
                                            },
                                            {
                                                "label": "Strongly Disagreed (Y/N vs N/Y)",
                                                "value": "STRONGLY_DISAGREED",
                                            },
                                        ],
                                        value=_NO_FILTER,
                                        clearable=False,
                                        style={
                                            "width": "250px",
                                            "display": "inline-block",
                                            "verticalAlign": "middle",
                                        },
                                    ),
                                ],
                                className="fade-in",
                            ),
                        ],
                    ),
                    html.Div(
                        id="rl-multi-country-msg",
                        style={
                            "display": "none",
                            "color": "#666",
                            "fontStyle": "italic",
                        },
                    ),
                ],
                style={"marginBottom": "20px", "minHeight": "5px"},
            ),
            # --- Vote Legend ---
            html.Div(
                [
                    html.Span("Vote key:", style={"fontWeight": "bold", "marginRight": "12px", "fontSize": "0.85em", "color": "#555"}),
                    *[
                        html.Span(
                            [html.Span(v["symbol"], style={"color": v["color"], "marginRight": "4px", "fontWeight": "bold"}), v["label"]],
                            style={"fontSize": "0.85em", "marginRight": "14px"},
                        )
                        for v in [*VOTE_MAP.values(), _VOTE_NA]
                    ],
                ],
                style={
                    "display": "flex",
                    "alignItems": "center",
                    "flexWrap": "wrap",
                    "padding": "8px 12px",
                    "backgroundColor": "#f8f9fa",
                    "borderRadius": "6px",
                    "border": "1px solid #e0e0e0",
                    "marginBottom": "14px",
                },
            ),
            # --- Results Area ---
            dcc.Loading(
                id="rl-loading",
                type="dot",
                children=[
                    html.Div(
                        id="rl-results-summary",
                        style={
                            "marginBottom": "10px",
                            "fontStyle": "italic",
                            "color": "#666",
                        },
                    ),
                    html.Div(id="rl-results-list"),
                    html.Button(
                        "Load More Accepted Resolutions",
                        id="rl-load-more-btn",
                        n_clicks=0,
                        style={
                            "width": "100%",
                            "marginTop": "15px",
                            "display": "none",
                            "padding": "10px",
                        },
                    ),
                ],
            ),
        ]
    )
]


def register_callbacks():

    # 1. Main Logic: Query, Filter & Render
    @callback(
        Output("rl-results-list", "children"),
        Output("rl-results-summary", "children"),
        Output("rl-load-more-btn", "style"),
        Output("rl-agreement-dropdown-wrapper", "style"),
        Output("rl-vote-filter-wrapper", "style"),
        Output("rl-multi-country-msg", "children"),
        Output("rl-multi-country-msg", "style"),
        Input("filter-component-data-store", "data"),
        Input("filter-component-filter-store", "data"),
        Input("rl-agreement-dropdown", "value"),
        Input("rl-vote-filter", "value"),
        Input("rl-load-more-btn", "n_clicks"),
        Input("rl-sort-dropdown", "value"),
        State("country1-iso-alpha3", "data"),
    )
    def update_resolution_list(
        data_store, filter_params, agreement_filter, vote_filter, n_clicks, sort_order, country1_backup
    ):
        # Default Styles
        btn_style_hidden = {"display": "none"}
        btn_style_visible = {
            "width": "100%",
            "marginTop": "15px",
            "display": "block",
            "padding": "10px",
            "cursor": "pointer",
        }

        if not data_store or not filter_params:
            return (
                html.Div("Loading...", style={"padding": "20px"}),
                "",
                btn_style_hidden,
                {"display": "none"},
                {"display": "none"},
                "",
                {"display": "none"},
            )

        # Extract Params
        country1 = filter_params.get("country1_alpha3") or country1_backup

        # Extract comparison countries (can be list, string or None)
        country2_raw = filter_params.get("country2")
        comparison_countries = []
        if isinstance(country2_raw, list):
            comparison_countries = country2_raw
        elif isinstance(country2_raw, str) and country2_raw:
            comparison_countries = [country2_raw]

        # UI State Logic: Filter Visibility
        show_agreement_filter = False
        agreement_container_style = {"display": "none"}
        vote_filter_style = {"display": "none"}
        multi_msg = ""
        multi_msg_style = {"display": "none"}

        if len(comparison_countries) == 1:
            # Only enable agreement filter if we have BOTH a main country AND exactly 1 comparison country
            if country1:
                show_agreement_filter = True
                agreement_container_style = {
                    "display": "flex",
                    "alignItems": "center",
                }
        elif len(comparison_countries) > 1:
            multi_msg = f"Comparing against {len(comparison_countries)} selected countries. Agreement filter disabled for multi-select."
            multi_msg_style = {"display": "block", "marginBottom": "10px"}
        elif country1:
            # Show vote filter only when main country is selected with no comparison countries
            vote_filter_style = {"display": "flex", "alignItems": "center"}

        # 1. Load pre-filtered resolutions from the shared data store,
        #    then join vote columns from the resolution table for display.
        try:
            df = pd.read_json(data_store, orient="split")

            # Join vote columns for country1 + comparison countries
            vote_cols_needed = [c for c in ([country1] + comparison_countries) if c]
            if vote_cols_needed and not df.empty:
                res_table = data.query_engine.resolution_table
                available = [c for c in vote_cols_needed if c in res_table.columns]
                if available:
                    vote_df = res_table.loc[
                        res_table["undl_id"].isin(df["undl_id"]), ["undl_id"] + available
                    ]
                    df = df.merge(vote_df, on="undl_id", how="left")

        except Exception as e:
            return (
                html.Div(f"Error loading data: {e}", style={"color": "red"}),
                "",
                btn_style_hidden,
                agreement_container_style,
                vote_filter_style,
                multi_msg,
                multi_msg_style,
            )

        if df.empty:
            return (
                html.Div(
                    "No resolutions found.", style={"padding": "20px", "color": "#777"}
                ),
                "0 results",
                btn_style_hidden,
                agreement_container_style,
                vote_filter_style,
                multi_msg,
                multi_msg_style,
            )

        # 2. Filter & Sort Logic
        if sort_order == "date_asc":
            sort_by, ascending = "date", True
        elif sort_order == "consensus_desc":
            sort_by, ascending = "consensus_score", False
        elif sort_order == "consensus_asc":
            sort_by, ascending = "consensus_score", True
        else:  # Default to date_desc
            sort_by, ascending = "date", False
        
        filtered_df = df.copy().sort_values(sort_by, ascending=ascending, na_position="last")

        if country1 and show_agreement_filter and agreement_filter != _NO_FILTER:
            c2 = comparison_countries[0]
            if country1 in filtered_df.columns and c2 in filtered_df.columns:
                filtered_df = filtered_df.dropna(subset=[country1, c2])
                if agreement_filter == "AGREED":
                    filtered_df = filtered_df[filtered_df[country1] == filtered_df[c2]]
                elif agreement_filter == "DISAGREED":
                    filtered_df = filtered_df[filtered_df[country1] != filtered_df[c2]]
                elif agreement_filter == "STRONGLY_DISAGREED":
                    cond1 = (filtered_df[country1] == "Y") & (filtered_df[c2] == "N")
                    cond2 = (filtered_df[country1] == "N") & (filtered_df[c2] == "Y")
                    filtered_df = filtered_df[cond1 | cond2]

        if country1 and not comparison_countries and vote_filter and vote_filter != _NO_FILTER:
            if country1 in filtered_df.columns:
                filtered_df = filtered_df[filtered_df[country1] == vote_filter]

        total_count = len(filtered_df)

        # 3. Pagination
        current_limit = _PAGE_SIZE + ((n_clicks or 0) * _LOAD_MORE_SIZE)
        display_df = filtered_df.head(current_limit)
        shown_count = len(display_df)

        # 4. Render
        summary = f"Showing {shown_count} of {total_count} resolutions"
        if len(comparison_countries) > 0:
            summary += f" (Comparing with: {', '.join([data.get_country_name(c) for c in comparison_countries[:3]])}{'...' if len(comparison_countries) > 3 else ''})"

        btn_style = btn_style_visible if shown_count < total_count else btn_style_hidden

        cards = []
        for _, row in display_df.iterrows():
            res_id = row.get("resolution", "N/A")
            link = row.get("undl_link", "#")
            date_val = row.get("date")
            date_str = (
                date_val.strftime("%d %b %Y") if pd.notna(date_val) else "Unknown"
            )
            session_val = row.get("session", "")
            session_str = f"Session {session_val}" if session_val else ""
            title = row.get("title", "Untitled")
            consensus_score = row.get("consensus_score")

            consensus_display = f"{consensus_score:.2f}" if pd.notna(consensus_score) else "N/A"

            vote_summary = create_vote_summary(
                row.get("total_yes"),
                row.get("total_no"),
                row.get("total_abstentions"),
            )

            indicators = []

            # Main Country (only if selected)
            if country1:
                c1_vote = row.get(country1) if country1 in row else None
                indicators.append(
                    create_vote_indicator(data.get_country_name(country1), c1_vote)
                )

            # Comparators (all selected countries)
            for c2 in comparison_countries:
                if c2 in filtered_df.columns:
                    indicators.append(
                        create_vote_indicator(data.get_country_name(c2), row.get(c2))
                    )

            card = html.Div(
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
                                    ) if session_str else None,
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
                                style={"marginBottom": "0.5rem", "display": "flex", "alignItems": "center", "flexWrap": "wrap"},
                            ),
                            html.Div(title),
                        ],
                        className="resolution-card-main",
                    ),
                    html.Div(
                        [html.Div(ind, className="voting-list-row") for ind in indicators],
                        className="voting-list",
                    )
                    if indicators
                    else None,
                ],
                className="resolution-card",
            )
            cards.append(card)

        return (
            cards,
            summary,
            btn_style,
            agreement_container_style,
            vote_filter_style,
            multi_msg,
            multi_msg_style,
        )
