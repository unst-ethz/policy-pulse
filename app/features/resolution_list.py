from dash import callback, Input, Output, State, html, dcc
import pandas as pd
from .. import data
from pathlib import Path

cwd = Path(__file__).resolve().parent
file_path = cwd.parent / "assets" / "joining_dates.csv"
joining_dates = pd.read_csv(file_path)


PAGE_SIZE = 10
LOAD_MORE_SIZE = 50


def create_vote_indicator(country_name, vote):
    VOTE_MAP = {
        "Y": {"color": "green", "label": "Yes"},
        "N": {"color": "red", "label": "No"},
        "A": {"color": "orange", "label": "Abstain"},
        "X": {"color": "blue", "label": "Not Voting"},
    }
    if pd.isna(vote) or vote not in VOTE_MAP:
        return html.Span(
            [
            html.Span("●", style={"marginRight": "4px"}),
            html.Span(f"{country_name}"),
        ],
            style={"color": "#999", "marginRight": "15px", "fontSize": "0.9em"},
        )
    config = VOTE_MAP[vote]
    return html.Span(
        [
            html.Span("●", style={"color": config["color"], "marginRight": "4px"}),
            html.Span(f"{country_name}"),
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
                                            {"label": "Show All", "value": "NO_FILTER"},
                                            {"label": "Yes", "value": "Y"},
                                            {"label": "No", "value": "N"},
                                            {"label": "Abstain", "value": "A"},
                                            {"label": "Not Voting", "value": "X"},
                                        ],
                                        value="NO_FILTER",
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
                                            {"label": "Show All", "value": "NO_FILTER"},
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
                                        value="NO_FILTER",
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
                    # Placeholder or info text when multiple countries selected?
                    # Making it empty if not single country usually looks cleaner.
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
                            [html.Span("●", style={"color": color, "marginRight": "4px"}), label],
                            style={"fontSize": "0.85em", "marginRight": "14px"},
                        )
                        for color, label in [
                            ("green", "Yes"),
                            ("red", "No"),
                            ("orange", "Abstain"),
                            ("blue", "Not Voting"),
                            ("#999", "N/A"),
                        ]
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
        Input(
            "filter-component-filter-store", "data"
        ),
        Input("rl-agreement-dropdown", "value"),
        Input("rl-vote-filter", "value"),
        Input("rl-load-more-btn", "n_clicks"),
        Input("rl-sort-dropdown", "value"),
        State("country1-iso-alpha3", "data"),
    )
    def update_resolution_list(
        filter_params, agreement_filter, vote_filter, n_clicks, sort_order, country1_backup
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

        if not filter_params:
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
        start_date = filter_params.get("start_date")
        end_date = filter_params.get("end_date")
        subject_ids = filter_params.get("subject_ids")
        # Try both the params and backup store for country1, but allow it to be None
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

        # 1. Query Data
        try:
            underlying_countries = []
            for i in comparison_countries:
                underlying_countries.append(i)
            if country1:
                underlying_countries.append(country1)
            if len(underlying_countries) == 0:
                df = data.query_engine.query_resolutions(
                    start_date=start_date,
                    end_date=end_date,
                    subject_ids=subject_ids,
                    include_descendants=True,
                )
            else:
                # TODO: Possible bug in date range calculation. Could it be that max_ending_date
                #  is only checked against the *last* country in the list (instead of all of them?)
                min_starting_date = joining_dates[joining_dates['country'] == underlying_countries[0]]['min_date'].to_list()[0]
                max_ending_date = joining_dates[joining_dates['country'] == underlying_countries[0]]['max_date'].to_list()[0]
                for c in underlying_countries:
                    if (
                        joining_dates[joining_dates["country"] == c][
                            "min_date"
                        ].to_list()[0]
                        < min_starting_date
                    ):
                        min_starting_date = joining_dates[
                            joining_dates["country"] == c
                        ]["min_date"].to_list()[0]
                if (
                        joining_dates[joining_dates["country"] == c][
                            "max_date"
                        ].to_list()[0]
                        > max_ending_date
                    ):
                        max_ending_date = joining_dates[
                            joining_dates["country"] == c
                        ]["max_date"].to_list()[0]
                df = data.query_engine.query_resolutions(
                    start_date=start_date
                    if min_starting_date < start_date
                    else min_starting_date,
                    end_date=end_date if max_ending_date > end_date else max_ending_date,
                    subject_ids=subject_ids,
                    include_descendants=True,
                )
        except Exception as e:
            return (
                html.Div(f"Error querying data: {e}", style={"color": "red"}),
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

        # Keyword search
        keyword = filter_params.get("keyword")
        if keyword and keyword.strip():
            from . import wordcloud_interactive
            matched_ids = wordcloud_interactive.get_keyword_matched_ids(df, keyword)
            df = df[df["undl_id"].isin(matched_ids)]

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

        if country1 and show_agreement_filter and agreement_filter != "NO_FILTER":
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

        if country1 and not comparison_countries and vote_filter and vote_filter != "NO_FILTER":
            if country1 in filtered_df.columns:
                filtered_df = filtered_df[filtered_df[country1] == vote_filter]

        total_count = len(filtered_df)

        # 3. Pagination
        current_limit = PAGE_SIZE + ((n_clicks or 0) * LOAD_MORE_SIZE)
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
                date_val.strftime("%Y-%m-%d") if pd.notnull(date_val) else "Unknown"
            )
            title = row.get("title", "Untitled")
            consensus_score = row.get("consensus_score")

            consensus_display = f"{consensus_score:.3f}" if pd.notnull(consensus_score) else "N/A"

            indicators = []

            # Main Country (only if selected)
            if country1:
                c1_vote = row.get(country1) if country1 in row else None
                indicators.append(
                    create_vote_indicator(data.get_country_name(country1), c1_vote)
                )

            # Comparators (all selected countries)
            for c2 in comparison_countries:
                if c2 in row:
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
                                            f"{res_id}",
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
                                        f"Consensus: {consensus_display}",
                                        style={
                                            "color": "#333" if pd.notnull(consensus_score) else "#999",
                                            "fontSize": "0.9em",
                                            "marginLeft": "12px",
                                            "fontWeight": "500"
                                        },
                                    ),
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
