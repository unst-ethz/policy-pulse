from dash import Input, Output, State, callback, clientside_callback, html, dcc
import pandas as pd
import urllib.parse

from .. import data

prefix = "filter-component"
ids = {
    "start_year": f"{prefix}-start-year",
    "end_year": f"{prefix}-end-year",
    "era_preset": f"{prefix}-era-preset",
    "subject_dropdown": f"{prefix}-subject-dropdown",
    "country": f"{prefix}-country-dropdown",
    "country2": f"{prefix}-country2-dropdown",
    "preset": f"{prefix}-preset-dropdown",
    "reset_btn": f"{prefix}-reset-btn",
    "filter_store": f"{prefix}-filter-store",
    "data_store": f"{prefix}-data-store",
    "location": f"{prefix}-location",
}

# Country group presets for quick comparison selection
PRESETS = {
    "p5": {
        "label": "P5 (Security Council)",
        "countries": ["USA", "GBR", "FRA", "RUS", "CHN"],
    },
    "g7": {
        "label": "G7",
        "countries": ["USA", "GBR", "FRA", "DEU", "ITA", "JPN", "CAN"],
    },
    "brics": {
        "label": "BRICS",
        "countries": ["BRA", "RUS", "IND", "CHN", "ZAF"],
    },
    "eu_major": {
        "label": "EU (Major)",
        "countries": ["DEU", "FRA", "ITA", "ESP", "NLD", "POL", "SWE"],
    },
    "nordic": {
        "label": "Nordic",
        "countries": ["SWE", "NOR", "FIN", "DNK", "ISL"],
    },
}

# Era presets for quick year range selection
ERA_PRESETS = {
    "cold_war": {
        "label": "Cold War (1947–1991)",
        "start": 1947,
        "end": 1991,
    },
    "post_cold_war": {
        "label": "Post Cold War (1992–2001)",
        "start": 1992,
        "end": 2001,
    },
    "war_on_terror": {
        "label": "War on Terror (2001–2014)",
        "start": 2001,
        "end": 2014,
    },
    "recent": {
        "label": "Recent (2015–present)",
        "start": 2015,
        "end": None,  # will use latest year
    },
}


def register_callbacks():

    # Callback: Apply era preset to year dropdowns
    @callback(
        Output(ids["start_year"], "value", allow_duplicate=True),
        Output(ids["end_year"], "value", allow_duplicate=True),
        Input(ids["era_preset"], "value"),
        prevent_initial_call=True,
    )
    def apply_era_preset(preset_key):
        if not preset_key or preset_key not in ERA_PRESETS:
            from dash import no_update
            return no_update, no_update
        era = ERA_PRESETS[preset_key]
        return era["start"], era["end"] or data.get_latest_year()

    # Callback: Apply preset to comparison countries
    @callback(
        Output(ids["country2"], "value"),
        Input(ids["preset"], "value"),
        prevent_initial_call=True,
    )
    def apply_preset(preset_key):
        if not preset_key or preset_key not in PRESETS:
            return []
        return PRESETS[preset_key]["countries"]

    # Callback: Reset all filters except main country
    @callback(
        Output(ids["start_year"], "value"),
        Output(ids["end_year"], "value"),
        Output(ids["era_preset"], "value"),
        Output(ids["subject_dropdown"], "value"),
        Output(ids["country2"], "value", allow_duplicate=True),
        Output(ids["preset"], "value", allow_duplicate=True),
        Input(ids["reset_btn"], "n_clicks"),
        prevent_initial_call=True,
    )
    def reset_filters(n_clicks):
        return (
            data.get_earliest_year(),
            data.get_latest_year(),
            None,
            None,
            [],
            None,
        )

    # Callback: Update filter store and print current selections when any filter changes
    @callback(
        Output(ids["filter_store"], "data"),
        Output(ids["location"], "search"),
        Input(ids["start_year"], "value"),
        Input(ids["end_year"], "value"),
        Input(ids["subject_dropdown"], "value"),
        Input(ids["country"], "value"),
        Input(ids["country2"], "value"),
        prevent_initial_call=False,
    )
    def update_filter_store(start_year, end_year, subject_ids, country_iso3, country2):
        """
        Register callbacks for the filter component.
        - Filter state management
        - Print current selections when filters change
        - Query data when filters change
        """
        # Convert years to inclusive date range (Jan 1 of start year to Dec 31 of end year)
        start_date = f"{start_year}-01-01" if start_year else None
        end_date = f"{end_year}-12-31" if end_year else None

        filter_data = {
            "start_date": start_date,
            "end_date": end_date,
            "start_year": start_year,
            "end_year": end_year,
            "subject_ids": subject_ids if subject_ids else None,
            "country1_alpha3": country_iso3,
            "country2": country2,
        }

        # Print current selections
        print("\n" + "=" * 50)
        print("Current Filter Selections:")
        print(f"  Year Range: {start_year} to {end_year}")
        print(f"  Subjects: {subject_ids if subject_ids else 'None'}")
        print(f"  Country: {country_iso3 if country_iso3 else 'None'}")
        print("=" * 50 + "\n")

        return filter_data, "?" + urllib.parse.urlencode(filter_data)

    # Callback: Query data when filters change
    @callback(
        Output(ids["data_store"], "data"),
        Input(ids["filter_store"], "data"),
        prevent_initial_call=False,
    )
    def query_data_on_filter_change(filter_data):
        """Query data based on current filter selections."""
        if not filter_data:
            return None

        try:
            start_date = filter_data.get("start_date")
            end_date = filter_data.get("end_date")
            subject_ids = filter_data.get("subject_ids")
            country = filter_data.get("country1_alpha3")

            # Query resolutions using the query engine
            df = data.query_engine.query_resolutions(
                start_date=start_date,
                end_date=end_date,
                subject_ids=subject_ids,
                include_descendants=True,
            )

            # If country filter is selected, filter by country vote
            if country and country in df.columns:
                # Only keep rows where the country has a vote (not NaN)
                df = df.dropna(subset=[country])

            # Convert to JSON for storage
            # Select key columns similar to app_mlkeyword.py
            result_df = (
                df[["undl_id", "resolution", "date", "title"]].copy()
                if not df.empty
                else pd.DataFrame()
            )

            print(f"\n✅ Queried {len(result_df)} resolutions")
            if country:
                print(f"   Filtered by country: {country}")

            return result_df.to_json(date_format="iso", orient="split")

        except Exception as e:
            print(f"\n❌ Error querying data: {e}")
            import traceback

            traceback.print_exc()
            return None


layout = (
    html.Div(
        style={
            "maxWidth": "1400px",
            "margin": "0 auto",
            "backgroundColor": "#ffffff",
            "padding": "30px",
            "borderRadius": "12px",
            "boxShadow": "0 2px 8px rgba(0,0,0,0.08)",
        },
        children=[
            # Hidden stores
            dcc.Store(id=ids["filter_store"]),
            dcc.Store(id=ids["data_store"]),
            dcc.Location(id=ids["location"], refresh=False),
            # Header row: title + reset button
            html.Div(
                [
                    html.H2(
                        "Filter the Data",
                        style={"color": "#212529", "margin": "0"},
                    ),
                    html.Button(
                        "↺ Reset Filters",
                        id=ids["reset_btn"],
                        n_clicks=0,
                        style={
                            "backgroundColor": "transparent",
                            "border": "1px solid #adb5bd",
                            "borderRadius": "4px",
                            "color": "#495057",
                            "padding": "6px 14px",
                            "fontSize": "13px",
                            "cursor": "pointer",
                            "fontFamily": "inherit",
                        },
                    ),
                ],
                style={
                    "display": "flex",
                    "justifyContent": "space-between",
                    "alignItems": "center",
                    "marginBottom": "10px",
                },
            ),
            # Filters container
            html.Div(
                [
                    # ROW 1: Main Country
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Span(
                                        "🌍",
                                        style={
                                            "fontSize": "18px",
                                            "marginRight": "8px",
                                        },
                                    ),
                                    html.Label(
                                        "Main Country",
                                        style={
                                            "fontWeight": "600",
                                            "color": "#495057",
                                            "fontSize": "15px",
                                            "marginBottom": "8px",
                                            "display": "block",
                                        },
                                    ),
                                ]
                            ),
                            dcc.Dropdown(
                                id=ids["country"],
                                options=[
                                    {
                                        "label": data.get_country_name(country),
                                        "value": country,
                                        "search": data.get_country_name(country),
                                    }
                                    for country in data.available_countries
                                ],
                                placeholder="Select a country...",
                                clearable=True,
                                style={
                                    "width": "100%",
                                    "fontSize": "14px",
                                },
                                searchable=True,
                            ),
                        ],
                        style={
                            "marginBottom": "20px",
                            "padding": "15px",
                            "backgroundColor": "#f8f9fa",
                            "borderRadius": "8px",
                            "border": "1px solid #e9ecef",
                        },
                    ),
                    # ROW 2: Comparison Countries + Preset
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Span(
                                                "🔄",
                                                style={
                                                    "fontSize": "18px",
                                                    "marginRight": "8px",
                                                },
                                            ),
                                            html.Label(
                                                "Compare with",
                                                style={
                                                    "fontWeight": "600",
                                                    "color": "#495057",
                                                    "fontSize": "15px",
                                                    "marginBottom": "8px",
                                                    "display": "block",
                                                },
                                            ),
                                        ]
                                    ),
                                    dcc.Dropdown(
                                        id=ids["country2"],
                                        options=[
                                            {
                                                "label": data.get_country_name(country),
                                                "value": country,
                                            }
                                            for country in data.available_countries
                                        ],
                                        value=[],
                                        multi=True,
                                        clearable=False,
                                        placeholder="Select countries to compare...",
                                        style={
                                            "width": "100%",
                                            "fontSize": "14px",
                                        },
                                        searchable=True,
                                    ),
                                ],
                                style={"flex": "1"},
                            ),
                            # Preset dropdown
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Span(
                                                "⚡",
                                                style={
                                                    "fontSize": "18px",
                                                    "marginRight": "8px",
                                                },
                                            ),
                                            html.Label(
                                                "Quick Select",
                                                style={
                                                    "fontWeight": "600",
                                                    "color": "#495057",
                                                    "fontSize": "15px",
                                                    "marginBottom": "8px",
                                                    "display": "block",
                                                },
                                            ),
                                        ]
                                    ),
                                    dcc.Dropdown(
                                        id=ids["preset"],
                                        options=[
                                            {"label": p["label"], "value": k}
                                            for k, p in PRESETS.items()
                                        ],
                                        placeholder="Choose a group...",
                                        clearable=True,
                                        style={
                                            "width": "100%",
                                            "fontSize": "14px",
                                        },
                                    ),
                                ],
                                style={"flex": "0 0 220px"},
                            ),
                        ],
                        style={
                            "display": "flex",
                            "flexDirection": "row",
                            "gap": "20px",
                            "marginBottom": "20px",
                            "padding": "15px",
                            "backgroundColor": "#f8f9fa",
                            "borderRadius": "8px",
                            "border": "1px solid #e9ecef",
                        },
                    ),
                    # ROW 3: Date Range and Subjects side by side
                    html.Div(
                        [
                            # Year Range Filter
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Span(
                                                "📅",
                                                style={
                                                    "fontSize": "18px",
                                                    "marginRight": "8px",
                                                },
                                            ),
                                            html.Label(
                                                "Year Range",
                                                style={
                                                    "fontWeight": "600",
                                                    "color": "#495057",
                                                    "fontSize": "15px",
                                                    "marginBottom": "8px",
                                                    "display": "block",
                                                },
                                            ),
                                        ]
                                    ),
                                    html.Div(
                                        [
                                            dcc.Dropdown(
                                                id=ids["start_year"],
                                                options=[
                                                    {"label": str(y), "value": y}
                                                    for y in range(
                                                        data.get_earliest_year(),
                                                        data.get_latest_year() + 1,
                                                    )
                                                ],
                                                value=data.get_earliest_year(),
                                                clearable=False,
                                                searchable=True,
                                                style={"flex": "1", "fontSize": "14px"},
                                            ),
                                            html.Span(
                                                "–",
                                                style={
                                                    "padding": "0 8px",
                                                    "color": "#6c757d",
                                                    "fontWeight": "600",
                                                    "alignSelf": "center",
                                                },
                                            ),
                                            dcc.Dropdown(
                                                id=ids["end_year"],
                                                options=[
                                                    {"label": str(y), "value": y}
                                                    for y in range(
                                                        data.get_earliest_year(),
                                                        data.get_latest_year() + 1,
                                                    )
                                                ],
                                                value=data.get_latest_year(),
                                                clearable=False,
                                                searchable=True,
                                                style={"flex": "1", "fontSize": "14px"},
                                            ),
                                        ],
                                        style={
                                            "display": "flex",
                                            "flexDirection": "row",
                                            "alignItems": "center",
                                            "gap": "4px",
                                        },
                                    ),
                                    dcc.Dropdown(
                                        id=ids["era_preset"],
                                        options=[
                                            {"label": e["label"], "value": k}
                                            for k, e in ERA_PRESETS.items()
                                        ],
                                        placeholder="Quick select era...",
                                        clearable=True,
                                        style={
                                            "width": "100%",
                                            "fontSize": "13px",
                                            "marginTop": "8px",
                                        },
                                    ),
                                ],
                                style={
                                    "flex": "0 0 23%",
                                    "padding": "15px",
                                    "backgroundColor": "#f8f9fa",
                                    "borderRadius": "8px",
                                    "border": "1px solid #e9ecef",
                                },
                            ),
                            # Subjects Filter
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Span(
                                                "📚",
                                                style={
                                                    "fontSize": "18px",
                                                    "marginRight": "8px",
                                                },
                                            ),
                                            html.Label(
                                                "Subjects",
                                                style={
                                                    "fontWeight": "600",
                                                    "color": "#495057",
                                                    "fontSize": "15px",
                                                    "marginBottom": "8px",
                                                    "display": "block",
                                                },
                                            ),
                                        ]
                                    ),
                                    dcc.Dropdown(
                                        id=ids["subject_dropdown"],
                                        options=data.available_subjects(),
                                        multi=True,
                                        placeholder="Select one or more subjects...",
                                        style={
                                            "width": "100%",
                                            "fontSize": "14px",
                                        },
                                        searchable=True,
                                    ),
                                ],
                                style={
                                    "flex": "1",
                                    "padding": "15px",
                                    "backgroundColor": "#f8f9fa",
                                    "borderRadius": "8px",
                                    "border": "1px solid #e9ecef",
                                },
                            ),
                        ],
                        style={
                            "display": "flex",
                            "flexDirection": "row",
                            "gap": "20px",
                        },
                    ),
                ]
            ),
        ],
    ),
)
