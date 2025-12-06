from dash import Input, Output, callback, clientside_callback, html, dcc
import pandas as pd
import urllib.parse

from .. import data

prefix = "filter-component"
ids = {
    "date_picker": f"{prefix}-date-picker-range",
    "subject_dropdown": f"{prefix}-subject-dropdown",
    "country": f"{prefix}-country-dropdown",
    "country2": f"{prefix}-country2-dropdown",
    "filter_store": f"{prefix}-filter-store",
    "data_store": f"{prefix}-data-store",  # Store for queried data
    "location": f"{prefix}-location",
}


def register_callbacks():

    # Callback: Update filter store and print current selections when any filter changes
    @callback(
        Output(ids["filter_store"], "data"),
        Output(ids["location"], "search"),
        Input(ids["date_picker"], "start_date"),
        Input(ids["date_picker"], "end_date"),
        Input(ids["subject_dropdown"], "value"),
        Input(ids["country"], "value"),
        Input(ids["country2"], "value"),
        prevent_initial_call=False,
    )
    def update_filter_store(start_date, end_date, subject_ids, country_iso3, country2):
        """
        Register callbacks for the filter component.
        - Filter state management
        - Print current selections when filters change
        - Query data when filters change
        """
        filter_data = {
            "start_date": start_date,
            "end_date": end_date,
            "subject_ids": subject_ids if subject_ids else None,
            "country1_alpha3": country_iso3,
            "country2": country2,
        }

        # Print current selections
        print("\n" + "=" * 50)
        print("Current Filter Selections:")
        print(f"  Date Range: {start_date} to {end_date}")
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
            # 存储组件
            dcc.Store(id=ids["filter_store"]),
            dcc.Store(id=ids["data_store"]),  # Store for queried data
            dcc.Location(id=ids["location"], refresh=False),
            html.H2(
                "Filter the Data",
                style={"color": "#212529", "marginBottom": "10px"},
            ),
            # Filters container with better layout
            html.Div(
                [
                    # Date Range Filter - Full width
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
                                        "Date Range",
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
                            dcc.DatePickerRange(
                                id=ids["date_picker"],
                                min_date_allowed=data.get_earliest_data_date(),
                                max_date_allowed=data.get_latest_data_date(),
                                start_date=data.get_earliest_data_date(),
                                end_date=data.get_latest_data_date(),
                                display_format="YYYY-MM-DD",
                                style={"width": "100%", "fontSize": "14px"},
                                calendar_orientation="vertical",
                            ),
                        ],
                        style={
                            "marginBottom": "25px",
                            "padding": "15px",
                            "backgroundColor": "#f8f9fa",
                            "borderRadius": "8px",
                            "border": "1px solid #e9ecef",
                        },
                    ),
                    # Subjects and Country in a row
                    html.Div(
                        [
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
                                    "marginRight": "20px",
                                    "padding": "15px",
                                    "backgroundColor": "#f8f9fa",
                                    "borderRadius": "8px",
                                    "border": "1px solid #e9ecef",
                                },
                            ),
                            # Country Filter
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
                                                "Country",
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
                                                "search": data.get_country_name(
                                                    country
                                                ),
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
                                    "flex": "0 0 280px",
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
                            "gap": "0",
                        },
                    ),
                    html.Div(
                        [
                            html.Label(
                                "Select a country to compare with:",
                                style={
                                    "fontWeight": "bold",
                                    "marginBottom": "5px",
                                },
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
                                style={"marginBottom": "15px"},
                            ),
                        ],
                        style={
                            "width": "30%",
                            "display": "inline-block",
                        },
                    ),
                    html.Div(
                        [
                            html.Label(
                                "Window Size (resolutions):",
                                style={
                                    "fontWeight": "bold",
                                    "marginBottom": "5px",
                                },
                            ),
                            dcc.Dropdown(
                                id="timespan-dropdown",
                                options=[
                                    {"label": "200 resolutions", "value": 200},
                                    {"label": "350 resolutions", "value": 350},
                                    {"label": "500 resolutions", "value": 500},
                                ],
                                value=350,
                                clearable=False,
                                style={"marginBottom": "15px"},
                            ),
                        ],
                        style={"width": "30%", "display": "inline-block"},
                    ),
                ]
            ),
        ],
    ),
)
