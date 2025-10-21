import functools
import time
from dash import Input, Output, callback, clientside_callback, html, dcc, register_page
import pandas as pd

from ..components import breadcrumb
from ..components import alignment_choropleth
from ..components import alignment_graph
from ..components import wordcloud_viz
from .. import data


def title(country_code_alpha3=None):
    return f"Country-specific Policy Pulse: {country_code_alpha3}"


register_page(__name__, path_template="/country/<country_code_alpha3>")


def layout(country_code_alpha3: str | None = None):
    return html.Div(
        [
            dcc.Store(id="country1-iso-alpha3", data=country_code_alpha3),
            dcc.Store(id="country1-localised-name"),
            html.Div(
                className="container",
                children=[
                    *breadcrumb.layout,
                    html.H1(
                        [
                            html.Span(
                                id="heading-country1-name", style={"fontWeight": "bold"}
                            ),
                            "'s Policy Pulse",
                        ]
                    ),
                    # Status and cache info
                    html.Div(
                        id="status-display",
                    ),
                    html.H2("Global Alignment Map"),
                    *alignment_choropleth.layout,
                    html.H2("Bi-country Alignment Comparison Graph"),
                    html.Div(
                        [
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
                                        id="country2-dropdown",
                                        options=[
                                            {"label": country, "value": country}
                                            for country in data.available_countries
                                        ],
                                        value=(
                                            data.available_countries[1]
                                            if len(data.available_countries) > 1
                                            else data.available_countries[0]
                                        ),
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
                                        "Time Span (days):",
                                        style={
                                            "fontWeight": "bold",
                                            "marginBottom": "5px",
                                        },
                                    ),
                                    dcc.Dropdown(
                                        id="timespan-dropdown",
                                        options=[
                                            {"label": "30 days", "value": 30},
                                            {"label": "90 days", "value": 90},
                                            {"label": "180 days", "value": 180},
                                            {"label": "365 days", "value": 365},
                                            {
                                                "label": "730 days (2 years)",
                                                "value": 730,
                                            },
                                        ],
                                        value=365,
                                        clearable=False,
                                        style={"marginBottom": "15px"},
                                    ),
                                ],
                                style={"width": "30%", "display": "inline-block"},
                            ),
                        ]
                    ),
                    *alignment_graph.layout,
                    html.H2(
                        "Keyword Wordcloud for GA Resolution Subjects (Not Country Specific)"
                    ),
                    *wordcloud_viz.layout,
                    # Footer with instructions
                    html.Div(
                        [
                            html.Hr(),
                            html.P(
                                [
                                    "💡 ",
                                    html.Strong("How it works:"),
                                    " Select countries and time span above. ",
                                    "Data is calculated on-demand and cached for fast re-access. ",
                                    "Alignment ranges from 0 (complete disalignment) to 1 (perfect alignment).",
                                ],
                                style={
                                    "color": "#7f8c8d",
                                    "textAlign": "center",
                                    "fontSize": "14px",
                                },
                            ),
                        ],
                        style={"padding": "20px", "marginTop": "40px"},
                    ),
                ],
            ),
        ]
    )


# Client-side callback from country1-iso-alpha3 (ISO alpha3) to localised name.
clientside_callback(
    """
    function localise_iso_country(iso_three_digit) {
        return new Intl.DisplayNames(["en"], { type: "region" }).of(window.getCountryISO2(iso_three_digit));
    }
    """,
    Output("country1-localised-name", "data"),
    Input("country1-iso-alpha3", "data"),
)

clientside_callback(
    """
    function store_to_heading(localised_name) {
        return localised_name;
    }
    """,
    Output("heading-country1-name", "children"),
    Input("country1-localised-name", "data"),
)

breadcrumb.register_callbacks()
alignment_choropleth.register_callbacks(data.query_engine)
alignment_graph.register_callbacks()
wordcloud_viz.register_callbacks()


@callback(
    [
        Output("status-display", "children"),
        Output("moving-average-data", "data"),
        Output("moving-average-calc-time", "data"),
    ],
    [
        Input("country1-iso-alpha3", "data"),
        Input("country2-dropdown", "value"),
        Input("timespan-dropdown", "value"),
    ],
)
@functools.lru_cache(maxsize=100)
def _calculate_data_uncached(country1: str, country2: str, time_span: int):
    """Calculate moving average data for country pair (cached)."""
    if country1 == country2:
        return (
            "Same country selected for both dropdowns. Please choose different countries.",
            None,
            None,
        )

    print(f"🔄 Calculating {country1} vs {country2} (span: {time_span})")
    start_time = time.time()

    try:
        # Calculate alignment
        def calc_alignment(row: pd.Series):
            vote_mapping = {"Y": 1, "A": 0, "N": -1}
            if row[country1] in vote_mapping and row[country2] in vote_mapping:
                diff = abs(vote_mapping[row[country1]] - vote_mapping[row[country2]])
                return 1 - (diff / 2)
            return float("nan")

        # Process data
        df_subset = data.df_ga_transformed[["date", country1, country2]].copy()
        df_subset["alignment"] = df_subset.apply(calc_alignment, axis=1)
        df_subset["date"] = pd.to_datetime(df_subset["date"])
        df_subset = df_subset.sort_values("date").reset_index(drop=True)

        # Apply date filters
        # if self.start_date:
        #     df_subset = df_subset[df_subset["date"] >= pd.to_datetime(self.start_date)]
        # if self.end_date:
        #     df_subset = df_subset[df_subset["date"] <= pd.to_datetime(self.end_date)]

        # Calculate moving averages with the specified time span
        df_subset["sma"] = (
            df_subset["alignment"].rolling(window=time_span, min_periods=1).mean()
        )
        df_subset["ema"] = (
            df_subset["alignment"].ewm(span=time_span, adjust=False).mean()
        )
        df_subset["cma"] = df_subset["alignment"].expanding(min_periods=1).mean()

        calc_time = time.time() - start_time
        print(f"✅ Calculated in {calc_time:.2f}s ({len(df_subset):,} points)")

        return None, df_subset.to_json(), calc_time

    except Exception as e:
        print(f"❌ Calculation error: {e}")
        return (
            html.Div(
                style={
                    "padding": "10px",
                    "marginBottom": "20px",
                    "backgroundColor": "#d5dbdb",
                    "border": "1px solid #bdc3c7",
                    "borderRadius": "5px",
                },
                children=str(e),
            ),
            None,
            None,
        )
