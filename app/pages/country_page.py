import functools
import time
import numpy as np
from typing import List
from dash import Input, Output, callback, clientside_callback, html, dcc, register_page
import pandas as pd
import random

from ..features import breadcrumb
from ..features import alignment_choropleth
from ..features import alignment_graph
from ..features import alignment_by_subject
from ..features import wordcloud_viz
from ..features import resolution_finder
from .. import data


def title(country_code_alpha3=None):
    return f"Country-specific Policy Pulse: {country_code_alpha3}"


register_page(__name__, path_template="/country/<country_code_alpha3>")


def layout(country_code_alpha3: str | None = None):

    available = [c for c in data.available_countries if c != country_code_alpha3]
    default_countries = random.sample(available, k=4)
    return html.Div(
        [
            dcc.Store(id="country1-iso-alpha3", data=country_code_alpha3),
            dcc.Store(id="country1-localised-name"),
            html.H1(
                [
                    html.Span(id="heading-country1-name", style={"fontWeight": "bold"}),
                    "'s Policy Pulse",
                ]
            ),
            # Status and cache info
            html.Div(
                id="status-display",
            ),
            # Tab Navigation
            dcc.Tabs(
                id="country-view-tabs",
                value="map",
                children=[
                    # TAB 1: Agreement Map
                    dcc.Tab(
                        label="Agreement Map",
                        value="map",
                        children=[
                            html.Div(
                                [
                                    html.H2("Global Alignment Map"),
                                    *alignment_choropleth.layout,
                                ],
                                className="tab-content",
                            )
                        ],
                    ),
                    # TAB 2: Agreement Timeline
                    dcc.Tab(
                        label="Agreement Timeline",
                        value="timeline",
                        children=[
                            html.Div(
                                [
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
                                                            {
                                                                "label": data.get_country_name(country),
                                                                "value": country,
                                                            }
                                                            for country in data.available_countries
                                                        ],
                                                        value=default_countries,
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
                                    *alignment_graph.layout,
                                ],
                                className="tab-content",
                            )
                        ],
                    ),
                    # TAB 3: Alignment by Subject
                    dcc.Tab(
                        label="Alignment by Subject",
                        value="subject",
                        children=[
                            html.Div(
                                [
                                    html.H2("Alignment by UN Subject Area"),
                                    html.P(
                                        "Compare voting alignment between two countries across different UN subject areas. "
                                        "The agreement score ranges from 0 (complete disagreement) to 1 (complete agreement). "
                                        "Only subjects with at least 30 shared votes are shown.",
                                        style={"color": "#7f8c8d", "marginBottom": "20px"}
                                    ),
                                    *alignment_by_subject.layout,
                                ],
                                className="tab-content",
                            )
                        ],
                    ),
                    # TAB 4: Resolution Finder
                    dcc.Tab(
                        label="Resolution Finder",
                        value="resolution_finder",
                        children=[
                            html.Div(
                                [
                                    html.H2("Find Resolutions"),
                                    html.P(
                                        "Search and filter UN resolutions to see how this country voted.",
                                        style={"color": "#7f8c8d", "marginBottom": "20px"}
                                    ),
                                    resolution_finder.layout,
                                ],
                                className="tab-content",
                            )
                        ],
                    ),
                ],
            ),
            # Footer with instructions
            html.Div(
                [
                    html.Hr(),
                    html.P(
                        [
                            "💡 ",
                            html.Strong("How it works:"),
                            " Use the tabs above to switch between visualizations. ",
                            "Select countries and time spans in the Timeline tab. ",
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
        ]
    )


# Client-side callback from country1-iso-alpha3 (ISO alpha3) to localised name.
clientside_callback(
    """
    function localise_iso_country(iso_three_digit, navbar_clicks) {
        // Check if triggered by navbar click
        const triggered = dash_clientside.callback_context.triggered;
        if (triggered && triggered[0] && triggered[0].prop_id === 'navbar-home-click.n_clicks') {
            return null;
        }
        
        // Otherwise localize the country code
        if (!iso_three_digit) return null;
        const iso2 = window.getCountryISO2(iso_three_digit);
        if (!iso2) return iso_three_digit;
        return new Intl.DisplayNames(["en"], { type: "region" }).of(iso2);
    }
    """,
    Output("country1-localised-name", "data"),
    [Input("country1-iso-alpha3", "data"), Input("navbar-home-click", "n_clicks")],
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

alignment_choropleth.register_callbacks(data.query_engine)
alignment_graph.register_callbacks()
alignment_by_subject.register_callbacks(data.query_engine)
resolution_finder.register_callbacks(data.query_engine)


@callback(
    Output("subject-country2-dropdown", "options"),
    Output("subject-country2-dropdown", "value"),
    Input("country1-iso-alpha3", "data"),
)
def populate_subject_country_dropdown(country1):
    """Populate the subject tab country dropdown with all countries except the selected one."""
    options = [
        {"label": data.get_country_name(country), "value": country}
        for country in data.available_countries
        if country != country1
    ]
    # Select a random default country
    available = [c for c in data.available_countries if c != country1]
    default_country = random.choice(available) if available else None
    return options, default_country


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
def _calculate_data_wrapper(country1: str, country2: List[str] | str, time_span: int):
    """Wrapper that converts list to tuple for caching."""
    # normalize country2 to tuple (hashable)
    if country2 is None:
        return ("No comparison country selected.", None, None)
    selected_tuple = tuple(country2) if isinstance(country2, list) else (country2,)

    return _calculate_data_uncached(country1, selected_tuple, time_span)


@functools.lru_cache(maxsize=100)
def _calculate_data_uncached(country1: str, selected_tuple: tuple, time_span: int):
    """Calculate moving average data for country pair (cached)."""
    # Normalize country2 to a list
    selected = list(selected_tuple)  # convert back to list for processing

    # remove country1 if accidentally selected
    selected = [c for c in selected if c != country1]
    if len(selected) == 0:
        return (
            "No valid countries selected. Please choose valid countries (excluding the primary country).",
            None,
            None,
        )

    print(f"🔄 Calculating {country1} vs {selected} (span: {time_span})")
    start_time = time.time()

    try:
        vote_mapping = {"Y": 1, "A": 0, "N": -1}

        # Pull only required columns
        cols = ["date", country1] + selected
        df = data.query_engine.query_resolutions()[cols].copy()

        # normalize and map votes to numeric (robust handling)
        vote_cols = [country1] + selected
        # convert to string, strip whitespace, uppercase
        df[vote_cols] = (
            df[vote_cols].astype(str).apply(lambda s: s.str.strip().str.upper())
        )
        # replace common null-like strings with actual NaN
        df[vote_cols] = df[vote_cols].replace(
            {"": pd.NA, "NAN": pd.NA, "NONE": pd.NA, "<NA>": pd.NA}
        )
        # map to numbers
        df[vote_cols] = df[vote_cols].replace(vote_mapping)
        # coerce any remaining non-numeric to NaN
        df[vote_cols] = df[vote_cols].apply(pd.to_numeric, errors="coerce")

        # ensure dates and sort
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.sort_values("date").reset_index(drop=True)

        # start after first row where country1 and at least one selected country voted
        mask_any = (~df[country1].isna()) & df[selected].notna().any(axis=1)
        if not mask_any.any():
            return ("No overlapping votes between the selected countries.", None, None)
        first_pos = mask_any.values.argmax()
        
        # Keep a reference to the original df for finding last vote dates
        df_original = df.copy()
        df = df.iloc[first_pos:].reset_index(drop=True)

        # compute per-country agreement (vectorized)
        diffs = df[selected].subtract(df[country1], axis=0).abs()
        agreement = 1.0 - (diffs / 2.0)

        # set agreement to NaN where either side didn't vote
        for col in selected:
            both_voted = df[[country1, col]].notna().all(axis=1)
            agreement[col] = agreement[col].where(both_voted, np.nan)

        # build output dataframe with per-country alignment and moving averages
        out = pd.DataFrame({"date": df["date"]})
        for col in selected:
            align_col = f"alignment_{col}"
            sma_col = f"sma_{col}"

            out[align_col] = agreement[col]
            
            out[sma_col] = (
                out[align_col]
                .rolling(window=time_span, min_periods=time_span // 4)
                .mean()
            )
            
            # Find the last date this country actually voted
            # Use the original unfiltered df to find the last non-null vote for this country
            last_vote_mask = df_original[col].notna()
            if last_vote_mask.any():
                last_vote_date = df_original.loc[last_vote_mask, 'date'].max()
                print(f"  📅 {col}: Last vote date = {last_vote_date}")
                # Set both alignment AND moving average to NaN for dates after the last vote
                rows_set_to_nan = (out['date'] > last_vote_date).sum()
                out.loc[out['date'] > last_vote_date, align_col] = np.nan
                out.loc[out['date'] > last_vote_date, sma_col] = np.nan
                print(f"      Set {rows_set_to_nan} rows to NaN after {last_vote_date}")

        calc_time = time.time() - start_time
        print(f"✅ Calculated in {calc_time:.2f}s ({len(out):,} points)")

        return (None, out.to_json(date_format="iso"), calc_time)

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
