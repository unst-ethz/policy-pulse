import functools
import time
import numpy as np
from typing import List
from dash import Input, Output, callback, clientside_callback, html, dcc, register_page
import pandas as pd
import random

from ..features import resolution_list
from ..features import alignment_choropleth
from ..features import alignment_graph
from ..features import alignment_by_subject
from ..features import wordcloud_interactive
from ..features import resolution_finder
from ..features import filters
from .. import data


def title(countr1_alpha3=None):
    return (
        "Policy Pulse Analysis" + f" (Country 1: {countr1_alpha3})"
        if countr1_alpha3
        else ""
    )


register_page(__name__, path_template="/trends")


def layout(countr1_alpha3: str | None = None, **other_keyword_arguments):

    available = [c for c in data.available_countries if c != countr1_alpha3]
    return html.Div(
        [
            dcc.Store(id="country1-iso-alpha3", data=countr1_alpha3),
            dcc.Store(id="country1-localised-name"),
            html.H1(
                [
                    "Analysis",
                ]
            ),
            # Status and cache info
            html.Div(
                id="status-display",
            ),
            *filters.layout,
            # Tab Navigation
            dcc.Tabs(
                id="country-view-tabs",
                value="resolution_list",
                children=[
                    dcc.Tab(
                        label="Resolutions",
                        value="resolution_list",
                        children=[
                            html.Div(
                                [
                                    html.H2("Preview of resolutions"),
                                    *resolution_list.layout,
                                ],
                                className="tab-content",
                            )
                        ],
                    ),
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
                    # TAB 3: Word Cloud
                    dcc.Tab(
                        label="Word Cloud",
                        value="wordcloud",
                        children=[
                            html.Div(
                                [
                                    html.H2("Resolution Title Word Cloud"),
                                    *wordcloud_interactive.layout,
                                ],
                                className="tab-content",
                            )
                        ],
                    ),
                ],
            ),
        ]
    )


# Client-side callback from filter component's country 1 (ISO alpha3) to
# localised name.
clientside_callback(
    """
    function localise_iso_country(filter_store, navbar_clicks) {
        // Check if triggered by navbar click
        const triggered = dash_clientside.callback_context.triggered;
        if (triggered && triggered[0] && triggered[0].prop_id === 'navbar-home-click.n_clicks') {
            return null;
        }

        const iso_three_digit = filter_store.country1_alpha3;

        // Otherwise localize the country code
        if (!iso_three_digit) return null;
        const iso2 = window.getCountryISO2(iso_three_digit);
        if (!iso2) return iso_three_digit;
        return new Intl.DisplayNames(["en"], { type: "region" }).of(iso2);
    }
    """,
    Output("country1-localised-name", "data"),
    [
        Input("filter-component-filter-store", "data"),
        Input("navbar-home-click", "n_clicks"),
    ],
)

# clientside_callback(
#     """
#     function store_to_heading(localised_name) {
#         return localised_name;
#     }
#     """,
#     Output("heading-country1-name", "children"),
#     Input("country1-localised-name", "data"),
# )

filters.register_callbacks()
resolution_list.register_callbacks()
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
wordcloud_interactive.register_callbacks()


@callback(
    [
        Output("status-display", "children"),
        Output("moving-average-data", "data"),
        Output("moving-average-calc-time", "data"),
    ],
    [
        Input("filter-component-filter-store", "data"),
        Input("timespan-dropdown", "value"),
    ],
)
def _calculate_data_wrapper(filter_store, time_span: int):
    """Wrapper that converts list to tuple for caching."""
    country1 = filter_store["country1_alpha3"]
    country2: List[str] | str = filter_store["country2"]

    # normalize country2 to tuple (hashable)
    if country2 is None:
        return ("No comparison country selected.", None, None)
    selected_tuple = tuple(country2) if isinstance(country2, list) else (country2,)

    return _calculate_data_uncached(country1, selected_tuple, time_span)


@functools.lru_cache(maxsize=100)
def _calculate_data_uncached(
    country1: str | None, selected_tuple: tuple, time_span: int
):
    """Calculate moving average data for country pair (cached)."""
    if country1 is None:
        # Don't calculate yet if no primary country is selected
        return (None, None, None)

    # Normalize country2 to a list
    selected = list(selected_tuple)  # convert back to list for processing

    # remove country1 if accidentally selected
    selected = [c for c in selected if c != country1]
    if len(selected) == 0:
        return (
            "No valid countries selected to compare against. Please choose valid countries (excluding the primary country).",
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
                last_vote_date = df_original.loc[last_vote_mask, "date"].max()
                print(f"  📅 {col}: Last vote date = {last_vote_date}")
                # Set both alignment AND moving average to NaN for dates after the last vote
                rows_set_to_nan = (out["date"] > last_vote_date).sum()
                out.loc[out["date"] > last_vote_date, align_col] = np.nan
                out.loc[out["date"] > last_vote_date, sma_col] = np.nan
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
