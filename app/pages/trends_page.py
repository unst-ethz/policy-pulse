import functools
import time
import numpy as np
from typing import List
from dash import Input, Output, State, callback, clientside_callback, html, dcc, register_page
import pandas as pd
import random

from ..features import resolution_list
from ..features import alignment_choropleth
from ..features import alignment_graph
from ..features import alignment_by_subject
from ..features import wordcloud_interactive
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
                    "Analysis of GA Votes",
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
                    # TAB 1: Resolution List
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
                    # TAB 2: Agreement Map
                    dcc.Tab(
                        label="Agreement Map",
                        value="map",
                        id="tab-agreement-map",
                        disabled=False,
                        children=[
                            html.Div(
                                id="tab-map-content",
                                className="tab-content",
                            )
                        ],
                    ),
                    # TAB 3: Agreement Timeline
                    dcc.Tab(
                        label="Agreement Timeline",
                        value="timeline",
                        id="tab-agreement-timeline",
                        disabled=False,
                        children=[
                            html.Div(
                                id="tab-timeline-content",
                                className="tab-content",
                            )
                        ],
                    ),
                    # TAB 4: Alignment by Subject
                    dcc.Tab(
                        label="Alignment by Subject",
                        value="subject",
                        id="tab-alignment-subject",
                        disabled=False,
                        children=[
                            html.Div(
                                id="tab-subject-content",
                                className="tab-content",
                            )
                        ],
                    ),
                    # TAB 5: Word Cloud
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
    function localise_iso_country(filter_store) {
        if (!filter_store) return null;
        const iso_three_digit = filter_store.country1_alpha3;

        // Localize the country code
        if (!iso_three_digit) return null;
        const iso2 = window.getCountryISO2(iso_three_digit);
        if (!iso2) return iso_three_digit;
        return new Intl.DisplayNames(["en"], { type: "region" }).of(iso2);
    }
    """,
    Output("country1-localised-name", "data"),
    [
        Input("filter-component-filter-store", "data"),
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




wordcloud_interactive.register_callbacks()


@callback(
    [
        Output("tab-agreement-map", "style"),
        Output("tab-agreement-timeline", "style"),
        Output("tab-alignment-subject", "style"),
        Output("tab-map-content", "children"),
        Output("tab-timeline-content", "children"),
        Output("tab-subject-content", "children"),
    ],
    Input("filter-component-filter-store", "data"),
    prevent_initial_call=False,
)
def update_tab_states(filter_store):
    """Gray out tabs visually based on country selection requirements."""
    if not filter_store:
        country1 = None
        country2 = None
    else:
        country1 = filter_store.get("country1_alpha3")
        country2_raw = filter_store.get("country2")
        # Handle country2 as list or string
        if isinstance(country2_raw, list):
            country2 = country2_raw[0] if country2_raw else None
        elif isinstance(country2_raw, str) and country2_raw:
            country2 = country2_raw
        else:
            country2 = None
    
    no_main_country = country1 is None or country1 == ""
    comparison_disabled = no_main_country or (country2 is None or country2 == "")
    tab_normal_style = {}
    tab_grayed_style = {
        "opacity": 0.45,
        "filter": "grayscale(100%)",
        "transition": "opacity 120ms ease",
    }
    # Agreement Map requires main country only.
    # Agreement Timeline / Alignment by Subject require both main + compare country.
    map_tab_style = tab_grayed_style if no_main_country else tab_normal_style
    timeline_tab_style = tab_grayed_style if comparison_disabled else tab_normal_style
    subject_tab_style = tab_grayed_style if comparison_disabled else tab_normal_style
    
    # Placeholder content for disabled tabs
    placeholder_no_country1 = html.Div(
        [
            html.P(
                "Please select a main country from the filters above to view this analysis.",
                style={
                    "color": "#7f8c8d",
                    "fontSize": "16px",
                    "textAlign": "center",
                    "padding": "40px",
                    "fontStyle": "italic",
                }
            )
        ]
    )
    
    placeholder_no_comparison = html.Div(
        [
            html.P(
                "Please select a comparison country from the filters above to view this analysis.",
                style={
                    "color": "#7f8c8d",
                    "fontSize": "16px",
                    "textAlign": "center",
                    "padding": "40px",
                    "fontStyle": "italic",
                }
            )
        ]
    )
    
    # Content when enabled - wrap in a list to match Dash's expected format
    if no_main_country:
        map_content = [placeholder_no_country1]
    else:
        map_content = [
            html.H2("Global Agreement Map"),
            *alignment_choropleth.layout,
        ]
    
    if comparison_disabled:
        if country1 is None or country1 == "":
            timeline_content = [placeholder_no_country1]
            subject_content = [placeholder_no_country1]
        else:
            timeline_content = [placeholder_no_comparison]
            subject_content = [placeholder_no_comparison]
    else:
        timeline_content = [
            html.H2("Bi-country Alignment Comparison Graph"),
            *alignment_graph.layout,
        ]
        subject_content = [
            html.H2("Alignment by UN Subject Area"),
            html.P(
                "Compare voting alignment between two countries across different UN subject areas. "
                "The agreement score ranges from 0 (complete disagreement) to 1 (complete agreement). "
                "Only subjects with at least 30 shared votes are shown.",
                style={"color": "#7f8c8d", "marginBottom": "20px"}
            ),
            *alignment_by_subject.layout,
        ]
    
    return (
        map_tab_style,
        timeline_tab_style,
        subject_tab_style,
        map_content,
        timeline_content,
        subject_content,
    )


@callback(
    Output("country-view-tabs", "value", allow_duplicate=True),
    Input("country-view-tabs", "value"),
    Input("filter-component-filter-store", "data"),
    prevent_initial_call=True,
)
def prevent_disabled_tab_switch(selected_tab, filter_store):
    """Keep tab switching enabled; tab availability is visual only."""
    return selected_tab


@callback(
    Output("download-btn", "disabled"),
    Output("download-btn", "style"),
    Input("filter-component-filter-store", "data"),
    prevent_initial_call=False,
)
def toggle_download_button(filter_store):
    base_style = {
        "border": "none",
        "borderRadius": "4px",
        "padding": "6px 14px",
        "fontSize": "13px",
        "fontFamily": "inherit",
        "fontWeight": "600",
    }
    country1 = (filter_store or {}).get("country1_alpha3")
    if not country1:
        return True, {**base_style, "backgroundColor": "#adb5bd", "color": "white", "cursor": "not-allowed"}
    return False, {**base_style, "backgroundColor": "#1a73e8", "color": "white", "cursor": "pointer"}


@callback(
    Output("download-resolutions-csv", "data"),
    Input("download-btn", "n_clicks"),
    State("filter-component-filter-store", "data"),
    prevent_initial_call=True,
)
def download_resolutions_csv(n_clicks, filter_store):
    """Build a CSV of all filtered resolutions + votes for selected countries."""
    if not filter_store:
        return None

    start_date = filter_store.get("start_date")
    end_date = filter_store.get("end_date")
    subject_ids = filter_store.get("subject_ids")
    country1 = filter_store.get("country1_alpha3")
    country2_raw = filter_store.get("country2")

    comparison: list[str] = []
    if isinstance(country2_raw, list):
        comparison = country2_raw
    elif isinstance(country2_raw, str) and country2_raw:
        comparison = [country2_raw]

    df = data.query_engine.query_resolutions(
        start_date=start_date,
        end_date=end_date,
        subject_ids=subject_ids,
        include_descendants=True,
    )

    # Apply keyword filter: OR logic across comma-separated phrases
    keyword = filter_store.get("keyword")
    if keyword and keyword.strip() and not df.empty:
        tokens = [t.strip() for t in keyword.split(",") if t.strip()]
        matched_ids: set = set()
        for token in tokens:
            title_match = df["title"].str.lower().str.contains(token.lower(), regex=False, na=False)
            matched_ids |= set(df.loc[title_match, "undl_id"].tolist())
            matched_ids |= wordcloud_interactive.search_keywords(token)
        df = df[df["undl_id"].isin(matched_ids)]

    base_cols = ["undl_id", "resolution", "date", "session", "title", "agenda_title", "subjects", "draft"]
    if "undl_link" in df.columns:
        base_cols.append("undl_link")

    vote_cols = []
    if country1 and country1 in df.columns:
        vote_cols.append(country1)
    for c in comparison:
        if c in df.columns and c not in vote_cols:
            vote_cols.append(c)

    cols = [c for c in base_cols + vote_cols if c in df.columns]
    result = df[cols].copy() if not df.empty else pd.DataFrame(columns=cols)

    # Rename vote columns to readable country names
    rename_map = {c: data.get_country_name(c) for c in vote_cols}
    result = result.rename(columns=rename_map)

    return dcc.send_data_frame(result.to_csv, "resolutions.csv", index=False)


@callback(
    [
        Output("filter-component-country-dropdown", "disabled"),
        Output("filter-component-country2-dropdown", "disabled"),
        Output("filter-component-preset-dropdown", "disabled"),
        Output("filter-component-subject-dropdown", "disabled"),
        Output("filter-component-country-dropdown", "style"),
        Output("filter-component-country2-dropdown", "style"),
        Output("filter-component-preset-dropdown", "style"),
        Output("filter-component-subject-dropdown", "style"),
    ],
    Input("country-view-tabs", "value"),
    prevent_initial_call=False,
)
def disable_filters_based_on_tab(selected_tab):
    """Disable filters based on selected tab:
    - Agreement Map: disable comparison dropdowns
    - Word Cloud: disable all country dropdowns (main, compare, quick select)
    """
    is_map_tab = selected_tab == "map"
    is_wordcloud_tab = selected_tab == "wordcloud"
    is_timeline_or_subject_tab = selected_tab in ["timeline", "subject"]
    
    # Base styles
    base_style = {
        "width": "100%",
        "fontSize": "14px",
    }
    
    # Disabled styles (grayed out)
    disabled_style = {
        **base_style,
        "opacity": "0.5",
        "cursor": "not-allowed",
        "backgroundColor": "#e9ecef",
    }
    
    if is_wordcloud_tab:
        # Word Cloud: disable all country dropdowns
        return True, True, True, False, disabled_style, disabled_style, disabled_style, base_style
    elif is_map_tab:
        # Agreement Map: disable only comparison dropdowns
        return False, True, True, False, base_style, disabled_style, disabled_style, base_style
    elif is_timeline_or_subject_tab:
        # Agreement Timeline / Alignment by Subject: disable subject dropdown
        return False, False, False, True, base_style, base_style, base_style, disabled_style
    else:
        # Other tabs: enable all
        return False, False, False, False, base_style, base_style, base_style, base_style


@callback(
    [
        Output("status-display", "children"),
        Output("moving-average-data", "data"),
        Output("moving-average-calc-time", "data"),
    ],
    Input("filter-component-filter-store", "data"),
)
def _calculate_data_wrapper(filter_store):
    """Wrapper that converts list to tuple for caching."""
    if not filter_store:
        return ("No country selected.", None, None)
    country1 = filter_store.get("country1_alpha3")
    country2: List[str] | str = filter_store.get("country2")

    # normalize country2 to tuple (hashable)
    if country2 is None:
        return ("No comparison country selected.", None, None)
    selected_tuple = tuple(country2) if isinstance(country2, list) else (country2,)

    return _calculate_data_uncached(country1, selected_tuple, 350)


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
