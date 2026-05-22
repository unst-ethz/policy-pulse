import functools
import time
from typing import List
from urllib.parse import urlencode

import numpy as np
import pandas as pd
from dash import (
    Input,
    Output,
    State,
    callback,
    clientside_callback,
    get_relative_path,
    html,
    dcc,
    register_page,
)

from .. import data
from ..features import (
    agreement_by_subject,
    agreement_choropleth,
    agreement_graph,
    filters,
    multilateral_scatter,
    resolution_list,
    wordcloud_interactive
)

# Tab IDs — defined once here so layout values and callback comparisons stay in sync
_TAB_RESOLUTION_LIST = "resolution_list"
_TAB_MAP            = "map"
_TAB_TIMELINE       = "timeline"
_TAB_SUBJECT        = "subject"
_TAB_MULTILATERAL   = "multilateral"
_TAB_WORDCLOUD      = "wordcloud"

# Country filter mode (CFM) options — controls whether the resolution list is
# restricted to resolutions where the selected country voted / was a member / not at all.
_CFM_OPTIONS = [
    {"label": " Voted on resolution", "value": "voted"},
    {"label": " Was UN member",       "value": "member"},
    {"label": " No filter",           "value": "none"},
]
_CFM_OPTIONS_DISABLED = [{**o, "disabled": True} for o in _CFM_OPTIONS]

# Base and greyed-out styles for all dropdown controls (country, subject, preset, era)
_DROPDOWN_STYLE     = {"width": "100%", "fontSize": "14px"}
_DROPDOWN_STYLE_OFF = {**_DROPDOWN_STYLE, "opacity": "0.5", "cursor": "not-allowed", "backgroundColor": "#e9ecef"}

# Base and greyed-out styles for the keyword text input
_INPUT_STYLE     = {
    "width": "100%", "fontSize": "14px", "padding": "8px 10px",
    "border": "1px solid #ced4da", "borderRadius": "4px",
    "fontFamily": "inherit", "boxSizing": "border-box",
}
_INPUT_STYLE_OFF = {**_INPUT_STYLE, "opacity": "0.5", "cursor": "not-allowed", "backgroundColor": "#e9ecef"}

# Base and greyed-out styles for the "Filter by membership dates" button
_BTN_STYLE     = {
    "fontSize": "13px", "fontFamily": "inherit", "border": "1px solid #adb5bd",
    "borderRadius": "4px", "padding": "4px 10px", "whiteSpace": "nowrap",
    "cursor": "pointer", "backgroundColor": "transparent", "color": "#495057",
}
_BTN_STYLE_OFF = {**_BTN_STYLE, "cursor": "not-allowed", "backgroundColor": "#e9ecef", "color": "#adb5bd"}


def title(countr1_alpha3=None):
    return (
        "Policy Pulse Analysis" + f" (Country 1: {countr1_alpha3})"
        if countr1_alpha3
        else ""
    )


register_page(__name__, path_template="/trends")


def layout(country1_alpha3: str | None = None, **other_keyword_arguments):
    """Render the full trends page, passing URL query params into the filter store."""

    available = [c for c in data.available_countries if c != country1_alpha3]
    return html.Div(
        [
            dcc.Store(id="country1-iso-alpha3", data=country1_alpha3),
            dcc.Store(id="country1-localised-name"),
            html.H1(
                [
                    "Explore GA Votes Yourself",
                ]
            ),
            filters.layout(other_keyword_arguments),
            html.Div(
                id="resolution-count-banner",
                style={
                    "padding": "5px 4px",
                    "fontSize": "13px",
                    "color": "#7f8c8d",
                },
            ),
            html.Div(
                id="status-display",
            ),
            # Tab Navigation
            dcc.Tabs(
                id="country-view-tabs",
                value=_TAB_RESOLUTION_LIST,
                style={"marginTop": "1rem"},
                children=[
                    # TAB 1: Resolution List
                    dcc.Tab(
                        label="Resolutions",
                        value=_TAB_RESOLUTION_LIST,
                        children=[
                            html.Div(
                                [
                                    html.H2("Preview of resolutions"),
                                    html.P(
                                        "Browse all UN General Assembly resolutions matching the current filters. "
                                        "Use the sort and search controls below to explore the list, and click any resolution to view its full details.",
                                        style={
                                            "color": "#7f8c8d",
                                            "marginBottom": "20px",
                                        },
                                    ),
                                    *resolution_list.layout,
                                ],
                                className="tab-content",
                            )
                        ],
                    ),
                    # TAB 2: Agreement Map
                    dcc.Tab(
                        label="Agreement Map",
                        value=_TAB_MAP,
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
                        value=_TAB_TIMELINE,
                        id="tab-agreement-timeline",
                        disabled=False,
                        children=[
                            html.Div(
                                id="tab-timeline-content",
                                className="tab-content",
                            )
                        ],
                    ),
                    # TAB 4: Agreement by Subject
                    dcc.Tab(
                        label="Agreement by Subject",
                        value=_TAB_SUBJECT,
                        id="tab-agreement-subject",
                        disabled=False,
                        children=[
                            html.Div(
                                id="tab-subject-content",
                                className="tab-content",
                            )
                        ],
                    ),
                    # TAB 5: Multilateral Alignment
                    dcc.Tab(
                        label="Multilateral Overview",
                        value=_TAB_MULTILATERAL,
                        children=[
                            html.Div(
                                [
                                    html.H2("Multilateral Agreement Overview"),
                                    html.P(
                                        "Explore how closely each country's voting aligns with the broader UN membership "
                                        "across the selected resolutions.",
                                        style={
                                            "color": "#7f8c8d",
                                            "marginBottom": "20px",
                                        },
                                    ),
                                    *multilateral_scatter.layout,
                                ],
                                className="tab-content",
                            )
                        ],
                    ),
                    # TAB 6: Word Cloud
                    dcc.Tab(
                        label="Word Cloud",
                        value=_TAB_WORDCLOUD,
                        children=[
                            html.Div(
                                [
                                    html.H2("Resolution Title Word Cloud"),
                                    html.P(
                                        "Explore the most searchable terms in the currently filtered resolutions. "
                                        "Word size and hover counts are computed from the active filters.",
                                        style={
                                            "color": "#7f8c8d",
                                            "marginBottom": "12px",
                                        },
                                    ),
                                    html.Div(
                                        [
                                            html.P(
                                                "How to use this panel:",
                                                style={
                                                    "color": "#374151",
                                                    "marginBottom": "8px",
                                                    "fontWeight": "600",
                                                },
                                            ),
                                            html.Ul(
                                                [
                                                    html.Li(
                                                        "Choose a tab by term source: Subjects uses the thesaurus subject system (same basis as the Subjects filter); "
                                                        "Default uses Hugging Face Voicelab/vlt5-base-keywords; "
                                                        "Geopolitical/Thematic/Action use OpenAI gpt-4.1-mini extraction."
                                                    ),
                                                    html.Li(
                                                        "Hover a phrase to see how many currently filtered resolutions can be found with it."
                                                    ),
                                                    html.Li(
                                                        "Click a phrase to jump to Accepted Resolutions. In Subjects, click updates the Subjects filter; "
                                                        "in other tabs, click updates Keyword Search."
                                                    ),
                                                    html.Li(
                                                        "Word-cloud-added keywords are exact phrases (quoted) and are appended with '&' (AND)."
                                                    ),
                                                ],
                                                style={
                                                    "color": "#6b7280",
                                                    "marginTop": "0",
                                                    "paddingLeft": "20px",
                                                    "marginBottom": "20px",
                                                },
                                            ),
                                        ]
                                    ),
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

# Client-side callback: resolves country 1 ISO alpha3 to a localised display name.
clientside_callback(
    """
    function localise_iso_country(filter_store) {
        if (!filter_store) return null;
        const iso_three_digit = filter_store.country1_alpha3;

        // Localize the country code (convert ISO3 to full name) 
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


@callback(
    Output("profile-page-link", "href"),
    Output("profile-page-link", "style"),
    Input("filter-component-filter-store", "data"),
)
def update_country_profile_link(filter_store):
    """Build the URL for the Country Profile page (/profile) from the current filter state.

    Encodes country1, date range and comparison countries as query parameters.
    Also styles the link button — greyed out when no primary country is selected.
    """
    _base = {
        "border": "none", "borderRadius": "4px", "padding": "6px 14px",
        "fontSize": "13px", "fontWeight": "600", "textDecoration": "none",
        "display": "inline-block",
    }
    _disabled = {**_base, "backgroundColor": "#adb5bd", "color": "white",
                 "cursor": "not-allowed", "opacity": "0.6"}
    _enabled  = {**_base, "backgroundColor": "#1a73e8", "color": "white",
                 "cursor": "pointer", "opacity": "1"}

    country1 = (filter_store or {}).get("country1_alpha3")
    if not country1:
        return "#", _disabled

    params = {"country1": country1}
    start_date = (filter_store or {}).get("start_date")
    end_date   = (filter_store or {}).get("end_date")
    if start_date:
        params["start_date"] = start_date
    if end_date:
        params["end_date"] = end_date

    country2 = (filter_store or {}).get("country2")
    if country2:
        if isinstance(country2, list):
            params["country2"] = ",".join(country2)
        else:
            params["country2"] = country2

    return get_relative_path("/profile?" + urlencode(params)), _enabled


# Wire up callbacks for each feature module. Must be called at import time
# so Dash can discover them before the server starts.
filters.register_callbacks()
resolution_list.register_callbacks()
agreement_choropleth.register_callbacks(data.query_engine)
agreement_graph.register_callbacks()
agreement_by_subject.register_callbacks(data.query_engine)
multilateral_scatter.register_callbacks(data.query_engine)
wordcloud_interactive.register_callbacks()


@callback(
    [
        Output("tab-agreement-map", "style"),
        Output("tab-agreement-timeline", "style"),
        Output("tab-agreement-subject", "style"),
        Output("tab-map-content", "children"),
        Output("tab-timeline-content", "children"),
        Output("tab-subject-content", "children"),
    ],
    Input("filter-component-filter-store", "data"),
    prevent_initial_call=False,
)
def grey_out_tabs_without_country(filter_store):
    """Grey out the Map tab until a main country is selected, and the
    Timeline and Subject tabs until a comparison country is also selected."""
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
    comparison_disabled = no_main_country or (
        country2 is None or country2 == "" or country2 == country1
    )
    tab_normal_style = {}
    tab_grayed_style = {
        "opacity": 0.45,
        "filter": "grayscale(100%)",
        "transition": "opacity 120ms ease",
    }
    # Agreement Map requires main country only.
    # Agreement Timeline / Agreement by Subject require both main + compare country.
    map_tab_style = tab_grayed_style if no_main_country else tab_normal_style
    timeline_tab_style = tab_grayed_style if comparison_disabled else tab_normal_style
    subject_tab_style = tab_grayed_style if comparison_disabled else tab_normal_style

    # Placeholder content for disabled tabs
    placeholder_no_country1 = html.Div(
        [
            html.P(
                "Please select a Main Country from the filters above to view this analysis.",
                style={
                    "color": "#7f8c8d",
                    "fontSize": "16px",
                    "textAlign": "center",
                    "padding": "40px",
                    "fontStyle": "italic",
                },
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
                },
            )
        ]
    )

    # Content when enabled - wrap in a list to match Dash's expected format
    if no_main_country:
        map_content = [placeholder_no_country1]
    else:
        country1_name = data.get_country_name(country1)
        map_content = [
            html.H2("Global Agreement Map"),
            html.P(
                f"Shows how closely each country's UN General Assembly voting agrees with {country1_name}. "
                "Agreement scores range from 0 (always opposed) to 1 (always agreeing).",
                style={"color": "#7f8c8d", "marginBottom": "20px"},
            ),
            *agreement_choropleth.layout,
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
            html.H2("Pairwise Agreement over Time"),
            html.P(
                f"Tracks how voting agreement between {country1_name} and the selected comparison countries has evolved across UN General Assembly sessions. "
                f"Each point represents the average agreement for one session. Only sessions with at least 3 shared votes are shown.",
                style={"color": "#7f8c8d", "marginBottom": "20px"},
            ),
            *agreement_graph.layout,
        ]
        subject_content = [
            html.H2("Agreement by UN Subject Area"),
            html.P(
                f"Compare voting agreement between {country1_name} and the selected comparison country across different UN subject areas. "
                "The agreement score ranges from 0 (complete disagreement) to 1 (complete agreement). "
                "Only subjects with at least 30 shared votes are shown.",
                style={"color": "#7f8c8d", "marginBottom": "20px"},
            ),
            *agreement_by_subject.layout,
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
    Output("resolution-count-banner", "children"),
    Input("filter-component-data-store", "data"),
    Input("country-view-tabs", "value"),
)
def update_resolution_count(data_store, active_tab):
    """Show how many resolutions match the current filters, or a note for the timeline tab."""
    if active_tab == _TAB_TIMELINE:
        return html.Span(
            "Timeline uses full session history — year range, subject and country filters do not apply.",
            style={"fontStyle": "italic"},
        )
    if not data_store:
        return ""
    n = len(pd.read_json(data_store, orient="split"))
    return [html.Strong(f"{n:,}"), f" resolution{'s' if n != 1 else ''} match current filters"]


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
        return True, {
            **base_style,
            "backgroundColor": "transparent",
            "color": "#adb5bd",
            "border": "1px solid #adb5bd",
            "cursor": "not-allowed",
        }
    return False, {
        **base_style,
        "backgroundColor": "transparent",
        "color": "#1a73e8",
        "border": "1px solid #1a73e8",
        "cursor": "pointer",
    }


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
        matched_ids = wordcloud_interactive.get_keyword_matched_ids(df, keyword)
        df = df[df["undl_id"].isin(matched_ids)]

    base_cols = [
        "undl_id",
        "resolution",
        "date",
        "session",
        "title",
        "agenda_title",
        "subjects",
        "draft",
    ]
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


# ---------------------------------------------------------------------------
# Filter control enable/disable logic
# Constants and helper shared by the two callbacks below, which together
# grey out filter controls that are irrelevant for the active tab.
# ---------------------------------------------------------------------------
def _compute_filter_state(selected_tab: str, country1: str | None) -> dict:
    """Return boolean flags indicating which filter controls should be disabled."""
    no_country = not country1

    c1_off = c2_off = preset_off = subj_off = yr_era_off = cfm_off = False

    if selected_tab == _TAB_WORDCLOUD:
        c1_off = c2_off = preset_off = cfm_off = True  # no country context needed
    elif selected_tab == _TAB_MULTILATERAL:
        c2_off = preset_off = True                     # comparison country not used
    elif selected_tab == _TAB_MAP:
        c2_off = preset_off = cfm_off = True           # choropleth filters by pair internally
    elif selected_tab == _TAB_TIMELINE:
        subj_off = yr_era_off = cfm_off = True         # timeline ignores subject + date range
    elif selected_tab == _TAB_SUBJECT:
        subj_off = cfm_off = True                      # tab hardwires "both countries voted"

    if no_country:
        cfm_off = True

    return {
        "c1_off":         c1_off,
        "c2_off":         c2_off,
        "preset_off":     preset_off,
        "subj_off":       subj_off,
        "kw_off":         selected_tab not in (_TAB_RESOLUTION_LIST, _TAB_WORDCLOUD),
        "cfm_off":        cfm_off,
        "yr_era_off":     yr_era_off,
        "membership_off": no_country or c1_off or yr_era_off,
    }


@callback(
    Output("filter-component-country-dropdown",    "disabled"),
    Output("filter-component-country2-dropdown",   "disabled"),
    Output("filter-component-preset-dropdown",     "disabled"),
    Output("filter-component-subject-dropdown",    "disabled"),
    Output("filter-component-keyword-search",      "disabled"),
    Output("filter-component-country-filter-mode", "options"),
    Output("filter-component-year-range",          "disabled"),
    Output("filter-component-era-preset",          "disabled"),
    Output("filter-component-era-prev-btn",        "disabled", allow_duplicate=True),
    Output("filter-component-era-next-btn",        "disabled", allow_duplicate=True),
    Output("filter-component-membership-dates-btn","disabled"),
    Input("country-view-tabs", "value"),
    Input("filter-component-country-dropdown", "value"),
    prevent_initial_call='initial_duplicate',
)
def _set_filter_disabled_states(selected_tab, country1):
    s = _compute_filter_state(selected_tab, country1)
    return (
        s["c1_off"], s["c2_off"], s["preset_off"], s["subj_off"],
        s["kw_off"],
        _CFM_OPTIONS_DISABLED if s["cfm_off"] else _CFM_OPTIONS,
        s["yr_era_off"], s["yr_era_off"], s["yr_era_off"], s["yr_era_off"],
        s["membership_off"],
    )


@callback(
    Output("filter-component-country-dropdown",    "style"),
    Output("filter-component-country2-dropdown",   "style"),
    Output("filter-component-preset-dropdown",     "style"),
    Output("filter-component-subject-dropdown",    "style"),
    Output("filter-component-keyword-search",      "style"),
    Output("filter-component-era-preset",          "style"),
    Output("filter-component-membership-dates-btn","style"),
    Input("country-view-tabs", "value"),
    Input("filter-component-country-dropdown", "value"),
    prevent_initial_call='initial_duplicate',
)
def _set_filter_styles(selected_tab, country1):
    s = _compute_filter_state(selected_tab, country1)

    def _dd(off):
        return _DROPDOWN_STYLE_OFF if off else _DROPDOWN_STYLE

    return (
        _dd(s["c1_off"]),
        _dd(s["c2_off"]),
        _dd(s["preset_off"]),
        _dd(s["subj_off"]),
        _INPUT_STYLE_OFF if s["kw_off"]         else _INPUT_STYLE,
        _dd(s["yr_era_off"]),
        _BTN_STYLE_OFF   if s["membership_off"] else _BTN_STYLE,
    )


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
        return ("", None, None)
    selected_tuple = tuple(country2) if isinstance(country2, list) else (country2,)

    return _calculate_data_uncached(country1, selected_tuple)


@functools.lru_cache(maxsize=100)
def _calculate_data_uncached(country1: str | None, selected_tuple: tuple):
    """Calculate per-session agreement data for country pairs (cached)."""
    if country1 is None:
        return (None, None, None)

    selected = list(selected_tuple)
    selected = [c for c in selected if c != country1]
    if len(selected) == 0:
        return (None, None, None)

    print(f"🔄 Calculating session agreement: {country1} vs {selected}")
    start_time = time.time()

    try:
        vote_mapping = {"Y": 1, "A": 0, "N": -1}

        cols = ["date", "session", country1] + selected
        df = data.query_engine.query_resolutions()[cols].copy()

        # normalize and map votes to numeric
        vote_cols = [country1] + selected
        df[vote_cols] = (
            df[vote_cols].astype(str).apply(lambda s: s.str.strip().str.upper())
        )
        df[vote_cols] = df[vote_cols].replace(
            {"": pd.NA, "NAN": pd.NA, "NONE": pd.NA, "<NA>": pd.NA}
        )
        df[vote_cols] = df[vote_cols].replace(vote_mapping)
        df[vote_cols] = df[vote_cols].apply(pd.to_numeric, errors="coerce")

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.sort_values("date").reset_index(drop=True)

        # check that at least one overlapping vote exists
        mask_any = (~df[country1].isna()) & df[selected].notna().any(axis=1)
        if not mask_any.any():
            return (
                html.Div(
                    style={
                        "padding": "10px",
                        "marginBlock": "20px",
                        "backgroundColor": "#efe1bb",
                        "border": "1px solid #e8da6f",
                        "borderRadius": "8px",
                    },
                    children=(
                        f"There are no resolutions for which "
                        f"{data.get_country_name(country1)} and any of the "
                        f"selected comparison countries both voted. "
                        f"Some features may be unavailable."
                    ),
                ),
                None,
                None,
            )

        # compute per-resolution agreement (vectorized)
        diffs = df[selected].subtract(df[country1], axis=0).abs()
        agreement = 1.0 - (diffs / 2.0)

        for col in selected:
            both_voted = df[[country1, col]].notna().all(axis=1)
            agreement[col] = agreement[col].where(both_voted, np.nan)

        # build temporary df for session-level groupby
        tmp = pd.DataFrame({"session": df["session"], "date": df["date"]})
        for col in selected:
            tmp[f"agreement_{col}"] = agreement[col]

        # derive year per session (median date's year)
        session_year = tmp.groupby("session")["date"].median().dt.year
        session_year.name = "year"

        # aggregate per session per country
        out_parts = [session_year]
        for col in selected:
            agree_col = f"agreement_{col}"
            avg_col = f"avg_agreement_{col}"
            n_col = f"n_votes_{col}"

            grouped = tmp.groupby("session")[agree_col]
            avg_agreement = grouped.mean()
            avg_agreement.name = avg_col
            n_votes = grouped.apply(lambda x: x.notna().sum())
            n_votes.name = n_col

            # minimum 3 votes threshold
            avg_agreement = avg_agreement.where(n_votes >= 3, np.nan)

            out_parts.extend([avg_agreement, n_votes])

        out = pd.concat(out_parts, axis=1).reset_index()
        out = out.sort_values(["year", "session"]).reset_index(drop=True)

        # last-vote cutoff: find last session each country voted in
        for col in selected:
            avg_col = f"avg_agreement_{col}"
            n_col = f"n_votes_{col}"
            has_votes = out[avg_col].notna()
            if has_votes.any():
                last_idx = has_votes.values[::-1].argmax()
                last_session_pos = len(has_votes) - 1 - last_idx
                out.loc[out.index[last_session_pos + 1:], avg_col] = np.nan
                print(f"  📅 {col}: last active session = {out.loc[out.index[last_session_pos], 'session']}")

        calc_time = time.time() - start_time
        print(f"✅ Calculated in {calc_time:.2f}s ({len(out)} sessions)")

        return (None, out.to_json(), calc_time)

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
