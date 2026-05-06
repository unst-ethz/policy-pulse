from dash import Input, Output, State, callback, clientside_callback, html, dcc, ctx, no_update
import feffery_antd_components as fac
import pandas as pd
import urllib.parse
from pathlib import Path

from .. import data

_joining_dates = pd.read_csv(
    Path(__file__).resolve().parent.parent / "assets" / "joining_dates.csv"
)

prefix = "filter-component"
ids = {
    "year_range": f"{prefix}-year-range",
    "era_preset": f"{prefix}-era-preset",
    "era_prev_btn": f"{prefix}-era-prev-btn",
    "era_next_btn": f"{prefix}-era-next-btn",
    "subject_dropdown": f"{prefix}-subject-dropdown",
    "country": f"{prefix}-country-dropdown",
    "country2": f"{prefix}-country2-dropdown",
    "preset": f"{prefix}-preset-dropdown",
    "reset_btn": f"{prefix}-reset-btn",
    "filter_store": f"{prefix}-filter-store",
    "data_store": f"{prefix}-data-store",
    "location": f"{prefix}-location",
    "keyword_search": f"{prefix}-keyword-search",
    "clear_country2_btn": f"{prefix}-clear-country2-btn",
    "clear_subjects_btn": f"{prefix}-clear-subjects-btn",
    "country_filter_mode": f"{prefix}-country-filter-mode",
}

# Country group presets for quick comparison selection
# Groups are based on official UN bodies and regional groupings
COUNTRY_PRESETS = {
    "p5": {
        "label": "P5 – Security Council Permanent Members",
        "countries": ["USA", "GBR", "FRA", "RUS", "CHN"],
    },
    "africa_group": {
        "label": "African Group (Representative)",
        "countries": ["NGA", "ZAF", "EGY", "KEN", "ETH"],
    },
    "asia_pacific": {
        "label": "Asia-Pacific Group (Representative)",
        "countries": ["IND", "JPN", "IDN", "BGD", "PAK"],
    },
    "grulac": {
        "label": "GRULAC – Latin America & Caribbean",
        "countries": ["BRA", "MEX", "ARG", "COL", "CHL"],
    },
    "sids": {
        "label": "Small Island Developing States (SIDS)",
        "countries": ["MDV", "FJI", "JAM", "TTO", "VUT"],
    },
}

# Era presets for quick year range selection
# Periods are anchored to UN institutional milestones, not geopolitical blocs
ERA_PRESETS = {
    "un_founding": {
        "label": "1945 to 1954 (UN Founding Era)",
        "start": 1945,
        "end": 1954,
    },
    "decolonization": {
        "label": "1955 to 1974 (Decolonization Era)",
        "start": 1955,
        "end": 1974,
    },
    "nieo_period": {
        "label": "1974 to 1991 (North–South Dialogue)",
        "start": 1974,
        "end": 1991,
    },
    "post_bipolarity": {
        "label": "1992 to 2000 (Post-Bipolarity Era)",
        "start": 1992,
        "end": 2000,
    },
    "mdg_era": {
        "label": "2001 to 2015 (Millennium Development Goals)",
        "start": 2001,
        "end": 2015,
    },
    "sdg_era": {
        "label": "2016 to present (Sustainable Development Goals)",
        "start": 2016,
        "end": None,  # will use latest year
    },
    "until_1991": {
        "label": "All years until 1991",
        "start": 1945,
        "end": 1991,
    },
    "since_1992": {
        "label": "All years since 1992",
        "start": 1992,
        "end": None,  # will use latest year
    },
}

# Ordered sequence of the six institutional eras for ◀ ▶ navigation
# Excludes the two cross-cutting presets (until_1991, since_1992)
ERA_SEQUENCE = [
    "un_founding", "decolonization", "nieo_period",
    "post_bipolarity", "mdg_era", "sdg_era",
]

_TABS_WITHOUT_COUNTRY_FILTER = {"wordcloud"}
_TABS_WITH_KEYWORD = {"resolution_list", "wordcloud"}


def get_default_filter_values():
    return {
        "start_year": data.get_earliest_year(),
        "end_year": data.get_latest_year(),
        "era_preset": None,
        "subject_ids": None,
        "country1_alpha3": None,
        "country2": [],
        "preset": None,
        "keyword": "",
        "country_filter_mode": "voted",
    }


def parse_query_list_value(raw_value):
    if isinstance(raw_value, list):
        return raw_value
    else:
        return [raw_value] if raw_value else []


def parse_page_query_params(page_query_params):
    parsed_filters = {}

    start_year = page_query_params.get("start_year", "")
    if start_year != "":
        try:
            parsed_filters["start_year"] = int(start_year)
        except (TypeError, ValueError):
            pass

    end_year = page_query_params.get("end_year", "")
    if end_year != "":
        try:
            parsed_filters["end_year"] = int(end_year)
        except (TypeError, ValueError):
            pass

    country1 = page_query_params.get("country1_alpha3", "")
    if country1 != "":
        parsed_filters["country1_alpha3"] = country1

    keyword = page_query_params.get("keyword", "")
    if keyword != "":
        parsed_filters["keyword"] = keyword

    parsed_filters["country2"] = parse_query_list_value(
        page_query_params.get("country2", "")
    )
    subject_ids = parse_query_list_value(page_query_params.get("subject_ids", ""))
    parsed_filters["subject_ids"] = subject_ids if subject_ids else None

    return parsed_filters


def register_callbacks():

    # Callback: Step through eras with ◀ ▶ buttons
    @callback(
        Output(ids["era_preset"], "value", allow_duplicate=True),
        Input(ids["era_prev_btn"], "n_clicks"),
        Input(ids["era_next_btn"], "n_clicks"),
        State(ids["era_preset"], "value"),
        prevent_initial_call=True,
    )
    def step_era(prev_clicks, next_clicks, current_era):
        if current_era in ERA_SEQUENCE:
            idx = ERA_SEQUENCE.index(current_era)
            new_idx = max(0, idx - 1) if ctx.triggered_id == ids["era_prev_btn"] else min(len(ERA_SEQUENCE) - 1, idx + 1)
        else:
            new_idx = 0 if ctx.triggered_id == ids["era_next_btn"] else len(ERA_SEQUENCE) - 1
        return ERA_SEQUENCE[new_idx]

    # Callback: Disable ◀ / ▶ at the ends of ERA_SEQUENCE
    @callback(
        Output(ids["era_prev_btn"], "disabled", allow_duplicate=True),
        Output(ids["era_next_btn"], "disabled", allow_duplicate=True),
        Input(ids["era_preset"], "value"),
        prevent_initial_call='initial_duplicate',
    )
    def update_era_nav_state(current_era):
        if current_era not in ERA_SEQUENCE:
            return False, False
        idx = ERA_SEQUENCE.index(current_era)
        return idx == 0, idx == len(ERA_SEQUENCE) - 1

    # Callback: Apply era preset to year range slider
    @callback(
        Output(ids["year_range"], "value", allow_duplicate=True),
        Input(ids["era_preset"], "value"),
        prevent_initial_call=True,
    )
    def apply_era_preset(preset_key):
        if not preset_key or preset_key not in ERA_PRESETS:
            return no_update
        era = ERA_PRESETS[preset_key]
        end_year = era["end"] or get_default_filter_values()["end_year"]
        return [era["start"], end_year]

    # Callback: Apply preset to comparison countries
    @callback(
        Output(ids["country2"], "value"),
        Input(ids["preset"], "value"),
        prevent_initial_call=True,
    )
    def apply_preset(preset_key):
        if not preset_key or preset_key not in COUNTRY_PRESETS:
            return []
        return COUNTRY_PRESETS[preset_key]["countries"]

    # Callback: Clear comparison countries
    @callback(
        Output(ids["country2"], "value", allow_duplicate=True),
        Output(ids["preset"], "value", allow_duplicate=True),
        Input(ids["clear_country2_btn"], "n_clicks"),
        prevent_initial_call=True,
    )
    def clear_comparison(n_clicks):
        return [], None

    # Callback: Clear subjects
    @callback(
        Output(ids["subject_dropdown"], "value", allow_duplicate=True),
        Input(ids["clear_subjects_btn"], "n_clicks"),
        prevent_initial_call=True,
    )
    def clear_subjects(n_clicks):
        return None

    # Callback: Reset all filters except main country
    @callback(
        Output(ids["year_range"], "value"),
        Output(ids["era_preset"], "value"),
        Output(ids["subject_dropdown"], "value"),
        Output(ids["country"], "value"),
        Output(ids["country2"], "value", allow_duplicate=True),
        Output(ids["preset"], "value", allow_duplicate=True),
        Output(ids["keyword_search"], "value", allow_duplicate=True),
        Output(ids["country_filter_mode"], "value"),
        Input(ids["reset_btn"], "n_clicks"),
        prevent_initial_call=True,
    )
    def reset_filters(n_clicks):
        default_filters = get_default_filter_values()
        return (
            [default_filters["start_year"], default_filters["end_year"]],
            default_filters["era_preset"],
            default_filters["subject_ids"],
            default_filters["country1_alpha3"],
            default_filters["country2"],
            default_filters["preset"],
            default_filters["keyword"],
            default_filters["country_filter_mode"],
        )

    # Callback: Update filter store and print current selections when any filter changes
    @callback(
        Output(ids["filter_store"], "data"),
        Output(ids["location"], "search"),
        Input(ids["year_range"], "value"),
        Input(ids["subject_dropdown"], "value"),
        Input(ids["country"], "value"),
        Input(ids["country2"], "value"),
        Input(ids["keyword_search"], "value"),
        Input(ids["country_filter_mode"], "value"),
        prevent_initial_call=False,
    )
    def update_filter_store(
        year_range, subject_ids, country_iso3, country2, keyword, country_filter_mode
    ):
        """
        Register callbacks for the filter component.
        - Filter state management
        - Print current selections when filters change
        - Query data when filters change
        """
        if isinstance(year_range, (list, tuple)) and len(year_range) == 2:
            start_year, end_year = year_range
        else:
            start_year, end_year = None, None
        start_year = data.get_earliest_year() if start_year is None else start_year
        end_year = data.get_latest_year() if end_year is None else end_year

        # Strip the synthetic root node — selecting "All Subjects" means no filter
        if isinstance(subject_ids, list):
            subject_ids = [s for s in subject_ids if s != "__all_subjects__"] or None

        # Normalize country2 to a list
        if isinstance(country2, str) and country2:
            country2_list = [country2]
        elif isinstance(country2, list):
            country2_list = country2
        else:
            country2_list = []

        effective_country1 = country_iso3
        effective_country2 = country2_list

        # If no main country is selected, promote the first comparison country
        if not effective_country1 and effective_country2:
            effective_country1 = effective_country2[0]
            effective_country2 = effective_country2[1:]

        filter_data = {
            "start_date": f"{start_year}-01-01" if start_year else None,
            "end_date": f"{end_year}-12-31" if end_year else None,
            "start_year": start_year,
            "end_year": end_year,
            "subject_ids": subject_ids,
            "country1_alpha3": effective_country1,
            "country2": effective_country2 or None,
            "keyword": keyword.strip() if keyword and keyword.strip() else None,
            "country_filter_mode": country_filter_mode or "voted",
        }

        # Remove all None values for cleaner URL and easier parsing
        url_filters = {}
        for k, v in filter_data.items():
            if v is None or v == []:
                continue
            if k in ("start_date", "end_date"):
                continue
            if k == "country_filter_mode" and v == "voted":
                continue  # omit default from URL
            # Strip synthetic UI-only sentinel values from URL
            if k == "subject_ids" and isinstance(v, list):
                v = [s for s in v if not s.startswith("__")]
                if not v:
                    continue
            url_filters[k] = v

        # Print current selections
        print("\n" + "=" * 50)
        print("Current Filter Selections:")
        print(f"  Year Range: {start_year} to {end_year}")
        print(f"  Subjects: {subject_ids if subject_ids else 'None'}")
        print(f"  Country: {country_iso3 if country_iso3 else 'None'}")
        print("=" * 50 + "\n")

        return filter_data, f"?{urllib.parse.urlencode(url_filters, doseq=True)}"

    # Callback: Query data when filters change
    # Tabs where country1 is disabled or highlight-only — participation filter must not apply


    @callback(
        Output(ids["data_store"], "data"),
        Input(ids["filter_store"], "data"),
        Input("country-view-tabs", "value"),
        prevent_initial_call=False,
    )
    def query_data_on_filter_change(filter_data, active_tab):
        """Query data based on current filter selections."""
        try:
            # Convert years to inclusive date range (Jan 1 of start year to Dec 31 of end year)
            start_date = filter_data.get("start_date") if filter_data else None
            end_date = filter_data.get("end_date") if filter_data else None
            subject_ids = filter_data.get("subject_ids") if filter_data else None
            country = filter_data.get("country1_alpha3") if filter_data else None

            # Query resolutions using the query engine
            df = data.query_engine.query_resolutions(
                start_date=start_date,
                end_date=end_date,
                subject_ids=subject_ids,
                include_descendants=True,
            )

            # Apply country filter only on tabs where it is meaningful
            mode = (filter_data.get("country_filter_mode") or "voted") if filter_data else "voted"
            if country and country in df.columns and active_tab not in _TABS_WITHOUT_COUNTRY_FILTER:
                if mode == "voted":
                    vote_cleaned = df[country].astype(str).str.strip().str.upper()
                    has_voted = vote_cleaned.isin(["Y", "N", "A"])
                    df = df[has_voted]
                elif mode == "member":
                    rows = _joining_dates[_joining_dates["country"] == country]
                    if not rows.empty:
                        min_date = pd.to_datetime(rows["min_date"].min())
                        max_date = pd.to_datetime(rows["max_date"].max())
                        df["date"] = pd.to_datetime(df["date"])
                        df = df[(df["date"] >= min_date) & (df["date"] <= max_date)]
                    # TODO: multi-period membership (suspended + readmitted countries)
                # "none": no filter applied

            keyword = filter_data.get("keyword") if filter_data else None
            if keyword and keyword.strip() and active_tab in _TABS_WITH_KEYWORD and not df.empty:
                from .wordcloud_interactive import get_keyword_matched_ids
                matched_ids = get_keyword_matched_ids(df, keyword)
                df = df[df["undl_id"].isin(matched_ids)]

            # Build column list: base columns + undl_link + vote columns when countries selected
            base_cols = ["undl_id", "resolution", "session", "date", "title", "consensus_score"]
            if "undl_link" in df.columns:
                base_cols.append("undl_link")

            # country2_raw = filter_data.get("country2")
            # comparison = []
            # if isinstance(country2_raw, list):
            #     comparison = country2_raw
            # elif isinstance(country2_raw, str) and country2_raw:
            #     comparison = [country2_raw]
            # vote_cols = []
            # if country and country in df.columns:
            #     vote_cols.append(country)
            # for c2 in comparison[:5]:
            #     if c2 in df.columns:
            #         vote_cols.append(c2)
            # cols = [c for c in base_cols + vote_cols if c in df.columns]

            # if not cols:
            cols = base_cols

            # Convert to JSON for storage
            result_df = df[cols].copy() if not df.empty else pd.DataFrame()

            print(f"\n✅ Queried {len(result_df)} resolutions")
            if country:
                print(f"   Filtered by country: {country}")

            return result_df.to_json(date_format="iso", orient="split")

        except Exception as e:
            print(f"\n❌ Error querying data: {e}")
            import traceback

            traceback.print_exc()
            return None


def layout(page_query_params: dict[str, str] | None = None):
    initial_filters = get_default_filter_values()
    earliest_year, latest_year = (
        initial_filters["start_year"],
        initial_filters["end_year"],
    )

    # If any filters are defined in the URL, those get priority over defaults.
    # Parse query values to expected component types.
    if page_query_params:
        initial_filters.update(parse_page_query_params(page_query_params))

    return html.Div(
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
            dcc.Download(id="download-resolutions-csv"),
            # Header row: title + buttons
            html.Div(
                [
                    html.H2(
                        "Filter the Data",
                        style={"color": "#212529", "margin": "0"},
                    ),
                    html.Div(
                        [
                            html.Button(
                                "Download CSV",
                                id="download-btn",
                                n_clicks=0,
                                style={
                                    "backgroundColor": "#1a73e8",
                                    "color": "white",
                                    "border": "none",
                                    "borderRadius": "4px",
                                    "padding": "6px 14px",
                                    "fontSize": "13px",
                                    "cursor": "pointer",
                                    "fontFamily": "inherit",
                                    "fontWeight": "600",
                                },
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
                        style={"display": "flex", "gap": "8px"},
                    ),
                ],
                style={
                    "display": "flex",
                    "justifyContent": "space-between",
                    "alignItems": "center",
                    "marginBottom": "10px",
                },
            ),
            html.P(
                "Use the controls below to narrow down the resolutions by keyword, country, year range, or subject area. "
                "Select a main country to enable voting agreement analysis across the tabs.",
                style={"color": "#7f8c8d", "marginBottom": "20px"}
            ),
            # Filters container
            html.Div(
                [
                    # ROW 0: Keyword Search
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Label(
                                        [
                                            html.Span(
                                                "🔍", style={"marginRight": "5px"}
                                            ),
                                            "Keyword Search",
                                        ],
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
                            dcc.Input(
                                id=ids["keyword_search"],
                                type="text",
                                placeholder="e.g. human rights, climate (comma-separated, press Enter)",
                                debounce=True,
                                value=initial_filters["keyword"],
                                style={
                                    "width": "100%",
                                    "fontSize": "14px",
                                    "padding": "8px 10px",
                                    "border": "1px solid #ced4da",
                                    "borderRadius": "4px",
                                    "fontFamily": "inherit",
                                    "boxSizing": "border-box",
                                },
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
                    # ROW 1: Main Country
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Label(
                                        [
                                            html.Span(
                                                "🌍", style={"marginRight": "5px"}
                                            ),
                                            "Main Country",
                                        ],
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
                                        "label": data.get_country_display_name(country),
                                        "value": country,
                                        "search": data.get_country_search_terms(country),
                                    }
                                    for country in data.available_countries
                                ],
                                value=initial_filters["country1_alpha3"],
                                placeholder="Select a country...",
                                clearable=True,
                                style={
                                    "width": "100%",
                                    "fontSize": "14px",
                                },
                                searchable=True,
                            ),
                            html.Div(
                                [
                                    html.Label(
                                        "Filter resolutions:",
                                        style={
                                            "fontSize": "13px",
                                            "color": "#6c757d",
                                            "marginRight": "10px",
                                        },
                                    ),
                                    dcc.RadioItems(
                                        id=ids["country_filter_mode"],
                                        options=[
                                            {"label": " Voted on resolution", "value": "voted"},
                                            {"label": " Was UN member", "value": "member"},
                                            {"label": " No filter", "value": "none"},
                                        ],
                                        value=initial_filters["country_filter_mode"],
                                        inline=True,
                                        style={"fontSize": "13px", "color": "#555"},
                                    ),
                                ],
                                style={
                                    "display": "flex",
                                    "alignItems": "center",
                                    "marginTop": "10px",
                                },
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
                                            html.Label(
                                                [
                                                    html.Span(
                                                        "🔄",
                                                        style={"marginRight": "5px"},
                                                    ),
                                                    "Compare with",
                                                ],
                                                style={
                                                    "fontWeight": "600",
                                                    "color": "#495057",
                                                    "fontSize": "15px",
                                                    "marginBottom": "0",
                                                    "display": "inline-block",
                                                },
                                            ),
                                            html.Button(
                                                "↺ Clear All",
                                                id=ids["clear_country2_btn"],
                                                n_clicks=0,
                                                style={
                                                    "marginLeft": "12px",
                                                    "padding": "4px 12px",
                                                    "fontSize": "13px",
                                                    "fontWeight": "400",
                                                    "color": "#495057",
                                                    "backgroundColor": "transparent",
                                                    "border": "1px solid #adb5bd",
                                                    "borderRadius": "4px",
                                                    "cursor": "pointer",
                                                    "fontFamily": "inherit",
                                                    "verticalAlign": "middle",
                                                },
                                            ),
                                        ],
                                        style={
                                            "marginBottom": "8px",
                                            "display": "flex",
                                            "alignItems": "center",
                                        },
                                    ),
                                    fac.AntdTreeSelect(
                                        id=ids["country2"],
                                        treeData=data.REGION_TREE_DATA,
                                        treeCheckable=True,
                                        showCheckedStrategy="show-child",
                                        treeNodeFilterProp="title",
                                        placeholder="Search or browse countries...",
                                        allowClear=True,
                                        multiple=True,
                                        treeDefaultExpandAll=False,
                                        treeLine=True,
                                        style={"width": "100%"},
                                        locale="en-us",
                                        value=initial_filters["country2"],
                                    ),
                                ],
                                style={"flex": "1"},
                            ),
                            # Preset dropdown
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Label(
                                                [
                                                    html.Span(
                                                        "⚡",
                                                        style={"marginRight": "5px"},
                                                    ),
                                                    "Quick Select",
                                                ],
                                                style={
                                                    "fontWeight": "600",
                                                    "color": "#495057",
                                                    "fontSize": "15px",
                                                    "marginBottom": "0",
                                                    "display": "inline-block",
                                                },
                                            ),
                                            # Invisible spacer matching Clear All button
                                            # to keep this label row the same height as
                                            # the Compare with row (which has the button).
                                            html.Button(
                                                "↺ Clear All",
                                                n_clicks=0,
                                                tabIndex=-1,
                                                style={
                                                    "marginLeft": "12px",
                                                    "padding": "4px 12px",
                                                    "fontSize": "13px",
                                                    "fontWeight": "400",
                                                    "border": "1px solid transparent",
                                                    "borderRadius": "4px",
                                                    "fontFamily": "inherit",
                                                    "verticalAlign": "middle",
                                                    "visibility": "hidden",
                                                    "pointerEvents": "none",
                                                },
                                            ),
                                        ],
                                        style={
                                            "marginBottom": "8px",
                                            "display": "flex",
                                            "alignItems": "center",
                                        },
                                    ),
                                    fac.AntdSelect(
                                        id=ids["preset"],
                                        options=[
                                            {"label": p["label"], "value": k}
                                            for k, p in COUNTRY_PRESETS.items()
                                        ],
                                        value=initial_filters["preset"],
                                        placeholder="Choose a group...",
                                        allowClear=True,
                                        style={"width": "100%"},
                                        locale="en-us",
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
                                            html.Label(
                                                [
                                                    html.Span(
                                                        "🗓️",
                                                        style={"marginRight": "5px"},
                                                    ),
                                                    "Year Range",
                                                ],
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
                                        dcc.RangeSlider(
                                            id=ids["year_range"],
                                            min=earliest_year,
                                            max=latest_year,
                                            step=1,
                                            value=[
                                                initial_filters["start_year"],
                                                initial_filters["end_year"],
                                            ],
                                            marks=None,
                                            tooltip=None,
                                            allowCross=False,
                                        ),
                                        style={},
                                    ),
                                    html.Div(
                                        [
                                            html.Button(
                                                "◀",
                                                id=ids["era_prev_btn"],
                                                n_clicks=0,
                                                style={
                                                    "fontSize": "14px",
                                                    "color": "#495057",
                                                    "backgroundColor": "transparent",
                                                    "border": "1px solid #adb5bd",
                                                    "borderRadius": "4px",
                                                    "cursor": "pointer",
                                                    "fontFamily": "inherit",
                                                },
                                            ),
                                            dcc.Dropdown(
                                                id=ids["era_preset"],
                                                options=[
                                                    {"label": e["label"], "value": k}
                                                    for k, e in ERA_PRESETS.items()
                                                ],
                                                value=initial_filters["era_preset"],
                                                placeholder="Quick select era...",
                                                clearable=True,
                                                style={
                                                    "fontSize": "14px",
                                                },
                                            ),
                                            html.Button(
                                                "▶",
                                                id=ids["era_next_btn"],
                                                n_clicks=0,
                                                style={
                                                    "fontSize": "14px",
                                                    "color": "#495057",
                                                    "backgroundColor": "transparent",
                                                    "border": "1px solid #adb5bd",
                                                    "borderRadius": "4px",
                                                    "cursor": "pointer",
                                                    "fontFamily": "inherit",
                                                },
                                            ),
                                        ],
                                        style={
                                            "display": "grid",
                                            "gridTemplateColumns": "28px 1fr 28px",
                                            "gap": "6px",
                                            "marginTop": "12px",
                                        },
                                    ),
                                ],
                                style={
                                    "flex": "0 0 30%",
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
                                            html.Label(
                                                [
                                                    html.Span(
                                                        "📚",
                                                        style={"marginRight": "5px"},
                                                    ),
                                                    "Subjects",
                                                ],
                                                style={
                                                    "fontWeight": "600",
                                                    "color": "#495057",
                                                    "fontSize": "15px",
                                                    "marginBottom": "0",
                                                    "display": "inline-block",
                                                },
                                            ),
                                            html.Button(
                                                "↺ Clear All",
                                                id=ids["clear_subjects_btn"],
                                                n_clicks=0,
                                                style={
                                                    "marginLeft": "12px",
                                                    "padding": "4px 12px",
                                                    "fontSize": "13px",
                                                    "fontWeight": "400",
                                                    "color": "#495057",
                                                    "backgroundColor": "transparent",
                                                    "border": "1px solid #adb5bd",
                                                    "borderRadius": "4px",
                                                    "cursor": "pointer",
                                                    "fontFamily": "inherit",
                                                    "verticalAlign": "middle",
                                                },
                                            ),
                                        ],
                                        style={
                                            "marginBottom": "8px",
                                            "display": "flex",
                                            "alignItems": "center",
                                        },
                                    ),
                                    fac.AntdTreeSelect(
                                        id=ids["subject_dropdown"],
                                        treeData=data.SUBJECT_TREE_DATA,
                                        treeCheckable=True,
                                        showCheckedStrategy="show-parent",
                                        treeNodeFilterProp="title",
                                        placeholder="Search or browse subjects...",
                                        allowClear=True,
                                        multiple=True,
                                        treeDefaultExpandAll=False,
                                        treeLine=True,
                                        style={"width": "100%"},
                                        locale="en-us",
                                        value=initial_filters["subject_ids"],
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
    )
