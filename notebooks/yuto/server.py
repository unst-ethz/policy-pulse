import json
import dash
from dash import callback, dcc, html, Input, Output
import pandas as pd
from functools import lru_cache
import time
import sys, os

# Add Janic's datastream module to search path
sys.path.append(
    os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "janic")
    )
)
from unDataStream import DataRepository, ResolutionQueryEngine

from .components import alignment_choropleth
from .components import alignment_graph
from .components import navbar
from .components import breadcrumb
from .components import wordcloud_viz


class DashMovingAverageApp:
    """
    Complete Dash application for interactive moving average visualization.
    Features true lazy loading, caching, and professional UI.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        time_span: int = 365,
        start_date: str | None = None,
        end_date: str | None = None,
        cache_size: int = 100,
    ):
        self.df = df
        self.time_span = time_span
        self.start_date = start_date
        self.end_date = end_date

        # Get available countries
        self.available_countries = [
            col
            for col in df.columns
            if col
            not in [
                "undl_id",
                "date",
                "session",
                "resolution",
                "draft",
                "committee_report",
                "meeting",
                "title",
                "agenda_title",
                "subjects",
                "total_yes",
                "total_no",
                "total_abstentions",
                "total_non_voting",
                "total_ms",
                "undl_link",
            ]
        ]

        # Set up caching for expensive calculations
        callback(
            [
                Output("status-display", "children"),
                Output("moving-average-data", "data"),
                Output("moving-average-calc-time", "data"),
            ],
            [
                Input("country1-dropdown", "value"),
                Input("country2-dropdown", "value"),
                Input("timespan-dropdown", "value"),
            ],
        )(lru_cache(maxsize=cache_size)(self._calculate_data_uncached))

        # Initialize Dash app
        self.app = dash.Dash(__name__)
        self.setup_layout()

    def _calculate_data_uncached(self, country1: str, country2: str, time_span: int):
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
            countries = [country1, country2]

            # Calculate alignment
            def calc_alignment(row: pd.Series):
                vote_mapping = {"Y": 1, "A": 0, "N": -1}
                if row[country1] in vote_mapping and row[country2] in vote_mapping:
                    diff = abs(
                        vote_mapping[row[country1]] - vote_mapping[row[country2]]
                    )
                    return 1 - (diff / 2)
                return float("nan")

            # Process data
            df_subset = self.df[["date", country1, country2]].copy()
            df_subset["alignment"] = df_subset.apply(calc_alignment, axis=1)
            df_subset["date"] = pd.to_datetime(df_subset["date"])
            df_subset = df_subset.sort_values("date").reset_index(drop=True)

            # Apply date filters
            if self.start_date:
                df_subset = df_subset[
                    df_subset["date"] >= pd.to_datetime(self.start_date)
                ]
            if self.end_date:
                df_subset = df_subset[
                    df_subset["date"] <= pd.to_datetime(self.end_date)
                ]

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

    def setup_layout(self):
        """Set up the Dash app layout."""
        self.app.layout = html.Div(
            [
                # Store intermediate results from callbacks.
                # Graphs can read from this by using Input("moving-average-data", "data")
                dcc.Store(id="moving-average-data"),
                dcc.Store(id="moving-average-calc-time"),
                dcc.Store(id="chosen-country-1"),
                # Nav bar
                *navbar.layout(self.available_countries),
                html.Div(
                    className="container",
                    children=[
                        *breadcrumb.layout,
                        # Status and cache info
                        html.Div(
                            id="status-display",
                        ),
                        *alignment_choropleth.layout,
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
                                                for country in self.available_countries
                                            ],
                                            value=(
                                                self.available_countries[1]
                                                if len(self.available_countries) > 1
                                                else self.available_countries[0]
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
                                            value=self.time_span,
                                            clearable=False,
                                            style={"marginBottom": "15px"},
                                        ),
                                    ],
                                    style={"width": "30%", "display": "inline-block"},
                                ),
                            ]
                        ),
                        *alignment_graph.layout,
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

    def run(self, debug: bool = True, port: int = 8050, host: str = "127.0.0.1"):
        """Run the Dash app."""
        print(f"🚀 Starting Dash app...")
        print(f"📊 Countries available: {len(self.available_countries)}")
        print(f"🌐 Open your browser to: http://{host}:{port}")

        self.app.run(debug=debug, port=port, host=host)


# Easy setup function
def create_dash_app(
    df: pd.DataFrame,
    time_span: int = 365,
    start_date: str | None = None,
    end_date: str | None = None,
    cache_size: int = 100,
):
    """
    Create a Dash app for interactive moving average visualization.

    Parameters:
    - df: Your DataFrame with voting data
    - time_span: Default moving average window
    - start_date, end_date: Date range filters
    - cache_size: Number of country pairs to cache

    Returns:
    - DashMovingAverageApp instance
    """
    return DashMovingAverageApp(df, time_span, start_date, end_date, cache_size)


# Usage examples and setup instructions
print("\n🌐 Production Deployment:")
print("- Set debug=False for production")
print("- Configure proper host/port for your environment")


def fetch_UN_data(dir_path: str | None = None):
    """
    Fetches and processes United Nations General Assembly and Security Council voting data.

    This function retrieves voting data from either local files or the UN Digital Library,
    and transforms the data into two formats: original and pivoted (transformed).

    Parameters:
    -----------
    dir_path : str, optional
        Path to directory where data should be read from or saved to.
        If None, data will be fetched from the UN Digital Library and not saved locally.

    Returns:
    --------
    tuple
        A tuple containing four DataFrames:
        - df_ga: Original GA voting data
        - df_ga_transformed: Pivoted GA voting data with countries as columns
        - df_sc: Original SC voting data
        - df_sc_transformed: Pivoted SC voting data with countries as columns

    Notes:
    ------
    - Currently, the Security Council data does not include veto information explicitly.
    - The filenames and URLs are hardcoded for the 2025 voting sessions. Must be updated when they change.
    """

    df_ga = None
    df_sc = None

    if dir_path:
        try:
            df_ga = pd.read_csv(f"{dir_path}/2025_9_19_ga_voting.csv")
            df_sc = pd.read_csv(f"{dir_path}/2025_7_21_sc_voting.csv")
        except FileNotFoundError:
            print("Not all data found locally. Fetching from UN Digital Library...")
    if df_ga is None or df_sc is None:
        ga_url = "https://digitallibrary.un.org/record/4060887/files/2025_9_19_ga_voting.csv?ln=en"
        sc_url = "https://digitallibrary.un.org/record/4055387/files/2025_7_21_sc_voting.csv?ln=en"

        try:
            df_ga = pd.read_csv(ga_url)
            df_sc = pd.read_csv(sc_url)

            # Save data locally if dir_path is provided
            if dir_path:
                # Check if directory exists, create it if it doesn't
                if not os.path.exists(dir_path):
                    os.makedirs(dir_path)
                    print(f"Created directory: {dir_path}")

                df_ga.to_csv(f"{dir_path}/2025_9_19_ga_voting.csv", index=False)
                df_sc.to_csv(f"{dir_path}/2025_7_21_sc_voting.csv", index=False)
        except Exception as e:
            print(
                "Error fetching data from UN Digital Library. The dataset might has been updated. Check the date in the URL."
            )
            print(f"Error: {e}")
            return None, None, None, None

    # Transform ga data
    ga_index_columns = [
        "undl_id",
        "date",
        "session",
        "resolution",
        "draft",
        "committee_report",
        "meeting",
        "title",
        "agenda_title",
        "subjects",
        "total_yes",
        "total_no",
        "total_abstentions",
        "total_non_voting",
        "total_ms",
        "undl_link",
    ]
    df_ga_transformed = df_ga.pivot(
        index=ga_index_columns, columns="ms_code", values="ms_vote"
    ).reset_index()
    df_ga_transformed.columns.name = None

    # Transform sc data
    sc_index_columns = [
        "undl_id",
        "date",
        "resolution",
        "draft",
        "meeting",
        "description",
        "agenda",
        "subjects",
        "modality",
        "total_yes",
        "total_no",
        "total_abstentions",
        "total_non_voting",
        "total_ms",
        "undl_link",
    ]
    df_sc_transformed = df_sc.pivot(
        index=sc_index_columns, columns="ms_code", values="ms_vote"
    ).reset_index()
    df_sc_transformed.columns.name = None

    return df_ga, df_ga_transformed, df_sc, df_sc_transformed


df_ga, df_ga_transformed, df_sc, df_sc_transformed = fetch_UN_data(dir_path="../data")


if df_ga_transformed is None:
    print("Failed to retrieve UN data")
    exit(1)

app = create_dash_app(df_ga_transformed, time_span=365, cache_size=100)

print("Initialising data repository...")
repo = DataRepository(
    config_path=os.path.normpath(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "janic",
            "config",
            "data_sources.yaml",
        )
    )
)
query_engine = ResolutionQueryEngine(repo=repo)
print("Data repository initialised!")

navbar.register_callbacks()
breadcrumb.register_callbacks()
alignment_choropleth.register_callbacks(query_engine)
alignment_graph.register_callbacks()
wordcloud_viz.register_callbacks()

app.run(debug=True, port=8050)
