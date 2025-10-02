import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import pandas as pd
from functools import lru_cache
import time

# Global variables (in production, you'd load this properly)
df_global = None  # Your dataframe goes here
available_countries = []


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
        self.calculate_data = lru_cache(maxsize=cache_size)(
            self._calculate_data_uncached
        )

        # Initialize Dash app
        self.app = dash.Dash(__name__)
        self.setup_layout()
        self.setup_callbacks()

    def _calculate_data_uncached(self, country1: str, country2: str, time_span: str):
        """Calculate moving average data for country pair (cached)."""
        print(f"🔄 Calculating {country1} vs {country2} (span: {time_span})")
        start_time = time.time()

        try:
            countries = [country1, country2]

            # Calculate agreement
            def calc_agreement(row: pd.Series):
                vote_mapping = {"Y": 1, "A": 0, "N": -1}
                if row[country1] in vote_mapping and row[country2] in vote_mapping:
                    diff = abs(
                        vote_mapping[row[country1]] - vote_mapping[row[country2]]
                    )
                    return 1 - (diff / 2)
                return float("nan")

            # Process data
            df_subset = self.df[["date", country1, country2]].copy()
            df_subset["agreement"] = df_subset.apply(calc_agreement, axis=1)
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
                df_subset["agreement"].rolling(window=time_span, min_periods=1).mean()
            )
            df_subset["ema"] = (
                df_subset["agreement"].ewm(span=time_span, adjust=False).mean()
            )
            df_subset["cma"] = df_subset["agreement"].expanding(min_periods=1).mean()

            calc_time = time.time() - start_time
            print(f"✅ Calculated in {calc_time:.2f}s ({len(df_subset):,} points)")

            return df_subset, calc_time

        except Exception as e:
            print(f"❌ Calculation error: {e}")
            raise

    def setup_layout(self):
        """Set up the Dash app layout."""
        self.app.layout = html.Div(
            [
                # Header
                html.Div(
                    [
                        html.H1(
                            "🌍 GA Voting Agreement Analysis",
                            style={
                                "textAlign": "center",
                                "color": "#2c3e50",
                                "marginBottom": "10px",
                            },
                        ),
                        html.P(
                            f"Interactive analysis of {len(self.available_countries)} countries • True lazy loading with caching",
                            style={
                                "textAlign": "center",
                                "color": "#7f8c8d",
                                "fontSize": "16px",
                            },
                        ),
                    ],
                    style={
                        "padding": "20px",
                        "backgroundColor": "#ecf0f1",
                        "marginBottom": "20px",
                    },
                ),
                # Controls
                html.Div(
                    [
                        html.Div(
                            [
                                html.Label(
                                    "Country 1:",
                                    style={"fontWeight": "bold", "marginBottom": "5px"},
                                ),
                                dcc.Dropdown(
                                    id="country1-dropdown",
                                    options=[
                                        {"label": country, "value": country}
                                        for country in self.available_countries
                                    ],
                                    value=self.available_countries[0],
                                    clearable=False,
                                    style={"marginBottom": "15px"},
                                ),
                            ],
                            style={
                                "width": "30%",
                                "display": "inline-block",
                                "paddingRight": "20px",
                            },
                        ),
                        html.Div(
                            [
                                html.Label(
                                    "Country 2:",
                                    style={"fontWeight": "bold", "marginBottom": "5px"},
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
                                "paddingRight": "20px",
                            },
                        ),
                        html.Div(
                            [
                                html.Label(
                                    "Time Span (days):",
                                    style={"fontWeight": "bold", "marginBottom": "5px"},
                                ),
                                dcc.Dropdown(
                                    id="timespan-dropdown",
                                    options=[
                                        {"label": "30 days", "value": 30},
                                        {"label": "90 days", "value": 90},
                                        {"label": "180 days", "value": 180},
                                        {"label": "365 days", "value": 365},
                                        {"label": "730 days (2 years)", "value": 730},
                                    ],
                                    value=self.time_span,
                                    clearable=False,
                                    style={"marginBottom": "15px"},
                                ),
                            ],
                            style={"width": "30%", "display": "inline-block"},
                        ),
                    ],
                    style={"padding": "0 20px", "marginBottom": "20px"},
                ),
                # Status and cache info
                html.Div(
                    [
                        html.Div(
                            id="status-display",
                            style={
                                "padding": "10px",
                                "backgroundColor": "#d5dbdb",
                                "border": "1px solid #bdc3c7",
                                "borderRadius": "5px",
                            },
                        )
                    ],
                    style={"padding": "0 20px", "marginBottom": "20px"},
                ),
                # Loading indicator and chart
                html.Div(
                    [
                        dcc.Loading(
                            id="loading-chart",
                            children=[
                                dcc.Graph(
                                    id="agreement-chart", style={"height": "600px"}
                                )
                            ],
                            type="cube",
                            color="#3498db",
                        )
                    ],
                    style={"padding": "0 20px"},
                ),
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
                                "Agreement ranges from 0 (complete disagreement) to 1 (perfect agreement).",
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

    def setup_callbacks(self):
        """Set up Dash callbacks for interactivity."""

        @self.app.callback(
            [Output("agreement-chart", "figure"), Output("status-display", "children")],
            [
                Input("country1-dropdown", "value"),
                Input("country2-dropdown", "value"),
                Input("timespan-dropdown", "value"),
            ],
        )
        def update_chart(country1: str, country2: str, time_span: float | None):
            """Update chart when countries or time span changes."""

            # Validation
            if country1 == country2:
                error_msg = html.Div(
                    [
                        html.I(
                            className="fas fa-exclamation-triangle",
                            style={"color": "orange", "marginRight": "5px"},
                        ),
                        html.Strong("Warning: "),
                        "Same country selected for both dropdowns. Please choose different countries.",
                    ],
                    style={"color": "orange"},
                )

                return (
                    go.Figure().add_annotation(
                        text="Please select different countries",
                        xref="paper",
                        yref="paper",
                        x=0.5,
                        y=0.5,
                        showarrow=False,
                        font_size=20,
                    ),
                    error_msg,
                )

            try:
                # Get cache info before calculation
                cache_info_before = self.calculate_data.cache_info()

                # Calculate data (uses cache if available)
                data, calc_time = self.calculate_data(country1, country2, time_span)

                # Get cache info after calculation
                cache_info_after = self.calculate_data.cache_info()
                was_cached = cache_info_before.hits < cache_info_after.hits

                # Create figure
                fig = go.Figure()

                # Add traces
                fig.add_trace(
                    go.Scatter(
                        x=data["date"],
                        y=data["sma"],
                        mode="lines",
                        name=f"{time_span}-Day SMA",
                        line=dict(color="#3498db", width=3),
                    )
                )

                fig.add_trace(
                    go.Scatter(
                        x=data["date"],
                        y=data["ema"],
                        mode="lines",
                        name=f"{time_span}-Day EMA",
                        line=dict(color="#e67e22", width=3),
                    )
                )

                fig.add_trace(
                    go.Scatter(
                        x=data["date"],
                        y=data["cma"],
                        mode="lines",
                        name="Cumulative MA",
                        line=dict(color="#27ae60", width=3),
                    )
                )

                # Add missing values
                missing_mask = pd.isna(data["agreement"])
                if missing_mask.any():
                    fig.add_trace(
                        go.Scatter(
                            x=data["date"][missing_mask],
                            y=[0.5] * missing_mask.sum(),
                            mode="markers",
                            name="Missing Data",
                            marker=dict(color="gray", symbol="x", size=8),
                            opacity=0.7,
                        )
                    )

                # Update layout
                missing_count = missing_mask.sum()
                total_count = len(data)
                cache_status = "📋 Cached" if was_cached else "🔄 Calculated"

                fig.update_layout(
                    title=f"GA Voting Agreement: {country1} vs {country2}<br>"
                    + f"<sub>{total_count:,} votes • {missing_count:,} missing ({missing_count/total_count*100:.1f}%) • {cache_status}</sub>",
                    xaxis_title="Date",
                    yaxis_title="Agreement Level",
                    yaxis=dict(range=[-0.05, 1.05]),
                    template="plotly_white",
                    hovermode="x unified",
                    legend=dict(
                        orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
                    ),
                )

                # Status message
                status_msg = html.Div(
                    [
                        html.Div(
                            [
                                html.I(
                                    className="fas fa-check-circle",
                                    style={"color": "green", "marginRight": "5px"},
                                ),
                                html.Strong("Chart Updated Successfully! "),
                                f"Processed {total_count:,} data points in {calc_time:.2f}s ",
                                f"({'from cache' if was_cached else 'newly calculated'})",
                            ]
                        ),
                        html.Div(
                            [
                                html.I(
                                    className="fas fa-database",
                                    style={"color": "blue", "marginRight": "5px"},
                                ),
                                f"Cache: {cache_info_after.currsize}/{cache_info_after.maxsize} pairs stored • ",
                                f"Hits: {cache_info_after.hits} • Misses: {cache_info_after.misses} • ",
                                f"Hit Rate: {cache_info_after.hits/(cache_info_after.hits + cache_info_after.misses)*100:.1f}%",
                            ],
                            style={
                                "fontSize": "12px",
                                "color": "#7f8c8d",
                                "marginTop": "5px",
                            },
                        ),
                    ]
                )

                return fig, status_msg

            except Exception as e:
                error_msg = html.Div(
                    [
                        html.I(
                            className="fas fa-exclamation-circle",
                            style={"color": "red", "marginRight": "5px"},
                        ),
                        html.Strong("Error: "),
                        f"Failed to generate chart: {str(e)}",
                    ],
                    style={"color": "red"},
                )

                return (
                    go.Figure().add_annotation(
                        text=f"Error: {str(e)}",
                        xref="paper",
                        yref="paper",
                        x=0.5,
                        y=0.5,
                        showarrow=False,
                        font_size=16,
                    ),
                    error_msg,
                )

    def run(self, debug: bool = True, port: int = 8050, host: str = "127.0.0.1"):
        """Run the Dash app."""
        print(f"🚀 Starting Dash app...")
        print(f"📊 Countries available: {len(self.available_countries)}")
        print(f"🌐 Open your browser to: http://{host}:{port}")
        print(f"🔄 True lazy loading enabled with caching")

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
print("🎯 Dash Lazy Loading Solution Ready!")
print("\n📋 Setup Instructions:")
print("1. Install Dash: pip install dash")
print("2. Load your data: df = pd.read_csv('your_data.csv')")
print("3. Create app: app = create_dash_app(df)")
print("4. Run app: app.run()")
print("5. Open browser to http://127.0.0.1:8050")

print("\n🚀 Quick Start Example:")
print(
    """
import pandas as pd
from your_module import create_dash_app

# Load your data
df = pd.read_csv('ga_voting_data.csv')

# Create and run the app
app = create_dash_app(df, time_span=365, cache_size=100)
app.run(debug=True, port=8050)
"""
)

print("\n✅ Features:")
print("- True lazy loading (calculates only when requested)")
print("- Intelligent caching (LRU eviction)")
print("- Real-time status updates")
print("- Professional web interface")
print("- Easy deployment to production")
print("- Handles hundreds of countries efficiently")
print("- No Jupyter dependency issues")

print("\n🌐 Production Deployment:")
print("- Deploy to Heroku, AWS, or any cloud platform")
print("- Set debug=False for production")
print("- Configure proper host/port for your environment")

import os
import pandas as pd


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
            df_ga = pd.read_csv(f"{dir_path}/2025_7_23_ga_voting.csv")
            df_sc = pd.read_csv(f"{dir_path}/2025_7_21_sc_voting.csv")
        except FileNotFoundError:
            print("Not all data found locally. Fetching from UN Digital Library...")
    if df_ga is None or df_sc is None:
        ga_url = "https://digitallibrary.un.org/record/4060887/files/2025_7_23_ga_voting.csv?ln=en"
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

                df_ga.to_csv(f"{dir_path}/2025_7_23_ga_voting.csv", index=False)
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


df_ga, df_ga_transformed, df_sc, df_sc_transformed = fetch_UN_data(
    dir_path="C:\\Users\\janic\\OneDrive\\Desktop\\ETH\\UN Projekt\\data"
)

app = create_dash_app(df_ga_transformed, time_span=365, cache_size=100)
app.run(debug=True, port=8050)
