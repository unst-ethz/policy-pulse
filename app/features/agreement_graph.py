from dash import dcc, Input, Output, callback, html
import plotly.graph_objects as go
import pandas as pd
from io import StringIO
from ..data import get_country_name


def register_callbacks():

    @callback(
        [
            Output("agreement-chart", "figure"),
            Output("agreement-chart-status", "children"),
        ],
        [
            Input("filter-component-filter-store", "data"),
            Input("moving-average-data", "data"),
        ],
    )
    def generate_chart(filter_store, moving_average_data):
        # moving_average_data is JSON produced by to_json; parse it
        if moving_average_data is None or not filter_store:
            return go.Figure(), html.Div("No data")

        country1 = filter_store.get("country1_alpha3")
        country2 = filter_store.get("country2")

        df = pd.read_json(StringIO(moving_average_data))
        df["date"] = pd.to_datetime(df["date"])

        # ensure country2 is a list
        selected = country2 if isinstance(country2, (list, tuple)) else [country2]

        fig = go.Figure()
        colors = ["blue", "orange", "green", "red", "purple", "brown", "pink", "gray", "olive", "cyan"]
        for i, c in enumerate(selected):
            sma_col = f"sma_{c}"
            align_col = f"agreement_{c}"
            if sma_col in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df["date"],
                        y=df[sma_col],
                        mode="lines",
                        name=f"{get_country_name(c)}",  # f"{c} ({country1}) SMA",
                        line=dict(color=colors[i % len(colors)]),
                    )
                )
            elif align_col in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df["date"],
                        y=df[align_col],
                        mode="lines",
                        name=f"{c} ({country1}) agreement",
                        line=dict(color=colors[i % len(colors)]),
                    )
                )

        fig.update_layout(
            title=f"Agreement: {get_country_name(country1)} vs {', '.join([get_country_name(c) for c in selected])}",
            xaxis_title="Date",
            yaxis_title="Agreement Score",
            yaxis=dict(range=[0, 1]),
            template="plotly_white",
        )
        # status message
        total_points = len(df)
        start_str = (
            df["date"].min().strftime("%Y-%m-%d")
            if not df["date"].isna().all()
            else "N/A"
        )
        end_str = (
            df["date"].max().strftime("%Y-%m-%d")
            if not df["date"].isna().all()
            else "N/A"
        )
        status_msg = html.Div(
            [
                html.Strong("Chart Updated Successfully. "),
                f"{total_points:,} points from {start_str} to {end_str}",
            ]
        )

        return fig, status_msg


layout = (
    html.Div(
        [
            html.Div(id="agreement-chart-status"),
            dcc.Loading(
                children=[dcc.Graph(id="agreement-chart", style={"height": "600px"})],
                type="cube",
                color="#3498db",
            ),
        ],
    ),
)
