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
            Output("agreement-chart-note", "children"),
        ],
        [
            Input("filter-component-filter-store", "data"),
            Input("moving-average-data", "data"),
        ],
    )
    def generate_chart(filter_store, moving_average_data):
        # moving_average_data is JSON produced by to_json; parse it
        if moving_average_data is None or not filter_store:
            return go.Figure(), html.Div("No data"), ""

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
        country1_name = get_country_name(country1)
        status_msg = html.Div(
            f"Overlapping vote period: {start_str} to {end_str} ({total_points:,} data points).",
            style={"color": "#7f8c8d", "fontSize": "14px", "padding": "4px 0"},
        )

        note_msg = html.P([
            html.Strong("Note: "),
            f"The chart shows the moving average of the pairwise vote agreement between {country1_name} "
            "and the selected comparison countries over time. An agreement score of 1 means that "
            "two countries voted the same on all General Assembly (GA) resolutions within the moving window. "
            'A score of 0 means that two countries always voted in opposite ways ("yes" vs. "no"). '
            "The data only covers GA resolutions that were passed (accepted)."
        ], style={
            "maxWidth": "100%",
            "margin": "0 0 0 0",
            "paddingLeft": "2%",
            "paddingTop": "10px",
            "color": "#7f8c8d",
            "fontSize": "16px",
            "lineHeight": "1.6",
            "textAlign": "left",
            "borderTop": "1px solid #eee"
        })

        return fig, status_msg, note_msg


layout = [
    html.Div(
        [
            html.Div(id="agreement-chart-status"),
            dcc.Loading(
                children=[dcc.Graph(id="agreement-chart", style={"height": "600px"})],
                type="cube",
                color="#3498db",
            ),
            html.Div(id="agreement-chart-note"),
        ],
    )
]