from dash import dcc, Input, Output, callback, html
import plotly.graph_objects as go
import pandas as pd
from io import StringIO


def register_callbacks():

    @callback(
        [
            Output("agreement-chart", "figure"),
            Output("agreement-chart-status", "children"),
        ],
        [
            Input("country1-dropdown", "value"),
            Input("country2-dropdown", "value"),
            Input("timespan-dropdown", "value"),
            Input("moving-average-data", "data"),
        ],
    )
    def generate_chart(country1, country2, time_span, moving_average_data):
        # Create figure
        fig = go.Figure()

        if moving_average_data is None:
            return fig, html.Div(
                [
                    html.Div(
                        [
                            html.I(
                                className="fas fa-check-circle",
                                style={"color": "red", "marginRight": "5px"},
                            ),
                            html.Strong("Chart could not be updated."),
                        ]
                    ),
                ]
            )

        data = pd.read_json(StringIO(moving_average_data))

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

        fig.update_layout(
            title=f"GA Voting Agreement: {country1} vs {country2}<br>"
            + f"<sub>{total_count:,} votes • {missing_count:,} missing ({missing_count/total_count*100:.1f}%)</sub>",
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
                        f"Processed {total_count:,} data points.",
                    ]
                ),
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
