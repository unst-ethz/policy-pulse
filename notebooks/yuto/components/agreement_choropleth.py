from dash import dcc, Input, Output, callback, html
import plotly.graph_objects as go
import pandas as pd
import sys, os
import plotly.express as px

# Add Janic's datastream module to search path
sys.path.append(
    os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "janic")
    )
)
from unDataStream import DataRepository, ResolutionQueryEngine


def register_callbacks():

    @callback(
        [
            Output("agreement-choropleth", "figure"),
            Output("agreement-choropleth-status", "children"),
        ],
        [
            Input("country1-dropdown", "value"),
        ],
    )
    def generate_chart(country1):

        repo = DataRepository(
            config_path=os.path.normpath(
                os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    "..",
                    "..",
                    "janic",
                    "config",
                    "data_sources.yaml",
                )
            )
        )
        query_engine = ResolutionQueryEngine(repo=repo)

        data = query_engine.query_agreement_between_countries(country1, average=True)

        # Transpose and remove the first two rows which are for the selected
        # country etc
        data = data.T[2:]
        data = data.reset_index()
        data.columns = ["three_letter_country", "agreement"]
        # Make sure the agreement column is numeric, so we can apply the
        # continuous color scale
        data[["agreement"]] = data[["agreement"]].apply(pd.to_numeric)

        fig = px.choropleth(
            data,
            color="agreement",
            color_continuous_scale=px.colors.sequential.RdBu,
            range_color=[0, 1],
            locations="three_letter_country",
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
                        f"Processed {len(data[["agreement"]])} data points.",
                    ]
                ),
            ]
        )

        return fig, status_msg


layout = (
    html.Div(
        [
            html.Div(id="agreement-choropleth-status"),
            dcc.Loading(
                children=[
                    dcc.Graph(id="agreement-choropleth", style={"height": "600px"})
                ],
                type="cube",
                color="#3498db",
            ),
        ],
        style={"padding": "0 20px"},
    ),
)
