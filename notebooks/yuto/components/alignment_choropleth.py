from dash import dcc, Input, Output, callback, html
import plotly.graph_objects as go
import pandas as pd
import plotly.express as px


def register_callbacks(query_engine):

    @callback(
        [
            Output("alignment-choropleth", "figure"),
            Output("alignment-choropleth-status", "children"),
        ],
        [
            Input("country1-dropdown", "value"),
        ],
    )
    def generate_chart(country1):
        data = query_engine.query_agreement_between_countries(country1, average=True)

        # Transpose and remove the first two rows which are for the selected
        # country etc
        data = data.T[2:]
        data = data.reset_index()
        data.columns = ["three_letter_country", "alignment"]
        # Make sure the alignment column is numeric, so we can apply the
        # continuous color scale
        data[["alignment"]] = data[["alignment"]].apply(pd.to_numeric)

        fig = px.choropleth(
            data,
            color="alignment",
            color_continuous_scale=px.colors.sequential.RdBu,
            range_color=[0, 1],
            locations="three_letter_country",
        )

        # Status message
        status_msg = html.Div(
            [
                html.Div(
                    [
                        html.Strong("Chart Updated Successfully! "),
                        f"Processed {len(data[["alignment"]])} data points.",
                    ]
                ),
            ]
        )

        return fig, status_msg


layout = (
    html.Div(
        [
            html.Div(id="alignment-choropleth-status"),
            dcc.Loading(
                children=[
                    dcc.Graph(id="alignment-choropleth", style={"height": "600px"})
                ],
                type="cube",
                color="#3498db",
            ),
        ],
    ),
)
