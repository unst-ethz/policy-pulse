from dash import dcc, Input, Output, callback, html
import plotly.graph_objects as go
import pandas as pd
import plotly.express as px

from .country_coordinates import get_country_longitude




def register_callbacks(query_engine):

    @callback(
        [
            Output("alignment-choropleth", "figure"),
            Output("alignment-choropleth-timeline", "min"),
            Output("alignment-choropleth-timeline", "value"),
            Output("alignment-choropleth-timeline", "max"),
            Output("alignment-choropleth-status", "children"),
        ],
        [
            Input("country1-iso-alpha3", "data"),
            Input("alignment-choropleth-timeline", "value"),
        ],
    )
    def generate_chart(country1, year: tuple[int, int]):
        # Find time range of all resolutions available: this should be precalculated
        # honestly
        all_resolutions = query_engine.query_resolutions()
        earliest_year = pd.to_datetime(
            all_resolutions["date"], errors="coerce"
        ).dt.year.min()
        latest_year = pd.to_datetime(
            all_resolutions["date"], errors="coerce"
        ).dt.year.max()

        if year[0] == 0 and year[1] == 0:
            year = (earliest_year, latest_year)

        resolutions_in_year = all_resolutions[
            (
                pd.to_datetime(all_resolutions["date"], errors="coerce").dt.year
                >= int(year[0])
            )
            & (
                pd.to_datetime(all_resolutions["date"], errors="coerce").dt.year
                <= int(year[1])
            )
        ]
        data = query_engine.query_agreement_between_countries(
            country1,
            resolution_ids=(resolutions_in_year["undl_id"].tolist()),
            average=True,
        )

        # Transpose and remove the first two rows which are for the selected
        # country etc
        data = data.T[2:]
        data = data.reset_index()

        if data.empty:
            # Simulate neutral 0.5 value for all countries
            data = pd.DataFrame(
                {
                    "three_letter_country": ["NAN"],
                    "alignment": [0.5],
                }
            )

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
            projection="robinson",
        )

        fig.add_trace(
            go.Choropleth(
                locations=[country1],
                z=[1],  # dummy value
                colorscale=[[0, "green"], [1, "green"]],  # solid green color
                showscale=False,
                hovertemplate=f"<b>{country1}</b> (Selected)<extra></extra>",
                marker_line_color="black",
                marker_line_width=2,
            )
        )

        # Center the map on the selected country's longitude (x-axis)
        # Keep y-axis at equator (latitude = 0)
        country_longitude = get_country_longitude(country1)
        fig.update_geos(
            projection_rotation_lon=-country_longitude
        )


        # Status message
        status_msg = html.Div(
            [
                html.Div(
                    [
                        html.Strong("Chart Updated Successfully! "),
                        f"Processed {len(data[['alignment']])} data points.",
                    ]
                ),
            ]
        )

        return fig, earliest_year, year, latest_year, status_msg


layout = (
    html.Div(
        [
            html.Div(id="alignment-choropleth-status"),
            dcc.Loading(
                children=[
                    dcc.Graph(
                        id="alignment-choropleth",
                        style={"height": "600px", "width": "100%"},
                    ),
                ],
                type="circle",
                color="#3498db",
            ),
            dcc.RangeSlider(
                min=0, max=1, step=1, value=[0, 0], id="alignment-choropleth-timeline"
            ),
        ],
    ),
)
