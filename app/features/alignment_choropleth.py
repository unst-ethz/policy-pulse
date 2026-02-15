from dash import dcc, Input, Output, callback, html
import plotly.graph_objects as go
import pandas as pd
import plotly.express as px

from .country_coordinates import get_country_longitude
from ..data import get_earliest_data_date, get_latest_data_date




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
            Input("filter-component-data-store", "data"),
            Input("filter-component-filter-store", "data"),
            Input("alignment-choropleth-timeline", "value"),
        ],
    )
    def generate_chart(filtered_data, filter_store, year: tuple[int, int]):
        if not filtered_data or not filter_store:
            return go.Figure(), None, year, None, ""
        all_resolutions = pd.read_json(filtered_data, orient="split")
        if all_resolutions.empty:
            # no resolutions matcing the filter
            status_msg = html.Div(
                [
                    html.Div([html.Strong("No resolutions to plot")]),
                ]
            )

            return None, None, year, None, status_msg

        # Find time range available resolutions for this filtered data
        earliest_year = get_earliest_data_date().year
        latest_year = get_latest_data_date().year

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
        country1 = filter_store.get("country1_alpha3")
        if country1 is None:
            # primary country is not yet selected
            status_msg = html.Div(
                [
                    html.Div([html.Strong("Please select a primary country")]),
                ]
            )

            return None, earliest_year, year, latest_year, status_msg

        alignment_data = query_engine.query_agreement_between_countries(
            country1,
            resolution_ids=(resolutions_in_year["undl_id"].tolist()),
            average=True,
        )

        # Transpose and remove the first two rows which are for the selected
        # country etc
        alignment_data = alignment_data.T[2:]
        alignment_data = alignment_data.reset_index()

        if alignment_data.empty:
            # Simulate neutral 0.5 value for all countries
            alignment_data = pd.DataFrame(
                {
                    "three_letter_country": ["NAN"],
                    "alignment": [0.5],
                }
            )

        alignment_data.columns = ["three_letter_country", "alignment"]
        # Make sure the alignment column is numeric, so we can apply the
        # continuous color scale
        alignment_data[["alignment"]] = alignment_data[["alignment"]].apply(pd.to_numeric)

        fig = px.choropleth(
            alignment_data,
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
        fig.update_geos(projection_rotation_lon=-country_longitude)

        # Status message
        status_msg = html.Div(
            [
                html.Div(
                    [
                        html.Strong("Chart Updated Successfully! "),
                        f"Processed {len(alignment_data[['alignment']])} data points.",
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
                min=0, 
                max=1, 
                step=1, 
                value=[0, 0], 
                id="alignment-choropleth-timeline",
                marks={
                    1946: '1946',
                    1950: '1950',
                    1960: '1960',
                    1970: '1970',
                    1980: '1980',
                    1990: '1990',
                    2000: '2000',
                    2010: '2010',
                    2020: '2020',
                    2025: '2025',
                }
            ),
        ],
    ),
)
