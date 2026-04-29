from dash import dcc, Input, Output, callback, html
import plotly.graph_objects as go
import pandas as pd
import plotly.express as px

from .. import data

from .country_coordinates import get_country_longitude


def register_callbacks(query_engine):

    @callback(
        [
            Output("agreement-choropleth", "figure"),
            Output("agreement-choropleth-status", "children"),
            Output("agreement-choropleth-note", "children"),
        ],
        [
            Input("filter-component-data-store", "data"),
            Input("filter-component-filter-store", "data"),
        ],
    )
    def generate_chart(filtered_data, filter_store):
        if not filtered_data or not filter_store:
            return go.Figure(), "", ""
        all_resolutions = pd.read_json(filtered_data, orient="split")
        if all_resolutions.empty:
            status_msg = html.Div([html.Div([html.Strong("No resolutions to plot")])])
            return go.Figure(), status_msg, ""

        country1 = filter_store.get("country1_alpha3")
        if country1 is None:
            status_msg = html.Div(
                [html.Div([html.Strong("Please select a primary country")])]
            )
            return go.Figure(), status_msg, ""

        start_year = filter_store.get("start_year")
        end_year = filter_store.get("end_year")

        resolutions_in_year = all_resolutions.copy()
        if start_year or end_year:
            dates = pd.to_datetime(resolutions_in_year["date"], errors="coerce").dt.year
            if start_year:
                resolutions_in_year = resolutions_in_year[dates >= int(start_year)]
            if end_year:
                resolutions_in_year = resolutions_in_year[dates <= int(end_year)]

        agreement_data = query_engine.query_agreement_between_countries(
            country1,
            resolution_ids=(resolutions_in_year["undl_id"].tolist()),
            average=True,
        )

        # Transpose and remove the first two rows which are for the selected
        # country etc
        agreement_data = agreement_data.T[2:]
        agreement_data = agreement_data.reset_index()
        agreement_data.columns = ["three_letter_country", "agreement_raw"]

        # Make new column by applying data.get_country_name to three_letter_country
        agreement_data["Country"] = agreement_data["three_letter_country"].apply(
            data.get_country_display_name
        )
        agreement_data["Agreement"] = agreement_data["agreement_raw"].apply(
            lambda x: f"{x:.2f} with {data.get_country_display_name(country1)}"
            if pd.notna(x) else "No shared vote"
        )

        if agreement_data.empty:
            # Simulate neutral 0.5 value for all countries
            agreement_data = pd.DataFrame(
                {
                    "three_letter_country": ["NAN"],
                    "Country": ["No data for selected filters"],
                    "agreement_raw": [0.5],
                }
            )

        # Make sure the agreement column is numeric, so we can apply the
        # continuous color scale
        agreement_data[["agreement_raw"]] = agreement_data[["agreement_raw"]].apply(
            pd.to_numeric
        )

        # Separate countries with no shared votes (NaN agreement)
        no_shared_votes = agreement_data[agreement_data["agreement_raw"].isna()].copy()
        agreement_data = agreement_data[agreement_data["agreement_raw"].notna()]

        # Plot the choropleth world map
        # Note: Recent `plotly` versions actually seem to use an official
        # UN data source to generate simplified geometries for the world map.
        # See the description of this plotly PR:
        # https://github.com/plotly/plotly.js/pull/7393
        fig = px.choropleth(
            agreement_data,
            color="agreement_raw",
            color_continuous_scale=px.colors.diverging.RdYlBu,  # Red-Yellow-Blue
            range_color=[0, 1],
            locations="three_letter_country",
            projection="robinson",
            hover_name="Country",
            hover_data={
                "Agreement": True,
                "three_letter_country": False,
                "agreement_raw": False,
            },
            labels={
                "agreement_raw": ""
            },  # For consistency with legend in the subject tab
        )

        # Change default colour for missing-data countries
        fig.update_geos(
            landcolor="#e1e1e1"  # light grey
        )

        # Add grey trace for countries with no shared votes
        if not no_shared_votes.empty:
            country1_name_hover = data.get_country_display_name(country1)
            no_vote_codes = no_shared_votes["three_letter_country"].tolist()
            no_vote_names = no_shared_votes["Country"].tolist()
            hover_texts = [
                f"<b>{name}</b><br><br>No shared vote with {country1_name_hover}<extra></extra>"
                for name in no_vote_names
            ]
            fig.add_trace(
                go.Choropleth(
                    locations=no_vote_codes,
                    z=[0] * len(no_vote_codes),
                    colorscale=[[0, "#e1e1e1"], [1, "#e1e1e1"]],
                    showscale=False,
                    hovertemplate=hover_texts,
                    marker_line_color="#999999",
                    marker_line_width=0.5,
                )
            )

        fig.add_trace(
            go.Choropleth(
                locations=[country1],
                z=[1],  # dummy value
                colorscale=[[0, "green"], [1, "green"]],  # solid green color
                showscale=False,
                hovertemplate=f"<b>{data.get_country_display_name(country1)}</b> (Selected)<extra></extra>",
                marker_line_color="black",
                marker_line_width=2,
            )
        )

        # Center the map on the selected country's longitude (x-axis)
        # Keep y-axis at the equator (latitude = 0)
        country_longitude = get_country_longitude(country1)
        fig.update_geos(projection_rotation_lon=-country_longitude)

        # Change internal padding
        fig.update_layout(
            margin=dict(l=10, r=10, t=0, b=0),
            coloraxis_colorbar=dict(
                len=0.9,  # Reduce legend height
                tickvals=[0, 0.25, 0.5, 0.75, 1.0],
                ticktext=[
                    "0 (Always voting opposed)",
                    "0.25",
                    "0.5",
                    "0.75",
                    "1 (Always voting the same)",
                ],
            ),
        )

        # Status message
        country1_name = data.get_country_display_name(country1)
        # f"{country1_name} is highlighted in green.
        note_msg = html.P(
            [
                html.Strong("Details: "),
                f"The map shows the pairwise vote agreement between {country1_name} and other countries. "
                "An agreement score of 1 (dark blue) means that two countries voted the same on all "
                "General Assembly (GA) resolutions. A score of 0 (dark red) means that two countries "
                'always voted in opposite ways ("yes" vs. "no"). The data only covers GA resolutions '
                "that were successfully passed. The map provides a simplified, static overview of "
                "political geography. Some smaller nations and territories are not shown and the  "
                "map does not reflect historical border changes over time."
            ],
            style={
                "maxWidth": "100%",
                "margin": "0 0 0 0",
                "paddingLeft": "2%",
                "paddingTop": "10px",
                "color": "#7f8c8d",
                "fontSize": "16px",
                "lineHeight": "1.6",
                "textAlign": "left",
                "borderTop": "1px solid #eee",
            },
        )

        return fig, None, note_msg


layout = [
    html.Div(
        [
            html.Div(id="agreement-choropleth-status"),
            dcc.Loading(
                children=[
                    dcc.Graph(
                        id="agreement-choropleth",
                        style={"height": "600px", "width": "100%"},
                    ),
                ],
                type="circle",
                color="#3498db",
            ),
            html.Div(id="agreement-choropleth-note"),
        ]
    )
]
