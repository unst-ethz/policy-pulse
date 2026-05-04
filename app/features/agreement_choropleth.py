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
            Input("choropleth-color-mode", "value"),
        ],
    )
    def generate_chart(filtered_data, filter_store, color_mode):
        color_mode = color_mode or "absolute"

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

        # Calculate the average "consensus score" for the filtered resolutions
        global_consensus_avg = None
        if 'consensus_score' in resolutions_in_year.columns:
            global_consensus_avg = resolutions_in_year['consensus_score'].mean()

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

        use_demeaned = color_mode == "demeaned" and pd.notna(global_consensus_avg)

        if use_demeaned:
            agreement_data["agreement_plot"] = agreement_data["agreement_raw"] - global_consensus_avg
            agreement_data["Agreement"] = agreement_data["agreement_raw"].apply(
                lambda x: f"{x - global_consensus_avg:+.2f} vs. average ({global_consensus_avg:.2f})"
                if pd.notna(x) else "No shared vote"
            )
            colorscale = px.colors.diverging.RdYlGn
            range_color = [-0.25, 0.25]
            colorbar_settings = dict(
                len=0.9,
                tickvals=[-0.25, 0, 0.25],
                ticktext=["−0.25", "0 (average)", "+0.25"],
            )
        else:
            agreement_data["agreement_plot"] = agreement_data["agreement_raw"]
            agreement_data["Agreement"] = agreement_data["agreement_raw"].apply(
                lambda x: f"{x:.2f} with {data.get_country_display_name(country1)}"
                if pd.notna(x) else "No shared vote"
            )
            colorscale = px.colors.diverging.RdYlBu
            range_color = [0, 1]
            colorbar_settings = dict(
                len=0.9,
                tickvals=[0, 0.25, 0.5, 0.75, 1.0],
                ticktext=["0", "0.25", "0.5", "0.75", "1"],
            )

        if agreement_data.empty:
            # Simulate neutral 0.5 value for all countries
            agreement_data = pd.DataFrame(
                {
                    "three_letter_country": ["NAN"],
                    "Country": ["No data for selected filters"],
                    "agreement_raw": [None],
                    "agreement_plot": [0.0 if use_demeaned else 0.5],
                    "Agreement": [""],
                }
            )

        agreement_data["agreement_plot"] = pd.to_numeric(
            agreement_data["agreement_plot"], errors="coerce"
        )

        # Separate countries with no shared votes (NaN agreement)
        no_shared_votes = agreement_data[agreement_data["agreement_raw"].isna()].copy()
        agreement_data = agreement_data[agreement_data["agreement_raw"].notna()]

        # Plot the choropleth world map
        fig = px.choropleth(
            agreement_data,
            color="agreement_plot",
            color_continuous_scale=colorscale,
            range_color=range_color,
            locations="three_letter_country",
            projection="robinson",
            hover_name="Country",
            hover_data={
                "Agreement": True,
                "three_letter_country": False,
                "agreement_plot": False,
                "agreement_raw": False,
            },
            labels={"agreement_plot": ""},
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
                colorscale=[[0, "#a078d3"], [1, "#a078d3"]],
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
            coloraxis_colorbar=colorbar_settings,
        )

        # Note message
        shared_disclaimer = (
            "The data only covers GA resolutions that were successfully passed. "
            "The map provides a simplified, static overview of political geography. Some smaller nations "
            "and territories are not shown and the map does not reflect historical border changes over time. "
            "The boundaries and names shown and the designations used on this map do not imply official "
            "endorsement or acceptance by the United Nations."
        )

        country1_name = data.get_country_display_name(country1)
        if use_demeaned:
            note_text = (
                f"The map shows how much more (green) or less (red) each country agreed with {country1_name} "
                f"compared to the average consensus score across all selected resolutions ({global_consensus_avg:.2f}). " 
                "Each resolution's consensus score is the average pairwise agreement among all voting country pairs. "
                f"A value of 0 (yellow) means that a given country agreed with {country1_name} at exactly the "
                f"average rate. {shared_disclaimer}"
            )
        else:
            note_text = (
                f"The map shows the pairwise vote agreement between {country1_name} and other countries. "
                "An agreement score of 1 (dark blue) means that two countries voted the same on all selected "
                "General Assembly (GA) resolutions. A score of 0 (dark red) means that two countries "
                f"always voted in opposite ways (Yes vs. No). {shared_disclaimer}"
            )

        note_msg = html.P(
            [html.Strong("Details: "), note_text],
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
            dcc.RadioItems(
                id="choropleth-color-mode",
                options=[
                    {"label": "Absolute (0–1)", "value": "absolute"},
                    {"label": "Relative to average", "value": "demeaned"},
                ],
                value="absolute",
                inline=True,
                style={"fontSize": "16px", "marginBottom": "8px", "paddingLeft": "2%"},
            ),
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
