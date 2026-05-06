from dash import dcc, Input, Output, callback, html
import plotly.graph_objects as go
import pandas as pd
import plotly.express as px

from .. import data

from .country_coordinates import get_country_longitude
from .color_utils import make_adaptive_colorscale_plotly


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
        adaptive_colour_scale = "adaptive" in (color_mode or [])

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

        agreement_data = query_engine.query_agreement_between_countries(
            country1,
            resolution_ids=all_resolutions["undl_id"].tolist(),
            average=True,
        )

        # Clean the returned agreement data
        agreement_data.drop(columns=['source_country', 'resolution_count'], inplace=True)
        agreement_data = agreement_data.T.reset_index()
        agreement_data.columns = ["three_letter_country", "agreement_score"]
        agreement_data["Country"] = agreement_data["three_letter_country"].apply(
            data.get_country_display_name
        )
        agreement_data["Agreement"] = agreement_data["agreement_score"].apply(
            lambda x: f"{x:.2f} with {data.get_country_display_name(country1)}"
            if pd.notna(x) else "No shared vote"
        )

        # Set up colour scale based on the distribution of consensus scores
        # Crucial: We need to restrict to resolutions where country1 actually voted
        # since these are the only ones that can contribute to any bilateral agreement
        # score on the Choropleth map.
        country1_resolutions = all_resolutions.dropna(subset=[country1]) if country1 in all_resolutions.columns else all_resolutions
        has_consensus = "consensus_score" in all_resolutions.columns and not country1_resolutions.empty
        use_adaptive = adaptive_colour_scale and has_consensus
        if use_adaptive:
            colorscale, lo, avg, hi = make_adaptive_colorscale_plotly(
                country1_resolutions["consensus_score"],
                base_colorscale="RdYlBu"
            )
            range_color = [lo, hi]
            colorbar_settings = dict(
                len=0.9,
                tickvals=[lo, avg, hi],
                ticktext=[f"{lo:.2f}", f"{avg:.2f} (avg)", f"{hi:.2f}"],
            )
        else:
            colorscale = px.colors.diverging.RdYlBu
            range_color = [0, 1]
            colorbar_settings = dict(
                len=0.9,
                tickvals=[0, 0.25, 0.5, 0.75, 1.0],
                ticktext=["0 (Always opposed)", "0.25", "0.5", "0.75", "1 (Always agreeing)"],
            )

        if agreement_data.empty:
            agreement_data = pd.DataFrame(
                {
                    "three_letter_country": ["NAN"],
                    "Country": ["No data for selected filters"],
                    "agreement_score": [None],
                    "Agreement": [""],
                }
            )

        agreement_data["agreement_score"] = pd.to_numeric(
            agreement_data["agreement_score"], errors="coerce"
        )

        # Separate countries with no shared votes (NaN agreement)
        no_shared_votes = agreement_data[agreement_data["agreement_score"].isna()].copy()
        agreement_data = agreement_data[agreement_data["agreement_score"].notna()]

        # Plot the choropleth world map
        fig = px.choropleth(
            agreement_data,
            color="agreement_score",
            color_continuous_scale=colorscale,
            range_color=range_color,
            locations="three_letter_country",
            projection="robinson",
            hover_name="Country",
            hover_data={
                "Agreement": True,
                "three_letter_country": False,
                "agreement_score": False,
            },
            labels={"agreement_score": ""},
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

        # Details annotation
        country1_name = data.get_country_display_name(country1)
        if use_adaptive:
            note_text = (
                f"The map shows the pairwise vote agreement between {country1_name} and other countries. "
                "A high score means that two countries often voted the same on the selected General Assembly (GA) "
                "resolutions; a low score means that two countries often voted in opposite ways (Yes vs. No). "
                "The colour scale runs from the 1st to the 99th percentile of consensus scores across all "
                f"selected resolutions ({lo:.2f}–{hi:.2f}), and its midpoint (yellow) is anchored at the global "
                f"consensus average ({avg:.2f}). "
                "Each resolution's consensus score is the average pairwise agreement among all voting country pairs. "
                f"Countries appearing blue thus agreed with {country1_name} more than the global average; "
                "those appearing red agreed less. Agreement values shown on hover remain absolute (0–1). "
            )
        else:
            note_text = (
                f"The map shows the pairwise vote agreement between {country1_name} and other countries. "
                "An agreement score of 1 (dark blue) means that two countries voted the same on all selected "
                "General Assembly (GA) resolutions; a score of 0 (dark red) means that two countries "
                "always voted in opposite ways (Yes vs. No). "
            )

        shared_disclaimer = (
            "The data only covers GA resolutions that were successfully passed. "
            "The map provides a simplified, static overview of political geography. Some smaller nations "
            "and territories are not shown and the map does not reflect historical border changes over time. "
            "The boundaries and names shown and the designations used on this map do not imply official "
            "endorsement or acceptance by the United Nations."
        )

        note_text += shared_disclaimer

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
            dcc.Checklist(
                id="choropleth-color-mode",
                options=[{"label": " Centre colours on global average and scale to range", "value": "adaptive"}],
                value=[],
                style={"fontSize": "14px", "color": "#555", "marginBottom": "8px", "paddingLeft": "2%"},
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
