from dash import dcc, Input, Output, callback, html
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from io import StringIO
import  plotly.express as px
from ..data import get_country_name


_SPECIAL_SESSION_COLOUR = 'black'


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
            Input("special-sessions-radio", "value"),
        ],
    )
    def generate_chart(filter_store, session_data, special_sessions):
        if session_data is None or not filter_store:
            return go.Figure(), html.Div("No data"), ""

        country1 = filter_store.get("country1_alpha3")
        country2 = filter_store.get("country2")

        df = pd.read_json(StringIO(session_data))

        # ensure country2 is a list
        selected = country2 if isinstance(country2, (list, tuple)) else [country2]

        # detect special/emergency sessions (contain "sp" e.g. "1sp", "7emsp")
        is_special = df["session"].astype(str).str.contains("sp", case=False, na=False)

        if not special_sessions:
            df = df[~is_special]
            is_special = is_special[~is_special]

        fig = go.Figure()
        colors = px.colors.qualitative.Plotly
        for i, c in enumerate(selected):
            avg_col = f"avg_agreement_{c}"
            n_col = f"n_votes_{c}"
            if avg_col not in df.columns:
                continue

            # filter to non-NaN rows so all dots are connected
            mask = df[avg_col].notna()
            df_c = df.loc[mask].copy()
            special_c = is_special.loc[mask]
            country_color = colors[i % len(colors)]

            hover_data = np.column_stack([
                df_c["session"].values,
                df_c[n_col].values if n_col in df_c.columns else np.zeros(len(df_c)),
            ])

            # per-point marker colors
            marker_colors = [_SPECIAL_SESSION_COLOUR if s else country_color for s in special_c]

            fig.add_trace(
                go.Scatter(
                    x=df_c["year"],
                    y=df_c[avg_col],
                    mode="markers+lines",
                    name=get_country_name(c),
                    line=dict(color=country_color),
                    marker=dict(size=6, color=marker_colors),
                    customdata=hover_data,
                    hovertemplate=(
                        "Session %{customdata[0]}<br>"
                        "Year: %{x}<br>"
                        "Agreement: %{y:.3f}<br>"
                        "Shared votes: %{customdata[1]:.0f}"
                        "<extra>%{fullData.name}</extra>"
                    ),
                )
            )

        fig.update_layout(
            title=f"Agreement: {get_country_name(country1)} vs {', '.join([get_country_name(c) for c in selected])}",
            xaxis_title="Year",
            yaxis_title="Agreement Score",
            yaxis=dict(range=[-0.03, 1.06]),
            template="plotly_white",
        )

        # status message
        n_sessions = len(df)
        year_min = int(df["year"].min()) if not df["year"].isna().all() else "N/A"
        year_max = int(df["year"].max()) if not df["year"].isna().all() else "N/A"
        country1_name = get_country_name(country1)
        status_msg = html.Div(
            f"Covers {n_sessions} sessions from {year_min} to {year_max}.",
            style={"color": "#7f8c8d", "fontSize": "14px", "padding": "4px 0"},
        )

        note_msg = html.P([
            html.Strong("Details: "),
            f"The chart shows the average pairwise vote agreement between {country1_name} "
            "and the selected comparison countries per UN General Assembly session. "
            "Each point represents the mean agreement score across all resolutions in that session "
            "where both countries voted. An agreement score of 1 means identical votes on all resolutions; "
            '0 means complete disagreement (Yes vs. No). '
            "Only sessions with at least 3 shared votes are shown. "
            "Back dots indicate special and emergency sessions; excluded by default "
            "(use the checkbox above to include them). "
            "The data only covers GA resolutions that were passed."
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
            html.Div(
                dcc.Checklist(
                    id="special-sessions-radio",
                    options=[{"label": " Include special / emergency sessions", "value": "include"}],
                    value=[],
                    style={"fontSize": "14px", "color": "#555"},
                ),
                style={"padding": "4px 0 8px 0"},
            ),
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
