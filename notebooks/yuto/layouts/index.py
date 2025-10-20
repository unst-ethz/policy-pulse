from dash import Input, Output, clientside_callback, html, dcc

from ..components import navbar
from ..components import breadcrumb
from ..components import alignment_choropleth
from ..components import alignment_graph
from ..components import wordcloud_viz


def register_callbacks(query_engine):

    # Client-side callback from country1-dropdown (ISO alpha3) to ISO alpha2.
    # This is useful to pipe into frontend JS code that can take ISO alpha2
    # and convert to the full localised country name.
    clientside_callback(
        """
        function convert_to_2(iso_three_digit) {
            return new Intl.DisplayNames(["en"], { type: "region" }).of(window.getCountryISO2(iso_three_digit));
        }
        """,
        Output("country1-localised-name", "data"),
        Input("country1-dropdown", "value"),
    )

    navbar.register_callbacks()
    breadcrumb.register_callbacks()
    alignment_choropleth.register_callbacks(query_engine)
    alignment_graph.register_callbacks()
    wordcloud_viz.register_callbacks()


def layout(available_countries: list[str]):
    return html.Div(
        [
            dcc.Store(id="country1-localised-name"),
            # Nav bar
            *navbar.layout(available_countries),
            html.Div(
                className="container",
                children=[
                    *breadcrumb.layout,
                    # Status and cache info
                    html.Div(
                        id="status-display",
                    ),
                    *alignment_choropleth.layout,
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Label(
                                        "Select a country to compare with:",
                                        style={
                                            "fontWeight": "bold",
                                            "marginBottom": "5px",
                                        },
                                    ),
                                    dcc.Dropdown(
                                        id="country2-dropdown",
                                        options=[
                                            {"label": country, "value": country}
                                            for country in available_countries
                                        ],
                                        value=(
                                            available_countries[1]
                                            if len(available_countries) > 1
                                            else available_countries[0]
                                        ),
                                        clearable=False,
                                        style={"marginBottom": "15px"},
                                    ),
                                ],
                                style={
                                    "width": "30%",
                                    "display": "inline-block",
                                },
                            ),
                            html.Div(
                                [
                                    html.Label(
                                        "Time Span (days):",
                                        style={
                                            "fontWeight": "bold",
                                            "marginBottom": "5px",
                                        },
                                    ),
                                    dcc.Dropdown(
                                        id="timespan-dropdown",
                                        options=[
                                            {"label": "30 days", "value": 30},
                                            {"label": "90 days", "value": 90},
                                            {"label": "180 days", "value": 180},
                                            {"label": "365 days", "value": 365},
                                            {
                                                "label": "730 days (2 years)",
                                                "value": 730,
                                            },
                                        ],
                                        value=365,
                                        clearable=False,
                                        style={"marginBottom": "15px"},
                                    ),
                                ],
                                style={"width": "30%", "display": "inline-block"},
                            ),
                        ]
                    ),
                    *alignment_graph.layout,
                    *wordcloud_viz.layout,
                    # Footer with instructions
                    html.Div(
                        [
                            html.Hr(),
                            html.P(
                                [
                                    "💡 ",
                                    html.Strong("How it works:"),
                                    " Select countries and time span above. ",
                                    "Data is calculated on-demand and cached for fast re-access. ",
                                    "Alignment ranges from 0 (complete disalignment) to 1 (perfect alignment).",
                                ],
                                style={
                                    "color": "#7f8c8d",
                                    "textAlign": "center",
                                    "fontSize": "14px",
                                },
                            ),
                        ],
                        style={"padding": "20px", "marginTop": "40px"},
                    ),
                ],
            ),
        ]
    )
