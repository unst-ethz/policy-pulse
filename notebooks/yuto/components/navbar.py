from dash import Input, Output, clientside_callback, html, dcc

from .. import data


def register_callbacks():
    pass


layout = (
    html.Header(
        [
            html.H1(
                "UN-ETH Policy Pulse",
            ),
            html.Div(
                dcc.Dropdown(
                    options=[
                        {
                            "label": dcc.Link([country], href="/country/" + country),
                            "value": country,
                        }
                        for country in data.available_countries
                    ],
                    # value=data.available_countries[0],
                    clearable=False,
                    className="navbar-dropdown",
                    placeholder="Search for a country...",
                ),
            ),
            html.Div(),
        ],
        className="navbar",
    ),
)
