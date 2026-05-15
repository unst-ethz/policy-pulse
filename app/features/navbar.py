from dash import get_relative_path, html, dcc

import os

experimental = os.getenv("WARN_EXPERIMENTAL", "True") == "True"

layout = (
    html.Header(
        [
            html.H1(
                html.Div(
                    id="navbar-home-click",
                    children=[
                        dcc.Link("UN-ETH Policy Pulse", href=get_relative_path("/")),
                        html.Span(" Experimental Branch; Unstable!")
                        if experimental
                        else None,
                    ],
                )
            ),
            html.Div(),
        ],
        className=f"navbar {'navbar-experimental' if experimental else ''}",
    ),
)
