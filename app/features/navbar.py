from dash import get_relative_path, html, dcc

import os

# At runtime, set this env variable to True if you want to show indicators
# that the web page is experimental (e.g. to distinguish from a stable version).
experimental = os.getenv("WARN_EXPERIMENTAL", "True") == "False"

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
