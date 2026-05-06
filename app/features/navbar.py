from dash import get_relative_path, html, dcc

layout = (
    html.Header(
        [
            html.H1(html.Div(id="navbar-home-click", children=dcc.Link("UN-ETH Policy Pulse", href=get_relative_path("/")))),
            html.Div(),
        ],
        className="navbar",
    ),
)
