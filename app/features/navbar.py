from dash import html, dcc

layout = (
    html.Header(
        [
            html.H1(html.Div(id="navbar-home-click", children=dcc.Link("UN-ETH Policy Pulse", href="/"))),
            html.Div(),
        ],
        className="navbar",
    ),
)
