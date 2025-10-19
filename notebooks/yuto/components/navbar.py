from dash import Input, Output, callback, html, dcc


def register_callbacks():
    pass


def layout(available_countries: list[str]):
    return (
        html.Header(
            [
                html.H1(
                    "UN-ETH Policy Pulse",
                ),
                html.Div(
                    dcc.Dropdown(
                        id="country1-dropdown",
                        options=[
                            {"label": country, "value": country}
                            for country in available_countries
                        ],
                        value=available_countries[0],
                        clearable=False,
                        className="navbar-dropdown",
                    ),
                ),
                html.Div(),
            ],
            className="navbar",
        ),
    )
