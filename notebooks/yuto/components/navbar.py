from dash import Input, Output, callback, clientside_callback, html, dcc

from .. import data


def register_callbacks():

    @callback(
        Output("navbar-country-dropdown", "value"),
        Input("navbar-home-click", "n_clicks"),
        prevent_initial_call=True
    )
    def reset_country_dropdown(n_clicks):
        return None


layout = (
    html.Header(
        [
            html.H1(html.Div(id="navbar-home-click", children=dcc.Link("UN-ETH Policy Pulse", href="/"))),
            html.Div(
                dcc.Dropdown(
                    id="navbar-country-dropdown",
                    options=[
                        {
                            "label": dcc.Link([data.get_country_name(country)], href="/country/" + country),
                            "value": country,
                            "search": data.get_country_name(country),
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
