from dash import Input, Output, callback, html, dcc


def register_callbacks():

    @callback(
        Output("breadcrumb", "children"),
        Input("country1-localised-name", "data"),
    )
    def update_breadcrumb(country_full_name: str | None):
        crumbs = [html.Span("Home")]
        if country_full_name is not None:
            crumbs.append(html.Span(">"))
            crumbs.append(html.Span("Country (" + country_full_name + ")"))
        return html.Div(children=crumbs)


layout = (html.Div(id="breadcrumb"),)
