from dash import Input, Output, callback, html, dcc


def register_callbacks():
    @callback(Output("breadcrumb", "children"), Input("country1-dropdown", "value"))
    def update_breadcrumb(country1_iso: str):
        crumbs = [html.Span("Home"), html.Span(">")]
        crumbs.append(html.Span("Country (" + country1_iso + ")"))
        return html.Div(children=crumbs)


layout = (html.Div(id="breadcrumb"),)
