import dash
from dash import dcc, html

from .components import navbar

app = dash.Dash(__package__, use_pages=True)
app.layout = html.Div(
    [
        # Global client-side store shared across all pages.
        # Graphs can read from this by using Input("moving-average-data", "data")
        dcc.Store(id="moving-average-data"),
        dcc.Store(id="moving-average-calc-time"),
        # Nav bar
        *navbar.layout,
        # Layout file
        dash.page_container,
    ]
)
navbar.register_callbacks()

if __name__ == "__main__":
    app.run(debug=True, port=8050, host="127.0.0.1")
