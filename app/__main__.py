import dash
from dash import dcc, html

external_scripts = [
    {
        "src": "https://unst-dev2.vsos.ethz.ch/api/script.js",
        "data-site-id": "9e3feb116761",
        "defer": True,
    }
]

app = dash.Dash(
    __package__,
    use_pages=True,
    suppress_callback_exceptions=True,
    external_scripts=external_scripts,
    # Base path may be set via DASH_URL_BASE_PATHNAME env variable
)

server = app.server

# Features must be imported after app is initialized, so they can use
# get_relative_path to resolve links based on the app's base path.
from .features import navbar  # noqa: E402
from .features import footer  # noqa: E402
from .features import breadcrumb  # noqa: E402

app.layout = html.Div(
    [
        # Global client-side store shared across all pages.
        # Graphs can read from this by using Input("moving-average-data", "data")
        dcc.Store(id="moving-average-data"),
        dcc.Store(id="moving-average-calc-time"),
        # Nav bar
        *navbar.layout,
        html.Div(
            className="container",
            children=[
                # *breadcrumb.layout,
                # Layout file
                dash.page_container,
            ],
        ),
        footer.layout,
    ]
)
breadcrumb.register_callbacks()

if __name__ == "__main__":
    app.run(debug=True, port=8050, host="127.0.0.1")
