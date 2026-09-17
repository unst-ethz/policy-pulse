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
from . import data  # noqa: E402
from .features import breadcrumb, footer, navbar  # noqa: E402

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


@server.before_request
def _ensure_data_reloader() -> None:
    """Start this worker's data-reload poller on its first request.

    Deliberately lazy rather than started at import: gunicorn runs with --preload, so import
    happens in the master and only the forking thread survives into each worker. A poller
    started at import time would live in the master, which serves no requests, and be missing
    from every worker. `start()` is a no-op after the first call per process, so this costs a
    dict lookup per request. It also covers the dev server, which never forks.
    """
    data.start_reloader()


def main() -> None:
    """Run the local development server (`uv run start-app`).

    Development only: it enables Dash's debug mode and binds to localhost. Production serves the
    `server` object above through gunicorn instead — see the Dockerfile.
    """
    app.run(debug=True, port=8050, host="127.0.0.1")


if __name__ == "__main__":
    main()
