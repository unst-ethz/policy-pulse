from dash import html

import os

build_commit = os.getenv("BUILD_COMMIT", "unknown")
build_date = os.getenv("BUILD_DATE", "unknown")
if build_date != "unknown":
    # GitHub provides with full ISO 8601 format, keep only date part
    build_date = build_date.split("T")[0]

layout = html.Footer(
    html.Div(
        [
            html.Div(
                [
                    html.Span("UN-ETH Policy Pulse", className="footer-title"),
                    html.Span(
                        "A volunteer project of the United Nations Student Team at ETH Zürich, "
                        "in collaboration with the UN Dag Hammarskjöld Library.",
                        className="footer-subtitle",
                    ),
                    html.Span(
                        "Commit {} - {}".format(build_commit, build_date),
                        className="footer-build-info",
                    )
                    if build_commit != "unknown" and build_date != "unknown"
                    else None,
                ],
                className="footer-column",
            ),
            html.Div(
                [
                    html.A(
                        "ETH Zürich",
                        href="https://ethz.ch",
                    ),
                    html.A(
                        "UN-ETH Student Team",
                        href="https://un-eth.ethz.ch/exchanges/un-eth-student-team.html",
                    ),
                    html.A(
                        "Dag Hammarskjöld Library",
                        href="https://www.un.org/en/library",
                    ),
                ],
                className="footer-column",
            ),
        ],
        className="container",
        style={
            "justifyContent": "space-between",
        },
    ),
    className="footer",
)
