from dash import html

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
