from dash import html

layout = html.Footer(
    html.Div(
        [
            html.Div(
                [
                    html.Span("UN-ETH Policy Pulse", className="footer-title"),
                    html.Span(
                        "A student project within the United Nations Student Team at ETH Zürich, in collaboration with the UN Dag Hammarskjöld Library.",
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
                        "United Nations Student Team",
                        href="https://un-eth.ethz.ch/exchanges/un-eth-student-team.html",
                    ),
                    html.A(
                        "UN-ETH Partnership",
                        href="https://un-eth.ethz.ch/",
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
