from dash import html

layout = html.Div(
    [
        html.P(
            "The Policy Pulse platform is built by volunteers of the United Nations Student Team (UNST) at ETH Zürich — "
            "a student-run initiative that bridges STEM fields and international policy. In a collaboration with "
            "the UN Digital Library (UNDL), we aim to make UN voting data more easily accessible, lowering barriers "
            "to entry for delegates, students, researchers, and anyone with an interest in international relations."
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.P(html.Strong("Features")),
                        html.Ul(
                            [
                                html.Li([
                                    html.Strong("Resolution List: "),
                                    "Browse every UN General Assembly (UNGA) resolution alongside the selected countries votes. "
                                    "Sort by date, filter by vote type (Yes, No, Abstain), and search by keyword or subject area.",
                                ]),
                                html.Li([
                                    html.Strong("Agreement Map: "),
                                    "A world map showing how closely every other UN member has voted with your selected country.",
                                ]),
                                html.Li([
                                    html.Strong("Agreement Timeline: "),
                                    "A timeline tracking how the voting alignment between two countries "
                                    "has evolved session by session — especially useful for spotting the temporal divergence between member states based on political shifts or major world events.",
                                ]),
                                html.Li([
                                    html.Strong("Alignment by Subject: "),
                                    "Breaks down the agreement aligment between two countries by UN subject area, "
                                    "revealing which topics they reliably agree on and where they diverge.",
                                ]),
                                html.Li([
                                    html.Strong("Word Cloud: "),
                                    "Visualises the most frequent terms in resolution titles for the current filter selection, "
                                    "giving a quick sense (or ",
                                    html.Em("pulse"),
                                    ") of which themes dominate the agenda.",
                                ]),
                            ]
                        ),
                    ]
                ),
                html.Div(
                    [
                        html.P(html.Strong("Limitations")),
                        html.Ul(
                            [
                                html.Li(
                                    "Only adopted resolutions at the UNGA are included. "
                                    "Resolutions that were withdrawn or rejected, as well as votes in other UN bodies, are not covered."
                                ),
                                html.Li(
                                    "Agreement scores are unweighted by the number of resolutions per subject. "
                                    "Subject areas with many more resolutions "
                                    "can have an outsized effect on the overall agreement figure."
                                ),
                                html.Li(
                                    "To be completed ..."
                                ),
                            ]
                        ),
                    ]
                ),
            ],
            style={
                "display": "grid",
                "gap": "1rem",
                "grid-template-columns": "repeat(auto-fit, minmax(300px, 1fr))",
            },
        ),
    ]
)
