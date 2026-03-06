from dash import html

layout = html.Div(
    [
        html.P(
            "The UN-ETH Policy Pulse is a platform designed by students in the United Nations Student Team at ETH Zürich to provide a more user-friendly interface for exploring the change in UN voting patterns over time and across different policy areas."
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.P("Some of the features we have are:"),
                        html.Ul(
                            [
                                html.Li(
                                    "Filter resolutions by policy area, year, or country to explore specific trends."
                                ),
                                html.Li(
                                    "Compare two countries' voting patterns over the years."
                                ),
                            ]
                        ),
                    ]
                ),
                html.Div(
                    [
                        html.P(
                            "There are several limitations with the platform at the moment that you should be aware of when analyzing data with it:"
                        ),
                        html.Ul(
                            [
                                html.Li(
                                    "The platform is still in development, so there may be some bugs or missing features."
                                ),
                                html.Li(
                                    "Only accepted resolutions at the United Nations General Assembly (UNGA) are taken into account. This may skew the results heavily towards certain policy areas or resolutions with higher consensus."
                                ),
                                html.Li(
                                    "Voting patterns, alignment, and trends are unweighted by the number of resolutions in the specific subject area. This means that some subjects with a disproportionally high amount of resolutions may skew the results when viewing two country's alignment."
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
