from dash import html, dcc


layout = html.Div(
    [
        html.Div(
            [
                html.Span(
                    "We'll walk you through an example use of the Policy Pulse platform, from the perspective of a general member of the public."
                ),
                html.H2("1."),
                html.Span("First, navigate to the 'Trends' page to explore trends: "),
                dcc.Link("Click here to explore Trends", href="/trends"),
                html.H2("2."),
                html.Span(
                    "Here, you can see a list of all the policies in the database. Let's filter the policies to only show those related to 'Education'."
                ),
                html.H2("3."),
                html.Span(
                    "Now we can see a list of policies related to education. Let's click on the first policy to see more details about it."
                ),
            ]
        )
    ]
)
