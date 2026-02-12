from dash import html, dcc, register_page

from ..features import wordcloud_viz


register_page(__name__, path="/", title="Policy Pulse: Homepage")

layout = html.Div(
    [
        html.H1("Homepage"),
        dcc.Link(
            "Explore Trends →",
            href="/trends",
            className="cta-button",
        ),
        html.H2("Keyword Wordcloud for GA Resolution Subjects (Not Country Specific)"),
        *wordcloud_viz.layout,
    ],
)

wordcloud_viz.register_callbacks()
