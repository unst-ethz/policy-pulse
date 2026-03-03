from dash import html, dcc, register_page

from ..features import recent_resolutions_panel, wordcloud_viz


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
        html.H2("Recent Resolutions"),
        recent_resolutions_panel.layout,
    ],
)


wordcloud_viz.register_callbacks()
recent_resolutions_panel.register_callbacks()
