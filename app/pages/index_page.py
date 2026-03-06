from dash import html, dcc, register_page

from ..features import general_stats_panel, recent_resolutions_panel, wordcloud_viz


register_page(__name__, path="/", title="Policy Pulse: Homepage")

layout = html.Div(
    [
        html.Div(
            [
                html.H1("Welcome to Policy Pulse"),
                html.Span(
                    "This project aims to make it easier to browse and understand trends on UN voting data.",
                    className="main-subtitle",
                ),
                html.Div(
                    [
                        dcc.Link(
                            "Explore Data →",
                            href="/trends",
                            className="cta-button",
                        ),
                        dcc.Link(
                            "Walk me through a case study ↓",
                            href="#case-study",
                            className="cta-button secondary",
                        ),
                    ],
                    className="main-header-buttons",
                ),
                html.Div(
                    [
                        html.Img(
                            src="/assets/main.webp",
                            className="main-header-image",
                        ),
                    ],
                    className="main-header-image-container",
                ),
            ],
            className="main-header",
        ),
        html.H2(html.Span("Recent updates"), className="section-title"),
        recent_resolutions_panel.layout,
        html.H2(html.Span("At a quick glance"), className="section-title"),
        html.Div(
            [
                general_stats_panel.layout,
                wordcloud_viz.layout,
            ],
            className="desktop-only-dual-column",
        ),
    ],
)


wordcloud_viz.register_callbacks()
general_stats_panel.register_callbacks()
recent_resolutions_panel.register_callbacks()
