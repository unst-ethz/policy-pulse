from dash import get_relative_path, html, dcc, register_page

from ..features import (
    feature_description,
    general_stats_panel,
    recent_resolutions_panel,
    wordcloud_viz,
    case_study,
)


register_page(__name__, path="/", title="Policy Pulse: Homepage")

layout = html.Div(
    [
        html.Div(
            [
                html.H1("Welcome to Policy Pulse"),
                html.Span(
                    [
                        "This project aims to make it easier to discover trends in ",
                        html.B("voting data "),
                        "from the United Nations. You can currently analyse voting data for ",
                        html.B("accepted resolutions "),
                        "of the ",
                        html.B("UN General Assembly"),
                        ". More data sources and functionalities will be added over time.",
                    ],
                    className="main-subtitle",
                ),
                html.Div(
                    [
                        dcc.Link(
                            "Explore Data →",
                            href=get_relative_path("/trends"),
                            className="cta-button",
                        ),
                        html.A(
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
                            src=get_relative_path("/assets/main.webp"),
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
        html.H2(html.Span("About the platform"), className="section-title"),
        feature_description.layout,
        html.H2(
            html.Span("Case Studies: How can you use the platform?"),
            id="case-study",
            className="section-title",
        ),
        case_study.layout,
    ],
)


wordcloud_viz.register_callbacks()
general_stats_panel.register_callbacks()
recent_resolutions_panel.register_callbacks()
