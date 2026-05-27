from dash import get_relative_path, html, dcc


def register_callbacks():
    return


_TABS = [
    ("Default",      "wordcloud_hd_default.png"),
    ("Geopolitical", "wordcloud_hd_geopolitical.png"),
    ("Thematic",     "wordcloud_hd_thematic.png"),
    ("Action",       "wordcloud_hd_action.png"),
    ("Subjects",     "wordcloud_hd_subjects.png"),
]

layout = html.Div(
    dcc.Tabs(
        [
            dcc.Tab(
                html.Img(
                    src=get_relative_path(f"/assets/{fname}"),
                    style={"width": "100%", "display": "block"},
                ),
                label=label,
            )
            for label, fname in _TABS
        ],
        persistence=True,
        persistence_type="session",
    )
)
