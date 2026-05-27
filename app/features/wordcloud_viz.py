from dash import get_relative_path, html, dcc, clientside_callback, Input, Output

_TABS = [
    ("Default",      "wordcloud_hd_default.png"),
    ("Geopolitical", "wordcloud_hd_geopolitical.png"),
    ("Thematic",     "wordcloud_hd_thematic.png"),
    ("Action",       "wordcloud_hd_action.png"),
    ("Subjects",     "wordcloud_hd_subjects.png"),
]
_N = len(_TABS)

layout = html.Div(
    [
        dcc.RadioItems(
            id="wc-tab-selector",
            options=[{"label": label, "value": i} for i, (label, _) in enumerate(_TABS)],
            value=0,
            inline=True,
            inputStyle={"display": "none"},
            labelStyle={
                "cursor": "pointer",
                "padding": "6px 14px",
                "borderBottom": "2px solid transparent",
                "fontSize": "0.85rem",
                "color": "#555",
                "userSelect": "none",
            },
            className="wc-tab-radio",
        ),
        html.Div(
            [
                html.Img(
                    src=get_relative_path(f"/assets/{fname}"),
                    id={"type": "wc-img", "index": i},
                    style={
                        "width": "100%",
                        "display": "block" if i == 0 else "none",
                    },
                )
                for i, (_, fname) in enumerate(_TABS)
            ]
        ),
    ]
)


def register_callbacks():
    clientside_callback(
        """
        function(selected) {
            return Array.from({length: """ + str(_N) + """}, (_, i) =>
                i === selected
                    ? {"width": "100%", "display": "block"}
                    : {"display": "none"}
            );
        }
        """,
        [Output({"type": "wc-img", "index": i}, "style") for i in range(_N)],
        Input("wc-tab-selector", "value"),
    )
