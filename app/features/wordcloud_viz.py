from dash import get_relative_path, html


def register_callbacks():
    return


layout = html.Div(
    [
        html.Img(src=get_relative_path("/assets/wordcloud.png"), width="100%"),
    ]
)
