from dash import callback, Input, Output, html, dcc


def register_callbacks():

    @callback(
        Output("wordcloud-viz-image", "children"),
        Input("wordcloud-viz-tabs", "value"),
    )
    def update_viz(chosen_tab):
        if chosen_tab == "all":
            return html.Img(src="/assets/all.png", width="100%")
        elif chosen_tab == "1625":
            return html.Img(src="/assets/16-25.png", width="100%")
        elif chosen_tab == "0615":
            return html.Img(src="/assets/06-15.png", width="100%")
        elif chosen_tab == "9605":
            return html.Img(src="/assets/96-05.png", width="100%")
        elif chosen_tab == "8695":
            return html.Img(src="/assets/86-95.png", width="100%")


layout = (
    dcc.Tabs(
        id="wordcloud-viz-tabs",
        value="all",
        children=[
            dcc.Tab(label="All Years", value="all"),
            dcc.Tab(label="2016–2025", value="1625"),
            dcc.Tab(label="2006–2015", value="0615"),
            dcc.Tab(label="1996–2005", value="9605"),
            dcc.Tab(label="1986–1995", value="8695"),
        ],
    ),
    html.Div(id="wordcloud-viz-image"),
)
