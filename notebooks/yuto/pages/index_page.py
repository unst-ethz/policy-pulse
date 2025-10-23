import functools
import time
from dash import Input, Output, callback, clientside_callback, html, dcc, register_page
import pandas as pd

from ..components import breadcrumb
from ..components import alignment_choropleth
from ..components import alignment_graph
from ..components import wordcloud_viz
from .. import data


register_page(__name__, path="/", title="Policy Pulse: Homepage")

layout = html.Div(
    [
        html.H1("Homepage"),
        html.H2("Keyword Wordcloud for GA Resolution Subjects (Not Country Specific)"),
        *wordcloud_viz.layout,
    ],
)

wordcloud_viz.register_callbacks()
