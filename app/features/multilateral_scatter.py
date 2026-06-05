import textwrap

from dash import dcc, Input, Output, callback, html
import plotly.graph_objects as go
import pandas as pd

from .. import data

# M49 top-level regions in a consistent display order with D3-palette colours
_REGION_ORDER = ["Africa", "Americas", "Asia", "Europe", "Oceania", "Other"]
_REGION_COLORS = {
    "Africa":   "#D62728",
    "Americas": "#1F77B4",
    "Asia":     "#FF7F0E",
    "Europe":   "#2CA02C",
    "Oceania":  "#9467BD",
    "Other":    "#7F7F7F",
}

_COLOR_DISAGREE = "#c0392b"
_COLOR_AGREE    = "#1558b0"
_COLOR_MEAN     = "#888888"

_MIN_VOTES_THRESHOLD = 10

_Y_AXIS_OPTIONS = [
    {"label": " Abstention Rate", "value": "abstention_rate"},
    {"label": " Yes Share",       "value": "yes_rate"},
    {"label": " No Share",        "value": "no_rate"},
    # TODO: Add "Not-Voting Share" (fraction of selected resolutions with no vote cast).
    #  Crucial: This would need to reflect countries' membership dates in the UN to be meaningful.
]
_Y_AXIS_LABELS = {o["value"]: o["label"].strip() for o in _Y_AXIS_OPTIONS}

_Y_AXIS_DETAILS = {
    "abstention_rate": (
        "The y-axis (Abstention Rate) is the fraction of votes actually cast that were abstentions (A) "
        "rather than Yes (Y) or No (N). "
        "Resolutions where the country did not participate at all are excluded from this calculation. "
    ),
    "yes_rate": (
        "The y-axis (Yes Share) is the fraction of votes actually cast that were Yes (Y) "
        "rather than No (N) or Abstain (A). "
        "Resolutions where the country did not participate at all are excluded from this calculation. "
    ),
    "no_rate": (
        "The y-axis (No Share) is the fraction of votes actually cast that were No (N) "
        "rather than Yes (Y) or Abstain (A). "
        "Resolutions where the country did not participate at all are excluded from this calculation. "
    ),
}


def _wrap(text: str, width: int = 36) -> str:
    """Break long callout text into lines for a Plotly annotation box."""
    return "<br>".join(textwrap.wrap(text, width))


def _callout_position(y_metric: str, cx: float, cy: float) -> tuple[float, float, str, str]:
    """Return (x, y, xanchor, yanchor) for the callout annotation.

    Primary positions keep the callout out of the dense data region:
      - yes_rate      → top-left    (data clusters top-right)
      - abstention/no → bottom-left (data clusters bottom-right)

    Falls back to the vertically mirrored corner (top-left ↔ bottom-left) when
    the selected country's paper-coordinate position (cx, cy) would be occluded
    by the primary box. The occlusion thresholds assume the box spans roughly
    0.30 × 0.20 paper units, which holds for the current font size and wrap width.
    """
    if y_metric == "yes_rate":
        occluded = cx < 0.30 and cy > 0.80
        if occluded:
            return 0.02, 0.04, "left", "bottom"   # top-left → bottom-left
        return 0.02, 0.96, "left", "top"
    else:
        occluded = cx < 0.30 and cy < 0.20
        if occluded:
            return 0.02, 0.96, "left", "top"      # bottom-left → top-left
        return 0.02, 0.04, "left", "bottom"


def register_callbacks(query_engine):

    @callback(
        [
            Output("multilateral-scatter", "figure"),
            Output("multilateral-scatter-status", "children"),
            Output("multilateral-details", "children"),
        ],
        [
            Input("filter-component-data-store", "data"),
            Input("filter-component-filter-store", "data"),
            Input("multilateral-y-axis", "value"),
        ],
    )
    def generate_scatter(filtered_data, filter_store, y_metric):
        y_metric = y_metric or "abstention_rate"
        if not filtered_data:
            return go.Figure(), "", ""

        resolutions = pd.read_json(filtered_data, orient="split")
        if resolutions.empty:
            return go.Figure(), html.Div([html.Strong("No resolutions to plot")]), ""

        stats = query_engine.query_multilateral_stats(resolutions["undl_id"].tolist())
        if stats.empty:
            return go.Figure(), html.Div([html.Strong("No data available")]), ""

        stats["country_name"] = stats["country"].apply(data.get_country_display_name)
        stats["region"] = stats["country"].apply(data.get_country_region)
        stats = stats.dropna(subset=["multilateral_alignment"])
        stats = stats[stats["participation_count"] >= _MIN_VOTES_THRESHOLD]

        mean_alignment = stats["multilateral_alignment"].mean()
        y_label = _Y_AXIS_LABELS[y_metric]
        fig = go.Figure()

        for region in _REGION_ORDER:
            group = stats[stats["region"] == region]
            if group.empty:
                continue
            fig.add_trace(go.Scatter(
                x=group["multilateral_alignment"],
                y=group[y_metric],
                mode="markers",
                name=region,
                marker=dict(size=10, color=_REGION_COLORS[region], opacity=0.65),
                text=group["country_name"],
                customdata=group[["multilateral_alignment", y_metric, "participation_count"]].values,
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "Multilateral vote agreement: %{customdata[0]:.3f}<br>"
                    f"{y_label}: %{{customdata[1]:.1%}}<br>"
                    "Votes cast: %{customdata[2]:.0f}"
                    "<extra></extra>"
                ),
            ))

        # Axis bounds computed here so the callout corner picker can use them.
        x_lo = min(0.295, stats["multilateral_alignment"].min() * 0.95)
        x_hi = max(0.95,  stats["multilateral_alignment"].max() * 1.05)
        y_lo = -0.01  # small gap so markers at y=0 aren't clipped by the axis line
        y_hi = max(0.505, stats[y_metric].max() * 1.06)

        country1 = (filter_store or {}).get("country1_alpha3")

        if country1:
            highlight = stats[stats["country"] == country1]
            if not highlight.empty:
                row = highlight.iloc[0]
                fig.add_trace(go.Scatter(
                    x=[row["multilateral_alignment"]],
                    y=[row[y_metric]],
                    mode="markers+text",
                    showlegend=False,
                    marker=dict(
                        size=14,
                        color=_REGION_COLORS.get(row["region"], _REGION_COLORS["Other"]),
                        line=dict(color="black", width=2),
                    ),
                    text=[row["country_name"]],
                    textposition="top right",
                    hovertemplate=(
                        f"<b>{row['country_name']}</b> (selected)<br>"
                        f"Multilateral vote agreement: {row['multilateral_alignment']:.3f}<br>"
                        f"{y_label}: {row[y_metric]:.1%}<br>"
                        f"Votes cast: {int(row['participation_count'])}"
                        "<extra></extra>"
                    ),
                ))

                country_name = row["country_name"]
                if row["multilateral_alignment"] >= mean_alignment:
                    side, tendency = "right", "tends to vote like the majority of Member States"
                else:
                    side, tendency = "left", "tends to vote against the majority of Member States"

                cx = (row["multilateral_alignment"] - x_lo) / (x_hi - x_lo)
                cy = (row[y_metric] - y_lo) / (y_hi - y_lo)
                ax, ay, xanchor, yanchor = _callout_position(y_metric, cx, cy)

                fig.add_annotation(
                    xref="paper", yref="paper",
                    x=ax, y=ay,
                    xanchor=xanchor, yanchor=yanchor,
                    text=_wrap(
                        f"{country_name} is located to the {side} of the mean "
                        f"line, showing that it {tendency}."
                    ),
                    showarrow=False,
                    align="left",
                    bgcolor="rgba(255, 255, 255, 0.88)",
                    bordercolor="#bbb",
                    borderwidth=1,
                    borderpad=8,
                    font=dict(size=13, color="#333"),
                )

        # mean_fraction: exact position of the mean line in paper coords (used for arrow tails).
        # mean_fraction_clamped: pulled into [0.2, 0.8] so text labels stay readable at extremes.
        mean_fraction         = (mean_alignment - x_lo) / (x_hi - x_lo)
        mean_fraction_clamped = max(0.2, min(0.8, mean_fraction))

        fig.update_layout(
            xaxis=dict(title="Multilateral Vote Agreement", range=[x_lo, x_hi]),
            yaxis=dict(title=y_label, tickformat=".0%", range=[y_lo, y_hi]),
            legend=dict(title="Region"),
            margin=dict(l=50, r=20, t=100, b=50),
            hovermode="closest",
        )

        fig.add_vline(
            x=mean_alignment,
            line=dict(color=_COLOR_MEAN, width=1, dash="dash"),
            annotation_text=f"mean ({mean_alignment:.2f})",
            annotation_position="top right",
            annotation_font=dict(color=_COLOR_MEAN, size=11),
        )

        # Row 1 — text labels, centred within each half (clamped position)
        fig.add_annotation(
            xref="paper", yref="paper",
            x=mean_fraction_clamped / 2,
            y=1.13,
            text="Tend to disagree with the majority",
            showarrow=False,
            font=dict(color=_COLOR_DISAGREE, size=13),
            xanchor="center",
            yanchor="bottom",
        )
        fig.add_annotation(
            xref="paper", yref="paper",
            x=mean_fraction_clamped + (1 - mean_fraction_clamped) / 2,
            y=1.13,
            text="Tend to agree with the majority",
            showarrow=False,
            font=dict(color=_COLOR_AGREE, size=13),
            xanchor="center",
            yanchor="bottom",
        )

        # Row 2 — spanning arrows.
        # Plotly does not render showarrow shafts that cross the plot/margin boundary,
        # so we use add_shape for the line (which supports yref="paper" values > 1)
        # and a tiny pixel-offset annotation for just the arrowhead triangle at each tip.
        arrow_y     = 1.06
        arrow_size  = 1.5  # arrowhead length multiplier
        arrow_width = 1.5  # arrowhead stroke width; equal to arrow_size here by coincidence
        fig.add_shape(type="line", xref="paper", yref="paper",
            x0=0.01, y0=arrow_y, x1=mean_fraction, y1=arrow_y,
            line=dict(color=_COLOR_DISAGREE, width=2))
        fig.add_shape(type="line", xref="paper", yref="paper",
            x0=mean_fraction, y0=arrow_y, x1=0.99, y1=arrow_y,
            line=dict(color=_COLOR_AGREE, width=2))
        # Arrowhead triangles: tail is 8px behind the tip so only the head is drawn
        fig.add_annotation(xref="paper", yref="paper", axref="pixel", ayref="pixel",
            x=0.01, y=arrow_y, ax=8, ay=0,
            text="", showarrow=True,
            arrowhead=2, arrowsize=arrow_size, arrowwidth=arrow_width, arrowcolor=_COLOR_DISAGREE)
        fig.add_annotation(xref="paper", yref="paper", axref="pixel", ayref="pixel",
            x=0.99, y=arrow_y, ax=-8, ay=0,
            text="", showarrow=True,
            arrowhead=2, arrowsize=arrow_size, arrowwidth=arrow_width, arrowcolor=_COLOR_AGREE)

        note_msg = html.P(
            [
                html.Strong("Details: "),
                "Each point represents a UN member state, coloured by geographical region. "
                "The x-axis (Multilateral Vote Agreement) is the average pairwise agreement between a country "
                "and all other countries that voted on the same resolution, averaged across all selected "
                f"resolutions. {_Y_AXIS_DETAILS[y_metric]} "
                "The dashed vertical line marks the mean multilateral voting agreement across all plotted countries. "
                "The more a country is located to the left of the mean line, the more often it votes "
                "against the majority. "
                f"Countries with fewer than {_MIN_VOTES_THRESHOLD} votes cast are excluded. "
                "The data only covers GA resolutions that were successfully passed."
            ],
            style={
                "maxWidth": "100%",
                "margin": "0 0 0 0",
                "paddingLeft": "2%",
                "paddingTop": "10px",
                "color": "#7f8c8d",
                "fontSize": "16px",
                "lineHeight": "1.6",
                "textAlign": "left",
                "borderTop": "1px solid #eee",
            },
        )
        return fig, None, note_msg


layout = [
    html.Div([
        html.Div(id="multilateral-scatter-status"),
        html.Div(
            [
                html.Span("Y-axis:", style={"fontSize": "14px", "color": "#555", "whiteSpace": "nowrap"}),
                dcc.Dropdown(
                    id="multilateral-y-axis",
                    options=_Y_AXIS_OPTIONS,
                    value="abstention_rate",
                    clearable=False,
                    style={"minWidth": "180px"},
                ),
            ],
            style={"display": "flex", "alignItems": "center", "gap": "8px", "marginBottom": "8px", "paddingLeft": "2%"},
        ),
        dcc.Loading(
            children=[dcc.Graph(id="multilateral-scatter", style={"height": "600px"})],
            type="circle",
            color="#3498db",
        ),
        html.Div(id="multilateral-details"),
    ])
]
