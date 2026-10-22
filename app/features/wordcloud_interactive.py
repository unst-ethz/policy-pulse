from typing import Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, callback, dcc, html, no_update
from dash.exceptions import PreventUpdate
from matplotlib import cm as mpl_cm
from matplotlib import colors as mpl_colors
from wordcloud import WordCloud

from .. import data
from ..un_data_stream.analysis import filtering, keywords, word_cloud
from . import filtered_resolutions
from .color_utils import make_adaptive_colorscale_plotly
from .resolution_list import create_vote_indicator

# The keyword indices, search and term ranking live in `un_data_stream.analysis` (shared with the
# API); this module only lays the terms out and colours them.
_DEFAULT_MODE = keywords.DEFAULT_MODE
_WORDCLOUD_MODES = keywords.MODES
_CONSENSUS_CMAP_COLORS = ["#ff66cc", "#e6b24b", "#33cc33"]


def _index() -> keywords.KeywordIndex:
    """The keyword index of the data currently served; a reload swaps in a rebuilt one."""
    return keywords.keyword_index(data.query_engine.snapshot)


def _get_wordcloud_layout(word_freq_dict, seed=42, canvas_width=1200):
    """Generate word positions using wordcloud library to avoid overlaps"""
    if not word_freq_dict:
        return {}, {}, {}

    try:
        # Create WordCloud object with appropriate settings
        # Use a larger canvas with 2:1 aspect ratio (width:height)
        # width, height = 1600, 900
        width, height = canvas_width, 800

        wordcloud = WordCloud(
            width=width,
            height=height,
            background_color="white",
            prefer_horizontal=1.0,
            relative_scaling=0.5,
            min_font_size=16,
            max_font_size=100,
            max_words=len(word_freq_dict),
            random_state=seed,
            collocation_threshold=0,
            scale=1,
        )

        # Generate word cloud layout
        wordcloud.generate_from_frequencies(word_freq_dict)
        # wordcloud.to_file("wordcloud.jpg")
        # # Force layout generation by creating the image
        # # This ensures layout_ is populated with actual positions
        # _ = wordcloud.to_image()

        # Extract positions from layout
        # The layout_ attribute contains tuples of (word, font_size, position, orientation, color)
        word_positions = {}
        sizes_from_wc = {}
        orientations_from_wc = {}

        if hasattr(wordcloud, "layout_") and wordcloud.layout_:
            for item in wordcloud.layout_:
                if len(item) >= 4:
                    word, font_size, position, orientation = (
                        item[0],
                        item[1],
                        item[2],
                        item[3],
                    )
                    # Handle case where word might be a tuple (first character) or string
                    if isinstance(word, (tuple, list)):
                        word = word[0] if len(word) > 0 else str(word)
                    word = str(word)  # Ensure word is a string

                    # Position is a tuple (x, y) in pixel coordinates
                    if isinstance(position, tuple) and len(position) == 2:
                        word_positions[word] = position
                        sizes_from_wc[word] = font_size
                        orientations_from_wc[word] = orientation
                else:
                    raise RuntimeError("wordcloud.layout_ item length < 3")
            # print(f"word_positions: {word_positions}")
            # print(f"sizes_from_wc: {sizes_from_wc}")
            # print(f"orientations_from_wc: {orientations_from_wc}")
        else:
            raise RuntimeError("wordcloud.layout_ not found")
        return word_positions, sizes_from_wc, orientations_from_wc
    except Exception as e:
        print(f"Error generating wordcloud layout: {e}")
        import traceback

        traceback.print_exc()
        return {}, {}, {}


def _assign_colors_from_blues_cmap(frequencies):
    """Map word frequencies to hex colours using the Blues matplotlib colormap.

    Normalises each frequency into [0.3, 1.0] to avoid very light colours at the low end;
    uniform frequencies get a fixed medium blue.
    """
    cmap = mpl_cm.get_cmap("Blues")
    freq_arr = np.array(frequencies, dtype=float)
    if len(freq_arr) == 0:
        return []
    if np.max(freq_arr) != np.min(freq_arr):
        normed = 0.3 + 0.7 * ((freq_arr - np.min(freq_arr)) / (np.max(freq_arr) - np.min(freq_arr)))
    else:
        normed = np.full_like(freq_arr, 0.6)
    colors = [mpl_colors.rgb2hex(cmap(v)) for v in normed]
    return colors


def _assign_colors_from_custom_cmap(
    consensus_scores: list, colorscale: list, lo: float, hi: float
) -> list:
    """Map consensus scores to colours using a caller-supplied Plotly colorscale.

    Normalises each score into [lo, hi], then looks up the nearest colour entry by
    binary-searching the colorscale's position array. Scores outside the range are
    clamped to the extremes; missing scores get a neutral grey.
    """
    span = hi - lo if hi > lo else 1.0
    positions = np.array([p for p, _ in colorscale])
    clrs = [c for _, c in colorscale]
    result = []
    for score in consensus_scores:
        if score is None or (isinstance(score, float) and np.isnan(score)):
            result.append("#aaaaaa")
        else:
            t = float(np.clip((score - lo) / span, 0.0, 1.0))
            idx = int(np.clip(np.searchsorted(positions, t), 0, len(clrs) - 1))
            result.append(clrs[idx])
    return result


def _apply_keyword_filter(df: pd.DataFrame, filter_store: dict | None) -> pd.DataFrame:
    """Apply keyword filter with the same token semantics as resolution list/trends."""
    if df.empty or "undl_id" not in df.columns:
        return df
    keyword = (filter_store or {}).get("keyword")
    return filtering.filter_by_keyword(data.query_engine, df, str(keyword) if keyword else None)


def _get_mode_label(mode: str) -> str:
    return _WORDCLOUD_MODES.get(mode, {}).get("label", "Word Cloud")


def _subjects_specific_message(generic_message: str) -> str:
    return (
        "No subject terms are available for the current filters. "
        "No resolutions are categorized under the current filters."
        if generic_message
        else generic_message
    )


def _mode_empty_message(mode: str, generic_message: str) -> str:
    if mode == "category":
        return _subjects_specific_message(generic_message)
    return generic_message


def _build_empty_wordcloud_figure(message: str) -> go.Figure:
    return go.Figure().add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
    )


def _build_wordcloud(
    filter_store: dict | None = None,
    active_tab: str | None = None,
    mode: str = _DEFAULT_MODE,
    color_mode: Literal["frequency", "consensus"] = "frequency",
):
    """Build word cloud figure for the resolutions matching the current filters."""
    mode_label = _get_mode_label(mode)
    if not filter_store:
        return _build_empty_wordcloud_figure(
            _mode_empty_message(
                mode,
                f"No data available for {mode_label}. Please adjust filters.",
            )
        )

    try:
        index = _index()
        if not index.terms(mode):
            return _build_empty_wordcloud_figure(
                _mode_empty_message(
                    mode,
                    f"No {mode_label.lower()} data source is available.",
                )
            )

        df = filtered_resolutions.resolutions_for(filter_store, active_tab)
        if df.empty:
            return _build_empty_wordcloud_figure(
                _mode_empty_message(
                    mode,
                    f"No resolutions match the current filters for {mode_label}.",
                )
            )

        # Check if 'undl_id' column exists
        if "undl_id" not in df.columns:
            return _build_empty_wordcloud_figure(
                _mode_empty_message(
                    mode,
                    f"No resolution IDs available to render {mode_label} word cloud.",
                )
            )

        # Keep word cloud aligned with active keyword search filter.
        df = _apply_keyword_filter(df, filter_store)
        if df.empty:
            return _build_empty_wordcloud_figure(
                _mode_empty_message(
                    mode,
                    f"No {mode_label.lower()} terms match the current keyword filter.",
                )
            )

        if not index.aggregate_term_frequencies(df["undl_id"], mode=mode):
            return _build_empty_wordcloud_figure(
                _mode_empty_message(
                    mode,
                    f"No {mode_label.lower()} terms found for filtered resolutions.",
                )
            )

        # Words are sized by searchable result count, so visual size reflects "how many
        # resolutions can be found by clicking this word".
        terms = word_cloud.cloud_terms(index, df, mode, keyword=filter_store.get("keyword"))
        if not terms:
            return _build_empty_wordcloud_figure(
                _mode_empty_message(
                    mode,
                    f"No searchable {mode_label.lower()} terms for current filters.",
                )
            )
        search_count_by_word = {t.term: t.count for t in terms}
        word_freq = dict(search_count_by_word)

        words = list(word_freq.keys())
        freqs = list(word_freq.values())

        # Get word positions from wordcloud library.
        # In consensus mode use a narrower canvas so words cluster in the left
        # ~75% of the plot, leaving the right side free for the colorbar.
        # The normalization formula below still references wc_norm_width=1200,
        # so the narrower pixel positions map to the left portion of the axis.
        seed = 42
        consensus_bar = color_mode == "consensus" and "consensus_score" in df.columns
        # wc_gen_width = 1150 if consensus_bar else 1200
        wc_gen_width = 1150
        word_positions, sizes_from_wc, orientations_from_wc = _get_wordcloud_layout(
            word_freq, seed=seed, canvas_width=wc_gen_width
        )
        word_keys = word_positions.keys()

        # Filter words to only those that were positioned
        words_filtered = []
        freqs_filtered = []

        for word, freq in zip(words, freqs):
            word_key = word
            if word_key in word_keys:
                words_filtered.append(word)
                freqs_filtered.append(freq)

        print(f"{len(words_filtered)}/{len(words)} words were positioned by WordCloud")

        if not words_filtered:
            print("Warning: No words were positioned by WordCloud")
            return _build_empty_wordcloud_figure(
                _mode_empty_message(
                    mode,
                    f"Could not place {mode_label.lower()} terms on the canvas.",
                )
            )

        words = words_filtered
        freqs = freqs_filtered

        # Extract positions and normalize to plotly coordinates
        # WordCloud uses pixel coordinates (0,0) at top-left, we need to center and scale
        x_positions = []
        y_positions = []
        sizes = []

        # WordCloud canvas dimensions (from _get_wordcloud_layout)
        wc_width, wc_height = wc_gen_width, 800
        # wc_width, wc_height = 1200, 800
        # wc_width, wc_height = 1600, 900

        for word in words:
            word_key = word
            x_pixel, y_pixel = word_positions[word_key]

            # Normalize coordinates to maintain 2:1 aspect ratio
            # Scale x to -1 to 1 range based on width
            # Scale y to -1 to 1 range based on height
            # This preserves the 2:1 aspect ratio
            x_normalized = (y_pixel / wc_width) * 3 - 1.5  # Scale to -1 to 1 range
            y_normalized = (
                1 - (x_pixel / wc_height) * 2
            )  # Scale to -1 to 1, flip y-axis (wordcloud uses top-left origin)

            x_positions.append(x_normalized)
            y_positions.append(y_normalized)

            # Use wordcloud's font size, but scale it appropriately for plotly
            wc_font_size = sizes_from_wc[word_key] * 0.9
            # Scale font size proportionally to maintain visual consistency
            # Increase the scaling factor to make words larger
            plotly_size = max(16, min(80, int(wc_font_size)))
            sizes.append(plotly_size)
        # print(f"words: {words}")
        # print(f"freqs: {freqs}")
        # print(f"x_positions: {x_positions}")
        # print(f"y_positions: {y_positions}")
        # print(f"sizes: {sizes}")

        click_result_counts = [search_count_by_word.get(word, 0) for word in words]

        lo = hi = avg = None
        # `.notna().any()` guards the same case as the choropleth: a filter can select only
        # resolutions with no recorded vote, whose consensus score is NaN, which would otherwise
        # put "nan" on the colourbar ticks.
        if (
            color_mode == "consensus"
            and "consensus_score" in df.columns
            and df["consensus_score"].notna().any()
        ):
            colorscale, lo, avg, hi = make_adaptive_colorscale_plotly(
                df["consensus_score"], _CONSENSUS_CMAP_COLORS
            )
            word_scores = word_cloud.mean_consensus(index, words, mode, df)
            word_colors = _assign_colors_from_custom_cmap(word_scores, colorscale, lo, hi)
            hover_text = [
                f"Click to search <b>{word}</b><br>{count} resolutions"
                + (f"<br>Avg. consensus: {score:.2f}" if score is not None else "")
                for word, count, score in zip(words, click_result_counts, word_scores)
            ]
        else:
            word_colors = _assign_colors_from_blues_cmap(freqs)
            hover_text = [
                f"Click to search <b>{word}</b><br>{count} resolutions"
                for word, count in zip(words, click_result_counts)
            ]

        # Calculate hover marker positions centered on words
        # Since textposition="bottom right", the anchor is at bottom-right corner
        # We need to shift left and up to center the hover area on the word
        hover_x_positions = []
        hover_y_positions = []
        hover_sizes = []

        # WordCloud canvas dimensions for coordinate conversion
        wc_width, wc_height = 1200, 800
        # wc_width, wc_height = 1600, 900
        coord_range_x = 2.2  # from -1.1 to 1.1
        coord_range_y = 2.2

        for word, size, x_pos, y_pos in zip(words, sizes, x_positions, y_positions):
            # Estimate word dimensions in pixels
            # Character width factor ~0.6, line height factor ~1.2
            word_width_px = size * len(word) * 0.8
            word_height_px = size * 1

            # Convert pixel shifts to normalized coordinates
            # Shift left by half width, up by half height
            shift_x_normalized = (word_width_px / 2) / wc_width * coord_range_x
            shift_y_normalized = (word_height_px / 2) / wc_height * coord_range_y

            # Center the hover marker on the word
            hover_x_positions.append(x_pos + shift_x_normalized)
            hover_y_positions.append(y_pos - shift_y_normalized)  # + because y increases upward

            # Size hover area to roughly match word size
            hover_sizes.append(size * 1.5)

        # customdata: [hover_display_text, word] so resolution-table callback can read the word
        hover_customdata = [[ht, w] for ht, w in zip(hover_text, words)]

        # Invisible marker trace with a larger hover area so that
        # hovering near the *center* of a word still shows the tooltip,
        # without changing the visual position of the words.
        hover_trace = go.Scatter(
            x=hover_x_positions,
            y=hover_y_positions,
            mode="markers",
            marker=dict(
                size=hover_sizes,
                opacity=0,
            ),
            hovertemplate="%{customdata[0]}<extra></extra>",
            customdata=hover_customdata,
            hoverlabel=dict(bgcolor="white", font=dict(color="black")),
            showlegend=False,
        )

        text_trace = go.Scatter(
            x=x_positions,
            y=y_positions,
            mode="text",
            text=words,
            textposition="bottom right",
            textfont=dict(size=sizes, color=word_colors),
            hoverinfo="skip",  # disable hover on text itself
            hovertemplate=None,
            showlegend=False,
        )

        traces = [hover_trace, text_trace]
        if consensus_bar and lo is not None:
            traces.insert(
                0,
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="markers",
                    marker=dict(
                        colorscale=colorscale,
                        cmin=lo,
                        cmax=hi,
                        color=[lo],
                        showscale=True,
                        colorbar=dict(
                            thickness=20,
                            len=0.5,
                            x=1.01,
                            tickvals=[lo, avg, hi],
                            ticktext=[f"{lo:.2f}", f"{avg:.2f} (avg)", f"{hi:.2f}"],
                        ),
                    ),
                    hoverinfo="none",
                    showlegend=False,
                ),
            )
        fig = go.Figure(data=traces)
        fig.update_layout(
            showlegend=False,
            xaxis=dict(visible=False, range=[-1.1, 1.1], scaleanchor="y", scaleratio=1.0),
            yaxis=dict(visible=False, range=[-1.1, 1.1]),
            margin=dict(l=10, r=10, t=10, b=10),
            plot_bgcolor="white",
            paper_bgcolor="white",
        )

        return fig

    except Exception as e:
        print(f"Error building word cloud: {e}")
        import traceback

        traceback.print_exc()
        return go.Figure().add_annotation(
            text=f"Error: {str(e)}",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
        )


def register_callbacks():
    """Register callbacks for the word cloud feature."""

    _DETAILS_FREQUENCY = (
        "Each word represents a unique keyword or phrase found in the titles of selected resolutions. "
        "Word size and colour both reflect the number of resolutions in whose title the word appears. "
        "The data only covers GA resolutions that were successfully passed."
    )
    _DETAILS_CONSENSUS = (
        "Each word represents a unique keyword or phrase found in the titles of selected resolutions. "
        "Word size reflects the number of resolutions in whose title the word appears. "
        "Word colour reflects the average consensus score of the resolutions associated with each word — "
        "the consensus score of a resolution is the average pairwise vote agreement across all country "
        "pairs that both cast a vote. A score of 1 means all countries voted identically; "
        "lower values indicate more divided votes. "
        "Green words thus appear in resolutions with broad consensus; pink words in more divisive resolutions. "
        "The colour scale runs from the 1st to the 99th percentile of consensus scores across the selected "
        "resolutions, and its midpoint (amber) is anchored at the global consensus average. "
        "The data only covers GA resolutions that were successfully passed."
    )
    _DETAILS_STYLE = {
        "maxWidth": "100%",
        "margin": "0 0 0 0",
        "paddingLeft": "2%",
        "paddingTop": "10px",
        "color": "#7f8c8d",
        "fontSize": "16px",
        "lineHeight": "1.6",
        "textAlign": "left",
        "borderTop": "1px solid #eee",
    }

    @callback(
        [
            Output("wordcloud-interactive-chart", "figure"),
            Output("wordcloud-details", "children"),
        ],
        Input("filter-component-filter-store", "data"),
        Input("country-view-tabs", "value"),
        Input("wordcloud-mode-tabs", "value"),
        Input("wordcloud-color-mode", "value"),
    )
    def update_wordcloud_chart(filter_store, active_tab, selected_mode, color_mode):
        """Update word cloud and details caption when filters or colour mode change."""
        resolved_color_mode = color_mode or "frequency"
        details_text = (
            _DETAILS_CONSENSUS if resolved_color_mode == "consensus" else _DETAILS_FREQUENCY
        )
        details = html.P([html.Strong("Details: "), details_text], style=_DETAILS_STYLE)

        if not filter_store:
            return (
                go.Figure().add_annotation(
                    text="Loading data...",
                    x=0.5,
                    y=0.5,
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    font=dict(size=16, color="#7f8c8d"),
                ),
                details,
            )
        mode = selected_mode if selected_mode in _WORDCLOUD_MODES else _DEFAULT_MODE
        return (
            _build_wordcloud(
                filter_store,
                active_tab,
                mode=mode,
                color_mode=resolved_color_mode,
            ),
            details,
        )

    @callback(
        Output("wordcloud-interactive-meta", "children"),
        Input("filter-component-filter-store", "data"),
        Input("country-view-tabs", "value"),
        Input("wordcloud-mode-tabs", "value"),
    )
    def update_wc_meta(filter_store, active_tab, selected_mode):
        """Update meta information about word cloud."""
        if not filter_store:
            return "No data available."

        try:
            df = filtered_resolutions.resolutions_for(filter_store, active_tab)
            if df.empty:
                return "No data available."
            if "undl_id" not in df.columns:
                return "No data available."
            # The meta line is currently disabled, so the aggregation it displayed
            # (term frequencies over df["undl_id"], by mode) is skipped rather than computed
            # and discarded. Restore both together if the counts come back.
            return ""
            # return f"Total accepted resolutions: {len(df):,}"  # • Unique words: {len(word_freq)}
        except Exception as e:
            return f"Error: {str(e)}"

    @callback(
        Output("filter-component-keyword-search", "value", allow_duplicate=True),
        Output("filter-component-subject-dropdown", "value", allow_duplicate=True),
        Output("country-view-tabs", "value", allow_duplicate=True),
        Input("wordcloud-interactive-chart", "clickData"),
        State("wordcloud-mode-tabs", "value"),
        State("filter-component-keyword-search", "value"),
        State("filter-component-subject-dropdown", "value"),
        prevent_initial_call=True,
    )
    def click_word_to_search(clickData, selected_mode, current_keyword, current_subject_ids):
        """Append clicked word cloud word to keyword search and switch to Resolutions tab."""
        if not clickData or "points" not in clickData or not clickData["points"]:
            raise PreventUpdate
        pt = clickData["points"][0]
        custom = pt.get("customdata")
        word = pt.get("text") or (
            custom[1] if isinstance(custom, (list, tuple)) and len(custom) > 1 else None
        )
        if not word:
            raise PreventUpdate

        if selected_mode == "category":
            target_subject_ids = _index().category_term_to_subject_ids.get(word.lower(), [])
            if not target_subject_ids:
                return no_update, no_update, "resolution_list"
            # In category mode, set subject filter (not keyword search).
            return no_update, target_subject_ids, "resolution_list"

        # Use quoted phrase so downstream keyword filtering can apply exact mode.
        cleaned_word = word.replace('"', "").strip()
        exact_phrase = f'"{cleaned_word}"'
        existing_expression = (current_keyword or "").strip()
        existing_terms = {
            t.strip().lower()
            for clause in existing_expression.split(",")
            for t in clause.split("&")
            if t and t.strip()
        }
        if exact_phrase.lower() in existing_terms:
            new_keyword_value = existing_expression
        else:
            new_keyword_value = (
                f"{existing_expression} & {exact_phrase}" if existing_expression else exact_phrase
            )
        return (
            new_keyword_value,
            current_subject_ids if current_subject_ids is not None else no_update,
            "resolution_list",
        )

    @callback(
        Output("wordcloud-interactive-table", "children"),
        Input("wordcloud-interactive-chart", "hoverData"),
        Input("filter-component-filter-store", "data"),
        Input("country-view-tabs", "value"),
        Input("wordcloud-mode-tabs", "value"),
        prevent_initial_call=True,
    )
    def update_resolution_table(hoverData, filter_params, active_tab, selected_mode):
        """Update resolution list (cards) when hovering over a word."""
        if (
            hoverData is None
            or not hoverData
            or "points" not in hoverData
            or not hoverData["points"]
        ):
            return html.Div(
                "Hover over a word to see related resolutions.",
                style={"color": "#7f8c8d"},
            )

        if not filter_params:
            return html.Div("No data available.", style={"color": "#7f8c8d"})

        try:
            pt = hoverData["points"][0]
            custom = pt.get("customdata")
            word = pt.get("text") or (
                custom[1] if isinstance(custom, (list, tuple)) and len(custom) > 1 else None
            )
            if not word:
                return html.Div("No word selected.", style={"color": "#7f8c8d"})

            df = filtered_resolutions.resolutions_for(filter_params, active_tab)

            if df.empty:
                return html.Div("No data available.", style={"color": "#7f8c8d"})

            if "undl_id" not in df.columns:
                return html.Div("No data available.", style={"color": "#7f8c8d"})

            mode = selected_mode if selected_mode in _WORDCLOUD_MODES else _DEFAULT_MODE
            mode_word_map = _index().terms(mode)
            if word not in mode_word_map:
                return html.Div(
                    f"No resolutions found for word '{word}'.",
                    style={"color": "#7f8c8d"},
                )

            matching = df[df["undl_id"].isin(mode_word_map[word])].copy()
            if matching.empty:
                return html.Div(
                    f"No resolutions found for word '{word}' in current filter.",
                    style={"color": "#7f8c8d"},
                )

            matching["date"] = pd.to_datetime(matching["date"], errors="coerce")
            matching = matching.sort_values(by=["date", "resolution"], ascending=[True, True])
            max_rows = 10000
            display_df = matching.head(max_rows)

            # Country filters for vote indicators (same as resolution_list)
            country1 = (filter_params or {}).get("country1_alpha3")
            country2_raw = (filter_params or {}).get("country2")
            comparison_countries = []
            if isinstance(country2_raw, list):
                comparison_countries = country2_raw
            elif isinstance(country2_raw, str) and country2_raw:
                comparison_countries = [country2_raw]

            cards = []
            for _, row in display_df.iterrows():
                res_id = row.get("resolution", "N/A")
                link = row.get("undl_link", "#")
                date_val = row.get("date")
                date_str = date_val.strftime("%Y-%m-%d") if pd.notnull(date_val) else "Unknown"
                title = row.get("title", "Untitled")
                indicators = []
                if country1 and country1 in row:
                    indicators.append(
                        create_vote_indicator(data.get_country_name(country1), row.get(country1))
                    )
                for c2 in comparison_countries[:5]:
                    if c2 in row:
                        indicators.append(
                            create_vote_indicator(data.get_country_name(c2), row.get(c2))
                        )
                card = html.Div(
                    [
                        html.Div(
                            [
                                html.A(
                                    html.Span(
                                        f"{res_id}",
                                        style={
                                            "color": "#007bff",
                                            "fontWeight": "bold",
                                        },
                                    ),
                                    href=link,
                                    target="_blank",
                                    style={"textDecoration": "none"},
                                ),
                                html.Span(
                                    date_str,
                                    style={
                                        "float": "right",
                                        "color": "#666",
                                        "fontSize": "0.9em",
                                    },
                                ),
                            ],
                            style={"marginBottom": "0.5rem"},
                        ),
                        html.Div(title),
                        html.Div(
                            indicators,
                            style={
                                "marginTop": "10px",
                                "paddingTop": "10px",
                                "borderTop": "1px solid #eee",
                                "display": "flex",
                                "flexWrap": "wrap",
                                "gap": "5px",
                            },
                        )
                        if indicators
                        else None,
                    ],
                    className="resolution-card",
                )
                cards.append(card)

            summary = html.Div(
                f"{len(matching)} resolutions for word '{word}'"
                + (f" (showing first {max_rows})" if len(matching) > max_rows else ""),
                style={"fontWeight": "bold", "marginBottom": "8px"},
            )
            return html.Div([summary] + cards)

        except Exception as e:
            print(f"Error updating resolution table: {e}")
            import traceback

            traceback.print_exc()
            return html.Div(f"Error: {str(e)}", style={"color": "red"})


layout = (
    html.Div(
        [
            # Meta info
            html.Div(
                id="wordcloud-interactive-meta",
                style={
                    "textAlign": "center",
                    "fontWeight": "bold",
                    "marginBottom": "8px",
                },
            ),
            dcc.Tabs(
                id="wordcloud-mode-tabs",
                value=_DEFAULT_MODE,
                children=[
                    dcc.Tab(label="Default", value="default"),
                    dcc.Tab(label="Geopolitical", value="geopolitical"),
                    dcc.Tab(label="Thematic", value="thematic"),
                    dcc.Tab(label="Action", value="action"),
                    dcc.Tab(label="Subjects", value="category"),
                ],
            ),
            html.Div(
                [
                    html.Span(
                        "Colour by:",
                        style={"fontSize": "14px", "color": "#555", "marginRight": "10px"},
                    ),
                    dcc.RadioItems(
                        id="wordcloud-color-mode",
                        options=[
                            {"label": "Frequency", "value": "frequency"},
                            {"label": "Avg. consensus score", "value": "consensus"},
                        ],
                        value="frequency",
                        inline=True,
                        labelStyle={"marginRight": "16px", "fontSize": "14px"},
                    ),
                ],
                style={
                    "display": "flex",
                    "alignItems": "center",
                    "padding": "8px 4px",
                    "marginTop": "6px",
                },
            ),
            # html.Div(
            #     "Use the camera icon (top-right) to download PNG",
            #     style={
            #         "textAlign": "center",
            #         "fontSize": "13px",
            #         "color": "#6b7280",
            #         "marginBottom": "10px",
            #     },
            # ),
            # Word cloud chart
            dcc.Loading(
                children=[
                    dcc.Graph(
                        id="wordcloud-interactive-chart",
                        style={"height": "800px"},
                        config={
                            "displayModeBar": True,
                            "toImageButtonOptions": {
                                "format": "png",
                                "filename": "wordcloud",
                                "height": 1000,
                                "width": 1800,
                                "scale": 2,
                            },
                        },
                    )
                ],
                type="cube",
                color="#3498db",
            ),
            html.P(id="wordcloud-details"),
            # Resolution table (hidden — click-to-search now handles this via the Resolutions tab)
            html.Div(
                [
                    html.Label(
                        "Accepted resolutions for hovered word:",
                        style={
                            "fontWeight": "bold",
                            "marginBottom": "10px",
                            "fontSize": "14px",
                            "color": "#2c3e50",
                        },
                    ),
                    html.Div(
                        id="wordcloud-interactive-table",
                        style={
                            "backgroundColor": "#ffffff",
                            "padding": "12px",
                            "borderRadius": "8px",
                            "boxShadow": "0 2px 4px rgba(0,0,0,0.1)",
                            "maxHeight": "300px",
                            "overflowY": "auto",
                        },
                    ),
                ],
                style={
                    "display": "none",
                    "marginTop": "20px",
                    "padding": "20px",
                    "backgroundColor": "#f8f9fa",
                    "borderRadius": "8px",
                    "boxShadow": "0 2px 4px rgba(0,0,0,0.1)",
                },
            ),
        ]
    ),
)
