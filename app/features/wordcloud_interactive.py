import os
import re
from collections import Counter
from io import StringIO
from dash import Input, Output, State, callback, html, dcc, no_update
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import random
from matplotlib import cm as mpl_cm
from matplotlib import colors as mpl_colors
from wordcloud import WordCloud
from rapidfuzz import process as fuzz_process

from .. import data
from .resolution_list import create_vote_indicator

# Initialize word cloud data on module load
_resolution_wc_data_by_mode = {}
_wc_word_undlid_map_by_mode = {}
_category_term_to_subject_ids = {}
_initialized = False
_DEFAULT_MODE = "default"
_MAX_WORDS_RENDER = 20
_MAX_WORD_CANDIDATES_FOR_SEARCH_COUNT = 30
_WORDCLOUD_MODES = {
    "default": {"label": "Default", "source": "undlid_keywords.csv:keywords"},
    "geopolitical": {"label": "Geopolitical", "source": "undlid_keywords_3d_noun_fixed.csv:Geopolitical"},
    "thematic": {"label": "Thematic", "source": "undlid_keywords_3d_noun_fixed.csv:Thematic"},
    "action": {"label": "Action", "source": "undlid_keywords_3d_noun_fixed.csv:Action"},
    "category": {"label": "Subjects", "source": "query_resolutions():subjects"},
}


def _init_wc_data():
    """Initialize word cloud data from keywords CSV file."""
    global _resolution_wc_data_by_mode, _wc_word_undlid_map_by_mode, _category_term_to_subject_ids, _initialized

    if _initialized:
        return

    print("Initializing word cloud data...")

    # Find the keywords CSV file in the app/assets directory
    # Get app directory (parent of features directory)
    app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        base_keywords_path = os.path.join(app_dir, "assets", "undlid_keywords.csv")
        three_d_keywords_path = os.path.join(
            app_dir, "assets", "undlid_keywords_3d_noun_fixed.csv"
        )
        resolutions_df = data.query_engine.query_resolutions()
        ignore_words = ["resolution", "general assembly"]
        split_pattern_general = re.compile(r"[;,]")
        # Subject/category values may be delimited by "|", "--", or ";".
        split_pattern_subject = re.compile(r"\||--|;")

        def build_indices(
            data_all: pd.DataFrame,
            keyword_col: str,
            split_pattern: re.Pattern,
            ignore_terms: list[str] | None = None,
        ):
            resolution_wc_data = {}
            wc_word_undlid_map = {}
            ignore_set = set(ignore_terms or [])

            for undl_id, keywords in data_all[["undl_id", keyword_col]].values:
                if pd.isna(keywords):
                    continue

                word_in_resolution = Counter()
                tokens_set = set()

                for keyword in split_pattern.split(str(keywords)):
                    keyword_clean = re.sub(r"\s+", " ", keyword.strip().lower())
                    if keyword_clean in ignore_set:
                        continue
                    if keyword_clean:
                        tokens_set.add(keyword_clean)

                for token in tokens_set:
                    word_in_resolution[token] += 1
                    if token not in wc_word_undlid_map:
                        wc_word_undlid_map[token] = []
                    wc_word_undlid_map[token].append(undl_id)

                resolution_wc_data[undl_id] = {"word_freq": dict(word_in_resolution)}

            return resolution_wc_data, wc_word_undlid_map

        resolution_wc_data_by_mode = {}
        wc_word_undlid_map_by_mode = {}
        category_term_to_subject_ids = {}

        if os.path.exists(base_keywords_path):
            base_keywords_df = pd.read_csv(base_keywords_path)
            base_data_all = pd.merge(
                resolutions_df, base_keywords_df, on="undl_id", how="left"
            )
            (
                resolution_wc_data_by_mode["default"],
                wc_word_undlid_map_by_mode["default"],
            ) = build_indices(
                base_data_all,
                "keywords",
                split_pattern_general,
                ignore_words,
            )
        else:
            print(f"Warning: Keywords file not found at {base_keywords_path}")
            resolution_wc_data_by_mode["default"] = {}
            wc_word_undlid_map_by_mode["default"] = {}

        if os.path.exists(three_d_keywords_path):
            three_d_df = pd.read_csv(three_d_keywords_path)
            three_d_df = three_d_df.rename(columns={"Original_ID": "undl_id"})
            three_d_data_all = pd.merge(
                resolutions_df, three_d_df, on="undl_id", how="left"
            )
            for mode_key, column_name in [
                ("geopolitical", "Geopolitical"),
                ("thematic", "Thematic"),
                ("action", "Action"),
            ]:
                if column_name not in three_d_data_all.columns:
                    print(f"Warning: Column '{column_name}' not found in 3D keywords CSV.")
                    resolution_wc_data_by_mode[mode_key] = {}
                    wc_word_undlid_map_by_mode[mode_key] = {}
                    continue
                (
                    resolution_wc_data_by_mode[mode_key],
                    wc_word_undlid_map_by_mode[mode_key],
                ) = build_indices(
                    three_d_data_all,
                    column_name,
                    split_pattern_general,
                    ignore_words,
                )
        else:
            print(f"Warning: 3D keywords file not found at {three_d_keywords_path}")
            for mode_key in ["geopolitical", "thematic", "action"]:
                resolution_wc_data_by_mode[mode_key] = {}
                wc_word_undlid_map_by_mode[mode_key] = {}

        resolution_subject_df = getattr(
            data.query_engine, "resolution_subject_table", pd.DataFrame()
        ).copy()
        subject_df = getattr(data.query_engine, "subject_table", pd.DataFrame()).copy()
        if (
            not resolution_subject_df.empty
            and "undl_id" in resolution_subject_df.columns
            and "subject_id" in resolution_subject_df.columns
            and not subject_df.empty
            and "subject_id" in subject_df.columns
        ):
            subject_label_col = (
                "label_en" if "label_en" in subject_df.columns else None
            )
            if subject_label_col is None:
                print("Warning: subject_table has no 'label_en' column; using subject_id.")
                subject_df["label_en"] = subject_df["subject_id"]
                subject_label_col = "label_en"

            subject_lookup = (
                subject_df[["subject_id", subject_label_col]]
                .dropna(subset=["subject_id"])
                .drop_duplicates(subset=["subject_id"])
                .set_index("subject_id")[subject_label_col]
                .to_dict()
            )
            for subject_id, label in subject_lookup.items():
                key = re.sub(r"\s+", " ", str(label).strip().lower())
                if key:
                    category_term_to_subject_ids.setdefault(key, []).append(subject_id)
            resolution_subject_df["category_term"] = resolution_subject_df[
                "subject_id"
            ].map(subject_lookup)
            resolution_subject_df["category_term"] = (
                resolution_subject_df["category_term"]
                .fillna(resolution_subject_df["subject_id"])
                .astype(str)
            )

            category_terms_by_resolution = (
                resolution_subject_df.groupby("undl_id")["category_term"]
                .apply(lambda s: "; ".join(sorted(set(s))))
                .reset_index(name="category_terms")
            )
            category_data_all = pd.merge(
                resolutions_df,
                category_terms_by_resolution,
                on="undl_id",
                how="left",
            )

            (
                resolution_wc_data_by_mode["category"],
                wc_word_undlid_map_by_mode["category"],
            ) = build_indices(
                category_data_all,
                "category_terms",
                split_pattern_subject,
                [],
            )
        else:
            print(
                "Warning: category word cloud data unavailable "
                "(need resolution_subject_table and subject_table)."
            )
            resolution_wc_data_by_mode["category"] = {}
            wc_word_undlid_map_by_mode["category"] = {}

        _resolution_wc_data_by_mode = resolution_wc_data_by_mode
        _wc_word_undlid_map_by_mode = wc_word_undlid_map_by_mode
        _category_term_to_subject_ids = category_term_to_subject_ids
        _initialized = True

        for mode_key in _WORDCLOUD_MODES:
            mode_words = len(_wc_word_undlid_map_by_mode.get(mode_key, {}))
            mode_resolutions = len(_resolution_wc_data_by_mode.get(mode_key, {}))
            print(
                f"✅ Word cloud mode '{mode_key}' initialized: {mode_resolutions} resolutions, {mode_words} unique words"
            )

    except Exception as e:
        print(f"❌ Error initializing word cloud data: {e}")
        import traceback

        traceback.print_exc()
        _initialized = True


def _aggregate_word_freq(undl_ids: pd.Series, mode: str = _DEFAULT_MODE) -> dict:
    """Combine word frequencies across given resolution IDs."""
    agg_counter = Counter()
    wc_data = _resolution_wc_data_by_mode.get(mode, {})
    for undl_id in undl_ids.values:
        if undl_id in wc_data:
            agg_counter.update(wc_data[undl_id]["word_freq"])
    return dict(agg_counter)


def _aggregate_word_undlids_map(
    word_list: list[str], undl_ids: pd.Series, mode: str = _DEFAULT_MODE
) -> dict:
    """Map words to resolution IDs that contain them."""
    agg_map = {}
    word_map = _wc_word_undlid_map_by_mode.get(mode, {})
    for word in word_list:
        agg_map[word] = []
        if word in word_map:
            matching_ids = undl_ids[undl_ids.isin(word_map[word])].values
            agg_map[word].extend(matching_ids)
    return agg_map


def search_keywords(
    token: str,
    score_cutoff: int = 80,
    mode: str = _DEFAULT_MODE,
    exact: bool = False,
) -> set:
    """Return the set of undl_ids whose keywords match token (exact substring + fuzzy).

    Requires _init_wc_data() to have been called first.
    """
    if not _initialized:
        _init_wc_data()

    matched_ids: set = set()
    token_lower = token.lower().strip()
    if not token_lower:
        return matched_ids

    word_map = _wc_word_undlid_map_by_mode.get(mode, {})
    all_keys = list(word_map.keys())

    if exact:
        if token_lower in word_map:
            matched_ids.update(word_map[token_lower])
        return matched_ids

    # Exact / substring matches first
    exact_matches = [k for k in all_keys if token_lower in k]
    for k in exact_matches:
        matched_ids.update(word_map[k])

    # Fuzzy matches
    if all_keys:
        fuzzy_results = fuzz_process.extract(
            token_lower, all_keys, score_cutoff=score_cutoff, limit=20
        )
        for match_key, _score, _idx in fuzzy_results:
            matched_ids.update(word_map[match_key])

    return matched_ids


def _parse_keyword_term(term: str) -> tuple[str, bool]:
    term = term.strip()
    exact_mode = len(term) >= 2 and term[0] == '"' and term[-1] == '"'
    parsed = term[1:-1].strip() if exact_mode else term
    return parsed, exact_mode


def get_keyword_matched_ids(
    df: pd.DataFrame, keyword_expression: str | None
) -> set:
    """
    Resolve keyword expression to matched undl_id set.
    Semantics:
    - ',' => OR between clauses
    - '&' => AND within a clause
    - quoted terms => exact keyword index match
    """
    if (
        df.empty
        or "undl_id" not in df.columns
        or not keyword_expression
        or not str(keyword_expression).strip()
    ):
        return set()

    matched_ids: set = set()
    clauses = [c.strip() for c in str(keyword_expression).split(",") if c.strip()]

    for clause in clauses:
        terms = [t.strip() for t in clause.split("&") if t.strip()]
        if not terms:
            continue
        clause_ids: set | None = None
        for term in terms:
            parsed_token, exact_mode = _parse_keyword_term(term)
            if not parsed_token:
                continue
            token_ids: set = set()
            title_match = df["title"].str.lower().str.contains(
                parsed_token.lower(), regex=False, na=False
            )
            token_ids |= set(df.loc[title_match, "undl_id"].tolist())
            token_ids |= search_keywords(parsed_token, exact=exact_mode)
            if clause_ids is None:
                clause_ids = token_ids
            else:
                clause_ids &= token_ids
            if not clause_ids:
                break
        if clause_ids:
            matched_ids |= clause_ids

    return matched_ids


def _get_wordcloud_layout(word_freq_dict, seed=42):
    """Generate word positions using wordcloud library to avoid overlaps"""
    if not word_freq_dict:
        return {}, {}, {}

    try:
        # Create WordCloud object with appropriate settings
        # Use a larger canvas with 2:1 aspect ratio (width:height)
        # width, height = 1600, 900
        width, height = 1200, 800

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


def _get_viridis_colors(frequencies):
    """Get color mapping for word frequencies using blue colormap."""
    # Use blue color scheme - 'Blues' for light to dark
    cmap = mpl_cm.get_cmap("Blues")
    freq_arr = np.array(frequencies, dtype=float)
    if len(freq_arr) == 0:
        return []
    if np.max(freq_arr) != np.min(freq_arr):
        # Normalize to 0.3-1.0 range instead of 0-1 to avoid very light/white colors
        # This keeps the blue theme but starts from a more visible blue
        normed = 0.3 + 0.7 * (
            (freq_arr - np.min(freq_arr)) / (np.max(freq_arr) - np.min(freq_arr))
        )
    else:
        # If all frequencies are the same, use a medium blue
        normed = np.full_like(freq_arr, 0.6)
    colors = [mpl_colors.rgb2hex(cmap(v)) for v in normed]
    return colors


def _apply_country_filter(df: pd.DataFrame, filter_store: dict | None) -> pd.DataFrame:
    """Apply main-country voted filter, matching filter-component data-store logic."""
    if df.empty:
        return df
    country1 = (filter_store or {}).get("country1_alpha3")
    if country1 and country1 in df.columns:
        return df.dropna(subset=[country1])
    return df


def _apply_keyword_filter(df: pd.DataFrame, filter_store: dict | None) -> pd.DataFrame:
    """Apply keyword filter with the same token semantics as resolution list/trends."""
    if df.empty or "undl_id" not in df.columns:
        return df
    keyword = (filter_store or {}).get("keyword")
    if not keyword or not str(keyword).strip():
        return df

    matched_ids = get_keyword_matched_ids(df, str(keyword))
    if not matched_ids:
        return df.iloc[0:0].copy()

    matched_ids_str = {str(x) for x in matched_ids}
    return df[df["undl_id"].astype(str).isin(matched_ids_str)].copy()


def _count_click_search_results(
    word: str,
    mode: str,
    filter_store: dict | None,
    filtered_df: pd.DataFrame,
) -> int:
    """
    Count resolutions that would be returned after clicking a word,
    aligned with Resolution tab search/filter behavior.
    """
    if not word:
        return 0

    if mode == "category":
        # Fast path: count within current filtered result set to avoid
        # per-word query_resolutions calls that can make rendering too slow.
        mode_word_map = _wc_word_undlid_map_by_mode.get(mode, {})
        candidate_ids = mode_word_map.get(word, [])
        if not candidate_ids or filtered_df.empty or "undl_id" not in filtered_df.columns:
            return 0
        filtered_ids = set(filtered_df["undl_id"].astype(str).tolist())
        return int(sum(1 for x in candidate_ids if str(x) in filtered_ids))

    if filtered_df.empty or "undl_id" not in filtered_df.columns:
        return 0

    cleaned_word = word.replace('"', "").strip()
    clicked_phrase = f'"{cleaned_word}"'
    existing_keyword = ((filter_store or {}).get("keyword") or "").strip()
    candidate_expression = (
        f"{existing_keyword} & {clicked_phrase}" if existing_keyword else clicked_phrase
    )
    matched_ids = get_keyword_matched_ids(filtered_df, candidate_expression)

    if not matched_ids:
        return 0

    matched_ids_str = {str(x) for x in matched_ids}
    return int(filtered_df["undl_id"].astype(str).isin(matched_ids_str).sum())


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
    filtered_data_json: str,
    mode: str = _DEFAULT_MODE,
    filter_store: dict | None = None,
):
    """Build word cloud figure from filtered data."""
    mode_label = _get_mode_label(mode)
    if not filtered_data_json:
        return _build_empty_wordcloud_figure(
            _mode_empty_message(
                mode,
                f"No data available for {mode_label}. Please adjust filters.",
            )
        )

    try:
        if not _wc_word_undlid_map_by_mode.get(mode, {}):
            return _build_empty_wordcloud_figure(
                _mode_empty_message(
                    mode,
                    f"No {mode_label.lower()} data source is available.",
                )
            )

        df = pd.read_json(StringIO(filtered_data_json), orient="split")
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

        # Get word frequencies for filtered resolutions
        word_freq = _aggregate_word_freq(df["undl_id"], mode=mode)

        if not word_freq:
            return _build_empty_wordcloud_figure(
                _mode_empty_message(
                    mode,
                    f"No {mode_label.lower()} terms found for filtered resolutions.",
                )
            )

        # Re-weight words by searchable result count so visual size reflects
        # "how many resolutions can be found by clicking this word".
        # To keep UI responsive, compute search-counts only for top raw-frequency candidates.
        raw_sorted_items = sorted(word_freq.items(), key=lambda x: (-x[1], x[0]))
        candidate_words = [
            w for w, _ in raw_sorted_items[:_MAX_WORD_CANDIDATES_FOR_SEARCH_COUNT]
        ]
        search_count_by_word = {
            w: _count_click_search_results(w, mode, filter_store, df)
            for w in candidate_words
        }
        weighted_items = [
            (w, c) for w, c in search_count_by_word.items() if c > 0
        ]
        if not weighted_items:
            return _build_empty_wordcloud_figure(
                _mode_empty_message(
                    mode,
                    f"No searchable {mode_label.lower()} terms for current filters.",
                )
            )
        # Sort and limit to top words by searchable count.
        weighted_items = sorted(weighted_items, key=lambda x: (-x[1], x[0]))
        word_freq = dict(weighted_items[:_MAX_WORDS_RENDER])

        words = list(word_freq.keys())
        freqs = list(word_freq.values())

        # Get word positions from wordcloud library
        seed = 42  # Deterministic seed based on words
        word_positions, sizes_from_wc, orientations_from_wc = _get_wordcloud_layout(
            word_freq, seed=seed
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
            print(f"Warning: No words were positioned by WordCloud")
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
        wc_width, wc_height = 1200, 800
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
            wc_font_size = sizes_from_wc[word_key]
            # Scale font size proportionally to maintain visual consistency
            # Increase the scaling factor to make words larger
            plotly_size = max(16, min(80, int(wc_font_size)))
            sizes.append(plotly_size)
        # print(f"words: {words}")
        # print(f"freqs: {freqs}")
        # print(f"x_positions: {x_positions}")
        # print(f"y_positions: {y_positions}")
        # print(f"sizes: {sizes}")

        # Colors based on frequency
        colors = _get_viridis_colors(freqs)

        click_result_counts = [search_count_by_word.get(word, 0) for word in words]
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
            hover_y_positions.append(
                y_pos - shift_y_normalized
            )  # + because y increases upward

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
            showlegend=False,
        )

        text_trace = go.Scatter(
            x=x_positions,
            y=y_positions,
            mode="text",
            text=words,
            textposition="bottom right",
            textfont=dict(size=sizes, color=colors),
            hoverinfo="skip",  # disable hover on text itself
            hovertemplate=None,
            showlegend=False,
        )

        fig = go.Figure(data=[hover_trace, text_trace])
        fig.update_layout(
            showlegend=False,
            xaxis=dict(
                visible=False, range=[-1.1, 1.1], scaleanchor="y", scaleratio=1.0
            ),
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

    # Initialize data on first callback registration
    _init_wc_data()

    @callback(
        Output("wordcloud-interactive-chart", "figure"),
        Input("filter-component-data-store", "data"),
        Input("wordcloud-mode-tabs", "value"),
        Input("filter-component-filter-store", "data"),
    )
    def update_wordcloud_chart(filtered_data, selected_mode, filter_store):
        """Update word cloud when filter data changes."""
        if not filtered_data:
            return go.Figure().add_annotation(
                text="Loading data...",
                x=0.5,
                y=0.5,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=16, color="#7f8c8d"),
            )
        mode = selected_mode if selected_mode in _WORDCLOUD_MODES else _DEFAULT_MODE
        return _build_wordcloud(filtered_data, mode=mode, filter_store=filter_store)

    @callback(
        Output("wordcloud-interactive-meta", "children"),
        Input("filter-component-data-store", "data"),
        Input("wordcloud-mode-tabs", "value"),
    )
    def update_wc_meta(filtered_data, selected_mode):
        """Update meta information about word cloud."""
        if not filtered_data:
            return "No data available."

        try:
            df = pd.read_json(StringIO(filtered_data), orient="split")
            if df.empty:
                return "No data available."
            if "undl_id" not in df.columns:
                return "No data available."
            mode = selected_mode if selected_mode in _WORDCLOUD_MODES else _DEFAULT_MODE
            word_freq = _aggregate_word_freq(df["undl_id"], mode=mode)
            return f""  # • Unique words: {len(word_freq)}
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
            custom[1]
            if isinstance(custom, (list, tuple)) and len(custom) > 1
            else None
        )
        if not word:
            raise PreventUpdate

        if selected_mode == "category":
            target_subject_ids = _category_term_to_subject_ids.get(word.lower(), [])
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
                f"{existing_expression} & {exact_phrase}"
                if existing_expression
                else exact_phrase
            )
        return new_keyword_value, current_subject_ids if current_subject_ids is not None else no_update, "resolution_list"

    @callback(
        Output("wordcloud-interactive-table", "children"),
        Input("wordcloud-interactive-chart", "hoverData"),
        Input("filter-component-data-store", "data"),
        Input("filter-component-filter-store", "data"),
        Input("wordcloud-mode-tabs", "value"),
        prevent_initial_call=True,
    )
    def update_resolution_table(hoverData, filtered_data, filter_params, selected_mode):
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

        if not filtered_data:
            return html.Div("No data available.", style={"color": "#7f8c8d"})

        try:
            pt = hoverData["points"][0]
            custom = pt.get("customdata")
            word = pt.get("text") or (
                custom[1]
                if isinstance(custom, (list, tuple)) and len(custom) > 1
                else None
            )
            if not word:
                return html.Div("No word selected.", style={"color": "#7f8c8d"})

            df = pd.read_json(StringIO(filtered_data), orient="split")

            if df.empty:
                return html.Div("No data available.", style={"color": "#7f8c8d"})

            if "undl_id" not in df.columns:
                return html.Div("No data available.", style={"color": "#7f8c8d"})

            mode = selected_mode if selected_mode in _WORDCLOUD_MODES else _DEFAULT_MODE
            mode_word_map = _wc_word_undlid_map_by_mode.get(mode, {})
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
            matching = matching.sort_values(
                by=["date", "resolution"], ascending=[True, True]
            )
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
                date_str = (
                    date_val.strftime("%Y-%m-%d") if pd.notnull(date_val) else "Unknown"
                )
                title = row.get("title", "Untitled")
                indicators = []
                if country1 and country1 in row:
                    indicators.append(
                        create_vote_indicator(
                            data.get_country_name(country1), row.get(country1)
                        )
                    )
                for c2 in comparison_countries[:5]:
                    if c2 in row:
                        indicators.append(
                            create_vote_indicator(
                                data.get_country_name(c2), row.get(c2)
                            )
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


# TODO: Add an annotation (explanatory caption) similar to the map and timeline tabs
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
