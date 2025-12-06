import os
from collections import Counter
from io import StringIO
from dash import Input, Output, callback, html, dcc
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import random
from matplotlib import cm as mpl_cm
from matplotlib import colors as mpl_colors
from wordcloud import WordCloud

from .. import data

# Initialize word cloud data on module load
_resolution_wc_data = {}
_wc_word_undlid_map = {}
_initialized = False


def _init_wc_data():
    """Initialize word cloud data from keywords CSV file."""
    global _resolution_wc_data, _wc_word_undlid_map, _initialized
    
    if _initialized:
        return
    
    print("Initializing word cloud data...")
    
    # Find the keywords CSV file in the app/assets directory
    # Get app directory (parent of features directory)
    app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    keywords_path = os.path.join(app_dir, "assets", "undlid_keywords.csv")
    
    if not os.path.exists(keywords_path):
        print(f"Warning: Keywords file not found at {keywords_path}")
        _initialized = True
        return
    
    try:
        keywords_df = pd.read_csv(keywords_path)
        data_all = pd.merge(
            data.query_engine.query_resolutions(),
            keywords_df,
            on='undl_id',
            how='left'
        )
        
        ignore_words = ["resolution", "general assembly"]
        
        resolution_wc_data = {}
        wc_word_undlid_map = {}
        
        for undl_id, keywords in data_all[['undl_id', 'keywords']].values:
            if pd.isna(keywords):
                continue
                
            word_in_resolution = Counter()
            tokens_set = set()
            
            for keyword in str(keywords).split(","):
                keyword_clean = keyword.strip().lower()
                if keyword_clean in ignore_words:
                    continue
                if keyword_clean:
                    tokens_set.add(keyword_clean)
            
            for token in tokens_set:
                word_in_resolution[token] += 1
                if token not in wc_word_undlid_map:
                    wc_word_undlid_map[token] = []
                wc_word_undlid_map[token].append(undl_id)
            
            resolution_wc_data[undl_id] = {
                'word_freq': dict(word_in_resolution)
            }
        
        _resolution_wc_data = resolution_wc_data
        _wc_word_undlid_map = wc_word_undlid_map
        _initialized = True
        
        print(f"✅ Word cloud data initialized: {len(resolution_wc_data)} resolutions, {len(wc_word_undlid_map)} unique words")
        
    except Exception as e:
        print(f"❌ Error initializing word cloud data: {e}")
        import traceback
        traceback.print_exc()
        _initialized = True


def _aggregate_word_freq(undl_ids: pd.Series) -> dict:
    """Combine word frequencies across given resolution IDs."""
    agg_counter = Counter()
    for undl_id in undl_ids.values:
        if undl_id in _resolution_wc_data:
            agg_counter.update(_resolution_wc_data[undl_id]['word_freq'])
    return dict(agg_counter)


def _aggregate_word_undlids_map(word_list: list[str], undl_ids: pd.Series) -> dict:
    """Map words to resolution IDs that contain them."""
    agg_map = {}
    for word in word_list:
        agg_map[word] = []
        if word in _wc_word_undlid_map:
            matching_ids = undl_ids[undl_ids.isin(_wc_word_undlid_map[word])].values
            agg_map[word].extend(matching_ids)
    return agg_map


def _get_wordcloud_layout(word_freq_dict, seed=42):
    """Generate word positions using wordcloud library to avoid overlaps"""
    if not word_freq_dict:
        return {}, {}, {}
    
    try:
        # Create WordCloud object with appropriate settings
        # Use a larger canvas with 2:1 aspect ratio (width:height)
        width, height = 1200, 800
        
        wordcloud = WordCloud(
            width=width,
            height=height,
            background_color='white',
            prefer_horizontal=1.0,
            relative_scaling=0.5,
            min_font_size=16,
            max_font_size=100,
            max_words=len(word_freq_dict),
            random_state=seed,
            collocation_threshold=0,
            scale=1
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
        
        if hasattr(wordcloud, 'layout_') and wordcloud.layout_:
            for item in wordcloud.layout_:
                if len(item) >= 4:
                    word, font_size, position, orientation = item[0], item[1], item[2], item[3]
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
    cmap = mpl_cm.get_cmap('Blues')
    freq_arr = np.array(frequencies, dtype=float)
    if len(freq_arr) == 0:
        return []
    if np.max(freq_arr) != np.min(freq_arr):
        # Normalize to 0.3-1.0 range instead of 0-1 to avoid very light/white colors
        # This keeps the blue theme but starts from a more visible blue
        normed = 0.3 + 0.7 * ((freq_arr - np.min(freq_arr)) / (np.max(freq_arr) - np.min(freq_arr)))
    else:
        # If all frequencies are the same, use a medium blue
        normed = np.full_like(freq_arr, 0.6)
    colors = [mpl_colors.rgb2hex(cmap(v)) for v in normed]
    return colors


def _build_wordcloud(filtered_data_json: str):
    """Build word cloud figure from filtered data."""
    if not filtered_data_json:
        return go.Figure().add_annotation(
            text="No data available. Please adjust filters.",
            x=0.5, y=0.5, xref="paper", yref="paper",
            showarrow=False
        )
    
    try:
        df = pd.read_json(StringIO(filtered_data_json), orient="split")
        if df.empty:
            return go.Figure().add_annotation(
                text="No resolutions match the current filters.",
                x=0.5, y=0.5, xref="paper", yref="paper",
                showarrow=False
            )
        
        # Check if 'undl_id' column exists
        if 'undl_id' not in df.columns:
            return go.Figure().add_annotation(
                text="No data available. Please adjust filters.",
                x=0.5, y=0.5, xref="paper", yref="paper",
                showarrow=False
            )
        
        # Get word frequencies for filtered resolutions
        word_freq = _aggregate_word_freq(df['undl_id'])
        
        if not word_freq:
            return go.Figure().add_annotation(
                text="No keywords found for the filtered resolutions.",
                x=0.5, y=0.5, xref="paper", yref="paper",
                showarrow=False
            )
        
        # Sort and limit to top 30 words
        sorted_word_freq = sorted(word_freq.items(), key=lambda x: (-x[1], x[0]))
        if len(sorted_word_freq) > 30:
            word_freq = dict(sorted_word_freq[:30])
        else:
            word_freq = dict(sorted_word_freq)
        
        words = list(word_freq.keys())
        freqs = list(word_freq.values())
        
        # Get word positions from wordcloud library
        seed = 42  # Deterministic seed based on words
        word_positions, sizes_from_wc, orientations_from_wc = _get_wordcloud_layout(word_freq, seed=seed)
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
            return go.Figure().add_annotation(
                text="No words positioned by WordCloud",
                x=0.5, y=0.5, xref="paper", yref="paper",
                showarrow=False
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
        
        for word in words:
            word_key = word
            x_pixel, y_pixel = word_positions[word_key]
            
            # Normalize coordinates to maintain 2:1 aspect ratio
            # Scale x to -1 to 1 range based on width
            # Scale y to -1 to 1 range based on height
            # This preserves the 2:1 aspect ratio
            x_normalized = (y_pixel / wc_width) * 3 - 1.5  # Scale to -1 to 1 range
            y_normalized = 1 - (x_pixel / wc_height) * 2  # Scale to -1 to 1, flip y-axis (wordcloud uses top-left origin)
            
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
        
        hover_text = [f"<b>{w}</b><br>Appears in {f} resolutions" for w, f in zip(words, freqs)]
        
        trace = go.Scatter(
            x=x_positions,
            y=y_positions,
            mode='text',
            text=words,
            textposition="bottom right",
            textfont=dict(size=sizes, color=colors),
            hovertemplate='%{customdata}<extra></extra>',
            customdata=hover_text,
        )
        
        fig = go.Figure(data=[trace])
        fig.update_layout(
            showlegend=False,
            xaxis=dict(visible=False, range=[-1.1, 1.1], scaleanchor="y", scaleratio=1.0),
            yaxis=dict(visible=False, range=[-1.1, 1.1]),
            margin=dict(l=10, r=10, t=10, b=10),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        return fig
        
    except Exception as e:
        print(f"Error building word cloud: {e}")
        import traceback
        traceback.print_exc()
        return go.Figure().add_annotation(
            text=f"Error: {str(e)}",
            x=0.5, y=0.5, xref="paper", yref="paper",
            showarrow=False
        )


def register_callbacks():
    """Register callbacks for the word cloud feature."""
    
    # Initialize data on first callback registration
    _init_wc_data()
    
    @callback(
        Output("wordcloud-interactive-chart", "figure"),
        Input("filter-component-data-store", "data"),
    )
    def update_wordcloud_chart(filtered_data):
        """Update word cloud when filter data changes."""
        if not filtered_data:
            raise PreventUpdate
        return _build_wordcloud(filtered_data)
    
    @callback(
        Output("wordcloud-interactive-meta", "children"),
        Input("filter-component-data-store", "data"),
    )
    def update_wc_meta(filtered_data):
        """Update meta information about word cloud."""
        if not filtered_data:
            return "No data available."
        
        try:
            df = pd.read_json(StringIO(filtered_data), orient="split")
            if df.empty:
                return "No data available."
            if 'undl_id' not in df.columns:
                return "No data available."
            word_freq = _aggregate_word_freq(df['undl_id'])
            return f"Total resolutions: {len(df):,}"    # • Unique words: {len(word_freq)}
        except Exception as e:
            return f"Error: {str(e)}"
    
    @callback(
        Output("wordcloud-interactive-table", "children"),
        Input("wordcloud-interactive-chart", "hoverData"),
        Input("filter-component-data-store", "data"),
        prevent_initial_call=True,
    )
    def update_resolution_table(hoverData, filtered_data):
        """Update resolution table when hovering over a word."""
        if hoverData is None or not hoverData or 'points' not in hoverData or not hoverData['points']:
            return html.Div("Hover over a word to see related resolutions.", style={'color': '#7f8c8d'})
        
        if not filtered_data:
            return html.Div("No data available.", style={'color': '#7f8c8d'})
        
        try:
            word = hoverData['points'][0].get('text')
            if not word:
                return html.Div("No word selected.", style={'color': '#7f8c8d'})
            
            df = pd.read_json(StringIO(filtered_data), orient="split")
            
            if df.empty:
                return html.Div("No data available.", style={'color': '#7f8c8d'})
            
            if 'undl_id' not in df.columns:
                return html.Div("No data available.", style={'color': '#7f8c8d'})
            
            # Get resolution IDs that contain this word
            if word not in _wc_word_undlid_map:
                return html.Div(f"No resolutions found for word '{word}'.", style={'color': '#7f8c8d'})
            
            matching_ids = df[df['undl_id'].isin(_wc_word_undlid_map[word])]
            
            if matching_ids.empty:
                return html.Div(f"No resolutions found for word '{word}' in current filter.", style={'color': '#7f8c8d'})
            
            # Sort by date then resolution
            matching_ids = matching_ids[['resolution', 'date', 'title']].copy()
            matching_ids['date'] = pd.to_datetime(matching_ids['date'], errors='coerce')
            matching_ids = matching_ids.sort_values(by=['date', 'resolution'], ascending=[True, True])
            
            # Limit number of rows
            max_rows = 10000
            df_limited = matching_ids.head(max_rows)
            
            header = html.Tr([
                html.Th("Resolution", style={'padding': '6px', 'textAlign': 'left', 'borderBottom': '2px solid #3498db'}),
                html.Th("Date", style={'padding': '6px', 'textAlign': 'left', 'borderBottom': '2px solid #3498db'}),
                html.Th("Title", style={'padding': '6px', 'textAlign': 'left', 'borderBottom': '2px solid #3498db'})
            ])
            
            rows = [
                html.Tr([
                    html.Td(str(r['resolution']), style={'padding': '6px', 'verticalAlign': 'top'}),
                    html.Td((r['date'].date().isoformat() if pd.notna(r['date']) else ''), style={'padding': '6px', 'verticalAlign': 'top'}),
                    html.Td(str(r['title']), style={'padding': '6px'})
                ]) for _, r in df_limited.iterrows()
            ]
            
            summary = html.Div(
                f"{len(matching_ids)} resolutions for word '{word}'" + 
                (f" (showing first {max_rows})" if len(matching_ids) > max_rows else ""),
                style={'fontWeight': 'bold', 'marginBottom': '8px'}
            )
            table = html.Table([header] + rows, style={'width': '100%', 'borderCollapse': 'collapse'})
            return html.Div([summary, table])
            
        except Exception as e:
            print(f"Error updating resolution table: {e}")
            import traceback
            traceback.print_exc()
            return html.Div(f"Error: {str(e)}", style={'color': 'red'})


layout = (
    html.Div(
        [
            # Meta info
            html.Div(
                id="wordcloud-interactive-meta",
                style={'textAlign': 'center', 'fontWeight': 'bold', 'marginBottom': '8px'}
            ),
            # Word cloud chart
            dcc.Loading(
                children=[
                    dcc.Graph(
                        id="wordcloud-interactive-chart",
                        style={'height': '800px'},
                        config={'displayModeBar': False}
                    )
                ],
                type="cube",
                color="#3498db"
            ),
            # Resolution table
            html.Div(
                [
                    html.Label(
                        "Resolutions for hovered word:",
                        style={
                            'fontWeight': 'bold',
                            'marginBottom': '10px',
                            'fontSize': '14px',
                            'color': '#2c3e50'
                        }
                    ),
                    html.Div(
                        id="wordcloud-interactive-table",
                        style={
                            'backgroundColor': '#ffffff',
                            'padding': '12px',
                            'borderRadius': '8px',
                            'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
                            'maxHeight': '300px',
                            'overflowY': 'auto'
                        }
                    )
                ],
                style={
                    'marginTop': '20px',
                    'padding': '20px',
                    'backgroundColor': '#f8f9fa',
                    'borderRadius': '8px',
                    'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
                }
            )
        ]
    ),
)

