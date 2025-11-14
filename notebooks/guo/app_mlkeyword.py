import dash
from dash import dcc, html, Input, Output, callback
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go
import pandas as pd
from functools import lru_cache
import time
from datetime import datetime
from collections import Counter
import numpy as np
import random
from plotly import colors as plotly_colors

# Global variables (in production, you'd load this properly)
df_global = None  # Your dataframe goes here
available_countries = []

def get_time_string() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def simp_normalize_texts(texts: pd.Series) -> pd.Series:
    return texts.fillna('').str.lower()\
        .str.replace(r'\s+', ' ', regex=True) \
        .str.strip() 

def remove_explicitRemove_strs(texts: pd.Series, strs: list[str], debug = False) -> pd.Series:
    texts = simp_normalize_texts(texts)
    for idx, str_ in enumerate(strs):
        contains_keywords = texts.str.contains(str_.lower())
        num_contains_keywords = contains_keywords.sum()
        if debug:
            print(f'# with "{str_}": {num_contains_keywords}')

        texts = simp_normalize_texts(texts.str.replace(str_, '', case=False, regex=True))
        if debug and idx == len(strs) - 1:
            texts.to_csv(f'texts_clean_{idx}_{get_time_string()}.csv', index=False)
    return texts

class DashWordCloudApp:
    
    def __init__(self, query_engine, analyzer):
        self.query_engine = query_engine
        self.analyzer = analyzer
        self.app = dash.Dash(__name__)

        start_time = time.time()    
        self.init_wc_data()
        end_time = time.time()
        print(f"Time taken to initialize word cloud data: {end_time - start_time} seconds")

        # word_freq_dict = self.query_wc_data(2025, 2025)
        # for word, freq in word_freq_dict.items():
        #     print(f"{word}: {freq}")
        # return
        self.setup_layout()
        self.setup_callbacks()

        self.current_word_resolution_map = {}

    def aggregate_word_freq(self, undl_ids : pd.Series) -> dict:
        """Combine word frequencies across years start_year to end_year (inclusive)"""
        agg_counter = Counter()
        for undl_id in undl_ids.values:
            if undl_id in self.resolution_wc_data:
                agg_counter.update(self.resolution_wc_data[undl_id]['word_freq'])
        return dict(agg_counter)
    
    def aggregate_word_undlids_map(self, word_list : list[str], undl_ids : pd.Series) -> dict:
        agg_map = {}
        for word in word_list:
            agg_map[word] = []
            agg_map[word].extend(undl_ids[undl_ids.isin(self.wc_word_undlid_map[word])].values)
        return agg_map

    def filter_data_by_undlid(self, data : pd.DataFrame, undl_ids : pd.Series) -> pd.DataFrame:
        return data[data['undl_id'].isin(undl_ids)]
    
    def query_data(self, start_year, end_year, subject_ids : list[str] = None, include_descendants : bool = True) -> pd.DataFrame:
        data = self.query_engine.query_resolutions(
            start_date=f'{start_year}-01-01',
            end_date=f'{end_year}-12-31',
            subject_ids=subject_ids,
            include_descendants=include_descendants
        )
        return data[['undl_id', 'resolution', 'date', 'title']]
    
    def query_wc_data(self, start_year, end_year, subject_ids : list[str] = None, include_descendants : bool = True) -> dict:
        data = self.query_data(start_year, end_year, subject_ids, include_descendants)
        word_freq_dict = self.aggregate_word_freq(data['undl_id'])
        sorted_word_freq_dict = sorted(word_freq_dict.items(), key=lambda x: (-x[1], x[0]))
        if len(sorted_word_freq_dict) > 30:
            word_freq_dict = dict(sorted_word_freq_dict[:30])
        else:
            word_freq_dict = dict(sorted_word_freq_dict)
        word_list = list(word_freq_dict.keys())
        word_undlids_map = self.aggregate_word_undlids_map(word_list, data['undl_id'])
        return word_freq_dict, word_undlids_map, len(data)

    # def query_resolution_name(self, )

    def scatter_wordcloud(self, words, sizes, seed=42):
        """Generate (x, y) positions with minimal overlap for plotly scatter"""
        np.random.seed(seed)
        x_positions, y_positions = [], []
        for i in range(len(words)):
            attempts = 0
            while attempts < 100:
                x = random.uniform(-50, 50)
                y = random.uniform(-30, 30)
                overlap = False
                for j in range(len(x_positions)):
                    distance = np.sqrt((x - x_positions[j]) ** 2 + (y - y_positions[j]) ** 2)
                    if distance < (sizes[i] + sizes[j]) / 4:
                        overlap = True
                        break
                if not overlap or attempts >= 50:
                    x_positions.append(x)
                    y_positions.append(y)
                    break
                attempts += 1
        return x_positions, y_positions

    def _get_viridis_colors(self, frequencies):
        # if not frequencies:
        #     print(f"ckp0.0")
        #     return []
        # print(f"ckp0.1")
        # f_min = min(frequencies)
        # print(f"ckp0.2")
        # f_max = max(frequencies)
        # print(f"ckp0.3")
        # print(f"f_min: {f_min}, f_max: {f_max}")
        # if f_max == f_min:
        #     scaled = [0.5] * len(frequencies)
        # else:
        #     scaled = [(f - f_min) / (f_max - f_min) for f in frequencies]
        # print(f"ckp0.1")
        # viridis = plotly_colors.PLOTLY_SCALES.get('Viridis', plotly_colors.sequential.Viridis)
        # # viridis = plotly_colors.PLOTLY_SCALES.get('Sunset', plotly_colors.sequential.Sunset)
        # print(f"ckp0.2")
        # tmp = [plotly_colors.find_intermediate_color(viridis[0][1], viridis[-1][1], s, colortype='rgb') for s in scaled]
        # print(f"ckp0.3")
        # print(f"tmp: {tmp}")
        # return tmp

        from matplotlib import cm as mpl_cm
        from matplotlib import colors as mpl_colors
        cmap = mpl_cm.get_cmap('copper_r')
        # cmap = mpl_cm.get_cmap('viridis_r')
        freq_arr = np.array(frequencies, dtype=float)
        if len(freq_arr) == 0:
            return []
        if np.max(freq_arr) != np.min(freq_arr):
            normed = (freq_arr - np.min(freq_arr)) / (np.max(freq_arr) - np.min(freq_arr))
        else:
            normed = np.zeros_like(freq_arr)
        colors = [mpl_colors.rgb2hex(cmap(v)) for v in normed]
        return colors

    def _build_wordcloud(self, start_year, end_year, subject_ids):
        word_freq, word_undlids_map, total_resolutions = self.query_wc_data(start_year, end_year, subject_ids)
        self.current_word_resolution_map = word_undlids_map
        words = list(word_freq.keys())
        freqs = list(word_freq.values())
        if not words:
            return go.Figure().add_annotation(text=f"No data for {start_year}-{end_year}", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)

        # Size scaling
        f_min, f_max = min(freqs), max(freqs)
        min_size, max_size = 12, 52
        if f_max == f_min:
            sizes = [int((min_size + max_size) / 2)] * len(freqs)
        else:
            sizes = [int(min_size + (f - f_min) * (max_size - min_size) / (f_max - f_min)) for f in freqs]
        # Colors and positions
        colors = self._get_viridis_colors(freqs)
        x_positions, y_positions = self.scatter_wordcloud(words, sizes, seed=42 + start_year + end_year)
        hover_text = [f"<b>{w}</b><br>Appears in {f} resolutions" for w, f in zip(words, freqs)]
        trace = go.Scatter(
            x=x_positions,
            y=y_positions,
            mode='text',
            text=words,
            textposition="middle center",
            textfont=dict(size=sizes, color=colors),
            hovertemplate='%{customdata}<extra></extra>',
            customdata=hover_text,
            name=f"{start_year}-{end_year}"
        )
        fig = go.Figure(data=[trace])
        fig.update_layout(
            showlegend=False,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            margin=dict(l=10, r=10, t=10, b=10),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        return fig

    def init_wc_data(self):
        print("Init word cloud data...")
        keywords_df = pd.read_csv(f"wc_data/undlid_keywords.csv")
        data_all = pd.merge(self.query_engine.query_resolutions(), keywords_df, on='undl_id', how='left')
        self.earliest_year = pd.to_datetime(data_all['date'], errors='coerce').dt.year.min()
        self.latest_year = pd.to_datetime(data_all['date'], errors='coerce').dt.year.max()
        print(f"Earliest year in the data: {self.earliest_year}")
        print(f"Latest year in the data: {self.latest_year}")
        print("data_all.head(): \n", data_all[['undl_id', 'keywords']].head())

        ignore_words = []
        # ignore_contain = []
        # remove_phrases = []

        ignore_words = [
            "resolution", "general assembly"
        ]

        # ignore_words = [
        #     "resolution", "outcomes", "use", "alternative approaches",
        #     "development", "other development issues", "developments", "cooperation",
        #     "major  conferences", "summits", "report", "general", "declaration", "peoples",
        #     "prevention", "promotion", "reports", "prohibition", "convention", 
        #     "system", "representatives", "situation", "rights", "strengthening",
        #     "action", "information", "context", "production", "s/2014/136",
        #     "letter", "sea", "programme", "organization", "rest", "law",
        #     "city", "right", "elimination", "charter", "advisory opinion",
        #     "review", "policies", "government", "granting", "biennium",
        #     "economic and social council", "relief and works agency",
        #     "recommendations", "decisions", "treaty", "its 10th special session",
        #     "protection", "activities", "respect", "trusteeship council",
        #     "financial year", "article", "principles", "work", "proliferation",
        #     "transfer", "role", "efforts", "forms", "establishment", "conclusion",
        #     "operation", "assistance", "signature", "ratification", "parties",
        #     "status", "specialized agencies", "threat", "total elimination",
        #     "admission", "ways", "committee", "means", "field", "resolutions",
        #     "secretariat", "membership", "member states", "related intolerance",
        #     "support", "population", "members", "measures", "decade",
        #     "year", "exercise", "commission", "people", "interests",
        #     "additional protocol", "[", "agenda", "department", "office",
        #     "persons", "principle", "non", "which", "cause", "steps", "conferences",
        #     "73 e", "plan", "monitoring", "proposal", "monitoring", "path", "paths", "format", "formats",
        #     "council", "complementary", "association", "achievement", "scope", "the", "i"
        # ]

        # ignore_contain = [ # 包含以下短语的都删除
        #     "implementation", "causes", "follow-up"
        # ]

        # remove_phrases = [  
        #     "general assembly", "general assemly", "united nations", "security council", "special committee",
        #     "questions", "question", "goals", "its"
        # ]

        resolution_wc_data = {}
        wc_word_undlid_map = {}
        for undl_id, keywords in data_all[['undl_id', 'keywords']].values:
            word_in_resolution = Counter()
            tokens_set = set()
            for keyword in keywords.split(","):
                if keyword.strip().lower() in ignore_words:
                    continue
                tokens_set.add(keyword.strip().lower())

            for token in tokens_set:
                word_in_resolution[token] += 1
                if token not in wc_word_undlid_map:
                    wc_word_undlid_map[token] = []
                wc_word_undlid_map[token].append(undl_id)
            
            resolution_wc_data[undl_id] = {
                'word_freq': dict(word_in_resolution)
            }
        
        self.resolution_wc_data = resolution_wc_data
        self.wc_word_undlid_map = wc_word_undlid_map
        return

    def setup_layout(self):
        """Set up the Dash app layout."""
        # Add custom CSS for better slider styling
        self.app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            /* Custom slider styling */
            .rc-slider-track {
                background-color: #3498db !important;
                height: 6px !important;
            }
            .rc-slider-rail {
                background-color: #ecf0f1 !important;
                height: 6px !important;
            }
            .rc-slider-handle {
                border: 3px solid #3498db !important;
                background-color: #ffffff !important;
                width: 20px !important;
                height: 20px !important;
                margin-top: -7px !important;
                box-shadow: 0 2px 6px rgba(52, 152, 219, 0.3) !important;
            }
            .rc-slider-handle:hover {
                border-color: #2980b9 !important;
                box-shadow: 0 2px 8px rgba(52, 152, 219, 0.5) !important;
            }
            .rc-slider-handle:active {
                border-color: #2980b9 !important;
                box-shadow: 0 2px 10px rgba(52, 152, 219, 0.7) !important;
            }
            /* End Year Slider specific styling */
            #end-year-slider .rc-slider-track {
                background-color: #e67e22 !important;
            }
            #end-year-slider .rc-slider-handle {
                border-color: #e67e22 !important;
                box-shadow: 0 2px 6px rgba(230, 126, 34, 0.3) !important;
            }
            #end-year-slider .rc-slider-handle:hover {
                border-color: #d35400 !important;
                box-shadow: 0 2px 8px rgba(230, 126, 34, 0.5) !important;
            }
            #end-year-slider .rc-slider-handle:active {
                border-color: #d35400 !important;
                box-shadow: 0 2px 10px rgba(230, 126, 34, 0.7) !important;
            }
            /* Tooltip styling */
            .rc-slider-tooltip {
                z-index: 1000 !important;
            }
            .rc-slider-tooltip-inner {
                background-color: #2c3e50 !important;
                color: #ffffff !important;
                font-weight: bold !important;
                padding: 4px 8px !important;
                border-radius: 4px !important;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
        '''
        
        self.app.layout = html.Div([
            # Header
            html.Div([
                html.H1("Resolution Title Word Cloud", 
                       style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': '10px'}),
            ], style={'padding': '20px', 'backgroundColor': '#ecf0f1', 'marginBottom': '20px'}),
            
            # Controls and resolution table in two columns
            html.Div([
                html.Div([
                    html.Div([
                        html.Label("Start Year:", style={
                            'fontWeight': 'bold', 
                            'marginBottom': '20px',
                            'fontSize': '14px',
                            'color': '#2c3e50'
                        }),
                        dcc.Slider(
                            id='start-year-slider',
                            min=self.earliest_year,
                            max=self.latest_year,
                            step=1,
                            value=self.earliest_year,
                            marks={
                                str(self.earliest_year): {
                                    'label': str(self.earliest_year),
                                    'style': {'fontSize': '12px', 'fontWeight': 'bold', 'color': '#3498db'}
                                },
                                str(self.latest_year): {
                                    'label': str(self.latest_year),
                                    'style': {'fontSize': '12px', 'fontWeight': 'bold', 'color': '#3498db'}
                                }
                            },
                            included=False,
                            tooltip={"placement": "bottom", "always_visible": True},
                            updatemode='drag'
                        ),
                    ], style={
                        'backgroundColor': '#f8f9fa',
                        'padding': '20px',
                        'borderRadius': '8px',
                        'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
                        'marginBottom': '16px'
                    }),
                    html.Div([
                        html.Label("End Year:", style={
                            'fontWeight': 'bold', 
                            'marginBottom': '10px',
                            'fontSize': '14px',
                            'color': '#2c3e50'
                        }),
                        dcc.Slider(
                            id='end-year-slider',
                            min=self.earliest_year,
                            max=self.latest_year,
                            step=1,
                            value=self.latest_year,
                            marks={
                                str(self.earliest_year): {
                                    'label': str(self.earliest_year),
                                    'style': {'fontSize': '12px', 'fontWeight': 'bold', 'color': '#e67e22'}
                                },
                                str(self.latest_year): {
                                    'label': str(self.latest_year),
                                    'style': {'fontSize': '12px', 'fontWeight': 'bold', 'color': '#e67e22'}
                                }
                            },
                            included=False,
                            tooltip={"placement": "bottom", "always_visible": True},
                            updatemode='drag'
                        ),
                    ], style={
                        'backgroundColor': '#f8f9fa',
                        'padding': '20px',
                        'borderRadius': '8px',
                        'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
                    })
                    ,
                    html.Div([
                        html.Label("Categories (optional):", style={
                            'fontWeight': 'bold', 
                            'marginBottom': '10px',
                            'fontSize': '14px',
                            'color': '#2c3e50'
                        }),
                        dcc.Dropdown(
                            id='subject-dropdown',
                            options=subject_options,
                            value=None,
                            multi=True,
                            placeholder='Select one or more categories (or leave empty)'
                        )
                    ], style={
                        'backgroundColor': '#f8f9fa',
                        'padding': '12px 20px',
                        'borderRadius': '8px',
                        'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
                        'marginTop': '16px'
                    })
                ], style={
                    'width': '40%', 
                    'display': 'inline-block', 
                    'paddingRight': '2%'
                }),
                html.Div([
                    html.Label("Resolutions for hovered word:", style={
                        'fontWeight': 'bold', 
                        'marginBottom': '10px',
                        'fontSize': '14px',
                        'color': '#2c3e50'
                    }),
                    html.Div(id='resolution-table', style={
                        'backgroundColor': '#ffffff',
                        'padding': '12px',
                        'borderRadius': '8px',
                        'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
                        'maxHeight': '220px',
                        'overflowY': 'auto'
                    })
                ], style={
                    'width': '55%', 
                    'display': 'inline-block',
                    'verticalAlign': 'top',
                    'backgroundColor': '#f8f9fa',
                    'padding': '20px',
                    'borderRadius': '8px',
                    'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
                }),
            ], style={'padding': '0 20px', 'marginBottom': '30px'}),
            
            # Meta info above chart
            html.Div([
                html.Div(id='wc-meta', style={'textAlign': 'center', 'fontWeight': 'bold', 'marginBottom': '8px'})
            ], style={'padding': '0 20px'}),

            # Loading indicator and chart
            html.Div([
                dcc.Loading(
                    id="loading-chart",
                    children=[dcc.Graph(id='wordcloud-chart', style={'height': '600px'})],
                    type="cube",
                    color="#3498db"
                )
            ], style={'padding': '0 20px'}),
            
            # Word frequency dictionary display
            html.Div([
                html.H2("Word Frequency Results", style={'textAlign': 'center', 'marginTop': '30px', 'marginBottom': '15px'}),
                dcc.Loading(
                    id="loading-word-freq",
                    children=[html.Div(id='word-freq-display', style={'padding': '20px'})],
                    type="default",
                    color="#3498db"
                )
            ], style={'padding': '0 20px', 'marginTop': '30px'}),
            
            # Footer with instructions
            html.Div([
                html.Hr(),
                html.P([
                    # "💡 ", html.Strong("How it works:"), " Select countries and time span above. ",
                    # "Data is calculated on-demand and cached for fast re-access. ",
                    # "Agreement ranges from 0 (complete disagreement) to 1 (perfect agreement)."
                ], style={'color': '#7f8c8d', 'textAlign': 'center', 'fontSize': '14px'})
            ], style={'padding': '20px', 'marginTop': '40px'})
        ])
    
    def setup_callbacks(self):
        """Set up Dash callbacks for interactivity."""
        @self.app.callback(
            Output('end-year-slider', 'value'),
            Input('start-year-slider', 'value'),
            Input('end-year-slider', 'value')
        )
        def enforce_year_order(start_year, end_year):
            """Ensure end year is not before start year"""
            if end_year < start_year:
                return start_year
            return end_year
        
        @self.app.callback(
            Output('word-freq-display', 'children'),
            Input('start-year-slider', 'value'),
            Input('end-year-slider', 'value'),
            Input('subject-dropdown', 'value')
        )
        def update_word_freq_display(start_year, end_year, subject_ids):
            """Update word frequency dictionary display when year sliders change."""
            # Prevent execution if end_year < start_year (enforce_year_order will fix it first)
            if end_year < start_year:
                raise PreventUpdate
            
            try:
                # Query word frequency data
                word_freq_dict, word_undlids_map, total_resolutions = self.query_wc_data(start_year, end_year, subject_ids)
                
                if not word_freq_dict:
                    return html.Div([
                        html.I(className="fas fa-info-circle", style={'color': 'blue', 'marginRight': '5px'}),
                        f"No word frequency data found for years {start_year}-{end_year}."
                    ], style={'color': 'blue', 'padding': '20px', 'textAlign': 'center'})
                
                # Create a table to display the results
                table_rows = []
                table_rows.append(html.Tr([
                    html.Th("Word", style={'padding': '10px', 'textAlign': 'left', 'borderBottom': '2px solid #3498db', 'fontWeight': 'bold'}),
                    html.Th("Frequency", style={'padding': '10px', 'textAlign': 'right', 'borderBottom': '2px solid #3498db', 'fontWeight': 'bold'})
                ]))
                
                # Ensure stable sorted display by frequency desc, then word asc
                sorted_items = sorted(word_freq_dict.items(), key=lambda x: (-x[1], x[0]))
                for word, freq in sorted_items:
                    table_rows.append(html.Tr([
                        html.Td(word, style={'padding': '8px', 'borderBottom': '1px solid #ecf0f1'}),
                        html.Td(f"{freq:,}", style={'padding': '8px', 'textAlign': 'right', 'borderBottom': '1px solid #ecf0f1'})
                    ]))
                
                return html.Div([
                    html.Div([
                        html.Strong(f"Year Range: {start_year} - {end_year}"),
                        # html.Br(),
                        # html.Span(f"Total unique words shown: {len(word_freq_dict)}", style={'color': '#7f8c8d', 'fontSize': '14px'})
                    ], style={'marginBottom': '15px'}),
                    html.Table(
                        table_rows,
                        style={
                            'width': '100%',
                            'borderCollapse': 'collapse',
                            'backgroundColor': 'white',
                            'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
                            'borderRadius': '5px'
                        }
                    )
                ])
                
            except Exception as e:
                return html.Div([
                    html.I(className="fas fa-exclamation-circle", style={'color': 'red', 'marginRight': '5px'}),
                    html.Strong("Error: "), f"Failed to load word frequency data: {str(e)}"
                ], style={'color': 'red', 'padding': '20px', 'textAlign': 'center'})

        # @self.app.callback(
        #     [Output('agreement-chart', 'figure'),
        #      Output('status-display', 'children')],
        #     [Input('country1-dropdown', 'value'),
        #      Input('country2-dropdown', 'value'),
        #      Input('timespan-dropdown', 'value')]
        # )
        # def update_chart(country1, country2, time_span):
        #     """Update chart when countries or time span changes."""
            
        #     # Validation
        #     if country1 == country2:
        #         error_msg = html.Div([
        #             html.I(className="fas fa-exclamation-triangle", style={'color': 'orange', 'marginRight': '5px'}),
        #             html.Strong("Warning: "), "Same country selected for both dropdowns. Please choose different countries."
        #         ], style={'color': 'orange'})
                
        #         return go.Figure().add_annotation(
        #             text="Please select different countries",
        #             xref="paper", yref="paper", x=0.5, y=0.5,
        #             showarrow=False, font_size=20
        #         ), error_msg
            
        #     try:
        #         # Get cache info before calculation
        #         cache_info_before = self.calculate_data.cache_info()
                
        #         # Calculate data (uses cache if available)
        #         data, calc_time = self.calculate_data(country1, country2, time_span)
                
        #         # Get cache info after calculation
        #         cache_info_after = self.calculate_data.cache_info()
        #         was_cached = cache_info_before.hits < cache_info_after.hits
                
        #         # Create figure
        #         fig = go.Figure()
                
        #         # Add traces
        #         fig.add_trace(go.Scatter(
        #             x=data['date'], y=data['sma'],
        #             mode='lines', name=f'{time_span}-Day SMA',
        #             line=dict(color='#3498db', width=3)
        #         ))
                
        #         fig.add_trace(go.Scatter(
        #             x=data['date'], y=data['ema'],
        #             mode='lines', name=f'{time_span}-Day EMA',
        #             line=dict(color='#e67e22', width=3)
        #         ))
                
        #         fig.add_trace(go.Scatter(
        #             x=data['date'], y=data['cma'],
        #             mode='lines', name='Cumulative MA',
        #             line=dict(color='#27ae60', width=3)
        #         ))
                
        #         # Add missing values
        #         missing_mask = pd.isna(data['agreement'])
        #         if missing_mask.any():
        #             fig.add_trace(go.Scatter(
        #                 x=data['date'][missing_mask],
        #                 y=[0.5] * missing_mask.sum(),
        #                 mode='markers', name='Missing Data',
        #                 marker=dict(color='gray', symbol='x', size=8),
        #                 opacity=0.7
        #             ))
                
        #         # Update layout
        #         missing_count = missing_mask.sum()
        #         total_count = len(data)
        #         cache_status = "📋 Cached" if was_cached else "🔄 Calculated"
                
        #         fig.update_layout(
        #             title=f'GA Voting Agreement: {country1} vs {country2}<br>' +
        #                   f'<sub>{total_count:,} votes • {missing_count:,} missing ({missing_count/total_count*100:.1f}%) • {cache_status}</sub>',
        #             xaxis_title='Date',
        #             yaxis_title='Agreement Level',
        #             yaxis=dict(range=[-0.05, 1.05]),
        #             template='plotly_white',
        #             hovermode='x unified',
        #             legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        #         )
                
        #         # Status message
        #         status_msg = html.Div([
        #             html.Div([
        #                 html.I(className="fas fa-check-circle", style={'color': 'green', 'marginRight': '5px'}),
        #                 html.Strong("Chart Updated Successfully! "),
        #                 f"Processed {total_count:,} data points in {calc_time:.2f}s ",
        #                 f"({'from cache' if was_cached else 'newly calculated'})"
        #             ]),
        #             html.Div([
        #                 html.I(className="fas fa-database", style={'color': 'blue', 'marginRight': '5px'}),
        #                 f"Cache: {cache_info_after.currsize}/{cache_info_after.maxsize} pairs stored • ",
        #                 f"Hits: {cache_info_after.hits} • Misses: {cache_info_after.misses} • ",
        #                 f"Hit Rate: {cache_info_after.hits/(cache_info_after.hits + cache_info_after.misses)*100:.1f}%"
        #             ], style={'fontSize': '12px', 'color': '#7f8c8d', 'marginTop': '5px'})
        #         ])
                
        #         return fig, status_msg
                
        #     except Exception as e:
        #         error_msg = html.Div([
        #             html.I(className="fas fa-exclamation-circle", style={'color': 'red', 'marginRight': '5px'}),
        #             html.Strong("Error: "), f"Failed to generate chart: {str(e)}"
        #         ], style={'color': 'red'})
                
        #         return go.Figure().add_annotation(
        #             text=f"Error: {str(e)}", xref="paper", yref="paper", 
        #             x=0.5, y=0.5, showarrow=False, font_size=16
        #         ), error_msg

        @self.app.callback(
            Output('wordcloud-chart', 'figure'),
            Input('start-year-slider', 'value'),
            Input('end-year-slider', 'value'),
            Input('subject-dropdown', 'value')
        )
        def update_wordcloud_chart(start_year, end_year, subject_ids):
            if end_year < start_year:
                raise PreventUpdate
            try:
                return self._build_wordcloud(start_year, end_year, subject_ids)
            except Exception as e:
                fig = go.Figure()
                fig.add_annotation(text=f"Error: {str(e)}", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
                fig.update_layout(xaxis=dict(visible=False), yaxis=dict(visible=False))
                return fig

        @self.app.callback(
            Output('wc-meta', 'children'),
            Input('start-year-slider', 'value'),
            Input('end-year-slider', 'value'),
            Input('subject-dropdown', 'value')
        )
        def update_wc_meta(start_year, end_year, subject_ids):
            if end_year < start_year:
                raise PreventUpdate
            try:
                _, _, total_resolutions = self.query_wc_data(start_year, end_year, subject_ids)
                return f"Total resolutions in range: {total_resolutions:,}"
            except Exception as e:
                return f"Error loading total: {str(e)}"

        @self.app.callback(
            Output('resolution-table', 'children'),
            Input('wordcloud-chart', 'hoverData'),
            Input('start-year-slider', 'value'),
            Input('end-year-slider', 'value'),
            Input('subject-dropdown', 'value')
        )
        def update_resolution_table(hoverData, start_year, end_year, subject_ids):
            if end_year < start_year:
                raise PreventUpdate
            if not hoverData or 'points' not in hoverData or not hoverData['points']:
                return html.Div("Hover over a word to see related resolutions.", style={'color': '#7f8c8d'})
            try:
                word = hoverData['points'][0].get('text')
                if not word or word not in self.current_word_resolution_map:
                    return html.Div("No resolutions for this word in range.", style={'color': '#7f8c8d'})
                undl_ids = pd.Series(self.current_word_resolution_map[word])
                data = self.query_data(start_year, end_year, subject_ids)
                df = self.filter_data_by_undlid(data, undl_ids)
                if df.empty:
                    return html.Div("No resolutions found.", style={'color': '#7f8c8d'})
                # Sort by date then id
                df = df[['resolution', 'date', 'title']].copy()
                # df = df[['undl_id', 'date', 'title']].copy()
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
                df = df.sort_values(by=['date', 'resolution'], ascending=[True, True])
                # df = df.sort_values(by=['date', 'undl_id'], ascending=[True, True])
                # Limit number of rows for readability
                max_rows = 50
                df_limited = df.head(max_rows)
                header = html.Tr([
                    html.Th("Resolution", style={'padding': '6px', 'textAlign': 'left', 'borderBottom': '2px solid #3498db'}),
                    html.Th("Date", style={'padding': '6px', 'textAlign': 'left', 'borderBottom': '2px solid #3498db'}),
                    html.Th("Title", style={'padding': '6px', 'textAlign': 'left', 'borderBottom': '2px solid #3498db'})
                ])
                rows = [
                    html.Tr([
                        html.Td(str(r['resolution']), style={'padding': '6px', 'verticalAlign': 'top'}),
                        # html.Td(str(r['undl_id']), style={'padding': '6px', 'verticalAlign': 'top'}),
                        html.Td((r['date'].date().isoformat() if pd.notna(r['date']) else ''), style={'padding': '6px', 'verticalAlign': 'top'}),
                        html.Td(str(r['title']), style={'padding': '6px'})
                    ]) for _, r in df_limited.iterrows()
                ]
                summary = html.Div(f"{len(df)} resolutions for word '{word}'", style={'fontWeight': 'bold', 'marginBottom': '8px'})
                table = html.Table([header] + rows, style={'width': '100%', 'borderCollapse': 'collapse'})
                return html.Div([summary, table])
            except Exception as e:
                return html.Div(f"Error: {str(e)}", style={'color': 'red'})
    
    def run(self, debug=True, port=8050, host='127.0.0.1'):
        """Run the Dash app."""
        print(f"🚀 Starting Dash app...")
        print(f"🌐 Open your browser to: http://{host}:{port}")
        
        self.app.run(debug=debug, port=port, host=host)


# Easy setup function
def create_dash_app(query_engine, analyzer):
    return DashWordCloudApp(query_engine, analyzer)

import os, sys

base_dir = None
try:
    base_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    base_dir = os.getcwd()

sys.path.append(
    os.path.normpath(
        os.path.join(base_dir, "..", "janic")
    )
)

from unDataStream import DataRepository, ResolutionQueryEngine

repo = DataRepository(
    config_path=os.path.normpath(
        os.path.join(
            base_dir,
            "..",
            "janic",
            "config",
            "data_sources.yaml",
        )
    )
)
query_engine = ResolutionQueryEngine(repo)
subject_table = repo.get_data()['subject']
# 'http://metadata.un.org/thesaurus/00', 
level_zero_subjects = {
    'http://metadata.un.org/thesaurus/10', 
    'http://metadata.un.org/thesaurus/09', 
    'http://metadata.un.org/thesaurus/16', 
    'http://metadata.un.org/thesaurus/07', 
    'http://metadata.un.org/thesaurus/04', 
    'http://metadata.un.org/thesaurus/06', 
    'http://metadata.un.org/thesaurus/15', 
    'http://metadata.un.org/thesaurus/05', 
    'http://metadata.un.org/thesaurus/03', 
    'http://metadata.un.org/thesaurus/17', 
    'http://metadata.un.org/thesaurus/11', 
    'http://metadata.un.org/thesaurus/12', 
    'http://metadata.un.org/thesaurus/13', 
    'http://metadata.un.org/thesaurus/14', 
    'http://metadata.un.org/thesaurus/18', 
    'http://metadata.un.org/thesaurus/08', 
    'http://metadata.un.org/thesaurus/01', 
    'http://metadata.un.org/thesaurus/02'
}
subject_options_list = subject_table[subject_table['subject_id'].isin(level_zero_subjects)].to_dict('records')
subject_options = [{"label": r["label_en"], "value": r["subject_id"]} for r in subject_options_list]
print("subject_options:")
for subject in subject_options:
    print(f"{subject['label']}")
print("Query engine initialized")

from TextAnalyzer import TextAnalyzer
analyzer = TextAnalyzer()
print("Analyzer initialized")

app = create_dash_app(query_engine, analyzer)
app.run(debug=True, port=8050)