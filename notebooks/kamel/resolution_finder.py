import dash
from dash import dcc, html, Input, Output, State, no_update
import pandas as pd
import datetime
import io
import numpy as np
from pathlib import Path
from unDataStream import DataRepository
from unDataStream import ResolutionQueryEngine


config_path = Path('config/data_sources.yaml')
repo = DataRepository(config_path)
data = repo.get_data()
analyzer = ResolutionQueryEngine(repo)


PAGE_SIZE = 100
MIN_UN_DATE = datetime.date(1945, 1, 1)
MAX_UN_DATE = datetime.date.today()

# Prepare subject options for dropdown
subject_options_list = data["subject"].to_dict('records')
subject_options = [{"label": r["label_en"], "value": r["subject_id"]} for r in subject_options_list]

# Prepare country options for dropdown
country_options = []
# Exclude non-country columns (adjust list as needed)
excluded_cols = [
    "undl_id", "date", "session", "resolution", "draft", "committee_report", 
    "meeting", "title", "agenda_title", "subjects", "total_yes", "total_no", 
    "total_abstentions", "total_non_voting", "total_ms", "undl_link"
]
for c in data["resolution"].columns:
    if c not in excluded_cols:
        country_options.append(c)
country_options.sort()
country_options_list = [{"label": c, "value": c} for c in country_options]


level_zero_subjects = {'http://metadata.un.org/thesaurus/10', 'http://metadata.un.org/thesaurus/09', 'http://metadata.un.org/thesaurus/16', 'http://metadata.un.org/thesaurus/00', 'http://metadata.un.org/thesaurus/07', 'http://metadata.un.org/thesaurus/04', 'http://metadata.un.org/thesaurus/06', 'http://metadata.un.org/thesaurus/15', 'http://metadata.un.org/thesaurus/05', 'http://metadata.un.org/thesaurus/03', 'http://metadata.un.org/thesaurus/17', 'http://metadata.un.org/thesaurus/11', 'http://metadata.un.org/thesaurus/12', 'http://metadata.un.org/thesaurus/13', 'http://metadata.un.org/thesaurus/14', 'http://metadata.un.org/thesaurus/18', 'http://metadata.un.org/thesaurus/08', 'http://metadata.un.org/thesaurus/01', 'http://metadata.un.org/thesaurus/02'}
level_one_subjects = {'http://metadata.un.org/thesaurus/170400', 'http://metadata.un.org/thesaurus/10', 'http://metadata.un.org/thesaurus/120600', 'http://metadata.un.org/thesaurus/160400', 'http://metadata.un.org/thesaurus/030200', 'http://metadata.un.org/thesaurus/170200', 'http://metadata.un.org/thesaurus/021200', 'http://metadata.un.org/thesaurus/150102', 'http://metadata.un.org/thesaurus/010300', 'http://metadata.un.org/thesaurus/00', 'http://metadata.un.org/thesaurus/080100', 'http://metadata.un.org/thesaurus/021000', 'http://metadata.un.org/thesaurus/010702', 'http://metadata.un.org/thesaurus/060500', 'http://metadata.un.org/thesaurus/07', 'http://metadata.un.org/thesaurus/160900', 'http://metadata.un.org/thesaurus/140501', 'http://metadata.un.org/thesaurus/020900', 'http://metadata.un.org/thesaurus/050500', 'http://metadata.un.org/thesaurus/120100', 'http://metadata.un.org/thesaurus/070202', 'http://metadata.un.org/thesaurus/06', 'http://metadata.un.org/thesaurus/140202', 'http://metadata.un.org/thesaurus/030400', 'http://metadata.un.org/thesaurus/150200', 'http://metadata.un.org/thesaurus/030500', 'http://metadata.un.org/thesaurus/161000', 'http://metadata.un.org/thesaurus/11', 'http://metadata.un.org/thesaurus/070500', 'http://metadata.un.org/thesaurus/100100', 'http://metadata.un.org/thesaurus/180702', 'http://metadata.un.org/thesaurus/120300', 'http://metadata.un.org/thesaurus/070201', 'http://metadata.un.org/thesaurus/110400', 'http://metadata.un.org/thesaurus/050400', 'http://metadata.un.org/thesaurus/100302', 'http://metadata.un.org/thesaurus/051000', 'http://metadata.un.org/thesaurus/18', 'http://metadata.un.org/thesaurus/100200', 'http://metadata.un.org/thesaurus/010200', 'http://metadata.un.org/thesaurus/140100', 'http://metadata.un.org/thesaurus/01', 'http://metadata.un.org/thesaurus/140402', 'http://metadata.un.org/thesaurus/160300', 'http://metadata.un.org/thesaurus/140502', 'http://metadata.un.org/thesaurus/010703', 'http://metadata.un.org/thesaurus/060200', 'http://metadata.un.org/thesaurus/150101', 'http://metadata.un.org/thesaurus/080302', 'http://metadata.un.org/thesaurus/150300', 'http://metadata.un.org/thesaurus/120500', 'http://metadata.un.org/thesaurus/100602', 'http://metadata.un.org/thesaurus/060400', 'http://metadata.un.org/thesaurus/040400', 'http://metadata.un.org/thesaurus/130200', 'http://metadata.un.org/thesaurus/170600', 'http://metadata.un.org/thesaurus/050000', 'http://metadata.un.org/thesaurus/050102', 'http://metadata.un.org/thesaurus/030800', 'http://metadata.un.org/thesaurus/060102', 'http://metadata.un.org/thesaurus/050600', 'http://metadata.un.org/thesaurus/020800', 'http://metadata.un.org/thesaurus/180701', 'http://metadata.un.org/thesaurus/180703', 'http://metadata.un.org/thesaurus/010701', 'http://metadata.un.org/thesaurus/031000', 'http://metadata.un.org/thesaurus/03', 'http://metadata.un.org/thesaurus/170302', 'http://metadata.un.org/thesaurus/010602', 'http://metadata.un.org/thesaurus/040201', 'http://metadata.un.org/thesaurus/010704', 'http://metadata.un.org/thesaurus/070400', 'http://metadata.un.org/thesaurus/160100', 'http://metadata.un.org/thesaurus/020100', 'http://metadata.un.org/thesaurus/12', 'http://metadata.un.org/thesaurus/110200', 'http://metadata.un.org/thesaurus/030102', 'http://metadata.un.org/thesaurus/150500', 'http://metadata.un.org/thesaurus/140300', 'http://metadata.un.org/thesaurus/100500', 'http://metadata.un.org/thesaurus/160700', 'http://metadata.un.org/thesaurus/070300', 'http://metadata.un.org/thesaurus/150600', 'http://metadata.un.org/thesaurus/050700', 'http://metadata.un.org/thesaurus/180500', 'http://metadata.un.org/thesaurus/020200', 'http://metadata.un.org/thesaurus/050101', 'http://metadata.un.org/thesaurus/160500', 'http://metadata.un.org/thesaurus/040101', 'http://metadata.un.org/thesaurus/030700', 'http://metadata.un.org/thesaurus/090100', 'http://metadata.un.org/thesaurus/150000', 'http://metadata.un.org/thesaurus/030103', 'http://metadata.un.org/thesaurus/160800', 'http://metadata.un.org/thesaurus/080200', 'http://metadata.un.org/thesaurus/020700', 'http://metadata.un.org/thesaurus/180400', 'http://metadata.un.org/thesaurus/150400', 'http://metadata.un.org/thesaurus/090200', 'http://metadata.un.org/thesaurus/170500', 'http://metadata.un.org/thesaurus/021100', 'http://metadata.un.org/thesaurus/170301', 'http://metadata.un.org/thesaurus/080301', 'http://metadata.un.org/thesaurus/030600', 'http://metadata.un.org/thesaurus/060600', 'http://metadata.un.org/thesaurus/020500', 'http://metadata.un.org/thesaurus/04', 'http://metadata.un.org/thesaurus/180200', 'http://metadata.un.org/thesaurus/050300', 'http://metadata.un.org/thesaurus/050800', 'http://metadata.un.org/thesaurus/15', 'http://metadata.un.org/thesaurus/05', 'http://metadata.un.org/thesaurus/17', 'http://metadata.un.org/thesaurus/010601', 'http://metadata.un.org/thesaurus/140201', 'http://metadata.un.org/thesaurus/010400', 'http://metadata.un.org/thesaurus/040300', 'http://metadata.un.org/thesaurus/100400', 'http://metadata.un.org/thesaurus/140504', 'http://metadata.un.org/thesaurus/140401', 'http://metadata.un.org/thesaurus/060101', 'http://metadata.un.org/thesaurus/020400', 'http://metadata.un.org/thesaurus/070100', 'http://metadata.un.org/thesaurus/180600', 'http://metadata.un.org/thesaurus/130100', 'http://metadata.un.org/thesaurus/180300', 'http://metadata.un.org/thesaurus/030300', 'http://metadata.un.org/thesaurus/100601', 'http://metadata.un.org/thesaurus/010700', 'http://metadata.un.org/thesaurus/02', 'http://metadata.un.org/thesaurus/110300', 'http://metadata.un.org/thesaurus/020601', 'http://metadata.un.org/thesaurus/110100', 'http://metadata.un.org/thesaurus/050900', 'http://metadata.un.org/thesaurus/040600', 'http://metadata.un.org/thesaurus/120200', 'http://metadata.un.org/thesaurus/010100', 'http://metadata.un.org/thesaurus/040202', 'http://metadata.un.org/thesaurus/020300', 'http://metadata.un.org/thesaurus/09', 'http://metadata.un.org/thesaurus/070000', 'http://metadata.un.org/thesaurus/16', 'http://metadata.un.org/thesaurus/010500', 'http://metadata.un.org/thesaurus/170100', 'http://metadata.un.org/thesaurus/030900', 'http://metadata.un.org/thesaurus/160600', 'http://metadata.un.org/thesaurus/130300', 'http://metadata.un.org/thesaurus/180100', 'http://metadata.un.org/thesaurus/050200', 'http://metadata.un.org/thesaurus/020602', 'http://metadata.un.org/thesaurus/140503', 'http://metadata.un.org/thesaurus/040500', 'http://metadata.un.org/thesaurus/030101', 'http://metadata.un.org/thesaurus/110500', 'http://metadata.un.org/thesaurus/13', 'http://metadata.un.org/thesaurus/14', 'http://metadata.un.org/thesaurus/060300', 'http://metadata.un.org/thesaurus/120400', 'http://metadata.un.org/thesaurus/08', 'http://metadata.un.org/thesaurus/160200', 'http://metadata.un.org/thesaurus/040102', 'http://metadata.un.org/thesaurus/100301'}
TOP_LEVEL_SUBJECTS = level_one_subjects
SUBJECT_ID_TO_LABEL_MAP = data['subject'].set_index('subject_id')['label_en'].to_dict()

# --- Helper Function for Vote Indicators ---
def create_vote_indicator(country_name, vote):
    VOTE_MAP = {
        'Y': {'color': 'green', 'label': 'Yes'},
        'N': {'color': 'red', 'label': 'No'},
        'A': {'color': 'orange', 'label': 'Abstain'},
        'X': {'color': 'blue', 'label': 'Not Voting'}
    }
    if pd.isna(vote) or vote not in VOTE_MAP:
        return html.Div(f"{country_name}: (Data N/A)", style={'color': 'grey', 'fontStyle': 'italic', 'margin-right': '15px', 'display': 'inline-block', 'fontSize': '0.9em'})
    config = VOTE_MAP[vote]
    return html.Div([
        html.Span('●', style={'margin-right': '5px'}),
        html.Span(f"{country_name}: {config['label']}")
    ], style={'color': config['color'], 'fontWeight': 'bold', 'margin-right': '15px', 'display': 'inline-block', 'fontSize': '0.9em'})


# --- [MODIFIED FUNCTION] ---
def calculate_agreement(analyzer_instance, c1, c2, start_date, end_date, subject_list, subject_map):
    """
    Calculates agreement scores based on the new metric.
    Score = avg(abs(val_c1 - val_c2))
    """
    print(f"Running new analysis for {c1} vs {c2} from {start_date} to {end_date}")
    
    # Define the new value mapping
    vote_map = {'Y': 1, 'N': -1, 'A': 0, 'X': 0}
    
    agreement_results = []

    for subject_id in subject_list:
        # 1. Query resolutions for this single topic
        df = analyzer_instance.query_resolutions(
            start_date=start_date, 
            end_date=end_date, 
            subject_ids=[subject_id]
        )
        
        # 2. Filter to relevant votes (drop NaNs for *both* countries)
        df_valid = df.dropna(subset=[c1, c2]).copy()
        total_votes = len(df_valid)
        if total_votes < 30:
            continue
        
        if total_votes == 0:
            agreement_results.append({
                'subject_label': subject_map.get(subject_id, f"ID: {subject_id}"),
                'disagreement_score': np.nan,
                'total_votes': 0
            })
            continue

        # 3. Map votes to numerical values
        df_valid['val_c1'] = df_valid[c1].map(vote_map)
        df_valid['val_c2'] = df_valid[c2].map(vote_map)
        
        # 4. Handle potential unmapped values (e.g., if vote_map is incomplete)
        # This drops rows where a vote was not 'Y', 'N', 'A', or 'X'
        df_valid = df_valid.dropna(subset=['val_c1', 'val_c2'])
        
        # Recalculate total_votes in case any were dropped
        total_votes = len(df_valid)
        if total_votes == 0:
            # This handles the edge case where votes were present but not in the map
            agreement_results.append({
                'subject_label': subject_map.get(subject_id, f"ID: {subject_id}"),
                'disagreement_score': np.nan,
                'total_votes': 0
            })
            continue
            
        # 5. Calculate the absolute difference for each resolution
        df_valid['abs_diff'] = (df_valid['val_c1'] - df_valid['val_c2']).abs()
        
        # 6. Calculate the final score (sum of diffs / total resolutions)
        disagreement_score = df_valid['abs_diff'].sum() / (2 * total_votes)
        
        # 7. Store results
        agreement_results.append({
            'subject_label': subject_map.get(subject_id, f"ID: {subject_id}"),
            'disagreement_score': disagreement_score,
            'total_votes': total_votes
        })
        
    return pd.DataFrame(agreement_results)

# --- Initialize the Dash App ---
app = dash.Dash(__name__, external_stylesheets=['https.codepen.io/chriddyp/pen/bWLwgP.css'])
server = app.server

# --- Define the App Layout ---
# (Layout code remains the same as before)
app.layout = html.Div([
    html.H1("UN Resolution Explorer"),
    dcc.Store(id='filtered-data-store'),
    dcc.Store(id='analysis-data-store'),
    html.Div(className='row', children=[
        html.Div(className='four columns', style={'border': '1px solid #ddd', 'padding': '10px', 'borderRadius': '5px'}, children=[
            html.H3("Filters"),
            html.Label("Date Range:"),
            dcc.DatePickerRange(id='date-picker-range', min_date_allowed=MIN_UN_DATE, max_date_allowed=MAX_UN_DATE, start_date=MIN_UN_DATE, end_date=MAX_UN_DATE, display_format='YYYY-MM-DD'),
            html.Hr(),
            html.Label("Subjects:"),
            dcc.Dropdown(id='subject-dropdown', options=subject_options, multi=True, placeholder="Filter by subjects..."),
            html.Hr(),
            html.Label("Country 1:"),
            dcc.Dropdown(id='country-1-dropdown', options=country_options_list, placeholder="Select first country...", clearable=True),
            html.Label("Country 2:"),
            dcc.Dropdown(id='country-2-dropdown', options=country_options_list, placeholder="Select second country (optional)...", clearable=True),
            html.Div(id='single-country-filter-div', style={'display': 'none'}, children=[
                html.Hr(),
                html.Label("Country 1 Vote (for Filtered List):"),
                dcc.RadioItems(id='single-vote-radio', options=[{'label': 'No Filter', 'value': 'NO_FILTER'}, {'label': 'Voted Yes', 'value': 'Y'}, {'label': 'Voted No', 'value': 'N'}, {'label': 'Abstained', 'value': 'A'}, {'label': "Didn't Vote", 'value': 'X'}], value='NO_FILTER', labelStyle={'display': 'block'})
            ]),
            html.Div(id='two-country-filter-div', style={'display': 'none'}, children=[
                html.Hr(),
                html.Label("Country Agreement (for Filtered List):"),
                dcc.RadioItems(id='agreement-radio', options=[{'label': 'No Filter', 'value': 'NO_FILTER'}, {'label': 'Agreed (Voted Same)', 'value': 'AGREED'}, {'label': 'Disagreed (Voted Differently)', 'value': 'DISAGREED'}, {'label': 'Strongly Disagreed (Y/N vs N/Y)', 'value': 'STRONGLY_DISAGREED'}], value='NO_FILTER', labelStyle={'display': 'block'})
            ]),
            html.Hr(),
            html.Button("Apply Filters", id='apply-button', n_clicks=0, style={'width': '100%', 'backgroundColor': '#007BFF', 'color': 'white'})
        ]),
        html.Div(className='eight columns', children=[
            dcc.Loading(id='loading-spinner', type='default', children=[
                dcc.Tabs(id='results-tabs', children=[
                    dcc.Tab(label='Filtered Results', id='results-list-tab', children=[
                        html.H4(id='results-summary'),
                        html.Div(id='results-output', style={'maxHeight': '70vh', 'overflowY': 'auto', 'border': '1px solid #eee', 'padding': '10px'}),
                        html.Button("Load More", id='load-more-button', n_clicks=0, style={'width': '100%', 'marginTop': '10px', 'display': 'none'})
                    ]),
                    dcc.Tab(label='Agreement Analysis', id='analysis-tab', disabled=True, children=[
                        html.Div(id='analysis-output', style={'padding': '15px'})
                    ])
                ])
            ])
        ])
    ])
], style={'padding': '20px'})


# --- Callbacks ---

# (update_country_filter_ui callback remains the same)
@app.callback(
    Output('single-country-filter-div', 'style'),
    Output('two-country-filter-div', 'style'),
    Input('country-1-dropdown', 'value'),
    Input('country-2-dropdown', 'value')
)
def update_country_filter_ui(country_1, country_2):
    if country_1 and country_2:
        return {'display': 'none'}, {'display': 'block'}
    elif country_1 and not country_2:
        return {'display': 'block'}, {'display': 'none'}
    else:
        return {'display': 'none'}, {'display': 'none'}

# (toggle_analysis_tab callback remains the same)
@app.callback(
    Output('analysis-tab', 'disabled'),
    Output('analysis-tab', 'label'),
    Input('country-1-dropdown', 'value'),
    Input('country-2-dropdown', 'value')
)
def toggle_analysis_tab(c1, c2):
    if c1 and c2:
        return False, f"Analysis: {c1} vs {c2}"
    else:
        return True, "Agreement Analysis (Select 2 Countries)"

# (query_base_resolutions callback remains the same)
@app.callback(
    Output('filtered-data-store', 'data'),
    Output('results-summary', 'children', allow_duplicate=True),
    Output('load-more-button', 'n_clicks'),
    Input('apply-button', 'n_clicks'),
    State('date-picker-range', 'start_date'),
    State('date-picker-range', 'end_date'),
    State('subject-dropdown', 'value'),
    prevent_initial_call=True
)
def query_base_resolutions(n_clicks, start_date, end_date, subject_ids):
    if n_clicks == 0:
        return no_update, no_update, no_update
    subjects = subject_ids if subject_ids else None
    start = start_date if start_date else None
    end = end_date if end_date else None
    df = analyzer.query_resolutions(start_date=start, end_date=end, subject_ids=subjects)
    return df.to_json(date_format='iso', orient='split'), f"Found {len(df)} resolutions. Applying country filters...", 0

# (run_agreement_analysis callback remains the same, it calls the *new* calculate_agreement)
@app.callback(
    Output('analysis-data-store', 'data'),
    Input('apply-button', 'n_clicks'),
    State('country-1-dropdown', 'value'),
    State('country-2-dropdown', 'value'),
    State('date-picker-range', 'start_date'),
    State('date-picker-range', 'end_date'),
    prevent_initial_call=True
)
def run_agreement_analysis(n_clicks, c1, c2, start_date, end_date):
    if n_clicks == 0:
        return no_update
    if c1 and c2:
        df_analysis = calculate_agreement(
            analyzer, c1, c2, start_date, end_date, 
            TOP_LEVEL_SUBJECTS, SUBJECT_ID_TO_LABEL_MAP
        )
        return df_analysis.to_json(orient='split')
    return None

# (display_filtered_results callback for Tab 1 remains the same)
@app.callback(
    Output('results-output', 'children'),
    Output('results-summary', 'children'),
    Output('load-more-button', 'style'), 
    Input('filtered-data-store', 'data'), 
    Input('load-more-button', 'n_clicks'),
    State('country-1-dropdown', 'value'),
    State('country-2-dropdown', 'value'),
    State('single-vote-radio', 'value'),
    State('agreement-radio', 'value'),
    prevent_initial_call=True
)
def display_filtered_results(json_data, n_clicks, c1, c2, single_vote, agreement):
    button_style = {'display': 'none'}
    if not json_data:
        return html.P("Click 'Apply Filters' to load data."), "", button_style
    df = pd.read_json(json_data, orient='split')
    if df.empty:
        return html.P("No resolutions found for the initial criteria."), "Total Results: 0", button_style
    filtered_df = df.copy()
    try:
        if c1 and not c2:
            filtered_df = filtered_df.dropna(subset=[c1])
            if single_vote and single_vote != 'NO_FILTER':
                filtered_df = filtered_df[filtered_df[c1] == single_vote]
        elif c1 and c2:
            filtered_df = filtered_df.dropna(subset=[c1, c2])
            if agreement == 'AGREED':
                filtered_df = filtered_df[filtered_df[c1] == filtered_df[c2]]
            elif agreement == 'DISAGREED':
                filtered_df = filtered_df[filtered_df[c1] != filtered_df[c2]]
            elif agreement == 'STRONGLY_DISAGREED':
                cond1 = (filtered_df[c1] == 'Y') & (filtered_df[c2] == 'N')
                cond2 = (filtered_df[c1] == 'N') & (filtered_df[c2] == 'Y')
                filtered_df = filtered_df[cond1 | cond2]
    except Exception as e:
         return html.P(f"An error occurred during filtering: {e}"), "Error", button_style
    if filtered_df.empty:
        return html.P("No resolutions matched all filter criteria."), "Total Results: 0", button_style
    total_found = len(filtered_df)
    current_page = n_clicks if n_clicks else 0
    num_to_show = (current_page + 1) * PAGE_SIZE
    display_df = filtered_df.head(num_to_show)
    num_shown = len(display_df)
    summary = f"Displaying {num_shown} of {total_found} resolutions."
    if num_shown < total_found:
        button_style = {'width': '100%', 'marginTop': '10px', 'display': 'block'}
    output_list = []
    for _, row in display_df.iterrows():
        date_str = pd.to_datetime(row['date']).strftime('%Y-%m-%d')
        markdown_text = f"**[{row['resolution']}]({row['undl_link']}) -- {date_str}**: {row['title'].split(' :')[0]}: {row['agenda_title']}"
        indicator_divs = []
        if c1: indicator_divs.append(create_vote_indicator(c1, row.get(c1)))
        if c2: indicator_divs.append(create_vote_indicator(c2, row.get(c2)))
        output_list.append(dcc.Markdown(markdown_text))
        if indicator_divs:
            output_list.append(html.Div(indicator_divs, style={'marginTop': '5px'}))
        output_list.append(html.Hr(style={'margin': '5px 0'}))
    return output_list, summary, button_style


# --- [MODIFIED FUNCTION] ---
@app.callback(
    Output('analysis-output', 'children'),
    Input('analysis-data-store', 'data'),
    prevent_initial_call=True
)
def display_analysis_results(json_data):
    """
    [FAST] Displays the *analysis results* from Tab 2.
    Reads the new 'disagreement_score' and sorts accordingly.
    """
    if not json_data:
        return html.P("Click 'Apply Filters' to run the analysis.")

    df = pd.read_json(json_data, orient='split').dropna(subset=['disagreement_score'])
    
    if df.empty:
        return html.P("No overlapping votes found for any top-level subjects in this period.")

    # --- NEW: Explanation of the score ---
    score_explanation = html.Details([
        html.Summary("How is this 'Disagreement Score' calculated?"),
        html.Div([
            html.P("A score from 0 to 2 is calculated for each topic, representing the average disagreement per resolution. A low score means high agreement.", style={'marginTop':'10px'}),
            html.Ul([
                html.Li([html.Strong("Yes = 1"), ", ", html.Strong("No = -1"), ", ", html.Strong("Abstain/No-Vote = 0")]),
                html.Li("Per-resolution score = abs(Country 1 Value - Country 2 Value)"),
                html.Li("Final Topic Score = Average of all per-resolution scores divided by 2 to normalize"),
                html.Li(html.Strong("Score 0:"), " Perfect Agreement (e.g., Y/Y or N/N)"),
                html.Li(html.Strong("Score 1:"), " 'Mild' Disagreement (e.g., Y/A or N/A)"),
                html.Li(html.Strong("Score 2:"), " 'Strong' Disagreement (e.g., Y/N)")
            ])
        ])
    ])

    # --- Helper function to create list items ---
    def create_list_item(row):
        return html.Li([
            f"{row['subject_label']}: ",
            html.Span(f"Score: {row['disagreement_score']:.2f}", style={'fontWeight': 'bold'}),
            f" (on {row['total_votes']} votes)"
        ])

    # --- Generate Top 5 Agreed (LOWEST score) ---
    top_5_agreed = df.sort_values(by='disagreement_score', ascending=True).head(5)
    agreed_list = [create_list_item(row) for _, row in top_5_agreed.iterrows()]
    
    # --- Generate Top 5 Disagreed (HIGHEST score) ---
    top_5_disagreed = df.sort_values(by='disagreement_score', ascending=False).head(5)
    disagreed_list = [create_list_item(row) for _, row in top_5_disagreed.iterrows()]

    return [
        score_explanation,
        html.Hr(),
        html.H5("Top 5 Agreed Topics (Lowest Score)"),
        html.P("Topics with the lowest average disagreement score (closest to 0)."),
        html.Ol(agreed_list),
        html.Hr(),
        html.H5("Top 5 Disagreed Topics (Highest Score)"),
        html.P("Topics with the highest average disagreement score (closest to 1)."),
        html.Ol(disagreed_list)
    ]

# --- Run the App ---
if __name__ == '__main__':
    app.run(debug=True)