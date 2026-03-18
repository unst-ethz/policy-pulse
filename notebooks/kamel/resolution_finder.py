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


import urllib.parse

# --- Initialize the Dash App ---
app = dash.Dash(__name__, external_stylesheets=['https.codepen.io/chriddyp/pen/bWLwgP.css'])
server = app.server

# --- Define the App Layout ---
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div([
        html.H1("UN Resolution Explorer", style={'display': 'inline-block', 'marginRight': '20px'}),
        html.A("Go to Country Agreement", id='nav-link', href="http://127.0.0.1:8050/", target="_self", style={'float': 'right', 'marginTop': '20px', 'fontSize': '18px', 'textDecoration': 'none', 'border': '1px solid #007BFF', 'padding': '10px', 'borderRadius': '5px', 'color': '#007BFF'})
    ], style={'marginBottom': '20px'}),
    
    dcc.Store(id='filtered-data-store'),
    
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
        ]),
        html.Div(className='eight columns', children=[
            dcc.Loading(id='loading-spinner', type='default', children=[
                html.H4(id='results-summary'),
                html.Div(id='results-output', style={'maxHeight': '70vh', 'overflowY': 'auto', 'border': '1px solid #eee', 'padding': '10px'}),
                html.Button("Load More", id='load-more-button', n_clicks=0, style={'width': '100%', 'marginTop': '10px', 'display': 'none'})
            ])
        ])
    ])
], style={'padding': '20px'})


# --- Callbacks ---

@app.callback(
    Output('country-1-dropdown', 'value'),
    Output('country-2-dropdown', 'value'),
    Output('date-picker-range', 'start_date'),
    Output('date-picker-range', 'end_date'),
    Output('subject-dropdown', 'value'),
    Input('url', 'search')
)
def initialize_filters_from_url(search):
    if not search:
        return no_update, no_update, no_update, no_update, no_update
    
    params = urllib.parse.parse_qs(search.lstrip('?'))
    
    c1 = params.get('c1', [None])[0]
    c2 = params.get('c2', [None])[0]
    start = params.get('start', [None])[0]
    end = params.get('end', [None])[0]
    subject = params.get('subject', [None])[0]
    
    # Handle subject being a list or single item, though parse_qs returns lists
    # If subject is present, we might want to ensure it's in the options or format it correctly
    # The dropdown expects a list for 'multi=True'
    subject_val = [subject] if subject else no_update
    
    return (
        c1 if c1 else no_update,
        c2 if c2 else no_update,
        start if start else no_update,
        end if end else no_update,
        subject_val
    )

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

# (toggle_analysis_tab callback removed)

# (query_base_resolutions callback modified for instant updates)
@app.callback(
    Output('filtered-data-store', 'data'),
    Output('results-summary', 'children', allow_duplicate=True),
    Output('load-more-button', 'n_clicks'),
    Input('date-picker-range', 'start_date'),
    Input('date-picker-range', 'end_date'),
    Input('subject-dropdown', 'value'),
    prevent_initial_call=True
)
def query_base_resolutions(start_date, end_date, subject_ids):
    subjects = subject_ids if subject_ids else None
    start = start_date if start_date else None
    end = end_date if end_date else None
    df = analyzer.query_resolutions(start_date=start, end_date=end, subject_ids=subjects)
    return df.to_json(date_format='iso', orient='split'), f"Found {len(df)} resolutions. Applying country filters...", 0

# (run_agreement_analysis callback removed)

# (display_filtered_results callback for Tab 1 remains the same)
@app.callback(
    Output('results-output', 'children'),
    Output('results-summary', 'children'),
    Output('load-more-button', 'style'), 
    Input('filtered-data-store', 'data'), 
    Input('load-more-button', 'n_clicks'),
    Input('country-1-dropdown', 'value'),
    Input('country-2-dropdown', 'value'),
    Input('single-vote-radio', 'value'),
    Input('agreement-radio', 'value'),
    prevent_initial_call=True
)
def display_filtered_results(json_data, n_clicks, c1, c2, single_vote, agreement):
    button_style = {'display': 'none'}
    if not json_data:
        return html.P("Adjust filters to load data."), "", button_style
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
        markdown_text = f"**[{row['resolution']}]({row['undl_link']}) -- {date_str}**: {row['title'].split(' :')[0]}: {row['agenda_title']}. Total countries that voted yes: {int(row['total_yes'])}"
        indicator_divs = []
        if c1: indicator_divs.append(create_vote_indicator(c1, row.get(c1)))
        if c2: indicator_divs.append(create_vote_indicator(c2, row.get(c2)))
        output_list.append(dcc.Markdown(markdown_text))
        if indicator_divs:
            output_list.append(html.Div(indicator_divs, style={'marginTop': '5px'}))
        output_list.append(html.Hr(style={'margin': '5px 0'}))
    return output_list, summary, button_style

# (display_analysis_results callback removed)

@app.callback(
    Output('nav-link', 'href'),
    Input('country-1-dropdown', 'value'),
    Input('country-2-dropdown', 'value'),
    Input('date-picker-range', 'start_date'),
    Input('date-picker-range', 'end_date')
)
def update_nav_link(c1, c2, start, end):
    base_url = "http://127.0.0.1:8050/"
    params = {}
    if c1: params['c1'] = c1
    if c2: params['c2'] = c2
    if start: params['start'] = start
    if end: params['end'] = end
    
    if params:
        return f"{base_url}?{urllib.parse.urlencode(params)}"
    return base_url

# --- Run the App ---
if __name__ == '__main__':
    app.run(debug=True, port=8051)