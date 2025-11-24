import dash
import urllib.parse
from dash import dcc, html, Input, Output, State, no_update
import pandas as pd
import datetime
import numpy as np
import plotly.express as px
from pathlib import Path
from unDataStream import DataRepository
from unDataStream import ResolutionQueryEngine

# --- Configuration & Data Loading ---
# Using the same config path as resolution_finder.py
config_path = Path('config/data_sources.yaml')
repo = DataRepository(config_path)
data = repo.get_data()
analyzer = ResolutionQueryEngine(repo)

# --- Constants ---
MIN_UN_DATE = datetime.date(1945, 1, 1)
MAX_UN_DATE = datetime.date.today()

# --- Prepare Options ---
# Subject options
subject_options_list = data["subject"].to_dict('records')
# Map subject_id to label for easy lookup
SUBJECT_ID_TO_LABEL_MAP = {r["subject_id"]: r["label_en"] for r in subject_options_list}

# Country options
country_options = []
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

# Top level subjects (copied from resolution_finder.py for consistency)
level_zero_subjects = {'http://metadata.un.org/thesaurus/10', 'http://metadata.un.org/thesaurus/09', 'http://metadata.un.org/thesaurus/16', 'http://metadata.un.org/thesaurus/00', 'http://metadata.un.org/thesaurus/07', 'http://metadata.un.org/thesaurus/04', 'http://metadata.un.org/thesaurus/06', 'http://metadata.un.org/thesaurus/15', 'http://metadata.un.org/thesaurus/05', 'http://metadata.un.org/thesaurus/03', 'http://metadata.un.org/thesaurus/17', 'http://metadata.un.org/thesaurus/11', 'http://metadata.un.org/thesaurus/12', 'http://metadata.un.org/thesaurus/13', 'http://metadata.un.org/thesaurus/14', 'http://metadata.un.org/thesaurus/18', 'http://metadata.un.org/thesaurus/08', 'http://metadata.un.org/thesaurus/01', 'http://metadata.un.org/thesaurus/02'}
TOP_LEVEL_SUBJECTS = level_zero_subjects

# --- Helper Function ---
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
            'subject_id': subject_id,
            'subject_label': subject_map.get(subject_id, f"ID: {subject_id}"),
            'disagreement_score': disagreement_score,
            'total_votes': total_votes
        })
        
    return pd.DataFrame(agreement_results)
# --- Dash App ---
app = dash.Dash(__name__, external_stylesheets=['https.codepen.io/chriddyp/pen/bWLwgP.css'])
server = app.server

app.layout = html.Div([
    html.H1("Country Alignment Dashboard"),
    html.Div(className='row', style={'padding': '20px', 'backgroundColor': '#f9f9f9', 'borderRadius': '5px'}, children=[
        html.Div(className='four columns', children=[
            html.Label("Country 1:"),
            dcc.Dropdown(id='country-1-dropdown', options=country_options_list, value='United States', clearable=False),
        ]),
        html.Div(className='four columns', children=[
            html.Label("Country 2:"),
            dcc.Dropdown(id='country-2-dropdown', options=country_options_list, value='China', clearable=False),
        ]),
        html.Div(className='four columns', children=[
            html.Label("Date Range:"),
            dcc.DatePickerRange(
                id='date-picker-range',
                min_date_allowed=MIN_UN_DATE,
                max_date_allowed=MAX_UN_DATE,
                start_date=MIN_UN_DATE,
                end_date=MAX_UN_DATE,
                display_format='YYYY-MM-DD'
            ),
        ]),
    ]),
    html.Br(),
    # Calculate button removed for instant updates
    html.Hr(),
    dcc.Loading(id='loading', children=[
        html.Div(id='error-container', style={'color': 'red', 'marginBottom': '10px'}),
        dcc.Graph(id='alignment-graph', style={'display': 'none'}),
        html.Div(id='table-container')
    ]),
    dcc.Location(id='url', refresh=True)
])

@app.callback(
    Output('country-1-dropdown', 'value'),
    Output('country-2-dropdown', 'value'),
    Output('date-picker-range', 'start_date'),
    Output('date-picker-range', 'end_date'),
    Input('url', 'search')
)
def initialize_filters_from_url(search):
    if not search:
        return no_update, no_update, no_update, no_update
    
    params = urllib.parse.parse_qs(search.lstrip('?'))
    
    c1 = params.get('c1', [None])[0]
    c2 = params.get('c2', [None])[0]
    start = params.get('start', [None])[0]
    end = params.get('end', [None])[0]
    
    return (
        c1 if c1 else no_update,
        c2 if c2 else no_update,
        start if start else no_update,
        end if end else no_update
    )

@app.callback(
    Output('alignment-graph', 'figure'),
    Output('alignment-graph', 'style'),
    Output('table-container', 'children'),
    Output('error-container', 'children'),
    Input('country-1-dropdown', 'value'),
    Input('country-2-dropdown', 'value'),
    Input('date-picker-range', 'start_date'),
    Input('date-picker-range', 'end_date'),
    prevent_initial_call=False
)
def update_graph(c1, c2, start_date, end_date):
    if not c1 or not c2:
        return no_update, {'display': 'none'}, no_update, "Please select two countries."
    
    df = calculate_agreement(analyzer, c1, c2, start_date, end_date, TOP_LEVEL_SUBJECTS, SUBJECT_ID_TO_LABEL_MAP)
    
    if df.empty:
        return no_update, {'display': 'none'}, no_update, "No common votes found for the selected criteria."
    
    # Sort by disagreement score (ascending: Agreement -> Disagreement)
    df = df.sort_values('disagreement_score')
    
    # Create Bar Chart
    fig = px.bar(
        df, 
        x='disagreement_score', 
        y='subject_label', 
        orientation='h',
        title=f"Disagreement Score: {c1} vs {c2} (0=Agree, 1=Disagree)",
        labels={'disagreement_score': 'Disagreement Score (0-1)', 'subject_label': 'Subject'},
        hover_data=['total_votes'],
        custom_data=['subject_id'], # Add subject_id to custom_data for click event
        color='disagreement_score',
        range_color=[0, 1],
        color_continuous_scale=[
            (0.0, "blue"), 
            (0.5, "#cfe6ff"),       # Very Light Blue (approx 0.37 on original scale)
            (0.5, "#ffcfcf"),       # Very Light Red (approx 0.63 on original scale)
            (1.0, "red")
        ],
        height=800
    )
    fig.update_layout(yaxis={'categoryorder':'total descending'}, clickmode='event+select') # Sort bars
    
    table = html.Div([
        html.Hr(),
        html.H4("Data Table"),
        html.Div(
            dash.dash_table.DataTable(
                data=df.to_dict('records'),
                columns=[
                    {'name': 'Subject', 'id': 'subject_label'},
                    {'name': 'Disagreement Score', 'id': 'disagreement_score', 'type': 'numeric', 'format': dash.dash_table.Format.Format(precision=2)},
                    {'name': 'Total Votes', 'id': 'total_votes'}
                ],
                sort_action='native',
                page_size=20,
                style_cell={'textAlign': 'left'},
                style_header={'fontWeight': 'bold'}
            )
        )
    ])
    
    return fig, {'display': 'block'}, table, ""

@app.callback(
    Output('url', 'href'),
    Input('alignment-graph', 'clickData'),
    State('country-1-dropdown', 'value'),
    State('country-2-dropdown', 'value'),
    State('date-picker-range', 'start_date'),
    State('date-picker-range', 'end_date'),
    prevent_initial_call=True
)
def navigate_to_resolution_finder(clickData, c1, c2, start_date, end_date):
    if not clickData:
        return no_update
    
    # Extract subject_id from clickData
    # clickData structure: {'points': [{'curveNumber': 0, 'pointNumber': 3, 'pointIndex': 3, 'x': 0.1, 'y': 'Subject Label', 'customdata': ['subject_id'], ...}]}
    try:
        subject_id = clickData['points'][0]['customdata'][0]
    except (KeyError, IndexError):
        return no_update
        
    # Construct URL parameters
    params = {
        'c1': c1,
        'c2': c2,
        'start': start_date,
        'end': end_date,
        'subject': subject_id
    }
    
    query_string = urllib.parse.urlencode(params)
    target_url = f"http://127.0.0.1:8051/?{query_string}"
    
    return target_url

if __name__ == '__main__':
    app.run(debug=True)
