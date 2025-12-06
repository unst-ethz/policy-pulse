from dash import dcc, html, Input, Output, State, no_update, callback
import pandas as pd
import datetime
from .. import data

PAGE_SIZE = 50

# Prepare subject options for dropdown
subject_options_list = data.subject_table.to_dict('records')
subject_options = [{"label": r["label_en"], "value": r["subject_id"]} for r in subject_options_list]

# Prepare country options for dropdown
country_options_list = [{"label": data.get_country_name(c), "value": c} for c in data.available_countries]


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


layout = html.Div([
    dcc.Store(id='rf-filtered-data-store'),
    
    html.Div(className='row', children=[
        html.Div(className='four columns', style={'border': '1px solid #ddd', 'padding': '10px', 'borderRadius': '5px', 'marginBottom': '20px'}, children=[
            html.H3("Filters"),
            html.Label("Date Range:"),
            dcc.DatePickerRange(
                id='rf-date-picker-range',
                min_date_allowed=data.MIN_UN_DATE,
                max_date_allowed=data.MAX_UN_DATE,
                start_date=data.MIN_UN_DATE,
                end_date=data.MAX_UN_DATE,
                display_format='YYYY-MM-DD'
            ),
            html.Hr(),
            html.Label("Subjects:"),
            dcc.Dropdown(id='rf-subject-dropdown', options=subject_options, multi=True, placeholder="Filter by subjects..."),
            html.Hr(),
            # Country 1 is implicitly the page's country
            html.Label("Compare with Country 2:"),
            dcc.Dropdown(id='rf-country-2-dropdown', options=country_options_list, placeholder="Select second country (optional)...", clearable=True),
            
            html.Div(id='rf-single-country-filter-div', style={'display': 'none'}, children=[
                html.Hr(),
                html.Label("Country 1 Vote (for Filtered List):"),
                dcc.RadioItems(
                    id='rf-single-vote-radio',
                    options=[
                        {'label': 'No Filter', 'value': 'NO_FILTER'},
                        {'label': 'Voted Yes', 'value': 'Y'},
                        {'label': 'Voted No', 'value': 'N'},
                        {'label': 'Abstained', 'value': 'A'},
                        {'label': "Didn't Vote", 'value': 'X'}
                    ],
                    value='NO_FILTER',
                    labelStyle={'display': 'block'}
                )
            ]),
            html.Div(id='rf-two-country-filter-div', style={'display': 'none'}, children=[
                html.Hr(),
                html.Label("Country Agreement (for Filtered List):"),
                dcc.RadioItems(
                    id='rf-agreement-radio',
                    options=[
                        {'label': 'No Filter', 'value': 'NO_FILTER'},
                        {'label': 'Agreed (Voted Same)', 'value': 'AGREED'},
                        {'label': 'Disagreed (Voted Differently)', 'value': 'DISAGREED'},
                        {'label': 'Strongly Disagreed (Y/N vs N/Y)', 'value': 'STRONGLY_DISAGREED'}
                    ],
                    value='NO_FILTER',
                    labelStyle={'display': 'block'}
                )
            ]),
        ]),
        html.Div(className='eight columns', children=[
            dcc.Loading(id='rf-loading-spinner', type='default', children=[
                html.H4(id='rf-results-summary'),
                html.Div(id='rf-results-output', style={'maxHeight': '70vh', 'overflowY': 'auto', 'border': '1px solid #eee', 'padding': '10px'}),
                html.Button("Load More", id='rf-load-more-button', n_clicks=0, style={'width': '100%', 'marginTop': '10px', 'display': 'none'})
            ])
        ])
    ])
], style={'padding': '20px'})


def register_callbacks(query_engine):
    
    @callback(
        Output('rf-single-country-filter-div', 'style'),
        Output('rf-two-country-filter-div', 'style'),
        Input('country1-iso-alpha3', 'data'),
        Input('rf-country-2-dropdown', 'value')
    )
    def update_country_filter_ui(country_1, country_2):
        if country_1 and country_2:
            return {'display': 'none'}, {'display': 'block'}
        elif country_1 and not country_2:
            return {'display': 'block'}, {'display': 'none'}
        else:
            return {'display': 'none'}, {'display': 'none'}

    @callback(
        Output('rf-filtered-data-store', 'data'),
        Output('rf-results-summary', 'children', allow_duplicate=True),
        Output('rf-load-more-button', 'n_clicks'),
        Input('rf-date-picker-range', 'start_date'),
        Input('rf-date-picker-range', 'end_date'),
        Input('rf-subject-dropdown', 'value'),
        prevent_initial_call=True
    )
    def query_base_resolutions(start_date, end_date, subject_ids):
        subjects = subject_ids if subject_ids else None
        start = start_date if start_date else None
        end = end_date if end_date else None
        df = query_engine.query_resolutions(start_date=start, end_date=end, subject_ids=subjects)
        return df.to_json(date_format='iso', orient='split'), f"Found {len(df)} resolutions. Applying country filters...", 0

    @callback(
        Output('rf-results-output', 'children'),
        Output('rf-results-summary', 'children'),
        Output('rf-load-more-button', 'style'), 
        Input('rf-filtered-data-store', 'data'), 
        Input('rf-load-more-button', 'n_clicks'),
        Input('country1-iso-alpha3', 'data'),
        Input('rf-country-2-dropdown', 'value'),
        Input('rf-single-vote-radio', 'value'),
        Input('rf-agreement-radio', 'value'),
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
                # Check if c1 is in columns (it might not be if data is filtered weirdly or c1 is invalid)
                if c1 in filtered_df.columns:
                    filtered_df = filtered_df.dropna(subset=[c1])
                    if single_vote and single_vote != 'NO_FILTER':
                        filtered_df = filtered_df[filtered_df[c1] == single_vote]
            elif c1 and c2:
                if c1 in filtered_df.columns and c2 in filtered_df.columns:
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
            # Use get_country_name for display if possible, but c1/c2 are codes
            c1_name = data.get_country_name(c1)
            c2_name = data.get_country_name(c2) if c2 else None
            
            markdown_text = f"**[{row['resolution']}]({row['undl_link']}) -- {date_str}**: {row['title'].split(' :')[0]}: {row['agenda_title']}. Total countries that voted yes: {int(row['total_yes'])}"
            indicator_divs = []
            if c1 and c1 in row: indicator_divs.append(create_vote_indicator(c1_name, row.get(c1)))
            if c2 and c2 in row: indicator_divs.append(create_vote_indicator(c2_name, row.get(c2)))
            output_list.append(dcc.Markdown(markdown_text))
            if indicator_divs:
                output_list.append(html.Div(indicator_divs, style={'marginTop': '5px'}))
            output_list.append(html.Hr(style={'margin': '5px 0'}))
        return output_list, summary, button_style
