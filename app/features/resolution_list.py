from dash import callback, Input, Output, State, html, dcc
import pandas as pd
from .. import data
import pandas as pd
from pathlib import Path

cwd = Path(__file__).resolve().parent
file_path = cwd.parent / "assets" / "joining_dates.csv"
joining_dates = pd.read_csv(file_path)


PAGE_SIZE = 10
LOAD_MORE_SIZE = 50

def create_vote_indicator(country_name, vote):
    VOTE_MAP = {
        'Y': {'color': 'green', 'label': 'Yes'},
        'N': {'color': 'red', 'label': 'No'},
        'A': {'color': 'orange', 'label': 'Abstain'},
        'X': {'color': 'blue', 'label': 'Not Voting'}
    }
    if pd.isna(vote) or vote not in VOTE_MAP:
         return html.Span(f"{country_name}: N/A", style={'color': '#999', 'fontSize': '0.85em', 'marginRight': '10px'})
    
    config = VOTE_MAP[vote]
    return html.Span([
        html.Span('●', style={'color': config['color'], 'marginRight': '4px'}),
        html.Span(f"{country_name}: {config['label']}")
    ], style={'fontWeight': '500', 'marginRight': '15px', 'fontSize': '0.9em'})


layout = [html.Div([
    # --- Local Controls ---
    html.Div([
        # Agreement Filter (Only visible when exactly 1 comparison country selected)
        html.Div(id='rl-agreement-filter-container', style={'display': 'none'}, children=[
            html.Label("Filter by Agreement:", style={'fontWeight': 'bold', 'marginRight': '10px'}),
            dcc.Dropdown(
                id='rl-agreement-dropdown',
                options=[
                    {'label': 'Show All', 'value': 'NO_FILTER'},
                    {'label': 'Agreed (Voted Same)', 'value': 'AGREED'},
                    {'label': 'Disagreed (Voted Differently)', 'value': 'DISAGREED'},
                    {'label': 'Strongly Disagreed (Y/N vs N/Y)', 'value': 'STRONGLY_DISAGREED'}
                ],
                value='NO_FILTER',
                clearable=False,
                style={'width': '250px', 'display': 'inline-block', 'verticalAlign': 'middle'}
            )
        ], className='fade-in'),
        
        # Placeholder or info text when multiple countries selected? 
        # Making it empty if not single country usually looks cleaner.
        html.Div(id='rl-multi-country-msg', style={'display': 'none', 'color': '#666', 'fontStyle': 'italic'})
        
    ], style={'marginBottom': '20px', 'minHeight': '5px'}), # Keep some space

    # --- Results Area ---
    dcc.Loading(id='rl-loading', type='dot', children=[
        html.Div(id='rl-results-summary', style={'marginBottom': '10px', 'fontStyle': 'italic', 'color': '#666'}),
        html.Div(id='rl-results-list'),
        html.Button("Load More Resolutions", id='rl-load-more-btn', n_clicks=0, 
                    style={'width': '100%', 'marginTop': '15px', 'display': 'none', 'padding': '10px'})
    ])
])]


def register_callbacks():
    
    # 1. Main Logic: Query, Filter & Render
    @callback(
        Output('rl-results-list', 'children'),
        Output('rl-results-summary', 'children'),
        Output('rl-load-more-btn', 'style'),
        Output('rl-agreement-filter-container', 'style'),
        Output('rl-multi-country-msg', 'children'),
        Output('rl-multi-country-msg', 'style'),
        Input('filter-component-filter-store', 'data'),     # Global Filter PARAMETERS (Start, End, Subject, Country1, Country2)
        Input('rl-agreement-dropdown', 'value'),            # Local Filter (Conditional)
        Input('rl-load-more-btn', 'n_clicks'),              # Pagination
        State('country1-iso-alpha3', 'data')                # Current Main Country (fallback)
    )
    def update_resolution_list(filter_params, agreement_filter, n_clicks, country1_backup):
        # Default Styles
        btn_style_hidden = {'display': 'none'}
        btn_style_visible = {'width': '100%', 'marginTop': '15px', 'display': 'block', 'padding': '10px', 'cursor': 'pointer'}
        
        if not filter_params:
             return html.Div("Loading...", style={'padding': '20px'}), "", btn_style_hidden, {'display': 'none'}, "", {'display': 'none'}

        # Extract Params
        start_date = filter_params.get("start_date")
        end_date = filter_params.get("end_date")
        subject_ids = filter_params.get("subject_ids")
        # Try both the params and backup store for country1, but allow it to be None
        country1 = filter_params.get("country1_alpha3") or country1_backup
        
        # Extract comparison countries (can be list, string or None)
        country2_raw = filter_params.get("country2")
        comparison_countries = []
        if isinstance(country2_raw, list):
            comparison_countries = country2_raw
        elif isinstance(country2_raw, str) and country2_raw:
            comparison_countries = [country2_raw]

        # UI State Logic: Agreement Filter Visibility
        show_agreement_filter = False
        agreement_container_style = {'display': 'none'}
        multi_msg = ""
        multi_msg_style = {'display': 'none'}
        
        if len(comparison_countries) == 1:
            # Only enable agreement filter if we have BOTH a main country AND exactly 1 comparison country
            if country1:
                show_agreement_filter = True
                agreement_container_style = {'display': 'block', 'padding': '15px', 'backgroundColor': '#f1f3f4', 'borderRadius': '8px'}
        elif len(comparison_countries) > 1:
            multi_msg = f"Comparing against top 5 of {len(comparison_countries)} selected countries. Agreement filter disabled for multi-select."
            multi_msg_style = {'display': 'block', 'marginBottom': '10px'}

        # 1. Query Data
        try:
            underlying_countries = []
            for i in comparison_countries:
                underlying_countries.append(i)
            if country1:
                underlying_countries.append(country1)
            if len(underlying_countries) == 0:
                 df = data.query_engine.query_resolutions(
                    start_date=start_date,
                    end_date=end_date,
                    subject_ids=subject_ids,
                    include_descendants=True
                )
            else:
               min_starting_date = '2099-01-01'
               for c in underlying_countries:
                  if joining_dates[joining_dates['country'] == c]['min_date'].to_list()[0] < min_starting_date:
                     min_starting_date = joining_dates[joining_dates['country'] == c]['min_date'].to_list()[0]
               df = data.query_engine.query_resolutions(
                start_date= start_date if min_starting_date < start_date else min_starting_date,
                end_date=end_date,
                subject_ids=subject_ids,
                include_descendants=True
            )
        except Exception as e:
            return html.Div(f"Error querying data: {e}", style={'color': 'red'}), "", btn_style_hidden, agreement_container_style, multi_msg, multi_msg_style

        if df.empty:
             return html.Div("No resolutions found.", style={'padding': '20px', 'color': '#777'}), "0 results", btn_style_hidden, agreement_container_style, multi_msg, multi_msg_style

        # 2. Filter Logic
        filtered_df = df.copy()
        
        if country1 and show_agreement_filter and agreement_filter != 'NO_FILTER':
            c2 = comparison_countries[0]
            if country1 in filtered_df.columns and c2 in filtered_df.columns:
                 filtered_df = filtered_df.dropna(subset=[country1, c2])
                 if agreement_filter == 'AGREED':
                    filtered_df = filtered_df[filtered_df[country1] == filtered_df[c2]]
                 elif agreement_filter == 'DISAGREED':
                    filtered_df = filtered_df[filtered_df[country1] != filtered_df[c2]]
                 elif agreement_filter == 'STRONGLY_DISAGREED':
                    cond1 = (filtered_df[country1] == 'Y') & (filtered_df[c2] == 'N')
                    cond2 = (filtered_df[country1] == 'N') & (filtered_df[c2] == 'Y')
                    filtered_df = filtered_df[cond1 | cond2]

        total_count = len(filtered_df)
        
        # 3. Pagination
        current_limit = PAGE_SIZE + ((n_clicks or 0) * LOAD_MORE_SIZE)
        display_df = filtered_df.head(current_limit)
        shown_count = len(display_df)
        
        # 4. Render
        summary = f"Showing {shown_count} of {total_count} resolutions"
        if len(comparison_countries) > 0:
            summary += f" (Comparing with: {', '.join([data.get_country_name(c) for c in comparison_countries[:3]])}{'...' if len(comparison_countries)>3 else ''})"

        btn_style = btn_style_visible if shown_count < total_count else btn_style_hidden

        cards = []
        for _, row in display_df.iterrows():
            # Basic Info
            res_id = row.get('resolution', 'N/A')
            link = row.get('undl_link', '#')
            date_val = row.get('date')
            date_str = date_val.strftime('%Y-%m-%d') if pd.notnull(date_val) else "Unknown"
            title = row.get('title', 'Untitled')
            
            # Vote Indicators
            indicators = []
            
            # Main Country (only if selected)
            if country1:
                c1_vote = row.get(country1) if country1 in row else None
                indicators.append(create_vote_indicator(data.get_country_name(country1), c1_vote))
            
            # Comparators (Force limit to 5 to avoid UI clutter)
            for c2 in comparison_countries[:5]:
                if c2 in row:
                    indicators.append(create_vote_indicator(data.get_country_name(c2), row.get(c2)))
            
            card = html.Div([
                html.Div([
                    html.A(
                        html.H4([html.I(className="fas fa-file-pdf", style={'marginRight': '8px'}), f"{res_id}"], 
                               style={'marginBottom': '5px', 'color': '#007bff', 'display': 'inline-block'}),
                        href=link, target="_blank", style={'textDecoration': 'none'}
                    ),
                    html.Span(date_str, style={'float': 'right', 'color': '#666', 'fontSize': '0.9em'})
                ]),
                html.Div(title, style={'fontWeight': 'bold', 'marginBottom': '5px', 'fontSize': '1.05em'}),
                
                html.Div(indicators, style={'marginTop': '10px', 'paddingTop': '10px', 'borderTop': '1px solid #eee', 'display': 'flex', 'flexWrap': 'wrap', 'gap': '5px'}) if indicators else None
                
            ], className='resolution-card', style={
                'backgroundColor': 'white',
                'border': '1px solid #e0e0e0',
                'borderRadius': '8px',
                'padding': '15px',
                'marginBottom': '15px',
                'boxShadow': '0 2px 4px rgba(0,0,0,0.05)'
            })
            cards.append(card)
            
        return cards, summary, btn_style, agreement_container_style, multi_msg, multi_msg_style
