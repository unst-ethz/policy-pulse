"""
Alignment by Subject Feature

This component visualizes voting alignment between two countries across
different UN subject areas, showing which topics they agree or disagree on.
"""

from dash import dcc, Input, Output, callback, html, dash_table
import plotly.express as px
import pandas as pd
import numpy as np
import datetime

from ..data import get_country_name, TOP_LEVEL_SUBJECTS, SUBJECT_ID_TO_LABEL_MAP, get_earliest_data_date, get_latest_data_date


# --- Constants ---
MIN_VOTES_THRESHOLD = 30  # Minimum votes required for a subject to be included


def calculate_agreement(query_engine, c1, c2, start_date, end_date, subject_list, subject_map):
    """
    Calculates agreement scores between two countries across subjects.
    
    Optimized to query agreement once and then filter by subject.
    
    Agreement Score ranges from 0 (complete disagreement) to 1 (complete agreement)
    
    Returns a DataFrame with columns: subject_id, subject_label, agreement_score, total_votes
    """
    # 1. Get all resolutions in the date range
    all_resolutions = query_engine.query_resolutions(
        start_date=start_date,
        end_date=end_date
    )
    
    if all_resolutions.empty:
        return pd.DataFrame()
    
    # 2. Query agreement between the two countries for ALL resolutions (not averaged)
    # This returns a DataFrame with columns: undl_id, c2 (and other countries we don't need)
    agreement_df = query_engine.query_agreement_between_countries(
        country_code=c1,
        #resolution_ids=all_resolutions['undl_id'].tolist(),
        average=False
    )
    
    if agreement_df.empty or c2 not in agreement_df.columns:
        return pd.DataFrame()
    
    # 3. Keep only the columns we need: undl_id and the target country's agreement
    agreement_df = agreement_df[['undl_id', c2]].copy()
    agreement_df.rename(columns={c2: 'agreement_score'}, inplace=True)
    
    # 4. Get resolution-subject mappings
    # We need to know which subjects each resolution belongs to
    resolution_subject_df = query_engine.resolution_subject_table.copy()
    
    # 5. For each subject, calculate the disagreement score
    agreement_results = []
    
    for subject_id in subject_list:
        # Find all resolutions with this subject (including descendants)
        if hasattr(query_engine, 'closure_table'):
            # Get descendants of this subject
            descendants = query_engine.closure_table[
                query_engine.closure_table['ancestor_id'] == subject_id
            ]['descendant_id'].unique()
            subject_filter = list(set([subject_id] + list(descendants)))
        else:
            subject_filter = [subject_id]
        
        # Get resolution IDs for this subject
        subject_resolution_ids = resolution_subject_df[
            resolution_subject_df['subject_id'].isin(subject_filter)
        ]['undl_id'].unique()
        
        # Filter agreement scores to only resolutions with this subject
        subject_agreements = agreement_df[
            agreement_df['undl_id'].isin(subject_resolution_ids)
        ].copy()
        
        # Remove NaN values
        subject_agreements = subject_agreements.dropna(subset=['agreement_score'])
        
        total_votes = len(subject_agreements)
        
        # Skip if insufficient votes
        if total_votes < MIN_VOTES_THRESHOLD:
            continue
        
        if total_votes == 0:
            agreement_results.append({
                'subject_id': subject_id,
                'subject_label': subject_map.get(subject_id, f"ID: {subject_id}"),
                'agreement_score': np.nan,
                'total_votes': 0
            })
            continue
        
        # Calculate agreement score
        # Agreement score is already 0-1 where 1=full agreement, 0=full disagreement
        agreement_score = subject_agreements['agreement_score'].mean()
        
        # Store results
        agreement_results.append({
            'subject_id': subject_id,
            'subject_label': subject_map.get(subject_id, f"ID: {subject_id}"),
            'agreement_score': agreement_score,
            'total_votes': total_votes
        })
    
    return pd.DataFrame(agreement_results)



def register_callbacks(query_engine):
    """Register callbacks for the alignment by subject feature."""
    
    @callback(
        [
            Output("alignment-by-subject-graph", "figure"),
            Output("alignment-by-subject-graph", "style"),
            Output("alignment-by-subject-table", "children"),
            Output("alignment-by-subject-status", "children"),
        ],
        [
            Input("country1-iso-alpha3", "data"),
            Input("subject-country2-dropdown", "value"),
            Input("subject-date-picker-range", "start_date"),
            Input("subject-date-picker-range", "end_date"),
        ],
        prevent_initial_call=False
    )
    def update_subject_alignment(c1, c2, start_date, end_date):
        """Update the alignment by subject visualization."""
        if not c1 or not c2:
            return {}, {'display': 'none'}, None, "Please select two countries."
        
        if c1 == c2:
            return {}, {'display': 'none'}, None, "Please select two different countries."
        
        # Calculate alignment
        df = calculate_agreement(
            query_engine, c1, c2, start_date, end_date, 
            TOP_LEVEL_SUBJECTS, SUBJECT_ID_TO_LABEL_MAP
        )
        
        if df.empty:
            return {}, {'display': 'none'}, None, "No common votes found for the selected criteria (minimum 30 votes per subject required)."
        
        # Sort by agreement score (descending: Agreement -> Disagreement)
        df = df.sort_values('agreement_score', ascending=False)
        
        # Create Bar Chart
        fig = px.bar(
            df, 
            x='agreement_score', 
            y='subject_label', 
            orientation='h',
            title=f"Voting Alignment: {get_country_name(c1)} vs {get_country_name(c2)} by Subject",
            labels={'agreement_score': 'Agreement Score', 'subject_label': 'Subject'},
            hover_data=['total_votes'],
            color='agreement_score',
            range_color=[0, 1],
            color_continuous_scale=[
                (0.0, "rgb(203, 24, 29)"),     # Dark Red (disagreement)
                (0.5, "rgb(251, 180, 174)"),   # Light Red
                (0.5, "rgb(179, 205, 227)"),   # Light Blue
                (1.0, "rgb(0, 68, 136)")       # Dark Blue (agreement)
            ],
            height=max(600, len(df) * 30)  # Dynamic height based on number of subjects
        )
        
        fig.update_layout(
            yaxis={'categoryorder': 'total ascending'},  # Sort bars by value
            xaxis={'range': [0, 1]},
            template="plotly_white",
            coloraxis_colorbar=dict(
                title="Score",
                tickvals=[0, 0.25, 0.5, 0.75, 1.0],
                ticktext=["0 (Disagree)", "0.25", "0.5", "0.75", "1 (Agree)"]
            )
        )
        
        # Create data table
        table = html.Div([
            html.Hr(),
            html.H4("Subject Alignment Data"),
            dash_table.DataTable(
                data=df.to_dict('records'),
                columns=[
                    {'name': 'Subject', 'id': 'subject_label'},
                    {
                        'name': 'Agreement Score', 
                        'id': 'agreement_score', 
                        'type': 'numeric', 
                        'format': dash_table.Format.Format(precision=3)
                    },
                    {'name': 'Total Votes', 'id': 'total_votes'}
                ],
                sort_action='native',
                page_size=20,
                style_cell={'textAlign': 'left'},
                style_header={'fontWeight': 'bold'},
                style_data_conditional=[
                    {
                        'if': {'column_id': 'agreement_score'},
                        'textAlign': 'right'
                    },
                    {
                        'if': {'column_id': 'total_votes'},
                        'textAlign': 'right'
                    }
                ]
            )
        ])
        
        status_msg = html.Div([
            html.Strong("Chart Updated Successfully! "),
            f"Analyzed {len(df)} subjects with sufficient voting data."
        ])
        
        return fig, {'display': 'block'}, table, status_msg


# Layout for the alignment by subject feature
layout = [
    html.Div(
        [
            html.Div(
                [
                    html.Label(
                        "Select a country to compare with:",
                        style={"fontWeight": "bold", "marginBottom": "5px"},
                    ),
                    dcc.Dropdown(
                        id="subject-country2-dropdown",
                        options=[],  # Will be populated dynamically
                        value=None,
                        clearable=False,
                        placeholder="Select a country...",
                        style={"marginBottom": "15px"},
                    ),
                ],
                style={"width": "30%", "display": "inline-block", "marginRight": "20px"},
            ),
            html.Div(
                [
                    html.Label(
                        "Date Range:",
                        style={"fontWeight": "bold", "marginBottom": "5px"},
                    ),
                    dcc.DatePickerRange(
                        id="subject-date-picker-range",
                        min_date_allowed=get_earliest_data_date(),
                        max_date_allowed=get_latest_data_date(),
                        start_date=get_earliest_data_date(),
                        end_date=get_latest_data_date(),
                        display_format='YYYY-MM-DD',
                        style={"marginBottom": "15px"},
                    ),
                ],
                style={"width": "40%", "display": "inline-block"},
            ),
        ],
        style={"marginBottom": "20px"}
    ),
    html.Hr(),
    html.Div(id="alignment-by-subject-status", style={"marginBottom": "10px"}),
    dcc.Loading(
        id="alignment-by-subject-loading",
        type="circle",
        color="#3498db",
        children=[
            dcc.Graph(
                id="alignment-by-subject-graph",
                style={'display': 'none'}
            ),
        ],
    ),
    html.Div(id="alignment-by-subject-table"),
]
