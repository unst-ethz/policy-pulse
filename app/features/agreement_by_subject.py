"""
Agreement by Subject Feature

This component visualizes voting agreement between two countries across
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
    
    agreement_df = query_engine.query_agreement_between_countries(
        country_code=c1,
        resolution_ids=all_resolutions['undl_id'].tolist(),
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
    
    # 5. For each subject, calculate the agreement score
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
    """Register callbacks for the agreement by subject feature."""

    @callback(
        Output("agreement-by-subject-country2-select", "options"),
        Output("agreement-by-subject-country2-select", "value"),
        Output("agreement-by-subject-country2-wrapper", "style"),
        Input("filter-component-filter-store", "data"),
        prevent_initial_call=False,
    )
    def update_country2_options(filter_params):
        """Populate the comparison country picker from the filter store."""
        hidden = {"display": "none"}
        visible = {"display": "flex", "alignItems": "center", "gap": "8px", "marginBottom": "10px"}

        if not filter_params:
            return [], None, hidden

        c2_input = filter_params.get("country2")
        countries = []
        if isinstance(c2_input, list):
            countries = c2_input
        elif isinstance(c2_input, str) and c2_input:
            countries = [c2_input]

        if not countries:
            return [], None, hidden

        options = [{"label": get_country_name(c), "value": c} for c in countries]
        # Default to first country
        return options, countries[0], visible

    @callback(
        [
            Output("agreement-by-subject-graph", "figure"),
            Output("agreement-by-subject-graph", "style"),
            Output("agreement-by-subject-table", "children"),
            Output("agreement-by-subject-status", "children"),
        ],
        [
            Input("filter-component-filter-store", "data"),
            Input("agreement-by-subject-country2-select", "value"),
        ],
        prevent_initial_call=False
    )
    def update_subject_agreement(filter_params, selected_c2):
        """Update the agreement by subject visualization using global filters."""
        if not filter_params:
             return {}, {'display': 'none'}, None, ""

        c1 = filter_params.get("country1_alpha3")
        start_date = filter_params.get("start_date")
        end_date = filter_params.get("end_date")

        c2 = selected_c2 or None
        
        if not c1 or not c2:
            msg = "Please select a comparison country in the sidebar settings." if c1 else "Please select a primary country."
            return {}, {'display': 'none'}, None, msg
        
        if c1 == c2:
            return {}, {'display': 'none'}, None, "Please select two different countries."
        
        # Calculate agreement
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
            title=f"Voting Agreement: {get_country_name(c1)} vs {get_country_name(c2)}",
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
                title="",  # Label is already shown on the x-axis; perhaps we don't need it again in the colourbar
                tickvals=[0, 0.25, 0.5, 0.75, 1.0],
                ticktext=["0 (Always opposed)", "0.25", "0.5", "0.75", "1 (Always agreeing)"]
            )
        )
        
        # Create data table
        table = html.Div([
            html.Hr(),
            html.H4("Subject Agreement Data"),
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
        
        status_msg = html.Div(
            f"Analyzed {len(df)} subjects based on resolutions from {start_date} to {end_date}.",
            style={"color": "#7f8c8d", "fontSize": "14px", "padding": "4px 0"},
        )
        
        return fig, {'display': 'block'}, table, status_msg


# Layout for the agreement by subject feature
# TODO: Add an annotation (explanatory caption) similar to the map and timeline tabs
layout = [
    html.Div(
        id="agreement-by-subject-country2-wrapper",
        style={"display": "none"},
        children=[
            html.Span("Comparing with:", style={"whiteSpace": "nowrap", "fontSize": "14px", "color": "#555"}),
            dcc.Dropdown(
                id="agreement-by-subject-country2-select",
                placeholder="Select comparison country…",
                clearable=False,
                style={"minWidth": "220px"},
            ),
        ],
    ),
    html.Div(id="agreement-by-subject-status", style={"marginBottom": "10px", "marginTop": "10px", "color": "#666"}),
    dcc.Loading(
        id="agreement-by-subject-loading",
        type="circle",
        color="#3498db",
        children=[
            dcc.Graph(
                id="agreement-by-subject-graph",
                style={'display': 'none'}
            ),
        ],
    ),
    html.Div(id="agreement-by-subject-table"),
]
