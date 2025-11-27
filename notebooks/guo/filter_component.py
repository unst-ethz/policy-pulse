"""
Minimal filter component for UN Resolution Explorer.
Handles filtering by time range, subject, and country.
"""

import dash
from dash import dcc, html, Input, Output, no_update
import datetime
import pandas as pd
from typing import List, Dict, Any

class FilterComponent:
    """
    A filter component for filtering UN resolutions by:
    - Date range (time)
    - Subject(s)
    - Country(ies)
    """
    
    def __init__(
        self,
        component_id_prefix: str = "filter",
    ):
        """
        Initialize the filter component.
        
        Args:
            component_id_prefix: Prefix for all component IDs to avoid conflicts
        """
        self.init_data()
        self.prefix = component_id_prefix
        
        # Component IDs
        self.ids = {
            'date_picker': f'{self.prefix}-date-picker-range',
            'subject_dropdown': f'{self.prefix}-subject-dropdown',
            'country': f'{self.prefix}-country-dropdown',
            'filter_store': f'{self.prefix}-filter-store',
            'data_store': f'{self.prefix}-data-store',  # Store for queried data
        }

    def init_data(self):
        import sys
        sys.path.append('../../app')
        from un_data_stream import DataRepository
        from un_data_stream import ResolutionQueryEngine
        from pathlib import Path

        config_path = Path('../../config/data_sources.yaml')
        repo = DataRepository(config_path)
        self.query_engine = ResolutionQueryEngine(repo)
        
        # data: Dict[str, Any]
        data = repo.get_data()

        # Prepare subject options for dropdown
        
        # data["subject"]: pd.DataFrame, totally 7341 subjects
        subject_options_list = data["subject"].to_dict('records')

        # If only want to show level zero subjects
        # level_zero_subjects = {
        #     'http://metadata.un.org/thesaurus/10', 
        #     'http://metadata.un.org/thesaurus/09', 
        #     'http://metadata.un.org/thesaurus/16', 
        #     'http://metadata.un.org/thesaurus/07', 
        #     'http://metadata.un.org/thesaurus/04', 
        #     'http://metadata.un.org/thesaurus/06', 
        #     'http://metadata.un.org/thesaurus/15', 
        #     'http://metadata.un.org/thesaurus/05', 
        #     'http://metadata.un.org/thesaurus/03', 
        #     'http://metadata.un.org/thesaurus/17', 
        #     'http://metadata.un.org/thesaurus/11', 
        #     'http://metadata.un.org/thesaurus/12', 
        #     'http://metadata.un.org/thesaurus/13', 
        #     'http://metadata.un.org/thesaurus/14', 
        #     'http://metadata.un.org/thesaurus/18', 
        #     'http://metadata.un.org/thesaurus/08', 
        #     'http://metadata.un.org/thesaurus/01', 
        #     'http://metadata.un.org/thesaurus/02'
        # }
        # subject_options_list = data["subject"][data["subject"]["subject_id"].isin(level_zero_subjects)].to_dict('records')
        
        self.subject_options = [{"label": r["label_en"], "value": r["subject_id"]} for r in subject_options_list]
        # ! not sure about sequence of subjects
        self.subject_options = self.subject_options[::-1]

        # Prepare country options for dropdown
        self.country_options = [{"label": c, "value": c} for c in data["country_columns"]]
        
        # 1946-01-26 2025-09-05
        self.min_date = data["resolution"]["date"].min()
        self.max_date = data["resolution"]["date"].max()

    def get_layout(self) -> html.Div:
        return html.Div(
            style={
                'backgroundColor': '#ffffff',
                'padding': '25px',
                'borderRadius': '12px',
                'boxShadow': '0 2px 8px rgba(0,0,0,0.08)',
                'marginBottom': '25px',
                'border': '1px solid #e9ecef'
            },
            children=[
                # Header with icon
                html.Div([
                    html.H4(
                        "🔍 Filters", 
                        style={
                            'margin': '0 0 20px 0', 
                            'color': '#212529',
                            'fontSize': '22px',
                            'fontWeight': '600',
                            'display': 'flex',
                            'alignItems': 'center',
                            'gap': '10px'
                        }
                    ),
                    html.P(
                        "Adjust filters to refine your data view",
                        style={
                            'margin': '0 0 25px 0',
                            'color': '#6c757d',
                            'fontSize': '14px'
                        }
                    )
                ]),
                
                # Filters container with better layout
                html.Div([
                    # Date Range Filter - Full width
                    html.Div([
                        html.Div([
                            html.Span("📅", style={'fontSize': '18px', 'marginRight': '8px'}),
                            html.Label(
                                "Date Range", 
                                style={
                                    'fontWeight': '600',
                                    'color': '#495057',
                                    'fontSize': '15px',
                                    'marginBottom': '8px',
                                    'display': 'block'
                                }
                            )
                        ]),
                        dcc.DatePickerRange(
                            id=self.ids['date_picker'],
                            min_date_allowed=self.min_date,
                            max_date_allowed=self.max_date,
                            start_date=self.min_date,
                            end_date=self.max_date,
                            display_format='YYYY-MM-DD',
                            style={
                                'width': '100%',
                                'fontSize': '14px'
                            },
                            calendar_orientation='vertical'
                        ),
                    ], style={
                        'marginBottom': '25px',
                        'padding': '15px',
                        'backgroundColor': '#f8f9fa',
                        'borderRadius': '8px',
                        'border': '1px solid #e9ecef'
                    }),
                    
                    # Subjects and Country in a row
                    html.Div([
                        # Subjects Filter
                        html.Div([
                            html.Div([
                                html.Span("📚", style={'fontSize': '18px', 'marginRight': '8px'}),
                                html.Label(
                                    "Subjects", 
                                    style={
                                        'fontWeight': '600',
                                        'color': '#495057',
                                        'fontSize': '15px',
                                        'marginBottom': '8px',
                                        'display': 'block'
                                    }
                                )
                            ]),
                            dcc.Dropdown(
                                id=self.ids['subject_dropdown'],
                                options=self.subject_options,
                                multi=True,
                                placeholder="Select one or more subjects...",
                                style={
                                    'width': '100%',
                                    'fontSize': '14px'
                                },
                                searchable=True
                            ),
                        ], style={
                            'flex': '1',
                            'marginRight': '20px',
                            'padding': '15px',
                            'backgroundColor': '#f8f9fa',
                            'borderRadius': '8px',
                            'border': '1px solid #e9ecef'
                        }),
                        
                        # Country Filter
                        html.Div([
                            html.Div([
                                html.Span("🌍", style={'fontSize': '18px', 'marginRight': '8px'}),
                                html.Label(
                                    "Country", 
                                    style={
                                        'fontWeight': '600',
                                        'color': '#495057',
                                        'fontSize': '15px',
                                        'marginBottom': '8px',
                                        'display': 'block'
                                    }
                                )
                            ]),
                            dcc.Dropdown(
                                id=self.ids['country'],
                                options=self.country_options,
                                placeholder="Select a country...",
                                clearable=True,
                                style={
                                    'width': '100%',
                                    'fontSize': '14px'
                                },
                                searchable=True
                            ),
                        ], style={
                            'flex': '0 0 280px',
                            'padding': '15px',
                            'backgroundColor': '#f8f9fa',
                            'borderRadius': '8px',
                            'border': '1px solid #e9ecef'
                        }),
                    ], style={
                        'display': 'flex',
                        'flexDirection': 'row',
                        'gap': '0'
                    }),
                ]),
                
                # 存储组件
                dcc.Store(id=self.ids['filter_store']),
                dcc.Store(id=self.ids['data_store'])  # Store for queried data
            ]
        )
    
    def register_callbacks(self, app: dash.Dash):
        """
        Register callbacks for the filter component.
        - Filter state management
        - Print current selections when filters change
        - Query data when filters change
        
        Args:
            app: Dash app instance.
        """
        if app is None:
            raise ValueError("App instance is required. Pass it to register_callbacks().")
        
        self.app = app
        
        # Callback: Update filter store and print current selections when any filter changes
        @app.callback(
            Output(self.ids['filter_store'], 'data'),
            Input(self.ids['date_picker'], 'start_date'),
            Input(self.ids['date_picker'], 'end_date'),
            Input(self.ids['subject_dropdown'], 'value'),
            Input(self.ids['country'], 'value'),
            prevent_initial_call=False
        )
        def update_filter_store(start_date, end_date, subject_ids, country):
            """Store filter values and print current selections."""
            filter_data = {
                'start_date': start_date,
                'end_date': end_date,
                'subject_ids': subject_ids if subject_ids else None,
                'country': country
            }
            
            # Print current selections
            print("\n" + "="*50)
            print("Current Filter Selections:")
            print(f"  Date Range: {start_date} to {end_date}")
            print(f"  Subjects: {subject_ids if subject_ids else 'None'}")
            print(f"  Country: {country if country else 'None'}")
            print("="*50 + "\n")
            
            return filter_data
        
        # Callback: Query data when filters change
        @app.callback(
            Output(self.ids['data_store'], 'data'),
            Input(self.ids['filter_store'], 'data'),
            prevent_initial_call=False
        )
        def query_data_on_filter_change(filter_data):
            """Query data based on current filter selections."""
            if not filter_data:
                return None
            
            try:
                start_date = filter_data.get('start_date')
                end_date = filter_data.get('end_date')
                subject_ids = filter_data.get('subject_ids')
                country = filter_data.get('country')
                
                # Query resolutions using the query engine
                df = self.query_engine.query_resolutions(
                    start_date=start_date,
                    end_date=end_date,
                    subject_ids=subject_ids,
                    include_descendants=True
                )
                
                # If country filter is selected, filter by country vote
                if country and country in df.columns:
                    # Only keep rows where the country has a vote (not NaN)
                    df = df.dropna(subset=[country])
                
                # Convert to JSON for storage
                # Select key columns similar to app_mlkeyword.py
                result_df = df[['undl_id', 'resolution', 'date', 'title']].copy() if not df.empty else pd.DataFrame()
                
                print(f"\n✅ Queried {len(result_df)} resolutions")
                if country:
                    print(f"   Filtered by country: {country}")
                
                return result_df.to_json(date_format='iso', orient='split')
                
            except Exception as e:
                print(f"\n❌ Error querying data: {e}")
                import traceback
                traceback.print_exc()
                return None
    
    def get_filter_store_id(self) -> str:
        """
        Returns the filter store component ID.
        Useful for creating callbacks that depend on filter values.
        
        Returns:
            String ID of the filter store component
        """
        return self.ids['filter_store']
    
    def get_data_store_id(self) -> str:
        """
        Returns the data store component ID.
        Useful for creating callbacks that depend on queried data.
        
        Returns:
            String ID of the data store component
        """
        return self.ids['data_store']

if __name__ == "__main__":
    # Create Dash app
    app = dash.Dash(__name__)
    
    # Create filter component
    filter_component = FilterComponent()
    
    # Register callbacks
    filter_component.register_callbacks(app)
    
    # Create app layout - just show the filter component layout
    app.layout = html.Div([
        filter_component.get_layout(),
        # Main content area
        html.Div(
            style={
                'padding': '20px',
                'minHeight': '100vh',
                'backgroundColor': '#f8f9fa'
            },
            children=[
                html.Div(
                    style={
                        'maxWidth': '1400px',
                        'margin': '0 auto',
                        'backgroundColor': '#ffffff',
                        'padding': '30px',
                        'borderRadius': '12px',
                        'boxShadow': '0 2px 8px rgba(0,0,0,0.08)'
                    },
                    children=[
                        html.H1("Filter Component Test", style={'color': '#212529', 'marginBottom': '10px'}),
                        html.P("Adjust the filters above to see data queries in the console.", style={'color': '#6c757d', 'marginBottom': '20px'}),
                        html.Div(
                            id='data-display',
                            style={
                                'marginTop': '20px',
                                'padding': '15px',
                                'backgroundColor': '#f8f9fa',
                                'borderRadius': '8px',
                                'border': '1px solid #dee2e6'
                            }
                        )
                    ]
                )
            ]
        )
    ])
    
    # Callback to display queried data
    @app.callback(
        Output('data-display', 'children'),
        Input(filter_component.get_data_store_id(), 'data')
    )
    def display_queried_data(data_json):
        """Display the queried data."""
        if not data_json:
            return html.P("No data yet. Adjust filters to query data.", style={'color': '#6c757d'})
        
        try:
            df = pd.read_json(data_json, orient='split')
            
            if df.empty:
                return html.P("No resolutions found for the selected filters.", style={'color': '#6c757d'})
            
            # Check if 'date' column exists
            date_info = ""
            if 'date' in df.columns:
                try:
                    df['date'] = pd.to_datetime(df['date'])
                    date_info = html.P(
                        f"Date Range: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}", 
                        style={'color': '#495057', 'marginBottom': '10px'}
                    )
                except:
                    date_info = html.P(f"Date Range: {df['date'].min()} to {df['date'].max()}", style={'color': '#495057', 'marginBottom': '10px'})
            
            # Build sample resolutions list
            sample_resolutions = []
            for _, row in df.head(10).iterrows():
                resolution = row.get('resolution', 'N/A')
                title = row.get('title', 'No title')
                title_short = title[:100] + "..." if len(str(title)) > 100 else str(title)
                sample_resolutions.append(
                    html.Li(f"{resolution} - {title_short}", style={'marginBottom': '5px'})
                )
            
            return html.Div([
                html.H3(f"✅ Queried Data: {len(df)} resolutions", style={'color': '#28a745', 'marginBottom': '15px'}),
                date_info,
                html.Div([
                    html.Strong("Sample resolutions (showing first 10):", style={'display': 'block', 'marginBottom': '10px'}),
                    html.Ul(sample_resolutions, style={'listStyleType': 'disc', 'paddingLeft': '20px'})
                ], style={'marginTop': '10px'})
            ])
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            return html.Div([
                html.P(f"❌ Error displaying data: {e}", style={'color': '#dc3545', 'fontWeight': 'bold'}),
                html.Pre(error_details, style={'fontSize': '12px', 'color': '#6c757d', 'overflow': 'auto', 'maxHeight': '200px'})
            ])
    
    # Run the app
    print("Starting filter component test...")
    print("Open http://127.0.0.1:8050 in your browser")
    print("Adjust filters to see data queries in the console and below.")
    app.run(debug=True, port=8050)