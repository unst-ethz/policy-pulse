from dash import callback, Input, Output, html, dcc
import pandas as pd


def register_callbacks():

    # Callback to display queried data
    @callback(
        Output("data-display", "children"),
        Input("filter-component-data-store", "data"),
    )
    def display_queried_data(data_json):
        """Display the queried data."""
        if not data_json:
            return html.P(
                "No data yet. Adjust filters to query data.", style={"color": "#6c757d"}
            )

        try:
            df = pd.read_json(data_json, orient="split")

            if df.empty:
                return html.P(
                    "No resolutions found for the selected filters.",
                    style={"color": "#6c757d"},
                )

            # Check if 'date' column exists
            date_info = ""
            if "date" in df.columns:
                try:
                    df["date"] = pd.to_datetime(df["date"])
                    date_info = html.P(
                        f"Date Range: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}",
                        style={"color": "#495057", "marginBottom": "10px"},
                    )
                except:
                    date_info = html.P(
                        f"Date Range: {df['date'].min()} to {df['date'].max()}",
                        style={"color": "#495057", "marginBottom": "10px"},
                    )

            # Build sample resolutions list
            sample_resolutions = []
            for _, row in df.head(10).iterrows():
                resolution = row.get("resolution", "N/A")
                title = row.get("title", "No title")
                title_short = (
                    title[:100] + "..." if len(str(title)) > 100 else str(title)
                )
                sample_resolutions.append(
                    html.Li(
                        f"{resolution} - {title_short}", style={"marginBottom": "5px"}
                    )
                )

            return html.Div(
                [
                    html.H3(
                        f"✅ Queried Data: {len(df)} resolutions",
                        style={"color": "#28a745", "marginBottom": "15px"},
                    ),
                    date_info,
                    html.Div(
                        [
                            html.Strong(
                                "Sample resolutions (showing first 10):",
                                style={"display": "block", "marginBottom": "10px"},
                            ),
                            html.Ul(
                                sample_resolutions,
                                style={"listStyleType": "disc", "paddingLeft": "20px"},
                            ),
                        ],
                        style={"marginTop": "10px"},
                    ),
                ]
            )
        except Exception as e:
            import traceback

            error_details = traceback.format_exc()
            return html.Div(
                [
                    html.P(
                        f"❌ Error displaying data: {e}",
                        style={"color": "#dc3545", "fontWeight": "bold"},
                    ),
                    html.Pre(
                        error_details,
                        style={
                            "fontSize": "12px",
                            "color": "#6c757d",
                            "overflow": "auto",
                            "maxHeight": "200px",
                        },
                    ),
                ]
            )


layout = (
    html.Div(
        id="data-display",
        style={
            "marginTop": "20px",
            "padding": "15px",
            "backgroundColor": "#f8f9fa",
            "borderRadius": "8px",
            "border": "1px solid #dee2e6",
        },
    ),
)
