import os
import pandas as pd

def fetch_UN_data(dir_path=None):
    """
    Fetches and processes United Nations General Assembly and Security Council voting data.
    
    This function retrieves voting data from either local files or the UN Digital Library,
    and transforms the data into two formats: original and pivoted (transformed).
    
    Parameters:
    -----------
    dir_path : str, optional
        Path to directory where data should be read from or saved to.
        If None, data will be fetched from the UN Digital Library and not saved locally.
    
    Returns:
    --------
    tuple
        A tuple containing four DataFrames:
        - df_ga: Original GA voting data
        - df_ga_transformed: Pivoted GA voting data with countries as columns
        - df_sc: Original SC voting data
        - df_sc_transformed: Pivoted SC voting data with countries as columns
    
    Notes:
    ------
    - Currently, the Security Council data does not include veto information explicitly.
    """

    df_ga = None
    df_sc = None

    if dir_path:
        try:
            df_ga = pd.read_csv(f"{dir_path}/2025_7_23_ga_voting.csv")
            df_sc = pd.read_csv(f"{dir_path}/2025_7_21_sc_voting.csv")
        except FileNotFoundError:
            print("Not all data found locally. Fetching from UN Digital Library...")
    if df_ga is None or df_sc is None:
        ga_url = "https://digitallibrary.un.org/record/4060887/files/2025_7_23_ga_voting.csv?ln=en"
        sc_url = "https://digitallibrary.un.org/record/4055387/files/2025_7_21_sc_voting.csv?ln=en"

        try:
            df_ga = pd.read_csv(ga_url)
            df_sc = pd.read_csv(sc_url)

            # Save data locally if dir_path is provided
            if dir_path:
                # Check if directory exists, create it if it doesn't
                if not os.path.exists(dir_path):
                    os.makedirs(dir_path)
                    print(f"Created directory: {dir_path}")
                
                df_ga.to_csv(f"{dir_path}/2025_7_23_ga_voting.csv", index=False)
                df_sc.to_csv(f"{dir_path}/2025_7_21_sc_voting.csv", index=False)
        except Exception as e:
            print("Error fetching data from UN Digital Library. The dataset might has been updated. Check the date in the URL.")
            print(f"Error: {e}")
            return None, None, None, None
    
    # Transform ga data
    ga_index_columns = ["undl_id", "date", "session", "resolution", "draft", "committee_report", "meeting", "title", "agenda_title", "subjects", "total_yes", "total_no", "total_abstentions", "total_non_voting", "total_ms", "undl_link"]
    df_ga_transformed = df_ga.pivot(index=ga_index_columns, columns='ms_name', values='ms_vote').reset_index()
    df_ga_transformed.columns.name = None

    # Transform sc data
    sc_index_columns = ["undl_id", "date", "resolution", "draft", "meeting", "description", "agenda", "subjects", "modality", "total_yes", "total_no", "total_abstentions", "total_non_voting", "total_ms", "undl_link"]
    df_sc_transformed = df_sc.pivot(index=sc_index_columns, columns='ms_name', values='ms_vote').reset_index()
    df_sc_transformed.columns.name = None

    return df_ga, df_ga_transformed, df_sc, df_sc_transformed
