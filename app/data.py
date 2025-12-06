import os
import sys
from typing import Any
import pandas as pd
import pycountry

from .un_data_stream import DataRepository, ResolutionQueryEngine


# def fetch_UN_data(dir_path: str | None = None):
#     """
#     Fetches and processes United Nations General Assembly and Security Council voting data.

#     This function retrieves voting data from either local files or the UN Digital Library,
#     and transforms the data into two formats: original and pivoted (transformed).

#     Parameters:
#     -----------
#     dir_path : str, optional
#         Path to directory where data should be read from or saved to.
#         If None, data will be fetched from the UN Digital Library and not saved locally.

#     Returns:
#     --------
#     tuple
#         A tuple containing four DataFrames:
#         - df_ga: Original GA voting data
#         - df_ga_transformed: Pivoted GA voting data with countries as columns
#         - df_sc: Original SC voting data
#         - df_sc_transformed: Pivoted SC voting data with countries as columns

#     Notes:
#     ------
#     - Currently, the Security Council data does not include veto information explicitly.
#     - The filenames and URLs are hardcoded for the 2025 voting sessions. Must be updated when they change.
#     """

#     df_ga = None
#     df_sc = None

#     if dir_path:
#         try:
#             df_ga = pd.read_csv(f"{dir_path}/2025_9_19_ga_voting.csv")
#             df_sc = pd.read_csv(f"{dir_path}/2025_7_21_sc_voting.csv")
#         except FileNotFoundError:
#             print("Not all data found locally. Fetching from UN Digital Library...")
#     if df_ga is None or df_sc is None:
#         ga_url = "https://digitallibrary.un.org/record/4060887/files/2025_9_19_ga_voting.csv?ln=en"
#         sc_url = "https://digitallibrary.un.org/record/4055387/files/2025_7_21_sc_voting.csv?ln=en"

#         try:
#             df_ga = pd.read_csv(ga_url)
#             df_sc = pd.read_csv(sc_url)

#             # Save data locally if dir_path is provided
#             if dir_path:
#                 # Check if directory exists, create it if it doesn't
#                 if not os.path.exists(dir_path):
#                     os.makedirs(dir_path)
#                     print(f"Created directory: {dir_path}")

#                 df_ga.to_csv(f"{dir_path}/2025_9_19_ga_voting.csv", index=False)
#                 df_sc.to_csv(f"{dir_path}/2025_7_21_sc_voting.csv", index=False)
#         except Exception as e:
#             print(
#                 "Error fetching data from UN Digital Library. The dataset might has been updated. Check the date in the URL."
#             )
#             raise e

#     # Transform ga data
#     ga_index_columns = [
#         "undl_id",
#         "date",
#         "session",
#         "resolution",
#         "draft",
#         "committee_report",
#         "meeting",
#         "title",
#         "agenda_title",
#         "subjects",
#         "total_yes",
#         "total_no",
#         "total_abstentions",
#         "total_non_voting",
#         "total_ms",
#         "undl_link",
#     ]
#     df_ga_transformed = df_ga.pivot(
#         index=ga_index_columns, columns="ms_code", values="ms_vote"
#     ).reset_index()
#     df_ga_transformed.columns.name = None

#     # Transform sc data
#     sc_index_columns = [
#         "undl_id",
#         "date",
#         "resolution",
#         "draft",
#         "meeting",
#         "description",
#         "agenda",
#         "subjects",
#         "modality",
#         "total_yes",
#         "total_no",
#         "total_abstentions",
#         "total_non_voting",
#         "total_ms",
#         "undl_link",
#     ]
#     df_sc_transformed = df_sc.pivot(
#         index=sc_index_columns, columns="ms_code", values="ms_vote"
#     ).reset_index()
#     df_sc_transformed.columns.name = None

#     return df_ga, df_ga_transformed, df_sc, df_sc_transformed


# df_ga, df_ga_transformed, df_sc, df_sc_transformed = fetch_UN_data(dir_path="../data")

# available_countries = [
#     col
#     for col in df_ga_transformed.columns
#     if col
#     not in [
#         "undl_id",
#         "date",
#         "session",
#         "resolution",
#         "draft",
#         "committee_report",
#         "meeting",
#         "title",
#         "agenda_title",
#         "subjects",
#         "total_yes",
#         "total_no",
#         "total_abstentions",
#         "total_non_voting",
#         "total_ms",
#         "undl_link",
#     ]
# ]

repo = DataRepository(config_path="config/data_sources.yaml")
query_engine = ResolutionQueryEngine(repo=repo)

available_countries = query_engine.get_available_countries()


def get_country_name(
    iso3_code: str | None,
) -> str:  # we need to support multiple languages at some point
    """Get English country name from ISO3 code."""
    if iso3_code is None:
        return "Unknown"
    try:
        country = pycountry.countries.get(alpha_3=iso3_code)
        return country.name if country else iso3_code
    except:
        return iso3_code


# Subject data for alignment by subject feature
_data = repo.get_data()
subject_table = _data["subject"]

# Top level subjects (level 0 in the hierarchy)
TOP_LEVEL_SUBJECTS = {
    'http://metadata.un.org/thesaurus/10', 
    'http://metadata.un.org/thesaurus/09', 
    'http://metadata.un.org/thesaurus/16', 
    'http://metadata.un.org/thesaurus/00', 
    'http://metadata.un.org/thesaurus/07', 
    'http://metadata.un.org/thesaurus/04', 
    'http://metadata.un.org/thesaurus/06', 
    'http://metadata.un.org/thesaurus/15', 
    'http://metadata.un.org/thesaurus/05', 
    'http://metadata.un.org/thesaurus/03', 
    'http://metadata.un.org/thesaurus/17', 
    'http://metadata.un.org/thesaurus/11', 
    'http://metadata.un.org/thesaurus/12', 
    'http://metadata.un.org/thesaurus/13', 
    'http://metadata.un.org/thesaurus/14', 
    'http://metadata.un.org/thesaurus/18', 
    'http://metadata.un.org/thesaurus/08', 
    'http://metadata.un.org/thesaurus/01', 
    'http://metadata.un.org/thesaurus/02'
}

# Map subject IDs to labels
SUBJECT_ID_TO_LABEL_MAP = {row["subject_id"]: row["label_en"] for _, row in subject_table.iterrows()}

def available_subjects() -> list[dict[str, Any]]:
    data = repo.get_data()

    # data["subject"]: pd.DataFrame, totally 7341 subjects
    subject_options_list = data["subject"].to_dict("records")

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

    subject_options = [
        {"label": r["label_en"], "value": r["subject_id"]} for r in subject_options_list
    ]
    # ! not sure about sequence of subjects
    subject_options = subject_options[::-1]

    return subject_options


def get_earliest_data_date():
    # 1946-01-26
    data = repo.get_data()

    return pd.to_datetime(data["resolution"]["date"].min())


def get_latest_data_date():
    # 2025-09-05
    data = repo.get_data()

    return pd.to_datetime(data["resolution"]["date"].max())
