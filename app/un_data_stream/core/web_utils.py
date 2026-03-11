import re
from typing import Optional

import requests

UNDL_API_URL = "https://digitallibrary.un.org/api/v1/file"

def fetch_latest_file_url_from_api(recid: str, file_name_pattern: str, file_format: str) -> Optional[str]:
    """
    Fetch the list of files for a given record ID from the UN Digital Library API
    and return the URL of the first file matching the name pattern and file format.

    # TODO: Add file duplication checks

    Args:
        recid: The bibliographic record ID (e.g., "4060887").
        file_name_pattern: A regex pattern to match against the file name.
        file_format: The expected file format (e.g., ".csv" or ".ttl".
        
    Returns:
        The full download URL of the matching file if found, otherwise None.
    """

    if not file_format.startswith('.'):
        file_format = '.' + file_format

    params = {'recid': recid}

    try:
        response = requests.get(UNDL_API_URL, params=params)
        response.raise_for_status()
        
        files_list = response.json()
        
        # Ensure we got a list
        if not isinstance(files_list, list):
            return None
            
        # Iterate through files to find a match
        for file_info in files_list:
            # We check against 'format' and 'name' separately
            # Example API response item:
            # {"format": ".csv", "name": "2026_02_06_ga_voting", "url": ".../2026_02_06_ga_voting.csv", ...}

            if file_info.get('format', '') == file_format:
                if re.fullmatch(file_name_pattern, file_info.get('name', '')):
                    return file_info.get('url', '')

        return None
            
    except (requests.RequestException, ValueError):
        # ValueError includes JSON decoding errors
        # TODO: Log specific exceptions for better observability instead of silently returning None
        return None
