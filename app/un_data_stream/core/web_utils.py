import logging
import re
from io import BytesIO
from typing import Optional

import pandas as pd
import requests

UNDL_API_URL = "https://digitallibrary.un.org/api/v1/file"

logger = logging.getLogger(__name__)


def download_bytes(url: str) -> bytes:
    """Reject empty/challenge responses before treating them as scientific data."""
    response = requests.get(url, timeout=(10, 120))
    response.raise_for_status()
    if response.status_code != 200 or not response.content:
        raise ValueError(f"Source download did not return a data file (HTTP {response.status_code}): {url}")
    if "text/html" in response.headers.get("Content-Type", "").lower():
        raise ValueError(f"Source returned HTML instead of a data file: {url}")
    return response.content


def read_csv_source(url: str) -> pd.DataFrame:
    return pd.read_csv(BytesIO(download_bytes(url)))

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
        response = requests.get(UNDL_API_URL, params=params, timeout=(10, 60))
        response.raise_for_status()
        
        files_list = response.json()
        
        # Ensure we got a list
        if not isinstance(files_list, list):
            logger.error(f"Unexpected API response format for recid {recid}: expected list, got {type(files_list).__name__}")
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
            
    except requests.RequestException as e:
        logger.error(f"HTTP error fetching file list for recid {recid} from {UNDL_API_URL}: {e}")
        return None
    except ValueError as e:
        logger.error(f"Failed to decode JSON response for recid {recid}: {e}")
        return None
