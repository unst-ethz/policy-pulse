"""
Unit tests for the `web_utils.py` module.

These tests focus on the low-level `fetch_latest_file_url_from_api` function,
mocking the `requests` library to verify correct API interaction and JSON
response handling without making real network calls.
"""

import pytest
import requests
from app.un_data_stream.core.web_utils import fetch_latest_file_url_from_api
from unittest.mock import Mock, patch
from requests import Response

def test_fetch_latest_file_url_success():
    """Test successful URL extraction with mocked JSON response."""
    mock_response = Mock(spec=Response)
    mock_response.status_code = 200
    mock_response.json.return_value = [
        {
            "name": "2026_01_01_ga_voting",
            "url": "https://digitallibrary.un.org/record/4060887/files/2026_01_01_ga_voting.csv",
            "format": ".csv"
        },
        {
            "name": "other_file",
            "url": "http://example.com/other.txt",
            "format": ".txt"
        }
    ]
    
    with patch('requests.get', return_value=mock_response) as mock_get:
        pattern = r".*_ga_voting$"
        recid = "4060887"
        
        url = fetch_latest_file_url_from_api(recid, pattern, '.csv')
        
        assert url == "https://digitallibrary.un.org/record/4060887/files/2026_01_01_ga_voting.csv"
        mock_get.assert_called_with("https://digitallibrary.un.org/api/v1/file", params={'recid': '4060887'})

def test_fetch_latest_file_url_not_found():
    """Test when no matching file is found."""
    mock_response = Mock(spec=Response)
    mock_response.status_code = 200
    mock_response.json.return_value = [
        {"name": "readme", "url": "http://example.com/readme.txt"}
    ]
    
    with patch('requests.get', return_value=mock_response):
        pattern = r".*_ga_voting.*"
        recid = "4060887"
        file_format = ".csv"
        
        url = fetch_latest_file_url_from_api(recid, pattern, file_format)
        
        assert url is None

def test_fetch_latest_file_url_request_error():
    """Test handling of request exceptions."""
    with patch('requests.get', side_effect=requests.RequestException("Connection error")):
        pattern = r".*_ga_voting.*"
        recid = "4060887"
        file_format = ".csv"
        
        url = fetch_latest_file_url_from_api(recid, pattern, file_format)
        
        assert url is None
