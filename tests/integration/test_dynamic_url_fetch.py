import pytest
from app.un_data_stream.core.web_utils import fetch_latest_file_url_from_api

@pytest.mark.needs_internet
def test_ga_voting_real_url_fetch():
    """Test actual fetching of GA voting file URL via API."""
    recid = "4060887"
    pattern = r".*_ga_voting$"
    file_format = ".csv"
    
    url = fetch_latest_file_url_from_api(recid, pattern, file_format)
    
    assert url is not None
    assert "ga_voting" in url
    assert url.startswith("https://digitallibrary.un.org/record/4060887/files/")

@pytest.mark.needs_internet
def test_sc_voting_real_url_fetch():
    """Test actual fetching of SC voting file URL via API."""
    recid = "4055387"
    pattern = r".*_sc_voting$"
    file_format = ".csv"
    
    url = fetch_latest_file_url_from_api(recid, pattern, file_format)
    
    assert url is not None
    assert "sc_voting" in url
    assert url.startswith("https://digitallibrary.un.org/record/4055387/files/")

@pytest.mark.needs_internet
def test_thesaurus_real_url_fetch():
    """Test actual fetching of Thesaurus file URL via API."""
    recid = "4075456"
    pattern = r"unbist-.*_2$"
    file_format = ".ttl"
    
    url = fetch_latest_file_url_from_api(recid, pattern, file_format)
    
    assert url is not None
    assert url.endswith("_2.ttl")
    assert url.startswith("https://digitallibrary.un.org/record/4075456/files/")

@pytest.mark.needs_internet
def test_fetch_nonexistent_pattern():
    """Test fetching with a valid recid but non-matching pattern."""
    recid = "4060887" # Valid GA record
    pattern = r"this_pattern_does_not_exist$"
    file_format = ".csv"
    
    url = fetch_latest_file_url_from_api(recid, pattern, file_format)
    
    assert url is None
