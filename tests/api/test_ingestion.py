import logging
import pickle
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
import requests
import yaml

from app.un_data_stream.core.web_utils import download_bytes, fetch_latest_file_url_from_api
from app.un_data_stream.data.repository import DataRepository
from tests.support.repository import MemoryRepository


@pytest.mark.parametrize(
    "status,content,content_type",
    [
        (202, b"", "text/html"),
        (200, b"<html>challenge</html>", "text/html"),
        (200, b"", "text/csv"),
    ],
)
def test_source_challenges_and_empty_files_are_not_ingested(
    monkeypatch, status, content, content_type
):
    response = Mock(status_code=status, content=content, headers={"Content-Type": content_type})
    monkeypatch.setattr(requests, "get", Mock(return_value=response))
    with pytest.raises(ValueError, match="Source"):
        download_bytes("https://digitallibrary.un.org/record/test/files/data.csv")


def test_current_and_legacy_source_names_resolve(monkeypatch):
    config = yaml.safe_load(Path("config/data_sources.yaml").read_text())
    for key, names, extension in [
        ("thesaurus", ["unbist-2026-02-01_2", "2026_08_20_thesaurus_2"], ".ttl"),
        ("member_states", ["member_states_auths_2026_01_01", "2026_08_17_unms_names"], ".csv"),
    ]:
        for name in names:
            response = Mock()
            response.json.return_value = [
                {
                    "name": name,
                    "format": extension,
                    "url": "https://digitallibrary.un.org/test-data",
                }
            ]
            monkeypatch.setattr(requests, "get", Mock(return_value=response))
            assert fetch_latest_file_url_from_api(
                "test", config["data_sources"][key]["file_name_pattern"].strip(), extension
            )


def test_legacy_uncompressed_cache_loads_without_writing_readonly_handle(tmp_path):
    source = MemoryRepository().get_data()
    mapping = {
        "resolution": "resolution_table",
        "resolution_subject": "resolution_subject_table",
        "subject": "subject_table",
        "closure": "closure_table",
        "broader": "broader_table",
        "member_states": "member_states_table",
    }
    for name, filename in mapping.items():
        source[name].to_csv(tmp_path / f"{filename}.csv", index=False)
    payload = {k: source[k] for k in ["country_columns", "multilateral_scores", "vote_bool_arrays"]}
    cache = tmp_path / "precomputed_agreement_data.pkl"
    original = pickle.dumps(payload)
    cache.write_bytes(original)
    repo = DataRepository.__new__(DataRepository)
    repo.logger = logging.getLogger("test-cache")
    repo.config = {"paths": {"data": str(tmp_path)}}
    repo._load_cached_data()
    np.testing.assert_array_equal(repo.multilateral_scores, source["multilateral_scores"])
    assert cache.read_bytes() == original


def test_failed_cache_write_invalidates_completion_marker(tmp_path, monkeypatch):
    repo = DataRepository.__new__(DataRepository)
    repo.config = {"paths": {"data": str(tmp_path)}}
    repo.resolution_table = pd.DataFrame({"undl_id": [1]})
    marker = tmp_path / "metadata.json"
    marker.write_text('{"version":"1.5"}')
    monkeypatch.setattr(pd.DataFrame, "to_csv", Mock(side_effect=OSError("disk full")))
    with pytest.raises(OSError):
        repo._save_cached_data()
    assert not marker.exists()
