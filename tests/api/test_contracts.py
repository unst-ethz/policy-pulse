import csv
import io
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.main import create_app
from backend.models import Filters
from tests.support.repository import CHILD_SUBJECT, ROOT_SUBJECT


def test_import_and_schema_do_not_download_data():
    code = "import requests; requests.get=lambda *a,**k: (_ for _ in ()).throw(AssertionError('network at import')); import backend.main; assert 'dash' not in __import__('sys').modules; assert backend.main.app.openapi()['info']['version']=='1.0.0'"
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True)


def test_schema_is_current():
    expected = json.loads(Path("backend/openapi.json").read_text())
    assert create_app().openapi() == expected


@pytest.mark.parametrize(
    "url",
    [
        "/api/v1/metadata",
        "/api/v1/metadata?language=fr",
        "/api/v1/methodology",
        "/api/v1/overview",
        "/api/v1/resolutions?country=USA&compare=CHE",
        "/api/v1/resolutions/9000000",
        "/api/v1/analysis/agreement/USA",
        "/api/v1/analysis/timeline/USA?compare=CHE",
        "/api/v1/analysis/subjects/USA/CHE",
        "/api/v1/analysis/multilateral",
        "/api/v1/analysis/words/default",
        "/api/v1/analysis/words/category",
        "/api/v1/countries/USA/profile?compare=CHE",
        "/health/live",
        "/health/ready",
    ],
)
def test_public_endpoints_are_strict_json(client, url):
    response = client.get(url)
    assert response.status_code == 200, response.text
    assert response.headers["x-content-type-options"] == "nosniff"
    if url.startswith("/api"):
        assert response.headers["x-request-id"]

    def invalid_constant(value):
        raise AssertionError(f"Non-JSON number: {value}")

    json.loads(response.text, parse_constant=invalid_constant)


@pytest.mark.parametrize(
    "query",
    [
        "limit=0",
        "limit=101",
        "offset=-1",
        "sort=bad",
        "country=INVALID",
        "country=ZZZ",
        "compare=ZZZ",
        "start_date=2024-02-30",
        "start_date=2025-01-01&end_date=2024-01-01",
        "vote=Y",
        "country=USA&compare=CHE&vote=Y",
        "agreement=AGREED",
        "subject=unknown",
        "extra=field",
        "country_mode=voted",
    ],
)
def test_invalid_selection_is_422_not_500(client, query):
    response = client.get(f"/api/v1/resolutions?{query}")
    assert response.status_code == 422, response.text
    assert response.json()["detail"]["message"]


@pytest.mark.parametrize(
    "path",
    [
        "/resolutions",
        "/resolutions/export.csv",
        "/analysis/agreement/USA",
        "/analysis/multilateral",
        "/analysis/words/category",
        "/analysis/subjects/USA/CHE",
        "/countries/USA/profile",
    ],
)
@pytest.mark.parametrize("query", ["start_date=0001-01-01", "end_date=9999-12-31"])
def test_dates_outside_engine_range_are_client_errors(client, path, query):
    response = client.get(f"/api/v1{path}?{query}")
    assert response.status_code == 422, response.text
    assert response.json()["detail"]["message"]


def test_pagination_is_stable_complete_and_export_matches(client):
    base = "/api/v1/resolutions?country=USA&compare=CHE&agreement=STRONGLY_DISAGREED&sort=consensus_desc"
    one = client.get(base + "&limit=7").json()
    two = client.get(base + "&limit=7&offset=7").json()
    ids = [r["id"] for r in one["items"] + two["items"]]
    assert len(ids) == len(set(ids))
    csv_response = client.get(
        base.replace("resolutions?", "resolutions/export.csv?") + "&limit=1&offset=10"
    )
    rows = list(csv.DictReader(io.StringIO(csv_response.text)))
    assert len(rows) == one["total"]
    assert [r["undl_id"] for r in rows[: len(ids)]] == ids
    assert "United States of America" in rows[0]
    assert all(r["United States of America"] != r["Switzerland"] for r in rows)


def test_missing_votes_stay_null_and_nonvotes_stay_x(client):
    record = client.get("/api/v1/resolutions/9000000").json()
    assert record["votes"]["GER"] is None
    assert record["votes"]["IND"] is None
    data = client.get("/api/v1/resolutions?country=IND&vote=X").json()
    assert data["total"] > 0
    assert all(r["votes"]["IND"] == "X" for r in data["items"])
    result = client.get("/api/v1/analysis/agreement/USA").json()
    germany = next(r for r in result["items"] if r["country"] == "GER")
    assert germany == {"country": "GER", "score": None, "shared_votes": 0}


def test_empty_selection_cannot_expand_to_all_data(client):
    suffix = "?start_date=1800-01-01&end_date=1800-12-31"
    for path in [
        "/resolutions",
        "/analysis/multilateral",
        "/analysis/agreement/USA",
        "/analysis/words/category",
    ]:
        result = client.get("/api/v1" + path + suffix).json()
        assert result["items"] == []
        assert result.get("total", result.get("resolution_count")) == 0


def test_inclusive_dates_and_descendants(client, service):
    result = client.get("/api/v1/resolutions?start_date=2024-01-02&end_date=2024-01-02").json()
    assert result["total"] == 1
    assert result["items"][0]["date"] == "2024-01-02"
    root = client.get("/api/v1/resolutions", params={"subject": ROOT_SUBJECT}).json()
    child = client.get("/api/v1/resolutions", params={"subject": CHILD_SUBJECT}).json()
    exact = client.get(
        "/api/v1/resolutions", params={"subject": ROOT_SUBJECT, "include_descendants": "false"}
    ).json()
    assert root["total"] == child["total"] > 0
    assert exact["total"] == 0
    assert (
        service.filtered(Filters(subject=["__all_subjects__", CHILD_SUBJECT])).shape[0]
        == child["total"]
    )


def test_no_subject_sentinel(client, service):
    result = client.get("/api/v1/resolutions?subject=__no_subject__&limit=100").json()
    linked = set(service.engine.resolution_subject_table.undl_id.astype(str))
    assert result["total"] > 0
    assert not {r["id"] for r in result["items"]} & linked


def test_keyword_boolean_expression(client):
    query = client.get(
        "/api/v1/resolutions", params={"keyword": 'nuclear & disarmament, "development"'}
    ).json()
    assert query["total"] == 160
    exact = client.get("/api/v1/resolutions", params={"keyword": '"nuclear" & cooperation'}).json()
    assert exact["total"] == 0
    literal = client.get("/api/v1/resolutions", params={"keyword": ".*"}).json()
    assert literal["total"] == 0


def test_health_and_methodology_available_when_data_fails():
    def failing(settings):
        raise RuntimeError("private diagnostic should not leak")

    with TestClient(create_app(service_factory=failing)) as client:
        assert client.get("/health/live").status_code == 200
        assert client.get("/api/v1/methodology").status_code == 200
        response = client.get("/api/v1/overview")
        assert response.status_code == 503
        assert response.headers["retry-after"] == "10"
        assert "private diagnostic" not in response.text
        assert client.get("/health/ready").status_code == 503


def test_no_write_api_and_unknown_resources(client):
    assert client.post("/api/v1/resolutions", json={}).status_code == 405
    assert client.get("/api/v1/resolutions/not-present").status_code == 404


def test_metadata_language_and_historical_names(client):
    en = client.get("/api/v1/metadata").json()
    fr = client.get("/api/v1/metadata?language=fr").json()
    assert next(c for c in fr["countries"] if c["code"] == "CHE")["name"] == "Suisse"
    germany = next(c for c in en["countries"] if c["code"] == "GER")
    assert germany["name"] == "Germany, Federal Republic of (1973–1990)"
    assert "West Germany" in germany["search_terms"]


def test_requests_do_not_mutate_shared_snapshot(client, service):
    before = service.engine.resolution_table.copy(deep=True)
    for url in [
        "/resolutions?country=USA&vote=N",
        "/analysis/agreement/USA",
        "/analysis/timeline/USA?compare=CHE",
        "/countries/CHE/profile",
        "/resolutions/export.csv",
    ]:
        assert client.get("/api/v1" + url).status_code == 200
    pd.testing.assert_frame_equal(before, service.engine.resolution_table)
