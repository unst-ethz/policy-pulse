"""Verify the production proxy/containers against the isolated synthetic cache.

Usage: python -m tests.support.smoke_stack http://127.0.0.1:18050
"""

import csv
import io
import re
import sys

import requests


def verify(base_url: str):
    def get(path):
        return requests.get(base_url.rstrip("/") + path, timeout=15)

    for path in [
        "/health/live",
        "/health/ready",
        "/api/v1/metadata",
        "/api/v1/methodology",
        "/api/v1/analysis/agreement/USA",
        "/api/v1/analysis/timeline/USA?compare=CHE",
        "/api/v1/analysis/subjects/USA/CHE",
        "/api/v1/analysis/multilateral",
        "/api/v1/analysis/words/category",
        "/api/v1/countries/USA/profile?compare=CHE",
    ]:
        response = get(path)
        assert response.status_code == 200, (path, response.status_code, response.text[:200])
        response.json()
        print("PASS", path)

    assert get("/api/v1/overview").json()["resolutions"] == 160, "Expected synthetic snapshot"
    records = get("/api/v1/resolutions?country=USA&compare=CHE&limit=2").json()
    assert len(records["items"]) == 2 and records["total"] == 160
    export = get("/api/v1/resolutions/export.csv?country=USA&compare=CHE&limit=2")
    assert len(list(csv.DictReader(io.StringIO(export.text)))) == 160
    assert get("/api/v1/resolutions?start_date=0001-01-01").status_code == 422
    assert get("/assets/not-real.js").status_code == 404
    for path in [
        "/",
        "/trends?country=USA",
        "/profile?country=USA",
        "/methodology",
        "/resolutions/9000000",
    ]:
        response = get(path)
        assert response.status_code == 200 and '<div id="root"></div>' in response.text
    asset = re.search(r'src="(/assets/[^\"]+\.js)"', get("/").text)
    assert asset, "Missing JavaScript entry point"
    response = get(asset.group(1))
    assert response.status_code == 200 and "immutable" in response.headers["Cache-Control"]
    print("PASS synthetic totals, pagination/export, validation, SPA deep links and asset caching")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python -m tests.support.smoke_stack http://127.0.0.1:18050")
    verify(sys.argv[1])
