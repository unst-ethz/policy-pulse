"""The HTTP API against the real data.

The numbers come from `un_data_stream.analysis`, which the unit tests check against hand-derived
values and which the Dash pages call too. These tests cover what the HTTP layer adds: parameter
parsing and validation, the filter selection, JSON shapes (no NaN), errors, and caching headers.
"""

import math

import pytest

pytestmark = pytest.mark.needs_db


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from app.api.main import app

    # Not used as a context manager on purpose: that would run the lifespan and start the
    # background reload poller inside the test process.
    return TestClient(app)


def get(client, url, status=200):
    response = client.get(url)
    assert response.status_code == status, response.text
    return response.json() if "json" in response.headers.get("content-type", "") else response


def _no_nan(value):
    """JSON has no NaN; Starlette would emit it as a bare token the browser cannot parse."""
    if isinstance(value, float):
        assert not math.isnan(value)
    elif isinstance(value, dict):
        for v in value.values():
            _no_nan(v)
    elif isinstance(value, list):
        for v in value:
            _no_nan(v)


ENDPOINTS = [
    "/api/v1/meta",
    "/api/v1/countries?lang=ar",
    "/api/v1/countries/che",
    "/api/v1/regions/tree",
    "/api/v1/subjects/tree?lang=zh",
    "/api/v1/stats/overview",
    "/api/v1/resolutions?votes=USA,CHN&sort=consensus_asc&limit=20",
    "/api/v1/resolutions/recent",
    "/api/v1/resolutions/facets?keyword=women&lang=fr",
    "/api/v1/agreement/map?country=BRA&start_date=1990-01-01",
    "/api/v1/agreement/timeline?country=IND&compare=PAK,USA",
    "/api/v1/agreement/by-subject?country=FRA&compare=DEU",
    "/api/v1/multilateral?start_date=2015-01-01",
    "/api/v1/keywords/cloud?mode=thematic",
    "/api/v1/countries/ZAF/profile?compare=NGA&compare=EGY",
    "/api/v1/resolutions?sponsor=CHE&original_sponsor=true&limit=20",
    "/api/v1/countries/CHE/sponsorship?start_date=2000-01-01",
    "/api/v1/subjects/domains?start_date=2000-01-01&lang=fr",
    "/api/v1/subjects/16/sponsors?start_date=2000-01-01",
]


@pytest.mark.parametrize("url", ENDPOINTS)
def test_endpoint_answers_without_nan(client, url):
    _no_nan(get(client, url))


def test_meta_matches_the_loaded_data(client):
    from app import data

    meta = get(client, "/api/v1/meta")
    assert meta["resolution_count"] == len(data.query_engine.resolution_table)
    assert meta["country_count"] == len(data.available_countries)
    assert set(meta["languages"]) == {"en", "fr", "es", "ar", "zh", "ru"}


def test_country_names_are_localised(client):
    assert get(client, "/api/v1/countries/CHE?lang=fr")["name"] == "Suisse"
    assert get(client, "/api/v1/countries/che")["name"] == "Switzerland"


def test_dissolved_states_are_marked(client):
    soviet_union = get(client, "/api/v1/countries/SUN")
    assert soviet_union["former_state"]
    assert soviet_union["membership"][-1]["end"] is not None
    assert not get(client, "/api/v1/countries/CHE")["former_state"]


def test_resolution_list_filters_and_paginates(client):
    first = get(client, "/api/v1/resolutions?start_date=2020-01-01&end_date=2020-12-31&limit=5")
    second = get(
        client, "/api/v1/resolutions?start_date=2020-01-01&end_date=2020-12-31&limit=5&offset=5"
    )
    assert first["total"] == second["total"] > 10
    assert len(first["items"]) == 5
    assert {r["undl_id"] for r in first["items"]}.isdisjoint(r["undl_id"] for r in second["items"])
    assert all(r["date"].startswith("2020") for r in first["items"] + second["items"])
    dates = [r["date"] for r in first["items"]]
    assert dates == sorted(dates, reverse=True)


def test_resolution_list_matches_the_dash_filter(client):
    """Same criteria, same set: the API and the Dash resolution tab share `filter_resolutions`."""
    from app.features import filtered_resolutions

    store = {
        "start_date": "2000-01-01",
        "end_date": "2010-12-31",
        "subject_ids": None,
        "country1_alpha3": "CHE",
        "country_filter_mode": "voted",
        "keyword": "nuclear, women",
    }
    expected = filtered_resolutions.resolutions_for(store, "resolution_list")
    page = get(
        client,
        "/api/v1/resolutions?start_date=2000-01-01&end_date=2010-12-31&country=CHE"
        "&country_mode=voted&keyword=nuclear,%20women&limit=500",
    )
    assert page["total"] == len(expected)
    assert {r["undl_id"] for r in page["items"]} == set(expected["undl_id"])


def test_votes_are_included_on_request(client):
    page = get(client, "/api/v1/resolutions?votes=usa&votes=CHN,USA&limit=3")
    for item in page["items"]:
        assert set(item["votes"]) == {"USA", "CHN"}
        assert set(item["votes"].values()) <= {"Y", "N", "A", "X"}


def test_vote_filter(client):
    page = get(client, "/api/v1/resolutions?country=USA&vote=N&votes=USA&limit=50")
    assert page["total"] > 0
    assert {r["votes"]["USA"] for r in page["items"]} == {"N"}


def test_resolution_detail(client):
    some_id = get(client, "/api/v1/resolutions/recent?limit=1")["items"][0]["undl_id"]
    detail = get(client, f"/api/v1/resolutions/{some_id}?lang=es")
    assert detail["undl_id"] == some_id
    assert len(detail["all_votes"]) == get(client, "/api/v1/meta")["country_count"]


def test_sponsor_filter_and_detail(client):
    page = get(client, "/api/v1/resolutions?sponsor=CHE&original_sponsor=true&limit=5")
    everything = get(client, "/api/v1/resolutions?sponsor=CHE&limit=5")
    assert 0 < page["total"] < everything["total"]
    detail = get(client, f"/api/v1/resolutions/{page['items'][0]['undl_id']}?lang=fr")
    swiss = [s for s in detail["sponsors"] if s["country"] == "CHE"]
    assert swiss == [{"name": "Suisse", "country": "CHE", "type": "state", "kind": "original"}]
    assert detail["sponsor_count"] == len(detail["sponsors"])
    assert detail["sponsorship_note"] is None
    kinds = [s["kind"] for s in detail["sponsors"]]
    assert kinds == sorted(kinds, key=lambda k: k != "original")  # originals first


def test_unknown_sponsorship_is_not_zero(client):
    page = get(client, "/api/v1/resolutions?end_date=1960-12-31&limit=500")
    unknown = [r for r in page["items"] if r["sponsor_count"] is None]
    assert unknown
    detail = get(client, f"/api/v1/resolutions/{unknown[0]['undl_id']}")
    assert detail["sponsors"] == [] and detail["sponsorship_note"]


def test_country_sponsorship(client):
    record = get(client, "/api/v1/countries/CHE/sponsorship?start_date=2010-01-01")
    assert 0 < record["original"] <= record["sponsored"] <= record["with_data"]
    assert record["with_data"] <= record["resolutions"]
    listed = get(client, "/api/v1/resolutions?sponsor=CHE&start_date=2010-01-01&limit=1")
    assert listed["total"] == record["sponsored"]
    assert sum(y["original"] + y["additional"] for y in record["per_year"]) == record["sponsored"]
    assert "CHE" not in {r["country"] for r in record["reception"]}


def test_facets_ignore_their_own_filter(client):
    base = "/api/v1/resolutions"
    facets = get(client, f"{base}/facets?keyword=nuclear&start_date=2000-01-01&end_date=2009-12-31")
    years = {y["year"]: y["resolutions"] for y in facets["years"]}
    assert min(years) < 2000  # the period does not narrow its own facet
    assert (
        years[2005]
        == get(client, f"{base}?keyword=nuclear&start_date=2005-01-01&end_date=2005-12-31")["total"]
    )

    first = facets["domains"][0]
    narrowed = get(
        client,
        f"{base}/facets?keyword=nuclear&start_date=2000-01-01&end_date=2009-12-31&subject={first['subject_id']}",
    )
    assert narrowed["domains"] == facets["domains"]
    listed = get(
        client,
        f"{base}?keyword=nuclear&start_date=2000-01-01&end_date=2009-12-31&subject={first['subject_id']}",
    )
    assert listed["total"] == first["resolutions"]
    get(client, f"{base}/facets?sponsor=XXX", status=422)


def test_group_vote_filter_and_facet(client):
    base = "/api/v1/resolutions"
    query = "group=CHE,FRA&group=155&group_vote=split&start_date=2000-01-01"  # 155: Western Europe
    counts = {
        g["group_vote"]: g["resolutions"]
        for g in get(client, f"{base}/facets?{query}")["group_votes"]
    }
    assert counts["split"] == get(client, f"{base}?{query}")["total"] > 0
    alike = get(client, f"{base}?{query.replace('split', 'alike')}")["total"]
    assert alike == counts["alike"]
    strict = get(client, f"{base}?{query}&group_all_voted=true")["total"]
    assert strict <= counts["split"]
    assert get(client, f"{base}/facets?start_date=2000-01-01")["group_votes"] == []
    get(client, f"{base}?group=XXX&group_vote=alike", status=422)


def test_adoption_filter_and_facet(client):
    base = "/api/v1/resolutions"
    counts = {
        a["adoption"]: a["resolutions"]
        for a in get(client, f"{base}/facets?adoption=recorded")["adoption"]
    }
    assert set(counts) == {"recorded", "non_recorded", "without_vote"}
    assert sum(counts.values()) == get(client, base)["total"]  # the facet ignores its own filter
    recorded = get(client, f"{base}?adoption=recorded")
    assert recorded["total"] == recorded["recorded_votes"] == counts["recorded"]
    both = get(client, f"{base}?adoption=non_recorded&adoption=without_vote")["total"]
    assert both == counts["non_recorded"] + counts["without_vote"]
    get(client, f"{base}?adoption=maybe", status=422)


def test_subject_domains_and_sponsors(client):
    domains = get(client, "/api/v1/subjects/domains?start_date=2010-01-01")
    counts = [d["resolutions"] for d in domains["domains"]]
    assert counts == sorted(counts, reverse=True)
    assert sum(y["resolutions"] for y in domains["per_year"]) == domains["resolutions"]
    first = domains["domains"][0]
    assert sum(y["resolutions"] for y in first["per_year"]) == first["resolutions"]
    assert first["recorded"] <= first["resolutions"]

    listed = get(client, f"/api/v1/resolutions?subject={first['subject_id']}&start_date=2010-01-01")
    assert listed["total"] == first["resolutions"]
    sponsors = get(client, f"/api/v1/subjects/{first['subject_id']}/sponsors?start_date=2010-01-01")
    assert sponsors["resolutions"] == first["resolutions"]
    assert all(s["original"] <= s["sponsored"] <= s["own_total"] for s in sponsors["sponsors"])
    get(client, "/api/v1/subjects/nope/sponsors", status=404)


def test_agreement_map(client):
    result = get(client, "/api/v1/agreement/map?country=CHE&start_date=2010-01-01")
    scores = {s["country"]: s["score"] for s in result["scores"]}
    assert "CHE" not in scores
    assert 0 <= scores["LIE"] <= 1
    assert result["consensus"]["lo"] <= result["consensus"]["avg"] <= result["consensus"]["hi"]


def test_timeline(client):
    result = get(client, "/api/v1/agreement/timeline?country=USA&compare=GBR&min_votes=3")
    values = [s["values"]["GBR"] for s in result["sessions"]]
    assert all(v["agreement"] is None or 0 <= v["agreement"] <= 1 for v in values)
    assert all(v["agreement"] is None for v in values if v["shared_votes"] < 3)


def test_by_subject_is_sorted_and_thresholded(client):
    result = get(client, "/api/v1/agreement/by-subject?country=USA&compare=GBR&min_votes=50")
    agreements = [s["agreement"] for s in result["subjects"]]
    assert agreements == sorted(agreements, reverse=True)
    assert all(s["shared_votes"] >= 50 for s in result["subjects"])


def test_word_cloud_category_terms_carry_subject_ids(client):
    result = get(client, "/api/v1/keywords/cloud?mode=category&limit=5")
    assert len(result["terms"]) == 5
    assert all(t["subject_ids"] for t in result["terms"])


def test_profile_clamps_to_membership(client):
    profile = get(client, "/api/v1/countries/CHE/profile")
    assert profile["start_date"] == "2002-01-01"  # UN member since 2002
    assert profile["comparison_is_default"]
    # The tally covers recorded votes; the rest were adopted without one or by a non-recorded vote
    assert sum(profile["votes"].values()) == profile["recorded_votes"]
    assert (
        profile["recorded_votes"] + profile["adopted_without_vote"] + profile["non_recorded_votes"]
        == profile["total_resolutions"]
    )


def test_csv_export(client):
    response = get(client, "/api/v1/resolutions.csv?country=CHE&votes=FRA&start_date=2025-01-01")
    assert response.headers["content-type"].startswith("text/csv")
    header = response.text.splitlines()[0].split(",")
    assert header[-2:] == ["Switzerland", "France"]


@pytest.mark.parametrize(
    "url, status",
    [
        ("/api/v1/countries/XYZ", 404),
        ("/api/v1/resolutions/does-not-exist", 404),
        ("/api/v1/resolutions?country=XYZ", 422),
        ("/api/v1/resolutions?vote=Y", 422),
        ("/api/v1/resolutions?country=USA&pair=GBR", 422),
        ("/api/v1/resolutions?limit=0", 422),
        ("/api/v1/resolutions?sponsor=XYZ", 422),
        ("/api/v1/countries/XYZ/sponsorship", 404),
        ("/api/v1/agreement/map", 422),
        ("/api/v1/agreement/timeline?country=USA&compare=USA", 422),
        ("/api/v1/agreement/by-subject?country=USA&compare=USA", 422),
        ("/api/v1/keywords/cloud?mode=bogus", 422),
        ("/api/v1/countries?lang=de", 422),
    ],
)
def test_bad_requests(client, url, status):
    get(client, url, status)


def test_unchanged_data_answers_304(client):
    first = client.get("/api/v1/stats/overview")
    etag = first.headers["etag"]
    assert first.headers["x-data-version"]
    again = client.get("/api/v1/stats/overview", headers={"If-None-Match": etag})
    assert again.status_code == 304
    other = client.get("/api/v1/stats/overview?x=1")
    assert other.headers["etag"] != etag


def test_openapi_schema_is_generated(client):
    schema = get(client, "/api/openapi.json")
    assert "/api/v1/resolutions" in schema["paths"]
    assert "CountryProfile" in schema["components"]["schemas"]
