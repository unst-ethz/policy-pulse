"""Independent arithmetic oracles protect definitions across the HTTP boundary."""

import ast
from itertools import product
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.main import create_app
from backend.models import Filters
from backend.service import AnalysisService
from tests.support.repository import ASSETS, CHILD_SUBJECT, ROOT_SUBJECT, MemoryRepository

ENCODING = {"Y": 1, "A": 0, "N": -1}


def bilateral(a, b):
    if a not in ENCODING or b not in ENCODING:
        return None
    return 1 - abs(ENCODING[a] - ENCODING[b]) / 2


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def assert_optional(actual, expected):
    if expected is None:
        assert actual is None
    else:
        assert actual == pytest.approx(expected, abs=1e-6)


def test_every_vote_pair_and_missing_case_survives_json():
    rows = [
        {"undl_id": i, "USA": a, "CHE": b}
        for i, (a, b) in enumerate(product(["Y", "N", "A", "X", None], repeat=2))
    ]
    service = AnalysisService(MemoryRepository(pd.DataFrame(rows)), ASSETS)
    with TestClient(create_app(service=service)) as client:
        for row in rows:
            record = client.get(f"/api/v1/resolutions/{row['undl_id']}").json()
            assert_optional(record["consensus"], bilateral(row["USA"], row["CHE"]))


def test_bilateral_average_and_denominators_match_raw_votes(client, service):
    frame = service.engine.resolution_table
    for country in service.engine.country_columns:
        result = client.get(f"/api/v1/analysis/agreement/{country}").json()
        for row in result["items"]:
            expected = [
                bilateral(a, b) for a, b in zip(frame[country], frame[row["country"]], strict=False)
            ]
            assert_optional(row["score"], mean(expected))
            assert row["shared_votes"] == sum(v is not None for v in expected)


def test_multilateral_is_mean_of_resolution_means_not_pooled_pairs(client, service):
    frame = service.engine.resolution_table
    result = client.get("/api/v1/analysis/multilateral").json()
    for item in result["items"]:
        c = item["country"]
        expected = []
        for row in frame.to_dict("records"):
            expected.append(
                mean(
                    [
                        bilateral(row[c], row[other])
                        for other in service.engine.country_columns
                        if other != c
                    ]
                )
            )
        assert_optional(item["multilateral_alignment"], mean(expected))
        counts = frame[c].value_counts()
        cast = sum(int(counts.get(v, 0)) for v in ENCODING)
        assert item["participation_count"] == cast
        for vote, field in [("Y", "yes_rate"), ("N", "no_rate"), ("A", "abstention_rate")]:
            assert item[field] == pytest.approx(int(counts.get(vote, 0)) / cast)
        assert sum(
            item[field] for field in ["yes_rate", "no_rate", "abstention_rate"]
        ) == pytest.approx(1)


def test_timeline_uses_session_median_year_and_threshold():
    rows = [
        {"undl_id": i, "date": date, "session": session, "USA": "Y", "CHE": vote}
        for i, (date, session, vote) in enumerate(
            [
                ("2023-12-01", "78", "Y"),
                ("2024-01-01", "78", "A"),
                ("2024-02-01", "78", "N"),
                ("2024-05-01", "79", "Y"),
                ("2024-05-02", "79", "Y"),
                ("2024-06-01", "1sp", "N"),
                ("2024-06-02", "1sp", "N"),
                ("2024-06-03", "1sp", "N"),
            ]
        )
    ]
    service = AnalysisService(MemoryRepository(pd.DataFrame(rows)), ASSETS)
    with TestClient(create_app(service=service)) as client:
        points = client.get("/api/v1/analysis/timeline/USA?compare=CHE").json()["items"]
        assert points == [
            {
                "session": "78",
                "year": 2024,
                "special": False,
                "scores": {"CHE": 0.5},
                "shared_votes": {"CHE": 3},
            },
            {
                "session": "79",
                "year": 2024,
                "special": False,
                "scores": {"CHE": None},
                "shared_votes": {"CHE": 2},
            },
        ]
        special = client.get(
            "/api/v1/analysis/timeline/USA?compare=CHE&include_special=true"
        ).json()["items"]
        assert next(p for p in special if p["special"])["scores"]["CHE"] == 0


def test_subject_threshold_descendants_and_no_duplicate_weighting(service):
    raw = pd.DataFrame(
        [{"undl_id": i, "USA": "Y", "CHE": "A" if i < 15 else "N"} for i in range(30)]
    )
    repo = MemoryRepository(raw)
    repo.data["resolution_subject"] = pd.DataFrame(
        [{"undl_id": i, "subject_id": s} for i in range(30) for s in [CHILD_SUBJECT, ROOT_SUBJECT]]
    )
    engine = AnalysisService(repo, ASSETS)
    result = engine.subjects("USA", "CHE")["items"]
    row = next(r for r in result if r["subject_id"] == ROOT_SUBJECT)
    assert row["shared_votes"] == 30
    assert row["score"] == 0.25
    repo.data["resolution_subject"] = repo.data["resolution_subject"][
        repo.data["resolution_subject"].undl_id != 29
    ]
    engine = AnalysisService(repo, ASSETS)
    assert engine.subjects("USA", "CHE")["items"] == []


def test_profile_rankings_yearly_means_and_membership(client, service):
    result = client.get("/api/v1/countries/USA/profile?compare=CHE").json()
    assert result["resolution_count"] == 160
    assert result["minimum_shared_votes"] == 100
    assert result["most_aligned"]
    raw = service.engine.resolution_table
    for row in result["most_aligned"] + result["least_aligned"]:
        expected = [bilateral(a, b) for a, b in zip(raw.USA, raw[row["country"]], strict=False)]
        assert_optional(row["score"], mean(expected))
        assert row["shared_votes"] >= 100
    assert_optional(
        result["yearly"][0]["scores"]["CHE"],
        mean([bilateral(a, b) for a, b in zip(raw.USA, raw.CHE, strict=False)]),
    )
    historical = client.get("/api/v1/countries/GER/profile").json()
    assert historical["resolution_count"] == 0
    assert historical["alignment"] is None
    assert historical["rank"] is None


def test_keyword_expression_matches_legacy_function(service):
    # Execute only the legacy pure functions, with explicitly injected data.
    # This compares against the still-present reference behavior without Dash.
    from rapidfuzz import process

    source = Path("app/features/wordcloud_interactive.py").read_text()
    names = {"search_keywords", "_parse_keyword_term", "get_keyword_matched_ids"}
    nodes = [
        n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name in names
    ]
    scope = {
        "pd": pd,
        "fuzz_process": process,
        "_initialized": True,
        "_DEFAULT_MODE": "default",
        "_wc_word_undlid_map_by_mode": service.keywords.words,
    }
    exec(
        compile(ast.Module(body=nodes, type_ignores=[]), "<legacy-keyword-reference>", "exec"),
        scope,
    )
    frame = service.engine.resolution_table
    for expression in [
        "nuclear",
        "cooperation, disarmament",
        "nuclear & disarmament",
        '"nuclear"',
        "nucler",
        ".*",
        '"" & nuclear',
        "nuclear & absent, development",
    ]:
        assert service.keywords.matching_ids(frame, expression) == scope["get_keyword_matched_ids"](
            frame, expression
        )


def test_word_counts_use_once_per_resolution_and_existing_source(service):
    result = service.words(Filters(), "category")
    words = {w["term"]: w for w in result["items"]}
    links = service.engine.resolution_subject_table
    expected = links[links.subject_id == CHILD_SUBJECT].undl_id.nunique()
    assert words["nuclear disarmament"]["count"] == expected
    frame = service.engine.resolution_table
    ids = set(links[links.subject_id == CHILD_SUBJECT].undl_id)
    assert words["nuclear disarmament"]["consensus"] == pytest.approx(
        frame[frame.undl_id.isin(ids)].consensus_score.mean()
    )


def test_nonvote_filter_keeps_legacy_recorded_code_semantics():
    repo = MemoryRepository(
        pd.DataFrame(
            [
                {"undl_id": 1, "USA": "X", "CHE": "X"},
                {"undl_id": 2, "USA": "X", "CHE": "Y"},
                {"undl_id": 3, "USA": None, "CHE": "Y"},
            ]
        )
    )
    with TestClient(create_app(service=AnalysisService(repo, ASSETS))) as client:
        agreed = client.get("/api/v1/resolutions?country=USA&compare=CHE&agreement=AGREED").json()
        assert [r["id"] for r in agreed["items"]] == ["1"]
        assert agreed["items"][0]["consensus"] is None
        disagreed = client.get(
            "/api/v1/resolutions?country=USA&compare=CHE&agreement=DISAGREED"
        ).json()
        assert [r["id"] for r in disagreed["items"]] == ["2"]


def test_map_midpoint_preserves_legacy_reference_selection(client, service):
    frame = service.engine.resolution_table
    expected = frame.loc[frame.IND.notna(), "consensus_score"].mean()
    result = client.get("/api/v1/analysis/agreement/IND").json()
    assert result["consensus_midpoint"] == pytest.approx(expected)
    assert result["reference_longitude"] == pytest.approx(78.96)
    empty = client.get("/api/v1/analysis/agreement/IND?end_date=1800-01-01").json()
    assert empty["consensus_midpoint"] is None
