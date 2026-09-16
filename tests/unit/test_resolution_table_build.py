"""Unit tests for the Postgres -> in-memory reshape in DataRepository.

These cover the parts of the load that turn normalized Postgres rows into the wide frame the query
engine consumes: the long-to-wide vote pivot, the 'X' fill for resolutions nobody voted on, the
`session_label` -> `session` rename, and the UN Digital Library search links.

No database needed — `_build_resolution_table` and `_undl_search_links` are pure functions of their
arguments, and importing the module does not connect to anything.
"""

from urllib.parse import quote

import pandas as pd
import pytest

from app.un_data_stream.data.repository import (
    UNDL_ID_QUERY,
    UNDL_SEARCH_URL,
    UNDL_SYMBOL_QUERY,
    DataRepository,
    _undl_search_links,
)


@pytest.fixture
def outcomes() -> pd.DataFrame:
    """Three resolutions as `resolution_outcomes` hands them over (column names included)."""
    return pd.DataFrame({
        "undl_id": ["100", "200", "300"],
        "resolution": ["A/RES/80/1", "A/RES/13(I)", "A/RES/80/3"],
        "date": ["2025-09-04", "1947-11-13", "2025-09-05"],
        "session_label": ["80", "2", "80"],
        "title": ["First", "Second", "Third"],
        "agenda_title": ["a", "b", "c"],
        "subjects": ["PEACE", None, "HEALTH | UN"],
        "draft": [None, None, None],
        "total_yes": [100, 50, None],
        "total_no": [2, 1, None],
        "total_abstentions": [3, 0, None],
        "total_non_voting": [1, 2, None],
        "total_ms": [106, 53, None],
    })


@pytest.fixture
def votes() -> pd.DataFrame:
    """Votes for two of the three resolutions; '300' was adopted without a vote."""
    return pd.DataFrame({
        "undl_id": ["100", "100", "100", "200", "200"],
        "country_code": ["USA", "CHN", "FRA", "USA", "CHN"],
        "vote": ["Y", "N", "A", "Y", "Y"],
    })


def test_pivot_produces_one_column_per_country(outcomes, votes):
    table, country_columns = DataRepository._build_resolution_table(outcomes, votes)

    assert country_columns == ["CHN", "FRA", "USA"], "country columns must be sorted"
    assert len(table) == 3, "one row per resolution, not per vote"
    row = table.set_index("undl_id").loc["100"]
    assert (row["USA"], row["CHN"], row["FRA"]) == ("Y", "N", "A")


def test_resolutions_with_no_votes_are_filled_with_X(outcomes, votes):
    """A resolution absent from resolution_votes was adopted without a (recorded) vote.

    Every country is 'X' (non-voting) there — that is real information, not missing data, and the
    agreement precomputation depends on it being a vote code rather than NaN.
    """
    table, country_columns = DataRepository._build_resolution_table(outcomes, votes)

    no_vote = table.set_index("undl_id").loc["300"]
    assert list(no_vote[country_columns]) == ["X", "X", "X"]
    # a country that simply didn't vote on an otherwise-voted resolution is also 'X'
    assert table.set_index("undl_id").loc["200", "FRA"] == "X"


def test_vote_columns_are_categorical(outcomes, votes):
    """Regression guard: object dtype here costs ~150 MB on the real dataset."""
    table, country_columns = DataRepository._build_resolution_table(outcomes, votes)

    for col in country_columns:
        assert isinstance(table[col].dtype, pd.CategoricalDtype)
        assert list(table[col].cat.categories) == ["Y", "N", "A", "X"]


def test_session_comes_from_session_label_as_string(outcomes, votes):
    """Postgres has a numeric `session` and a `session_label`; the app needs the label.

    Special sessions are only representable in the label ('10sp', '10emsp'), and
    app/features/agreement_graph.py detects them with a substring match, so the column must stay
    a string.
    """
    table, _ = DataRepository._build_resolution_table(outcomes, votes)

    assert "session_label" not in table.columns
    assert table["session"].tolist() == ["80", "2", "80"]
    assert table["session"].dtype == object


def test_special_session_labels_survive():
    outcomes = pd.DataFrame({
        "undl_id": ["1"], "resolution": ["A/RES/S-10/2"], "date": ["1978-06-30"],
        "session_label": ["10sp"], "title": ["t"], "agenda_title": [None],
        "subjects": [None], "draft": [None], "total_yes": [None], "total_no": [None],
        "total_abstentions": [None], "total_non_voting": [None], "total_ms": [None],
    })
    votes = pd.DataFrame({"undl_id": ["1"], "country_code": ["USA"], "vote": ["Y"]})

    table, _ = DataRepository._build_resolution_table(outcomes, votes)

    assert table["session"].iloc[0] == "10sp"


def test_dates_are_parsed(outcomes, votes):
    table, _ = DataRepository._build_resolution_table(outcomes, votes)

    assert pd.api.types.is_datetime64_any_dtype(table["date"])


# ---------------------------------------------------------------------------
# UN Digital Library links
# ---------------------------------------------------------------------------

def test_symbol_search_link_is_percent_encoded():
    """`undl_id` has no digitallibrary equivalent, so links are searches on the symbol.

    The exact query pattern is being tuned (`UNDL_SYMBOL_QUERY` — field-qualified on MARC 791, or a
    bare phrase for a broader search), so these tests derive the expectation from the constant and
    pin the *encoding* instead: that's the part with a real failure mode, since an unencoded
    symbol's '/' would be read as a URL path.
    """
    (link,) = _undl_search_links(pd.Series(["A/RES/80/311"]), pd.Series(["1472800"]))

    assert link == UNDL_SEARCH_URL + quote(UNDL_SYMBOL_QUERY.format("A/RES/80/311"), safe=":*")
    assert "A%2FRES%2F80%2F311" in link, "the symbol's slashes must be encoded"


@pytest.mark.parametrize("symbol, must_contain", [
    ("A/RES/13(I)", "%28I%29"),          # parentheses, early sessions
    ("A/RES/71/101[B]", "%5BB%5D"),      # bracketed parts
    ("A/RES/2/1", "A%2FRES%2F2%2F1"),    # plain slashes
])
def test_awkward_symbols_are_encoded(symbol, must_contain):
    (link,) = _undl_search_links(pd.Series([symbol]), pd.Series(["1"]))

    assert must_contain in link
    assert link == UNDL_SEARCH_URL + quote(UNDL_SYMBOL_QUERY.format(symbol), safe=":*")


def test_symbols_are_stripped():
    (padded,) = _undl_search_links(pd.Series(["  A/RES/2/1  "]), pd.Series(["1"]))
    (clean,) = _undl_search_links(pd.Series(["A/RES/2/1"]), pd.Series(["1"]))

    assert padded == clean


@pytest.mark.parametrize("missing", [None, "", "   ", float("nan")])
def test_missing_symbol_falls_back_to_id_search(missing):
    """`resolution` is nullable in the schema, and a row without one still needs a link."""
    (link,) = _undl_search_links(pd.Series([missing]), pd.Series(["999001"]))

    assert link == UNDL_SEARCH_URL + UNDL_ID_QUERY.format("999001")
    assert "999001" in link


def test_links_are_attached_to_the_table(outcomes, votes):
    table, _ = DataRepository._build_resolution_table(outcomes, votes)

    assert table["undl_link"].str.startswith(UNDL_SEARCH_URL).all()
    assert table["undl_link"].nunique() == 3
    assert not table["undl_link"].str.contains('"').any(), "quotes must be encoded, not literal"
