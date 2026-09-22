"""The word-cloud keyword indices, built from the `resolution_keyword_cloud` view.

Integration rather than unit tests only because importing the feature module pulls in
`app.data`, which reads Postgres at import time. The invariant tests below still run against
hand-built frames, so they fail for the reason they name rather than because live data moved.
"""

import pandas as pd
import pytest

pytestmark = pytest.mark.needs_postgres

# Mirrors the app's four keyword panels; 'category' is excluded deliberately -- it is built from
# `resolution_subject`, not from keywords.
KEYWORD_MODES = ("default", "geopolitical", "thematic", "action")


@pytest.fixture(scope="module")
def wc():
    from app.features import wordcloud_interactive

    wordcloud_interactive._init_wc_data()
    return wordcloud_interactive


def _pairs(rows: list[tuple[str, str]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["undl_id", "keyword"])


def test_a_term_containing_a_comma_is_indexed_whole(wc):
    """The CSV assets this replaced were split on `[;,]` at read time, because they packed a
    resolution's keywords into one cell. The table stores one term per row, and ~100 of those
    terms contain a comma -- re-introducing a split would shatter each into fragments that match
    no resolution and clutter the cloud with words nobody can click through.
    """
    term = "national or ethnic, religious and linguistic minorities"
    resolution_data, word_map = wc._build_keyword_indices(_pairs([("1", term)]))

    assert list(word_map) == [term]
    assert list(resolution_data["1"]["word_freq"]) == [term]


def test_excluded_terms_leave_neither_index(wc):
    """A term dropped from the cloud but left in the reverse index would still be searchable and
    still colour the resolution list, from a word the user was never shown."""
    resolution_data, word_map = wc._build_keyword_indices(
        _pairs([("1", "states"), ("1", "cyprus")]), exclude={"states"}
    )

    assert set(word_map) == {"cyprus"}
    assert set(resolution_data["1"]["word_freq"]) == {"cyprus"}


def test_a_resolution_counts_once_per_term(wc):
    """The cloud sizes a word by how many resolutions carry it. A per-resolution count above 1
    would let one resolution outvote several."""
    resolution_data, word_map = wc._build_keyword_indices(
        _pairs([("1", "disarmament"), ("2", "disarmament")])
    )

    assert set(resolution_data["1"]["word_freq"].values()) == {1}
    assert word_map["disarmament"] == ["1", "2"]


def test_every_keyword_mode_is_populated(wc):
    """All four read one dimension of the same view; an empty one means a dimension is missing
    from it, which no page would report -- the tab just renders a blank cloud."""
    for mode in KEYWORD_MODES:
        assert wc._word_undlid_map(mode), f"no words indexed for mode {mode!r}"
        assert wc._wc_data(mode), f"no resolutions indexed for mode {mode!r}"


def test_the_two_indices_describe_the_same_data(wc):
    """`_wc_data` sizes the words and `_word_undlid_map` answers what a click opens. They are
    built together, so a mismatch means a word could render and then resolve to nothing."""
    for mode in KEYWORD_MODES:
        resolution_data = wc._wc_data(mode)
        word_map = wc._word_undlid_map(mode)

        for word, undl_ids in word_map.items():
            assert undl_ids, f"{mode}: {word!r} maps to no resolutions"
            sample = undl_ids[0]
            assert word in resolution_data[sample]["word_freq"]


def test_default_mode_is_the_union_of_the_three_axes(wc):
    """The view derives 'general' as that union, and the default panel is the only place a user
    sees all of it. If the view ever stopped being a union, the panel would silently narrow.
    """
    default_data = wc._wc_data("default")
    axes = [wc._wc_data(m) for m in ("geopolitical", "thematic", "action")]

    suppressed = wc._EXCLUDED_TERMS_BY_MODE["default"]
    for undl_id in list(default_data)[:500]:
        from_axes: set[str] = set()
        for axis in axes:
            from_axes |= set(axis.get(undl_id, {}).get("word_freq", {}))
        # Exclusions cut both ways: the default panel suppresses terms an axis may still carry,
        # and an axis suppresses terms the union has. Only the terms neither side drops are
        # comparable.
        assert from_axes - suppressed <= set(default_data[undl_id]["word_freq"])


def test_default_panel_suppresses_what_the_axes_suppress(wc):
    """'general' is the union of the axes, so a term excluded as noise on its own tab arrives in
    the default panel too unless the default inherits that exclusion — unlabelled, on the tab
    most readers open first.
    """
    default_map = wc._word_undlid_map("default")

    for axis, terms in wc._EXCLUDED_TERMS_BY_AXIS.items():
        assert terms <= wc._EXCLUDED_TERMS_BY_MODE["default"], f"{axis} exclusions not inherited"
        for term in terms:
            assert term not in default_map, f"{term!r} reached the default panel"


def test_keywords_cover_the_corpus(wc):
    """Every resolution should have been through the extractor. A shortfall here is the backfill
    being incomplete, which is otherwise invisible: the cloud renders from whatever it has.
    """
    from app import data

    served = len(data.query_engine.query_resolutions())
    covered = len(wc._wc_data("default"))

    assert covered / served >= 0.95, f"only {covered}/{served} resolutions have keywords"
