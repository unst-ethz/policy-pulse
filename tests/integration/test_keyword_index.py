"""The keyword indices, built from the live `resolution_keyword_cloud` view.

Integration tests because they check properties of the real data (every mode populated, the
default panel being the union of the axes, corpus coverage). The index-building rules themselves
are unit-tested against hand-built frames in tests/unit/test_keywords.py.
"""

import pytest

from app.un_data_stream.analysis.keywords import EXCLUDED_TERMS_BY_AXIS, EXCLUDED_TERMS_BY_MODE

pytestmark = pytest.mark.needs_postgres

# Mirrors the app's four keyword panels; 'category' is excluded deliberately -- it is built from
# `resolution_subject`, not from keywords.
KEYWORD_MODES = ("default", "geopolitical", "thematic", "action")


@pytest.fixture(scope="module")
def index():
    from app import data
    from app.un_data_stream.analysis import keywords

    return keywords.keyword_index(data.query_engine.snapshot)


def test_every_keyword_mode_is_populated(index):
    """All four read one dimension of the same view; an empty one means a dimension is missing
    from it, which no page would report -- the tab just renders a blank cloud."""
    for mode in KEYWORD_MODES:
        assert index.terms(mode), f"no words indexed for mode {mode!r}"
        assert index.resolutions(mode), f"no resolutions indexed for mode {mode!r}"


def test_the_two_indices_describe_the_same_data(index):
    """`resolutions()` sizes the words and `terms()` answers what a click opens. They are
    built together, so a mismatch means a word could render and then resolve to nothing."""
    for mode in KEYWORD_MODES:
        resolution_data = index.resolutions(mode)
        word_map = index.terms(mode)

        for word, undl_ids in word_map.items():
            assert undl_ids, f"{mode}: {word!r} maps to no resolutions"
            sample = undl_ids[0]
            assert word in resolution_data[sample]["word_freq"]


def test_default_mode_is_the_union_of_the_three_axes(index):
    """The view derives 'general' as that union, and the default panel is the only place a user
    sees all of it. If the view ever stopped being a union, the panel would silently narrow.
    """
    default_data = index.resolutions("default")
    axes = [index.resolutions(m) for m in ("geopolitical", "thematic", "action")]

    suppressed = EXCLUDED_TERMS_BY_MODE["default"]
    for undl_id in list(default_data)[:500]:
        from_axes: set[str] = set()
        for axis in axes:
            from_axes |= set(axis.get(undl_id, {}).get("word_freq", {}))
        # Exclusions cut both ways: the default panel suppresses terms an axis may still carry,
        # and an axis suppresses terms the union has. Only the terms neither side drops are
        # comparable.
        assert from_axes - suppressed <= set(default_data[undl_id]["word_freq"])


def test_default_panel_suppresses_what_the_axes_suppress(index):
    """'general' is the union of the axes, so a term excluded as noise on its own tab arrives in
    the default panel too unless the default inherits that exclusion — unlabelled, on the tab
    most readers open first.
    """
    default_map = index.terms("default")

    for axis, terms in EXCLUDED_TERMS_BY_AXIS.items():
        assert terms <= EXCLUDED_TERMS_BY_MODE["default"], f"{axis} exclusions not inherited"
        for term in terms:
            assert term not in default_map, f"{term!r} reached the default panel"


def test_keywords_cover_the_corpus(index):
    """Every resolution should have been through the extractor. A shortfall here is the backfill
    being incomplete, which is otherwise invisible: the cloud renders from whatever it has.
    """
    from app import data

    served = len(data.query_engine.query_resolutions())
    covered = len(index.resolutions("default"))

    assert covered / served >= 0.95, f"only {covered}/{served} resolutions have keywords"
