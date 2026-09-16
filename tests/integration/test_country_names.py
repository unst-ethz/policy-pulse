"""Country name resolution against the real `member_states` table.

`app/data.py`'s `_build_name_index` was rewritten when the app moved to Postgres: the authority
table now arrives as `iso_code`/`name_*`/`other_names`/`coverage_periods`/`record_type` instead of
the old bulk-CSV headers, and "is this state current?" is answered by `record_type == 'ms'` rather
than by comparing counts of start/end dates.

These assert behaviour rather than exact upstream strings wherever the underlying record could
legitimately change (states do get renamed — NRU became "Naoero" in 2026), so a failure here means
our logic broke, not that the UN edited a record.
"""

import re

import pytest

pytestmark = pytest.mark.needs_postgres

# (start–end) appended to retired ISOs by get_country_name; en dash, not hyphen.
YEAR_RANGE = re.compile(r"\((\d{4})–(\d{4})\)")


@pytest.fixture(scope="module")
def data():
    from app import data as app_data
    return app_data


def test_current_state_resolves_to_a_real_name(data):
    name = data.get_country_name("USA")

    assert name and name != "USA"
    assert not YEAR_RANGE.search(name), "a current member state must not carry a year range"


def test_every_voting_country_resolves(data):
    """Every code appearing in the vote data must resolve, including legacy ones."""
    unresolved = [c for c in data.available_countries if data.get_country_name(c) == c]

    assert not unresolved, f"codes falling back to the bare ISO code: {unresolved}"


@pytest.mark.parametrize("iso", ["SUN", "CSK", "YUG", "DDR"])
def test_retired_isos_carry_a_year_range(data, iso):
    """ISOs with no `record_type == 'ms'` row are dissolved states; show when they existed."""
    match = YEAR_RANGE.search(data.get_country_name(iso))

    assert match, f"{iso} should show a year range"
    start, end = int(match.group(1)), int(match.group(2))
    assert 1945 <= start <= end <= 2026


def test_predecessor_names_are_offered_for_display(data):
    """A former-state row under the same ISO is a real historical name, not a spelling variant."""
    display = data.get_country_display_name("MMR")

    assert "historical:" in display
    assert "Burma" in display


def test_renamed_current_state_keeps_its_former_name_as_history(data):
    """NRU has both an `ms` row and an `fs` row; the current one must win the display slot."""
    display = data.get_country_display_name("NRU")
    plain = data.get_country_name("NRU")

    assert display.startswith(plain)
    assert "historical:" in display
    assert not YEAR_RANGE.search(plain), "NRU is still a member, so no year range"


@pytest.mark.parametrize("legacy, expected_range", [("GER", (1973, 1990)), ("SCG", (2003, 2006))])
def test_legacy_voting_codes_resolve_via_their_successor_record(data, legacy, expected_range):
    """GER and SCG appear in voting data but have no member_states row of their own.

    They are resolved through the former-state row filed under the successor's ISO (DEU, SRB), so
    the name and the year range come from the database rather than from hardcoded strings.
    """
    name = data.get_country_name(legacy)

    assert name != legacy
    match = YEAR_RANGE.search(name)
    assert match, f"{legacy} should show the years it voted"
    assert (int(match.group(1)), int(match.group(2))) == expected_range


def test_translations_are_used_when_present(data):
    """Current member states carry all six languages."""
    assert data.get_country_name("DEU", lang="fr") != data.get_country_name("DEU", lang="en")
    assert data.get_country_name("DEU", lang="fr") == "Allemagne"


def test_translation_falls_back_to_english(data):
    """Former-state records almost never carry translations (1 of 50 does).

    Documenting the gap rather than pretending otherwise: a retired ISO renders its English name in
    a non-English UI. If upstream ever fills these in, this test starts failing and should simply
    be deleted.
    """
    assert "USSR" in data.get_country_name("SUN", lang="ru")


def test_legacy_codes_keep_their_patched_translations(data):
    """The one place hardcoded translations remain, because the source rows have none."""
    assert data.get_country_name("GER", lang="fr").startswith("République fédérale")


def test_search_terms_include_variants_and_english(data):
    """Search must match variant spellings and abbreviations from `other_names`."""
    terms = data.get_country_search_terms("SUN").lower()

    assert "ussr" in terms
    assert "soviet union" in terms

    myanmar = data.get_country_search_terms("MMR").lower()
    assert "myanmar" in myanmar and "burma" in myanmar


def test_search_terms_include_the_localised_name(data):
    terms = data.get_country_search_terms("DEU", lang="fr")

    assert "Allemagne" in terms
    assert "Germany" in terms, "the English name stays searchable in any UI language"
