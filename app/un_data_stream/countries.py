"""UN authority names, shared by API and legacy UI; no I/O on import."""

from typing import Any
import pandas as pd

SUPPORTED_LANGS = ("en", "fr", "es", "ar", "zh", "ru")

_LANG_COL = {
    "en": "Member State",
    "fr": "French",
    "es": "Spanish",
    "ar": "Arabic",
    "zh": "Chinese",
    "ru": "Russian",
}

_LEGACY_VOTING_CODES: dict[str, dict[str, Any]] = {
    "GER": {  # West Germany; voting records 1973-1990
        "en": "Germany, Federal Republic of",
        "fr": "République fédérale d'Allemagne",
        "es": "República Federal de Alemania",
        "ru": "Федеративная Республика Германия",
        "_aliases": ["West Germany", "BRD"],
        "_year_range": (1973, 1990),
    },
    "SCG": {  # Serbia and Montenegro; voting records 2003-2006
        "en": "Serbia and Montenegro",
        "fr": "Serbie-et-Monténégro",
        "es": "Serbia y Montenegro",
        "ru": "Сербия и Черногория",
        "_aliases": ["Federal Republic of Yugoslavia"],
        "_year_range": (2003, 2006),
    },
}


def _parse_year(date_str) -> int | None:
    """Extract the earliest 4-digit year from a possibly comma-separated date string."""
    if pd.isna(date_str):
        return None
    for part in str(date_str).split(","):
        head = part.strip()[:4]
        if head.isdigit():
            return int(head)
    return None


def _parse_years(date_str) -> list[int]:
    """Return all 4-digit years found in a possibly comma-separated date string."""
    if pd.isna(date_str):
        return []
    years: list[int] = []
    for part in str(date_str).split(","):
        head = part.strip()[:4]
        if head.isdigit():
            years.append(int(head))
    return years


def _build_name_index(member_states_df: pd.DataFrame) -> dict[str, dict]:
    """Build per-ISO name records keyed by ISO3.

    Each record contains:
        names:           dict[lang, str]        - canonical name per language (English fallback)
        display_aliases: list[str]              - real historical names (predecessor entities) for display
        search_aliases:  list[str]              - display aliases plus variant spellings, translations,
                                                  abbreviations from the 'Other Names' column
        year_range:      tuple[int, int] | None - set only for retired ISOs (no active row)
    """

    # A row is "currently active" if either:
    #  (a) End date is NaN, OR
    #  (b) it encodes multi-period membership where the final period is open
    #      (more Start segments than End segments in the comma-separated strings,
    #      e.g. KHM Cambodia: starts 1955/1975/1990, ends 1970/1976 -> third period is open).
    def _is_active(row) -> bool:
        if pd.isna(row["End date"]):
            return True
        return len(_parse_years(row["Start date"])) > len(_parse_years(row["End date"]))

    index: dict[str, dict] = {}

    for iso, group in member_states_df.groupby("ISO Code"):
        if not isinstance(iso, str) or not iso.strip():
            continue

        active_mask = group.apply(_is_active, axis=1)
        active = group[active_mask]
        if len(active) >= 1:
            canonical = active.iloc[0]
            year_range = None
        else:
            # Retired ISO: pick row with latest end year as canonical, attach year range
            with_end = group.assign(_end=group["End date"].apply(_parse_year))
            canonical = with_end.sort_values("_end").iloc[-1]
            start_years = [y for s in group["Start date"] for y in _parse_years(s)]
            end_years = [y for s in group["End date"] for y in _parse_years(s)]
            year_range = (min(start_years), max(end_years)) if start_years and end_years else None

        en_name = canonical["Member State"]
        names: dict[str, str] = {"en": en_name}
        for lang in SUPPORTED_LANGS:
            if lang == "en":
                continue
            val = canonical.get(_LANG_COL[lang])
            names[lang] = val if isinstance(val, str) and val.strip() else en_name

        # Display aliases: only the Member State names of non-canonical rows. These are the
        # real predecessor entities (e.g. "Burma" -> "Myanmar"), not variant spellings.
        display_aliases: list[str] = []
        for _, row in group.iterrows():
            other_name = row["Member State"]
            if (
                other_name != en_name
                and isinstance(other_name, str)
                and other_name not in display_aliases
            ):
                display_aliases.append(other_name)

        # Search aliases: display aliases plus everything in 'Other Names' (variant spellings,
        # translations, abbreviations like "FYROM"/"BRD"/"Soviet Union") so search remains permissive.
        search_aliases: list[str] = list(display_aliases)
        for other in group["Other Names"].dropna():
            token = str(other).strip()
            if token and token != en_name and token not in search_aliases:
                search_aliases.append(token)

        index[iso] = {
            "names": names,
            "display_aliases": display_aliases,
            "search_aliases": search_aliases,
            "year_range": year_range,
        }

    # Merge legacy voting-only codes (see _LEGACY_VOTING_CODES above).
    for iso, payload in _LEGACY_VOTING_CODES.items():
        if iso in index:
            continue
        names = {lang: payload.get(lang, payload["en"]) for lang in SUPPORTED_LANGS}
        legacy_aliases = list(payload.get("_aliases", []))
        index[iso] = {
            "names": names,
            "display_aliases": legacy_aliases,
            "search_aliases": legacy_aliases,
            "year_range": payload.get("_year_range"),
        }

    return index


class CountryCatalog:
    def __init__(self, member_states: pd.DataFrame):
        self.index = _build_name_index(member_states)

    def get_country_name(self, iso3_code: str | None, lang: str = "en") -> str:
        """Return the country name for an ISO3 code in the requested language.

        Falls back to English when the requested language has no translation.
        Appends a (start-end) year range for retired ISOs (e.g. SUN, CSK).
        """
        if not iso3_code:
            return "Unknown"
        rec = self.index.get(iso3_code)
        if not rec:
            return iso3_code
        name = rec["names"].get(lang) or rec["names"]["en"]
        if rec["year_range"]:
            start, end = rec["year_range"]
            return f"{name} ({start}–{end})"
        return name

    def get_country_display_name(self, iso3_code: str, lang: str = "en") -> str:
        """Country name with historical aliases in brackets, e.g. 'Myanmar (historical: Burma)'."""
        base_name = self.get_country_name(iso3_code, lang=lang)
        rec = self.index.get(iso3_code)
        if not rec or not rec["display_aliases"]:
            return base_name
        return f"{base_name} (historical: {', '.join(rec['display_aliases'])})"

    def get_country_search_terms(self, iso3_code: str, lang: str = "en") -> str:
        """Search string including the localised name, the English name, and all variant aliases."""
        rec = self.index.get(iso3_code)
        if not rec:
            return iso3_code
        terms = {rec["names"]["en"], rec["names"].get(lang, rec["names"]["en"])}
        terms.update(rec["search_aliases"])
        return " ".join(t for t in terms if t)
