"""Keyword and subject-term indices, and the keyword search expression built on them.

Moved out of the word-cloud feature so that the resolution filter, the word cloud and the API all
search the same way. Two indices per mode, built together from one snapshot:

- `resolution_terms[mode]`: `{undl_id: {"word_freq": {term: count}}}` — sizes the cloud
- `term_resolutions[mode]`: `{term: [undl_id, ...]}` — answers what a term (or a click) matches

The keyword modes read one dimension each of `resolution_keyword_cloud` (the `keyword_table`);
the `category` mode is built from the thesaurus subjects assigned to each resolution instead.

The index is cached against the snapshot it was built from (`keyword_index(snapshot)`), so a data
reload — which swaps in a new snapshot object — invalidates it without anyone having to remember to.
"""

import logging
import re
import threading
from collections import Counter
from dataclasses import dataclass, field

import pandas as pd
from rapidfuzz import process as fuzz_process

from .snapshot import DataSnapshot

logger = logging.getLogger(__name__)

DEFAULT_MODE = "default"

# mode -> keyword_table dimension it reads. `category` reads the subject tables instead.
MODES: dict[str, dict] = {
    "default": {"label": "Default", "dimension": "general"},
    "geopolitical": {"label": "Geopolitical", "dimension": "geopolitical"},
    "thematic": {"label": "Thematic", "dimension": "thematic"},
    "action": {"label": "Action", "dimension": "action"},
    "category": {"label": "Subjects", "dimension": None},
}

EXCLUDED_TERMS_BY_AXIS = {
    "geopolitical": {"peoples", "states", "united nations"},
}
# Derived rather than repeated so a later addition to an axis cannot forget the union.
EXCLUDED_TERMS_BY_MODE = {
    **EXCLUDED_TERMS_BY_AXIS,
    "default": set().union(*EXCLUDED_TERMS_BY_AXIS.values()),
}

# Subject/category values may be delimited by "|", "--", or ";".
_SUBJECT_SPLIT = re.compile(r"\||--|;")


def build_keyword_indices(pairs: pd.DataFrame, exclude: set[str] | None = None):
    """Index one dimension of `resolution_keyword_cloud` into the two dicts the cloud reads.

    Returns `({undl_id: {"word_freq": {term: 1}}}, {term: [undl_id, ...]})`.
    """
    resolution_terms: dict = {}
    term_resolutions: dict = {}
    exclude_set = exclude or set()

    for undl_id, keyword in pairs[["undl_id", "keyword"]].itertuples(index=False):
        if keyword in exclude_set:
            continue
        entry = resolution_terms.setdefault(undl_id, {"word_freq": {}})
        # Always 1: a keyword either describes a resolution or it does not, and the cloud sizes
        # a word by how many resolutions carry it, not by repetition within one.
        entry["word_freq"][keyword] = 1
        term_resolutions.setdefault(keyword, []).append(undl_id)

    return resolution_terms, term_resolutions


def _build_delimited_indices(frame: pd.DataFrame, column: str, split_pattern: re.Pattern):
    """Index a column of delimiter-joined terms (the `category` mode's subject labels)."""
    resolution_terms: dict = {}
    term_resolutions: dict = {}

    for undl_id, keywords in frame[["undl_id", column]].values:
        if pd.isna(keywords):
            continue

        word_in_resolution: Counter = Counter()
        tokens_set = set()
        for keyword in split_pattern.split(str(keywords)):
            keyword_clean = re.sub(r"\s+", " ", keyword.strip().lower())
            if keyword_clean:
                tokens_set.add(keyword_clean)

        for token in tokens_set:
            word_in_resolution[token] += 1
            term_resolutions.setdefault(token, []).append(undl_id)

        resolution_terms[undl_id] = {"word_freq": dict(word_in_resolution)}

    return resolution_terms, term_resolutions


@dataclass
class KeywordIndex:
    """Every mode's term indices for one snapshot. Treat as read-only once built."""

    resolution_terms: dict[str, dict] = field(default_factory=dict)
    term_resolutions: dict[str, dict] = field(default_factory=dict)
    # Lower-cased subject label -> subject ids, so a click on a `category` term can select the
    # subject filter rather than run a text search.
    category_term_to_subject_ids: dict[str, list[str]] = field(default_factory=dict)

    def terms(self, mode: str) -> dict:
        """`{term: [undl_id, ...]}` for `mode` (empty for an unknown or unpopulated mode)."""
        return self.term_resolutions.get(mode, {})

    def resolutions(self, mode: str) -> dict:
        """`{undl_id: {"word_freq": ...}}` for `mode`."""
        return self.resolution_terms.get(mode, {})

    @classmethod
    def build(cls, snapshot: DataSnapshot) -> "KeywordIndex":
        index = cls()
        resolutions_df = snapshot.resolution_table
        served_ids = (
            set(resolutions_df["undl_id"].astype(str)) if not resolutions_df.empty else set()
        )

        keyword_df = snapshot.keyword_table
        if keyword_df is not None and not keyword_df.empty:
            keyword_df = keyword_df[keyword_df["undl_id"].astype(str).isin(served_ids)]
            by_dimension = dict(tuple(keyword_df.groupby("dimension", observed=True)))
        else:
            logger.warning("resolution_keyword_cloud is empty; keyword word clouds unavailable")
            by_dimension = {}

        for mode, spec in MODES.items():
            dimension = spec["dimension"]
            if dimension is None:
                continue
            pairs = by_dimension.get(dimension)
            if pairs is None or pairs.empty:
                logger.warning("No '%s' rows in resolution_keyword_cloud", dimension)
                index.resolution_terms[mode] = {}
                index.term_resolutions[mode] = {}
                continue
            index.resolution_terms[mode], index.term_resolutions[mode] = build_keyword_indices(
                pairs, EXCLUDED_TERMS_BY_MODE.get(mode)
            )

        index._build_category_mode(snapshot)

        for mode in MODES:
            logger.info(
                "Keyword mode '%s': %d resolutions, %d unique terms",
                mode,
                len(index.resolutions(mode)),
                len(index.terms(mode)),
            )
        return index

    def _build_category_mode(self, snapshot: DataSnapshot) -> None:
        resolution_subject_df = snapshot.resolution_subject_table.copy()
        subject_df = snapshot.subject_table.copy()
        if (
            resolution_subject_df.empty
            or not {"undl_id", "subject_id"} <= set(resolution_subject_df.columns)
            or subject_df.empty
            or "subject_id" not in subject_df.columns
        ):
            logger.warning(
                "Category word cloud data unavailable (need resolution_subject and subject tables)"
            )
            self.resolution_terms["category"] = {}
            self.term_resolutions["category"] = {}
            return

        if "label_en" not in subject_df.columns:
            logger.warning("subject table has no 'label_en' column; using subject_id")
            subject_df["label_en"] = subject_df["subject_id"]

        subject_lookup = (
            subject_df[["subject_id", "label_en"]]
            .dropna(subset=["subject_id"])
            .drop_duplicates(subset=["subject_id"])
            .set_index("subject_id")["label_en"]
            .to_dict()
        )
        for subject_id, label in subject_lookup.items():
            key = re.sub(r"\s+", " ", str(label).strip().lower())
            if key:
                self.category_term_to_subject_ids.setdefault(key, []).append(subject_id)

        resolution_subject_df["category_term"] = (
            resolution_subject_df["subject_id"]
            .map(subject_lookup)
            .fillna(resolution_subject_df["subject_id"])
            .astype(str)
        )
        category_terms_by_resolution = (
            resolution_subject_df.groupby("undl_id")["category_term"]
            .apply(lambda s: "; ".join(sorted(set(s))))
            .reset_index(name="category_terms")
        )
        category_data_all = pd.merge(
            snapshot.resolution_table[["undl_id"]],
            category_terms_by_resolution,
            on="undl_id",
            how="left",
        )
        self.resolution_terms["category"], self.term_resolutions["category"] = (
            _build_delimited_indices(category_data_all, "category_terms", _SUBJECT_SPLIT)
        )

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self, token: str, mode: str = DEFAULT_MODE, exact: bool = False, score_cutoff: int = 80
    ) -> set:
        """undl_ids whose terms match `token`: exact term if `exact`, else substring + fuzzy."""
        matched_ids: set = set()
        token_lower = token.lower().strip()
        if not token_lower:
            return matched_ids

        word_map = self.terms(mode)
        if exact:
            if token_lower in word_map:
                matched_ids.update(word_map[token_lower])
            return matched_ids

        all_keys = list(word_map.keys())
        for key in all_keys:
            if token_lower in key:
                matched_ids.update(word_map[key])

        if all_keys:
            fuzzy_results = fuzz_process.extract(
                token_lower, all_keys, score_cutoff=score_cutoff, limit=20
            )
            for match_key, _score, _idx in fuzzy_results:
                matched_ids.update(word_map[match_key])

        return matched_ids

    def aggregate_term_frequencies(self, undl_ids: pd.Series, mode: str = DEFAULT_MODE) -> dict:
        """Sum each term's per-resolution counts over `undl_ids`."""
        counter: Counter = Counter()
        resolution_terms = self.resolutions(mode)
        for undl_id in undl_ids.values:
            if undl_id in resolution_terms:
                counter.update(resolution_terms[undl_id]["word_freq"])
        return dict(counter)


def parse_keyword_term(term: str) -> tuple[str, bool]:
    """Strip one search term; a term wrapped in double quotes asks for an exact keyword match."""
    term = term.strip()
    exact_mode = len(term) >= 2 and term[0] == '"' and term[-1] == '"'
    parsed = term[1:-1].strip() if exact_mode else term
    return parsed, exact_mode


def matched_ids(df: pd.DataFrame, keyword_expression: str | None, index: KeywordIndex) -> set:
    """Resolve a keyword expression to the undl_ids it matches.

    Semantics:
    - ',' => OR between clauses
    - '&' => AND within a clause
    - quoted terms => exact keyword index match

    A term matches a resolution if it occurs in the title of a row of `df`, or (via the `default`
    keyword index) in its keywords. Index hits are not restricted to `df`; callers intersect.
    """
    if (
        df.empty
        or "undl_id" not in df.columns
        or not keyword_expression
        or not str(keyword_expression).strip()
    ):
        return set()

    result: set = set()
    clauses = [c.strip() for c in str(keyword_expression).split(",") if c.strip()]

    for clause in clauses:
        terms = [t.strip() for t in clause.split("&") if t.strip()]
        if not terms:
            continue
        clause_ids: set | None = None
        for term in terms:
            parsed_token, exact_mode = parse_keyword_term(term)
            if not parsed_token:
                continue
            token_ids: set = set()
            title_match = (
                df["title"].str.lower().str.contains(parsed_token.lower(), regex=False, na=False)
            )
            token_ids |= set(df.loc[title_match, "undl_id"].tolist())
            token_ids |= index.search(parsed_token, exact=exact_mode)
            if clause_ids is None:
                clause_ids = token_ids
            else:
                clause_ids &= token_ids
            if not clause_ids:
                break
        if clause_ids:
            result |= clause_ids

    return result


# ---------------------------------------------------------------------------
# Per-snapshot cache
# ---------------------------------------------------------------------------

_cached: tuple[DataSnapshot, KeywordIndex] | None = None
_lock = threading.Lock()


def keyword_index(snapshot: DataSnapshot) -> KeywordIndex:
    """The index for `snapshot`, built on first use and reused until the snapshot is replaced.

    Keyed on identity, not equality: a reload always produces a new snapshot object, and comparing
    snapshots by value would mean comparing every DataFrame in them.
    """
    global _cached
    cached = _cached
    if cached is not None and cached[0] is snapshot:
        return cached[1]
    with _lock:
        if _cached is not None and _cached[0] is snapshot:
            return _cached[1]
        index = KeywordIndex.build(snapshot)
        _cached = (snapshot, index)
        return index
