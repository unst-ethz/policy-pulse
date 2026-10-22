"""Which terms a word cloud shows for a set of resolutions, and how big each one is.

Presentation-free: positions, fonts and colours are the renderer's job. A term is sized by its
*searchable count* — how many of the given resolutions a click on it would return, using the same
keyword search as the resolution filter — rather than by raw frequency, so the number shown next
to a word always matches what clicking it finds.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .keywords import EXCLUDED_TERMS_BY_MODE, KeywordIndex, matched_ids

MAX_TERMS = 20
# Searchable counts cost a keyword search each, so only the top raw-frequency terms are scored.
MAX_CANDIDATES = 30


@dataclass(frozen=True)
class CloudTerm:
    term: str
    count: int  # resolutions a click on the term returns
    frequency: int  # resolutions among the given ones carrying the term


def searchable_count(
    index: KeywordIndex, term: str, mode: str, keyword: str | None, df: pd.DataFrame
) -> int:
    """How many rows of `df` clicking `term` would return (on top of the active `keyword`)."""
    if not term:
        return 0

    if mode == "category":
        # Subject terms select resolutions directly, not through the text search.
        candidate_ids = index.terms(mode).get(term, [])
        if not candidate_ids or df.empty or "undl_id" not in df.columns:
            return 0
        filtered_ids = set(df["undl_id"].astype(str).tolist())
        return int(sum(1 for x in candidate_ids if str(x) in filtered_ids))

    if df.empty or "undl_id" not in df.columns:
        return 0

    cleaned = term.replace('"', "").strip()
    clicked_phrase = f'"{cleaned}"'
    existing_keyword = (keyword or "").strip()
    expression = f"{existing_keyword} & {clicked_phrase}" if existing_keyword else clicked_phrase
    ids = matched_ids(df, expression, index)
    if not ids:
        return 0
    ids_str = {str(x) for x in ids}
    return int(df["undl_id"].astype(str).isin(ids_str).sum())


def cloud_terms(
    index: KeywordIndex,
    df: pd.DataFrame,
    mode: str,
    keyword: str | None = None,
    limit: int = MAX_TERMS,
    candidates: int = MAX_CANDIDATES,
) -> list[CloudTerm]:
    """The top `limit` terms of `mode` across the resolutions in `df`, biggest first.

    `df` is the already-filtered resolution set; `keyword` is the active keyword expression, which
    a click on a term narrows further.
    """
    if df.empty or "undl_id" not in df.columns:
        return []
    frequencies = index.aggregate_term_frequencies(df["undl_id"], mode=mode)
    if not frequencies:
        return []

    by_frequency = sorted(frequencies.items(), key=lambda x: (-x[1], x[0]))
    counts = {
        term: searchable_count(index, term, mode, keyword, df)
        for term, _ in by_frequency[:candidates]
    }
    ranked = sorted(((t, c) for t, c in counts.items() if c > 0), key=lambda x: (-x[1], x[0]))
    excluded = EXCLUDED_TERMS_BY_MODE.get(mode, set())
    if excluded:
        ranked = [(t, c) for t, c in ranked if t.strip().lower() not in excluded]

    # Taking the first N after exclusions promotes later-ranked terms in place of removed ones.
    return [CloudTerm(t, c, frequencies[t]) for t, c in ranked[:limit]]


def mean_consensus(
    index: KeywordIndex, terms: list[str], mode: str, df: pd.DataFrame
) -> list[float | None]:
    """Average consensus score of each term's resolutions within `df` (None if none is scored).

    Note: a term's resolutions are taken from the whole index and then looked up in `df`, so only
    resolutions in `df` with a consensus score contribute.
    """
    term_map = index.terms(mode)
    score_by_id = df.set_index(df["undl_id"].astype(str))["consensus_score"].dropna().to_dict()
    result: list[float | None] = []
    for term in terms:
        scores = [
            score_by_id[rid] for x in term_map.get(term, []) if (rid := str(x)) in score_by_id
        ]
        result.append(float(np.mean(scores)) if scores else None)
    return result
