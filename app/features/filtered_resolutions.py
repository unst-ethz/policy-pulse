"""The filtered resolution set every tab works from.

Each tab derives this for itself, per callback, instead of reading a `dcc.Store` that one central
callback published. The store held the whole filtered table as JSON, which meant the browser
downloaded up to 6.3 MB and then **uploaded it back** with every callback that took it as an
input — eight of them, so ~57 MB over the wire and ~600 ms of JSON parsing per filter change,
to cache a query that costs 3-15 ms to run. Deriving it is both cheaper and simpler: the criteria
already live in `filter-component-filter-store` (a few hundred bytes).

It also removes the dtype hazard the round-trip carried: `undl_id` is all digits, so pandas' JSON
reader inferred int64 and the ids silently stopped matching the query engine's string keys.

Kept in its own module so the tabs and `filters.py` can share it without an import cycle
(`filters.py` imports `wordcloud_interactive`, which reads this).
"""

import pandas as pd

from .. import data

# The country participation filter is meaningless where country1 is disabled or highlight-only.
TABS_WITHOUT_COUNTRY_FILTER = {"wordcloud"}
# Keyword search only narrows the tabs that actually present resolutions.
TABS_WITH_KEYWORD = {"resolution_list", "wordcloud"}

# The columns tabs read. Deliberately not the whole frame: `resolution_list` and
# `agreement_choropleth` branch on `if country1 in df.columns`, so handing them the 202 vote
# columns would silently switch on per-country display paths that have been dormant. Those
# branches predate this module — see the commented-out `vote_cols` block in filters.py — and
# turning them on is a feature decision, not part of removing the store.
COLUMNS = [
    "undl_id",
    "resolution",
    "session",
    "date",
    "title",
    "consensus_score",
    "total_yes",
    "total_no",
    "total_abstentions",
    "modality",
    "undl_link",
]


def resolutions_for(filter_data: dict | None, active_tab: str | None = None) -> pd.DataFrame:
    """The resolutions matching `filter_data`, as `active_tab` should see them.

    `active_tab` matters because two filters are tab-dependent: the country participation filter
    does not apply on tabs where country1 is highlight-only, and keyword search only applies to
    the tabs that list resolutions. Passing the *active* tab (rather than the calling tab's own
    id) keeps every tab showing the same set, which is what the shared store did.
    """
    if not filter_data:
        return pd.DataFrame(columns=COLUMNS)

    df = data.query_engine.query_resolutions(
        start_date=filter_data.get("start_date"),
        end_date=filter_data.get("end_date"),
        subject_ids=filter_data.get("subject_ids"),
        include_descendants=True,
    )

    country = filter_data.get("country1_alpha3")
    mode = filter_data.get("country_filter_mode") or "none"  # 'none' applies no country filter
    if (
        mode in ("voted", "member")
        and country in df.columns
        and active_tab not in TABS_WITHOUT_COUNTRY_FILTER
    ):
        # No normalization: the vote columns are a category of exactly Y/N/A/X with no nulls
        # (repository.py), and normalizing would decategorize 20k values on every callback.
        cast_a_vote = df[country].isin(["Y", "N", "A"])
        if mode == "voted":
            df = df[cast_a_vote]
        else:
            # Member on the day, *or* voted on this resolution — see "Membership and voting
            # activity" in app/data.py for why the second half is not redundant.
            df = df[data.membership_mask(country, df["date"]) | cast_a_vote]

    keyword = filter_data.get("keyword")
    if keyword and keyword.strip() and active_tab in TABS_WITH_KEYWORD and not df.empty:
        # Imported here, not at module scope: wordcloud_interactive reads this module.
        from .wordcloud_interactive import get_keyword_matched_ids

        matched_ids = get_keyword_matched_ids(df, keyword)
        df = df[df["undl_id"].isin(matched_ids)]

    return df[[column for column in COLUMNS if column in df.columns]]
