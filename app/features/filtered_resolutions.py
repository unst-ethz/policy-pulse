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
from ..un_data_stream.analysis import filtering

# The country participation filter is meaningless where country1 is disabled or highlight-only.
TABS_WITHOUT_COUNTRY_FILTER = {"wordcloud"}
# Keyword search only narrows the tabs that actually present resolutions.
TABS_WITH_KEYWORD = {"resolution_list", "wordcloud"}

# The columns tabs read. Deliberately not the whole frame: `resolution_list` and
# `agreement_choropleth` branch on `if country1 in df.columns`, so handing them the 202 vote
# columns would silently switch on per-country display paths that have been dormant. Those
# branches predate this module — see the commented-out `vote_cols` block in filters.py — and
# turning them on is a feature decision, not part of removing the store.
COLUMNS = filtering.COLUMNS


def resolutions_for(filter_data: dict | None, active_tab: str | None = None) -> pd.DataFrame:
    """The resolutions matching `filter_data`, as `active_tab` should see them.

    `active_tab` matters because two filters are tab-dependent: the country participation filter
    does not apply on tabs where country1 is highlight-only, and keyword search only applies to
    the tabs that list resolutions. Passing the *active* tab (rather than the calling tab's own
    id) keeps every tab showing the same set, which is what the shared store did.

    The filtering itself is `un_data_stream.analysis.filtering`, shared with the API.
    """
    if not filter_data:
        return pd.DataFrame(columns=COLUMNS)

    return filtering.filter_resolutions(
        data.query_engine,
        filtering.ResolutionFilter.from_filter_store(filter_data),
        membership_mask=data.membership_mask,
        apply_country=active_tab not in TABS_WITHOUT_COUNTRY_FILTER,
        apply_keyword=active_tab in TABS_WITH_KEYWORD,
    )
