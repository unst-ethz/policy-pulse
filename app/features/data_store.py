"""Shared (de)serialisation for the filtered-resolutions `dcc.Store`.

`filters.py` publishes the filtered resolution table into a single store that every tab reads
(map, timeline, subject breakdown, multilateral scatter, word cloud, resolution list). Both ends
go through here so the round-trip can't change a dtype the query engine cares about.
"""

from io import StringIO

import pandas as pd

# The join key between the store and everything else. `undl_id` is TEXT in Postgres, and
# ResolutionQueryEngine indexes its precomputed arrays by that exact string
# (`resolution_ids` are matched with `id in self._row_index`).
RESOLUTION_ID = "undl_id"


def dump_resolutions(df: pd.DataFrame) -> str:
    """Serialise the filtered resolution table for the store."""
    return df.to_json(date_format="iso", orient="split")


def load_resolutions(payload: str | None) -> pd.DataFrame:
    """Read the store payload back, keeping `undl_id` a string.

    undl_ids are all digits ('725817'), so pandas' JSON reader infers int64 for the column and
    the ids silently stop matching the query engine's string keys — every id is dropped as
    "unknown", which surfaces as an empty chart or a bare KeyError rather than as an error about
    types. Coercing back on read keeps that invariant in one place instead of in each tab.
    """
    if not payload:
        return pd.DataFrame()
    df = pd.read_json(StringIO(payload), orient="split")
    if RESOLUTION_ID in df.columns:
        df[RESOLUTION_ID] = df[RESOLUTION_ID].astype(str)
    return df
