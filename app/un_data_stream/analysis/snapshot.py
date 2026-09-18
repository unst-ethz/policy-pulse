"""The immutable bundle of data a query is answered from.

Every table and precomputed array the query engine needs lives in one frozen object, so replacing
the app's data is a **single reference assignment** (`engine.swap(new_snapshot)`) rather than
eleven separate attribute writes.

That matters because the arrays are positionally coupled: `country_columns` is the column index
into `multilateral_scores` and the vote arrays, and `row_index` is the row index into all of them.
A reader that picked up a new `resolution_table` alongside old arrays would attribute scores to
the wrong countries — silently, with plausible-looking numbers. Keeping them in one object makes
that combination unrepresentable.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DataSnapshot:
    """One coherent view of the data. Treat every field as read-only."""

    resolution_table: pd.DataFrame
    resolution_subject_table: pd.DataFrame
    subject_table: pd.DataFrame
    closure_table: pd.DataFrame
    country_columns: List[str]

    # (resolutions x countries), indexed by row_index / country_columns
    multilateral_scores: Optional[np.ndarray]
    yes: np.ndarray
    no: np.ndarray
    abstained: np.ndarray
    voted: np.ndarray
    row_index: Dict[str, int]

    loaded_at: datetime
    # MAX(completed_at) of successful ingestion runs at the moment this data was read. The
    # reloader compares it against the live marker to decide whether a rebuild is due.
    source_marker: Optional[datetime]

    @classmethod
    def from_repo(cls, repo: Any) -> "DataSnapshot":
        """Build a snapshot from a loaded `DataRepository`."""
        data = repo.get_data()
        resolution_table = data.get("resolution", pd.DataFrame())
        country_columns = data.get("country_columns") or []
        vote_bool_arrays = data.get("vote_bool_arrays")

        if vote_bool_arrays is not None and country_columns:
            yes, no, abstained, voted = vote_bool_arrays
            row_index = {rid: i for i, rid in enumerate(resolution_table["undl_id"].tolist())}
        else:
            # No vote data (an empty or partial load): keep the shapes valid so queries return
            # empty results instead of raising.
            empty = np.empty((0, 0), dtype=bool)
            yes = no = abstained = voted = empty
            row_index = {}

        return cls(
            resolution_table=resolution_table,
            resolution_subject_table=data.get("resolution_subject", pd.DataFrame()),
            subject_table=data.get("subject", pd.DataFrame()),
            closure_table=data.get("closure", pd.DataFrame()),
            country_columns=country_columns,
            multilateral_scores=data.get("multilateral_scores"),
            yes=yes,
            no=no,
            abstained=abstained,
            voted=voted,
            row_index=row_index,
            loaded_at=datetime.now(timezone.utc),
            source_marker=getattr(repo, "source_marker", None),
        )

    def describe(self) -> str:
        """One-line summary for logs."""
        marker = self.source_marker.isoformat() if self.source_marker else "unknown"
        return (
            f"{len(self.resolution_table)} resolutions, {len(self.country_columns)} countries, "
            f"ingested up to {marker}"
        )


def empty_snapshot() -> "DataSnapshot":
    """An all-empty snapshot, for tests and for a failed first load."""
    empty = np.empty((0, 0), dtype=bool)
    # Column names match what a real (if empty) Postgres read returns, so callers see the same
    # shape they would in production rather than a columnless frame.
    return DataSnapshot(
        resolution_table=pd.DataFrame(columns=["undl_id", "date"]),
        resolution_subject_table=pd.DataFrame(columns=["undl_id", "subject_id"]),
        subject_table=pd.DataFrame(columns=["subject_id", "label_en", "node_type"]),
        closure_table=pd.DataFrame(columns=["ancestor_id", "descendant_id", "depth"]),
        country_columns=[],
        multilateral_scores=None,
        yes=empty,
        no=empty,
        abstained=empty,
        voted=empty,
        row_index={},
        loaded_at=datetime.now(timezone.utc),
        source_marker=None,
    )
