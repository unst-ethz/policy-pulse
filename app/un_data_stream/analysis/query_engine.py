"""
Query engine for resolution data analysis.

This module provides querying capabilities for UN resolution data,
including subject-based and date filtering, bilateral vote agreement,
lookups and multilateral alignment statistics.
"""

import warnings
from typing import List, Optional

import numpy as np
import pandas as pd

from ..data import DataRepository
from .snapshot import DataSnapshot


class ResolutionQueryEngine:
    """Advanced query engine for resolution data analysis."""

    def __init__(self, repo: DataRepository):
        """
        Initialize query engine with processed data.

        Args:
            repo: DataRepository instance providing all precomputed tables and arrays.
        """
        self.logger = repo.logger
        self._snap = DataSnapshot.from_repo(repo)

    # ------------------------------------------------------------------
    # Data handle
    #
    # The engine instance is stable for the life of the process: `app/data.py` publishes it once
    # and three features close over it at callback-registration time
    # (trends_page.py -> agreement_choropleth / agreement_by_subject / multilateral_scatter).
    # Rebinding a *new* engine would leave those three serving stale data forever while every
    # other tab moved on, so a reload replaces the snapshot inside this instance instead.
    # ------------------------------------------------------------------

    @property
    def snapshot(self) -> DataSnapshot:
        """The data currently being served."""
        return self._snap

    def swap(self, snapshot: DataSnapshot) -> DataSnapshot:
        """Atomically replace the served data, returning the snapshot that was in place.

        One assignment, so a reader either sees the whole old snapshot or the whole new one.
        Queries in flight keep the snapshot they started with — they took their reference at
        method entry — and finish against consistent data.
        """
        previous, self._snap = self._snap, snapshot
        self.logger.info(f"Swapped in new data: {snapshot.describe()}")
        return previous

    # Read-only views onto the current snapshot, kept under their historical names because
    # features read them directly (resolution_list.py, recent_resolutions_panel.py,
    # wordcloud_interactive.py, general_stats_panel.py) and so do the tests.

    @property
    def resolution_table(self) -> pd.DataFrame:
        return self._snap.resolution_table

    @property
    def resolution_subject_table(self) -> pd.DataFrame:
        return self._snap.resolution_subject_table

    @property
    def subject_table(self) -> pd.DataFrame:
        return self._snap.subject_table

    @property
    def closure_table(self) -> pd.DataFrame:
        return self._snap.closure_table

    @property
    def country_columns(self) -> List[str]:
        return self._snap.country_columns

    def query_resolutions(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        subject_ids: Optional[List[str]] = None,
        language: str = "en",
        include_descendants: bool = True,
    ) -> pd.DataFrame:
        """
        Query resolutions based on date range and subject filters.

        Args:
            start_date: Start date in format 'YYYY-MM-DD' (None = from beginning)
            end_date: End date in format 'YYYY-MM-DD' (None = until today)
            subject_ids: List of subject URIs to filter by (None = all subjects)
            include_descendants: If True, include all descendants of specified subjects

        Returns:
            pd.DataFrame: Filtered resolutions with all metadata
        """
        # One read of the snapshot reference: everything below is answered from this
        # single coherent view, even if a reload swaps the engine's data mid-query.
        snap = self._snap

        # Start with all resolutions
        filtered_df = snap.resolution_table.copy()
        if filtered_df.empty:
            # Nothing to filter. Also guards a degenerate snapshot (no rows *and* no columns),
            # where reaching for "date" below would raise instead of returning nothing.
            self.logger.info("No resolutions loaded")
            return filtered_df

        filtered_df["date"] = pd.to_datetime(filtered_df["date"])

        # 1. Apply date filters
        if start_date:
            filtered_df = filtered_df[filtered_df["date"] >= pd.to_datetime(start_date)]
        if end_date:
            filtered_df = filtered_df[filtered_df["date"] <= pd.to_datetime(end_date)]

        # 2. Apply subject filters
        if subject_ids is not None and len(subject_ids) > 0:
            # Separate the synthetic "no subject" sentinel from real subject IDs
            include_no_subject = "__no_subject__" in subject_ids
            real_subject_ids = [
                s for s in subject_ids if s != "__no_subject__" and s != "__all_subjects__"
            ]

            matching_ids = set()

            if include_no_subject:
                # Resolutions that have no entry in the subject table at all
                all_with_subject = set(snap.resolution_subject_table["undl_id"].unique())
                no_subject_ids = set(filtered_df["undl_id"].unique()) - all_with_subject
                matching_ids.update(no_subject_ids)
                self.logger.info(f"No-subject resolutions: {len(no_subject_ids)}")

            if real_subject_ids:
                if include_descendants:
                    expanded_subjects = set(real_subject_ids)
                    for subject_id in real_subject_ids:
                        descendants = snap.closure_table[
                            snap.closure_table["ancestor_id"] == subject_id
                        ]["descendant_id"].unique()
                        expanded_subjects.update(descendants)
                    self.logger.info(
                        f"Expanded {len(real_subject_ids)} subjects to {len(expanded_subjects)} (including descendants)"
                    )
                    subject_filter = list(expanded_subjects)
                else:
                    subject_filter = real_subject_ids

                subject_resolution_ids = snap.resolution_subject_table[
                    snap.resolution_subject_table["subject_id"].isin(subject_filter)
                ]["undl_id"].unique()
                matching_ids.update(subject_resolution_ids)

            filtered_df = filtered_df[filtered_df["undl_id"].isin(matching_ids)]
            self.logger.info(f"After subject filter: {len(filtered_df)} resolutions")

        self.logger.info(f"Final result: {len(filtered_df)} resolutions")
        return filtered_df

    def query_agreement_between_countries(
        self, country_code: str, resolution_ids: Optional[List[str]] = None, average: bool = False
    ) -> pd.DataFrame:
        """
        Get bilateral agreement scores between a selected country and all other countries.

        Scores are computed on demand from the precomputed vote_bool_arrays via a single
        vectorised NumPy broadcast — no (C x C) matrices are stored or iterated.

        Args:
            country_code: ISO3 country code to query
            resolution_ids: Resolution IDs to include (None = all resolutions)
            average: If True, return mean scores across all resolutions;
                     if False, return per-resolution scores

        Returns:
            pd.DataFrame in wide format — one column per country (ISO3 code), excluding
            country_code itself:
                - If average=False: one row per resolution; columns ['undl_id', <iso3>, ...]
                  where each cell is the bilateral agreement score between country_code and
                  that country on that resolution.
                - If average=True: a single row; columns ['source_country',
                  'resolution_count', <iso3>, ...] where each cell is the mean bilateral
                  agreement score across all selected resolutions.
        """
        # One read of the snapshot reference: everything below is answered from this
        # single coherent view, even if a reload swaps the engine's data mid-query.
        snap = self._snap
        if not snap.row_index:
            self.logger.error("No vote data available")
            return pd.DataFrame()

        if country_code not in snap.country_columns:
            self.logger.error(f"Country '{country_code}' not found in country columns")
            self.logger.info(f"Available countries (first 10): {snap.country_columns[:10]}")
            return pd.DataFrame()

        country_idx = snap.country_columns.index(country_code)

        if resolution_ids is None:
            rid_list = snap.resolution_table["undl_id"].tolist()
            rows = list(range(len(rid_list)))
        else:
            pairs = [(r, snap.row_index[r]) for r in resolution_ids if r in snap.row_index]
            missing = len(resolution_ids) - len(pairs)
            if missing:
                self.logger.warning(f"Missing vote data for {missing} resolution IDs")
            if not pairs:
                self.logger.warning("No valid resolutions found for agreement analysis")
                return pd.DataFrame()
            rid_list, rows = map(list, zip(*pairs))

        self.logger.info(f"Analyzing agreement for '{country_code}' across {len(rows)} resolutions")

        # Reconstruct numeric vote indicators by subtracting bool arrays for "yes" - "no":
        # True - False = 1 (yes),
        # False - True = -1 (no),
        # False - False = 0 (abstained);
        # non-voters masked to nan below.
        yes_float = snap.yes[rows].astype(np.float32)
        no_float = snap.no[rows].astype(np.float32)
        v = yes_float - no_float  # (R', C) — all countries' votes
        v[~snap.voted[rows]] = np.nan  # X / missing → excluded from scoring

        v_c = v[:, country_idx]  # (R',) — reference country's votes
        # broadcast (R',1) vs (R',C): one subtraction covers all resolutions × all countries at once
        agree = (
            1.0 - np.abs(v_c[:, np.newaxis] - v) / 2.0
        )  # (R', C) — agreement scores ∈ {0.0, 0.5, 1.0, nan}
        agree[:, country_idx] = np.nan  # mask self-comparison

        other_idx = [i for i in range(len(snap.country_columns)) if i != country_idx]
        other_cols = [snap.country_columns[i] for i in other_idx]

        if average:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                avg = np.nanmean(
                    agree[:, other_idx], axis=0
                )  # (C-1,); NaN for non-participating countries
            avg_df = pd.DataFrame([avg], columns=other_cols)
            avg_df.insert(0, "resolution_count", len(rows))
            avg_df.insert(0, "source_country", country_code)
            self.logger.info(f"Calculated average agreements across {len(rows)} resolutions")
            return avg_df
        else:
            scores_df = pd.DataFrame(agree[:, other_idx], columns=other_cols)
            scores_df.insert(0, "undl_id", rid_list)
            self.logger.info(f"Retrieved agreement scores for {len(scores_df)} resolutions")
            return scores_df.sort_values("undl_id").reset_index(drop=True)

    def query_multilateral_stats(self, resolution_ids: Optional[List[str]] = None) -> pd.DataFrame:
        """
        For each country, compute multilateral alignment and voting rate statistics
        across the given resolutions, based on the precomputed (R x C) arrays.

        Args:
            resolution_ids: List of undl_ids to include. If None or empty, uses all
                resolutions in the dataset.

        Returns:
            pd.DataFrame with one row per country and columns:
                - country: ISO3 country code
                - multilateral_alignment: mean pairwise agreement with all other voting
                  countries, averaged across selected resolutions (NaN if no participation)
                - yes_rate: fraction of votes cast as Yes (NaN if no votes)
                - no_rate: fraction of votes cast as No (NaN if no votes)
                - abstention_rate: fraction of votes cast as abstentions (NaN if no votes)
                - participation_count: number of selected resolutions the country voted on
        """
        # One read of the snapshot reference: everything below is answered from this
        # single coherent view, even if a reload swaps the engine's data mid-query.
        snap = self._snap
        if snap.multilateral_scores is None or not snap.country_columns:
            return pd.DataFrame()

        if resolution_ids is None or len(resolution_ids) == 0:
            rows = list(snap.row_index.values())
        else:
            rows = [snap.row_index[r] for r in resolution_ids if r in snap.row_index]

        if not rows:
            return pd.DataFrame()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            avg_alignment = np.nanmean(snap.multilateral_scores[rows], axis=0)  # (C,)

        voted_slice = snap.voted[rows]  # (R', C) bool
        abstained_slice = snap.abstained[rows]  # (R', C) bool
        yes_slice = snap.yes[rows]  # (R', C) bool
        no_slice = snap.no[rows]  # (R', C) bool
        participation = voted_slice.sum(axis=0)  # (C,) int
        abstentions = abstained_slice.sum(axis=0)  # (C,) int
        yes_votes = yes_slice.sum(axis=0)  # (C,) int
        no_votes = no_slice.sum(axis=0)  # (C,) int

        # out= / where= writes results into pre-filled nan arrays, leaving nan
        # where participation == 0, without triggering divide-by-zero warnings
        voted = participation > 0
        abstention_rate = np.full(len(participation), np.nan)
        yes_rate = np.full(len(participation), np.nan)
        no_rate = np.full(len(participation), np.nan)
        np.divide(abstentions, participation, out=abstention_rate, where=voted)
        np.divide(yes_votes, participation, out=yes_rate, where=voted)
        np.divide(no_votes, participation, out=no_rate, where=voted)
        # TODO: Add "Not-Voting Share" (fraction of selected resolutions with no vote cast).
        #  Crucial: This would need to reflect countries' membership dates in the UN to be meaningful.

        return pd.DataFrame(
            {
                "country": snap.country_columns,
                "multilateral_alignment": avg_alignment,
                "abstention_rate": abstention_rate,
                "yes_rate": yes_rate,
                "no_rate": no_rate,
                "participation_count": participation,
            }
        )

    def get_available_countries(self) -> List[str]:
        """Get list of available country codes in the dataset."""
        # One read of the snapshot reference: everything below is answered from this
        # single coherent view, even if a reload swaps the engine's data mid-query.
        snap = self._snap
        return snap.country_columns
