"""
Vote-agreement precomputation.

Turns the wide resolution table into the compact arrays every agreement/alignment query
broadcasts over: a per-resolution consensus score, a (resolutions x countries) multilateral
alignment matrix, and four boolean vote-type arrays.

This module used to also orchestrate per-dataset fetch/processing via a registry of
`DatasetProcessor` implementations. That work moved to the `undl-ingest` repo, which writes the
already-normalized tables the app now reads.
"""

import logging
import time
import warnings
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from app.un_data_stream.data.progress import progressbar


class DataProcessor:
    """Precomputes the vote-agreement arrays the query engine broadcasts over."""

    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self.config = config
        self.logger = logger

    def calculate_agreement_data(
        self, resolutions_df: pd.DataFrame, country_columns: List[str]
    ) -> Tuple[Dict[str, float], np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """
        For each resolution, compute the consensus score (scalar) and per-country
        multilateral alignment scores.  Also build four compact bool arrays encoding
        each country's raw vote per resolution.

        A per-resolution (C x C) agreement matrix is constructed transiently to derive
        consensus and multilateral scores, but is never stored or returned.

        Row order in multilateral_scores and the bool arrays matches resolutions_df — the
        same ordering that resolution_table preserves — so one row-index dict covers all.

        Args:
            resolutions_df : pd.DataFrame
                DataFrame with one row per resolution, containing voting columns
                for each member state and metadata columns
            country_columns : List[str]
                The vote columns to treat as countries, in the order the returned arrays should
                be indexed by. Passed in rather than inferred: the caller knows this exactly
                (from `resolution_votes.country_code`), and the returned arrays are indexed
                positionally by it, so inferring it here would silently misattribute every score
                if the frame ever gained an unrecognised metadata column.

        Returns:
            Tuple of:
                - consensus_scores: Dict mapping undl_id to its consensus score (scalar).
                - multilateral_scores: (R x C) float32 array — per-resolution per-country
                  row-mean alignment score (NaN where country did not vote).
                - vote_bool_arrays: 4-tuple of (R x C) bool arrays (yes, no, abstained, voted)
                  in the same row order as resolutions_df / multilateral_scores.
        """
        self.logger.info("Starting consensus and multilateral scores calculation")
        start_time = time.time()

        self.logger.info(f"Using {len(country_columns)} country columns")
        self.logger.info(f"Processing {len(resolutions_df)} resolutions")

        n = len(country_columns)
        off_diag_mask = ~np.eye(n, dtype=bool)

        # For each resolution, compute consensus score and multilateral scores. The agreement
        # matrix is built transiently per resolution and not retained.
        consensus_scores = {}
        multilateral_rows = []

        for idx, row in progressbar(resolutions_df.iterrows(), total=len(resolutions_df)):
            undl_id = row["undl_id"]
            agreement_matrix = self._calculate_single_resolution_matrix(row, country_columns)
            consensus_scores[undl_id] = self._calculate_single_consensus_score(agreement_matrix)

            # Per-country multilateral score: mean agreement with every *other* voting country.
            # Diagonal is masked to nan so self-agreement (always 1.0) does not inflate the mean.
            mat_no_diag = np.where(off_diag_mask, agreement_matrix, np.nan)  # (C, C)
            num_valid = np.sum(~np.isnan(mat_no_diag), axis=1)  # (C,) — voting partners per country
            row_means = np.full(n, np.nan)  # (C,) — default nan for non-voters
            row_agreement_sums = np.nansum(mat_no_diag, axis=1)  # (C,)
            # out= / where= writes results directly into row_means, leaving nan where num_valid == 0
            np.divide(row_agreement_sums, num_valid, out=row_means, where=num_valid > 0)
            multilateral_rows.append(row_means)

        multilateral_scores = np.array(
            multilateral_rows, dtype=np.float32
        )  # float32 saves disk space when pickling

        # Boolean arrays with vote-type indicators, so the query engine never has to look at a
        # vote column again.
        vote_yes, vote_no, vote_abstained, vote_voted = self._vote_type_masks(
            resolutions_df, country_columns
        )

        elapsed_time = time.time() - start_time
        n_res = len(consensus_scores)
        assert n_res == len(resolutions_df)
        self.logger.info(
            f"Calculated {n_res} consensus scores "
            f"and multilateral_scores ({multilateral_scores.shape}) in {elapsed_time:.2f}s"
        )

        return (
            consensus_scores,
            multilateral_scores,
            (vote_yes, vote_no, vote_abstained, vote_voted),
        )

    @staticmethod
    def _vote_type_masks(
        resolutions_df: pd.DataFrame, country_columns: List[str]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Build the (yes, no, abstained, voted) boolean arrays, one column at a time.

        Deliberately avoids `resolutions_df[country_columns].astype(str)`. On the real dataset
        that materialises 20,832 x 202 ~= 4.2 million Python strings, and the `.apply(...)` that
        followed built another full object-dtype copy per column — together the largest transient
        allocation in a load, and the reason a reloading worker's RSS plateaued a few hundred MB
        above its baseline.

        Vote columns arrive as `CategoricalDtype` with categories Y/N/A/X (see
        `DataRepository.VOTE_DTYPE`), so the comparison runs on the integer codes and only ever
        normalises the handful of category labels. Object columns — what the unit tests pass, and
        historically tolerant of stray case or whitespace — take a per-column string comparison,
        so at most one column is expanded at a time instead of the whole frame.
        """
        n_rows = len(resolutions_df)
        n_cols = len(country_columns)
        masks = {code: np.zeros((n_rows, n_cols), dtype=bool) for code in ("Y", "N", "A")}

        for position, column in enumerate(country_columns):
            values = resolutions_df[column]
            if isinstance(values.dtype, pd.CategoricalDtype):
                codes = values.cat.codes.to_numpy()
                normalised = [str(category).strip().upper() for category in values.cat.categories]
                for code, mask in masks.items():
                    # More than one category can normalise to the same code (e.g. 'Y' and 'y'),
                    # so match every code that does rather than just the first.
                    matching = [i for i, category in enumerate(normalised) if category == code]
                    if matching:
                        mask[:, position] = np.isin(codes, matching)
            else:
                text = values.astype(str).str.strip().str.upper()
                for code, mask in masks.items():
                    mask[:, position] = (text == code).to_numpy()

        vote_yes, vote_no, vote_abstained = masks["Y"], masks["N"], masks["A"]
        return vote_yes, vote_no, vote_abstained, vote_yes | vote_no | vote_abstained

    @staticmethod
    def _calculate_single_resolution_matrix(
        resolution_row: pd.Series, country_columns: List[str]
    ) -> np.ndarray:
        """
        Quickly calculate the full vote-agreement matrix for a single resolution.

        Uses vectorized calculation via NumPy broadcasting.
        Result is float32 to keep the transient matrix small during the build loop.

        Args:
            resolution_row: Series containing votes for all countries
            country_columns: List of country column names

        Returns:
            np.ndarray: 2D agreement matrix (C x C)
        """
        # 1. Map votes to numeric values
        vote_mapping = {"Y": np.float32(1.0), "A": np.float32(0.0), "N": np.float32(-1.0)}

        # Extract the country votes as a NumPy array (floats to accommodate NaN)
        # Using .get() for safety, defaulting to np.nan
        votes = np.array(
            [vote_mapping.get(resolution_row[c], np.nan) for c in country_columns], dtype=np.float32
        )

        # broadcast (C,1) vs (1,C): all pairwise vote differences in one operation
        abs_diff_mat = np.abs(votes[:, np.newaxis] - votes[np.newaxis, :])  # (C, C)

        # agreement formula: scores ∈ {0.0, 0.5, 1.0, nan}
        agreement_matrix = (1.0 - (abs_diff_mat / 2.0)).astype(np.float32)  # (C, C)

        return agreement_matrix

    @staticmethod
    def _calculate_single_consensus_score(agreement_matrix: np.ndarray) -> float:
        """
        Calculate the consensus score for a given resolution, based on the
        resolution's full vote-agreement matrix.

        The consensus score is simply the average vote-agreement score across
        all country pairs where both sides voted on the resolution at hand.

        Args:
            agreement_matrix: np.ndarray: 2D agreement matrix (C x C)

        Returns:
            float
        """

        # The matrix is symmetric, so the lower triangle contains every unique country pair
        # exactly once. Taking its nanmean gives the average pairwise agreement without
        # double-counting. k=-1 excludes the diagonal (self-agreement, always 1.0).
        n_countries = agreement_matrix.shape[0]
        tril_indices = np.tril_indices(n_countries, k=-1)
        lower_triangle_values = agreement_matrix[tril_indices]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            consensus_score = np.nanmean(lower_triangle_values)

        return float(consensus_score)
