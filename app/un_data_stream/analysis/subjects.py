"""Existing subject agreement calculation, shared by both frontends."""

import pandas as pd
import numpy as np

MIN_VOTES_THRESHOLD = 30


def calculate_agreement(query_engine, c1, c2, start_date, end_date, subject_list, subject_map):
    """
    Calculates agreement scores between two countries across subjects.

    Optimized to query agreement once and then filter by subject.

    Agreement Score ranges from 0 (complete disagreement) to 1 (complete agreement)

    Returns a DataFrame with columns: subject_id, subject_label, agreement_score, total_votes
    """
    # 1. Get all resolutions in the date range
    all_resolutions = query_engine.query_resolutions(start_date=start_date, end_date=end_date)

    if all_resolutions.empty:
        return pd.DataFrame()

    agreement_df = query_engine.query_agreement_between_countries(
        country_code=c1, resolution_ids=all_resolutions["undl_id"].tolist(), average=False
    )

    if agreement_df.empty or c2 not in agreement_df.columns:
        return pd.DataFrame()

    # 3. Keep only the columns we need: undl_id and the target country's agreement
    agreement_df = agreement_df[["undl_id", c2]].copy()
    agreement_df.rename(columns={c2: "agreement_score"}, inplace=True)

    # 4. Get resolution-subject mappings
    # We need to know which subjects each resolution belongs to
    resolution_subject_df = query_engine.resolution_subject_table.copy()

    # 5. For each subject, calculate the agreement score
    agreement_results = []

    for subject_id in subject_list:
        # Find all resolutions with this subject (including descendants)
        if hasattr(query_engine, "closure_table"):
            # Get descendants of this subject
            descendants = query_engine.closure_table[
                query_engine.closure_table["ancestor_id"] == subject_id
            ]["descendant_id"].unique()
            subject_filter = list(set([subject_id] + list(descendants)))
        else:
            subject_filter = [subject_id]

        # Get resolution IDs for this subject
        subject_resolution_ids = resolution_subject_df[
            resolution_subject_df["subject_id"].isin(subject_filter)
        ]["undl_id"].unique()

        # Filter agreement scores to only resolutions with this subject
        subject_agreements = agreement_df[
            agreement_df["undl_id"].isin(subject_resolution_ids)
        ].copy()

        # Remove NaN values
        subject_agreements = subject_agreements.dropna(subset=["agreement_score"])

        total_votes = len(subject_agreements)

        # Skip if insufficient votes
        if total_votes < MIN_VOTES_THRESHOLD:
            continue

        if total_votes == 0:
            agreement_results.append(
                {
                    "subject_id": subject_id,
                    "subject_label": subject_map.get(subject_id, f"ID: {subject_id}"),
                    "agreement_score": np.nan,
                    "total_votes": 0,
                }
            )
            continue

        # Calculate agreement score
        # Agreement score is already 0-1 where 1=full agreement, 0=full disagreement
        agreement_score = subject_agreements["agreement_score"].mean()

        # Store results
        agreement_results.append(
            {
                "subject_id": subject_id,
                "subject_label": subject_map.get(subject_id, f"ID: {subject_id}"),
                "agreement_score": agreement_score,
                "total_votes": total_votes,
            }
        )

    return pd.DataFrame(agreement_results)
