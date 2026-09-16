"""
GA subject-string matching.

Matches a resolution's raw subject strings (MARC `991.d`, flattened into the `subjects` column)
against the UNBIS thesaurus, producing resolution -> subject_id pairs.

INTERIM: this is the app's last piece of source-data processing and belongs in the ingestion job,
which sees the raw repeated `991.d` fields rather than the flattened column, and can log unmatched
strings as a QA finding. See T11 in plans/app_postgres_migration_plan.md — when that lands, this
module goes away.

This class used to also pivot the bulk CSV export from country-per-row to resolution-per-row
(`process()`), and to declare a dataset type for a processor registry. Both went with the export
and the registry; only the matching grammar below is still used.
"""

import logging

import pandas as pd


class GAResolutionProcessor:
    """Matches GA resolution subject strings to thesaurus subject ids."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def _parse_subjects(self, df: pd.DataFrame, subject_table: pd.DataFrame) -> pd.DataFrame:
        """
        Parse the 'subjects' column in the DataFrame to create a separate row for each subject.
        Uses GA-specific parsing grammar.

        Args:
            df (pd.DataFrame): DataFrame containing a 'subjects' column with special parsing grammar.
            subject_table (pd.DataFrame): Subject table for matching subject IDs.

        Returns:
            pd.DataFrame: DataFrame with each subject in a separate row and subject_id mapping.

        Notes:
            The parsing grammar includes splitting at | and --
        """
        self.logger.info("Parsing GA subjects using | and -- delimiters")

        # Store original subject count for logging
        original_subject_count = len(df["subjects"].unique()) if "subjects" in df.columns else 0

        # First we split the subjects by | and explode the list into separate rows
        df_expanded = df.assign(subjects=df["subjects"].str.split("|")).explode("subjects")

        # For now we just take the first element if --
        df_expanded = df_expanded.assign(
            subjects=df_expanded["subjects"].str.split("--").str[0]
        ).explode("subjects")

        # Clean up subjects (strip whitespace)
        df_expanded["subjects"] = df_expanded["subjects"].str.strip()

        # Remove empty subjects
        # df_expanded = df_expanded[df_expanded['subjects'].notna() & (df_expanded['subjects'] != '')]

        final_subject_count = (
            len(df_expanded["subjects"].unique()) if "subjects" in df_expanded.columns else 0
        )
        self.logger.info(
            f"Expanded subjects from {original_subject_count} to {final_subject_count} unique subjects."
        )

        # Add subject IDs by matching with subject_table
        df_with_ids = self._add_subject_ids(df_expanded, subject_table)

        return df_with_ids

    def _add_subject_ids(
        self, resolution_df: pd.DataFrame, subjects_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Add subject_id column to resolutions dataframe by matching subject strings.

        Args:
            resolution_df: DataFrame with resolutions, already parsed to one row per resolution-subject pair
            subjects_df: DataFrame with subject_id and multilingual labels

        Returns:
            pd.DataFrame: Original resolution_df with added 'subject_id' column
        """
        self.logger.info("Matching GA subjects to subject IDs")

        # Build matching index from subjects_df
        matching_index = {}

        for _, row in subjects_df.iterrows():
            subject_id = row["subject_id"]

            # Index all labels
            for col in subjects_df.columns:
                if col.startswith("label_") and pd.notna(row[col]):
                    label_lower = str(row[col]).lower().strip()
                    matching_index[label_lower] = subject_id

                # Also index alternative labels if they exist
                if col.startswith("alt_labels_") and pd.notna(row[col]) and str(row[col]).strip():
                    for alt_label in str(row[col]).split(";"):
                        alt_label_lower = alt_label.lower().strip()
                        if alt_label_lower:
                            matching_index[alt_label_lower] = subject_id

        # Map subjects to subject_ids
        def map_subject(subject_string):
            if pd.isna(subject_string):
                return None
            subject_clean = str(subject_string).lower().strip()
            return matching_index.get(subject_clean, None)

        # Add the subject_id column
        resolution_df["subject_id"] = resolution_df["subjects"].apply(map_subject)

        # Report statistics
        total_rows = len(resolution_df)
        matched_rows = resolution_df["subject_id"].notna().sum()
        unmatched_rows = resolution_df["subject_id"].isna().sum()

        self.logger.info("GA Subject Mapping Results:")
        self.logger.info(f"  Total rows: {total_rows}")
        self.logger.info(f"  Matched: {matched_rows} ({matched_rows / total_rows * 100:.1f}%)")
        self.logger.info(
            f"  Unmatched: {unmatched_rows} ({unmatched_rows / total_rows * 100:.1f}%)"
        )

        if unmatched_rows > 0:
            self.logger.info("Sample unmatched GA subjects:")
            unmatched_samples = (
                resolution_df[resolution_df["subject_id"].isna()]["subjects"]
                .drop_duplicates()
                .head(5)
            )
            for subject in unmatched_samples:
                self.logger.info(f"  - '{subject}'")

        return resolution_df
