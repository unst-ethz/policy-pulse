"""
Security Council resolution data processor.

This module handles processing SC resolution data from country-per-row to 
resolution-per-row format.
"""

import logging
import pandas as pd
from typing import Dict

from ..core.abstractions import DatasetProcessor


class SCResolutionProcessor(DatasetProcessor):
    """Processes Security Council resolution data."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger
      
    def process(self, raw_data: pd.DataFrame, **kwargs) -> Dict[str, pd.DataFrame]:
        """Transform SC data from country-per-row to resolution-per-row format."""
        self.logger.info("Processing SC resolution data")

        # SC-specific index columns
        sc_index_cols = [
            "undl_id", "date", "resolution", "draft", "meeting", "description", 
            "agenda", "subjects", "modality", "total_yes", "total_no", 
            "total_abstentions", "total_non_voting", "total_ms", "undl_link"
        ]

        # Transform to resolution-per-row format
        transformed_df = raw_data.pivot(
            index=sc_index_cols,
            columns='ms_code',
            values='ms_vote'
        ).reset_index()

        transformed_df.columns.name = None

        self.logger.info(f"Processed {len(transformed_df)} SC resolutions")
        return {'sc_resolutions': transformed_df}
    
    def get_dataset_type(self) -> str:
        return 'sc_resolutions'