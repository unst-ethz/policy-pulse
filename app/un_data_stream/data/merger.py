"""
Data merger for combining multiple resolution datasets.

This module handles merging multiple resolution datasets into unified formats
with consistent schemas across different data sources.
"""

import logging
import pandas as pd
from typing import Dict


class DataMerger:
    """Handles merging multiple resolution datasets into unified formats."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def _get_unified_schema(self) -> Dict[str, str]:
        """Define the unified schema for the merged resolutions table."""
        return {
            # Core identifier fields (present in all datasets)
            'undl_id': 'str',
            'date': 'datetime',
            'resolution': 'str',
            'draft': 'str',
            'meeting': 'str',
            'subjects': 'str',
            'undl_link': 'str',
            
            # Voting summary fields (present in all datasets)
            'total_yes': 'int',
            'total_no': 'int', 
            'total_abstentions': 'int',
            'total_non_voting': 'int',
            'total_ms': 'int',
            
            # Dataset source identification
            'source_dataset': 'str',  # 'GA', 'SC', 'HRC', etc.
            
            # Content fields (may vary by dataset, nullable)
            'title': 'str',           # GA has this, SC uses 'description'
            'agenda_title': 'str',    # GA specific
            'agenda': 'str',          # SC specific  
            'session': 'str',         # GA specific
            'committee_report': 'str', # GA specific
            'modality': 'str',        # SC specific (voting type)
            'description': 'str'      # SC specific
        }
    
    def _normalize_to_unified_schema(self, df: pd.DataFrame, dataset_type: str) -> pd.DataFrame:
        """Normalize a dataset-specific DataFrame to the unified schema."""
        self.logger.info(f"Normalizing {dataset_type} to unified schema")
        
        unified_df = pd.DataFrame()
        schema = self._get_unified_schema()
        
        # Add source dataset identifier
        unified_df['source_dataset'] = dataset_type.upper().replace('_RESOLUTIONS', '')
        
        # Map common fields directly
        common_fields = ['undl_id', 'date', 'resolution', 'draft', 'meeting', 
                        'subjects', 'undl_link', 'total_yes', 'total_no', 
                        'total_abstentions', 'total_non_voting', 'total_ms']
        
        for field in common_fields:
            if field in df.columns:
                unified_df[field] = df[field]
            else:
                self.logger.warning(f"Missing expected field '{field}' in {dataset_type}")
                unified_df[field] = None
        
        # Handle dataset-specific field mappings
        if dataset_type == 'ga_resolutions':
            # GA-specific fields
            unified_df['title'] = df.get('title')
            unified_df['agenda_title'] = df.get('agenda_title') 
            unified_df['session'] = df.get('session')
            unified_df['committee_report'] = df.get('committee_report')
            # Set SC/HRC fields to None
            unified_df['description'] = None
            unified_df['agenda'] = None
            unified_df['modality'] = None
            
        elif dataset_type == 'sc_resolutions':
            # SC-specific fields
            unified_df['description'] = df.get('description')
            unified_df['agenda'] = df.get('agenda')
            unified_df['modality'] = df.get('modality')
            # Map SC description to title for consistency
            unified_df['title'] = df.get('description')
            # Set GA fields to None
            unified_df['agenda_title'] = None
            unified_df['session'] = None
            unified_df['committee_report'] = None
        
        # Copy voting columns (country votes)
        voting_columns = [col for col in df.columns if col not in schema.keys()]
        for col in voting_columns:
            unified_df[col] = df[col]
        
        self.logger.info(f"Normalized {len(unified_df)} {dataset_type} records")
        return unified_df
    
    def merge_resolutions(self, processed_datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Merge multiple resolution datasets into a unified table."""
        self.logger.info("Merging resolution datasets into unified table")
        
        unified_datasets = []
        
        for dataset_type, df in processed_datasets.items():
            if dataset_type.endswith('_resolutions'):
                normalized_df = self._normalize_to_unified_schema(df, dataset_type)
                unified_datasets.append(normalized_df)
                self.logger.info(f"Added {len(normalized_df)} records from {dataset_type}")
        
        if not unified_datasets:
            self.logger.warning("No resolution datasets found to merge")
            return pd.DataFrame()
        
        # Concatenate all datasets
        merged_df = pd.concat(unified_datasets, ignore_index=True, sort=False)
        
        # Sort by date and source for consistent ordering
        merged_df = merged_df.sort_values(['date', 'source_dataset', 'undl_id'])
        
        self.logger.info(f"Successfully merged {len(merged_df)} total resolutions from {len(unified_datasets)} datasets")
        return merged_df