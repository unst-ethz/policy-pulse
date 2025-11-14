"""
Resolution analyzer for advanced data analysis and insights.

This module provides comprehensive analysis capabilities for UN resolution data,
including trend analysis, subject analysis, and statistical computations.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from collections import Counter, defaultdict
from datetime import datetime, timedelta


class ResolutionAnalyzer:
    """Advanced analyzer for UN resolution data with statistical and trend analysis."""
    
    def __init__(self, data: Dict[str, pd.DataFrame]):
        """
        Initialize analyzer with processed data.
        
        Args:
            data: Dictionary containing processed DataFrames
        """
        self.resolution_table = data.get('resolution', pd.DataFrame())
        self.resolution_subject_table = data.get('resolution_subject', pd.DataFrame())
        self.subject_table = data.get('subject', pd.DataFrame())
        self.closure_table = data.get('closure', pd.DataFrame())

    def analyze_temporal_trends(self, subject_label: Optional[str] = None, 
                              time_period: str = 'year') -> pd.DataFrame:
        """
        Analyze temporal trends in resolution adoption.
        
        Args:
            subject_label: Optional subject to filter by
            time_period: Grouping period ('year', 'month', 'quarter')
            
        Returns:
            DataFrame with temporal trend analysis
        """
        # Get base dataset
        if subject_label:
            resolutions = self._get_resolutions_by_subject(subject_label)
        else:
            resolutions = self.resolution_table.copy()
        
        if resolutions.empty:
            return pd.DataFrame()
        
        # Find date column
        date_col = self._find_date_column(resolutions)
        if not date_col:
            return pd.DataFrame()
        
        # Convert to datetime
        resolutions['datetime'] = pd.to_datetime(resolutions[date_col], errors='coerce')
        resolutions = resolutions.dropna(subset=['datetime'])
        
        if resolutions.empty:
            return pd.DataFrame()
        
        # Group by time period
        if time_period == 'year':
            resolutions['period'] = resolutions['datetime'].dt.year
        elif time_period == 'month':
            resolutions['period'] = resolutions['datetime'].dt.to_period('M')
        elif time_period == 'quarter':
            resolutions['period'] = resolutions['datetime'].dt.to_period('Q')
        else:
            raise ValueError("time_period must be 'year', 'month', or 'quarter'")
        
        # Calculate trends
        trends = resolutions.groupby('period').agg({
            'resolution_id': 'count',
            'datetime': ['min', 'max']
        }).round(2)
        
        trends.columns = ['resolution_count', 'period_start', 'period_end']
        trends = trends.reset_index()
        
        return trends

    def analyze_subject_popularity(self, top_n: int = 20) -> pd.DataFrame:
        """
        Analyze the popularity of subjects across resolutions.
        
        Args:
            top_n: Number of top subjects to return
            
        Returns:
            DataFrame with subject popularity analysis
        """
        if self.resolution_subject_table.empty or self.subject_table.empty:
            return pd.DataFrame()
        
        # Count subject usage
        subject_counts = self.resolution_subject_table['subject_id'].value_counts()
        
        # Get subject details
        popular_subjects = self.subject_table[
            self.subject_table['subject_id'].isin(subject_counts.index[:top_n])
        ].copy()
        
        # Add usage statistics
        popular_subjects['usage_count'] = popular_subjects['subject_id'].map(subject_counts)
        popular_subjects['usage_percentage'] = (
            popular_subjects['usage_count'] / len(self.resolution_table) * 100
        ).round(2)
        
        return popular_subjects.sort_values('usage_count', ascending=False)

    def analyze_co_occurrence(self, subject_label: str, top_n: int = 10) -> pd.DataFrame:
        """
        Analyze which subjects frequently co-occur with a given subject.
        
        Args:
            subject_label: The subject to analyze co-occurrences for
            top_n: Number of top co-occurring subjects to return
            
        Returns:
            DataFrame with co-occurrence analysis
        """
        # Find the target subject
        target_subjects = self.subject_table[
            self.subject_table['label'].str.contains(subject_label, case=False, na=False)
        ]
        
        if target_subjects.empty:
            return pd.DataFrame()
        
        target_subject_ids = set(target_subjects['subject_id'])
        
        # Find resolutions with the target subject
        target_resolutions = self.resolution_subject_table[
            self.resolution_subject_table['subject_id'].isin(target_subject_ids)
        ]['resolution_id'].unique()
        
        if len(target_resolutions) == 0:
            return pd.DataFrame()
        
        # Find all subjects in these resolutions
        co_occurring_subjects = self.resolution_subject_table[
            self.resolution_subject_table['resolution_id'].isin(target_resolutions)
        ]
        
        # Remove the target subjects themselves
        co_occurring_subjects = co_occurring_subjects[
            ~co_occurring_subjects['subject_id'].isin(target_subject_ids)
        ]
        
        # Count co-occurrences
        co_occurrence_counts = co_occurring_subjects['subject_id'].value_counts()
        
        if co_occurrence_counts.empty:
            return pd.DataFrame()
        
        # Get subject details
        top_co_occurring = self.subject_table[
            self.subject_table['subject_id'].isin(co_occurrence_counts.index[:top_n])
        ].copy()
        
        # Add co-occurrence statistics
        top_co_occurring['co_occurrence_count'] = top_co_occurring['subject_id'].map(co_occurrence_counts)
        top_co_occurring['co_occurrence_percentage'] = (
            top_co_occurring['co_occurrence_count'] / len(target_resolutions) * 100
        ).round(2)
        
        return top_co_occurring.sort_values('co_occurrence_count', ascending=False)

    def analyze_subject_evolution(self, subject_label: str, 
                                time_period: str = 'year') -> pd.DataFrame:
        """
        Analyze how usage of a specific subject has evolved over time.
        
        Args:
            subject_label: The subject to analyze
            time_period: Time grouping ('year', 'month', 'quarter')
            
        Returns:
            DataFrame with subject evolution analysis
        """
        return self.analyze_temporal_trends(subject_label, time_period)

    def get_resolution_complexity(self) -> pd.DataFrame:
        """
        Analyze the complexity of resolutions based on number of subjects.
        
        Returns:
            DataFrame with complexity analysis
        """
        if self.resolution_subject_table.empty:
            return pd.DataFrame()
        
        # Count subjects per resolution
        complexity = self.resolution_subject_table.groupby('resolution_id')['subject_id'].count()
        complexity_stats = pd.DataFrame({
            'resolution_id': complexity.index,
            'subject_count': complexity.values
        })
        
        # Add resolution details if available
        if not self.resolution_table.empty:
            complexity_stats = complexity_stats.merge(
                self.resolution_table, on='resolution_id', how='left'
            )
        
        return complexity_stats.sort_values('subject_count', ascending=False)

    def detect_trending_subjects(self, recent_period_months: int = 12, 
                               min_resolution_count: int = 5) -> pd.DataFrame:
        """
        Detect subjects that are trending in recent resolutions.
        
        Args:
            recent_period_months: Number of recent months to consider
            min_resolution_count: Minimum resolutions needed to be considered trending
            
        Returns:
            DataFrame with trending subjects analysis
        """
        date_col = self._find_date_column(self.resolution_table)
        if not date_col:
            return pd.DataFrame()
        
        # Calculate cutoff date
        cutoff_date = datetime.now() - timedelta(days=recent_period_months * 30)
        
        # Get recent resolutions
        recent_resolutions = self.resolution_table.copy()
        recent_resolutions['datetime'] = pd.to_datetime(recent_resolutions[date_col], errors='coerce')
        recent_resolutions = recent_resolutions[
            recent_resolutions['datetime'] >= cutoff_date
        ]
        
        if recent_resolutions.empty:
            return pd.DataFrame()
        
        recent_resolution_ids = recent_resolutions['resolution_id'].unique()
        
        # Get subjects from recent resolutions
        recent_subjects = self.resolution_subject_table[
            self.resolution_subject_table['resolution_id'].isin(recent_resolution_ids)
        ]
        
        # Count recent usage
        recent_counts = recent_subjects['subject_id'].value_counts()
        
        # Filter by minimum count
        trending_subject_ids = recent_counts[recent_counts >= min_resolution_count].index
        
        if len(trending_subject_ids) == 0:
            return pd.DataFrame()
        
        # Get subject details
        trending_subjects = self.subject_table[
            self.subject_table['subject_id'].isin(trending_subject_ids)
        ].copy()
        
        # Add trending statistics
        trending_subjects['recent_usage_count'] = trending_subjects['subject_id'].map(recent_counts)
        
        # Calculate overall usage for comparison
        overall_counts = self.resolution_subject_table['subject_id'].value_counts()
        trending_subjects['total_usage_count'] = trending_subjects['subject_id'].map(overall_counts)
        
        # Calculate trend ratio
        trending_subjects['trend_ratio'] = (
            trending_subjects['recent_usage_count'] / 
            trending_subjects['total_usage_count']
        ).round(3)
        
        return trending_subjects.sort_values('trend_ratio', ascending=False)

    def _get_resolutions_by_subject(self, subject_label: str) -> pd.DataFrame:
        """Helper method to get resolutions by subject."""
        matching_subjects = self.subject_table[
            self.subject_table['label'].str.contains(subject_label, case=False, na=False)
        ]
        
        if matching_subjects.empty:
            return pd.DataFrame()
        
        subject_ids = matching_subjects['subject_id'].unique()
        
        matching_res_subjects = self.resolution_subject_table[
            self.resolution_subject_table['subject_id'].isin(subject_ids)
        ]
        
        if matching_res_subjects.empty:
            return pd.DataFrame()
        
        resolution_ids = matching_res_subjects['resolution_id'].unique()
        
        return self.resolution_table[
            self.resolution_table['resolution_id'].isin(resolution_ids)
        ]

    def _find_date_column(self, df: pd.DataFrame) -> Optional[str]:
        """Helper method to find the date column in a DataFrame."""
        for col in ['date', 'Date', 'DATE']:
            if col in df.columns:
                return col
        
        # Look for columns containing 'date'
        date_cols = [col for col in df.columns if 'date' in col.lower()]
        return date_cols[0] if date_cols else None