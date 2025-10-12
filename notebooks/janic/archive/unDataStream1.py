# unDataStream.py
import logging
import sys
import yaml
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, Any, List, Optional, Protocol
from io import BytesIO
import pandas as pd
import requests
from rdflib import Graph, RDF, SKOS

# Abstract base classes for extensibility
class DatasetFetcher(ABC):
    """Abstract base class for dataset-specific fetchers."""
    
    @abstractmethod
    def fetch(self, source_config: Dict[str, Any]) -> pd.DataFrame:
        """Fetch raw data for this dataset type."""
        pass
    
    @abstractmethod
    def get_dataset_type(self) -> str:
        """Return the dataset type identifier (e.g., 'ga_resolutions', 'sc_resolutions')."""
        pass

class DatasetProcessor(ABC):
    """Abstract base class for dataset-specific processors."""
    
    @abstractmethod
    def process(self, raw_data: pd.DataFrame, **kwargs) -> Dict[str, pd.DataFrame]:
        """Process raw data into normalized tables."""
        pass
    
    @abstractmethod 
    def get_dataset_type(self) -> str:
        """Return the dataset type this processor handles."""
        pass

# Implementations for GA resolutions
class GAResolutionFetcher(DatasetFetcher):
    """Fetches General Assembly resolution data."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def fetch(self, source_config: Dict[str, Any]) -> pd.DataFrame:
        """Fetch GA resolution data from URL."""
        self.logger.info(f"Fetching GA resolutions from {source_config['url']}")
        try:
            df = pd.read_csv(source_config['url'])
            self.logger.info(f"Successfully fetched {len(df)} GA resolution records")
            
            df['session'] = df['session'].astype(str) # Ensure session is consistent
            df['date'] = pd.to_datetime(df['date']) # Convert date to datetime
            
            return df
        except Exception as e:
            self.logger.error(f"Failed to fetch GA resolutions: {e}")
            raise
    
        
    def get_dataset_type(self) -> str:
        return "ga_resolutions"

class GAResolutionProcessor(DatasetProcessor):
    """Processes General Assembly resolution data."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def process(self, raw_data: pd.DataFrame, **kwargs) -> Dict[str, pd.DataFrame]:
        """Transform GA data from country-per-row to resolution-per-row format."""
        self.logger.info("Processing GA resolution data")
        
        # GA-specific index columns
        ga_index_columns = [
            "undl_id", "date", "session", "resolution", "draft", 
            "committee_report", "meeting", "title", "agenda_title", 
            "subjects", "total_yes", "total_no", "total_abstentions", 
            "total_non_voting", "total_ms", "undl_link"
        ]
        
        # Transform to resolution-per-row format
        transformed_df = raw_data.pivot(
            index=ga_index_columns, 
            columns='ms_code', 
            values='ms_vote'
        ).reset_index()
        transformed_df.columns.name = None
        
        self.logger.info(f"Transformed {len(transformed_df)} GA resolutions")
        
        # Parse subjects using GA-specific logic
        subject_table = kwargs.get('subject_table')
        if subject_table is not None and not subject_table.empty:
            parsed_df = self._parse_subjects(transformed_df, subject_table)
            self.logger.info(f"Processed {len(parsed_df)} GA resolutions with parsed subjects")
            return {"ga_resolutions": parsed_df}
        else:
            self.logger.warning("No subject_table provided, skipping subject parsing")
            return {"ga_resolutions": transformed_df}
    
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
        original_subject_count = len(df['subjects'].unique()) if 'subjects' in df.columns else 0
        
        # First we split the subjects by | and explode the list into separate rows
        df_expanded = df.assign(subjects=df['subjects'].str.split('|')).explode('subjects')

        # For now we just take the first element if --
        df_expanded = df_expanded.assign(subjects=df_expanded['subjects'].str.split('--').str[0]).explode('subjects')

        # Clean up subjects (strip whitespace)
        df_expanded['subjects'] = df_expanded['subjects'].str.strip()
        
        # Remove empty subjects
        #df_expanded = df_expanded[df_expanded['subjects'].notna() & (df_expanded['subjects'] != '')]
        
        final_subject_count = len(df_expanded['subjects'].unique()) if 'subjects' in df_expanded.columns else 0
        self.logger.info(f"Expanded subjects from {original_subject_count} to {final_subject_count} unique subjects.")
        
        # Add subject IDs by matching with subject_table
        df_with_ids = self._add_subject_ids(df_expanded, subject_table)
        
        return df_with_ids
    
    def _add_subject_ids(self, resolution_df: pd.DataFrame, subjects_df: pd.DataFrame) -> pd.DataFrame:
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
            subject_id = row['subject_id']
            
            # Index all labels
            for col in subjects_df.columns:
                if col.startswith('label_') and pd.notna(row[col]):
                    label_lower = str(row[col]).lower().strip()
                    matching_index[label_lower] = subject_id
                        
                # Also index alternative labels if they exist
                if col.startswith('alt_labels_') and pd.notna(row[col]) and str(row[col]).strip():
                    for alt_label in str(row[col]).split(';'):
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
        resolution_df['subject_id'] = resolution_df['subjects'].apply(map_subject)
        
        # Report statistics
        total_rows = len(resolution_df)
        matched_rows = resolution_df['subject_id'].notna().sum()
        unmatched_rows = resolution_df['subject_id'].isna().sum()
        
        self.logger.info(f"GA Subject Mapping Results:")
        self.logger.info(f"  Total rows: {total_rows}")
        self.logger.info(f"  Matched: {matched_rows} ({matched_rows/total_rows*100:.1f}%)")
        self.logger.info(f"  Unmatched: {unmatched_rows} ({unmatched_rows/total_rows*100:.1f}%)")
        
        if unmatched_rows > 0:
            self.logger.info(f"Sample unmatched GA subjects:")
            unmatched_samples = resolution_df[resolution_df['subject_id'].isna()]['subjects'].drop_duplicates().head(5)
            for subject in unmatched_samples:
                self.logger.info(f"  - '{subject}'")
        
        return resolution_df
    
    def get_dataset_type(self) -> str:
        return "ga_resolutions"

# Implementations for SC resolutions
class SCResolutionFetcher(DatasetFetcher):
    """Fetches Security Council resolution data"""

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def fetch(self, source_config: Dict[str, Any]) -> pd.DataFrame:
        """Fetch SC resolution data from URL."""
        self.logger.info(f"Fetching SC resolutions from {source_config['url']}")
        try:
            df = pd.read_csv(source_config['url'])
            self.logger.info(f"Successfully fetched {len(df)} SC resolution records")
            return df
        except Exception as e:
            self.logger.error(f"Failed to fetch SC resolutions: {e}")
            raise
        
    def get_dataset_type(self) -> str:
        return "sc_resolutions"

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

    
# Thesaurus handling (separate from resolution datasets)
class ThesaurusFetcher:
    """Fetches thesaurus data (RDF/TTL format)."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def fetch(self, source_config: Dict[str, Any]) -> Graph:
        """Fetch and parse thesaurus graph."""
        self.logger.info(f"Fetching thesaurus from {source_config['url']}")

        thesaurus_url = source_config['data_sources']['thesaurus']['url']

        graph = Graph()

        try:
            response = requests.get(thesaurus_url)
        except Exception as e:
            self.logger.info("Error fetching thesaurus file. The dataset might has been updated. Check the date in the URL.")
            self.logger.info("Thesaurus URL:", thesaurus_url)
            self.logger.info(f"Error: {e}")
            
            raise ValueError("Failed to fetch thesaurus")

        if response.status_code != 200:
            self.logger.info("Error fetching thesaurus file. The dataset might has been updated. Check the date in the URL. Response code:", response.status_code)
            self.logger.info("Thesaurus URL:", thesaurus_url)

            raise ValueError("Failed to fetch thesaurus")
        
        ttl_content = BytesIO(response.content)
        self.logger.info(f"Downloaded thesaurus file successfully")

        graph.parse(ttl_content, format="turtle")

        return graph

class ThesaurusProcessor:
    """Processes thesaurus data into subject and closure tables."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def process(self, thesaurus_graph: Graph) -> Dict[str, pd.DataFrame]:
        """Process thesaurus into subject and closure tables."""
        self.logger.info("Processing thesaurus data")
        
        # Extract subjects data with multilingual labels
        subjects_data = self._extract_subjects_data(thesaurus_graph)
        
        # Create subjects table
        subject_table = self._create_subjects_table(subjects_data)
        
        # Create closure table for hierarchical relationships
        closure_table = self._create_closure_table(thesaurus_graph)
        
        return {
            "subject_table": subject_table,
            "closure_table": closure_table
        }
    
    def _extract_subjects_data(self, g: Graph) -> Dict[str, Dict[str, Any]]:
        """
        Extract all subjects and schemes with their multilingual labels from the thesaurus.
        
        Args:
            g (Graph): RDFLib Graph containing the thesaurus data.
            
        Returns:
            Dict: Dictionary with subject_id as key and metadata as value
        """
        subjects_data = {}
        
        # First, get all SKOS Concepts
        for subject in g.subjects(RDF.type, SKOS.Concept):
            subject_id = str(subject)
            
            if subject_id not in subjects_data:
                subjects_data[subject_id] = {
                    'subject_id': subject_id,
                    'labels': {},
                    'alt_labels': {},
                    'node_type': 'concept'
                }
            
            # Get all prefLabels with language tags
            for label in g.objects(subject, SKOS.prefLabel):
                if hasattr(label, 'language') and label.language:
                    subjects_data[subject_id]['labels'][label.language] = str(label)
                else:
                    subjects_data[subject_id]['labels']['unknown'] = str(label)
            
            # Get alternative labels
            for alt_label in g.objects(subject, SKOS.altLabel):
                if hasattr(alt_label, 'language') and alt_label.language:
                    if alt_label.language not in subjects_data[subject_id]['alt_labels']:
                        subjects_data[subject_id]['alt_labels'][alt_label.language] = []
                    subjects_data[subject_id]['alt_labels'][alt_label.language].append(str(alt_label))
        
        # Now, get all Concept Schemes (top level containers)
        for scheme in g.subjects(RDF.type, SKOS.ConceptScheme):
            scheme_id = str(scheme)
            
            if scheme_id not in subjects_data:
                subjects_data[scheme_id] = {
                    'subject_id': scheme_id,
                    'labels': {},
                    'alt_labels': {},
                    'node_type': 'scheme'
                }
            
            # Get all prefLabels for schemes
            for label in g.objects(scheme, SKOS.prefLabel):
                if hasattr(label, 'language') and label.language:
                    subjects_data[scheme_id]['labels'][label.language] = str(label)
                else:
                    subjects_data[scheme_id]['labels']['unknown'] = str(label)
        
        self.logger.info(f"Extracted {len(subjects_data)} subjects/schemes:")
        self.logger.info(f"  - Concepts: {sum(1 for d in subjects_data.values() if d.get('node_type') == 'concept')}")
        self.logger.info(f"  - Schemes: {sum(1 for d in subjects_data.values() if d.get('node_type') == 'scheme')}")
        
        return subjects_data

    def _create_subjects_table(self, subjects_data: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """
        Convert subjects dictionary to a DataFrame with language columns.
        
        Args:
            subjects_data (Dict): Dictionary containing subject metadata.
            
        Returns:
            pd.DataFrame: DataFrame with columns for subject_id, labels in different languages, alt_labels, and node_type.
        """
        rows = []
        
        # Common language codes in UNBIS
        languages = ['en', 'es', 'fr', 'ar', 'ru', 'zh']
        
        for subject_id, data in subjects_data.items():
            row = {'subject_id': subject_id}
            
            # Add labels for each language
            for lang in languages:
                row[f'label_{lang}'] = data['labels'].get(lang, None)
                # Store alt labels as semicolon-separated string
                alt_labels = data['alt_labels'].get(lang, [])
                row[f'alt_labels_{lang}'] = '; '.join(alt_labels) if alt_labels else ''
            
            # Add node type (concept or scheme)
            row['node_type'] = data.get('node_type', 'concept')
            
            rows.append(row)
        
        return pd.DataFrame(rows)

    def _create_closure_table(self, g: Graph) -> pd.DataFrame:
        """
        Create a closure table directly from RDFLib Graph using SKOS relationships.
        
        This function traverses the SKOS broader/narrower relationships to build
        a closure table without converting to NetworkX first.
        
        Args:
            g (Graph): RDFLib Graph containing the thesaurus with SKOS relationships
        
        Returns:
            pd.DataFrame: Closure table with columns:
                - ancestor_id (str): URI of the ancestor subject/scheme
                - descendant_id (str): URI of the descendant subject/scheme  
                - depth (int): Number of edges between ancestor and descendant
        """
        
        # First, build adjacency list from SKOS broader relationships
        # broader means parent, so child -> parent edges
        children_to_parents = defaultdict(set)
        parents_to_children = defaultdict(set)
        
        # Get all subjects (concepts and schemes)
        all_subjects = set()
        
        # Add concepts
        for concept in g.subjects(RDF.type, SKOS.Concept):
            all_subjects.add(str(concept))
        
        # Add schemes
        for scheme in g.subjects(RDF.type, SKOS.ConceptScheme):
            all_subjects.add(str(scheme))
        
        # Build parent-child relationships from broader
        for child, parent in g.subject_objects(SKOS.broader):
            child_str = str(child)
            parent_str = str(parent)
            children_to_parents[child_str].add(parent_str)
            parents_to_children[parent_str].add(child_str)
            all_subjects.add(child_str)
            all_subjects.add(parent_str)
        
        # Add scheme relationships (top concepts)
        for scheme in g.subjects(RDF.type, SKOS.ConceptScheme):
            scheme_str = str(scheme)
            # Get top concepts
            for top_concept in g.objects(scheme, SKOS.hasTopConcept):
                concept_str = str(top_concept)
                children_to_parents[concept_str].add(scheme_str)
                parents_to_children[scheme_str].add(concept_str)
        
        self.logger.info(f"Found {len(all_subjects)} subjects/schemes")
        self.logger.info(f"Found {sum(len(p) for p in children_to_parents.values())} parent-child relationships")
        
        # Now build closure table
        closure_data = []
        
        def get_all_ancestors(subject_id):
            """BFS to find all ancestors and their depths"""
            ancestors = {}
            visited = set()
            queue = deque([(subject_id, 0)])
            
            while queue:
                current, depth = queue.popleft()
                
                if current in visited:
                    continue
                visited.add(current)
                
                # Add current as ancestor at this depth (for self-reference at depth 0)
                if current not in ancestors or ancestors[current] > depth:
                    ancestors[current] = depth
                
                # Add parents to queue
                for parent in children_to_parents.get(current, []):
                    if parent not in visited:
                        queue.append((parent, depth + 1))
            
            return ancestors
        
        # Process each subject
        nodes_with_ancestors = 0
        for subject in all_subjects:
            ancestors = get_all_ancestors(subject)
            
            if len(ancestors) > 1:  # More than just self
                nodes_with_ancestors += 1
            
            for ancestor, depth in ancestors.items():
                closure_data.append({
                    'ancestor_id': ancestor,
                    'descendant_id': subject,
                    'depth': depth
                })
        
        closure_df = pd.DataFrame(closure_data)
        
        self.logger.info(f"Closure table statistics:")
        self.logger.info(f"  Total rows: {len(closure_df):,}")
        self.logger.info(f"  Nodes with ancestors: {nodes_with_ancestors:,}")
        self.logger.info(f"  Unique ancestors: {closure_df['ancestor_id'].nunique():,}")
        self.logger.info(f"  Unique descendants: {closure_df['descendant_id'].nunique():,}")
        if len(closure_df) > 0:
            self.logger.info(f"  Max depth: {closure_df['depth'].max()}")
            
            # Show depth distribution
            depth_counts = closure_df['depth'].value_counts().sort_index()
            self.logger.info(f"Depth distribution:")
            for depth, count in depth_counts.items():
                self.logger.info(f"  Depth {depth}: {count:,} relationships")
        
        return closure_df

class DataFetcher:
    """Orchestrates fetching from multiple data sources."""

    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self.config = config
        self.logger = logger
        
        # Registry of dataset fetchers
        self._dataset_fetchers: Dict[str, DatasetFetcher] = {}
        self._register_default_fetchers()
        
        # Thesaurus fetcher (separate from datasets)
        self.thesaurus_fetcher = ThesaurusFetcher(logger)
    
    def _register_default_fetchers(self):
        """Register default dataset fetchers."""
        ga_fetcher = GAResolutionFetcher(self.logger)
        self._dataset_fetchers[ga_fetcher.get_dataset_type()] = ga_fetcher
        
        # Future fetchers can be added here:
        sc_fetcher = SCResolutionFetcher(self.logger)
        self._dataset_fetchers[sc_fetcher.get_dataset_type()] = sc_fetcher
    
    def register_fetcher(self, fetcher: DatasetFetcher):
        """Register a new dataset fetcher."""
        self._dataset_fetchers[fetcher.get_dataset_type()] = fetcher
    
    def fetch_resolutions(self) -> Dict[str, pd.DataFrame]:
        """Fetch all configured resolution datasets."""
        results = {}
        
        for dataset_type, source_config in self.config['data_sources']['resolutions'].items():
                
            if dataset_type in self._dataset_fetchers:
                fetcher = self._dataset_fetchers[dataset_type]
                results[dataset_type] = fetcher.fetch(source_config)
            else:
                self.logger.warning(f"No fetcher registered for dataset type: {dataset_type}")
        
        return results
    
    def fetch_thesaurus(self) -> Graph:
        """Fetch thesaurus data."""
        thesaurus_config = self.config['data_sources'].get('thesaurus')
        if not thesaurus_config:
            raise ValueError("No thesaurus configuration found")
        
        return self.thesaurus_fetcher.fetch(thesaurus_config)

class DataProcessor:
    """Orchestrates processing of individual datasets."""
    
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self.config = config
        self.logger = logger
        
        # Registry of dataset processors
        self._dataset_processors: Dict[str, DatasetProcessor] = {}
        self._register_default_processors()
        
        # Thesaurus processor (separate from datasets)
        self.thesaurus_processor = ThesaurusProcessor(logger)
    
    def _register_default_processors(self):
        """Register default dataset processors."""
        ga_processor = GAResolutionProcessor(self.logger)
        self._dataset_processors[ga_processor.get_dataset_type()] = ga_processor
        
        # Future processors can be added here:
        sc_processor = SCResolutionProcessor(self.logger)
        self._dataset_processors[sc_processor.get_dataset_type()] = sc_processor
    
    def register_processor(self, processor: DatasetProcessor):
        """Register a new dataset processor."""
        self._dataset_processors[processor.get_dataset_type()] = processor
    
    def process_resolutions(self, raw_datasets: Dict[str, pd.DataFrame], **kwargs) -> Dict[str, pd.DataFrame]:
        """Process individual resolution datasets."""
        processed_datasets = {}
        
        for dataset_type, raw_data in raw_datasets.items():
            if dataset_type in self._dataset_processors:
                processor = self._dataset_processors[dataset_type]
                processed_data = processor.process(raw_data, **kwargs)
                processed_datasets.update(processed_data)
            else:
                self.logger.warning(f"No processor registered for dataset type: {dataset_type}")
        
        return processed_datasets
    
    def process_thesaurus(self, thesaurus_graph: Graph) -> Dict[str, pd.DataFrame]:
        """Process thesaurus data."""
        return self.thesaurus_processor.process(thesaurus_graph)
    
    def normalize_resolutions(self, resolutions_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Normalize the resolutions dataframe into separate tables.
        
        Args:
            resolutions_df : pd.DataFrame
                DataFrame with one row per resolution-subject pair, containing
                resolution metadata and subject_id column
            
        Returns:
            tuple of (resolutions_normalized_df, resolution_subjects_df)
                - resolutions_normalized_df: One row per resolution with metadata
                - resolution_subjects_df: Resolution-subject pairs mapping table
        """
        
        # Identify columns that belong to resolution metadata vs subject mapping
        subject_columns = ['subjects', 'subject_id']
        resolution_columns = [col for col in resolutions_df.columns if col not in subject_columns]
        
        # 1. Create normalized resolutions table (one row per resolution)
        resolutions_normalized_df = resolutions_df[resolution_columns].drop_duplicates()
        
        # 2. Create resolution-subject mapping table
        # Only keep rows with valid subject_ids
        valid_mappings = resolutions_df[resolutions_df['subject_id'].notna()]
        resolution_subjects_df = valid_mappings[['undl_id', 'subject_id']].copy()
        
        # Remove duplicates (in case same subject appears multiple times for a resolution)
        resolution_subjects_df = resolution_subjects_df.drop_duplicates()
        
        # 3. Report statistics
        # self.logger.info("Normalization Results:")
        # self.logger.info(f"\nResolutions Table:")
        # self.logger.info(f"  Total resolutions: {len(resolutions_normalized_df)}")
        # self.logger.info(f"  Columns: {', '.join(resolutions_normalized_df.columns)}")
        
        # self.logger.info(f"\nResolution-Subject Mapping Table:")
        # self.logger.info(f"  Total mappings: {len(resolution_subjects_df)}")
        # self.logger.info(f"  Unique resolutions with subjects: {resolution_subjects_df['undl_id'].nunique()}")
        # self.logger.info(f"  Unique subjects used: {resolution_subjects_df['subject_id'].nunique()}")
        # self.logger.info(f"  Avg subjects per resolution: {len(resolution_subjects_df) / resolution_subjects_df['undl_id'].nunique():.2f}")
        
        # Check for resolutions without subjects
        resolutions_without_subjects = set(resolutions_normalized_df['undl_id']) - set(resolution_subjects_df['undl_id'])
        if resolutions_without_subjects:
            self.logger.info(f"\nWarning: {len(resolutions_without_subjects)} resolutions have no mapped subjects")
        
        return resolutions_normalized_df, resolution_subjects_df

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

class DataRepository:
    """Handles storage and retrieval of processed UN data."""
    
    def __init__(self, config_path: Path):
        self.config_path = config_path
        
        # Initialize data attributes
        self.resolution_table: pd.DataFrame
        self.resolution_subject_table: pd.DataFrame
        self.subject_table: pd.DataFrame
        self.closure_table: pd.DataFrame

        # Load configuration
        self._load_config()

        # Initialize Logging
        self._setup_logging()

        self.logger.info("Initializing UNDataRepository")

        # Check if links are still valid
        if not self._check_URLS():
            self.logger.error("One or more data URLs are invalid. The dataset might have been updated. Check the date in the URL.")
            raise ValueError("Invalid data URLs in configuration.")
        
        # Check if data is already processed and available
        if self._has_cached_data():
            # Cached data found -> load
            self._load_cached_data() 
            self.logger.info("Initialization Complete with Cached Data.")
            return
        
        self._build_data()
        self.logger.info("Initialization complete with fetched Data.")

    def get_data(self) -> Dict[str, pd.DataFrame]:
        """Return processed data as a dictionary of DataFrames."""
        return {
            'resolution': self.resolution_table,
            'resolution_subject': self.resolution_subject_table,
            'subject': self.subject_table,
            'closure': self.closure_table
        }
    
    def _load_config(self):
        """Load configuration from YAML file."""
        with open(self.config_path, 'r') as file:
            self.config = yaml.safe_load(file)

    def _setup_logging(self):
        """
        Setup logging configuration with file and console handlers.

        """
        # Create logger
        self.logger = logging.getLogger('UNResolutionAnalyzer')
        
        if not self.config['logs']:
            self.logger.disabled = True
            return

        self.logger.setLevel(logging.DEBUG if self.config['debug'] else logging.INFO)

        # Clear any existing handlers
        self.logger.handlers.clear()

        # Create formatters
        detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        simple_formatter = logging.Formatter(
            '%(levelname)s - %(message)s'
        )

        log_dir = Path(self.config['paths']['logs'])
        log_dir.mkdir(exist_ok=True)
        log_file = log_dir / 'un_resolution_analyzer.log'

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_formatter)
        self.logger.addHandler(file_handler)

        # Console handler
        if self.config['debug']:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.DEBUG)
            console_handler.setFormatter(simple_formatter)
            self.logger.addHandler(console_handler)

        self.logger.info("Logging setup complete.")

    def _check_URLS(self) -> bool:
        """Check if the URLs in the configuration are reachable."""
        urls = [
            source['url'] 
            for source in self.config['data_sources'].values() 
            if 'url' in source
        ]
        all_valid = True
        for url in urls:
            try:
                response = requests.head(url, allow_redirects=True, timeout=10)
                if response.status_code != 200:
                    self.logger.error(f"URL not reachable: {url} (Status code: {response.status_code})")
                    all_valid = False
                else:
                    self.logger.info(f"URL is valid: {url}")
            except requests.RequestException as e:
                self.logger.error(f"Error reaching URL: {url} ({e})")
                all_valid = False
        return all_valid
    
    def _has_cached_data(self) -> bool:
        """Check if cached data files exist."""
        data_path = Path(self.config['paths']['data'])
        required_files = [
            'resolution_table.csv',
            'resolution_subject_table.csv',
            'subject_table.csv',
            'closure_table.csv'
        ]
        all_exist = all((data_path / file).exists() for file in required_files)
        if all_exist:
            self.logger.info("Cached Data files found.")
        else:
            self.logger.info("Cached Data files not found.")
        return all_exist
    
    def _load_cached_data(self):
        """Load cached data files into DataFrames."""
        data_path = Path(self.config['paths']['data'])
        data_path.mkdir(exist_ok=True)
        self.resolution_table = pd.read_csv(data_path / 'resolution_table.csv')
        self.resolution_subject_table = pd.read_csv(data_path / 'resolution_subject_table.csv')
        self.subject_table = pd.read_csv(data_path / 'subject_table.csv')
        self.closure_table = pd.read_csv(data_path / 'closure_table.csv')
        self.logger.info("Cached data loaded successfully.")
    
    def _save_cached_data(self):
        """Save data files into DataFrames."""
        data_path = Path(self.config['paths']['data'])
        data_path.mkdir(exist_ok=True)
        # Save the tables in the defined folder
        self.resolution_table.to_csv(data_path / 'resolution_table.csv', index=False)
        self.resolution_subject_table.to_csv(data_path / 'resolution_subject_table.csv', index=False)
        self.subject_table.to_csv(data_path / 'subject_table.csv', index=False)
        self.closure_table.to_csv(data_path / 'closure_table.csv', index=False)

    def _build_data(self):
        """Build processed data tables from raw sources."""
        
        # Fetch Data
        fetcher = DataFetcher(self.config, self.logger)
        resolutions_raw = fetcher.fetch_resolutions()
        thesaurus_graph = fetcher.fetch_thesaurus()

        # Process data
        processor = DataProcessor(self.config, self.logger)
        merger = DataMerger(self.logger)
        
        # Process thesaurus first (needed for subject matching)
        thesaurus_tables = processor.process_thesaurus(thesaurus_graph)
        self.subject_table = thesaurus_tables.get('subject_table', pd.DataFrame())
        self.closure_table = thesaurus_tables.get('closure_table', pd.DataFrame())
        
        # Process individual resolution datasets
        processed_datasets = processor.process_resolutions(resolutions_raw, kwargs={'subject_table': self.subject_table})
        
        # Continue with GA for now
        # unified_resolutions = merger.merge_resolutions(processed_datasets)
        ga_resolutions = processed_datasets.get('ga_resolutions', pd.DataFrame())

        # Normalize ga_resolutions
        self.resolution_table, self.resolution_subject_table = processor.normalize_resolutions(ga_resolutions)

        # Save processed data
        self._save_cached_data()

class ResolutionQueryEngine:
    """Provides querying capabilities on processed resolution data."""
    
    def __init__(self, resolution_df: pd.DataFrame, resolution_subject_df: pd.DataFrame, subject_df: pd.DataFrame, closure_df: pd.DataFrame):
        self.resolution_df = resolution_df
        self.resolution_subject_df = resolution_subject_df
        self.subject_df = subject_df
        self.closure_df = closure_df
        
    def query(self, start_date: Optional[str] = None, end_date: Optional[str] = None, subject_ids: Optional[List[str]] = None, include_descendants: bool = True) -> pd.DataFrame:
        """Query resolutions based on date range and subject filters."""
        # TODO: Implement querying logic
        return pd.DataFrame()

class ResolutionAnalyzer:
    """Analyzes resolution data for trends and patterns."""
    
    def __init__(self, resolution_df: pd.DataFrame, resolution_subject_df: pd.DataFrame):
        self.resolution_df = resolution_df
        self.resolution_subject_df = resolution_subject_df
        
    def voting_trends(self) -> pd.DataFrame:
        """Analyze voting trends over time."""
        # TODO: Implement voting trends analysis
        return pd.DataFrame()
    
    def subject_distribution(self) -> pd.DataFrame:
        """Analyze distribution of subjects across resolutions."""
        # TODO: Implement subject distribution analysis
        return pd.DataFrame()