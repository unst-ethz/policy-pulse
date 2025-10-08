import logging
import sys
import yaml
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, Any, List, Optional
from io import BytesIO
import pandas as pd
import requests
from rdflib import Graph, RDF, SKOS


class UNResolutionAnalyzer:
    """
    Main interface for querying UN resolutions with hierarchical subject filtering.
    
    This class handles data loading, caching, and querying of UN resolutions
    using the UNBIS thesaurus for hierarchical subject classification.
    """
    
    def __init__(
        self,
        config_path: str,
    ):
        """
        Initialize the analyzer with configuration.
        
        Args:
            config_path: Path to YAML configuration file
            cache_dir: Directory for cached dataframes (if None, use config default)
            force_refresh: If True, bypass cache and rebuild all data
            check_updates: If True, check if remote data is newer than cache

        Notes:
            Add security council data to the pipeline
        """
        self.config_path = config_path

        # Load configuration
        self._load_config()

        # Initialize Logging
        self._setup_logging()

        self.logger.info("Initializing UNResolutionAnalyzer")

        # Check if data is present
        data_folder = Path(self.config['paths']['data'])
        data_folder.mkdir(exist_ok=True)
        resolution_table_path = data_folder / "resolution_table.csv"
        resolution_subject_table_path = data_folder / "resolution_subject_table.csv"
        subject_table = data_folder / "subject_table.csv"
        closure_table = data_folder / "closure_table.csv"

        try:
            self.resolution_table = pd.read_csv(resolution_table_path)
            self.resolution_subject_table = pd.read_csv(resolution_subject_table_path)
            self.subject_df = pd.read_csv(subject_table)
            self.closure_table = pd.read_csv(closure_table)
            self.logger.info("Loaded data from local source successfully.")
            return
        except Exception as e:
            self.logger.info("Local data not found or incomplete, fetching and processing data.")
        
        # Fetch data, transform it and parse subjects
        df_ga, df_sc = self._fetch_resolutions()

        df_ga_transformed, df_sc_transformed = self._transform_resolutions(df_ga, df_sc)

        df_ga_parsed = self._parse_subjects(df_ga_transformed)

        # Fetch and parse thesaurus
        g = self._fetch_thesaurus()
    
        subjects_data = self._extract_subjects_data(g)

        self.subject_df = self._create_subjects_table(subjects_data)

        df_ga_mapped = self._add_subject_ids(df_ga_parsed, self.subject_df)

        self.resolution_table, self.resolution_subject_table = self._normalize_resolution_dataframe(df_ga_mapped)

        self.closure_table = self._create_closure_table(g)

        # Save the tables in the defined folder
        self.resolution_table.to_csv(resolution_table_path, index=False)
        self.resolution_subject_table.to_csv(resolution_subject_table_path, index=False)
        self.subject_df.to_csv(subject_table, index=False)
        self.closure_table.to_csv(closure_table, index=False)

        

    # Public API methods
    def query(self, start_date: Optional[str] = None, end_date: Optional[str] = None, subject_ids: Optional[List[str]] = None, language: str = "en", include_descendants: bool = True) -> pd.DataFrame:
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

        # Start with all resolutions
        filtered_df = self.resolution_table.copy()

        filtered_df['date'] = pd.to_datetime(filtered_df['date'])

        # 1. Apply date filters
        if start_date:
            filtered_df = filtered_df[filtered_df['date'] >= pd.to_datetime(start_date)]
        if end_date:
            filtered_df = filtered_df[filtered_df['date'] <= pd.to_datetime(end_date)]
        
        # 2. Apply subject filters
        if subject_ids is not None and len(subject_ids) > 0:
            if include_descendants:
                # Expand subject_ids to include all descendants
                expanded_subjects = set(subject_ids)
                
                for subject_id in subject_ids:
                    # Find all descendants of this subject
                    descendants = self.closure_table[
                        self.closure_table['ancestor_id'] == subject_id
                    ]['descendant_id'].unique()
                    expanded_subjects.update(descendants)
                
                self.logger.info(f"Expanded {len(subject_ids)} subjects to {len(expanded_subjects)} (including descendants)")
                subject_filter = list(expanded_subjects)
            else:
                subject_filter = subject_ids
            
            # Find resolutions with these subjects
            matching_resolution_ids = self.resolution_subject_table[
                self.resolution_subject_table['subject_id'].isin(subject_filter)
            ]['undl_id'].unique()
            
            # Filter resolutions
            filtered_df = filtered_df[
                filtered_df['undl_id'].isin(matching_resolution_ids)
            ]
            self.logger.info(f"After subject filter: {len(filtered_df)} resolutions")
        
        self.logger.info(f"\nFinal result: {len(filtered_df)} resolutions")
        return filtered_df
    
    #def get_subjects_by_name(self, ...): pass
    #def get_subject_hierarchy(self, ...): pass
    
    # Pipeline steps (private)
    def _load_config(self):
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)

    def _fetch_resolutions(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """    
        This function retrieves voting data from the UN Digital Library.

        Args:
            config (dict): Configuration dictionary containing data source information.

        Returns:
            tuple: A tuple containing 2 DataFrames:
                    - df_ga: GA voting data with one entry per resolution-country pair
                    - df_sc: SC voting data with one entry per resolution-country pair

        Notes:
            Currently, the Security Council data does not include veto information explicitly.
        """
        ga_url = self.config['data_sources']['ga_resolutions']['url']
        sc_url = self.config['data_sources']['sc_resolutions']['url']

        try:
            df_ga = pd.read_csv(ga_url)
            df_sc = pd.read_csv(sc_url)
        except Exception as e:
            self.logger.info("Error fetching data from UN Digital Library. The dataset might has been updated. Check the date in the URL.")
            self.logger.info("GA URL:", ga_url)
            self.logger.info("SC URL:", sc_url)
            self.logger.info(f"Error: {e}")

            raise ValueError("Failed to fetch UN resolutions")
        
        df_ga['session'] = df_ga['session'].astype(str) # Ensure session is consistent
        df_ga['date'] = pd.to_datetime(df_ga['date']) # Convert date to datetime

        return df_ga, df_sc
    
    def _transform_resolutions(self, df_ga: pd.DataFrame, df_sc: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Transform the GA and SC resolutions DataFrames to have one entry per resolution.

        Args:
            df_ga (pd.DataFrame): Original GA voting data.
            df_sc (pd.DataFrame): Original SC voting data.

        Returns:
            tuple: Transformed DataFrames (df_ga_transformed, df_sc_transformed).
        """
        
        ga_index_cols = ["undl_id", "date", "session", "resolution", "draft", "committee_report", "meeting", "title", "agenda_title", "subjects", "total_yes", "total_no", "total_abstentions", "total_non_voting", "total_ms", "undl_link"]
        df_ga_transformed = df_ga.pivot(index=ga_index_cols, columns='ms_code', values='ms_vote').reset_index()
        df_ga_transformed.columns.name = None

        # Transform SC DataFrame
        sc_index_cols = ["undl_id", "date", "resolution", "draft", "meeting", "description", "agenda", "subjects", "modality", "total_yes", "total_no", "total_abstentions", "total_non_voting", "total_ms", "undl_link"]
        df_sc_transformed = df_sc.pivot(index=sc_index_cols, columns='ms_code', values='ms_vote').reset_index()
        df_sc_transformed.columns.name = None

        return df_ga_transformed, df_sc_transformed

    def _parse_subjects(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Parse the 'subjects' column in the DataFrame to create a separate row for each subject.

        Args:
            df (pd.DataFrame): DataFrame containing a 'subjects' column with special parsing grammar.

        Returns:
            pd.DataFrame: DataFrame with each subject in a separate row.
        
        Notes:
            The parsing grammar includes splitting at | and --
        """

        # First we split the subjects by | and explode the list into separate rows
        df_expanded = df.assign(subjects=df['subjects'].str.split('|')).explode('subjects')

        # For now we just take the first element if --
        df_expanded = df_expanded.assign(subjects=df_expanded['subjects'].str.split('--').str[0]).explode('subjects')

        self.logger.info(f"Expanded subjects from {len(df['subjects'].unique())} to {len(df_expanded['subjects'].unique())} rows.")
        return df_expanded
    
    def _fetch_thesaurus(self):
        """
        Fetch and parse the UN thesaurus TTL file.
        
        """
        thesaurus_url = self.config['data_sources']['thesaurus']['url']

        g = Graph()

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

        g.parse(ttl_content, format="turtle")

        return g
    
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
                row[f'alt_labels_{lang}'] = '; '.join(alt_labels) if alt_labels else None
            
            # Add node type (concept or scheme)
            row['node_type'] = data.get('node_type', 'concept')
            
            rows.append(row)
        
        return pd.DataFrame(rows)

    def _add_subject_ids(self, resolution_df: pd.DataFrame, subjects_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add subject_id column to resolutions dataframe by matching subject strings.
        
        Args:
            resolutions_df : pd.DataFrame
                DataFrame with resolutions, already parsed to one row per resolution-subject pair
            subjects_df : pd.DataFrame
                DataFrame with subject_id and multilingual labels
            subject_column : str
                Name of the column in resolutions_df containing subject strings
            
        Returns:
            pd.DataFrame
                Original resolutions_df with added 'subject_id' column
        """
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
                if col.startswith('alt_labels_') and pd.notna(row[col]):
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
        
        self.logger.info(f"Mapping Results:")
        self.logger.info(f"  Total rows: {total_rows}")
        self.logger.info(f"  Matched: {matched_rows} ({matched_rows/total_rows*100:.1f}%)")
        self.logger.info(f"  Unmatched: {unmatched_rows} ({unmatched_rows/total_rows*100:.1f}%)")
        
        if unmatched_rows > 0:
            self.logger.info(f"\nSample unmatched subjects:")
            unmatched_samples = resolution_df[resolution_df['subject_id'].isna()]['subjects'].drop_duplicates().head(10)
            for subject in unmatched_samples:
                self.logger.info(f"  - '{subject}'")
        
        return resolution_df
    
    def _normalize_resolution_dataframe(self, resolutions_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
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
        self.logger.info("Normalization Results:")
        self.logger.info(f"\nResolutions Table:")
        self.logger.info(f"  Total resolutions: {len(resolutions_normalized_df)}")
        self.logger.info(f"  Columns: {', '.join(resolutions_normalized_df.columns)}")
        
        self.logger.info(f"\nResolution-Subject Mapping Table:")
        self.logger.info(f"  Total mappings: {len(resolution_subjects_df)}")
        self.logger.info(f"  Unique resolutions with subjects: {resolution_subjects_df['undl_id'].nunique()}")
        self.logger.info(f"  Unique subjects used: {resolution_subjects_df['subject_id'].nunique()}")
        self.logger.info(f"  Avg subjects per resolution: {len(resolution_subjects_df) / resolution_subjects_df['undl_id'].nunique():.2f}")
        
        # Check for resolutions without subjects
        resolutions_without_subjects = set(resolutions_normalized_df['undl_id']) - set(resolution_subjects_df['undl_id'])
        if resolutions_without_subjects:
            self.logger.info(f"\nWarning: {len(resolutions_without_subjects)} resolutions have no mapped subjects")
        
        return resolutions_normalized_df, resolution_subjects_df
    
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
        
        self.logger.info(f"\nClosure table statistics:")
        self.logger.info(f"  Total rows: {len(closure_df):,}")
        self.logger.info(f"  Nodes with ancestors: {nodes_with_ancestors:,}")
        self.logger.info(f"  Unique ancestors: {closure_df['ancestor_id'].nunique():,}")
        self.logger.info(f"  Unique descendants: {closure_df['descendant_id'].nunique():,}")
        self.logger.info(f"  Max depth: {closure_df['depth'].max()}")
        
        # Show depth distribution
        depth_counts = closure_df['depth'].value_counts().sort_index()
        self.logger.info(f"\nDepth distribution:")
        for depth, count in depth_counts.items():
            self.logger.info(f"  Depth {depth}: {count:,} relationships")
        
        return closure_df

    # Utility methods
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