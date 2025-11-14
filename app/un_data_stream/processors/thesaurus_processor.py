"""
Thesaurus data processor.

This module handles processing thesaurus data from RDF/TTL format into
structured subject and closure tables.
"""

import logging
import pandas as pd
from collections import defaultdict, deque
from typing import Dict, Any
from rdflib import RDF, SKOS

# Note: rdflib imports are handled at runtime to avoid import errors
# from rdflib import Graph, RDF, SKOS


class ThesaurusProcessor:
    """Processes thesaurus data into subject and closure tables."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def process(self, thesaurus_graph) -> Dict[str, pd.DataFrame]:
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
    
    def _extract_subjects_data(self, g) -> Dict[str, Dict[str, Any]]:
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

    def _create_closure_table(self, g) -> pd.DataFrame:
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
        
        # self.logger.info(f"Closure table statistics:")
        # self.logger.info(f"  Total rows: {len(closure_df):,}")
        # self.logger.info(f"  Nodes with ancestors: {nodes_with_ancestors:,}")
        # self.logger.info(f"  Unique ancestors: {closure_df['ancestor_id'].nunique():,}")
        # self.logger.info(f"  Unique descendants: {closure_df['descendant_id'].nunique():,}")
        # if len(closure_df) > 0:
        #     self.logger.info(f"  Max depth: {closure_df['depth'].max()}")
            
        #     # Show depth distribution
        #     depth_counts = closure_df['depth'].value_counts().sort_index()
        #     self.logger.info(f"Depth distribution:")
        #     for depth, count in depth_counts.items():
        #         self.logger.info(f"  Depth {depth}: {count:,} relationships")
        
        return closure_df