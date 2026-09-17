"""
Data repository for the app's in-memory copy of the storage layer.

Reads the already-normalized tables out of Postgres (written by the `undl-ingest` jobs), reshapes
them into the shapes the app's query layer expects, and precomputes the agreement/alignment arrays
`ResolutionQueryEngine` broadcasts over. The app is a read-only consumer — it does not fetch from
UNDL, and it does not process raw source data.
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple
from urllib.parse import quote

import pandas as pd
import yaml

from . import db
from .processor import DataProcessor

# Resolution columns the app actually reads, as {app-facing name: resolution_outcomes column}.
#
# Deliberately a subset: `resolution_outcomes` also stores `vote_note`, `meeting`,
# `committee_report`, `amended_draft`, `related_documents`, `description`, `agenda`, `modality` and
# `source_dataset`, none of which any feature reads today. Adding one back is a one-line change
# here and nothing else — the country columns are stated explicitly rather than inferred from
# "whatever isn't known metadata".
#
# `session` is the one real rename: Postgres stores a numeric `session` plus a `session_label`
# holding the format the app has always used ('10', '10sp', '10emsp'), and special-session
# detection (`app/features/agreement_graph.py`) depends on the label form.
RESOLUTION_COLUMNS: Dict[str, str] = {
    "undl_id": "undl_id",
    "resolution": "resolution",
    "date": "date",
    "session": "session_label",
    "title": "title",
    "agenda_title": "agenda_title",
    "subjects": "subjects",
    "draft": "draft",
    "total_yes": "total_yes",
    "total_no": "total_no",
    "total_abstentions": "total_abstentions",
    "total_non_voting": "total_non_voting",
    "total_ms": "total_ms",
    # Derived at ingestion time: 'Vote, recorded' | 'Vote, non-recorded' | 'Without a vote'.
    # Carried because only 'Vote, recorded' resolutions have per-country vote rows, and the UI
    # needs to say so — every score is computed over that subset even when the filter matches more.
    "modality": "modality",
}

VOTE_COLUMNS = ["undl_id", "country_code", "vote"]

# Vote codes: Yes / No / Abstained / non-voting. Stored as a category rather than object dtype —
# 19 MB instead of 167 MB for the wide frame, and the same dtype the old CSV cache used.
VOTE_DTYPE = pd.CategoricalDtype(categories=["Y", "N", "A", "X"], ordered=False)

# Resolution -> UN Digital Library links, built here rather than in each of the five features
# that read `undl_link`.
#
# These are deliberately *search* links, not record links. `undl_id` identifies a
# metadata.un.org MARC bib record — the source of truth these tables are ingested from — and
# digitallibrary.un.org keeps its own, different ids for the same resolution, with no mapping
# published between the two systems. Linking to a search lands the user on a results page
# listing the documents for that resolution, which is the best available behaviour;
# treat them as "look it up" links, never as guaranteed single-hit record links.
UNDL_SEARCH_URL = "https://digitallibrary.un.org/search?p="

# Preferred: the resolution symbol. Can be with or without MARC field 791 — e.g. 791:"A/RES/80/311".
UNDL_SYMBOL_QUERY = '"{}"'  #'791:"{}"' Default without for now for more extensive search

# Fallback: match on field 035, which carries our own undl_id. `resolution` is nullable
# in the schema (currently populated on every row, but not guaranteed), and a row without a
# symbol would otherwise get no link at all.
UNDL_ID_QUERY = "035:*{}"


def _undl_search_links(symbols: pd.Series, undl_ids: pd.Series) -> List[str]:
    """Build one UN Digital Library search URL per resolution.

    Percent-encodes the query: symbols carry '/', and some older ones also carry parentheses or
    brackets ('A/RES/13(I)', 'A/RES/71/101[B]'). ':' and '*' are left literal so the field prefix
    and the wildcard stay legible in the URL.
    """
    links = []
    for symbol, undl_id in zip(symbols, undl_ids):
        if isinstance(symbol, str) and symbol.strip():
            query = UNDL_SYMBOL_QUERY.format(symbol.strip())
        else:
            query = UNDL_ID_QUERY.format(undl_id)
        links.append(UNDL_SEARCH_URL + quote(query, safe=":*"))
    return links


class DataRepository:
    """Loads and holds the app's processed UN data, sourced from Postgres."""

    def __init__(self, config_path: str):
        self.config_path = config_path

        # Initialize data attributes
        self.resolution_table: pd.DataFrame
        self.resolution_subject_table: pd.DataFrame
        self.subject_table: pd.DataFrame
        self.closure_table: pd.DataFrame
        self.broader_table: pd.DataFrame
        self.member_states_table: pd.DataFrame

        # Load configuration
        self._load_config()

        # Initialize Logging
        self._setup_logging()

        self.logger.info("Initializing UNDataRepository from Postgres")

        self._load_from_postgres()

        self.logger.info("Initialization complete. Below are memory footprints:")
        self._log_memory_footprints()

    def get_data(self) -> Dict[str, Any]:
        """Return all processed data as a dict consumed by ResolutionQueryEngine.

        Keys:
            resolution, resolution_subject, subject, closure, broader — pd.DataFrames
            country_columns     — List[str] of ISO3 country codes (length C)
            multilateral_scores — (R x C) np.ndarray, float32
            vote_bool_arrays    — 4-tuple of (R x C) bool arrays (yes, no, abstained, voted)
        """
        return {
            "resolution": self.resolution_table,
            "resolution_subject": self.resolution_subject_table,
            "subject": self.subject_table,
            "closure": self.closure_table,
            "broader": self.broader_table,
            "country_columns": self.country_columns,
            "member_states": self.member_states_table,
            "multilateral_scores": self.multilateral_scores,
            "vote_bool_arrays": self.vote_bool_arrays,
        }

    def _load_config(self):
        """Load configuration from YAML file."""
        with open(self.config_path, "r") as file:
            self.config = yaml.safe_load(file)

        # Resolve relative paths in config relative to project root (parent of config dir)
        project_root = Path(self.config_path).resolve().parent.parent
        for key, val in self.config.get("paths", {}).items():
            if not Path(val).is_absolute():
                self.config["paths"][key] = str(project_root / val)

    def _setup_logging(self):
        """Setup logging configuration with file and console handlers."""
        # Create logger
        self.logger = logging.getLogger("UNResolutionAnalyzer")

        if not self.config["logs"]:
            self.logger.disabled = True
            return

        self.logger.setLevel(logging.DEBUG if self.config["debug"] else logging.INFO)

        # Clear any existing handlers
        self.logger.handlers.clear()

        # Create formatters
        detailed_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
        )
        simple_formatter = logging.Formatter("%(levelname)s - %(message)s")

        log_dir = Path(self.config["paths"]["logs"])
        log_dir.mkdir(exist_ok=True)
        log_file = log_dir / "un_resolution_analyzer.log"

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_formatter)
        self.logger.addHandler(file_handler)

        # Console handler
        if self.config["debug"]:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.DEBUG)
            console_handler.setFormatter(simple_formatter)
            self.logger.addHandler(console_handler)

        self.logger.info("Logging setup complete.")

    def _load_from_postgres(self):
        """Read every table the app needs, then derive the in-memory representation.

        The connection is opened, drained and closed inside this method. Nothing holds a live
        connection afterwards: this runs at import time, which under gunicorn's `--preload` is the
        master process before it forks, and a connection that survived the fork would be shared
        across workers. See `db.create_engine`.
        """
        db.load_env()
        engine = db.create_engine()
        try:
            self.logger.info("Reading tables from Postgres")
            with db.connect_or_explain(engine, self.logger) as conn:
                outcomes = db.read_table(
                    conn, "resolution_outcomes", columns=list(RESOLUTION_COLUMNS.values())
                )
                votes = db.read_table(conn, "resolution_votes", columns=VOTE_COLUMNS)
                self.subject_table = db.read_table(conn, "subject")
                self.resolution_subject_table = db.read_table(
                    conn, "resolution_subject", columns=["undl_id", "subject_id"]
                )
                self.closure_table = db.read_table(conn, "subject_closure")
                self.broader_table = db.read_table(conn, "subject_broader")
                self.member_states_table = db.read_table(conn, "member_states")
        finally:
            engine.dispose()

        self.logger.info(
            f"Read {len(outcomes)} resolutions, {len(votes)} votes, "
            f"{len(self.subject_table)} subjects, "
            f"{len(self.resolution_subject_table)} resolution-subject pairs, "
            f"{len(self.member_states_table)} member states"
        )

        self.resolution_table, self.country_columns = self._build_resolution_table(outcomes, votes)

        # Precompute the arrays every agreement/alignment query broadcasts over.
        # `country_columns` is passed in, not inferred: the query engine turns a country code into
        # a *column index* into these arrays, so their column order has to be exactly the list the
        # wide frame was built from.
        processor = DataProcessor(self.config, self.logger)
        (
            consensus_scores,
            self.multilateral_scores,
            self.vote_bool_arrays,
        ) = processor.calculate_agreement_data(self.resolution_table, self.country_columns)

        # Add consensus scores to the resolution table
        self.resolution_table["consensus_score"] = self.resolution_table["undl_id"].map(
            consensus_scores
        )

        self._prune_unused_subjects()

    @staticmethod
    def _build_resolution_table(
        outcomes: pd.DataFrame, votes: pd.DataFrame
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Turn the normalized outcome/vote tables into the wide one-row-per-resolution frame.

        `resolution_votes` is stored long (one row per resolution x country); the query engine
        wants one row per resolution with a column per country, so it gets pivoted and joined back
        onto the resolution metadata.

        Args:
            outcomes: `resolution_outcomes` rows, restricted to `RESOLUTION_COLUMNS`
            votes: `resolution_votes` rows (undl_id, country_code, vote)

        Returns:
            (wide resolution frame, country column names in frame order)
        """
        rename = {source: app_facing for app_facing, source in RESOLUTION_COLUMNS.items()}
        meta = outcomes.rename(columns=rename)
        meta["date"] = pd.to_datetime(meta["date"])
        meta["session"] = meta["session"].astype(str)
        meta["undl_link"] = _undl_search_links(meta["resolution"], meta["undl_id"])

        votes_wide = votes.pivot(index="undl_id", columns="country_code", values="vote")
        votes_wide.columns.name = None
        country_columns = sorted(votes_wide.columns)

        resolution_table = meta.merge(votes_wide, on="undl_id", how="left")

        # A resolution with no `resolution_votes` rows at all is one that was never voted on
        # per-country — adopted without a vote, or adopted by a non-recorded vote. 'X'
        # (non-voting) is the correct code for every country there, not missing data.
        resolution_table[country_columns] = (
            resolution_table[country_columns].fillna("X").astype(VOTE_DTYPE)
        )

        return resolution_table, country_columns

    def _prune_unused_subjects(self):
        """Drop thesaurus entries no resolution maps to, directly or as an ancestor.

        The filter UI builds its subject tree from these tables, so without this it would offer
        thousands of subjects that can never match a resolution. Keeping a subject's ancestors is
        what preserves the path from a top-level domain down to it; `subject_closure` includes
        depth-0 self-pairs, so the matched subjects themselves are retained too.
        """
        matched_ids = self.resolution_subject_table["subject_id"].unique()
        used_ids = self.closure_table.loc[
            self.closure_table["descendant_id"].isin(matched_ids), "ancestor_id"
        ].unique()

        before = len(self.subject_table)
        self.subject_table = self.subject_table[self.subject_table["subject_id"].isin(used_ids)]
        self.closure_table = self.closure_table[self.closure_table["ancestor_id"].isin(used_ids)]
        self.broader_table = self.broader_table[self.broader_table["parent_id"].isin(used_ids)]
        self.logger.info(
            f"Pruned subject table from {before} to {len(self.subject_table)} subjects "
            f"reachable from the {len(matched_ids)} matched by a resolution"
        )

    def _log_memory_footprints(self):
        """Log the memory footprint of every table and array held in memory."""
        for name, table in (
            ("Resolution Table", self.resolution_table),
            ("Resolution Subject Table", self.resolution_subject_table),
            ("Subject Table", self.subject_table),
            ("Closure Table", self.closure_table),
            ("Broader Table", self.broader_table),
            ("Member States Table", self.member_states_table),
        ):
            self.logger.info(f"{name}: {table.memory_usage(index=True).sum() / (1024**2):.2f} MB")
        self.logger.info(
            f"Multilateral Scores: {self.multilateral_scores.nbytes / (1024**2):.2f} MB"
        )
        self.logger.info(
            f"Vote Bool Arrays: {sum(a.nbytes for a in self.vote_bool_arrays) / (1024**2):.2f} MB"
        )
