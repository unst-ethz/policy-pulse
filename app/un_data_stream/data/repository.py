"""
Data repository for the app's in-memory copy of the storage layer.

Reads the collections out of MongoDB (written by the `undl-ingest` jobs), flattens them into the
frames the old Postgres tables were, reshapes those into the shapes the app's query layer expects,
and precomputes the agreement/alignment arrays `ResolutionQueryEngine` broadcasts over. The app is
a read-only consumer — it does not fetch from UNDL, and it does not process raw source data.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple
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
# `session` is the one real rename: the store keeps a numeric `session` plus a `session_label`
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

# `resolution_sponsor`: one row per sponsor of a resolution, or one placeholder row (sponsor NULL,
# `note` saying why) when the draft could not be read. NULL sponsor means "unknown", never
# "unsponsored". States always carry a `country_code` matching the vote columns.
SPONSOR_COLUMNS = ["undl_id", "sponsor", "country_code", "sponsor_type", "sponsor_kind", "note"]

# Spellings of one non-state sponsor as catalogued over the years -> the name we show.
_SPONSOR_ALIASES = {"palestine": "State of Palestine", "state of palestine": "State of Palestine"}

# The `resolutions` fields one load reads: the outcome columns (`_id` is `undl_id`), the votes
# and the three derived arrays. The keyword run bookkeeping stays on the server.
RESOLUTION_PROJECTION = [
    *(c for c in RESOLUTION_COLUMNS.values() if c != "undl_id"),
    "votes",
    "derived.subject_ids",
    "derived.keywords",
    "derived.sponsors",
]

# The old `subject` and `member_states` table columns, key first. `subjects` also carries an
# `inserted_at` the table never had; it stays out so the frame keeps its columns.
SUBJECT_COLUMNS = [
    "subject_id",
    "record_id",
    *(
        f"{kind}_{lang}"
        for kind in ("label", "alt_labels")
        for lang in ("en", "es", "fr", "ar", "ru", "zh")
    ),
    "domain_code",
    "node_type",
    "source_updated_at",
]
SUBJECT_PROJECTION = [*SUBJECT_COLUMNS[1:], "broader", "ancestors"]
MEMBER_STATE_COLUMNS = [
    "record_id",
    "iso_code",
    "m49_code",
    "record_type",
    "name_en",
    *(f"name_{lang}" for lang in ("fr", "es", "ar", "zh", "ru")),
    "other_names",
    "coverage_periods",
    "founding_member",
    "membership_resolution",
    "scope_note",
    "earlier_names",
    "later_names",
    "subject_xref_id",
    "source_updated_at",
    "inserted_at",
]

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


def clean_sponsors(raw: pd.DataFrame) -> pd.DataFrame:
    """Tidy `resolution_sponsor` names: collapse whitespace, drop a leading "the ", merge aliases.

    Only the display name changes. States are identified by `country_code` anyway; this matters
    for observers and groups, which have no code (e.g. "Palestine" vs "State of Palestine").
    """
    sponsors = raw[SPONSOR_COLUMNS].copy()
    names = sponsors["sponsor"].str.replace(r"\s+", " ", regex=True).str.strip()
    names = names.str.replace(r"^the ", "", case=False, regex=True)
    sponsors["sponsor"] = names.map(
        lambda n: _SPONSOR_ALIASES.get(n.lower(), n), na_action="ignore"
    )
    return sponsors


def sponsor_counts(sponsors: pd.DataFrame) -> pd.Series:
    """undl_id -> number of sponsors; resolutions without sponsorship data are absent."""
    return sponsors.dropna(subset=["sponsor"]).groupby("undl_id").size()


class DataRepository:
    """Loads and holds the app's processed UN data, sourced from MongoDB."""

    def __init__(self, config_path: str):
        self.config_path = config_path

        # Initialize data attributes
        self.resolution_table: pd.DataFrame
        self.resolution_subject_table: pd.DataFrame
        self.keyword_table: pd.DataFrame
        self.sponsor_table: pd.DataFrame
        self.subject_table: pd.DataFrame
        self.closure_table: pd.DataFrame
        self.broader_table: pd.DataFrame
        self.member_states_table: pd.DataFrame
        # One row per country code that has ever cast a vote: first_vote, last_vote.
        self.voting_activity_table: pd.DataFrame
        # MAX(completed_at) of successful ingestion runs as of this load; the reloader compares
        # the live value against it. None when nothing has ever been ingested successfully.
        self.source_marker: datetime | None = None

        # Load configuration
        self._load_config()

        # Initialize Logging
        self._setup_logging()

        self.logger.info("Initializing UNDataRepository from MongoDB")

        self._load()

        self.logger.info("Initialization complete. Below are memory footprints:")
        self._log_memory_footprints()

    def get_data(self) -> Dict[str, Any]:
        """Return all processed data as a dict consumed by ResolutionQueryEngine.

        Keys:
            resolution, resolution_subject, resolution_keyword, resolution_sponsor, subject,
            closure, broader, member_states, voting_activity
                                — pd.DataFrames
            country_columns     — List[str] of ISO3 country codes (length C)
            multilateral_scores — (R x C) np.ndarray, float32
            vote_bool_arrays    — 4-tuple of (R x C) bool arrays (yes, no, abstained, voted)
        """
        return {
            "resolution": self.resolution_table,
            "resolution_subject": self.resolution_subject_table,
            "resolution_keyword": self.keyword_table,
            "resolution_sponsor": self.sponsor_table,
            "subject": self.subject_table,
            "closure": self.closure_table,
            "broader": self.broader_table,
            "country_columns": self.country_columns,
            "member_states": self.member_states_table,
            "voting_activity": self.voting_activity_table,
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

    def _load(self):
        """Read every collection the app needs, then derive the in-memory representation.

        The client is opened, drained and closed inside this method. Nothing holds a live
        connection afterwards: this runs at import time, which under gunicorn's `--preload` is the
        master process before it forks, and a client that survived the fork would be shared
        across workers. See `db.connect`.
        """
        db.load_env()
        self.logger.info("Reading collections from MongoDB")
        with db.connect_or_explain(self.logger) as client:
            database = db.database(client)
            # Read the freshness marker *before* the data, never after. If an ingestion run
            # completes while we are reading, a marker taken first is older than the data we
            # got, so the reloader simply rebuilds once more — harmless. A marker taken last
            # could be newer than the data, and the reloader would then never rebuild, leaving
            # this worker permanently behind.
            self.source_marker = db.read_success_marker(database)
            frames = self._frames_from_documents(
                database[db.RESOLUTIONS].find({}, RESOLUTION_PROJECTION),
                database[db.SUBJECTS].find({}, SUBJECT_PROJECTION),
                database[db.MEMBER_STATES].find({}, MEMBER_STATE_COLUMNS[1:]),
            )
        outcomes = frames["outcomes"]
        votes = frames["votes"]
        self.resolution_subject_table = frames["resolution_subject"]
        self.keyword_table = frames["keyword"]
        self.subject_table = frames["subject"]
        self.closure_table = frames["closure"]
        self.broader_table = frames["broader"]
        self.member_states_table = frames["member_states"]
        raw_sponsors = frames["sponsor"]

        self.voting_activity_table = self._build_voting_activity(outcomes, votes)
        self.sponsor_table = clean_sponsors(raw_sponsors)

        self.logger.info(f"Data ingested up to {self.source_marker}")
        self.logger.info(
            f"Read {len(outcomes)} resolutions, {len(votes)} votes, "
            f"{len(self.subject_table)} subjects, "
            f"{len(self.resolution_subject_table)} resolution-subject pairs, "
            f"{len(self.keyword_table)} keyword rows, "
            f"{len(self.sponsor_table)} sponsor rows, "
            f"{len(self.member_states_table)} member states, "
            f"{len(self.voting_activity_table)} countries with voting activity"
        )

        self.resolution_table, self.country_columns = self._build_resolution_table(outcomes, votes)
        # NaN where the sponsorship is unknown, so "no data" never reads as "no sponsors".
        self.resolution_table["sponsor_count"] = self.resolution_table["undl_id"].map(
            sponsor_counts(self.sponsor_table)
        )

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
    def _frames_from_documents(
        resolutions: Iterable[dict], subjects: Iterable[dict], member_states: Iterable[dict]
    ) -> Dict[str, pd.DataFrame]:
        """Flatten the three collections into the nine frames the Postgres tables used to be.

        Same columns as the old tables (and the `resolution_keyword_cloud` view); `_id` goes back
        to its column name. Each cursor is drained exactly once.
        """
        resolutions = list(resolutions)
        subjects = list(subjects)

        outcomes = pd.DataFrame(
            [{**d, "undl_id": d["_id"]} for d in resolutions],
            columns=list(RESOLUTION_COLUMNS.values()),
        )
        # BSON has no date-only type, so `date` is stored as midnight UTC; Postgres's DATE read
        # back as a naive datetime64, which is what the features compare against.
        outcomes["date"] = pd.to_datetime(outcomes["date"], utc=True).dt.tz_localize(None)

        votes = pd.DataFrame(
            [(d["_id"], c, v) for d in resolutions for c, v in d.get("votes", {}).items()],
            columns=VOTE_COLUMNS,
        )
        resolution_subject = pd.DataFrame(
            [
                (d["_id"], s)
                for d in resolutions
                for s in d.get("derived", {}).get("subject_ids", [])
            ],
            columns=["undl_id", "subject_id"],
        )
        keywords = pd.DataFrame(
            [
                (d["_id"], dimension, keyword)
                for d in resolutions
                for dimension, words in d.get("derived", {}).get("keywords", {}).items()
                for keyword in words
            ],
            columns=["undl_id", "dimension", "keyword"],
        )
        # 'general' is the union of the three dimensions, as the old view's UNION built it.
        general = keywords.drop_duplicates(["undl_id", "keyword"]).assign(dimension="general")
        keyword_table = pd.concat([keywords, general]).drop_duplicates(ignore_index=True)

        subject_table = pd.DataFrame(
            [{**d, "subject_id": d["_id"]} for d in subjects], columns=SUBJECT_COLUMNS
        )
        closure = pd.DataFrame(
            [(a["id"], d["_id"], a["depth"]) for d in subjects for a in d.get("ancestors", [])],
            columns=["ancestor_id", "descendant_id", "depth"],
        )
        broader = pd.DataFrame(
            [(p, d["_id"]) for d in subjects for p in d.get("broader", [])],
            columns=["parent_id", "child_id"],
        )
        member_states_table = pd.DataFrame(
            [{**d, "record_id": d["_id"]} for d in member_states], columns=MEMBER_STATE_COLUMNS
        )
        # `.get` per field, so an absent field reads as None exactly like a null one (and like
        # the NULLs the old table gave), rather than as the NaN a missing dict key would become.
        sponsors = pd.DataFrame(
            [
                (d["_id"], *(s.get(c) for c in SPONSOR_COLUMNS[1:]))
                for d in resolutions
                for s in d.get("derived", {}).get("sponsors") or []
            ],
            columns=SPONSOR_COLUMNS,
        )

        # PyMongo's tz-aware datetimes carry bson's own UTC tzinfo; pandas keeps it as a distinct
        # FixedOffset dtype unless told it is plain UTC, which is what the Postgres read gave.
        subject_table["source_updated_at"] = pd.to_datetime(
            subject_table["source_updated_at"], utc=True
        )
        for column in ("source_updated_at", "inserted_at"):
            member_states_table[column] = pd.to_datetime(member_states_table[column], utc=True)

        return {
            "outcomes": outcomes,
            "votes": votes,
            "resolution_subject": resolution_subject,
            "keyword": keyword_table,
            "subject": subject_table,
            "closure": closure,
            "broader": broader,
            "member_states": member_states_table,
            "sponsor": sponsors,
        }

    @staticmethod
    def _build_voting_activity(outcomes: pd.DataFrame, votes: pd.DataFrame) -> pd.DataFrame:
        """First and last date each country code actually cast a vote."""
        cast = votes.loc[votes["vote"] != "X", ["undl_id", "country_code"]]
        dated = cast.merge(outcomes[["undl_id", "date"]], on="undl_id")
        dated["date"] = pd.to_datetime(dated["date"])
        return (
            dated.groupby("country_code")["date"]
            .agg(first_vote="min", last_vote="max")
            .reset_index()
        )

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
        """Log the memory footprint of every table and array held in memory.

        `deep=True` because every frame here is string-heavy and the shallow measure counts only
        the pointer array — it reports single-digit MB for tables several times that, which
        defeats the point of a log that exists to catch memory regressions.
        """
        for name, table in (
            ("Resolution Table", self.resolution_table),
            ("Resolution Subject Table", self.resolution_subject_table),
            ("Keyword Table", self.keyword_table),
            ("Sponsor Table", self.sponsor_table),
            ("Subject Table", self.subject_table),
            ("Closure Table", self.closure_table),
            ("Broader Table", self.broader_table),
            ("Member States Table", self.member_states_table),
            ("Voting Activity Table", self.voting_activity_table),
        ):
            size = table.memory_usage(index=True, deep=True).sum()
            self.logger.info(f"{name}: {size / (1024**2):.2f} MB")
        self.logger.info(
            f"Multilateral Scores: {self.multilateral_scores.nbytes / (1024**2):.2f} MB"
        )
        self.logger.info(
            f"Vote Bool Arrays: {sum(a.nbytes for a in self.vote_bool_arrays) / (1024**2):.2f} MB"
        )
