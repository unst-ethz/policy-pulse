-- Prototype schema for the intermediate storage layer.
-- Source of truth: plans/intermediate_storage_layer_plan.md ("Postgres Schema (voting data)",
-- "Postgres Schema (member states)", "Postgres Schema (subject hierarchy)").
-- resolution_subject is intentionally left out for now -- no notebook produces resolution-to-
-- subject matches yet (needs ingestion_prototype.py's and thesaurus_ingestion_prototype.py's
-- output together), out of scope for this ingestion prototype.

CREATE TABLE resolution_outcomes (
    undl_id           TEXT PRIMARY KEY,      -- UNDL control number
    source_dataset    TEXT NOT NULL,         -- 'GA' | 'SC' | 'HRC' | ...
    resolution        TEXT,                  -- e.g. 'A/RES/78/1'
    date              DATE NOT NULL,
    modality          TEXT NOT NULL,         -- derived, not raw MARC -- see plan's "modality
                                              -- derivation" decision: 'Vote, recorded' | 'Vote, non-
                                              -- recorded' | 'Without a vote'
    draft              TEXT,
    meeting            TEXT,
    subjects           TEXT,                 -- raw `991.d` values joined for reference/QA display;
                                              -- resolution_subject (not created here) is what's
                                              -- actually queried once the subject tables exist
    vote_note          TEXT,
    total_yes          INTEGER,
    total_no           INTEGER,
    total_abstentions  INTEGER,
    total_non_voting   INTEGER,
    total_ms           INTEGER,
    -- undl_link intentionally not stored -- https://digitallibrary.un.org/record/{undl_id},
    -- reconstructed on the fly, see plan doc's "Derived URL columns" decision
    -- GA-specific (null for other bodies)
    title              TEXT,
    session            INTEGER,
    committee_report   TEXT,
    amended_draft      TEXT,
    related_documents  TEXT,
    agenda_title       TEXT,
    -- SC-specific (null for GA) -- field names TBD once SC ingestion is actually designed
    description        TEXT,
    agenda             TEXT,
    source_updated_at  TIMESTAMPTZ NOT NULL, -- from the API's own `updated` field
    inserted_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE resolution_votes (
    undl_id      TEXT NOT NULL REFERENCES resolution_outcomes(undl_id),
    country_code TEXT NOT NULL,
    vote         TEXT NOT NULL CHECK (vote IN ('Y','N','A','X')),
    inserted_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (undl_id, country_code)
);

CREATE TABLE member_states (
    record_id           TEXT PRIMARY KEY,      -- MARC auth record id
    iso_code             TEXT NOT NULL,         -- ISO alpha-3 -- join key everywhere else in the app
    m49_code              TEXT,
    record_type           TEXT NOT NULL,        -- 'ms' = current member state | 'fs' = former name
    name_en TEXT NOT NULL, name_fr TEXT, name_es TEXT, name_ar TEXT, name_zh TEXT, name_ru TEXT,
    other_names           TEXT,                 -- semicolon-joined variant spellings/abbreviations (410)
    coverage_periods      TEXT,                 -- 'start:end' pairs joined with '|', blank end = ongoing
    founding_member        BOOLEAN NOT NULL,
    membership_resolution TEXT,                 -- admitting GA resolution symbol, NULL for founding members
    scope_note             TEXT,                 -- historical/succession context (680), '|'-joined if repeated
    earlier_names          TEXT,                 -- semicolon-joined 510 $w=a targets
    later_names            TEXT,                 -- semicolon-joined 510 $w=b targets
    subject_xref_id        BIGINT,               -- raw MARC record id from 550 (thesaurus concept
                                                  -- xref) -- NOT resolved to subject.subject_id and no
                                                  -- FK yet, see plan doc's "subject_xref_id" decision
    -- unms_ontology_link intentionally not stored -- http://metadata.un.org/UNMS/{m49_code},
    -- 'ms' rows only, reconstructed on the fly, see plan doc's "Derived URL columns" decision
    source_updated_at      TIMESTAMPTZ NOT NULL,
    inserted_at             TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE subject (
    subject_id     TEXT PRIMARY KEY,      -- thesaurus URI's trailing id only, e.g. '1006488' or
                                           -- '170400' -- full URI = 'http://metadata.un.org/
                                           -- thesaurus/' || subject_id, reconstructed on the fly
    record_id      TEXT,                  -- MARC auth record id; NULL for Skosmos-bootstrap rows
    label_en TEXT, label_es TEXT, label_fr TEXT, label_ar TEXT, label_ru TEXT, label_zh TEXT,
    alt_labels_en TEXT, alt_labels_es TEXT, alt_labels_fr TEXT, alt_labels_ar TEXT,
    alt_labels_ru TEXT, alt_labels_zh TEXT, -- only alt_labels_en is populated today, see plan doc
    domain_code    TEXT,                  -- 072 hierarchical domain code(s), ';'-joined if >1
    node_type      TEXT NOT NULL,         -- 'concept' | 'scheme' | 'micro_thesaurus' | 'root'
    source_updated_at TIMESTAMPTZ         -- NULL for Skosmos-bootstrap rows
);

CREATE TABLE subject_broader (       -- direct SKOS `broader` edges only, for tree navigation
    parent_id TEXT NOT NULL REFERENCES subject(subject_id),
    child_id  TEXT NOT NULL REFERENCES subject(subject_id),
    PRIMARY KEY (parent_id, child_id)
);

CREATE TABLE subject_closure (       -- full transitive ancestor/descendant pairs, for hierarchy filters
    ancestor_id   TEXT NOT NULL REFERENCES subject(subject_id),
    descendant_id TEXT NOT NULL REFERENCES subject(subject_id),
    depth         INTEGER NOT NULL,
    PRIMARY KEY (ancestor_id, descendant_id)
);

CREATE TABLE ingestion_runs (
    id                   SERIAL PRIMARY KEY,
    source_dataset       TEXT NOT NULL,      -- 'GA' | 'SC' | 'HRC' | 'thesaurus' | ...
    started_at           TIMESTAMPTZ NOT NULL,
    completed_at         TIMESTAMPTZ,
    status               TEXT NOT NULL,      -- 'success' | 'failed' | 'partial'
    resolutions_upserted INTEGER,
    votes_upserted       INTEGER,
    issues_count         INTEGER DEFAULT 0,
    issues_detail        JSONB                -- e.g. name-mismatch / vote-count-mismatch findings
);
