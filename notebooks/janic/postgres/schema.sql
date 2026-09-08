-- Prototype schema for the intermediate storage layer.
-- Source of truth: plans/intermediate_storage_layer_plan.md ("Postgres Schema (voting data)").
-- Subject-hierarchy tables (subject/subject_broader/subject_closure/resolution_subject) are
-- intentionally left out for now -- out of scope for this ingestion prototype.

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
    undl_link          TEXT,
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
