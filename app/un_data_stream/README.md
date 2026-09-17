# unDataStream

The app's data layer: loads the normalized UN resolution tables out of Postgres, precomputes the
vote-agreement arrays, and answers queries over them in memory.

## Overview

`policy-pulse` is a **read-only consumer** of a Postgres storage layer. Fetching from the UN
Digital Library, parsing MARC records and building the thesaurus tables all happen in the separate
[`undl-ingest`](https://github.com/unst-ethz) repo, whose jobs run on a schedule and write the
tables described below. This package never talks to UNDL.

This is a change from how it used to work: until 2026-09 the app fetched bulk CSV/TTL exports
itself on a cache miss, processed them, and wrote a local CSV + pickle cache — which meant a
multi-minute startup after every deploy. That pipeline (`fetchers/`, `core/`, `data/fetcher.py`,
`data/merger.py`, the thesaurus and SC processors) was removed once the storage layer landed; see
`plans/app_postgres_migration_plan.md` and `git log` if you need the old implementation.

### Key features

- 🐘 **Postgres-sourced**: 7 tables read once at startup (~25s for ~20.8k resolutions and ~950k
  votes), no local cache to invalidate
- 📊 **Precomputed agreement arrays**: per-resolution consensus scores plus a
  (resolutions × countries) alignment matrix, built once at load
- 🔍 **Vectorised querying**: date and subject-hierarchy filters, bilateral agreement, per-country
  multilateral statistics — all NumPy broadcasting, no per-row iteration
- 🌳 **Subject hierarchy**: UNBIS thesaurus closure/broader tables for ancestor/descendant filters

## Architecture

```
un_data_stream/
├── data/
│   ├── db.py          # Postgres connection + table reads (SQLAlchemy Core)
│   ├── repository.py   # Main entry point: read → reshape → precompute
│   ├── processor.py    # Vote-agreement precomputation
│   └── progress.py     # Progress bar for the precompute loop
└── analysis/
    └── query_engine.py # The query layer the app calls
```

## Usage

```python
from app.un_data_stream import DataRepository, ResolutionQueryEngine

repo = DataRepository(config_path="config/data_sources.yaml")
engine = ResolutionQueryEngine(repo=repo)

# All resolutions, or filtered by date and subject hierarchy
df = engine.query_resolutions(start_date="2000-01-01", end_date="2025-12-31")
df = engine.query_resolutions(subject_ids=["1006488"], include_descendants=True)

# Bilateral agreement: one row per resolution, or averaged across them
engine.query_agreement_between_countries("USA", average=True)

# Per-country alignment and vote-rate statistics
engine.query_multilateral_stats(resolution_ids=df["undl_id"].tolist())
```

In the app itself, don't construct these yourself — `app/data.py` builds one `DataRepository` and
one `ResolutionQueryEngine` at import time and treats them as read-only process-wide singletons.

## Configuration

Connection settings come from the environment (`PGHOST`, `PGPORT`, `PGDATABASE`, `PGUSER`,
`PGPASSWORD`); copy `.env.example` to `.env` for local development. `db.load_env()` validates that
they are all present and fails with a clear message if not.

`config/data_sources.yaml` now only carries runtime settings:

```yaml
debug: true   # DEBUG-level logging + console handler
logs: true    # false disables the logger entirely
paths:
  logs: "logs/"
```

## Tables read

| table | used for |
|---|---|
| `resolution_outcomes` | resolution metadata (one row per resolution) |
| `resolution_votes` | normalized long votes, pivoted to one column per country |
| `subject` | thesaurus labels, domains, node types |
| `subject_broader` | direct SKOS `broader` edges, for tree navigation |
| `subject_closure` | transitive ancestor/descendant pairs, for hierarchy filters |
| `resolution_subject` | which thesaurus concepts a resolution is about, for subject filters |
| `member_states` | ISO codes and multi-language country names |

Everything here is read as written by `undl-ingest`; the app derives nothing from raw source data.
Note that `subject`/`subject_broader`/`subject_closure` are **pruned at load** to the subjects
reachable from a `resolution_subject` row (627 of 7,341 today), so the filter UI only offers
subjects that can actually match something.

## Performance model

Vote columns are held as `pandas.CategoricalDtype` (Y/N/A/X), but the hot path never touches the
DataFrame's vote columns. `DataProcessor.calculate_agreement_data()` precomputes, once, a
`(resolutions × countries)` boolean array per vote type (`yes`/`no`/`abstained`/`voted`) plus a
float32 `multilateral_scores` matrix; every query-time agreement/alignment computation is
vectorised NumPy broadcasting over those. When touching `query_engine.py`, prefer array operations
over DataFrame loops.

One invariant worth knowing: `country_columns` is the positional index into those arrays, so its
order must match the wide frame's vote columns exactly. It is passed explicitly into
`calculate_agreement_data()` rather than inferred, precisely so the two can't drift.
