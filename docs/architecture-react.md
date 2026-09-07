# React and API architecture

This branch separates the browser application from the scientific calculation engine. The existing Dash UI remains available for migration comparisons; the production images run React and FastAPI.

```text
Browser (React + TypeScript)
        │ GET /api/v1/* — JSON / CSV
        ▼
Nginx (static files, SPA routes, same-origin API proxy)
        │
        ▼
FastAPI routes → validated request/response models → application services
        │
        ▼
Existing app/un_data_stream engine → immutable in-memory dataset snapshot
        │
        ▼
Persistent cache ← configured UN Digital Library sources
```

## Ownership and dependency direction

| Directory | Responsibility |
| --- | --- |
| `frontend/src/pages` | Overview, explorer, resolution detail, printable profile, methodology |
| `frontend/src/views` | Each independent analysis view |
| `frontend/src/components` | Shared filters, selection controls, resolution cards, charts, error/empty states |
| `frontend/src/api` | HTTP transport and generated OpenAPI types |
| `frontend/src/lib` | URL state and filter translation, including existing shared links |
| `backend/routes.py`, `models.py` | Versioned public HTTP contract and validation |
| `backend/service.py` | Resolution selection and presentation-ready aggregation over the existing engine |
| `backend/catalog.py`, `keywords.py` | Source names, subject hierarchy, existing keyword assets |
| `backend/provider.py`, `config.py` | Snapshot initialization, process lock, explicit configuration |
| `app/un_data_stream` | Shared scientific engine, processors and UN source ingestion; no frontend dependency |
| `app/pages`, `app/features` | Legacy Dash UI retained as a reference, excluded from production containers |
| `tests/api`, `tests/support` | Offline contracts, independent formula oracles, isolated synthetic browser fixtures |

The scientific package retains its original import path so existing notebooks, tests and the legacy application continue to work. It is shared Python domain code, not a web server. Neither the API nor its production dependencies import Dash. Country authority/name logic, geographic utilities, institutional presets and subject agreement calculation were extracted into that package and re-exported to the old UI.

## Frontend identity and map provenance

The React frontend reuses the original UN-ETH Policy Pulse name, navy header (`#1b3357`) with its cyan rule (`#3bc4ff`), IBM Plex Sans, welcome text, and pulse-and-globe illustration. The font is bundled locally. `frontend/public/policy-pulse.webp` is an unchanged copy of `app/assets/main.webp`. The home page retains recent updates, the statistics/word-cloud section, project background and case studies. Explorer filters sit above the tabbed analysis views as in the original interface.

The interactive map uses Natural Earth boundaries from the pinned `world-atlas` package. It is **not an official UN map**. Its source and geographic limitations are visible below the map, alongside a link to a UN-published reference map. A future move to official UN boundary data requires a verified dataset and explicit review of geographic coverage and disputed boundaries; changing the projection does not change provenance.

The map restores the original Robinson projection, reference-country longitude, RdYlBu scale, purple reference country and grey missing-data states. Its optional yellow midpoint uses the same legacy mean of resolution consensus after excluding missing reference-country vote codes (recorded X codes remain in that selection). This setting changes colours only. The API supplies the midpoint and longitude; agreement scores and denominators remain unchanged. A degenerate or unavailable midpoint disables the optional scale.

## Scientific invariants

`DataProcessor`'s scoring methods and `ResolutionQueryEngine` are unchanged. The HTTP layer uses those implementations directly. It does not implement agreement scoring in JavaScript.

- Y = +1; A = 0; N = −1. Agreement is `1 - abs(a - b) / 2`.
- X and missing observations do not contribute to agreement or vote-rate denominators. Missing scores are JSON `null` and display as “—”; observed zero remains zero.
- Consensus averages unique country pairs, excluding self-comparisons.
- Multilateral alignment averages a country's per-resolution means across valid resolutions, rather than pooling all pairs across history.
- Timeline: full history, mean by session, median-date year, minimum **3** shared votes. Special/emergency sessions are optional and off by default.
- Subject comparison: selected date range, descendant expansion, unique resolutions per subject, minimum **30** shared votes.
- Multilateral scatter: minimum **10** votes cast. Profile bilateral rankings: minimum **100** shared votes.
- Membership filtering keeps the existing widest interval. Profile dates use its year boundaries. Gaps in multi-period membership remain an explicitly documented limitation.
- Legacy resolution-list “same/different recorded vote” filters include X after excluding missing data. They are labeled as recorded-code comparisons, distinct from numerical agreement.
- Existing keyword files, OR/AND/quoted search semantics, fuzzy cutoff and subject label indexing are retained. Missing keyword coverage stays missing.

The machine-readable explanations are in `backend/methodology.py` and served at `/api/v1/methodology`. Any future methodological change needs a separately reviewed version change, updated explanations, and independent expected-value tests.

## API behavior

The public API is read-only and namespaced under `/api/v1`. It provides metadata, methodology, overview, paginated resolutions, full filtered CSV export, resolution detail, bilateral agreement, timelines, subjects, multilateral statistics, word frequencies and printable profile data. Interactive documentation is at `/api/docs`; the OpenAPI document is `/api/openapi.json`.

- Selection failures and invalid dates/pagination return **422**. Missing resolutions return **404**.
- Loading or failed ingestion returns **503**, with a retry hint. `/health/live` and methodology remain available. `/health/ready` only succeeds once a complete snapshot is available.
- Empty filter results stay empty. In particular, the adapter never passes an empty selection to the legacy multilateral method, whose empty-list convention means “all records.”
- Page size is capped at 100. Country and subject selections are validated and bounded. Sorting has a deterministic ID tie-breaker.
- CSV uses the same filter/sort service as the list, ignores pagination and escapes spreadsheet formula prefixes. Score values are not rounded by the API.
- Requests receive an ID; diagnostic details stay in server logs. CORS is disabled by default because the browser uses the same-origin proxy. Separate deployments can explicitly allow origins.

Run `make schema` after changing a response model. Commit both `backend/openapi.json` and `frontend/src/api/schema.ts`. CI regenerates them and fails if they differ.

## Reliability and scaling

One API worker loads one snapshot. Cache initialization is protected by a file lock across workers. The completion marker is removed before a cache write so interrupted writes cannot masquerade as a complete snapshot. Legacy uncompressed caches can be read without writing through a read-only file handle. HTTP ingestion has bounded timeouts and rejects empty/HTML/challenge responses.

The reference deployment uses one worker with bounded concurrency. Scale by running measured replicas with enough memory for each snapshot. Do not assume the previous Gunicorn copy-on-write memory behavior applies to ASGI workers. For larger datasets, the service boundary allows future database/columnar storage work without changing the frontend contract.

Persistent volumes prevent refetching data on every restart. Prepare a replacement snapshot in a separate directory, validate it, then roll out both images and that snapshot together. Keep the old images and snapshot for rollback. There is no unauthenticated administrative refresh endpoint.

## Verification and remaining release gate

The offline suite runs without UN availability: existing matrix tests, all vote-pair combinations across the API, independent raw-vote averages/denominators, subject thresholds, timeline aggregation, profiles, keyword parity, source response validation, cache failure cases, frontend component tests and browser journeys.

Browser and container tests use visibly named **synthetic test resolutions**. Production does not have a demo-mode switch or fallback to these fixtures. `compose.test.yaml` is only for isolated test containers.

Live-source tests remain explicit: `POLICY_PULSE_LIVE_TESTS=1 .venv/bin/python -m pytest tests/integration tests/e2e -q`. Do not count skipped live tests as passed. On 2026-09-07 the official file-list API was reachable and confirmed renamed thesaurus/member-state files, but the file downloads returned empty HTTP 202 responses from the development environment. A real-data smoke test from the deployment environment or a trusted existing cache is still required before traffic cutover.

Local verification on 2026-09-07: **90 Python tests passed, 34 live tests skipped; 13 frontend tests and 9 browser journeys passed**. A fresh installation from the dependency lockfiles was also verified. Type checking, lint, formatting and production frontend builds passed. Desktop and mobile screenshots were reviewed after restoring the original frontend identity. Both production containers built and became healthy; the isolated synthetic-cache smoke test verified every analysis endpoint through Nginx, pagination and full CSV export, invalid-date responses, SPA routes and hashed-asset handling. The production API runtime was also checked for its non-root user and absence of Dash/test fixtures. These checks validate the implementation and deployment packaging; they do not replace the real-data release gate above.
