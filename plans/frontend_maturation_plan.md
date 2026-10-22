# From Dash prototype to a mature prototype (Vue 3 frontend)

Status: Phase 1 done (branch `feature/api-phase1`, 2026-10-22); Phase 2 next.

## 1. Where we are

**Keep — this is the asset:**
- The data layer. `undl-ingest` fills Postgres; the app reads it into an immutable `DataSnapshot`
  that is swapped atomically on reload; `ResolutionQueryEngine` answers agreement/alignment
  queries vectorised in NumPy; the data layer has unit + integration tests. None of it depends on
  Dash.
- The analytical ideas, already validated: agreement map, agreement timeline, agreement by subject,
  multilateral scatter, word cloud, country profile, case studies.
- One-command local setup (`init-env`) against a snapshot of production data.

**What holds it back:**
1. *UI, data loading and computation live in one process.* Every filter change is a server
   round-trip through Dash callbacks, and ~6k of ~9.6k lines are UI code with analytics embedded
   in it (e.g. the session-agreement timeline in `trends_page.py` returns either an `html.Div` or a
   JSON string).
2. *One page does too much.* `/trends` shares one filter panel across 6 tabs and needs callbacks to
   grey out filters and tabs depending on which tab is active (`_set_filter_disabled_states`,
   `grey_out_tabs_without_country`). That is an information-architecture smell: users can't tell
   why a control is disabled or what to do about it.
3. *No navigation.* The navbar links only to home; the breadcrumb is commented out; the profile
   page is only reachable through a link built from the current filters.
4. *No design system.* `dcc`, `feffery-antd` and raw `html` components are mixed; ~200 inline
   `style={}` dicts; colours hard-coded in Python; inconsistent loading/empty/error states.
5. *The six UN languages are unused.* Country names and subject labels exist in en/fr/es/ar/zh/ru,
   the UI is English only.
6. *Domain content needs review:* e.g. "African Group (Representative)" presets are five hand-picked
   countries; there is no methodology page for the agreement/consensus scores.
7. *Operational gaps:* CI only builds the image (no tests/lint gate), no HTTPS, site goes down on
   every deploy, `dash==4.0.0rc4` (a release candidate) in production, stale `app/requirements.txt`.

## 2. Target architecture

```
undl-ingest ──► Postgres ──► API  (FastAPI + existing un_data_stream, in-memory snapshot)
                                 │  OpenAPI ──► generated TypeScript client
                                 ▼
                             web/ (Vue 3 + Vite + TS, static files behind nginx)
```

**Backend:** FastAPI around the existing snapshot/query engine (keeps gunicorn `--preload` and the
reloader). Endpoints return **data, not Plotly figures**. Pydantic models generate the OpenAPI
schema; `openapi-typescript` turns it into a typed client so the contract can't drift silently.
ETag keyed on the data version, so browsers/nginx can cache.

**Frontend (Vue 3):**

| Concern | Choice | Why |
|---|---|---|
| Build / language | Vite + TypeScript | standard |
| Routing / state | Vue Router; **filters live in the URL**; Pinia only for UI state | every view shareable |
| Data fetching | TanStack Query (Vue) | caching, loading/error states, request cancellation |
| Components | PrimeVue (or the partners' UN component library if one is mandated) | TreeSelect for the thesaurus, virtualised DataTable, accessible MultiSelect |
| Charts | ECharts via `vue-echarts`; D3 for bespoke pieces | maps, graphs, scatter, time series, word cloud extension, SVG export |
| i18n | `vue-i18n`, 6 UN languages incl. RTL for Arabic | data already supports it |

## 3. Information architecture & UX

Entity-centric pages, each with only the filters that apply to it:

| Page | Content |
|---|---|
| Home | recent resolutions, key stats, entry points ("pick a country", "pick a topic") |
| Resolutions | searchable/filterable table → **resolution detail page** (vote map, breakdown, subjects, UNDL link) — new |
| Country | the profile grown into a hub: voting record, agreement map + ranking, agreement over time, by subject |
| Compare | 2..n countries or groups (groups from M49 / actual UN regional groups) |
| Topics | thesaurus subjects + keywords: trends, word cloud, divisiveness (consensus) |
| Methodology & data | definitions of agreement/consensus scores, sources, coverage, last update |

Cross-cutting: active-filter chips; explain unavailable views instead of greying them out;
skeleton/empty/error states; per chart a "how to read this", export (SVG/PNG/CSV), a link to the
underlying resolutions and an accessible table alternative; WCAG 2.1 AA; responsive layouts; a
"data as of …" stamp.

**Extensibility:** a visualization is *API endpoint + Vue component + declared filters +
methodology text*, registered in one place, so new analyses (Security Council votes/vetoes, voting
bloc clustering, topic divisiveness over time, sponsorship) don't require rewiring.

## 4. Engineering maturity

- CI on every PR: ruff + pytest (API), ESLint + `vue-tsc` + Vitest (web), Playwright for 3–5
  critical flows; deploy only from green builds.
- Deployment: HTTPS, separate API and web containers, health-checked rollout, explicit
  `main` → staging and `stable` → production.
- Error tracking (Sentry / self-hosted GlitchTip).
- Hygiene: remove `app/requirements.txt`, fill in the `pyproject` description, move committed
  data dumps out of `notebooks/`.

## 5. Phases

1. **API first** — extract the analytics out of the Dash callbacks into framework-free functions,
   put FastAPI on top, test it. Dash keeps working the whole time and calls the same functions, so
   it doubles as a parity oracle.
2. **Vue skeleton** — design tokens, layout, navigation, i18n, URL-backed filter model, CI.
3. **Port pages by value** — Resolutions + detail → Country → Compare → Topics → Home, each with
   a parity check against Dash.
4. **Switch over** — retire Dash (delete `app/features`, `app/pages`, `app/assets`,
   `app/__main__.py`), methodology page, accessibility pass, then new visualizations.

Open questions for the UN partners: primary audience (delegates, analysts, researchers, public);
mandated design system / style guide; required languages at launch; hosting (ETH VM vs UN infra);
any need for authentication.

---

## 6. Phase 1 — detailed tasks

Principle: **one implementation per computation.** Every number the API serves is produced by a
Dash-free function in `app/un_data_stream/analysis/`, and the Dash feature that showed it before
now calls that same function. Behaviour is pinned by comparing the refactored Dash paths against
outputs captured from the pre-refactor code.

Layout decision: the API lives in `app/api/` rather than a new top-level package. It reuses the
installed `app` package, the Dockerfile already copies `app/`, and once Dash is retired (phase 4)
`app/` simply *is* the backend. `app/data.py` stays the reload-aware reference-data module (names,
membership periods, trees); it has no Dash dependency.

| # | Task | Status |
|---|---|---|
| T1 | `analysis/keywords.py`: keyword/subject indices + keyword-expression matching (from `wordcloud_interactive`), cached per snapshot instead of via `invalidate()` hooks | done |
| T2 | `analysis/filtering.py`: `ResolutionFilter` + `filter_resolutions()` (from `filtered_resolutions` / CSV export) | done |
| T3 | `analysis/agreement.py`: agreement map, session timeline (from `trends_page`), agreement by subject | done |
| T4 | `analysis/word_cloud.py`: ranked cloud terms with searchable counts + consensus (from `_build_wordcloud`, minus layout/colours) | done |
| T5 | `analysis/profile.py` + `analysis/overview.py`: country profile, dataset stats, recent resolutions | done |
| T6 | Dash call sites switched to T1–T5; parity verified against pre-refactor outputs | done |
| T7 | FastAPI app (`app/api/`): schemas, filter dependency, `lang`, error handling, CORS, ETag/data-version, reloader in lifespan | done |
| T8 | Endpoints (see below) | done |
| T9 | Tests: unit tests for T1–T5 on synthetic data, API integration tests against Postgres | done |
| T10 | Run scripts (`uv run start-api`), Docker command, README / deployment docs | done |

### API surface (v1)

All under `/api/v1`. Shared filter query params: `start_date`, `end_date` (YYYY-MM-DD),
`subject` (repeatable), `country`, `country_mode` (`voted` | `member` | `none`), `keyword`
(`,` = OR, `&` = AND, `"…"` = exact). `lang` selects names/labels (en, fr, es, ar, zh, ru).

| Endpoint | Replaces |
|---|---|
| `GET /meta` | data date range, counts, data version / last ingestion |
| `GET /countries`, `GET /countries/{iso3}` | country dropdown, membership/participation ranges |
| `GET /regions/tree`, `GET /subjects/tree` | the two AntD tree selects |
| `GET /resolutions` (paginated, sortable), `GET /resolutions.csv` | resolution list, CSV download |
| `GET /resolutions/recent` | home page "recent updates" |
| `GET /resolutions/{undl_id}` | (new) resolution detail incl. every country's vote |
| `GET /stats/overview` | home page "at a quick glance" |
| `GET /agreement/map` | agreement choropleth |
| `GET /agreement/timeline` | agreement timeline |
| `GET /agreement/by-subject` | agreement by subject |
| `GET /multilateral` | multilateral scatter |
| `GET /keywords/cloud` | word cloud |
| `GET /countries/{iso3}/profile` | country profile page |

### Phase 1 outcome (2026-10-22)

- `app/un_data_stream/analysis/` gained `filtering`, `keywords`, `agreement`, `word_cloud`,
  `profile`, `overview`. The Dash features now call these: `app/features` + `app/pages` lost ~980
  lines and gained ~115 (`wordcloud_interactive.py` alone shrank by ~500).
- Parity: 130 outputs of the Dash code paths (filtered sets per tab, figures, profile pages, CSV
  exports, resolution lists, timeline data, …) were captured before the refactor and are
  byte-identical after it.
- The keyword index is cached per snapshot (identity), so the word cloud's `invalidate()` hook is
  gone — a reload can't leave it stale.
- `app/api/`: 16 endpoints, OpenAPI at `/api/openapi.json`, docs at `/api/docs`, ETag/304 keyed
  on the data version, gzip, CORS for the Vite dev server, reloader started per worker in the
  lifespan. Verified under the production command (gunicorn + uvicorn workers + `--preload`).
- Tests: 120 → 228. Unit tests for every analysis module on a hand-checkable synthetic world
  (`tests/unit/conftest.py`), API integration tests (`tests/integration/test_api.py`).

### Behaviour found during extraction (kept as-is; needs a decision)

Phase 1 deliberately preserved behaviour, so these are reproduced by both Dash and the API:

1. **"Agreed" pair filter counts two non-votes as agreement.** `filter_by_pair_agreement` compares
   raw codes, so X/X matches — which includes every resolution adopted without a vote (~72% of
   the table). The resolution list's "Agreed" option is inflated by these.
2. **Profile "did not vote" includes resolutions adopted without a vote.** E.g. Switzerland since
   2002: 5,994 of 8,109 "absent". Should probably be computed over recorded votes only, with
   non-recorded adoptions shown separately.
3. **Home stats card mislabel.** "10,631 resolutions with assigned subjects" is the number of
   (resolution, subject) links; 9,692 resolutions have a subject. The API returns both numbers.
4. **Choropleth consensus scale.** The code means to anchor the scale on resolutions where the
   main country voted, but its condition (`country1 in all_resolutions.columns`) is never true, so
   it uses all filtered resolutions.
5. **CSV export ignores the country participation filter** while the list it sits under applies it.
6. Agreement-by-subject row order depended on hash seed (the UI sorts afterwards, so invisible);
   the shared function now sorts by subject id.

### Phase 2 — first steps

- Scaffold `web/` (Vite + Vue 3 + TS, Vue Router, Pinia, TanStack Query, PrimeVue, vue-i18n,
  ECharts), dev proxy `/api` → `:8000`.
- Generate the TS client from `/api/openapi.json` (`openapi-typescript`), checked in CI.
- URL-backed filter model mirroring the API's filter parameters.
- CI workflow: ruff + pytest (unit always, integration against the snapshot image), and lint /
  type-check / test for `web/`.
- Decide items 1–5 above with the team before porting the affected views.
