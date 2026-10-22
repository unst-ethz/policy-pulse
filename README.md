# Policy Pulse

Policy Pulse is a project by the UN-ETH Student Team on creating a powerful interactive data analysis platform for UN voting records.

Website: http://unst-dev.vsos.ethz.ch/

## Development

- `app/` is the website (Dash) and the HTTP API (`app/api/`), `notebooks/` is data exploration,
  `tests/` covers the data layer, the analyses and the API
- Dependencies are managed with [uv](https://docs.astral.sh/uv/): `uv sync`

```bash
uv sync                  # install dependencies
uv run init-env          # start a local database with a copy of the production data (see below)
uv run start-app         # Dash website: http://127.0.0.1:8050
uv run start-api         # HTTP API: http://127.0.0.1:8000/api/docs
uv run pytest            # tests (integration tests skip themselves without the database)
```

The Vue 3 prototype that will replace the Dash site lives in `web/` (see [web/README.md](web/README.md)):
with the API running, `cd web && npm install && npm run dev` serves it on http://localhost:5173.

We are moving from the Dash prototype to a Vue 3 frontend on top of an HTTP API; the plan and its
status are in [plans/frontend_maturation_plan.md](plans/frontend_maturation_plan.md).

## Code layout

```
app/un_data_stream/data/       loading from Postgres, the precomputed vote arrays, periodic reload
app/un_data_stream/analysis/   every analysis, framework-free: filtering, keywords, agreement,
                               word cloud, country profile, overview — used by Dash *and* the API
app/data.py                    the loaded data plus country names, membership periods, trees
app/api/                       FastAPI: routes, query parameters, response schemas
app/features/, app/pages/      the Dash UI (to be retired once the Vue frontend reaches parity)
web/                           the Vue 3 frontend (prototype), on top of app/api
```

New analyses go into `app/un_data_stream/analysis/` as plain functions with unit tests on the
synthetic data in `tests/unit/conftest.py`; the API and the UI then only present them.

## HTTP API

`uv run start-api` serves the API on port 8000 (`POLICY_PULSE_API_PORT` to change it). The
interactive documentation at `/api/docs` lists every endpoint with its parameters, and
`/api/openapi.json` is the schema the frontend's TypeScript client is generated from.

- Endpoints live under `/api/v1`. List and analysis endpoints share one set of filter parameters
  (`start_date`, `end_date`, `subject`, `country`, `country_mode`, `keyword`); `lang` picks the
  language of country and subject names (en, fr, es, ar, zh, ru).
- Responses carry an `ETag` and an `X-Data-Version` header; repeat requests with `If-None-Match`
  get a `304` until the data is reloaded.
- Browsers may call it from the origins in `POLICY_PULSE_CORS_ORIGINS` (comma-separated; defaults
  to the Vite dev server, `http://localhost:5173`).
- Startup loads the data like the Dash app does (about 25 s), so `--reload` restarts are slow.

## Local database

The app is a read-only consumer of a Postgres database that the separate `undl-ingest` repo fills.
You do **not** need a production connection or to run any ingestion job to work on the frontend:
the ingest VM dumps its database every Monday and publishes it as a ready-to-run Postgres image,
and `init-env` pulls that image and starts it.

```bash
docker login ghcr.io -u <your-github-username>   # once: private package, token needs read:packages
uv run init-env
```

That writes a `.env` if you don't have one, pulls and starts the database, waits for it to finish
restoring, then connects through the app's own settings and counts the core tables — so it fails
loudly if the container came up empty rather than leaving you to discover it as a blank page. When
it prints `Ready`, `uv run start-app` will work.

If port 5432 is already taken by a Postgres of your own:

```bash
POLICY_PULSE_DB_PORT=55432 uv run init-env       # remember to set PGPORT=55432 in .env too
```

### Without the Python wrapper

`init-env` is a convenience around `docker-compose.yml`; everything it does can be done by hand:

```bash
cp .env.example .env     # then set PGPASSWORD=policy_pulse
docker login ghcr.io
docker compose pull
docker compose up -d
docker compose ps        # wait for the healthcheck to say "healthy"
```

The container is `policy-pulse-db`, listening on `127.0.0.1:5432`, database/user/password all
`policy_pulse`.

### Updating to a newer snapshot

```bash
uv run update-db          # --force re-seeds even when you are already current
```

It pulls the newest image, wipes the data volume, re-seeds, and prints the row counts with the
change against what you had. Restart the app afterwards — data is read once, at import.

Note that **`docker compose pull && docker compose up -d` does not update the data.** It recreates
the container on the new image, but the image restores its dump only into a *fresh* data
directory, so the old volume is served as-is: new image, old rows, healthy container, no warning.
The volume has to go, which is what `update-db` does (and why it skips the wipe when it can tell
you are already current — it keeps a marker inside the volume recording what seeded it). By hand:

```bash
docker compose pull && docker compose down -v && docker compose up -d
```

Either way this **discards anything you changed in the local database** — it is a disposable copy
of the snapshot, not somewhere to keep work.

### What the data is

A **snapshot**, not a live mirror: it is published weekly, so expect it to be a few days behind
production.

The publishing side lives in `undl-ingest` (`deploy/undl-snapshot`, `snapshot/`); its
`deploy/README.md` documents why the snapshot works this way and how to trigger one early.

Unrelated to all this, `notebooks/janic/postgres/` builds an **empty-schema** Postgres image. It
belongs to the ingestion prototype that has since moved to `undl-ingest` and is kept only as a
record — it is not what you want for frontend work.
