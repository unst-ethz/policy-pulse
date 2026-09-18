# How Our Deployment Works

## The Flow

```
You push to main
       ↓
GitHub Actions builds a Docker image (~1-2 min)
       ↓
Watchtower (running on our VM) notices the new image
       ↓
Old container is removed, new one starts (Site goes down)
       ↓
App loads its data from Postgres (~25 sec)
       ↓
Site is up again
```

> Until 2026-09 this step refetched and reprocessed the UN bulk exports on every deploy, which is
> why older notes say the site is down for ~5 minutes. The app now reads already-processed tables
> from Postgres (filled by the separate `undl-ingest` jobs), so a deploy costs seconds, not
> minutes.

---

## Key Components

**GitHub Actions** — Automatically builds a Docker image whenever we push to `main` using the `Dockerfile`. The image contains our code and dependencies.

**uv** — Dependency manager. The image installs from `uv.lock` (`uv sync --locked --no-dev`), so a deploy gets exactly the versions the lockfile pins, and the `notebooks` group (marimo/altair) is left out. There is no `requirements.txt` any more — editing dependencies means editing `pyproject.toml` and committing the refreshed `uv.lock`.

**Watchtower** — A service running on our VM that checks every 60 seconds if there's a new image. When it finds one, it pulls it and restarts the app.

**Nginx** — A reverse proxy that sits in front of our app. It handles incoming web requests and forwards them to our Dash app. Can use Nginx to add SSL later.

**Gunicorn** — The server that runs our Python app. We use the `--preload` flag, which loads the resolution data once and then spawns worker processes that share it copy-on-write.

**Postgres** — Holds the resolution/vote/thesaurus/member-state tables the app reads at startup. It is written by the `undl-ingest` jobs, not by this app. The app needs `PGHOST`, `PGPORT`, `PGDATABASE`, `PGUSER` and `PGPASSWORD` in its environment, and fails fast at startup if they're missing. Note that any DB connection must be opened *after* gunicorn forks its workers — the data load opens and closes its own connection inside the master, so nothing is shared across the fork.

---

## What Happens on Deploy

When we push to `main`:
1. The current container is removed
2. A new container starts with the updated code
3. The app reads its tables from Postgres and precomputes the vote-agreement arrays
4. **Site is unavailable for roughly half a minute**

Data freshness is no longer tied to deploys: the ingestion jobs refresh Postgres on their own
schedule, so new resolutions arrive without anyone pushing to `main`.

---

## Checking Status

Everything runs with Docker. To see what's happening on the VM:

```bash
cd /opt/dash-app

# View logs for the app
docker compose logs -f app

# View logs for everything
docker compose logs -f
```

---

## Environment

The container needs the Postgres connection settings at **run** time — they are never baked into
the image:

```bash
docker run --rm -p 8050:8050 --env-file .env policy-pulse
```

`PGHOST`, `PGPORT`, `PGDATABASE`, `PGUSER`, `PGPASSWORD`. If any are missing the app refuses to
start with a message naming them; if the database is unreachable it reports the host, port,
database and user it tried (never the password), so a broken deploy is diagnosable from
`docker compose logs app` alone.

Note the image copies only `app/` and `config/`, so a local `.env` in the build context cannot end
up in a published layer (`.dockerignore` excludes it as well).

---

## Code Requirement

Your Dash app must expose the server object for Gunicorn:
```python
app = dash.Dash(__name__)
server = app.server  # ← Required for deployment
```

Without this line, the deployment will fail.

---

## File Locations

All Docker configuration lives on the VM at:
```
/opt/dash-app/
```

---

## Next Steps

- [ ] Add SSL/HTTPS for security
- [ ] Configure firewall
