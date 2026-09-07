# Policy Pulse

> **TEST ONLY — experimental `policy-pulse-react` branch.** This React/FastAPI migration is for review and testing, not production use. Browser and container checks use synthetic data. Real-data validation is still pending because UN downloads returned empty HTTP 202 responses. Do not deploy or switch production traffic to this branch until that release gate is completed and reviewed.

Policy Pulse is a project by the UN-ETH Student Team on creating a powerful interactive data analysis platform for UN voting records.

## React frontend and Python API

The frontend is React + TypeScript. A versioned FastAPI backend calls the existing Python scientific engine, preserving vote encoding, formulas and sample thresholds. The legacy Dash UI remains in `app/pages` and `app/features` for comparison; it is excluded from the production images.

Requirements: Python 3.13, Node.js 22.12+, and optionally Docker Compose.

```bash
make install
make api        # terminal 1: http://127.0.0.1:8000/api/docs
make frontend   # terminal 2: http://127.0.0.1:5173
```

The API initializes the configured UN dataset in the background. `data/` and `logs/` are local persistent directories. A cold start requires access to the UN source files; initialization failures remain visible as HTTP 503. Neither importing the API nor running offline tests downloads data.

For an explicitly synthetic, offline development preview, run `.venv/bin/uvicorn tests.support.server:app --host 127.0.0.1 --port 8000` instead of `make api`. This fixture server is not part of production.

## Verification

```bash
.venv/bin/python -m pytest -q
.venv/bin/ruff check backend tests/api tests/support
cd frontend
npm test
npm run build
npx playwright install chromium
npm run test:e2e
```

Browser tests start their own local API and frontend, so stop existing servers on ports 8000 and 5173 first. To use an installed Chrome test browser locally, set `PLAYWRIGHT_CHANNEL=chrome`. Live UN integration tests are opt-in with `POLICY_PULSE_LIVE_TESTS=1`; they are never counted as offline passes.

Run `make schema` after changing API models, and commit both generated schema files. Python dependencies are hash-locked in `backend/requirements*.txt`; JavaScript dependencies are locked in `frontend/package-lock.json`.

To update Python dependencies, edit `backend/requirements*.in` and run `make lock`. This resolves exact versions with pip-tools, then attaches the artifact digests published by PyPI. Review the lockfile changes and run the verification suite; changes to the pinned scientific stack need formula parity review.

## Deployment

```bash
docker compose up --build -d
curl --fail http://127.0.0.1:8050/health/ready
```

Nginx serves React on port 8050 and proxies `/api/*` to the separately running backend. The backend is not published to the host. Data and logs live in persistent named volumes. See [deployment instructions](deployment-overview.md) for warm-up, HTTPS proxying, release pinning, verification and rollback.

See [architecture and scientific invariants](docs/architecture-react.md) for module responsibilities, API contracts, migration details and remaining live-data release verification.

Existing website: http://unst-dev.vsos.ethz.ch/
