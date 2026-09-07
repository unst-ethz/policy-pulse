# Deploying the React frontend and API

> **TEST ONLY:** The `policy-pulse-react` branch is an experimental migration for review and testing. These instructions document candidate deployment packaging, not an approved production release. Real-data validation remains required before any production cutover.

Policy Pulse now uses two independently built containers: an Nginx container serving React, and a FastAPI container running the existing scientific engine. The Compose frontend binds to loopback port 8050 by default, preserving the previous reverse-proxy destination. Only the frontend is exposed to the host.

## Release process

Pull requests and branch pushes run the offline scientific/API suite, generated-schema check, frontend tests, browser journeys, and production container smoke test. Pushes to `main` or `stable` run the same checks before publishing two images:

- `ghcr.io/unst-ethz/policy-pulse/backend:<full-commit-sha>`
- `ghcr.io/unst-ethz/policy-pulse/frontend:<full-commit-sha>`

The workflow publishes versioned images. It does not change the VM or advance a floating `latest` tag. The previous single-container Watchtower deployment must be explicitly replaced with this Compose stack during a reviewed traffic cutover. Do not independently auto-update the two images.

## Before switching traffic

1. Retain the old Compose configuration and running image digest for rollback.
2. Use the same tested commit for both new image variables in `.env`, starting from `.env.example`.
3. Prepare a trusted existing cache or let the backend fetch the configured UN sources. Its persistent volume includes `metadata.json`, processed CSVs and `precomputed_agreement_data.pkl`. Existing compatible 1.4 caches remain valid; the filename compatibility update does not change their formulas or force a rebuild.
4. Verify the new stack on an unused local port, for example `POLICY_PULSE_PORT=18050 docker compose -p policy-pulse-candidate up -d`. With published images use `--no-build` after `docker compose pull`; for a local source build use `--build`.
5. Wait for `/health/ready` to return HTTP 200. Check `/api/v1/overview` against the trusted snapshot and inspect representative resolution, map, timeline, subject, multilateral and profile results. Run the opt-in live integration suite against the same source/cache where feasible.
6. Point the existing HTTPS reverse proxy at the candidate frontend only after these checks pass. Keep the old stack and its data until the new release is accepted.

As of 2026-09-07, source-file listings were reachable from development, but direct UN file downloads returned empty HTTP 202 responses. Ingestion now rejects those responses explicitly. A successful warm-up using real data from the VM or an existing trusted cache is a required release gate; synthetic browser/container fixtures do not establish live-source readiness.

## Operations

```bash
# Build and start locally; React is served through Nginx
# Run from the repository root.
docker compose up --build -d

# The process can be alive while its dataset is still loading.
curl --fail http://127.0.0.1:8050/health/live
curl --fail http://127.0.0.1:8050/health/ready
curl --fail http://127.0.0.1:8050/api/v1/overview

docker compose logs --tail=100 backend
docker compose logs --tail=100 frontend
docker compose ps
```

The initial warm-up may take several minutes. Subsequent restarts reuse the data volume. `docker compose down` preserves named volumes; do not add `-v` to normal restart or rollback commands.

Both containers run as non-root. The Python image does not include Dash, notebooks, test fixtures, or frontend build tools. The browser makes same-origin API requests; a separate-origin installation must explicitly set `POLICY_PULSE_CORS_ORIGINS`. Keep TLS termination on the existing external reverse proxy. Nginx returns the SPA shell for application deep links and returns 404 for missing hashed assets. API responses are never cached as static files.

There is one API worker by default, with bounded concurrency. Each additional process/replica loads its own snapshot; measure memory before increasing workers. Startup initialization is serialized using a file lock. Readiness stays false if data is unavailable or inconsistent. Monitor readiness externally; Docker health status alone does not automatically restart an unhealthy process.

## Refreshing data

Data is a versioned deployment input. To update it without damaging the active cache:

1. Build and validate a fresh snapshot in a separate directory/volume using `POLICY_PULSE_DATA_DIR=/absolute/new-snapshot .venv/bin/python -m backend.provider` from a network environment with source-file access.
2. Review changes in date coverage, counts, country authority names and subject metadata, and run scientific parity tests.
3. Start the candidate API against that snapshot and verify readiness.
4. Roll out the candidate with both matching image versions. Keep the previous snapshot for rollback.

The API exposes no administrative or public refresh endpoint. Source selection is controlled by `config/data_sources.yaml`; scientific definition changes require a separate review.

## Rollback

Restore both previous image tags/digests and the previous compatible data snapshot, start them on the candidate port, confirm readiness, then restore the proxy target. A cache copied from an unknown source is unsafe because the legacy scientific cache contains a pickle; accept only trusted project snapshots.

## Isolated container smoke test

This test uses labeled synthetic records and does not exercise the UN network:

```bash
.venv/bin/python -m tests.support.export_cache /tmp/policy-pulse-smoke-data
chmod -R a+rwX /tmp/policy-pulse-smoke-data
POLICY_PULSE_TEST_CACHE=/tmp/policy-pulse-smoke-data POLICY_PULSE_PORT=18050 \
  docker compose -p policy-pulse-smoke -f compose.yaml -f compose.test.yaml up --build --wait
.venv/bin/python -m tests.support.smoke_stack http://127.0.0.1:18050
POLICY_PULSE_TEST_CACHE=/tmp/policy-pulse-smoke-data POLICY_PULSE_PORT=18050 \
  docker compose -p policy-pulse-smoke -f compose.yaml -f compose.test.yaml down
```

Never use `compose.test.yaml` or the generated synthetic cache in a real deployment.
