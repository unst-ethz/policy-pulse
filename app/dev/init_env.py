"""
`uv run init-env` — one command from a fresh clone to a working local database.

It writes a `.env` if there isn't one, starts the snapshot Postgres from the repo's
`docker-compose.yml`, waits for it to actually hold data, and then connects to it through the
app's own data layer (`app.un_data_stream.data.db`) and counts the core tables. That last step is
the point of the whole thing: the container can start, report healthy and still be the wrong
database, so this exits nonzero unless the rows the app needs are really there.

It mirrors `undl-ingest`'s own `init-env` step for step, the same way this repo's `data/db.py`
mirrors that repo's `common/db.py` — same container, same tables, same checks, no shared code.
The difference is that this one verifies through `app.data`'s connection settings, so a `.env`
pointing somewhere else is caught here rather than as a blank page later.

Deliberately dev-only. It is never imported by the app and never runs in the container — it just
lives under `app/` because that is the one package the wheel installs, which is what makes
`[project.scripts]` able to find it.
"""

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# app/dev/init_env.py -> app/dev -> app -> repo root
REPO_ROOT = Path(__file__).resolve().parents[2]
COMPOSE_FILE = REPO_ROOT / "docker-compose.yml"
ENV_FILE = REPO_ROOT / ".env"
ENV_EXAMPLE = REPO_ROOT / ".env.example"

CONTAINER_NAME = "policy-pulse-db"
IMAGE = "ghcr.io/unst-ethz/policy-pulse-db"

# The snapshot image bakes in these credentials (undl-ingest's snapshot/Dockerfile) and accepts no
# others. `.env.example` ships PGPASSWORD blank because the same file also has to describe a real
# production connection, so filling it in is this script's job.
DEV_PASSWORD = "policy_pulse"

# The tables the app reads on startup. `ingestion_runs` and `subject_closure` are deliberately not
# checked: the first is ingestion bookkeeping the app never touches, and the second is legitimately
# derived, so an empty one is a data question rather than a "did the snapshot restore" question.
REQUIRED_TABLES = (
    "member_states",
    "resolution_outcomes",
    "resolution_votes",
    "subject",
    "subject_broader",
)

HEALTH_TIMEOUT_SECONDS = 60
HEALTH_POLL_SECONDS = 2


class InitEnvError(RuntimeError):
    """A failure with a message already written for the person running the command."""


def _say(message: str) -> None:
    print(message, flush=True)


def _run(args: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    """Run a command, capturing both streams as text. Never raises on a nonzero exit."""
    return subprocess.run(args, cwd=cwd, capture_output=True, text=True)


def _stream_run(args: list[str], cwd: Path | None = None) -> int:
    """Run a command with its output going straight to the terminal, returning the exit code.

    Used where the output is the useful part (`docker logs`), rather than something to match on.
    """
    return subprocess.run(args, cwd=cwd).returncode


def _compose(*args: str) -> list[str]:
    return ["docker", "compose", "-f", str(COMPOSE_FILE), *args]


def db_port() -> str:
    """The host port the database is published on — the same default the compose file uses."""
    return os.environ.get("POLICY_PULSE_DB_PORT", "5432")


def ensure_env_file() -> None:
    """Create `.env` from `.env.example` with the dev password filled in, if it doesn't exist."""
    if ENV_FILE.exists():
        _say(f"==> {ENV_FILE.name} already exists, leaving it alone")
        return
    if not ENV_EXAMPLE.exists():
        raise InitEnvError(f"Neither {ENV_FILE} nor {ENV_EXAMPLE} exists — is this the repo root?")

    port = db_port()
    lines = []
    for line in ENV_EXAMPLE.read_text().splitlines():
        # The example's own header explains how to fill the file in by hand, which is no longer
        # the reader's situation once this has written it; the header below replaces it.
        if not lines and (line.startswith("#") or not line.strip()):
            continue
        if line.startswith("PGPASSWORD="):
            line = f"PGPASSWORD={DEV_PASSWORD}"
        elif line.startswith("PGPORT="):
            line = f"PGPORT={port}"
        lines.append(line)
    header = (
        "# Written by `uv run init-env` for the local snapshot database (docker-compose.yml).\n"
        "# Yours to edit — it is gitignored and never committed.\n"
    )
    ENV_FILE.write_text(header + "\n".join(lines) + "\n")
    _say(f"==> Wrote {ENV_FILE.name} (local dev database on port {port})")


def check_docker() -> None:
    """Check that docker is installed and its daemon is actually reachable."""
    if shutil.which("docker") is None:
        raise InitEnvError(
            "`docker` is not installed (or not on PATH).\n"
            "Install Docker Engine or Docker Desktop, then run `uv run init-env` again."
        )
    result = _run(["docker", "info"])
    if result.returncode != 0:
        raise InitEnvError(
            "`docker info` failed — the Docker daemon is not reachable.\n"
            "Start Docker (e.g. `sudo systemctl start docker`, or open Docker Desktop) and "
            "make sure your user can talk to it, then run `uv run init-env` again.\n\n"
            f"{result.stderr.strip()}"
        )
    _say("==> Docker is available")


def compose_pull() -> None:
    """Pull the snapshot image, translating the two failures people actually hit."""
    _say(f"==> Pulling {IMAGE} (private package — needs `docker login ghcr.io`)")
    result = _run(_compose("pull"), cwd=REPO_ROOT)
    if result.returncode == 0:
        return

    stderr = result.stderr.strip()
    lowered = stderr.lower()
    if "unauthorized" in lowered or "denied" in lowered or "authentication required" in lowered:
        raise InitEnvError(
            "Not authorised to pull the snapshot image.\n"
            "It is a private GHCR package, so Docker needs a GitHub token with `read:packages`:\n"
            "    docker login ghcr.io -u <your-github-username>\n"
            "(paste the token as the password), then run `uv run init-env` again.\n\n"
            f"{stderr}"
        )
    if "not found" in lowered or "manifest unknown" in lowered:
        raise InitEnvError(
            f"{IMAGE} has no image to pull.\n"
            "If you are logged in, the VM may not have published a snapshot yet — check "
            "`undl-snapshot.service` on the ingest VM (see undl-ingest's deploy/README.md).\n\n"
            f"{stderr}"
        )
    raise InitEnvError(f"`docker compose pull` failed.\n\n{stderr}")


def compose_up() -> None:
    """Start the database, translating a busy host port into the override that fixes it."""
    port = db_port()
    _say(f"==> Starting {CONTAINER_NAME} on 127.0.0.1:{port}")
    result = _run(_compose("up", "-d"), cwd=REPO_ROOT)
    if result.returncode == 0:
        return

    stderr = result.stderr.strip()
    lowered = stderr.lower()
    if "already allocated" in lowered or "address already in use" in lowered:
        raise InitEnvError(
            f"Host port {port} is already in use — most likely by a Postgres you already run.\n"
            "Start the snapshot database on a different port instead:\n"
            "    POLICY_PULSE_DB_PORT=55432 uv run init-env\n"
            f"(then make sure PGPORT in {ENV_FILE.name} says 55432 too).\n\n"
            f"{stderr}"
        )
    raise InitEnvError(f"`docker compose up -d` failed.\n\n{stderr}")


def _health_status() -> str | None:
    """The container's healthcheck status, or None if it cannot be inspected (e.g. it is gone)."""
    result = _run(["docker", "inspect", "--format", "{{.State.Health.Status}}", CONTAINER_NAME])
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def wait_for_healthy() -> None:
    """Poll until the container's healthcheck passes.

    The healthcheck queries `member_states` rather than running `pg_isready`, because the snapshot
    is restored by the entrypoint *before* Postgres starts accepting external connections: a plain
    readiness probe can go green against a database whose tables are still empty.
    """
    _say(f"==> Waiting for {CONTAINER_NAME} to be healthy (restoring the snapshot)")
    deadline = time.monotonic() + HEALTH_TIMEOUT_SECONDS
    status = None
    while time.monotonic() < deadline:
        status = _health_status()
        if status == "healthy":
            _say("    healthy")
            return
        if status is None:
            break
        time.sleep(HEALTH_POLL_SECONDS)

    _say(f"    last health status: {status or 'could not inspect the container'}")
    _say(f"--- last lines of `docker logs {CONTAINER_NAME}` ---")
    _stream_run(["docker", "logs", "--tail", "30", CONTAINER_NAME])
    raise InitEnvError(
        f"{CONTAINER_NAME} did not become healthy within {HEALTH_TIMEOUT_SECONDS}s.\n"
        "If the volume predates the current image the snapshot is not re-applied — "
        "`docker compose down -v && uv run init-env` wipes it and re-seeds from the image."
    )


def check_tables() -> None:
    """Connect the way the app does and count the core tables, failing if any is empty."""
    # Imported here, not at module scope: this pulls in pandas and SQLAlchemy, and the Docker and
    # compose steps above should be able to fail fast without paying for that.
    import sqlalchemy as sa

    from app.un_data_stream.data import db

    _say("==> Checking the data through the app's own connection settings")
    db.load_env()
    configured_port = os.environ.get("PGPORT")
    if configured_port != db_port():
        raise InitEnvError(
            f"PGPORT in {ENV_FILE.name} is {configured_port}, but the database is published on "
            f"{db_port()}.\nUpdate {ENV_FILE.name} (or unset POLICY_PULSE_DB_PORT) so the app "
            "connects to the container this command started."
        )

    engine = db.create_engine()
    try:
        with db.connect_or_explain(engine) as conn:
            counts = {
                table: conn.execute(sa.text(f"select count(*) from {table}")).scalar_one()
                for table in REQUIRED_TABLES
            }
    finally:
        # NullPool or not, the engine is disposed explicitly — see db.create_engine's docstring.
        engine.dispose()

    width = max(len(table) for table in counts)
    for table, count in counts.items():
        _say(f"    {table.ljust(width)}  {count:>9,}")

    empty = [table for table, count in counts.items() if count == 0]
    if empty:
        raise InitEnvError(
            f"These tables are empty: {', '.join(empty)}.\n"
            "The snapshot did not restore — most likely an old `pgdata` volume survived, which "
            "the image's restore step deliberately leaves alone. Wipe it and try again:\n"
            "    docker compose down -v && uv run init-env"
        )


def main() -> None:
    """Entry point for `uv run init-env`."""
    try:
        ensure_env_file()
        check_docker()
        compose_pull()
        compose_up()
        wait_for_healthy()
        check_tables()
    except InitEnvError as exc:
        print(f"\ninit-env failed.\n\n{exc}", file=sys.stderr)
        raise SystemExit(1) from None

    port = db_port()
    _say("")
    _say(f"Ready. Postgres is on localhost:{port} (database policy_pulse, user policy_pulse).")
    _say("Start the app with `uv run start-app` — http://127.0.0.1:8050")
    _say("The data is a weekly snapshot: `docker compose pull && docker compose up -d` for newer.")


if __name__ == "__main__":
    main()
