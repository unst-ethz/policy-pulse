"""
`uv run init-env` — one command from a fresh clone to a working local database.

It writes a `.env` if there isn't one, starts the local MongoDB from the repo's
`docker-compose.yml`, waits for it to be healthy, and then connects to it through the app's own
data layer (`app.un_data_stream.data.db`) and counts the core collections. That last step is
the point of the whole thing: the container can start, report healthy and still be the wrong
database, so this exits nonzero unless the rows the app needs are really there.

It mirrors `undl-ingest`'s own `init-env` step for step, the same way this repo's `data/db.py`
mirrors that repo's `common/db.py` — same container, same collections, same checks, no shared
code.
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

CONTAINER_NAME = "policy-pulse-mongo"
# TODO(mongodb phase 5): the MongoDB dev snapshot image, once undl-ingest publishes one (see
# docker-compose.yml). Until then this is the empty base image and the count check below fails.
IMAGE = "mongodb/mongodb-atlas-local:8.0.15"

# What `DataRepository` reads on startup, as {label: (collection, filter)}: the three collections,
# plus the embedded data that replaced the old vote, subject-pair, broader and closure tables. An
# empty one of those fails nothing outright — no `ancestors` silently breaks subject-hierarchy
# expansion — so each is counted on its own. `ingestion_runs` is bookkeeping, only the marker.
REQUIRED_COUNTS = {
    "member_states": ("member_states", {}),
    "resolutions": ("resolutions", {}),
    "resolutions with votes": ("resolutions", {"votes": {"$exists": True}}),
    "resolutions with subjects": ("resolutions", {"derived.subject_ids.0": {"$exists": True}}),
    "subjects": ("subjects", {}),
    "subjects with broader": ("subjects", {"broader.0": {"$exists": True}}),
    "subjects with ancestors": ("subjects", {"ancestors.0": {"$exists": True}}),
}

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
    return os.environ.get("POLICY_PULSE_DB_PORT", "27017")


def ensure_env_file() -> None:
    """Create `.env` from `.env.example` with the local port filled in, if it doesn't exist."""
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
        if line.startswith("MONGODB_URI="):
            line = f"MONGODB_URI=mongodb://localhost:{port}/?directConnection=true"
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


def compose_pull(image: str | None = None) -> None:
    """Pull the snapshot image, translating the two failures people actually hit."""
    _say(f"==> Pulling {image or IMAGE}")
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
            f"Host port {port} is already in use — most likely by a MongoDB you already run.\n"
            "Start the local database on a different port instead:\n"
            "    POLICY_PULSE_DB_PORT=27018 uv run init-env\n"
            f"(then make sure MONGODB_URI in {ENV_FILE.name} says localhost:27018 too).\n\n"
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

    Healthy only means MongoDB is up (see docker-compose.yml's TODO for the snapshot's data
    check); whether the data is there is `check_counts`' question.
    """
    _say(f"==> Waiting for {CONTAINER_NAME} to be healthy")
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


def read_counts() -> dict[str, int]:
    """Count the core collections over the app's own connection settings.

    Going through `app.un_data_stream.data.db` rather than `docker exec ... mongosh` is the point:
    it proves the database is reachable *the way the app reaches it*, `.env` included, instead of
    only proving that something is alive inside the container.
    """
    # Imported here, not at module scope: this pulls in pandas and PyMongo, and the Docker and
    # compose steps above should be able to fail fast without paying for that.
    from app.un_data_stream.data import db

    db.load_env()
    configured_port = db.host().rpartition(":")[2]
    if configured_port != db_port():
        raise InitEnvError(
            f"MONGODB_URI in {ENV_FILE.name} points at {db.host()}, but the database is published "
            f"on port {db_port()}.\nUpdate {ENV_FILE.name} (or unset POLICY_PULSE_DB_PORT) so the "
            "app connects to the container this command started."
        )

    with db.connect_or_explain() as client:
        database = db.database(client)
        return {
            label: database[collection].count_documents(query)
            for label, (collection, query) in REQUIRED_COUNTS.items()
        }


def check_counts(previous: dict[str, int] | None = None) -> dict[str, int]:
    """Print the core document counts, with deltas when we know the old ones.

    Fails if any is zero — that is the snapshot not having restored.
    """
    _say("==> Checking the data through the app's own connection settings")
    counts = read_counts()

    width = max(len(label) for label in counts)
    for label, count in counts.items():
        line = f"    {label.ljust(width)}  {count:>9,}"
        if previous is not None:
            delta = count - previous.get(label, 0)
            line += f"   {delta:+,}" if delta else "   ="
        _say(line)

    empty = [label for label, count in counts.items() if count == 0]
    if empty:
        # TODO(mongodb phase 5): until the snapshot image exists, the database starts empty and
        # this is expected; see docker-compose.yml.
        raise InitEnvError(
            f"These collections are empty: {', '.join(empty)}.\n"
            "The snapshot did not restore — most likely an old `mongodata` volume survived, which "
            "the image's restore step deliberately leaves alone. Wipe it and try again:\n"
            "    docker compose down -v && uv run init-env\n"
            "(Until the MongoDB snapshot image exists, the local database starts empty: see the "
            "TODO in docker-compose.yml.)"
        )
    return counts


def compose_image() -> str:
    """The image reference the compose file resolves to, env interpolation and all."""
    result = _run(_compose("config", "--images"), cwd=REPO_ROOT)
    images = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if result.returncode != 0 or not images:
        raise InitEnvError(f"Could not read the image from {COMPOSE_FILE.name}.\n\n{result.stderr}")
    return images[0]


def _image_id(reference: str) -> str | None:
    """The local image id `reference` resolves to, or None if it isn't present locally."""
    result = _run(["docker", "image", "inspect", "--format", "{{.Id}}", reference])
    return result.stdout.strip() if result.returncode == 0 else None


def _running_image_id() -> str | None:
    """The image id the database container is actually running, or None if there is no container."""
    result = _run(["docker", "inspect", "--format", "{{.Image}}", CONTAINER_NAME])
    return result.stdout.strip() if result.returncode == 0 else None


# Written into the data volume itself once a seed has been verified. The volume is the thing whose
# contents we are asking about, and this marker shares its lifetime exactly: `docker compose down
# -v` destroys both together, so the marker can never outlive the data it describes.
#
# The image id alone cannot answer "is my data current?". `docker compose pull && up -d` recreates
# the container on the new image *without* re-running the restore, which leaves a container that
# reports the new image while serving the old rows — so comparing the running image to the pulled
# one says "up to date" precisely when the data is stale. This marker records what actually seeded
# the volume, which is the question.
SEED_MARKER = "/data/db/.policy-pulse-seeded-from"


def read_seed_marker() -> str | None:
    """The image id this volume was seeded from, or None if unknown (no container, or seeded by
    something other than these commands)."""
    result = _run(["docker", "exec", CONTAINER_NAME, "cat", SEED_MARKER])
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def write_seed_marker(image_id: str) -> None:
    """Record what seeded this volume. Best effort: failing to write it costs a future no-op
    update, which is not worth failing an otherwise successful setup over."""
    _run(["docker", "exec", CONTAINER_NAME, "sh", "-c", f"printf %s '{image_id}' > {SEED_MARKER}"])


def compose_down(remove_volumes: bool) -> None:
    """Stop the database, optionally destroying its data volume."""
    args = ["down", "-v"] if remove_volumes else ["down"]
    result = _run(_compose(*args), cwd=REPO_ROOT)
    if result.returncode != 0:
        raise InitEnvError(f"`docker compose {' '.join(args)}` failed.\n\n{result.stderr.strip()}")


def main() -> None:
    """Entry point for `uv run init-env`."""
    try:
        ensure_env_file()
        check_docker()
        compose_pull()
        compose_up()
        wait_for_healthy()
        check_counts()
        # Record what seeded this volume so `update-db` can tell later whether it is still current.
        image_id = _image_id(compose_image())
        if image_id is not None:
            write_seed_marker(image_id)
    except InitEnvError as exc:
        print(f"\ninit-env failed.\n\n{exc}", file=sys.stderr)
        raise SystemExit(1) from None

    port = db_port()
    _say("")
    _say(f"Ready. MongoDB is on localhost:{port} (database policy_pulse).")
    _say("Start the app with `uv run start-app` — http://127.0.0.1:8050")
    _say("The data is a weekly snapshot; `uv run update-db` moves you to a newer one.")


def update() -> None:
    """Entry point for `uv run update-db` — move the local database to the newest snapshot.

    `docker compose pull && docker compose up -d` is *not* enough on its own, and fails silently
    when it is wrong. The image restores its dump only on a fresh data directory, so an existing
    `mongodata` volume means the new image starts up against the old data: the container is
    healthy, the collections are full, and nothing has changed. The volume has to go for the new
    snapshot to land, which is why this wipes it rather than just recreating the container.

    Since that wipe is destructive, it is skipped when the pulled image is the one already running
    (`--force` re-seeds anyway, for a dev database someone has scribbled on).
    """
    force = "--force" in sys.argv[1:]
    try:
        check_docker()
        image = compose_image()

        running = _running_image_id()
        seeded_from = read_seed_marker() if running is not None else None
        previous: dict[str, int] | None = None
        if running is not None:
            # Best effort: if the current database is unreachable there is simply nothing to
            # compare against, which is not a reason to refuse to update.
            try:
                previous = read_counts()
            except Exception:
                previous = None

        compose_pull(image)
        pulled = _image_id(image)

        if seeded_from is not None and seeded_from == pulled and not force:
            _say("")
            _say("Already on the newest published snapshot — nothing to do.")
            _say("(`uv run update-db --force` re-seeds from it anyway.)")
            return

        if running is None:
            _say("==> No database container yet — creating one")
        elif seeded_from is None:
            # Either an older volume from before these commands existed, or one that a manual
            # `docker compose up -d` recreated without re-seeding. Both mean the data's provenance
            # is unknown, and re-seeding is the only way to make it known.
            _say("==> Cannot tell what this volume was seeded from — re-seeding to be sure")
        elif seeded_from != pulled:
            _say("==> New snapshot pulled; re-seeding from it")
        else:
            _say("==> Already current, re-seeding anyway (--force)")

        # The wipe and the re-create are one step from the caller's point of view, but `down -v`
        # is the part that actually discards data, so it is announced before it happens.
        _say("==> Clearing the data volume (`docker compose down -v`) — the snapshot only ever")
        _say("    restores into a fresh one")
        compose_down(remove_volumes=True)
        compose_up()
        wait_for_healthy()
        check_counts(previous=previous)
        if pulled is not None:
            write_seed_marker(pulled)
    except InitEnvError as exc:
        print(f"\nupdate-db failed.\n\n{exc}", file=sys.stderr)
        raise SystemExit(1) from None

    _say("")
    _say(f"Updated. MongoDB is on localhost:{db_port()} with the newest snapshot.")
    _say("Restart the app (`uv run start-app`) to load it — data is read once, at import.")


if __name__ == "__main__":
    main()
