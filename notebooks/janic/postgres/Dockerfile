# Local prototype Postgres for the intermediate-storage-layer ingestion notebook.
# Not for production use -- see plans/intermediate_storage_layer_plan.md's "Ingestion job
# Postgres credentials" open question for the real deployment story.
#
# Build:  docker build -t policy-pulse-pg notebooks/janic/postgres
# Run:    docker run -d --name policy-pulse-pg -p 5432:5432 \
#           -v policy_pulse_pgdata:/var/lib/postgresql/data policy-pulse-pg
#
# schema.sql only runs automatically on a *fresh* data directory (Postgres's own
# /docker-entrypoint-initdb.d convention -- see https://hub.docker.com/_/postgres, "Initialization
# scripts"). If you already have a policy_pulse_pgdata volume from a previous `docker run`, remove
# it first (`docker volume rm policy_pulse_pgdata`) or the schema won't be (re-)applied.
FROM postgres:16

ENV POSTGRES_USER=policy_pulse
ENV POSTGRES_PASSWORD=policy_pulse
ENV POSTGRES_DB=policy_pulse

COPY schema.sql /docker-entrypoint-initdb.d/01-schema.sql
