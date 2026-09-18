#!/usr/bin/env bash
# Quick reset of the local prototype DB's *data* -- schema/tables are untouched (see schema.sql).
# Usage: notebooks/janic/postgres/reset.sh
set -euo pipefail

docker exec policy-pulse-pg psql -U policy_pulse -d policy_pulse -c \
  "TRUNCATE resolution_votes, resolution_outcomes;"

echo "Wiped resolution_outcomes / resolution_votes."
