.PHONY: install api frontend test schema lock build
install:
	python3.13 -m venv .venv
	.venv/bin/python -m pip install --require-hashes -r backend/requirements-dev.txt
	cd frontend && npm ci
api:
	.venv/bin/uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
frontend:
	cd frontend && npm run dev
test:
	.venv/bin/python -m pytest -q
	.venv/bin/ruff check backend tests/api tests/support scripts/add_requirement_hashes.py
	cd frontend && npm test && npm run format:check && npm run build && npm run test:e2e
schema:
	.venv/bin/python -m backend.export_openapi
	cd frontend && npm run api:generate
lock:
	.venv/bin/pip-compile --allow-unsafe --strip-extras --output-file backend/requirements.txt backend/requirements.in
	.venv/bin/pip-compile --allow-unsafe --strip-extras --constraint backend/requirements.txt --output-file backend/requirements-dev.txt backend/requirements-dev.in
	.venv/bin/python scripts/add_requirement_hashes.py backend/requirements.txt backend/requirements-dev.txt
build:
	docker compose build
