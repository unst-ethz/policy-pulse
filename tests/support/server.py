"""Offline E2E server. Never imported or enabled by the production entrypoint."""

from backend.main import create_app

from .repository import build_service

app = create_app(service=build_service())
