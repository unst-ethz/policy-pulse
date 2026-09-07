"""Generate a deterministic, reviewable API schema without starting ingestion."""

import json
from pathlib import Path

from .main import create_app

if __name__ == "__main__":
    Path(__file__).with_name("openapi.json").write_text(
        json.dumps(create_app().openapi(), indent=2, sort_keys=True) + "\n"
    )
