"""Create an isolated SYNTHETIC cache to smoke-test production container startup."""

import logging
import sys
from pathlib import Path

import yaml

from app.un_data_stream.data.repository import DataRepository

from .repository import MemoryRepository


def export_cache(destination: Path):
    data = MemoryRepository().get_data()
    repo = DataRepository.__new__(DataRepository)
    repo.logger = logging.getLogger("synthetic-cache")
    repo.config = yaml.safe_load(Path("config/data_sources.yaml").read_text())
    repo.config["paths"]["data"] = str(destination)
    for key, attribute in {
        "resolution": "resolution_table",
        "resolution_subject": "resolution_subject_table",
        "subject": "subject_table",
        "closure": "closure_table",
        "broader": "broader_table",
        "member_states": "member_states_table",
        "country_columns": "country_columns",
        "multilateral_scores": "multilateral_scores",
        "vote_bool_arrays": "vote_bool_arrays",
    }.items():
        setattr(repo, attribute, data[key])
    repo._save_cached_data()
    (destination / "SYNTHETIC_TEST_DATA").write_text(
        "For automated tests only. Never deploy this cache as real UN data.\n"
    )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python -m tests.support.export_cache /absolute/test/cache")
    export_cache(Path(sys.argv[1]).resolve())
