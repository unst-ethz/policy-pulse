"""Load one immutable snapshot per worker, with serialized cache initialization."""

from filelock import FileLock

from app.un_data_stream import DataRepository

from .config import Settings
from .service import AnalysisService


def build_service(settings: Settings) -> AnalysisService:
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    with FileLock(str(settings.data_dir / ".initialize.lock"), timeout=900):
        repository = DataRepository(
            str(settings.config_path),
            data_dir=str(settings.data_dir),
            log_dir=str(settings.log_dir),
        )
        if repository.resolution_table.empty:
            raise ValueError("The configured dataset contains no resolutions")
        if repository.resolution_table.undl_id.duplicated().any():
            raise ValueError("Resolution identifiers must be unique")
        return AnalysisService(repository, settings.assets_dir)


if __name__ == "__main__":
    service = build_service(Settings())
    print(f"Snapshot ready: {len(service.engine.resolution_table):,} resolutions")
