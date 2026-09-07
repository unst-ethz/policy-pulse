import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    config_path: Path = field(
        default_factory=lambda: Path(
            os.getenv("POLICY_PULSE_CONFIG", ROOT / "config/data_sources.yaml")
        )
    )
    data_dir: Path = field(
        default_factory=lambda: Path(os.getenv("POLICY_PULSE_DATA_DIR", ROOT / "data"))
    )
    log_dir: Path = field(
        default_factory=lambda: Path(os.getenv("POLICY_PULSE_LOG_DIR", ROOT / "logs"))
    )
    assets_dir: Path = ROOT / "app/assets"
    cors_origins: tuple[str, ...] = field(
        default_factory=lambda: tuple(
            v.strip() for v in os.getenv("POLICY_PULSE_CORS_ORIGINS", "").split(",") if v.strip()
        )
    )
