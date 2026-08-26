from pathlib import Path
from typing import Any
import yaml


def load_pipeline_config(path: Path) -> dict[str, Any]:
    path = Path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    for name, value in data.get("paths", {}).items():
        if Path(str(value)).is_absolute():
            raise ValueError(f"paths.{name} must be relative to repository root")
    return data
