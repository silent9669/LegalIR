from pathlib import Path
from typing import Any
import yaml


class PipelineConfig(dict[str, Any]):
    """Mapping-backed pipeline configuration with attribute access."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def to_dict(self) -> dict[str, Any]:
        """Return a plain-Python representation safe for YAML serialization."""
        return _to_plain(self)


def _to_plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _to_plain(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_plain(item) for item in value]
    if isinstance(value, tuple):
        return [_to_plain(item) for item in value]
    return value


def _wrap_config(value: Any) -> Any:
    if isinstance(value, dict):
        return PipelineConfig({key: _wrap_config(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_wrap_config(item) for item in value]
    return value


def load_pipeline_config(path: Path | str = "configs/pipeline.yaml") -> PipelineConfig:
    path = Path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    for name, value in data.get("paths", {}).items():
        if Path(str(value)).is_absolute():
            raise ValueError(f"paths.{name} must be relative to repository root")
    return _wrap_config(data)
