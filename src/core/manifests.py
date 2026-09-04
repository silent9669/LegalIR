"""Manifest schema and serialization for jobs, preflight, and bundles."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union


@dataclass
class Manifest:
    """Base manifest providing JSON serialization and deserialization."""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_json())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        return cls(**data)

    @classmethod
    def load(cls, path: Union[str, Path]):
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


@dataclass
class PreflightManifest(Manifest):
    """Manifest produced by the preflight stage verifying dataset and environment."""

    dataset_name: str
    dataset_version: str
    runtime_commit: str
    status: str
    details: Dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


@dataclass
class JobManifest(Manifest):
    """Manifest produced by an individual fold or doc-disjoint job."""

    job_id: str
    job_type: str  # "fold" or "doc_disjoint"
    status: str  # "PASS" or "FAIL"
    runtime_commit: str
    dataset_sha256: str
    inputs: Dict[str, str] = field(default_factory=dict)
    outputs: Dict[str, str] = field(default_factory=dict)  # file_path -> sha256
    metrics: Dict[str, float] = field(default_factory=dict)
    duration_seconds: float = 0.0


@dataclass
class BundleManifest(Manifest):
    """Manifest of the frozen immutable production bundle."""

    bundle_version: str
    runtime_commit: str
    dataset_version: str
    dataset_fingerprint: str
    config_sha256: str
    files: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # rel_path -> {sha256, size_bytes, rows}
    status: str = "PASS"
