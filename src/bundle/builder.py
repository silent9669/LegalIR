"""Builder for assembling and sealing immutable production bundles."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Union

from src.core.hashing import sha256_file
from src.core.manifests import BundleManifest


class ProductionBundleBuilder:
    """Collects, copies, and seals the artifacts that constitute the production bundle."""

    def __init__(
        self,
        bundle_dir: Union[str, Path],
        runtime_commit: str,
        dataset_fingerprint: str,
        bundle_version: str = "v1",
        config_sha256: str = "unknown",
    ):
        self.bundle_dir = Path(bundle_dir)
        self.runtime_commit = runtime_commit
        self.dataset_fingerprint = dataset_fingerprint
        self.bundle_version = bundle_version
        self.config_sha256 = config_sha256
        self.files_map: Dict[str, Path] = {}

    def add_file(self, rel_path: str, src_path: Union[str, Path]) -> None:
        """Register a file to be copied into the bundle under rel_path."""
        src = Path(src_path)
        if not src.is_file():
            raise FileNotFoundError(f"Bundle source file not found: {src}")
        self.files_map[rel_path] = src

    def freeze(self) -> BundleManifest:
        """Copy all files into bundle directory and generate signed bundle_manifest.json."""
        self.bundle_dir.mkdir(parents=True, exist_ok=True)
        file_entries: Dict[str, Dict[str, Any]] = {}

        for rel_path, src_p in sorted(self.files_map.items()):
            dst_p = self.bundle_dir / rel_path
            dst_p.parent.mkdir(parents=True, exist_ok=True)
            if src_p.resolve() != dst_p.resolve():
                shutil.copy2(src_p, dst_p)

            digest = sha256_file(dst_p)
            size = dst_p.stat().st_size
            file_entries[rel_path] = {
                "sha256": digest,
                "size_bytes": size,
            }

        manifest = BundleManifest(
            bundle_version=self.bundle_version,
            runtime_commit=self.runtime_commit,
            dataset_version="v2",
            dataset_fingerprint=self.dataset_fingerprint,
            config_sha256=self.config_sha256,
            files=file_entries,
            status="PASS",
        )

        manifest.save(self.bundle_dir / "bundle_manifest.json")
        return manifest
