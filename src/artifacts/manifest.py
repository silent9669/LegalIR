import json
from pathlib import Path
from typing import Any
from src.artifacts.checksums import sha256_file


def build_inventory(root: Path) -> dict[str, dict[str, Any]]:
    root = Path(root).resolve()
    return {
        path.relative_to(root).as_posix(): {
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def verify_inventory(root: Path, manifest_path: Path) -> list[str]:
    root = Path(root).resolve()
    manifest_path = Path(manifest_path)
    expected = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors = []
    for rel, record in expected.items():
        path = root / rel
        if not path.exists():
            errors.append(f"missing: {rel}")
        elif sha256_file(path) != record["sha256"]:
            errors.append(f"checksum mismatch: {rel}")
    return errors
