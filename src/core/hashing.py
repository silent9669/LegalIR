"""Cryptographic hashing utilities for files, dataframes, and directories."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Union
import pandas as pd


def sha256_bytes(data: bytes) -> str:
    """Compute SHA-256 hex digest for raw bytes."""
    return hashlib.sha256(data).hexdigest()


def sha256_string(text: str) -> str:
    """Compute SHA-256 hex digest for a UTF-8 string."""
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: Union[str, Path], chunk_size: int = 65536) -> str:
    """Compute SHA-256 hex digest for a file streaming in fixed chunks."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def sha256_directory(dir_path: Union[str, Path], pattern: str = "*") -> str:
    """
    Compute a deterministic SHA-256 hex digest for a directory tree.
    Hashes sorted relative file paths concatenated with their SHA-256 digests.
    """
    dir_path = Path(dir_path)
    if not dir_path.is_dir():
        raise NotADirectoryError(f"Directory not found: {dir_path}")

    files = sorted([p for p in dir_path.rglob(pattern) if p.is_file()])
    h = hashlib.sha256()
    for p in files:
        rel_path = str(p.relative_to(dir_path)).replace("\\", "/")
        file_hash = sha256_file(p)
        h.update(f"{rel_path}:{file_hash}\n".encode("utf-8"))
    return h.hexdigest()


def sha256_dataframe(df: pd.DataFrame) -> str:
    """Compute deterministic SHA-256 hex digest for a Pandas DataFrame."""
    # Convert to CSV or binary bytes deterministically
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    return sha256_bytes(csv_bytes)
