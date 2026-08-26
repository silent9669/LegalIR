from collections.abc import Iterator
from pathlib import Path
from typing import Any
import json
import zipfile


def iter_official_contexts(zip_path: Path) -> Iterator[dict[str, Any]]:
    zip_path = Path(zip_path)
    seen = set()
    with zipfile.ZipFile(zip_path) as archive:
        names = sorted(
            name for name in archive.namelist()
            if Path(name).name.startswith("context_") and name.endswith(".json")
        )
        if not names:
            raise ValueError(f"no context_*.json members in {zip_path}")
        for name in names:
            row = json.loads(archive.read(name).decode("utf-8"))
            row["id"] = str(row["id"])
            if row["id"] in seen:
                raise ValueError(f"duplicate document ID {row['id']}")
            seen.add(row["id"])
            yield row
