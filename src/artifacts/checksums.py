from hashlib import sha256
from pathlib import Path


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    path = Path(path)
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()
