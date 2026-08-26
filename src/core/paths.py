from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    repo: Path
    shared: Path
    canonical: Path
    local: Path
    local_models: Path
    local_indexes: Path
    local_runs: Path

    @classmethod
    def from_repo(cls, repo_root: Path | None = None) -> "ProjectPaths":
        repo = (repo_root or Path(__file__).resolve().parents[2]).resolve()
        shared = repo / "artifacts" / "shared"
        local = repo / "artifacts" / "local"
        return cls(
            repo=repo,
            shared=shared,
            canonical=shared / "canonical" / "v2",
            local=local,
            local_models=local / "models",
            local_indexes=local / "indexes",
            local_runs=local / "runs",
        )
