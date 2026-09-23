"""Score-averaged cross-encoder ensemble for private inference.

OOF/disjoint evaluation stays single-adapter (honest, leak-free). Only the
final private submission inference averages member scores:

    members = [final_adapter, fold_0_adapter, ..., fold_4_adapter]

Each member is an independent CrossEncoderReranker (own base + LoRA weights);
the wrapper exposes the identical scoring interface via ``score_fn`` injection,
so ``rerank_batch``/aggregation/fusion code paths are untouched.

Members must all share the same base model id; adapters may differ in rank,
objective, or training population — heterogeneity is the point.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable


def _mean(scores_list: list[list[float]]) -> list[float]:
    n = len(scores_list[0])
    out = [0.0] * n
    for scores in scores_list:
        if len(scores) != n:
            raise ValueError(
                f"Ensemble member returned {len(scores)} scores for {n} pairs"
            )
        for i, s in enumerate(scores):
            out[i] += float(s)
    return [v / len(scores_list) for v in out]


def build_ensemble_score_fn(
    members: list[Any],
) -> Callable[..., list[float]]:
    """Return a ``score_fn(pairs, batch_size, max_length)`` averaging members.

    The wrapper records lightweight per-member timing on
    ``_score_fn.timings`` (durations/pair counts only; never pair text) so a
    pilot can rank ensemble overhead by measurement. Scoring stays strictly
    sequential and output-identical; no token/forward reuse is attempted here.
    """
    if not members:
        raise ValueError("Ensemble needs at least one member reranker")

    def _score_fn(
        pairs: list[tuple[str, str]],
        batch_size: int | None = None,
        max_length: int | None = None,
    ) -> list[float]:
        import time as _time

        per_member = []
        timings = getattr(_score_fn, "timings", None)
        if not isinstance(timings, dict):
            timings = {"calls": 0, "pairs_scored": 0, "total_seconds": 0.0,
                       "per_member_seconds": [0.0] * len(members)}
            _score_fn.timings = timings  # type: ignore[attr-defined]
        call_t0 = _time.perf_counter()
        for i, m in enumerate(members):
            t0 = _time.perf_counter()
            per_member.append(m.score_pairs(pairs, batch_size=batch_size, max_length=max_length))
            dt = _time.perf_counter() - t0
            try:
                timings["per_member_seconds"][i] += dt
            except Exception:
                pass
        timings["calls"] += 1
        timings["pairs_scored"] += len(pairs)
        timings["total_seconds"] += _time.perf_counter() - call_t0
        return _mean(per_member)

    _score_fn.timings = {"calls": 0, "pairs_scored": 0, "total_seconds": 0.0,  # type: ignore[attr-defined]
                         "per_member_seconds": [0.0] * len(members)}
    return _score_fn


def build_ensemble_reranker(
    reranker_cls: Any,
    *,
    model_name: str,
    adapter_paths: list[str | Path],
    device: Any = None,
    batch_size: int = 128,
    max_length: int = 512,
    precision: str | None = None,
    revision: str | None = None,
) -> Any:
    """Build one reranker object whose scores average ``adapter_paths`` members.

    Returns a ``reranker_cls`` instance with ``score_fn`` injected, so every
    downstream consumer (rerank_batch, aggregation, RRF fusion, selectors)
    works unchanged. Raises if fewer than 2 valid adapter dirs are found.
    """
    valid = [Path(p) for p in adapter_paths if p and Path(p).is_dir()]
    if len(valid) < 2:
        raise ValueError(
            f"Ensemble needs >=2 adapter dirs, found {len(valid)} in {adapter_paths}"
        )
    members = [
        reranker_cls(
            model_name=model_name,
            adapter_path=d,
            device=device,
            batch_size=batch_size,
            max_length=max_length,
            precision=precision,
            revision=revision,
        )
        for d in valid
    ]
    for m in members:
        m.ensure_loaded()
    wrapper = reranker_cls(
        model_name=model_name,
        device=device,
        batch_size=batch_size,
        max_length=max_length,
        precision=precision,
        revision=revision,
        score_fn=build_ensemble_score_fn(members),
    )
    wrapper.ensemble_members = members  # type: ignore[attr-defined]
    wrapper.ensemble_adapter_paths = [str(d) for d in valid]  # type: ignore[attr-defined]
    # Representative handles so parameter audits, strict artifact checks, and
    # fresh-reload probes keep working unchanged (member[0] is the final adapter).
    wrapper.model = members[0].model  # type: ignore[attr-defined]
    wrapper.tokenizer = members[0].tokenizer  # type: ignore[attr-defined]
    wrapper.adapter_path = members[0].adapter_path  # type: ignore[attr-defined]
    return wrapper


def resolve_ensemble_members(
    final_adapter_dir: str | Path | None,
    checkpoints_dir: str | Path | None = None,
    num_folds: int = 5,
    max_members: int = 6,
    oof_cv_dir: str | Path | None = None,
) -> list[str]:
    """Collect [final, fold_0..fold_4] adapter dirs that exist on disk.

    Fold adapters live under the OOF cv dir (``<working>/cv/fold_<i>/
    reranker_adapter``); ``checkpoints_dir`` is a legacy fallback location.
    """
    members: list[str] = []
    if final_adapter_dir and Path(final_adapter_dir).is_dir():
        members.append(str(final_adapter_dir))
    search_roots = [d for d in (oof_cv_dir, checkpoints_dir) if d]
    for root in search_roots:
        for i in range(num_folds):
            for cand in (
                Path(root) / f"fold_{i}" / "reranker_adapter",
                Path(root) / f"fold_{i}",
            ):
                if cand.is_dir() and (cand / "adapter_config.json").is_file():
                    members.append(str(cand))
                    break
            if len(members) >= max_members:
                break
        if len(members) >= max_members:
            break
    # De-dup preserving order.
    seen: set[str] = set()
    out: list[str] = []
    for m in members[:max_members]:
        if m not in seen:
            seen.add(m)
            out.append(m)
    return out
