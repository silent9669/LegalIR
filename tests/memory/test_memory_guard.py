import pytest
from src.core.memory import (
    MemorySnapshot,
    take_memory_snapshot,
    release_memory,
    check_memory_guard,
    format_memory_report,
)


def test_memory_snapshot():
    snap = take_memory_snapshot()
    assert snap.rss_bytes > 0
    assert snap.system_total_bytes > 0
    assert snap.system_available_bytes > 0
    assert isinstance(snap.to_dict(), dict)


def test_format_memory_report():
    snap = take_memory_snapshot()
    report = format_memory_report(snap, stage="test_stage")
    assert "test_stage" in report
    assert "RSS" in report
    assert "Available" in report


def test_memory_guard_passes_normal():
    # Calling check_memory_guard with small required memory should pass
    check_memory_guard(min_available_bytes=1024 * 1024, max_rss_fraction=0.99)


def test_memory_guard_raises_on_exhaustion(monkeypatch):
    from src.core import memory

    fake_snap = memory.MemorySnapshot(
        rss_bytes=100 * 1024**3,
        system_total_bytes=16 * 1024**3,
        system_available_bytes=500 * 1024**2,  # 500MB
        system_used_bytes=15 * 1024**3,
        gpu_allocated_bytes=0,
        gpu_reserved_bytes=0,
    )
    monkeypatch.setattr(memory, "take_memory_snapshot", lambda: fake_snap)
    with pytest.raises(MemoryError) as exc_info:
        check_memory_guard(min_available_bytes=3 * 1024**3, max_rss_fraction=0.70)
    assert "Low memory guard triggered" in str(exc_info.value)
