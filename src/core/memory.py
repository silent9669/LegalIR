"""Host and GPU memory telemetry, garbage collection, and threshold guards."""

from __future__ import annotations

import ctypes
import gc
import os
import sys
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional
import psutil

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@dataclass
class MemorySnapshot:
    """Snapshot of process and system memory metrics."""

    rss_bytes: int
    system_total_bytes: int
    system_available_bytes: int
    system_used_bytes: int
    gpu_allocated_bytes: int = 0
    gpu_reserved_bytes: int = 0
    gpu_peak_allocated_bytes: int = 0
    gpu_peak_reserved_bytes: int = 0
    gpu_devices: Dict[int, Dict[str, int]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def take_memory_snapshot() -> MemorySnapshot:
    """Take a comprehensive snapshot of host RAM and CUDA VRAM."""
    process = psutil.Process(os.getpid())
    rss = process.memory_info().rss
    vm = psutil.virtual_memory()

    gpu_alloc = 0
    gpu_res = 0
    gpu_peak_alloc = 0
    gpu_peak_res = 0
    gpu_devices: Dict[int, Dict[str, int]] = {}

    if HAS_TORCH and torch.cuda.is_available():
        num_devices = torch.cuda.device_count()
        for d in range(num_devices):
            alloc = torch.cuda.memory_allocated(d)
            res = torch.cuda.memory_reserved(d)
            p_alloc = torch.cuda.max_memory_allocated(d)
            p_res = torch.cuda.max_memory_reserved(d)

            gpu_devices[d] = {
                "allocated_bytes": alloc,
                "reserved_bytes": res,
                "peak_allocated_bytes": p_alloc,
                "peak_reserved_bytes": p_res,
            }
            gpu_alloc += alloc
            gpu_res += res
            gpu_peak_alloc += p_alloc
            gpu_peak_res += p_res

    return MemorySnapshot(
        rss_bytes=rss,
        system_total_bytes=vm.total,
        system_available_bytes=vm.available,
        system_used_bytes=vm.used,
        gpu_allocated_bytes=gpu_alloc,
        gpu_reserved_bytes=gpu_res,
        gpu_peak_allocated_bytes=gpu_peak_alloc,
        gpu_peak_reserved_bytes=gpu_peak_res,
        gpu_devices=gpu_devices,
    )


def release_memory() -> None:
    """Force garbage collection, CUDA cache release, and C library heap trim."""
    gc.collect()

    if HAS_TORCH and torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Try Linux malloc_trim if on glibc
    if sys.platform.startswith("linux"):
        try:
            libc = ctypes.CDLL("libc.so.6")
            libc.malloc_trim(0)
        except Exception:
            pass


def format_memory_report(snap: MemorySnapshot, stage: str = "") -> str:
    """Format memory snapshot into a human-readable telemetry line."""
    to_gib = lambda b: f"{b / (1024**3):.2f} GiB"
    rss_pct = (snap.rss_bytes / snap.system_total_bytes) * 100 if snap.system_total_bytes > 0 else 0

    lines = [
        f"[{stage or 'Memory'}] Process RSS: {to_gib(snap.rss_bytes)} ({rss_pct:.1f}%), "
        f"Available: {to_gib(snap.system_available_bytes)} / Total: {to_gib(snap.system_total_bytes)}"
    ]

    if snap.gpu_devices:
        gpu_parts = []
        for d, stats in snap.gpu_devices.items():
            gpu_parts.append(
                f"GPU{d}: alloc {to_gib(stats['allocated_bytes'])} (peak {to_gib(stats['peak_allocated_bytes'])})"
            )
        lines.append(" | ".join(gpu_parts))

    return " | ".join(lines)


def check_memory_guard(
    min_available_bytes: int = 3 * 1024**3,  # 3 GiB default
    max_rss_fraction: float = 0.70,  # 70% physical RAM cap
    stage: str = "",
) -> None:
    """
    Assert that current memory usage satisfies the host RAM contract:
    - available RAM >= max(min_available_bytes, 10% of total)
    - peak RSS <= max_rss_fraction * total RAM
    If violated, triggers release_memory() and re-checks before raising MemoryError.
    """
    snap = take_memory_snapshot()
    threshold_available = max(min_available_bytes, int(0.10 * snap.system_total_bytes))
    threshold_rss = int(max_rss_fraction * snap.system_total_bytes)

    is_low_available = snap.system_available_bytes < threshold_available
    is_high_rss = snap.rss_bytes > threshold_rss

    if is_low_available or is_high_rss:
        release_memory()
        snap = take_memory_snapshot()
        is_low_available = snap.system_available_bytes < threshold_available
        is_high_rss = snap.rss_bytes > threshold_rss

        if is_low_available or is_high_rss:
            report = format_memory_report(snap, stage=f"Low Memory Guard Triggered ({stage})")
            raise MemoryError(
                f"Low memory guard triggered: {report}. "
                f"Required available >= {threshold_available / 1024**3:.2f} GiB, "
                f"Max RSS <= {threshold_rss / 1024**3:.2f} GiB"
            )
