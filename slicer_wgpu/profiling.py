"""Performance profiling utilities for slicer_wgpu.

Provides CPU frame timing, GPU shadow-compute timing (via timestamp
queries when available), and a step-count heatmap debug mode.

Usage::

    from slicer_wgpu.profiling import FrameProfiler

    profiler = FrameProfiler()
    # ... in your render loop:
    profiler.begin_frame()
    renderer.render(scene, camera)
    profiler.end_frame()
    print(profiler.report())
"""

from __future__ import annotations

import time
from collections import deque


class FrameProfiler:
    """Collects per-frame timing statistics.

    Tracks CPU wall-clock frame time and, when a ShadowVolume with
    timestamp-query support is provided, GPU shadow-compute time.
    """

    def __init__(self, history: int = 120):
        self._history = history
        self._cpu_times: deque[float] = deque(maxlen=history)
        self._shadow_times: deque[int] = deque(maxlen=history)
        self._frame_start: float = 0.0
        self._shadow_volume = None

    def set_shadow_volume(self, sv) -> None:
        """Attach a ShadowVolume to collect GPU timing from."""
        self._shadow_volume = sv

    def begin_frame(self) -> None:
        """Call immediately before the render dispatch."""
        self._frame_start = time.perf_counter()

    def end_frame(self) -> None:
        """Call immediately after the render dispatch returns."""
        elapsed_ms = (time.perf_counter() - self._frame_start) * 1000.0
        self._cpu_times.append(elapsed_ms)
        if self._shadow_volume is not None:
            gpu_ns = self._shadow_volume.last_gpu_time_ns
            if gpu_ns is not None:
                self._shadow_times.append(gpu_ns)

    def report(self) -> dict:
        """Return summary statistics.

        Keys:
            cpu_frame_ms_avg, cpu_frame_ms_p95, cpu_frame_ms_last,
            shadow_gpu_ms_avg, shadow_gpu_ms_last,
            n_frames
        """
        out: dict = {"n_frames": len(self._cpu_times)}
        if self._cpu_times:
            times = list(self._cpu_times)
            out["cpu_frame_ms_avg"] = sum(times) / len(times)
            out["cpu_frame_ms_last"] = times[-1]
            sorted_t = sorted(times)
            out["cpu_frame_ms_p95"] = sorted_t[int(len(sorted_t) * 0.95)]
        if self._shadow_times:
            stimes = list(self._shadow_times)
            out["shadow_gpu_ms_avg"] = sum(stimes) / len(stimes) / 1e6
            out["shadow_gpu_ms_last"] = stimes[-1] / 1e6
        return out

    def reset(self) -> None:
        self._cpu_times.clear()
        self._shadow_times.clear()

    @staticmethod
    def enable_step_heatmap(renderer) -> None:
        """Turn on step-count heatmap visualization."""
        renderer.material.debug_step_count = 1.0

    @staticmethod
    def disable_step_heatmap(renderer) -> None:
        """Restore normal rendering."""
        renderer.material.debug_step_count = 0.0
