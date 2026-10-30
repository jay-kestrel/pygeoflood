"""Runtime timing instrumentation for pyGeoFlood pipeline runs.

Every hook wraps live callables at runtime — instance-attribute shadowing
for pipeline stages, module-attribute patching for kernels and the
WhiteboxTools factory, class-attribute patching for CSV/vector writers and
a delegating proxy for rasterio datasets — so the measured source stays
byte-identical to what upstream ships. All wrappers are installed by
:class:`PipelineProbe` and fully removed on exit.
"""

from __future__ import annotations

import functools
import time
from typing import Any, Callable

import geopandas as gpd
import pandas as pd
import rasterio

from pygeoflood import tools as pgf_tools

from .schema import IO_KINDS, KERNELS, STAGES, WBT_TOOLS


class _TimedWBT:
    """Proxy that times each WhiteboxTools tool call (each is a subprocess).

    Only the known terrain-tool entry points are timed; configuration
    setters delegate untimed so the recorded calls are exactly the WBT
    subprocess work the spec wants separated.
    """

    def __init__(self, inner: Any, probe: "PipelineProbe"):
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_probe", probe)

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._inner, name)
        if not callable(attr) or name not in WBT_TOOLS:
            return attr

        probe = self._probe

        @functools.wraps(attr)
        def timed_tool(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            try:
                return attr(*args, **kwargs)
            finally:
                probe.record_wbt(name, time.perf_counter() - start)

        return timed_tool


class _TimedDataset:
    """Delegating rasterio dataset proxy timing read/write/close."""

    def __init__(self, dataset: Any, probe: "PipelineProbe", kind: str):
        object.__setattr__(self, "_ds", dataset)
        object.__setattr__(self, "_probe", probe)
        object.__setattr__(self, "_kind", kind)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._ds, name)

    def __enter__(self) -> "_TimedDataset":
        self._ds.__enter__()
        return self

    def __exit__(self, *exc_info: Any) -> bool | None:
        return self._ds.__exit__(*exc_info)

    def read(self, *args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        data = self._ds.read(*args, **kwargs)
        self._probe.record_io("raster_read", time.perf_counter() - start)
        return data

    def read_masks(self, *args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        data = self._ds.read_masks(*args, **kwargs)
        self._probe.record_io("raster_read", time.perf_counter() - start)
        return data

    def write(self, *args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        result = self._ds.write(*args, **kwargs)
        self._probe.record_io("raster_write", time.perf_counter() - start)
        return result

    def close(self) -> None:
        start = time.perf_counter()
        self._ds.close()
        self._probe.record_io(self._kind, time.perf_counter() - start)


class PipelineProbe:
    """Install per-run timers around a pyGeoFlood instance and collect them.

    Usage::

        probe = PipelineProbe()
        with probe.install(pgf):
            pgf.run_fim_workflow()
        result = probe.as_stages()  # per-stage durations and statuses
    """

    def __init__(self) -> None:
        self.stage_times: dict[str, list[float]] = {name: [] for name in STAGES}
        self.wbt_calls: list[dict[str, Any]] = []
        self.kernel_calls: list[dict[str, Any]] = []
        self.io: dict[str, dict[str, float]] = {
            kind: {"count": 0, "seconds": 0.0} for kind in IO_KINDS
        }
        self.failed_stage: str | None = None
        self._current_stage: str | None = None
        self._undo: list[Callable[[], None]] = []
        self._installed = False

    # -- recording ---------------------------------------------------------

    @property
    def current_stage(self) -> str | None:
        return self._current_stage

    def record_wbt(self, tool: str, seconds: float) -> None:
        self.wbt_calls.append(
            {"tool": tool, "stage": self._current_stage, "duration_s": seconds}
        )

    def record_kernel(self, kernel: str, seconds: float) -> None:
        self.kernel_calls.append(
            {"kernel": kernel, "stage": self._current_stage, "duration_s": seconds}
        )

    def record_io(self, kind: str, seconds: float) -> None:
        bucket = self.io[kind]
        bucket["count"] += 1
        bucket["seconds"] += seconds

    # -- installation ------------------------------------------------------

    def install(self, pgf: Any) -> "PipelineProbe":
        """Wrap all observation points around *pgf*. Returns self (usable as
        a context manager, which removes every wrapper on exit)."""
        if self._installed:
            raise RuntimeError("PipelineProbe.install called twice")
        self._installed = True

        for name in STAGES:
            self._wrap_stage(pgf, name)
        for kernel in KERNELS:
            self._wrap_kernel(kernel)
        self._patch_wbt_factory()
        self._patch_rasterio_open()
        self._patch_dataframe_to_csv()
        self._patch_geodataframe_to_file()
        return self

    def __enter__(self) -> "PipelineProbe":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.remove()

    def remove(self) -> None:
        while self._undo:
            self._undo.pop()()

    # -- wrappers ----------------------------------------------------------

    def _wrap_stage(self, pgf: Any, name: str) -> None:
        original = getattr(pgf, name)
        probe = self

        @functools.wraps(original)
        def timed_stage(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            try:
                with self._stage_context(name):
                    return original(*args, **kwargs)
            except Exception:
                probe.failed_stage = name
                raise
            finally:
                probe.stage_times[name].append(time.perf_counter() - start)

        pgf.__dict__[name] = timed_stage
        self._undo.append(
            lambda: pgf.__dict__.pop(name, None)
        )

    def _stage_context(self, name: str):
        """Attribute nested records (WBT calls, I/O, kernels) to the
        pipeline stage that is currently executing."""
        probe = self

        class _StageScope:
            def __enter__(self) -> None:
                probe._current_stage = name

            def __exit__(self, *exc_info: Any) -> None:
                probe._current_stage = None

        return _StageScope()

    def _wrap_kernel(self, kernel: str) -> None:
        original = getattr(pgf_tools, kernel, None)
        if original is None:  # kernel absent in this upstream version
            return
        probe = self

        @functools.wraps(original)
        def timed_kernel(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                probe.record_kernel(kernel, time.perf_counter() - start)

        setattr(pgf_tools, kernel, timed_kernel)
        self._undo.append(
            lambda: setattr(pgf_tools, kernel, original)
        )

    def _patch_wbt_factory(self) -> None:
        original_get = pgf_tools.get_WhiteboxTools
        probe = self

        @functools.wraps(original_get)
        def timed_get(*args: Any, **kwargs: Any) -> _TimedWBT:
            inner = original_get(*args, **kwargs)
            return _TimedWBT(inner, probe)

        pgf_tools.get_WhiteboxTools = timed_get
        self._undo.append(
            lambda: setattr(pgf_tools, "get_WhiteboxTools", original_get)
        )

    def _patch_rasterio_open(self) -> None:
        real_open = rasterio.open
        probe = self

        @functools.wraps(real_open)
        def timed_open(*args: Any, **kwargs: Any) -> _TimedDataset:
            mode = kwargs.get("mode")
            if mode is None and len(args) >= 2:
                mode = args[1]
            mode = mode or "r"
            kind = "raster_read" if mode == "r" else "raster_write"
            start = time.perf_counter()
            dataset = real_open(*args, **kwargs)
            probe.record_io(kind, time.perf_counter() - start)
            return _TimedDataset(dataset, probe, kind)

        rasterio.open = timed_open
        self._undo.append(lambda: setattr(rasterio, "open", real_open))

    def _patch_dataframe_to_csv(self) -> None:
        original = pd.DataFrame.to_csv
        probe = self

        @functools.wraps(original)
        def timed_to_csv(frame: pd.DataFrame, *args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            try:
                return original(frame, *args, **kwargs)
            finally:
                probe.record_io("csv_write", time.perf_counter() - start)

        pd.DataFrame.to_csv = timed_to_csv
        self._undo.append(lambda: setattr(pd.DataFrame, "to_csv", original))

    def _patch_geodataframe_to_file(self) -> None:
        original = gpd.GeoDataFrame.to_file
        probe = self

        @functools.wraps(original)
        def timed_to_file(frame: gpd.GeoDataFrame, *args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            try:
                return original(frame, *args, **kwargs)
            finally:
                probe.record_io("vector_write", time.perf_counter() - start)

        gpd.GeoDataFrame.to_file = timed_to_file
        self._undo.append(lambda: setattr(gpd.GeoDataFrame, "to_file", original))

    # -- results -----------------------------------------------------------

    def as_stages(self) -> list[dict[str, Any]]:
        """Per-stage entries in canonical order.

        A stage with no recorded time either did not run (skipped — only
        the conditional rasterize_custom_flowline qualifies when the
        workflow completed) or the workflow failed (failed)."""
        stages: list[dict[str, Any]] = []
        for name in STAGES:
            durations = self.stage_times[name]
            if durations:
                stages.append(
                    {"name": name, "duration_s": durations[0], "status": "timed"}
                )
            elif self.failed_stage is not None:
                status = "failed" if name == self.failed_stage else "skipped"
                stages.append({"name": name, "duration_s": None, "status": status})
            else:
                stages.append(
                    {"name": name, "duration_s": None, "status": "skipped"}
                )
        return stages

    def as_wbt_calls(self) -> list[dict[str, Any]]:
        return [dict(call) for call in self.wbt_calls]

    def as_kernels(self) -> list[dict[str, Any]]:
        """One entry per observed kernel invocation, in call order."""
        return [dict(call) for call in self.kernel_calls]
