"""Numba cache=True variant toggle (spec D1 comparison baseline).

Re-jits the numba kernels with ``cache=True`` at runtime, so compiled
machine code is written to disk (``__pycache__`` beside the kernel
source) and fresh processes skip the one-off JIT compile after the
first-ever run has populated the cache. This is the second comparison
baseline for deliverable D4: alongside the numba-warm numbers, a cold
process on a warm disk cache measures the workload with no JIT compile
tax — the regime an AOT-compiled port lives in.

Like the timing instrumentation, this patches live module attributes —
the measured source stays byte-identical to upstream. All kernel call
sites resolve through ``pygeoflood.tools`` module globals at call time
(pygeoflood.py imports ``tools as t``; get_binary_hand and
catchhydrogeo call the kernels from within tools.py), so replacing the
module attribute is sufficient.
"""

from __future__ import annotations

from .schema import KERNELS


def enable_numba_cache() -> dict[str, str]:
    """Re-jit every numba kernel in ``pygeoflood.tools`` with cache=True.

    Returns a per-kernel report: ``"cached"`` (dispatcher re-jitted with
    cache=True), ``"not-numba"`` (no JIT compile to cache — the pure
    NumPy anisodiff), or ``"absent"`` (kernel not in this upstream
    version). Raises if a jitted kernel fails to re-jit: a silently
    non-cached variant would poison the baseline comparison.
    """
    import numba

    try:
        from numba.core.dispatcher import Dispatcher
    except ImportError:  # numba moved the class (defensive, pinned env)
        from numba.dispatcher import Dispatcher  # type: ignore[no-redef]

    from pygeoflood import tools as pgf_tools

    report: dict[str, str] = {}
    for name in KERNELS:
        original = getattr(pgf_tools, name, None)
        if original is None:
            report[name] = "absent"
            continue
        if not isinstance(original, Dispatcher):
            report[name] = "not-numba"
            continue
        # Preserve the kernel's own compile options (@jit(nopython=True,
        # parallel=True)); only add cache=True. Keys numba injects
        # internally (leading underscore) are dropped.
        options = {
            key: value
            for key, value in getattr(original, "targetoptions", {}).items()
            if not key.startswith("_")
        }
        if not options:
            options = {"nopython": True, "parallel": True}
        options["cache"] = True
        cached = numba.jit(**options)(original.py_func)
        setattr(pgf_tools, name, cached)
        report[name] = "cached"
    return report
