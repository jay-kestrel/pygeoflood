"""ctypes binding for the COBOL anisodiff kernel (spec D1, art_QEojt4gd).

Loads ``cobol/lib/pygeoflood_cobol.so`` — the cobc-compiled kernel plus
shim.c — and mirrors the exact call the Rust port makes
(:func:`pygeoflood._rust.anisodiff`). Routing lives in
:mod:`pygeoflood.tools`; this module is imported eagerly but the .so is
loaded lazily, so the default (no COBOL) path never pays the load.

Contract notes carried over from the Rust port:

- No silent fallback: a missing or unloadable backend raises. Callers
  must see backend problems, not quietly different numbers.
- float32 in, float32 out; no silent float64 promotion through the shim.
- Strided, read-only, and Fortran-ordered inputs are normalized to a
  C-contiguous snapshot at the adapter (the Rust binding snapshots via
  ``view.iter()`` for the same reason).
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

import numpy as np

ENV_VAR = "PYGEOFLOOD_COBOL"

# Hard OCCURS ceiling in anisodiff.cob (GnuCOBOL table-size limit).
_MAX_ELEMENTS = 536_870_911

_RC_MESSAGES = {
    2: "out of memory allocating the COBOL scratch plane",
}


class CobolKernelError(RuntimeError):
    """The COBOL backend is unusable or a kernel call failed.

    Raised instead of falling back so a benchmark row can never quietly
    change backend (spec contract C2, no silent fallbacks).
    """


def enabled() -> bool:
    """True when PYGEOFLOOD_COBOL routes anisodiff to the COBOL kernel."""
    return os.environ.get(ENV_VAR) == "1"


def _library_paths() -> tuple[Path, ...]:
    """Candidate .so locations: packaged beside the module, then the repo.

    The editable/dev install resolves to <repo>/cobol/lib (what
    `make -C cobol lib` builds); a package-dir install would look inside
    pygeoflood/ itself.
    """
    here = Path(__file__).resolve().parent
    return (
        here / "lib" / "pygeoflood_cobol.so",
        here.parents[1] / "cobol" / "lib" / "pygeoflood_cobol.so",
    )


class _Kernel:
    """Lazily loaded, cached entry point into pygeoflood_cobol.so."""

    def __init__(self) -> None:
        candidates = _library_paths()
        path = next((p for p in candidates if p.exists()), candidates[0])
        if not path.exists():
            raise FileNotFoundError(
                f"{ENV_VAR}=1 but the COBOL backend is not built — run "
                "`make -C cobol lib` (requires GnuCOBOL 3); expected " + str(path)
            )
        try:
            # RTLD_GLOBAL puts the shim's symbols (cobol_expf/cobol_exp)
            # in the global namespace: the kernel resolves them through
            # the COBOL runtime's dynamic CALL machinery (dlsym), which
            # cannot see RTLD_LOCAL-loaded libraries.
            lib = ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
        except OSError as exc:  # broken build/env — surface it, never fall back
            raise CobolKernelError(f"failed to load {path}: {exc}") from exc
        try:
            entry = lib.anisodiff_cobol
        except AttributeError as exc:
            raise CobolKernelError(
                f"{path} does not export anisodiff_cobol — rebuild with "
                "`make -C cobol clean lib`"
            ) from exc
        entry.restype = ctypes.c_int
        entry.argtypes = [
            ctypes.POINTER(ctypes.c_float),  # plane (contiguous snapshot)
            ctypes.c_int,  # rows
            ctypes.c_int,  # cols
            ctypes.c_int,  # niter (> 0; tools.anisodiff short-circuits earlier)
            ctypes.c_int,  # option code: 0 linear, 1 PeronaMalik1, 2 PeronaMalik2
            ctypes.c_double,  # kappa (NumPy scalar as float64)
            ctypes.c_double,  # gamma (NumPy scalar as float64)
            ctypes.c_double,  # step[0]
            ctypes.c_double,  # step[1]
            ctypes.c_int,  # prec: explicit-float64 promotion bitmask
            ctypes.POINTER(ctypes.c_float),  # out
        ]
        self._entry = entry

    def __call__(
        self,
        plane: np.ndarray,
        rows: int,
        cols: int,
        niter: int,
        option_code: int,
        kappa: float,
        gamma: float,
        step_s: float,
        step_e: float,
        prec: int,
        out: np.ndarray,
    ) -> None:
        rc = self._entry(
            plane.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int(rows),
            ctypes.c_int(cols),
            ctypes.c_int(niter),
            ctypes.c_int(option_code),
            ctypes.c_double(kappa),
            ctypes.c_double(gamma),
            ctypes.c_double(step_s),
            ctypes.c_double(step_e),
            ctypes.c_int(prec),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )
        if rc != 0:
            raise CobolKernelError(
                f"COBOL anisodiff failed, rc={rc}: "
                f"{_RC_MESSAGES.get(rc, 'unrecognized exit status')}"
            )


_kernel: _Kernel | None = None


def _lib() -> _Kernel:
    """Process-wide cached kernel; the shim handles one-time cob_init."""
    global _kernel
    if _kernel is None:
        _kernel = _Kernel()
    return _kernel


def anisodiff(
    image: np.ndarray,
    niter: int,
    kappa: float,
    gamma: float,
    step: tuple[float, float],
    option: str,
    precision: tuple[bool, bool, bool, bool],
) -> np.ndarray:
    """Run the COBOL kernel; mirrors the Rust binding's call contract.

    ``image`` is the float32 (rows, cols) plane tools.anisodiff hands to
    ``_rust.anisodiff``; ``precision`` is the explicit-float64 promotion
    tuple ``(kappa, south step, east step, gamma)`` computed against
    NumPy's scalar-promotion rules at the call site.
    """
    plane = np.ascontiguousarray(image, dtype=np.float32)
    rows, cols = plane.shape
    if rows * cols > _MAX_ELEMENTS:
        raise CobolKernelError(
            f"plane of {rows * cols} elements exceeds the COBOL kernel's "
            f"{_MAX_ELEMENTS}-element ceiling"
        )
    out = np.empty_like(plane)
    if out.size == 0 or niter <= 0:
        # The kernel loop would be a no-op; match the reference's
        # zero-iteration result (a copy of the input) without a call.
        out[...] = plane
        return out
    option_code = {"PeronaMalik1": 1, "PeronaMalik2": 2}.get(option, 0)
    kappa_d, south_d, east_d, gamma_d = precision
    prec = int(kappa_d) | int(south_d) << 1 | int(east_d) << 2 | int(gamma_d) << 3
    _lib()(
        plane,
        rows,
        cols,
        int(niter),
        option_code,
        float(kappa),
        float(gamma),
        float(step[0]),
        float(step[1]),
        prec,
        out,
    )
    return out
