"""Environment manifest collection for benchmark runs.

Every result embeds the manifest so numbers are only ever interpreted
against the machine and library versions that produced them. The DEM is
identified by sha256 — same bytes, same workload.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from importlib import metadata
from pathlib import Path

from .schema import REQUIRED_MANIFEST_KEYS

# Package versions recorded on every run. pygeoflood itself is included so
# a baseline is never mistaken for a port measurement.
PACKAGES = [
    "pygeoflood",
    "numpy",
    "numba",
    "scipy",
    "scikit-image",
    "scikit-fmm",
    "rasterio",
    "geopandas",
    "pandas",
    "xarray",
    "netcdf4",
    "fiona",
    "shapely",
    "whitebox",
    "psutil",
]


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Stream a file through sha256 without loading it whole."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def total_ram_bytes() -> int | None:
    """Total physical RAM via psutil, with a /proc/meminfo fallback."""
    try:
        import psutil

        return psutil.virtual_memory().total
    except ImportError:
        pass
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


def cpu_model() -> str | None:
    """CPU model string on Linux; None elsewhere (recorded as null)."""
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return None


def numba_threads() -> dict:
    """Numba's actual thread-pool size and the env var that set it."""
    try:
        import numba

        return {
            "numba_num_threads": numba.config.NUMBA_NUM_THREADS,
            "numba_env_threads": os.environ.get("NUMBA_NUM_THREADS"),
        }
    except ImportError:
        return {"numba_num_threads": None, "numba_env_threads": None}


def wbt_versions() -> dict:
    """WhiteboxTools binary version (runs the binary) and pip package version."""
    binary_version = None
    try:
        from whitebox import WhiteboxTools

        binary_version = WhiteboxTools().version()
    except Exception as error:  # binary download failure is a manifest fact
        binary_version = f"unavailable: {error}"
    try:
        package_version = metadata.version("whitebox")
    except metadata.PackageNotFoundError:
        package_version = None
    return {
        "wbt_binary_version": binary_version,
        "wbt_package_version": package_version,
        "wbt_max_procs": wbt_max_procs(),
    }


def wbt_max_procs() -> int | str:
    """Effective WhiteboxTools thread pin, read from the binary's own
    settings.json (the per-tool subprocesses read it from there).
    The pip lib's get_max_procs() reads a cwd-relative settings.json,
    which misreports -1 whenever the process cwd is not the exe dir.
    """
    try:
        from whitebox.whitebox_tools import WhiteboxTools

        exe_dir = Path(WhiteboxTools().exe_path)
        return json.loads((exe_dir / "settings.json").read_text())["max_procs"]
    except (OSError, ValueError, KeyError) as error:
        return f"unavailable: {error}"


def gnucobol_version() -> str | None:
    """GnuCOBOL compiler version from `cobc --version`, or None when the
    compiler is not installed (probed like wbt_versions: absence is a
    manifest fact, never an error)."""
    try:
        output = subprocess.run(
            ["cobc", "--version"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    match = re.search(r"(\d+\.\d+\.\d+)", output)
    return match.group(1) if match else None


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def git_commit(repo_root: Path) -> str | None:
    """Best-effort HEAD sha of the checkout the harness ran from."""
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
    except Exception:
        return None


def dem_info(dem_path: Path) -> dict:
    """Grid metadata for the benchmark DEM (or the crop)."""
    import rasterio

    with rasterio.open(dem_path) as src:
        return {
            "path": str(dem_path),
            "shape": [src.height, src.width],
            "crs": str(src.crs),
            "resolution": [src.res[0], src.res[1]],
            "cell_count": src.height * src.width,
            "dtype": str(src.dtypes[0]),
            "nodata": src.nodata,
            "bounds": [src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top],
        }


def collect_manifest(dem_path: Path, repo_root: Path, crop: dict | None) -> dict:
    """Assemble the full environment manifest (all REQUIRED_MANIFEST_KEYS)."""
    import time

    threads = numba_threads()
    manifest = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "platform": platform.platform(),
        "python_version": sys.version,
        "cpu_count": os.cpu_count(),
        "cpu_model": cpu_model(),
        "ram_total_bytes": total_ram_bytes(),
        **threads,
        **wbt_versions(),
        # Which backend produced the anisodiff kernel rows in this run,
        # plus the COBOL toolchain version when installed. Optional keys
        # (schema allows extras) so baselines stay valid.
        "implementation": (
            "cobol" if os.environ.get("PYGEOFLOOD_COBOL") == "1" else "rust"
        ),
        "gnucobol_version": gnucobol_version(),
        "packages": package_versions(),
        "git_commit": git_commit(repo_root),
        "dem_sha256": sha256_file(dem_path),
        "dem": dem_info(dem_path),
        "crop": crop,
    }
    missing = [key for key in REQUIRED_MANIFEST_KEYS if key not in manifest]
    if missing:
        raise RuntimeError(f"manifest incomplete, missing {missing}")
    return manifest
