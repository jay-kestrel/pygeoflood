"""Result serialization: per-run JSON and a deterministic markdown summary.

The markdown writer is a pure function of the result dicts — same results
in, same bytes out — so committed baselines diff cleanly.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .schema import BYPASSED_STAGES, IO_KINDS, SCHEMA_VERSION, STAGES

VARIANCE_THRESHOLD_PCT = 10.0


def write_run_json(result: dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")


def summarize_variance(
    durations: list[float], threshold_pct: float = VARIANCE_THRESHOLD_PCT
) -> dict | None:
    """Spread of repeated end-to-end runs, as a percentage of their mean.

    Returns None for fewer than two runs (nothing to compare).
    """
    if len(durations) < 2:
        return None
    mean = sum(durations) / len(durations)
    spread_pct = (max(durations) - min(durations)) / mean * 100.0
    return {
        "runs_s": list(durations),
        "mean_s": mean,
        "spread_pct": spread_pct,
        "threshold_pct": threshold_pct,
        "within_threshold": spread_pct <= threshold_pct,
    }


def _fmt(seconds: float | None, digits: int = 2) -> str:
    if seconds is None:
        return "—"
    return f"{seconds:.{digits}f}"


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def _aggregate_stages(results: list[dict]) -> dict[str, dict[str, float | None]]:
    """Per-stage cold and warm (median) durations in canonical order."""
    columns: dict[str, dict[str, list[float]]] = {"cold": {}, "warm": {}}
    for result in results:
        mode = result["mode"]
        for stage in result["stages"]:
            if stage["duration_s"] is not None:
                columns[mode].setdefault(stage["name"], []).append(stage["duration_s"])

    aggregated: dict[str, dict[str, float | None]] = {}
    for name in STAGES:
        cold = columns["cold"].get(name)
        warm = columns["warm"].get(name)
        if cold or warm:
            aggregated[name] = {
                "cold_s": _median(cold) if cold else None,
                "warm_s": _median(warm) if warm else None,
            }
    return aggregated


def _aggregate_wbt(results: list[dict]) -> dict[str, dict[str, Any]]:
    """Per-WBT-tool totals: median seconds per mode, calls per run, host stage."""
    by_tool: dict[str, dict[str, list[float]]] = {}
    stages: dict[str, str] = {}
    calls_per_run: dict[str, int] = {}
    for result in results:
        totals: dict[str, float] = {}
        counts: dict[str, int] = {}
        for call in result["wbt_calls"]:
            tool = call["tool"]
            totals[tool] = totals.get(tool, 0.0) + call["duration_s"]
            counts[tool] = counts.get(tool, 0) + 1
            if call.get("stage"):
                stages[tool] = call["stage"]
        mode = result["mode"]
        for tool, seconds in totals.items():
            by_tool.setdefault(tool, {}).setdefault(mode, []).append(seconds)
        calls_per_run.update(counts)

    return {
        tool: {
            "cold_s": _median(modes["cold"]) if "cold" in modes else None,
            "warm_s": _median(modes["warm"]) if "warm" in modes else None,
            "calls_per_run": calls_per_run.get(tool, 0),
            "stage": stages.get(tool),
        }
        for tool, modes in by_tool.items()
    }


def _aggregate_kernels(results: list[dict]) -> dict[str, dict[str, Any]]:
    """Per-kernel totals: cold (includes JIT compile) vs warm medians."""
    by_kernel: dict[str, dict[str, list[float]]] = {}
    stages: dict[str, str] = {}
    for result in results:
        totals: dict[str, float] = {}
        for entry in result["kernels"]:
            kernel = entry["kernel"]
            totals[kernel] = totals.get(kernel, 0.0) + entry["duration_s"]
            if entry.get("stage"):
                stages[kernel] = entry["stage"]
        mode = result["mode"]
        for kernel, seconds in totals.items():
            by_kernel.setdefault(kernel, {}).setdefault(mode, []).append(seconds)

    return {
        kernel: {
            "cold_s": _median(modes["cold"]) if "cold" in modes else None,
            "warm_s": _median(modes["warm"]) if "warm" in modes else None,
            "stage": stages.get(kernel),
        }
        for kernel, modes in by_kernel.items()
    }


def _aggregate_io(results: list[dict]) -> dict[str, dict[str, Any]]:
    """Per-I/O-kind medians and warm-run call counts."""
    aggregated: dict[str, dict[str, Any]] = {}
    for kind in IO_KINDS:
        cold_values, warm_values, warm_counts = [], [], []
        for result in results:
            cold_values.append(result["io"][f"{kind}_seconds"])
            if result["mode"] == "warm":
                warm_values.append(result["io"][f"{kind}_seconds"])
                warm_counts.append(result["io"][f"{kind}_count"])
        aggregated[kind] = {
            "cold_s": _median(cold_values),
            "warm_s": _median(warm_values) if warm_values else None,
            "warm_count": warm_counts[-1] if warm_counts else 0,
        }
    return aggregated


def render_summary(results: list[dict], title: str) -> str:
    """Deterministic markdown summary for a list of run results (one
    profile, one harness invocation)."""
    first = results[0]
    manifest = first["manifest"]
    dem_sha = manifest.get("dem_sha256")
    dem_ref = f"`{dem_sha[:12]}…`" if dem_sha else "`n/a`"
    lines: list[str] = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append(
        f"Schema version: {SCHEMA_VERSION} · profile: `{first['profile']}` · "
        f"DEM sha256: {dem_ref}"
    )
    if manifest.get("crop"):
        crop = manifest["crop"]
        lines.append(
            f"Crop window: origin (row {crop['origin_row']}, col {crop['origin_col']}), "
            f"{crop['height']}x{crop['width']} cells"
        )
    lines.append("")

    lines.append("## End-to-end")
    lines.append("")
    lines.append("| mode | iteration | wall clock (s) |")
    lines.append("|---|---|---|")
    for result in results:
        lines.append(
            f"| {result['mode']} | {result['iteration']} | {_fmt(result['wall_clock_s'])} |"
        )
    lines.append("")

    lines.append("## Run-to-run variance")
    lines.append("")
    variance = first["variance"]
    if variance:
        verdict = "within" if variance["within_threshold"] else "EXCEEDS"
        lines.append(
            f"Warm-run spread {variance['spread_pct']:.2f}% "
            f"(mean {variance['mean_s']:.2f} s, {len(variance['runs_s'])} runs) — "
            f"{verdict} the {variance['threshold_pct']:.0f}% guidance."
        )
    else:
        lines.append(
            "Fewer than two warm runs — no variance computed for this invocation."
        )
    lines.append("")

    lines.append("## Per-stage timings (seconds)")
    lines.append("")
    lines.append(
        "Bypassed in upstream `run_fim_workflow()` at 3650c9b (geodesic path, "
        f"upstream issue #12): {', '.join(BYPASSED_STAGES)}. "
        "`rasterize_custom_flowline` is conditional upstream and does not run in "
        "the canonical workload (no custom flowline input)."
    )
    lines.append("")
    lines.append("| stage | cold | warm (median) | status |")
    lines.append("|---|---|---|---|")
    statuses = {stage["name"]: stage["status"] for stage in first["stages"]}
    for name, row in _aggregate_stages(results).items():
        lines.append(
            f"| {name} | {_fmt(row['cold_s'])} | {_fmt(row['warm_s'])} | "
            f"{statuses[name]} |"
        )
    lines.append("")

    lines.append("## WhiteboxTools calls (C++ subprocesses)")
    lines.append("")
    lines.append("| tool | host stage | calls/run | cold (s) | warm (s) |")
    lines.append("|---|---|---|---|---|")
    for tool, row in _aggregate_wbt(results).items():
        lines.append(
            f"| {tool} | {row['stage'] or '—'} | {row['calls_per_run']} | "
            f"{_fmt(row['cold_s'])} | {_fmt(row['warm_s'])} |"
        )
    lines.append("")

    lines.append("## Python/numba kernels")
    lines.append("")
    lines.append(
        "Cold timings include the one-off numba JIT compile; the cold-vs-warm "
        "delta is the JIT tax that an AOT port removes."
    )
    lines.append("")
    lines.append("| kernel | host stage | cold (s) | warm (s) |")
    lines.append("|---|---|---|---|")
    for kernel, row in _aggregate_kernels(results).items():
        lines.append(
            f"| {kernel} | {row['stage'] or '—'} | {_fmt(row['cold_s'])} | "
            f"{_fmt(row['warm_s'])} |"
        )
    lines.append("")

    lines.append("## Raster/CSV/vector I/O")
    lines.append("")
    lines.append("| kind | warm calls/run | cold (s) | warm (s) |")
    lines.append("|---|---|---|---|")
    for kind, row in _aggregate_io(results).items():
        lines.append(
            f"| {kind} | {row['warm_count']} | {_fmt(row['cold_s'])} | "
            f"{_fmt(row['warm_s'])} |"
        )
    lines.append("")

    lines.append("## Environment manifest")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(manifest, indent=2, sort_keys=True))
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


def write_summary(results: list[dict], path: Path, title: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(render_summary(results, title))
