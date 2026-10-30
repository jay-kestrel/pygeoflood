"""Fail-fast wrapper for WhiteboxTools tool calls.

The whitebox 2.3.6 Python wrapper's ``run_tool`` never checks the child
process's exit status: a tool that runs and panics prints the panic through
its stdout callback and still returns 0. pygeoflood caches stage outputs
under deterministic filenames, so downstream stages used to read the
*previous* run's stale intermediates after a panicked stage. Every
``wbt.*`` call site routes through :func:`run_wbt`, which turns a failed
tool call into an exception.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from whitebox import WhiteboxTools

# Observed failure signatures of the WBT CLI: Rust panics print
# "thread 'main' panicked", user errors print capitalized "Problem"/
# "Error"/"Exception" banners — all while run_tool still returns 0.
_PANIC_MARKERS = ("error", "panic", "Problem", "Exception")


def run_wbt(wbt: "WhiteboxTools", tool: str, **kwargs) -> str:
    """Run a WhiteboxTools tool; raise on failure instead of returning 0.

    The WBT CLI reports panic through its stdout callback AND a nonzero
    exit code; the old call sites saw neither. This wrapper is the single
    choke point so a panicked stage aborts the pipeline instead of letting
    downstream stages read the previous run's stale intermediates.

    Parameters
    ----------
    wbt : WhiteboxTools
        Instance (or proxy) whose tool method to invoke. The instance's
        default callback, if any, still sees every output line.
    tool : str
        Tool method name, e.g. ``"fill_depressions"``.
    **kwargs
        Keyword arguments forwarded verbatim to the tool method. A
        ``callback`` kwarg, if present, is wrapped: it still sees every
        line, after the output has been captured.

    Returns
    -------
    str
        The captured tool output.

    Raises
    ------
    AttributeError
        If the instance has no tool named ``tool``.
    RuntimeError
        If the tool reports a nonzero exit code, or a panic marker appears
        in its output.
    """
    if not hasattr(wbt, tool):
        raise AttributeError(f"Unknown WhiteboxTools tool: {tool}")

    lines: list[str] = []
    user_callback = kwargs.pop("callback", None)
    default_callback = getattr(wbt, "default_callback", None)

    def collector(line: str) -> None:
        lines.append(line)
        if user_callback is not None:
            user_callback(line)
        elif default_callback is not None:
            default_callback(line)

    result = getattr(wbt, tool)(callback=collector, **kwargs)
    # normalize line endings only for the captured text; the chained
    # callback above received each raw line unchanged
    output = "".join(line if line.endswith("\n") else line + "\n" for line in lines)
    if isinstance(result, int) and result != 0:
        raise RuntimeError(
            f"WhiteboxTools '{tool}' failed with exit code {result}:\n{output}"
        )
    if any(marker in output for marker in _PANIC_MARKERS):
        raise RuntimeError(f"WhiteboxTools '{tool}' failed:\n{output}")
    return output
