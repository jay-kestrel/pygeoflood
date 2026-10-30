"""Fail-fast WhiteboxTools regression tests (PG-1 finding F1).

The whitebox 2.3.6 Python wrapper's run_tool never checks the child
process's exit status: a tool that runs and panics prints the panic through
the callback stream and still returns 0 (whitebox_tools.py:563-649). Stage
outputs are cached under deterministic names, so downstream stages used to
read the *previous* run's stale intermediates. run_wbt is the single choke
point that must abort instead: raise RuntimeError on a nonzero exit code or
on panic markers in the tool's output stream.
"""

import pytest


class StubWBT:
    """WhiteboxTools stand-in mimicking the 2.3.6 wrapper contract.

    Tool methods forward an optional callback to run_tool; run_tool pumps
    each subprocess stdout line through it and returns the exit code — 0
    even when the binary panicked.
    """

    def __init__(self, output_lines=(), exit_code=0):
        self.default_callback = None
        self._output_lines = list(output_lines)
        self._exit_code = exit_code

    def watershed(self, callback=None, **kwargs):
        callback = callback if callback is not None else self.default_callback
        for line in self._output_lines:
            if callback is not None:
                callback(line)
        return self._exit_code


PANIC_OUTPUT = (
    "thread 'main' panicked at whitebox-tools-app/src/main.rs:72:21:",
    "note: run with `RUST_BACKTRACE=1` environment variable to display a "
    "backtrace",
)


def test_run_wbt_raises_on_panic_markers_even_with_zero_exit_code():
    """The review's live probe: watershed panics, the wrapper returns 0.

    Downstream stages must not proceed to read the previous run's stale
    intermediate, so run_wbt must raise."""
    from pygeoflood.wrapper import run_wbt  # new entry point under test

    wbt = StubWBT(output_lines=PANIC_OUTPUT, exit_code=0)
    with pytest.raises(RuntimeError, match="panicked"):
        run_wbt(
            wbt,
            "watershed",
            d8_pntr="d8.tif",
            pour_pts="pp.tif",
            output="out.tif",
        )


def test_run_wbt_raises_on_nonzero_exit_code():
    from pygeoflood.wrapper import run_wbt

    wbt = StubWBT(output_lines=[], exit_code=1)
    with pytest.raises(RuntimeError, match="exit code 1"):
        run_wbt(wbt, "watershed", d8_pntr="d8.tif")


def test_run_wbt_raises_on_unknown_tool():
    from pygeoflood.wrapper import run_wbt

    with pytest.raises(AttributeError, match="no_such_tool"):
        run_wbt(StubWBT(), "no_such_tool")


def test_run_wbt_chains_instance_callback():
    """The instance's own callback still sees every line — output behavior
    is unchanged, collection is additive."""
    from pygeoflood.wrapper import run_wbt

    seen = []
    wbt = StubWBT(output_lines=["line one", "line two"])
    wbt.default_callback = seen.append
    output = run_wbt(wbt, "watershed")
    assert seen == ["line one", "line two"]
    assert "line two" in output


def test_run_wbt_returns_clean_output():
    from pygeoflood.wrapper import run_wbt

    wbt = StubWBT(output_lines=["Elapsed time: 0.1s"])
    output = run_wbt(wbt, "watershed", d8_pntr="d8.tif")
    assert isinstance(output, str)
    assert "Elapsed time: 0.1s" in output
