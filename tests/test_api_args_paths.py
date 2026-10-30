"""Argument-precedence and custom-path regression tests (PG-1: F7, F9).

F7: get_WhiteboxTools silently discarded its whitebox_exe_dir argument —
the WBT_PATH environment variable always won, so the documented
compiled-from-source HPC use case could not work. Pinned behavior: the
explicit argument wins; WBT_PATH is only the fallback.

F9: inundate unconditionally appended ".csv" to a custom flood-stage path,
so "my_stage.csv" was looked up as "my_stage.csv.csv".
"""

from pathlib import Path

import pytest

from pygeoflood.tools import get_WhiteboxTools


def test_get_whitebox_tools_explicit_dir_wins_over_env(monkeypatch):
    monkeypatch.setenv("WBT_PATH", "/env/wbt/dir")
    wbt = get_WhiteboxTools(whitebox_exe_dir="/explicit/wbt/dir")
    assert wbt.exe_path == "/explicit/wbt/dir"


def test_get_whitebox_tools_falls_back_to_env_without_argument(monkeypatch):
    monkeypatch.setenv("WBT_PATH", "/env/wbt/dir")
    wbt = get_WhiteboxTools()
    assert wbt.exe_path == "/env/wbt/dir"


def test_ensure_csv_suffix_appends_when_missing():
    from pygeoflood.tools import ensure_csv_suffix  # new helper under test

    assert ensure_csv_suffix("my_stage") == "my_stage.csv"


def test_ensure_csv_suffix_no_double_suffix():
    from pygeoflood.tools import ensure_csv_suffix

    assert ensure_csv_suffix("my_stage.csv") == "my_stage.csv"


def test_ensure_csv_suffix_keeps_existing_other_suffix():
    from pygeoflood.tools import ensure_csv_suffix

    assert ensure_csv_suffix("my_stage.txt") == "my_stage.txt"


def test_ensure_csv_suffix_accepts_pathlike():
    from pygeoflood.tools import ensure_csv_suffix

    assert ensure_csv_suffix(Path("/data/stage")) == "/data/stage.csv"
