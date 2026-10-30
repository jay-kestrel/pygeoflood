"""Tests for the manifest collector's WhiteboxTools probes.

The pip whitebox lib's version() RETURNS (rather than raises) an OSError
object when the binary is missing — feeding that object into the result
manifest crashed json.dump mid-write. The probes must also point at the
SAME binary the benchmark runs use (WBT_PATH), not the pip package dir.
"""

from benchmarks import manifest


class _FakeWBT:
    def __init__(self, version_return="WhiteboxTools v2.4.0 (c) test"):
        self.exe_path = "/fake/exe/dir"
        self.version_return = version_return
        self.set_dir_calls = []

    def set_whitebox_dir(self, path_str):
        self.set_dir_calls.append(path_str)

    def version(self):
        return self.version_return


def test_wbt_versions_records_string_version(monkeypatch):
    fake = _FakeWBT()
    monkeypatch.setattr("whitebox.WhiteboxTools", lambda: fake)
    versions = manifest.wbt_versions()
    assert versions["wbt_binary_version"] == "WhiteboxTools v2.4.0 (c) test"


def test_wbt_versions_coerces_returned_exception(monkeypatch):
    # The whitebox lib returns a FileNotFoundError object when the
    # binary is absent — it must land in the manifest as a string.
    fake = _FakeWBT(version_return=FileNotFoundError("no binary here"))
    monkeypatch.setattr("whitebox.WhiteboxTools", lambda: fake)
    versions = manifest.wbt_versions()
    assert isinstance(versions["wbt_binary_version"], str)
    assert versions["wbt_binary_version"].startswith("unavailable:")


def test_wbt_probes_honor_wbt_path(monkeypatch):
    fake = _FakeWBT()
    monkeypatch.setenv("WBT_PATH", "/opt/wbt-pinned")
    monkeypatch.setattr("whitebox.WhiteboxTools", lambda: fake)
    manifest.wbt_versions()
    # wbt_versions() builds handles twice (binary version + max_procs);
    # every one of them must have been pointed at WBT_PATH.
    assert fake.set_dir_calls, "probe never pointed the WBT handle anywhere"
    assert set(fake.set_dir_calls) == {"/opt/wbt-pinned"}


def test_wbt_max_procs_reads_settings_beside_wbt_path_binary(
    tmp_path, monkeypatch
):
    import json as jsonlib

    settings = tmp_path / "settings.json"
    settings.write_text(jsonlib.dumps({"max_procs": 1}))
    fake = _FakeWBT()
    fake.exe_path = str(tmp_path)
    monkeypatch.setenv("WBT_PATH", str(tmp_path))
    monkeypatch.setattr("whitebox.WhiteboxTools", lambda: fake)
    assert manifest.wbt_max_procs() == 1


def test_wbt_max_procs_reports_missing_settings_as_string(
    tmp_path, monkeypatch
):
    fake = _FakeWBT()
    fake.exe_path = str(tmp_path)  # no settings.json in here
    monkeypatch.setattr("whitebox.WhiteboxTools", lambda: fake)
    result = manifest.wbt_max_procs()
    assert isinstance(result, str)
    assert result.startswith("unavailable:")
