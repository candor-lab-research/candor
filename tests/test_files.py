import fcntl
from pathlib import Path

import pytest

from candor import config, files

from .test_config import settings


def recorded_syncs(monkeypatch) -> list[tuple[str, str]]:
    """Each file flushed, by name, with what it held then."""
    synced = []
    real = files.sync

    def sync(path: Path) -> None:
        synced.append((path.name, path.read_text()))
        real(path)

    monkeypatch.setattr(files, "sync", sync)
    return synced


@pytest.mark.skipif(not hasattr(fcntl, "F_FULLFSYNC"), reason="only macOS has F_FULLFSYNC")
def test_results_are_fully_flushed_on_macos(tmp_path, monkeypatch):
    calls = []
    real = fcntl.fcntl

    def recording(descriptor, command, *args):
        calls.append(command)
        return real(descriptor, command, *args)

    monkeypatch.setattr(fcntl, "fcntl", recording)
    (tmp_path / "result").write_text("x")
    files.sync(tmp_path / "result")
    assert calls.count(fcntl.F_FULLFSYNC) >= 2


def test_a_file_is_on_the_disk_whole_before_it_replaces_the_old_and_after(tmp_path, monkeypatch):
    synced = recorded_syncs(monkeypatch)
    path = tmp_path / "result.json"
    path.write_text("old")
    files.write_atomic(path, "new")
    assert synced == [("result.json.building", "new"), ("result.json", "new")]
    assert [p.name for p in tmp_path.iterdir()] == ["result.json"]


def test_a_write_that_fails_leaves_the_old_file_and_nothing_else(tmp_path, monkeypatch):
    path = tmp_path / "result.json"
    path.write_text("old")

    def full(path: Path) -> None:
        raise OSError("no space left")

    monkeypatch.setattr(files, "sync", full)
    with pytest.raises(OSError, match="no space left"):
        files.write_atomic(path, "new")
    assert path.read_text() == "old" and [p.name for p in tmp_path.iterdir()] == ["result.json"]


def test_a_runs_settings_are_on_the_disk_once_rewritten(tmp_path, monkeypatch):
    config.write(tmp_path, "run", settings(), "h")
    synced = recorded_syncs(monkeypatch)
    config.rewrite(tmp_path / "run", settings(), "upgraded")
    assert [name for name, _ in synced] == [f"{config.FILE}.building", config.FILE]
