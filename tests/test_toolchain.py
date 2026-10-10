import io
import tarfile

import pytest

from candor import toolchain
from candor.toolchain import Asset, ToolchainError


def fake_release(tmp_path, monkeypatch, tamper: bool = False):
    """Serve a tarball holding llama-server and llama-quantize instead of downloading one."""
    archive = tmp_path / "release.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for tool in ("llama-server", "llama-quantize"):
            data = b"#!/bin/sh\n"
            info = tarfile.TarInfo(f"llama-b0/{tool}")
            info.size, info.mode = len(data), 0o755
            tar.addfile(info, io.BytesIO(data))
    digest = toolchain.sha256(archive)
    if tamper:
        archive.write_bytes(archive.read_bytes() + b"x")

    def retrieve(_url, destination):
        destination.write_bytes(archive.read_bytes())

    monkeypatch.setattr(toolchain.urllib.request, "urlretrieve", retrieve)
    monkeypatch.setattr(toolchain, "this_platform", lambda: "test")
    monkeypatch.setitem(toolchain.ASSETS, "test", [Asset("release.tar.gz", digest)])
    monkeypatch.setenv("CANDOR_CACHE", str(tmp_path / "cache"))
    monkeypatch.delenv("CANDOR_LLAMA_BIN", raising=False)


def test_binaries_are_fetched_once_and_checked(tmp_path, monkeypatch):
    fake_release(tmp_path, monkeypatch)
    folder = toolchain.binaries()
    assert (folder / "llama-server").exists()
    assert (folder / "llama-server").stat().st_mode & 0o100
    assert toolchain.binaries() == folder


def test_a_download_with_the_wrong_checksum_is_refused(tmp_path, monkeypatch):
    fake_release(tmp_path, monkeypatch, tamper=True)
    with pytest.raises(ToolchainError, match="sha256"):
        toolchain.binaries()
    assert not list(toolchain.cache().parent.rglob("llama-server"))


def test_an_existing_folder_can_be_used_instead(tmp_path, monkeypatch):
    for tool in ("llama-server", "llama-quantize"):
        (tmp_path / tool).touch()
    monkeypatch.setenv("CANDOR_LLAMA_BIN", str(tmp_path))
    assert toolchain.binaries() == tmp_path


def test_a_folder_missing_a_tool_is_refused(tmp_path, monkeypatch):
    (tmp_path / "llama-server").touch()
    monkeypatch.setenv("CANDOR_LLAMA_BIN", str(tmp_path))
    with pytest.raises(ToolchainError, match="llama-quantize"):
        toolchain.binaries()


def test_this_platform_is_known():
    assert toolchain.this_platform() in toolchain.ASSETS
