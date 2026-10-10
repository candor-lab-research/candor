import io
import os
import shutil
import subprocess
import sys
import tarfile
import time

import pytest

from candor import toolchain
from candor.toolchain import Asset, ToolchainError

SERVER = "#!/bin/sh\necho 'version: 0.6.0-dev (build {build}, commit 86a2835)'\n"


def tarball(path, files: dict[str, str]) -> None:
    with tarfile.open(path, "w:gz") as tar:
        for name, text in files.items():
            data = text.encode()
            info = tarfile.TarInfo(name)
            info.size, info.mode = len(data), 0o755
            tar.addfile(info, io.BytesIO(data))


@pytest.fixture
def release(tmp_path, monkeypatch):
    """Serve fake release archives instead of downloading them, and count the downloads."""
    archives = {}
    downloads = []

    def add(name: str, files: dict[str, str]) -> Asset:
        archives[name] = tmp_path / name
        tarball(archives[name], files)
        return Asset(name, toolchain.sha256(archives[name]))

    def download(url, destination):
        downloads.append(url)
        destination.write_bytes(archives[url.rsplit("/", 1)[1]].read_bytes())

    monkeypatch.setattr(toolchain, "_download", download)
    monkeypatch.setattr(toolchain, "this_platform", lambda: "test")
    monkeypatch.setenv("CANDOR_CACHE", str(tmp_path / "cache"))
    monkeypatch.delenv("CANDOR_LLAMA_BIN", raising=False)
    return add, downloads, archives


def binaries_release(add, build="11524"):
    return add(
        "llama.tar.gz",
        {"llama-b0/llama-server": SERVER.format(build=build), "llama-b0/llama-quantize": "#!/bin/sh\n"},
    )


def test_binaries_are_fetched_once_and_checked(release, monkeypatch):
    add, downloads, _ = release
    monkeypatch.setitem(toolchain.ASSETS, "test", [binaries_release(add)])
    folder = toolchain.binaries()
    assert (folder / "llama-server").stat().st_mode & 0o100
    assert toolchain.binaries() == folder
    assert len(downloads) == 1


def test_a_second_archive_lands_beside_llama_server(release, monkeypatch):
    add, _, _ = release
    runtime = add("cudart.tar.gz", {"cudart-b0/libcudart.so.12": "lib", "cudart-b0/libcublas.so.12": "lib"})
    monkeypatch.setitem(toolchain.ASSETS, "test", [binaries_release(add), runtime])
    folder = toolchain.binaries()
    assert (folder / "libcudart.so.12").exists() and (folder / "libcublas.so.12").exists()


def test_a_download_with_the_wrong_checksum_is_refused_and_leaves_nothing(release, monkeypatch):
    add, _, archives = release
    asset = binaries_release(add)
    archives[asset.name].write_bytes(archives[asset.name].read_bytes() + b"x")
    monkeypatch.setitem(toolchain.ASSETS, "test", [asset])
    with pytest.raises(ToolchainError, match="sha256"):
        toolchain.binaries()
    assert not list(toolchain.cache().rglob("llama-server"))


def test_a_different_release_is_refused(release, monkeypatch):
    add, _, _ = release
    monkeypatch.setitem(toolchain.ASSETS, "test", [binaries_release(add, build="9999")])
    with pytest.raises(ToolchainError, match=r"isn't llama\.cpp b11524"):
        toolchain.binaries()


def test_an_existing_folder_must_hold_the_same_release(tmp_path, monkeypatch):
    (tmp_path / "llama-server").write_text(SERVER.format(build="11524"))
    (tmp_path / "llama-quantize").write_text("#!/bin/sh\n")
    for tool in toolchain.TOOLS:
        (tmp_path / tool).chmod(0o755)
    monkeypatch.setenv("CANDOR_LLAMA_BIN", str(tmp_path))
    assert toolchain.binaries() == tmp_path
    (tmp_path / "llama-server").write_text(SERVER.format(build="1"))
    with pytest.raises(ToolchainError, match=r"isn't llama\.cpp"):
        toolchain.binaries()


def test_a_folder_missing_a_tool_is_refused(tmp_path, monkeypatch):
    (tmp_path / "llama-server").touch()
    monkeypatch.setenv("CANDOR_LLAMA_BIN", str(tmp_path))
    with pytest.raises(ToolchainError, match="llama-quantize"):
        toolchain.binaries()


def test_this_platform_is_known():
    assert toolchain.this_platform() in toolchain.ASSETS


@pytest.fixture
def converter_source(tmp_path, monkeypatch):
    """A local git repository standing in for llama.cpp, and a fake uv that can be told to fail."""
    monkeypatch.setenv("CANDOR_CACHE", str(tmp_path / "cache"))
    source = toolchain.cache() / "source"
    (source / "requirements").mkdir(parents=True)
    (source / "requirements" / "requirements-convert_hf_to_gguf.txt").write_text("")
    (source / "convert_hf_to_gguf.py").write_text("")
    git = ["git", "-C", str(source), "-c", "user.name=t", "-c", "user.email=t@t"]
    subprocess.run(["git", "init", "-q", str(source)], check=True)
    subprocess.run([*git, "add", "."], check=True)
    subprocess.run([*git, "commit", "-qm", "pinned"], check=True)
    head = subprocess.run(
        [*git, "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    monkeypatch.setattr(toolchain, "COMMIT", head)
    uv = tmp_path / "uv"
    uv.write_text(
        '#!/bin/sh\nif [ "$1" = venv ]; then mkdir -p "$5/bin" && touch "$5/bin/python"; exit 0; fi\n'
        'echo "$*" >> "$UV_CALLS"; exit "${UV_FAIL:-0}"\n'
    )
    uv.chmod(0o755)
    monkeypatch.setattr(toolchain.shutil, "which", lambda name: str(uv) if name == "uv" else None)
    monkeypatch.setenv("UV_CALLS", str(tmp_path / "uv-calls"))
    return source, tmp_path / "uv-calls"


def test_a_converter_whose_install_failed_is_installed_again(converter_source, monkeypatch):
    _, calls = converter_source
    monkeypatch.setenv("UV_FAIL", "1")
    with pytest.raises(subprocess.CalledProcessError):
        toolchain.converter()
    monkeypatch.setenv("UV_FAIL", "0")
    python, script = toolchain.converter()
    assert python.exists() and script.name == "convert_hf_to_gguf.py"
    assert len(calls.read_text().splitlines()) == 2
    toolchain.converter()
    assert len(calls.read_text().splitlines()) == 2


def test_a_changed_converter_checkout_is_refused(converter_source):
    source, _ = converter_source
    (source / "convert_hf_to_gguf.py").write_text("print('edited')")
    with pytest.raises(ToolchainError, match="local changes"):
        toolchain.converter()


def test_a_converter_at_another_commit_is_refused(converter_source, monkeypatch):
    monkeypatch.setattr(toolchain, "COMMIT", "0" * 40)
    with pytest.raises(ToolchainError, match=r"isn't llama\.cpp"):
        toolchain.converter()


def test_setup_waits_for_another_process_holding_the_lock(tmp_path, monkeypatch):
    monkeypatch.setenv("CANDOR_CACHE", str(tmp_path))
    script = (
        "import time, sys\nfrom candor import toolchain\n"
        "with toolchain._locked('x'):\n    print('locked', flush=True)\n    time.sleep(1)\n"
    )
    other = subprocess.Popen(
        [sys.executable, "-c", script], stdout=subprocess.PIPE, text=True, env={**os.environ}
    )
    assert other.stdout is not None and other.stdout.readline().strip() == "locked"
    started = time.perf_counter()
    with toolchain._locked("x"):
        waited = time.perf_counter() - started
    other.wait()
    assert waited > 0.5


def test_a_server_that_wont_start_is_reported_not_downloaded_again(release, monkeypatch):
    add, downloads, _ = release
    broken = add("llama.tar.gz", {"llama-b0/llama-server": "#!/bin/sh\necho 'missing libfoo' >&2\nexit 127\n",
                                  "llama-b0/llama-quantize": "#!/bin/sh\n"})  # fmt: skip
    monkeypatch.setitem(toolchain.ASSETS, "test", [broken])
    for _ in range(2):
        with pytest.raises(ToolchainError, match="won't start: missing libfoo"):
            toolchain.binaries()
    assert len(downloads) == 1


def test_a_file_in_two_archives_must_be_identical(release, monkeypatch):
    add, _, _ = release
    same = add("same.tar.gz", {"other/llama-quantize": "#!/bin/sh\n", "other/libcudart.so.12": "lib"})
    monkeypatch.setitem(toolchain.ASSETS, "test", [binaries_release(add), same])
    assert (toolchain.binaries() / "libcudart.so.12").exists()
    shutil.rmtree(toolchain.cache())
    clash = add("clash.tar.gz", {"other/llama-quantize": "#!/bin/sh\necho different\n"})
    monkeypatch.setitem(toolchain.ASSETS, "test", [binaries_release(add), clash])
    with pytest.raises(ToolchainError, match="differs between two of the release archives"):
        toolchain.binaries()


def test_a_damaged_install_is_fetched_again(release, monkeypatch):
    add, downloads, _ = release
    monkeypatch.setitem(toolchain.ASSETS, "test", [binaries_release(add)])
    folder = toolchain.binaries()
    (folder / "llama-quantize").unlink()
    assert (toolchain.binaries() / "llama-quantize").exists()
    assert len(downloads) == 2


def test_a_stray_source_folder_is_replaced_by_a_fresh_clone(converter_source, monkeypatch):
    source, _ = converter_source
    upstream = source.with_name("upstream")
    source.rename(upstream)
    subprocess.run(["git", "-C", str(upstream), "tag", toolchain.LLAMA_CPP_RELEASE], check=True)
    source.mkdir()
    (source / "half-cloned").write_text("x")
    monkeypatch.setattr(toolchain, "SOURCE", str(upstream))
    _, script = toolchain.converter()
    assert script.exists() and (source / ".git").exists() and not (source / "half-cloned").exists()
