import importlib.metadata
import io
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tarfile
import threading
import time
import tomllib
from importlib import resources
from pathlib import Path

import pytest

from candor import toolchain
from candor.engine import LLAMA_CPP_RELEASE
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


@pytest.mark.parametrize("build", ["9999", "115240", "1152"])
def test_a_different_release_is_refused(release, monkeypatch, build):
    add, _, _ = release
    monkeypatch.setitem(toolchain.ASSETS, "test", [binaries_release(add, build=build)])
    with pytest.raises(ToolchainError) as raised:
        toolchain.binaries()
    message = str(raised.value)
    assert message.count("delete") == 1 and message.endswith("/bin to fetch it again")


def test_an_existing_folder_must_hold_the_same_release(tmp_path, monkeypatch):
    (tmp_path / "llama-server").write_text(SERVER.format(build="11524"))
    (tmp_path / "llama-quantize").write_text("#!/bin/sh\n")
    for tool in toolchain.TOOLS:
        (tmp_path / tool).chmod(0o755)
    monkeypatch.setenv("CANDOR_LLAMA_BIN", str(tmp_path))
    assert toolchain.binaries() == tmp_path
    (tmp_path / "llama-server").write_text(SERVER.format(build="1"))
    with pytest.raises(
        ToolchainError, match=r"isn't llama\.cpp b11524; point CANDOR_LLAMA_BIN at a b11524 build$"
    ):
        toolchain.binaries()


def test_a_folder_missing_a_tool_is_refused(tmp_path, monkeypatch):
    (tmp_path / "llama-server").touch()
    monkeypatch.setenv("CANDOR_LLAMA_BIN", str(tmp_path))
    with pytest.raises(ToolchainError, match="llama-quantize"):
        toolchain.binaries()


def test_this_platform_is_known():
    assert toolchain.this_platform() in toolchain.ASSETS


def test_linux_always_takes_the_cuda_build(monkeypatch):
    monkeypatch.setattr(toolchain.platform, "system", lambda: "Linux")
    monkeypatch.setattr(toolchain.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(toolchain.shutil, "which", lambda _: None)
    assert toolchain.this_platform() == "linux-cuda"


def listing_devices(stdout: str, stderr: str) -> str:
    """A llama-server whose --list-devices prints the given text."""
    listing = f"if [ \"$1\" = --list-devices ]; then printf '{stdout}'; printf '{stderr}' >&2; exit; fi\n"
    return SERVER.format(build="11524").replace("\n", "\n" + listing, 1)


@pytest.mark.parametrize(
    ("stdout", "stderr", "refused"),
    [
        ("Available devices:\n  CUDA0: NVIDIA A40 (45498 MiB, 45229 MiB free)\n", "", None),
        (
            "Available devices:\n  (none)\n",
            "ggml_cuda_init: failed to initialize CUDA: no CUDA-capable device is detected\n",
            "failed to initialize CUDA",
        ),
        ("Available devices:\n  (none)\n", "", "no CUDA device listed"),
    ],
)
def test_a_cuda_build_must_list_a_cuda_device(release, monkeypatch, stdout, stderr, refused):
    add, _, _ = release
    monkeypatch.setattr(toolchain, "this_platform", lambda: "linux-cuda")
    monkeypatch.setattr(toolchain, "on_cuda", lambda: True)
    server = listing_devices(stdout, stderr)
    archive = add("cuda.tar.gz", {"llama-b0/llama-server": server, "llama-b0/llama-quantize": "#!/bin/sh\n"})
    monkeypatch.setitem(toolchain.ASSETS, "linux-cuda", [archive])
    if refused is None:
        assert (toolchain.binaries() / "llama-server").exists()
    else:
        with pytest.raises(ToolchainError, match=f"can't see the GPU.*{refused}"):
            toolchain.binaries()


def test_the_gpus_are_listed_once_and_named_to_every_server(release, monkeypatch, tmp_path):
    add, _, _ = release
    monkeypatch.setattr(toolchain, "this_platform", lambda: "linux-cuda")
    monkeypatch.setattr(toolchain, "on_cuda", lambda: True)
    listed = tmp_path / "listed"
    gpus = (
        "  CUDA0: NVIDIA A40 (45498 MiB, 45229 MiB free)\\n  CUDA1: NVIDIA A40 (45498 MiB, 45229 MiB free)\\n"
    )
    server = listing_devices(gpus, "").replace("then printf", f"then echo >> '{listed}'; printf")
    archive = add("cuda.tar.gz", {"llama-b0/llama-server": server, "llama-b0/llama-quantize": "#!/bin/sh\n"})
    monkeypatch.setitem(toolchain.ASSETS, "linux-cuda", [archive])
    for _ in range(3):
        folder = toolchain.binaries()
    # Listing them starts CUDA; a server started after a GPU fails refuses its --device instead.
    assert toolchain.device_flags(folder) == ["--device", "CUDA0,CUDA1"]
    assert listed.read_text() == "\n"


def test_the_machine_names_the_platform_and_the_gpus_listed(tmp_path, monkeypatch):
    monkeypatch.setattr(toolchain, "this_platform", lambda: "linux-cuda")
    server = tmp_path / "llama-server"
    gpus = (
        "Available devices:\\n  CUDA0: NVIDIA A40 (45498 MiB, 45229 MiB free)\\n"
        "  BLAS: OpenBLAS (0 MiB, 0 MiB free)\\n"
    )
    server.write_text(listing_devices(gpus, ""))
    server.chmod(0o755)
    assert toolchain.machine(tmp_path) == {"platform": "linux-cuda", "gpus": ["NVIDIA A40"]}
    assert toolchain.described(toolchain.machine(tmp_path)) == "NVIDIA A40 (linux-cuda)"
    # One that can't list its devices still names the platform.
    assert toolchain.machine(tmp_path / "missing") == {"platform": "linux-cuda", "gpus": []}


def test_an_override_on_linux_must_see_the_gpu_too(tmp_path, monkeypatch):
    monkeypatch.setattr(toolchain, "on_cuda", lambda: True)
    server = tmp_path / "llama-server"
    server.write_text(listing_devices("Available devices:\\n", ""))
    server.chmod(0o755)
    (tmp_path / "llama-quantize").touch()
    monkeypatch.setenv("CANDOR_LLAMA_BIN", str(tmp_path))
    with pytest.raises(ToolchainError, match="can't see the GPU"):
        toolchain.binaries()


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


def test_the_converter_is_installed_again_when_its_pins_change(converter_source, monkeypatch, tmp_path):
    _, calls = converter_source
    toolchain.converter()
    assert f"-c {toolchain.CONSTRAINTS}" in calls.read_text()
    changed = tmp_path / "constraints.txt"
    changed.write_text("numpy==2.2.6\n")
    monkeypatch.setattr(toolchain, "CONSTRAINTS", changed)
    toolchain.converter()
    toolchain.converter()
    assert calls.read_text().count(f"-c {changed}") == 1


def test_every_pin_of_llama_cpp_is_the_same_commit():
    table = json.loads((resources.files("candor") / "data" / "llama_cpp_architectures.json").read_text())
    installed = json.loads(importlib.metadata.distribution("gguf").read_text("direct_url.json") or "{}")
    project = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text())
    gguf = next(spec for spec in project["project"]["dependencies"] if spec.startswith("gguf "))
    assert gguf.endswith(f"@{LLAMA_CPP_RELEASE}#subdirectory=gguf-py")
    assert installed["vcs_info"]["requested_revision"] == LLAMA_CPP_RELEASE
    assert installed["vcs_info"]["commit_id"] == toolchain.COMMIT
    assert table["commit"] == toolchain.COMMIT


def test_the_converter_pins_were_made_for_the_pinned_release():
    header = [line for line in toolchain.CONSTRAINTS.read_text().splitlines() if line.startswith("#")]
    assert re.search(rf"\bllama\.cpp {LLAMA_CPP_RELEASE}\b", "\n".join(header))


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


def test_an_install_of_other_archives_is_replaced(release, monkeypatch):
    add, downloads, _ = release
    monkeypatch.setitem(toolchain.ASSETS, "test", [binaries_release(add)])
    toolchain.binaries()
    runtime = add("cudart.tar.gz", {"cudart-b0/libcudart.so.12": "lib"})
    monkeypatch.setitem(toolchain.ASSETS, "test", [binaries_release(add), runtime])
    assert (toolchain.binaries() / "libcudart.so.12").exists()
    assert len(downloads) == 3


def test_binaries_wait_for_another_setup(release, monkeypatch):
    add, downloads, _ = release
    monkeypatch.setitem(toolchain.ASSETS, "test", [binaries_release(add)])
    with toolchain._locked("bin-test"):
        waiting = threading.Thread(target=toolchain.binaries)
        waiting.start()
        waiting.join(0.5)
        assert waiting.is_alive() and not downloads
    waiting.join(10)
    assert len(downloads) == 1


def test_a_stalled_download_fails_instead_of_hanging(tmp_path, monkeypatch):
    previous = socket.getdefaulttimeout()
    socket.setdefaulttimeout(5)  # so a download without its own timeout shows up as slow, not as a hang
    try:
        with socket.socket() as silent:
            silent.bind(("127.0.0.1", 0))
            silent.listen()
            monkeypatch.setattr(toolchain, "RELEASES", f"http://127.0.0.1:{silent.getsockname()[1]}")
            monkeypatch.setattr(toolchain, "TIMEOUT", 0.2)
            started = time.perf_counter()
            with pytest.raises(ToolchainError, match=r"couldn't download x\.tar\.gz"):
                toolchain._fetch(Asset("x.tar.gz", "0"), tmp_path)
            assert time.perf_counter() - started < 2
    finally:
        socket.setdefaulttimeout(previous)


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
