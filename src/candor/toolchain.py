"""The pinned llama.cpp release: the binaries that run and quantise models, and the converter that
turns a Hugging Face model into GGUF.

Everything is fetched once into ~/.cache/candor (or CANDOR_CACHE). Release archives are checked
against their sha256, the binaries against their version, and the converter's source against its
commit and for local changes, so every machine and pod runs exactly the same llama.cpp. Setup holds
a lock, so two runs starting at once can't trip over each other.
"""

import fcntl
import hashlib
import os
import platform
import shutil
import subprocess
import tarfile
import urllib.request
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from .engine import LLAMA_CPP_RELEASE

COMMIT = "86a283532072722c5f3363d37d59a874d09fa99b"
RELEASES = f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA_CPP_RELEASE}"
SOURCE = "https://github.com/ggml-org/llama.cpp"
TIMEOUT = 60
TOOLS = ("llama-server", "llama-quantize")


@dataclass(frozen=True)
class Asset:
    name: str
    sha256: str


ASSETS = {
    "macos-arm64": [
        Asset(
            f"llama-{LLAMA_CPP_RELEASE}-bin-macos-arm64.tar.gz",
            "726e586ce19199f71f00a743c7baae737da9b2c7b1a2402b9661c2ce66d87ca3",
        )
    ],
    "linux-cuda": [
        Asset(
            f"llama-{LLAMA_CPP_RELEASE}-bin-ubuntu-cuda-12.8-x64.tar.gz",
            "5a68281f130a3ea793042f4534939b352669138ff5229ef4127030a01161f171",
        ),
        Asset(
            f"cudart-llama-{LLAMA_CPP_RELEASE}-bin-ubuntu-cuda-12.8-x64.tar.gz",
            "3286cf2eefefc4ebda4c21a215fca38e65732dd14d6a7ba6a8ab8d6aae6dcb99",
        ),
    ],
    "linux-cpu": [
        Asset(
            f"llama-{LLAMA_CPP_RELEASE}-bin-ubuntu-x64.tar.gz",
            "2f44f8e794a9e6290bacb2b811e6a83019a106c057473682aeeb445bd49c5083",
        )
    ],
}


class ToolchainError(RuntimeError):
    pass


def cache() -> Path:
    root = Path(os.environ.get("CANDOR_CACHE", "~/.cache/candor")).expanduser()
    return root / f"llama.cpp-{LLAMA_CPP_RELEASE}"


@contextmanager
def _locked(name: str) -> Iterator[None]:
    root = cache()
    root.mkdir(parents=True, exist_ok=True)
    with (root / f".{name}.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def this_platform() -> str:
    system, machine = platform.system(), platform.machine()
    if system == "Darwin" and machine == "arm64":
        return "macos-arm64"
    if system == "Linux" and machine == "x86_64":
        return "linux-cuda" if shutil.which("nvidia-smi") else "linux-cpu"
    raise ToolchainError(f"no llama.cpp {LLAMA_CPP_RELEASE} build is set up for {system} {machine}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, destination: Path) -> None:
    with urllib.request.urlopen(url, timeout=TIMEOUT) as response, destination.open("wb") as out:
        shutil.copyfileobj(response, out)


def _fetch(asset: Asset, folder: Path) -> Path:
    """Download and unpack one archive into its own folder, and return that folder."""
    archive = folder / asset.name
    _download(f"{RELEASES}/{asset.name}", archive)
    found = sha256(archive)
    if found != asset.sha256:
        raise ToolchainError(f"{asset.name} has sha256 {found}, expected {asset.sha256}")
    unpacked = folder / f"{asset.name}.unpacked"
    with tarfile.open(archive) as tar:
        tar.extractall(unpacked, filter="data")
    archive.unlink()
    return unpacked


def _version_ok(folder: Path) -> bool:
    try:
        result = subprocess.run(
            [str(folder / "llama-server"), "--version"], capture_output=True, text=True, timeout=60
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return f"build {LLAMA_CPP_RELEASE.removeprefix('b')}," in result.stdout + result.stderr


def _check(folder: Path) -> Path:
    for tool in TOOLS:
        if not (folder / tool).exists():
            raise ToolchainError(f"{folder} has no {tool}")
    if not _version_ok(folder):
        raise ToolchainError(f"the llama-server in {folder} isn't llama.cpp {LLAMA_CPP_RELEASE}")
    return folder


def _install(target: Path, assets: list[Asset]) -> None:
    """Unpack every archive and gather all their files in one folder beside llama-server: the CUDA
    runtime ships as a separate archive whose libraries llama-server must find next to itself."""
    partial = target.with_name(target.name + ".partial")
    shutil.rmtree(partial, ignore_errors=True)
    partial.mkdir(parents=True)
    try:
        unpacked = [_fetch(asset, partial) for asset in assets]
        servers = sorted(unpacked[0].rglob("llama-server"))
        if not servers:
            raise ToolchainError(f"{assets[0].name} holds no llama-server")
        home = servers[0].parent
        for extra in unpacked[1:]:
            for item in sorted(extra.rglob("*")):
                if item.is_file() or item.is_symlink():
                    item.rename(home / item.name)
        shutil.rmtree(target, ignore_errors=True)
        home.rename(target)
    finally:
        shutil.rmtree(partial, ignore_errors=True)


def binaries() -> Path:
    """The folder holding llama-server and llama-quantize, fetched and checked on first use.

    CANDOR_LLAMA_BIN points at an existing folder instead; it must hold the same release.
    """
    if override := os.environ.get("CANDOR_LLAMA_BIN"):
        return _check(Path(override).expanduser())
    name = this_platform()
    target = cache() / name / "bin"
    with _locked(f"bin-{name}"):
        if not target.exists() or not _version_ok(target):
            _install(target, ASSETS[name])
        folder = _check(target)
    if name == "linux-cuda":
        server = str(folder / "llama-server")
        devices = subprocess.run([server, "--list-devices"], capture_output=True, text=True, timeout=120)
        if "CUDA" not in devices.stdout + devices.stderr:
            raise ToolchainError("llama-server can't see the GPU, so it would silently run on the CPU")
    return folder


def _git(source: Path, *args: str) -> str:
    result = subprocess.run(["git", "-C", str(source), *args], capture_output=True, text=True, check=True)
    return result.stdout.strip()


def converter() -> tuple[Path, Path]:
    """The Python that runs llama.cpp's converter, and the converter script, set up on first use.

    The converter needs transformers 4, which can't share an environment with candor's
    transformers 5, so it gets its own, built by uv from llama.cpp's own requirements file. The
    environment counts as ready only once its install has finished.
    """
    root = cache()
    source, venv = root / "source", root / "converter-venv"
    python, installed = venv / "bin" / "python", venv / ".candor-installed"
    with _locked("converter"):
        if not (source / ".git").exists():
            partial = root / "source.partial"
            shutil.rmtree(partial, ignore_errors=True)
            clone = ["git", "-c", "advice.detachedHead=false", "clone", "--quiet", "--depth", "1",
                     "--branch", LLAMA_CPP_RELEASE, SOURCE, str(partial)]  # fmt: skip
            subprocess.run(clone, check=True, timeout=600)
            partial.rename(source)
        if _git(source, "rev-parse", "HEAD") != COMMIT:
            raise ToolchainError(f"{source} isn't llama.cpp {LLAMA_CPP_RELEASE}; delete it to fetch again")
        if _git(source, "status", "--porcelain"):
            raise ToolchainError(f"{source} has local changes; delete it to fetch a clean copy")
        if not installed.exists():
            uv = shutil.which("uv")
            if uv is None:
                raise ToolchainError("setting up llama.cpp's converter needs uv: https://docs.astral.sh/uv/")
            requirements = source / "requirements" / "requirements-convert_hf_to_gguf.txt"
            shutil.rmtree(venv, ignore_errors=True)
            subprocess.run([uv, "venv", "--quiet", "--python", "3.12", str(venv)], check=True)
            install = [uv, "pip", "install", "--quiet", "--python", str(python), "-r", str(requirements)]
            subprocess.run(install, check=True)
            installed.touch()
    return python, source / "convert_hf_to_gguf.py"
