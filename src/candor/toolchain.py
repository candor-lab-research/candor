"""The pinned llama.cpp release: the binaries that run and quantise models, and the converter that
turns a Hugging Face model into GGUF.

Everything is fetched once into ~/.cache/candor (or CANDOR_CACHE). Release archives are checked
against their sha256 and the converter's source against its commit, so every machine and pod runs
exactly the same llama.cpp.
"""

import hashlib
import os
import platform
import shutil
import subprocess
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from .engine import LLAMA_CPP_RELEASE

COMMIT = "86a283532072722c5f3363d37d59a874d09fa99b"
RELEASES = f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA_CPP_RELEASE}"
SOURCE = "https://github.com/ggml-org/llama.cpp"


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
    return (
        Path(os.environ.get("CANDOR_CACHE", "~/.cache/candor")).expanduser()
        / f"llama.cpp-{LLAMA_CPP_RELEASE}"
    )


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


def _fetch(asset: Asset, folder: Path) -> None:
    archive = folder / asset.name
    urllib.request.urlretrieve(f"{RELEASES}/{asset.name}", archive)
    found = sha256(archive)
    if found != asset.sha256:
        archive.unlink()
        raise ToolchainError(f"{asset.name} has sha256 {found}, expected {asset.sha256}")
    with tarfile.open(archive) as tar:
        tar.extractall(folder / "bin", filter="data")
    archive.unlink()


def binaries() -> Path:
    """The folder holding llama-server and llama-quantize, fetched and checked on first use.

    CANDOR_LLAMA_BIN points at an existing folder of the same release instead.
    """
    if override := os.environ.get("CANDOR_LLAMA_BIN"):
        folder = Path(override).expanduser()
    else:
        root = cache()
        servers = sorted((root / "bin").rglob("llama-server")) if (root / "bin").exists() else []
        if not servers:
            partial = root.with_name(root.name + ".partial")
            shutil.rmtree(partial, ignore_errors=True)
            partial.mkdir(parents=True)
            for asset in ASSETS[this_platform()]:
                _fetch(asset, partial)
            root.mkdir(parents=True, exist_ok=True)
            shutil.rmtree(root / "bin", ignore_errors=True)
            (partial / "bin").rename(root / "bin")
            shutil.rmtree(partial)
            servers = sorted((root / "bin").rglob("llama-server"))
        folder = servers[0].parent
    for tool in ("llama-server", "llama-quantize"):
        if not (folder / tool).exists():
            raise ToolchainError(f"{folder} has no {tool}")
    return folder


def converter() -> tuple[Path, Path]:
    """The Python that runs llama.cpp's converter, and the converter script, set up on first use.

    The converter needs transformers 4, which can't share an environment with candor's
    transformers 5, so it gets its own, built by uv from llama.cpp's own requirements file.
    """
    root = cache()
    source, venv = root / "source", root / "converter-venv"
    python = venv / "bin" / "python"
    if not (source / ".git").exists():
        shutil.rmtree(source, ignore_errors=True)
        clone = [
            "git",
            "clone",
            "--quiet",
            "--depth",
            "1",
            "--branch",
            LLAMA_CPP_RELEASE,
            SOURCE,
            str(source),
        ]
        subprocess.run(clone, check=True)
    head = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    )
    if head.stdout.strip() != COMMIT:
        raise ToolchainError(
            f"{source} is at {head.stdout.strip()}, expected llama.cpp {LLAMA_CPP_RELEASE} ({COMMIT})"
        )
    if not python.exists():
        uv = shutil.which("uv")
        if uv is None:
            raise ToolchainError("setting up llama.cpp's converter needs uv: https://docs.astral.sh/uv/")
        requirements = source / "requirements" / "requirements-convert_hf_to_gguf.txt"
        subprocess.run([uv, "venv", "--quiet", "--python", "3.12", str(venv)], check=True)
        subprocess.run(
            [uv, "pip", "install", "--quiet", "--python", str(python), "-r", str(requirements)], check=True
        )
    return python, source / "convert_hf_to_gguf.py"
