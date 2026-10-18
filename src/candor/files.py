"""Files written so that a crash or power cut leaves each one whole, or as it was."""

import fcntl
import os
from pathlib import Path


def _flush(descriptor: int) -> None:
    """On macOS a plain fsync can leave data in the drive's cache; F_FULLFSYNC waits for the disk."""
    if hasattr(fcntl, "F_FULLFSYNC"):
        try:
            fcntl.fcntl(descriptor, fcntl.F_FULLFSYNC)
            return
        except OSError:
            pass
    os.fsync(descriptor)


def sync(path: Path) -> None:
    """Flush a file, or every file in a folder, and the folder entries, to the disk."""
    files = [path] if path.is_file() else [p for p in path.rglob("*") if p.is_file()]
    for item in [*files, *{path.parent, *(f.parent for f in files)}]:
        descriptor = os.open(item, os.O_RDONLY)
        try:
            _flush(descriptor)
        finally:
            os.close(descriptor)


def write_atomic(path: Path, text: str) -> None:
    """Replace path with text in one step, on the disk before this returns."""
    partial = path.with_name(path.name + ".building")
    try:
        partial.write_text(text, encoding="utf-8")
        sync(partial)
        partial.replace(path)
    finally:
        partial.unlink(missing_ok=True)
    sync(path)
