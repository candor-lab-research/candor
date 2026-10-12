"""Merge a finished edit into the model's weights and convert it to the GGUF you run.

The edit is merged as its adapters apply it: the same directions, rank-1 factors and strengths, added
to the model's own weights before converting, then quantised as the run deploys. Merging before
quantising shifts the result a little (phase 0: KL 0.01 between adapters and merged weights in
Q4_K_M), so the exported file is measured again before it ships.
"""

import json
import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

from torch import Tensor

from . import convert, merge, source
from .edit import RowNorm, rank_one
from .run import Run, RunError, record_of
from .weights import ModelFiles

if TYPE_CHECKING:
    from .measure import Edit


def deltas(run: Run, edit: "Edit") -> dict[str, tuple[Tensor, Tensor]]:
    """Each edited weight's change as rank-1 factors (b, a), by its name in the model, with the
    strength folded into b."""
    _, _, _, writers = run.shape
    directions = run.edit_directions(edit.name, edit.direction_layer)
    files = ModelFiles(run.model_folder)
    row_norm = RowNorm(run.settings.edit.row_norm)
    made = {}
    for writer in writers:
        if strength := edit.strength(writer):
            b, a = rank_one(files.tensor(writer.name), directions[writer.layer], row_norm)
            made[writer.name] = (strength * b, a)
    return made


def _planned(run: Run, edit: "Edit") -> tuple[Path, dict[str, Any]]:
    """Where the edit's export goes and what it is built from."""
    run.gguf()
    run.edit_directions(edit.name, edit.direction_layer)
    quantization = run.settings.deployment.quantization
    path = run.path("exports", edit.name, f"{run.folder.name}-{edit.name}-{quantization}.gguf")
    inputs = {
        **run.adapter_inputs(),
        "base": run.inputs["base"],
        "edit": edit.record(),
        "quantization": quantization,
    }
    return path, inputs


def export(run: Run, edit: "Edit") -> Path:
    """The edit merged into the model, converted and quantised as the run deploys; reused while it
    matches. Nothing is left behind but the GGUF and the converter's log."""
    with run.locked():
        run.check_remote_code()
        shipped, inputs = _planned(run, edit)
        if run.fresh(shipped, inputs):
            return shipped
        _sweep(run, shipped)
        quantization = inputs["quantization"]
        float_type = convert.float_type(run.model_folder)
        copy = sum((run.model_folder / name).stat().st_size for name in source.weight_files(run.model_folder))
        run.check_disk(1, quantization != float_type, copy)
        folder = shipped.parent
        folder.mkdir(parents=True, exist_ok=True)
        log = folder / "convert.log"
        _, layers, arch, writers = run.shape
        with tempfile.TemporaryDirectory(dir=folder) as work:
            # The converter names the GGUF after the folder, so the copy keeps the model's name.
            merged = Path(work) / run.model_folder.name
            run.report(f"Merging {edit.name} into the weights")
            left_out = merge.merge(run.model_folder, merged, deltas(run, edit))
            if left_out:
                run.report(f"Left out of the copy, as weights in another format: {', '.join(left_out)}")
            run.report("Converting to GGUF")
            unquantized = Path(work) / "model.gguf"
            convert.to_gguf(merged, unquantized, log)
            if problems := convert.check_targets(unquantized, merged, writers, arch, layers):
                raise RunError(
                    "the exported GGUF doesn't hold the edited weights unchanged:\n  "
                    + "\n  ".join(problems[:5])
                )
            partial = Path(work) / f"{shipped.name}.building"
            if quantization == float_type:
                unquantized.rename(partial)
            else:
                run.report(f"Quantising to {quantization}")
                convert.quantize(unquantized, partial, quantization, log)
            run.commit(partial, shipped, inputs)
    return shipped


def _sweep(run: Run, shipped: Path) -> None:
    """Remove the work folders exports killed part way left in any edit's folder, which hold copies
    of the model, and this edit's exports in another quantisation."""
    root = run.path("exports")
    kept = {shipped.name, record_of(shipped).name, "convert.log"}
    for folder in root.iterdir() if root.exists() else []:
        for leftover in folder.iterdir() if folder.is_dir() else []:
            if leftover.is_dir():
                shutil.rmtree(leftover)
            elif folder == shipped.parent and leftover.name not in kept:
                leftover.unlink()


def exported(run: Run, edit: "Edit") -> tuple[Path, dict[str, Any]]:
    """The edit's export and its record, which holds what it was built from and the code that built
    it, for a caller holding the run's lock. Fails unless it matches the run and the edit as they
    are now."""
    path, inputs = _planned(run, edit)
    if not run.current(path, inputs):
        raise RunError(f"{edit.name} has no export matching this run; run candor export first")
    return path, json.loads(record_of(path).read_text())
