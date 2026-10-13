"""Merge a finished edit into the model's weights and convert it to the GGUF you run.

The edit is merged as its adapters apply it: the same directions, rank-1 factors and strengths, added
to the model's own weights before converting, then quantised as the run deploys. Quantising can round
part of a small edit away. rude-llama's weights were 4-bit before they were saved in 16 bits, and
Q4_K finds that grid again, so its weights rarely move by an edit's few percent of a step: Q4_K_M
kept 55-98% of trial 535's edit and its refusals came back (5 of 288 became 161). On weights that were
never quantised, Q4_K keeps nearly all of an edit. So the export measures how much of each weight's
edit the quantised file holds, quantises any weight that lost too much as Q8_0 instead, which keeps
it, and refuses the file if a weight still falls short. The file is then measured before it ships.

A model edited elsewhere, such as heretic's own save of one of its trials, is exported the same way:
the run's model with each edited weight taken from it, so it is converted, quantised and checked
exactly as candor's own edits are.
"""

import json
import re
import shutil
import tempfile
from collections.abc import Mapping
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

import gguf
import numpy as np
import torch
from torch import Tensor

from . import convert, merge, source
from .edit import RowNorm, rank_one
from .merge import Change, edited
from .run import Run, RunError, digest, record_of
from .source import INDEX
from .weights import ModelFiles

if TYPE_CHECKING:
    from .measure import Edit

# The least share of each weight's edit the shipped file may hold, and the type that keeps it when
# the run's quantisation doesn't.
LEAST_KEPT = 0.98
KEEPING_TYPE = "Q8_0"
# The names of candor's own edits and measurements, which a model edited elsewhere can't take.
RESERVED = re.compile(r"(base|(trial|heretic)-\d+)(-.*)?")


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
        run.report(f"Merging {edit.name} into the weights")
        return _ship(run, shipped, inputs, deltas(run, edit))


def export_outside(run: Run, name: str, folder: Path) -> Path:
    """A copy of the run's model edited elsewhere, such as heretic's save of one of its trials,
    exported as name: the run's model with each edit target's weight taken from that copy, converted
    and quantised as the run deploys, and checked as candor's own edits are. Reused while it matches."""
    if RESERVED.fullmatch(name):
        raise RunError(f"{name} is how candor names its own edits and measurements; pick another name")
    with run.locked():
        run.check_remote_code()
        shipped, inputs = _planned_outside(run, name, folder)
        if run.fresh(shipped, inputs):
            return shipped
        changes = taken_from(run, folder)
        if not changes:
            raise RunError(f"{folder} holds the run's model unchanged in every edit target")
        run.report(f"Taking the {len(changes)} edited weights from {folder}")
        return _ship(run, shipped, inputs, changes)


def taken_from(run: Run, folder: Path) -> dict[str, Change]:
    """Each edit target whose weight in folder differs from the run's model, by its name in the model,
    replaced by folder's. A save without the multimodal wrapper names its layers without
    language_model, as llama.cpp's converter does."""
    _, _, _, writers = run.shape
    theirs, ours = ModelFiles(folder), ModelFiles(run.model_folder)
    changes: dict[str, Change] = {}
    for writer in writers:
        found = [
            name
            for name in (writer.name, writer.name.replace(".language_model.", ".", 1))
            if name in theirs.shard_of
        ]
        if not found:
            raise RunError(f"{folder} has no weight for the edit target {writer.name}")
        weight, original = theirs.tensor(found[0]), ours.tensor(writer.name)
        if weight.shape != original.shape:
            raise RunError(
                f"{folder}'s {found[0]} is {list(weight.shape)}, not {list(original.shape)} as in the "
                "run's model"
            )
        if not torch.equal(weight.float(), original.float()):
            changes[writer.name] = partial(_replaced, theirs, found[0])
    return changes


def _replaced(files: ModelFiles, name: str, _original: Tensor) -> Tensor:
    return files.tensor(name)


def _planned_outside(run: Run, name: str, folder: Path) -> tuple[Path, dict[str, Any]]:
    """Where a model edited elsewhere is exported and what it is built from: the folder's weight files
    by name, size and time, and the folder itself, so a measurement can check it is unchanged."""
    run.gguf()
    folder = folder.expanduser().absolute()
    if not folder.is_dir():
        raise RunError(f"{folder} isn't a folder")
    weights = [p for p in folder.iterdir() if p.suffix == ".safetensors" or p.name == INDEX]
    quantization = run.settings.deployment.quantization
    path = run.path("exports", name, f"{run.folder.name}-{name}-{quantization}.gguf")
    inputs = {
        "targets": sorted(run.target_names.items()),
        "base": run.inputs["base"],
        "folder": str(folder),
        "weights": digest(sorted((p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in weights)),
        "quantization": quantization,
    }
    return path, inputs


def _ship(run: Run, shipped: Path, inputs: dict[str, Any], changes: Mapping[str, Change]) -> Path:
    """The run's model with these weights changed, converted and quantised as the run deploys, with
    any weight quantising loses too much of its change from quantised to keep it, committed as
    shipped."""
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
        left_out = merge.merge(run.model_folder, merged, changes)
        if left_out:
            run.report(f"Left out of the copy, as weights in another format: {', '.join(left_out)}")
        run.report("Converting to GGUF")
        unquantized = Path(work) / "model.gguf"
        convert.to_gguf(merged, unquantized, log)
        if problems := convert.check_targets(unquantized, merged, writers, arch, layers):
            raise RunError(
                "the exported GGUF doesn't hold the edited weights unchanged:\n  " + "\n  ".join(problems[:5])
            )
        partial_file = Path(work) / f"{shipped.name}.building"
        if quantization == float_type:
            unquantized.rename(partial_file)
            held = kept(run, partial_file, changes)
        else:
            run.report(f"Quantising to {quantization}")
            convert.quantize(unquantized, partial_file, quantization, log)
            held = kept(run, partial_file, changes)
            if lost := _short(held):
                run.report(
                    f"{quantization} keeps {min(lost.values()):.1%} to {max(lost.values()):.1%} of the "
                    f"edit in {len(lost)} of {len(held)} edited weights; quantising those as {KEEPING_TYPE}"
                )
                convert.quantize(
                    unquantized, partial_file, quantization, log, dict.fromkeys(lost, KEEPING_TYPE)
                )
                held = kept(run, partial_file, changes)
        if lost := _short(held):
            worst = sorted(lost.items(), key=lambda item: item[1])[:5]
            raise RunError(
                f"{len(lost)} of {len(held)} edited weights keep less than {LEAST_KEPT:.0%} of the edit "
                "even quantised to keep it, e.g. " + ", ".join(f"{name} {share:.1%}" for name, share in worst)
            )
        run.report(
            f"Each edited weight keeps {min(held.values()):.1%} to {max(held.values()):.1%} of its edit"
        )
        run.commit(partial_file, shipped, inputs)
    return shipped


def _short(held: dict[str, float]) -> dict[str, float]:
    return {name: share for name, share in held.items() if share < LEAST_KEPT}


def kept(run: Run, model_gguf: Path, made: Mapping[str, Change]) -> dict[str, float]:
    """For each edited weight, by its GGUF name, the share of its edit model_gguf holds: the part of
    its change from the original weight that lies along the edit, over the edit. 1 when the file
    holds the edit exactly; less when quantising rounded part of it away."""
    _, _, arch, _ = run.shape
    model_config = source.read_config(str(run.model_folder))
    tensors = {tensor.name: tensor for tensor in gguf.GGUFReader(model_gguf).tensors}
    files = ModelFiles(run.model_folder)
    shares = {}
    for name, change in made.items():
        tensor = tensors[run.target_names[name]]
        values = gguf.quants.dequantize(tensor.data, tensor.tensor_type)
        stored = torch.from_numpy(np.array(values, dtype=np.float32)).reshape(
            list(reversed(tensor.shape.tolist()))
        )
        weight = files.tensor(name)
        original = convert.as_converted(weight, name, arch, model_config).float()
        edit = convert.as_converted(edited(weight, change) - weight.float(), name, arch, model_config)
        shares[tensor.name] = float(((stored - original) * edit).sum() / (edit * edit).sum())
    return shares


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


def outside_export(run: Run, name: str) -> tuple[Path, dict[str, Any]]:
    """A model edited elsewhere as export_outside shipped it, and its record, for a caller holding the
    run's lock. Fails unless it matches the run and the folder it was taken from as they are now."""
    quantization = run.settings.deployment.quantization
    path = run.path("exports", name, f"{run.folder.name}-{name}-{quantization}.gguf")
    if not record_of(path).exists():
        raise RunError(f"{name} has no export in {quantization}; run candor export --outside first")
    folder = Path(json.loads(record_of(path).read_text())["folder"])
    planned, inputs = _planned_outside(run, name, folder)
    if not run.current(planned, inputs):
        raise RunError(f"{name}'s export no longer matches this run or {folder}; export it again")
    return planned, json.loads(record_of(planned).read_text())


def exported(run: Run, edit: "Edit") -> tuple[Path, dict[str, Any]]:
    """The edit's export and its record, which holds what it was built from and the code that built
    it, for a caller holding the run's lock. Fails unless it matches the run and the edit as they
    are now."""
    path, inputs = _planned(run, edit)
    if not run.current(path, inputs):
        raise RunError(f"{edit.name} has no export matching this run; run candor export first")
    return path, json.loads(record_of(path).read_text())
