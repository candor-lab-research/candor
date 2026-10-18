"""Merge a finished edit into the model's weights and convert it to the GGUF you run.

The edit is merged as its adapters apply it: the same directions, rank-1 factors and strengths, added
to the model's own weights before converting, then quantised as the run deploys. Quantising can round
part of a small edit away. rude-llama's weights were 4-bit before they were saved in 16 bits, and
Q4_K finds that grid again, so its weights rarely move by an edit's few percent of a step: Q4_K_M
kept 55-98% of trial 535's edit and its refusals came back (5 of 288 became 161). On weights that were
never quantised, Q4_K keeps nearly all of an edit. Merging can round a tiny edit away too: merged into
weights stored in 16 bits, an edit under half a step of theirs rounds back to the weight. So the export
measures how much of each weight's edit the quantised file holds. A weight that lost too much is merged
again unrounded, in F32, and quantised as Q8_0 from that, or left in F32 where Q8_0 still loses too
much and straight away for a run deploying Q8_0 or finer, while every other tensor keeps the type the
plain quantisation gave it. The file is refused if a
weight still falls short, and is then measured before it ships.

A model edited elsewhere, such as heretic's own save of one of its trials, is exported the same way:
the run's model with each edited weight taken from it, so it is converted, quantised and checked
exactly as candor's own edits are. It can be given as a folder of safetensors or as one unquantised
GGUF, as some edits are published only that way.
"""

import json
import math
import re
import shutil
import tempfile
from collections.abc import Callable, Iterable, Mapping
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

import gguf
import numpy as np
import torch
from torch import Tensor

from . import config, convert, merge, source
from .edit import RowNorm, edit_factors
from .merge import Change, edited
from .run import Run, RunError, digest, record_of
from .source import INDEX
from .weights import ModelFiles

if TYPE_CHECKING:
    from .measure import Edit

# The least share of each weight's edit the shipped file may hold, and the types tried in turn, each
# finer than the run's quantisation, for a weight that holds less: each from the weight merged unrounded.
LEAST_KEPT = 0.98
KEEPING_TYPES = ("Q8_0", "F32")
BITS = config.QUANTIZATIONS | {"F32": 32}
# The names of candor's own edits and measurements, which a model edited elsewhere can't take.
RESERVED = re.compile(r"(base|(trial|heretic)-\d+)(-.*)?")
# The types a GGUF edited elsewhere may hold its edit targets in: unquantised, so quantising them as the
# run deploys rounds them once, as candor's own edits are.
UNQUANTISED = {
    gguf.GGMLQuantizationType.F32: torch.float32,
    gguf.GGMLQuantizationType.F16: torch.float16,
    gguf.GGMLQuantizationType.BF16: torch.bfloat16,
}
# A part of a GGUF split into several files, as llama-gguf-split names them.
SPLIT = re.compile(r"-\d{5}-of-\d{5}\.gguf$")


def deltas(run: Run, edit: "Edit") -> dict[str, tuple[Tensor, Tensor]]:
    """Each edited weight's change as factors (b, a), by its name in the model, with the strength
    folded in: rank 1, or for a "full" edit the rank its adapters have, so the file shipped is the edit
    scored."""
    _, _, _, writers = run.shape
    directions = run.edit_directions(edit.name, edit.direction_layer, edit.answering)
    files = ModelFiles(run.model_folder)
    row_norm = RowNorm(run.settings.edit.row_norm)
    made = {}
    for writer in writers:
        if strength := edit.strength(writer):
            weight = files.tensor(writer.name)
            made[writer.name] = edit_factors(weight, directions[writer.layer], strength, row_norm)
    return made


def _path(run: Run, name: str) -> Path:
    quantization = run.settings.deployment.quantization
    return run.path("exports", name, f"{run.folder.name}-{name}-{quantization}.gguf")


def _planned(run: Run, edit: "Edit") -> tuple[Path, dict[str, Any]]:
    """Where the edit's export goes and what it is built from."""
    run.gguf()
    run.edit_directions(edit.name, edit.direction_layer, edit.answering)
    inputs = {
        **run.adapter_inputs(edit.answering),
        "base": run.inputs["base"],
        "edit": edit.record(),
        "quantization": run.settings.deployment.quantization,
    }
    return _path(run, edit.name), inputs


def export(run: Run, edit: "Edit") -> Path:
    """The edit merged into the model, converted and quantised as the run deploys; reused while it
    matches. Nothing is left behind but the GGUF and the converter's log."""
    with run.locked():
        run.check_remote_code()
        shipped, inputs = _planned(run, edit)
        if run.fresh(shipped, inputs | _listed_apart(run, shipped, partial(_edited, run, edit))):
            return shipped
        run.report(f"Merging {edit.name} into the weights")
        return _ship(run, shipped, inputs, deltas(run, edit))


def export_outside(run: Run, name: str, given: Path) -> Path:
    """A copy of the run's model edited elsewhere, such as heretic's save of one of its trials, as a
    folder or an unquantised GGUF, exported as name: the run's model with each edit target's weight
    taken from that copy, converted and quantised as the run deploys, and checked as candor's own edits
    are. Reused while it matches."""
    if RESERVED.fullmatch(name):
        raise RunError(f"{name} is how candor names its own edits and measurements; pick another name")
    given = given.expanduser().absolute()
    with run.locked():
        run.check_remote_code()
        shipped, inputs = _planned_outside(run, name, given)
        if run.fresh(shipped, inputs | _listed_apart(run, shipped, partial(_edited_outside, run, given))):
            return shipped
        changes = _taken(run, given)
        if not changes:
            raise RunError(f"{given} holds the run's model unchanged in every edit target")
        run.report(f"Taking the {len(changes)} edited weights from {given}")
        return _ship(run, shipped, inputs, changes)


def _taken(run: Run, given: Path) -> dict[str, Change]:
    return taken_from_gguf(run, given) if given.is_file() else taken_from(run, given)


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


def taken_from_gguf(run: Run, path: Path) -> dict[str, Change]:
    """Each edit target whose weight in an unquantised GGUF of the run's model differs from the run's
    own converted to the GGUF's type, by its name in the model, replaced by the GGUF's. Converting to a
    narrower type rounds every weight a little, so only a weight that differs by more than that counts
    as edited."""
    _, _, arch, writers = run.shape
    reader = gguf.GGUFReader(path)
    field = reader.get_field("general.architecture")
    if field is None:
        raise RunError(f"{path} doesn't say which architecture it holds, so it can't be checked")
    if (theirs := field.contents()) != arch:
        raise RunError(f"{path} holds a {theirs} model, and the run's model is {arch}")
    tensors = {tensor.name: tensor for tensor in reader.tensors}
    model_config = source.read_config(str(run.model_folder))
    ours = ModelFiles(run.model_folder)
    changes: dict[str, Change] = {}
    for writer in writers:
        name = run.target_names[writer.name]
        if name not in tensors:
            raise RunError(f"{path} has no tensor {name} for the edit target {writer.name}")
        tensor = tensors[name]
        if tensor.tensor_type not in UNQUANTISED:
            raise RunError(
                f"{path} holds {name} as {tensor.tensor_type.name}, quantised already; give the GGUF it was "
                "quantised from, so it is quantised once, as the run deploys"
            )
        weight = _stored(tensor)
        original = convert.as_converted(ours.tensor(writer.name), writer.name, arch, model_config)
        if list(weight.shape) != list(original.shape):
            raise RunError(
                f"{path}'s {name} is {list(weight.shape)}, not {list(original.shape)} as in the run's model"
            )
        if not torch.equal(weight, original.to(UNQUANTISED[tensor.tensor_type]).float()):
            changes[writer.name] = partial(_converted_back, tensor, writer.name, arch, model_config)
    return changes


def _stored(tensor: gguf.ReaderTensor) -> Tensor:
    """A GGUF tensor's values in float32, laid out as torch lays out the model's weight."""
    values = gguf.quants.dequantize(tensor.data, tensor.tensor_type)
    return torch.from_numpy(np.array(values, dtype=np.float32)).reshape(list(reversed(tensor.shape.tolist())))


def _converted_back(
    tensor: gguf.ReaderTensor, name: str, arch: str, model_config: dict[str, Any], _original: Tensor
) -> Tensor:
    return convert.as_original(_stored(tensor), name, arch, model_config)


def _planned_outside(run: Run, name: str, given: Path) -> tuple[Path, dict[str, Any]]:
    """Where a model edited elsewhere, given by its full path, is exported and what it is built from: its
    weight files by name, size and time, and the folder or GGUF itself, so a measurement can check it is
    unchanged."""
    run.gguf()
    if given.is_dir():
        kind = "folder"
        weights = [p for p in given.iterdir() if p.suffix == ".safetensors" or p.name == INDEX]
    elif SPLIT.search(given.name):
        raise RunError(
            f"{given} is one part of a split GGUF, which candor doesn't take; join the parts into one file "
            "first (llama.cpp's llama-gguf-split --merge)"
        )
    elif given.is_file() and given.suffix == ".gguf":
        kind, weights = "file", [given]
    else:
        raise RunError(f"{given} isn't a folder or a GGUF file")
    inputs = {
        "targets": sorted(run.target_names.items()),
        "base": run.inputs["base"],
        kind: str(given),
        "weights": digest(sorted((p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in weights)),
        "quantization": run.settings.deployment.quantization,
    }
    return _path(run, name), inputs


def _ship(run: Run, shipped: Path, inputs: dict[str, Any], changes: Mapping[str, Change]) -> Path:
    """The run's model with these weights changed, converted and quantised as the run deploys, any
    weight that keeps too little of its change kept apart, committed as shipped."""
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
        shutil.rmtree(merged)
        if quantization == float_type:
            built = unquantized
        else:
            built = Path(work) / f"{shipped.name}.building"
            run.report(f"Quantising to {quantization}")
            convert.quantize(unquantized, built, quantization, log)
        held = kept(run, built, changes)
        record = inputs
        if _short(held):
            held = _kept_apart(run, unquantized, built, log, changes, held)
            if apart := _apart(run, built):
                record = inputs | {"apart": apart}
        if lost := _short(held):
            raise RunError(
                f"{len(lost)} of {len(held)} edited weights keep less than {LEAST_KEPT:.0%} of the edit "
                f"even stored as F32, e.g. {_worst(lost)}"
            )
        run.report(f"Each edited weight keeps {_span(held)} of its edit")
        run.commit(built, shipped, record)
    return shipped


def _kept_apart(
    run: Run, unquantized: Path, built: Path, log: Path, changes: Mapping[str, Change], held: dict[str, float]
) -> dict[str, float]:
    """The shares held again, once each edited weight built keeps too little of is merged again
    unrounded and stored at each type finer than the run's quantisation in turn, until it keeps
    enough. Every other tensor is given the type the plain quantisation gave it: given some tensors'
    types, llama-quantize stops counting layers past them and changes the types of others. built is
    unquantized when the run deploys its float type."""
    quantization = run.settings.deployment.quantization
    plain = convert.tensor_types(built)
    apart: dict[str, str] = {}
    rounding = quantization
    for keeping in (kind for kind in KEEPING_TYPES if BITS[kind] > BITS[quantization]):
        if not (lost := _short(held)):
            break
        run.report(
            f"{rounding} keeps {_span(lost)} of the edit in {len(lost)} of {len(held)} edited weights; "
            f"storing those as {keeping}"
        )
        if not apart:
            _merge_unrounded(run, unquantized, lost, changes)
        apart |= dict.fromkeys(lost, keeping)
        if built != unquantized:
            convert.quantize(unquantized, built, quantization, log, plain | apart)
        held = kept(run, built, changes)
        rounding = keeping
    return held


def _merge_unrounded(run: Run, unquantized: Path, lost: Iterable[str], changes: Mapping[str, Change]) -> None:
    """Replace these edited weights, by GGUF name, in the unquantised GGUF with their merge in F32,
    unrounded, refusing them if even that keeps too little of the edit. Merged into weights stored in
    16 bits, an edit under half a step of theirs rounds back to the weight, and no quantising after can
    bring it back."""
    by_gguf_name = {run.target_names[name]: name for name in changes}
    redone = {by_gguf_name[name]: changes[by_gguf_name[name]] for name in lost}
    rounded = kept(run, unquantized, redone)
    shapes = source.tensor_shapes(str(run.model_folder))
    run.check_disk(1, False, 2 * sum(math.prod(shapes[name]) for name in redone))
    rewritten = unquantized.with_name(f"unrounded-{unquantized.name}")
    _in_float32(
        unquantized,
        rewritten,
        {run.target_names[name]: partial(_unrounded, run, name, change) for name, change in redone.items()},
    )
    rewritten.replace(unquantized)
    exact = kept(run, unquantized, redone)
    if short := _short(exact):
        raise RunError(
            f"{len(short)} of {len(changes)} edited weights keep less than {LEAST_KEPT:.0%} of the edit even "
            f"merged unrounded, in F32, before quantising, e.g. {_worst(short)}"
        )
    run.report(
        f"Merged in {convert.float_type(run.model_folder)}, those kept {_span(rounded)} of the edit; merged "
        f"unrounded, in F32, they keep {_span(exact)} and are stored from that"
    )


def _unrounded(run: Run, name: str, change: Change) -> np.ndarray:
    """The weight with its change in float32, laid out as the converter writes it."""
    _, _, arch, _ = run.shape
    weight = edited(ModelFiles(run.model_folder).tensor(name), change)
    model_config = source.read_config(str(run.model_folder))
    return convert.as_converted(weight, name, arch, model_config).contiguous().numpy()


def _in_float32(
    model_gguf: Path, destination: Path, replaced: Mapping[str, Callable[[], np.ndarray]]
) -> None:
    """A copy of a GGUF with these tensors, by name, holding the values their functions give in F32,
    and everything else as it was, copied as gguf-py's gguf_new_metadata copies a GGUF."""
    reader = gguf.GGUFReader(model_gguf)
    arch = reader.get_field(gguf.Keys.General.ARCHITECTURE)
    if arch is None:
        raise RunError(f"{model_gguf} doesn't say which architecture it holds")
    writer = gguf.GGUFWriter(destination, arch.contents())
    for field in reader.fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
            continue
        if field.name == gguf.Keys.General.ALIGNMENT:
            writer.data_alignment = field.contents()
        kind = field.types[0]
        item = field.types[-1] if kind == gguf.GGUFValueType.ARRAY else None
        writer.add_key_value(field.name, field.contents(), kind, sub_type=item)
    f32 = gguf.GGMLQuantizationType.F32
    for tensor in reader.tensors:
        if tensor.name in replaced:
            shape = list(reversed(tensor.shape.tolist()))
            writer.add_tensor_info(tensor.name, shape, np.dtype(np.float32), 4 * tensor.n_elements, f32)
        else:
            writer.add_tensor_info(
                tensor.name, tensor.data.shape, tensor.data.dtype, tensor.data.nbytes, tensor.tensor_type
            )
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    for tensor in reader.tensors:
        writer.write_tensor_data(replaced[tensor.name]() if tensor.name in replaced else tensor.data)
    writer.close()


def _apart(run: Run, model_gguf: Path) -> dict[str, str]:
    """The tensors a GGUF holds at another type than the run's plain quantisation gives them, by name."""
    plain = convert.tensor_types(run.gguf()[1])
    return {name: kind for name, kind in convert.tensor_types(model_gguf).items() if plain.get(name) != kind}


def _listed_apart(run: Run, path: Path, changed: Callable[[], set[str]]) -> dict[str, Any]:
    """What an export's record must hold besides its inputs: the tensors it holds apart, as listed. A
    record from before candor listed them needs nothing when they are all edited weights, as those
    exports gave only edited weights a type of their own; where an unedited tensor's type moved too, as
    llama-quantize then moved some past a tensor given its own, the export no longer matches."""
    if not path.exists() or not (apart := _apart(run, path)):
        return {}
    record = record_of(path)
    unlisted = record.exists() and "apart" not in json.loads(record.read_text())
    return {} if unlisted and apart.keys() <= changed() else {"apart": apart}


def _edited(run: Run, edit: "Edit") -> set[str]:
    """The GGUF names of the weights the edit changes."""
    _, _, _, writers = run.shape
    return {run.target_names[writer.name] for writer in writers if edit.strength(writer)}


def _edited_outside(run: Run, given: Path) -> set[str]:
    """The GGUF names of the weights taken from a model edited elsewhere."""
    return {run.target_names[name] for name in _taken(run, given)}


def _short(held: dict[str, float]) -> dict[str, float]:
    return {name: share for name, share in held.items() if share < LEAST_KEPT}


def _span(shares: dict[str, float]) -> str:
    return f"{min(shares.values()):.1%} to {max(shares.values()):.1%}"


def _worst(shares: dict[str, float]) -> str:
    worst = sorted(shares.items(), key=lambda item: item[1])[:5]
    return ", ".join(f"{name} {share:.1%}" for name, share in worst)


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
        stored = _stored(tensor)
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
    made = json.loads(record_of(path).read_text())
    given = Path(made["file"] if "file" in made else made["folder"])
    planned, inputs = _planned_outside(run, name, given)
    if not run.current(planned, inputs | _listed_apart(run, planned, partial(_edited_outside, run, given))):
        raise RunError(f"{name}'s export no longer matches this run or {given}; export it again")
    return planned, json.loads(record_of(planned).read_text())


def exported(run: Run, edit: "Edit") -> tuple[Path, dict[str, Any]]:
    """The edit's export and its record, which holds what it was built from and the code that built
    it, for a caller holding the run's lock. Fails unless it matches the run and the edit as they
    are now."""
    path, inputs = _planned(run, edit)
    if not run.current(path, inputs | _listed_apart(run, path, partial(_edited, run, edit))):
        raise RunError(f"{edit.name} has no export matching this run; run candor export first")
    return path, json.loads(record_of(path).read_text())


def built(run: Run, edit: "Edit") -> Path:
    """The edit's export as it was built, to chat with: made from this edit at the run's edit
    settings, but not checked against the model, whose full-size files a machine that only serves may
    not hold. Measurements take exported()."""
    path = _path(run, edit.name)
    record = record_of(path)
    if not (path.exists() and record.exists()):
        raise RunError(f"{edit.name} has no export; run candor export first")
    made = json.loads(record.read_text())
    wanted = json.loads(json.dumps({"edit": edit.record(), "row_norm": run.settings.edit.row_norm}))
    if {key: made.get(key) for key in wanted} != wanted:
        raise RunError(f"{edit.name}'s export was made from another edit; run candor export again")
    return path
