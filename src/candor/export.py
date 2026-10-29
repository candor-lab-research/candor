"""Write a finished edit into the model's weights, as the GGUF you run.

The edit is written as its adapters apply it: the same directions, factors and strengths, added to
the model's own weights in a copy of the run's unquantised GGUF, then quantised as the run deploys.
Quantising can round part of a small edit away. rude-llama's weights were 4-bit before they were
saved in 16 bits, and Q4_K finds that grid again, so its weights rarely move by an edit's few percent
of a step: Q4_K_M kept 55-98% of trial 535's edit and its refusals came back (5 of 288 became 161).
On weights that were never quantised, Q4_K keeps nearly all of an edit. Writing can round a tiny edit
away too: written into weights stored in 16 bits, an edit under half a step of theirs rounds back to
the weight. So the export measures how much of each weight's edit the quantised file holds. A weight
that lost too much is written again unrounded, in F32, and quantised as Q8_0 from that, or left in
F32 where Q8_0 still loses too much (straight away for a run deploying Q8_0 or finer); every other
tensor keeps the type the plain quantisation gave it. The file is refused if a weight still falls
short, and is then measured before it ships.

A model edited elsewhere, such as heretic's own save of one of its trials, is exported the same way:
the run's model with each edited weight taken from it, quantised and checked exactly as candor's own
edits are. It can be given as a folder of safetensors or as one unquantised GGUF, as some edits are
published only that way.
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

from . import config, convert, source
from .run import Run, RunError, digest, record_of
from .source import INDEX
from .weights import ModelFiles

if TYPE_CHECKING:
    from .measure import Edit

# The least share of each weight's edit the shipped file may hold, and the types tried in turn, each
# finer than the run's quantisation, for a weight that holds less: each from the weight written unrounded.
LEAST_KEPT = 0.98
KEEPING_TYPES = ("Q8_0", "F32")
BITS = config.QUANTIZATIONS | {"F32": 32}
# The names of candor's own edits and measurements, which a model edited elsewhere can't take.
RESERVED = re.compile(r"(base|(trial|heretic)-\d+)(-.*)?")
# The types a GGUF edited elsewhere may hold its edit targets in: unquantised, so quantising them as the
# run deploys rounds them once, as candor's own edits are.
UNQUANTISED = convert.STORED_AS | {gguf.GGMLQuantizationType.F32: torch.float32}
# A part of a GGUF split into several files, as llama-gguf-split names them.
SPLIT = re.compile(r"-\d{5}-of-\d{5}\.gguf$")
F32 = gguf.GGMLQuantizationType.F32
# How a weight is edited: by adding b @ a, or by a function of the original weight giving the edited one.
Change = tuple[Tensor, Tensor] | Callable[[Tensor], Tensor]


def edited(weight: Tensor, change: Change) -> Tensor:
    """The weight after its change, in float32."""
    if callable(change):
        return change(weight).float()
    b, a = change
    return weight.float() + b.float() @ a.float()


def deltas(run: Run, edit: "Edit") -> dict[str, tuple[Tensor, Tensor]]:
    """Each edited weight's change as factors (b, a), by its edit target's name, with the strength
    folded in: the change its adapter holds (run.changes), laid out as the model stores the weight, a
    layer's experts stacked."""
    _, _, arch, _ = run.shape
    model_config = source.read_config(str(run.model_folder))
    return {
        name: (b, convert.as_original(a, name, arch, model_config))
        for name, (b, a) in run.changes(edit).items()
    }


def _path(run: Run, name: str) -> Path:
    quantization = run.settings.deployment.quantization
    return run.path("exports", name, f"{run.folder.name}-{name}-{quantization}.gguf")


def _planned(run: Run, edit: "Edit", read: bool = True) -> tuple[Path, dict[str, Any]]:
    """Where the edit's export goes and what it is built from. Without read the directions are only
    planned, so an export built from others is found out before any residual is read again."""
    run.gguf()
    if read:
        run.edit_directions(edit.name, edit.direction_layer, edit.answering)
    else:
        run.planned_directions()
    inputs = {
        **run.adapter_inputs(edit.answering),
        "base": run.inputs["base"],
        "edit": edit.record(),
        "quantization": run.settings.deployment.quantization,
    }
    return _path(run, edit.name), inputs


def export(run: Run, edit: "Edit") -> Path:
    """The edit written into a copy of the run's GGUF and quantised as the run deploys; reused while
    it matches. Nothing is left behind but the GGUF and llama-quantize's log, convert.log."""
    with run.locked():
        run.check_remote_code()
        shipped, inputs = _planned(run, edit)
        if run.fresh(shipped, inputs | _listed_apart(run, shipped, partial(_edited, run, edit))):
            return shipped
        run.report(f"Writing {edit.name} into the weights")
        return _ship(run, shipped, inputs, deltas(run, edit))


def export_outside(run: Run, name: str, given: Path) -> Path:
    """A copy of the run's model edited elsewhere, such as heretic's save of one of its trials, as a
    folder or an unquantised GGUF, exported as name: the run's model with each edit target's weight
    taken from that copy, written into a copy of the run's GGUF and quantised as the run deploys, and
    checked as candor's own edits are. Reused while it matches."""
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
    """Each edit target whose weight in folder differs from the run's model, by its name, replaced by
    folder's. A save without the multimodal wrapper names its layers without language_model, as
    llama.cpp's converter does."""
    theirs, ours = ModelFiles(folder), ModelFiles(run.model_folder)
    changes: dict[str, Change] = {}
    for target in run.edit_targets.values():
        found = []
        for name in target.names:
            named = [n for n in (name, name.replace(".language_model.", ".", 1)) if n in theirs.shard_of]
            if not named:
                raise RunError(f"{folder} has no weight for the edit target {name}")
            found.append(named[0])
        weight, original = theirs.read(found, target.stacked), ours.read(target.names, target.stacked)
        if weight.shape != original.shape:
            raise RunError(
                f"{folder}'s {found[0]} is {list(weight.shape)}, not {list(original.shape)} as in the "
                "run's model"
            )
        if not torch.equal(weight.float(), original.float()):
            changes[target.name] = partial(_replaced, theirs, found, target.stacked)
    return changes


def _replaced(files: ModelFiles, names: list[str], stacked: bool, _original: Tensor) -> Tensor:
    return files.read(names, stacked)


def taken_from_gguf(run: Run, path: Path) -> dict[str, Change]:
    """Each edit target whose weight in an unquantised GGUF of the run's model differs from the run's
    own converted to the GGUF's type, by its edit target's name, replaced by the GGUF's. Converting to a
    narrower type rounds every weight a little, so only a weight that differs by more than that counts
    as edited."""
    _, _, arch, _ = run.shape
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
    for target in run.edit_targets.values():
        name = target.gguf
        if name not in tensors:
            raise RunError(f"{path} has no tensor {name} for the edit target {target.name}")
        tensor = tensors[name]
        if tensor.tensor_type not in UNQUANTISED:
            raise RunError(
                f"{path} holds {name} as {tensor.tensor_type.name}, quantised already; give the GGUF it was "
                "quantised from, so it is quantised once, as the run deploys"
            )
        weight = convert.stored(tensor)
        original = run.target_weight(ours, target, model_config)
        if list(weight.shape) != list(original.shape):
            raise RunError(
                f"{path}'s {name} is {list(weight.shape)}, not {list(original.shape)} as in the run's model"
            )
        if not torch.equal(weight, original.to(UNQUANTISED[tensor.tensor_type]).float()):
            changes[target.name] = partial(_converted_back, tensor, target.name, arch, model_config)
    return changes


def _converted_back(
    tensor: gguf.ReaderTensor, name: str, arch: str, model_config: dict[str, Any], _original: Tensor
) -> Tensor:
    return convert.as_original(convert.stored(tensor), name, arch, model_config)


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
    """The run's model with these weights changed, written into a copy of its unquantised GGUF and
    quantised as the run deploys, any weight that keeps too little of its change kept apart, committed
    as shipped. The GGUF's edit targets are the model's as the converter writes them (run.gguf checks
    each), so each changed weight is written as the converter would write it."""
    _sweep(run, shipped)
    quantization = inputs["quantization"]
    base, _ = run.gguf()
    float_type = convert.float_type(run.model_folder)
    run.check_disk(1, quantization != float_type)
    folder = shipped.parent
    folder.mkdir(parents=True, exist_ok=True)
    log = folder / "convert.log"
    with tempfile.TemporaryDirectory(dir=folder) as work:
        unquantized = Path(work) / "model.gguf"
        _rewritten(
            base,
            unquantized,
            {
                run.edit_targets[name].gguf: partial(_written, run, name, change)
                for name, change in changes.items()
            },
        )
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
    """The shares held again, once each edited weight built keeps too little of is written again
    unrounded and stored at each type finer than the run's quantisation in turn, until it keeps
    enough. Every other tensor is given the type the plain quantisation gave it: given some tensors'
    types, llama-quantize stops counting layers past them and changes the types of others. F32 is
    written straight into built, as llama-quantize keeps no F32 given a float type to quantise to. built
    is unquantized when the run deploys its float type."""
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
            _write_unrounded(run, unquantized, lost, changes)
        apart |= dict.fromkeys(lost, keeping)
        if built != unquantized:
            if keeping == "F32":
                unrounded = {tensor.name: tensor for tensor in gguf.GGUFReader(unquantized).tensors}
                rewritten = built.with_name(f"f32-{built.name}")
                _rewritten(built, rewritten, {name: partial(_values, unrounded[name]) for name in lost}, F32)
                rewritten.replace(built)
            else:
                convert.quantize(unquantized, built, quantization, log, plain | apart)
        held = kept(run, built, changes)
        rounding = keeping
    return held


def _write_unrounded(run: Run, unquantized: Path, lost: Iterable[str], changes: Mapping[str, Change]) -> None:
    """Replace these edited weights, by GGUF name, in the unquantised GGUF with their change written in
    F32, unrounded, refusing them if even that keeps too little of the edit. Written into weights stored
    in 16 bits, an edit under half a step of theirs rounds back to the weight, and no quantising after
    can bring it back."""
    by_gguf_name = {run.edit_targets[name].gguf: name for name in changes}
    redone = {by_gguf_name[name]: changes[by_gguf_name[name]] for name in lost}
    rounded = kept(run, unquantized, redone)
    shapes = source.tensor_shapes(str(run.model_folder))
    size = sum(math.prod(shapes[weight]) for name in redone for weight in run.edit_targets[name].names)
    run.check_disk(1, False, 2 * size)
    rewritten = unquantized.with_name(f"unrounded-{unquantized.name}")
    _rewritten(
        unquantized,
        rewritten,
        {
            run.edit_targets[name].gguf: partial(_written, run, name, change)
            for name, change in redone.items()
        },
        F32,
    )
    rewritten.replace(unquantized)
    exact = kept(run, unquantized, redone)
    if short := _short(exact):
        raise RunError(
            f"{len(short)} of {len(changes)} edited weights keep less than {LEAST_KEPT:.0%} of the edit even "
            f"written unrounded, in F32, before quantising, e.g. {_worst(short)}"
        )
    run.report(
        f"Written in {convert.float_type(run.model_folder)}, those kept {_span(rounded)} of the edit; "
        f"written unrounded, in F32, they keep {_span(exact)} and are stored from that"
    )


def _written(run: Run, name: str, change: Change) -> np.ndarray:
    """The edit target's weight with its change in float32, laid out as the converter writes it."""
    _, _, arch, _ = run.shape
    target = run.edit_targets[name]
    weight = edited(ModelFiles(run.model_folder).read(target.names, target.stacked), change)
    model_config = source.read_config(str(run.model_folder))
    return convert.as_converted(weight, name, arch, model_config).contiguous().numpy()


def _values(tensor: gguf.ReaderTensor) -> np.ndarray:
    return np.array(tensor.data, dtype=np.float32)


def _rewritten(
    model_gguf: Path,
    destination: Path,
    replaced: Mapping[str, Callable[[], np.ndarray]],
    kind: gguf.GGMLQuantizationType | None = None,
) -> None:
    """A copy of a GGUF with these tensors, by name, holding the values their functions give in float32,
    stored as kind or, without one, at the type each had, rounded as llama.cpp's converter rounds; and
    everything else as it was, copied as gguf-py's gguf_new_metadata copies a GGUF."""
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
        item = field.types[-1] if field.types[0] == gguf.GGUFValueType.ARRAY else None
        writer.add_key_value(field.name, field.contents(), field.types[0], sub_type=item)
    kinds = {t.name: t.tensor_type if kind is None else kind for t in reader.tensors if t.name in replaced}
    for tensor in reader.tensors:
        if tensor.name in replaced:
            shape, dtype = _laid_out(list(reversed(tensor.shape.tolist())), kinds[tensor.name])
            nbytes = math.prod(shape) * dtype.itemsize
            writer.add_tensor_info(tensor.name, shape, dtype, nbytes, kinds[tensor.name])
        else:
            writer.add_tensor_info(
                tensor.name, tensor.data.shape, tensor.data.dtype, tensor.data.nbytes, tensor.tensor_type
            )
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    for tensor in reader.tensors:
        if tensor.name in replaced:
            writer.write_tensor_data(gguf.quants.quantize(replaced[tensor.name](), kinds[tensor.name]))
        else:
            writer.write_tensor_data(tensor.data)
    writer.close()


def _laid_out(shape: list[int], kind: gguf.GGMLQuantizationType) -> tuple[list[int], np.dtype]:
    """The shape and type gguf-py holds a tensor of this shape in, stored as kind: floats for F32 and
    F16, bytes for the rest."""
    if kind == F32:
        return shape, np.dtype(np.float32)
    if kind == gguf.GGMLQuantizationType.F16:
        return shape, np.dtype(np.float16)
    return list(gguf.quant_shape_to_byte_shape(shape, kind)), np.dtype(np.uint8)


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
    return {target.gguf for target in run.edit_targets.values() if edit.strength(target.writers[0])}


def _edited_outside(run: Run, given: Path) -> set[str]:
    """The GGUF names of the weights taken from a model edited elsewhere."""
    return {run.edit_targets[name].gguf for name in _taken(run, given)}


def _short(held: dict[str, float]) -> dict[str, float]:
    return {name: share for name, share in held.items() if share < LEAST_KEPT}


def _span(shares: dict[str, float]) -> str:
    return f"{min(shares.values()):.1%} to {max(shares.values()):.1%}"


def _worst(shares: dict[str, float]) -> str:
    worst = sorted(shares.items(), key=lambda item: item[1])[:5]
    return ", ".join(f"{name} {share:.1%}" for name, share in worst)


def kept(run: Run, model_gguf: Path, made: Mapping[str, Change]) -> dict[str, float]:
    """For each edited weight, by its GGUF name, the share of its edit model_gguf holds: the part of
    its change from the original weight that lies along the edit, over the edit; for a layer's
    experts, the least any expert holds. 1 when the file holds the edit exactly; less when quantising
    rounded part of it away."""
    _, _, arch, _ = run.shape
    model_config = source.read_config(str(run.model_folder))
    tensors = {tensor.name: tensor for tensor in gguf.GGUFReader(model_gguf).tensors}
    files = ModelFiles(run.model_folder)
    shares = {}
    for name, change in made.items():
        tensor = tensors[run.edit_targets[name].gguf]
        stored = convert.stored(tensor)
        target = run.edit_targets[name]
        weight = files.read(target.names, target.stacked)
        original = convert.as_converted(weight, name, arch, model_config).float()
        edit = convert.as_converted(edited(weight, change) - weight.float(), name, arch, model_config)
        # A share for each expert, or one for a single weight; an expert the edit leaves alone has none.
        along, size = ((stored - original) * edit).sum((-2, -1)), (edit * edit).sum((-2, -1))
        moved = size > 0
        shares[tensor.name] = float((along[moved] / size[moved]).min()) if moved.any() else 1.0
    return shares


def _sweep(run: Run, shipped: Path) -> None:
    """Remove the work folders exports killed part way left in any edit's folder, which hold copies
    of the model, and this edit's exports in another quantisation, saying so."""
    root = run.path("exports")
    kept = {shipped.name, record_of(shipped).name, "convert.log"}
    for folder in root.iterdir() if root.exists() else []:
        for leftover in folder.iterdir() if folder.is_dir() else []:
            if leftover.is_dir():
                run.report(f"Removing {leftover.relative_to(run.folder)}, left by an export stopped part way")
                shutil.rmtree(leftover)
            elif folder == shipped.parent and leftover.name not in kept:
                run.report(f"Removing {leftover.relative_to(run.folder)}, which this run no longer uses")
                leftover.unlink()


def outside_export(run: Run, name: str) -> tuple[Path, dict[str, Any]]:
    """A model edited elsewhere as export_outside shipped it, and its record, for a caller holding the
    run's lock. Fails unless it matches the run and the folder it was taken from as they are now."""
    path = _path(run, name)
    if not record_of(path).exists():
        quantization = run.settings.deployment.quantization
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
    path, inputs = _planned(run, edit, read=False)
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
