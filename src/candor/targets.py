"""Find the weights that write into the residual stream, which are what an edit has to change.

Every per-layer tensor whose output has the model's hidden size is a candidate. Names then say
whether it writes into the residual stream (an output projection) or reads from it (a query or
gate projection of the same shape). Anything the names can't place is reported rather than
skipped, because a writer that is silently left out weakens the edit without any error.
"""

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from functools import cache
from importlib import resources
from typing import Any

import gguf

WRITERS = {
    "o_proj",
    "out_proj",
    "down_proj",
    "w2",
    "wo",
    "c_proj",
    "fc2",
    "dense",
    "dense_4h_to_h",
    "output_linear",
}
READERS = {
    "q_proj", "k_proj", "v_proj", "qkv_proj", "query_key_value", "gate_proj", "up_proj", "gate_up_proj",
    "w1", "w3", "fc1", "c_attn", "c_fc", "wq", "wk", "wv", "wqkv", "dense_h_to_4h",
}  # fmt: skip
NOT_WEIGHTS = {"bias", "biases", "scales", "weight_scale", "weight_scale_inv", "input_scale"}
QUANTIZATION = NOT_WEIGHTS - {"bias"}
OUTPUT_NORMS = {"post_feedforward_layernorm", "post_self_attn_layernorm", "post_mlp_layernorm"}
ATTENTION = {"self_attn", "attn", "attention", "linear_attn", "mixer", "conv"}
MLP = {"mlp", "feed_forward", "ffn", "moe", "block_sparse_moe", "shared_mlp"}
OTHER_STACKS = ("vision", "visual", "audio", "encoder", "mtp")
LAYER = re.compile(r"^(.*?layers)\.(\d+)\.(.+)$")


@dataclass(frozen=True)
class Writer:
    name: str
    layer: int
    component: str
    shape: tuple[int, ...]
    expert: bool = False

    @property
    def kind(self) -> str:
        return self.name.split(f".{self.layer}.", 1)[1].removesuffix(".weight")


@dataclass
class Targets:
    layers: int
    stack: str
    writers: list[Writer] = field(default_factory=list)
    unclassified: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    extra_layers: list[int] = field(default_factory=list)
    output_norms: bool = False
    quantized: list[str] = field(default_factory=list)

    @property
    def experts(self) -> list[Writer]:
        return [w for w in self.writers if w.expert]

    @property
    def ok(self) -> bool:
        return bool(self.writers) and not self.unclassified and not self.missing

    def summary(self) -> dict[str, Counter]:
        """For each component, how many layers use each kind of writer."""
        kinds: dict[str, Counter] = defaultdict(Counter)
        for writer in self.writers:
            kinds[writer.component][re.sub(r"experts\.\d+\.", "experts.N.", writer.kind)] += 1
        return dict(kinds)


def _role(leaf: str) -> str | None:
    if leaf in WRITERS:
        return "writer"
    if leaf in READERS or leaf.startswith("in_proj"):
        return "reader"
    return None


def _component(path: list[str]) -> str | None:
    if any(part in ATTENTION for part in path):
        return "attn"
    if any(part in MLP or part == "experts" for part in path):
        return "mlp"
    return None


def _choose_stack(stacks: dict[str, set[int]], layers: int) -> tuple[str | None, str]:
    fits = [prefix for prefix, found in stacks.items() if set(range(layers)) <= found]
    if len(fits) > 1:
        fits = [prefix for prefix in fits if not any(word in prefix for word in OTHER_STACKS)] or fits
    if len(fits) == 1:
        return fits[0], ""
    if fits:
        return None, f"several layer stacks fit {layers} layers: {', '.join(sorted(fits))}"
    return None, f"no stack holds layers 0 to {layers - 1}; found {', '.join(sorted(stacks)) or 'none'}"


def find_targets(shapes: dict[str, list[int]], hidden: int, layers: int) -> Targets:
    stacks: dict[str, set[int]] = defaultdict(set)
    for name in shapes:
        if match := LAYER.match(name):
            stacks[match[1]].add(int(match[2]))
    stack, problem = _choose_stack(stacks, layers)
    if stack is None:
        return Targets(layers, "", missing=[problem])

    targets = Targets(layers, stack, extra_layers=sorted(stacks[stack] - set(range(layers))))
    scaled: set[str] = set()
    for name, shape in sorted(shapes.items()):
        match = LAYER.match(name)
        if not match or match[1] != stack or int(match[2]) >= layers:
            continue
        path = match[3].split(".")
        if OUTPUT_NORMS & set(path):
            targets.output_norms = True
        if path[-1] in QUANTIZATION:
            scaled.add(name.rsplit(".", 1)[0])
        if path[-1] in NOT_WEIGHTS:
            continue
        if path[-1] == "weight":
            path = path[:-1]
        writes_hidden = (len(shape) == 2 and shape[0] == hidden) or (len(shape) == 3 and hidden in shape[1:])
        if not writes_hidden:
            continue
        role, component = _role(path[-1]), _component(path[:-1])
        if role == "reader":
            continue
        if role == "writer" and component:
            targets.writers.append(Writer(name, int(match[2]), component, tuple(shape), "experts" in path))
        else:
            targets.unclassified.append(name)
    targets.writers.sort(key=lambda w: (w.layer, w.component, w.name))
    targets.quantized = [w.name for w in targets.writers if w.name.removesuffix(".weight") in scaled]
    for layer in range(layers):
        for component in ("attn", "mlp"):
            if not any(w.layer == layer and w.component == component for w in targets.writers):
                targets.missing.append(f"layer {layer} has no {component} writer")
    return targets


@cache
def _converter_table() -> dict[str, Any]:
    path = resources.files("candor") / "data" / "llama_cpp_architectures.json"
    return json.loads(path.read_text())


def gguf_architecture(config: dict[str, Any]) -> str | None:
    """llama.cpp's name for the model, chosen the way its converter chooses: the first top-level
    architecture, replaced by text_config's unless it is one of the converter's exceptions."""
    table = _converter_table()
    architecture = (config.get("architectures") or [None])[0]
    text_architectures = (config.get("text_config") or {}).get("architectures")
    if architecture not in table["top_level_only"] and text_architectures:
        architecture = text_architectures[0]
    return table["architectures"].get(architecture)


@cache
def _name_map(arch: str, layers: int) -> gguf.TensorNameMap:
    arch_enum = next(a for a, n in gguf.MODEL_ARCH_NAMES.items() if n == arch)
    return gguf.get_tensor_name_map(arch_enum, layers)


def gguf_name(writer: Writer, arch: str, layers: int) -> str | None:
    """The name llama.cpp's converter will give this weight, mirroring how it strips language_model."""
    module = writer.name.removesuffix(".weight").replace("language_model.", "")
    mapped = _name_map(arch, layers).get_name(module)
    return f"{mapped}.weight" if mapped else None
