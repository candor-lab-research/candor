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
# One expert's weight in a layer that stores each of its experts apart, as Laguna and Qwen3-MoE do.
EXPERT = re.compile(r"\.experts\.(\d+)\.")
# The llama.cpp architectures whose memory candor knows: a K and a V of each KV head for every token of
# context in each layer that attends to all of it, and in each of Qwen3.5's linear-attention layers a
# state per slot that doesn't grow with the context.
KV_ARCHITECTURES = {"llama", "qwen2", "qwen3", "qwen35", "k2-horizon", "qwen3moe", "olmoe"}
FULL, LINEAR = "full_attention", "linear_attention"


@dataclass(frozen=True)
class Writer:
    name: str
    layer: int
    component: str
    expert: bool = False

    @property
    def kind(self) -> str:
        return self.name.split(f".{self.layer}.", 1)[1].removesuffix(".weight")


@dataclass(frozen=True)
class Target:
    """One weight an edit changes, as llama.cpp stores it: a writer, or a layer's experts stacked along
    a first axis in expert order, as the converter stacks them. name is what the converter maps to the
    GGUF name: the writer's own, or for experts stored apart, theirs without the expert number."""

    name: str
    gguf: str
    writers: tuple[Writer, ...]

    @property
    def layer(self) -> int:
        return self.writers[0].layer

    @property
    def names(self) -> list[str]:
        return [w.name for w in self.writers]

    @property
    def stacked(self) -> bool:
        """Whether the GGUF stacks the writers, as it does experts stored apart, even one alone."""
        return self.name != self.writers[0].name


@dataclass
class Targets:
    writers: list[Writer] = field(default_factory=list)
    unclassified: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    extra_layers: list[int] = field(default_factory=list)
    output_norms: bool = False
    quantized: list[str] = field(default_factory=list)
    # Experts held as one tensor with the hidden size last, [experts, in, hidden], which the converter
    # transposes and candor doesn't edit yet, or square, so which way they are held can't be told.
    hidden_last: list[str] = field(default_factory=list)

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
        return Targets(missing=[problem])

    targets = Targets(extra_layers=sorted(stacks[stack] - set(range(layers))))
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
            targets.writers.append(
                Writer(name, int(match[2]), component, "experts" in path or len(shape) == 3)
            )
            if len(shape) == 3 and shape[2] == hidden:
                targets.hidden_last.append(name)
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
    """The name llama.cpp's converter will give this weight, mirroring how it strips language_model and
    stacks experts stored apart into one tensor named without the expert number."""
    module = _stacked_name(writer.name).removesuffix(".weight").replace("language_model.", "")
    mapped = _name_map(arch, layers).get_name(module)
    return f"{mapped}.weight" if mapped else None


def _stacked_name(name: str) -> str:
    return EXPERT.sub(".experts.", name)


def edit_targets(writers: list[Writer], arch: str, layers: int) -> dict[str, Target]:
    """The weights an edit changes, by name, each writer in one: on its own, or with the rest of its
    layer's experts, stacked in expert order. Raises ValueError for a writer with no GGUF name, or
    experts not numbered 0 to n - 1."""
    grouped: dict[str, list[Writer]] = defaultdict(list)
    for writer in writers:
        grouped[_stacked_name(writer.name)].append(writer)
    found = {}
    for name, members in grouped.items():
        gguf = gguf_name(members[0], arch, layers)
        if gguf is None:
            raise ValueError(f"no llama.cpp name for {members[0].name}")
        numbered = [EXPERT.search(w.name) for w in members]
        if len(members) > 1 or numbered[0]:
            if not all(numbered):
                raise ValueError(f"{name} is held both as one tensor and as experts stored apart")
            by_number = {int(match[1]): w for match, w in zip(numbered, members, strict=True) if match}
            if sorted(by_number) != list(range(len(members))):
                raise ValueError(f"{name}'s experts aren't numbered 0 to {len(members) - 1}")
            members = [by_number[i] for i in range(len(members))]
        found[name] = Target(name, gguf, tuple(members))
    return found


def _layer_kinds(text: dict[str, Any], arch: str | None) -> list[str] | None:
    """Each layer's kind of attention as llama.cpp reads it: from layer_types, or without them every
    layer's full, but Qwen3.5's every full_attention_interval-th (4 by default); None for a kind
    candor doesn't know."""
    kinds = text.get("layer_types")
    if kinds is not None:
        return kinds if set(kinds) <= {FULL, LINEAR} else None
    layers = text.get("num_hidden_layers")
    if not isinstance(layers, int):
        return None
    if arch != "qwen35":
        return [FULL] * layers
    interval = text.get("full_attention_interval", 4)
    if not isinstance(interval, int) or interval < 1:
        return None
    return [FULL if (i + 1) % interval == 0 else LINEAR for i in range(layers)]


def kv_bytes_per_token(config: dict[str, Any], arch: str | None, bits: float) -> float | None:
    """The KV cache a token of context takes at this many bits a value, where candor knows how the
    architecture lays it out and the config gives its sizes, the head size and KV heads defaulting as in
    transformers and llama.cpp's converter. None otherwise, or for a model no layer of which has one."""
    if arch not in KV_ARCHITECTURES:
        return None
    text = config.get("text_config", config)
    heads, hidden = text.get("num_attention_heads"), text.get("hidden_size")
    kv_heads = text.get("num_key_value_heads", heads)
    whole = isinstance(hidden, int) and isinstance(heads, int) and heads > 0
    size = text.get("head_dim") or (hidden // heads if whole else None)
    kinds = _layer_kinds(text, arch)
    if kinds is None or not (isinstance(kv_heads, int) and isinstance(size, int)):
        return None
    return kinds.count(FULL) * 2 * kv_heads * size * bits / 8 or None


def recurrent_bytes_per_slot(config: dict[str, Any], arch: str | None) -> int | None:
    """The state each slot holds in Qwen3.5's linear-attention layers, the same whatever its context, in
    f32 whatever the KV cache type, sized as llama.cpp's n_embd_r and n_embd_s size it from the
    converter's settings. 0 for a model with no such layer; None where the config doesn't give its sizes."""
    text = config.get("text_config", config)
    kinds = _layer_kinds(text, arch) or []
    if arch != "qwen35" or LINEAR not in kinds:
        return 0
    names = (
        "linear_conv_kernel_dim",
        "linear_key_head_dim",
        "linear_num_key_heads",
        "linear_value_head_dim",
        "linear_num_value_heads",
    )
    sizes = [text.get(name) for name in names]
    if not all(isinstance(n, int) for n in sizes):
        return None
    kernel, key_size, key_heads, value_size, value_heads = sizes
    inner = value_size * value_heads
    conv = (kernel - 1) * (inner + 2 * key_heads * key_size)
    return kinds.count(LINEAR) * (conv + key_size * inner) * 4
