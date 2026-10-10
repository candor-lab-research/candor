"""Find the weights that write into the residual stream, which are what an edit has to change.

Every per-layer tensor whose output has the model's hidden size is a candidate. Names then say
whether it writes into the residual stream (an output projection) or reads from it (a query or
gate projection of the same shape). Anything the names can't place is reported rather than
skipped, because a writer that is silently left out weakens the edit without any error.
"""

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field

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
ATTENTION = {"self_attn", "attn", "attention", "linear_attn", "mixer", "conv"}
MLP = {"mlp", "feed_forward", "ffn", "moe", "block_sparse_moe", "shared_mlp"}
LAYER = re.compile(r"^(.*?layers)\.(\d+)\.(.+)\.weight$")


@dataclass(frozen=True)
class Writer:
    name: str
    layer: int
    component: str
    shape: tuple[int, ...]


@dataclass
class Targets:
    layers: int
    stack: str
    writers: list[Writer] = field(default_factory=list)
    unclassified: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return bool(self.writers) and not self.unclassified and not self.missing

    def summary(self) -> dict[str, Counter]:
        """For each component, how many layers use each kind of writer."""
        kinds: dict[str, Counter] = defaultdict(Counter)
        for writer in self.writers:
            kinds[writer.component][writer.name.split(f".{writer.layer}.", 1)[1].removesuffix(".weight")] += 1
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


def find_targets(shapes: dict[str, list[int]], hidden: int, layers: int) -> Targets:
    stacks: dict[str, set[int]] = defaultdict(set)
    for name in shapes:
        if match := LAYER.match(name):
            stacks[match[1]].add(int(match[2]))
    stack = next((prefix for prefix, found in stacks.items() if found == set(range(layers))), None)
    if stack is None:
        return Targets(layers, "", missing=[f"no stack of exactly {layers} layers among {sorted(stacks)}"])

    targets = Targets(layers, stack)
    for name, shape in sorted(shapes.items()):
        match = LAYER.match(name)
        if not match or match[1] != stack:
            continue
        writes_hidden = (len(shape) == 2 and shape[0] == hidden) or (len(shape) == 3 and hidden in shape[1:])
        if not writes_hidden:
            continue
        path = match[3].split(".")
        role, component = _role(path[-1]), _component(path[:-1])
        if role == "reader":
            continue
        if role == "writer" and component:
            targets.writers.append(Writer(name, int(match[2]), component, tuple(shape)))
        else:
            targets.unclassified.append(name)
    targets.writers.sort(key=lambda w: (w.layer, w.component, w.name))
    for layer in range(layers):
        for component in ("attn", "mlp"):
            if not any(w.layer == layer and w.component == component for w in targets.writers):
                targets.missing.append(f"layer {layer} has no {component} writer")
    return targets


def gguf_architecture(model_type: str) -> str | None:
    """llama.cpp's name for a Hugging Face model_type, if the pinned gguf package knows it."""
    wanted = model_type.replace("_", "").replace("-", "")
    for name in gguf.MODEL_ARCH_NAMES.values():
        if name.replace("_", "").replace("-", "") == wanted:
            return name
    return None


def gguf_name(writer: Writer, arch: str) -> str | None:
    """The name llama.cpp's converter will give this weight, mirroring how it strips language_model."""
    arch_enum = next(a for a, n in gguf.MODEL_ARCH_NAMES.items() if n == arch)
    module = writer.name.removesuffix(".weight").replace("language_model.", "")
    mapped = gguf.get_tensor_name_map(arch_enum, writer.layer + 1).get_name(module)
    return f"{mapped}.weight" if mapped else None
