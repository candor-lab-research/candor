"""Which tensors of an export differ from the unedited GGUF, by kind, with their layers and types."""
import hashlib
import re
import sys
from collections import defaultdict

from gguf import GGUFReader


def tensors(path):
    return {t.name: (t.tensor_type.name, hashlib.sha256(t.data.tobytes()).hexdigest()) for t in GGUFReader(path).tensors}


base, edited = tensors(sys.argv[1]), tensors(sys.argv[2])
assert base.keys() == edited.keys(), "different tensor names"
kinds = defaultdict(list)
for name in base:
    if base[name] != edited[name]:
        layer, kind = re.match(r"blk\.(\d+)\.(.+)\.weight", name).groups()
        kinds[kind].append((int(layer), edited[name][0]))
print("tensors", len(base), "differ", sum(map(len, kinds.values())))
for kind, rows in kinds.items():
    layers = sorted(l for l, _ in rows)
    print(kind, len(rows), f"layers {layers[0]}-{layers[-1]}", "contiguous" if layers == list(range(layers[0], layers[-1] + 1)) else layers, sorted({t for _, t in rows}))
