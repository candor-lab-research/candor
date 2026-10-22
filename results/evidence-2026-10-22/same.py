"""Checks a release GGUF against the measured export: every tensor byte-identical, and which metadata
keys differ."""

import hashlib
import sys

from gguf import GGUFReader

a, b = GGUFReader(sys.argv[1]), GGUFReader(sys.argv[2])


def tensors(r):
    return {
        t.name: (t.tensor_type, tuple(t.shape), hashlib.sha256(t.data.tobytes()).hexdigest())
        for t in r.tensors
    }


def meta(r):
    return {
        k: hashlib.sha256(b"".join(bytes(f.parts[i]) for i in f.data)).hexdigest()
        for k, f in r.fields.items()
        if not k.startswith("GGUF.")
    }


ta, tb = tensors(a), tensors(b)
print("tensors", len(ta), len(tb), "identical" if ta == tb else "DIFFER")
ma, mb = meta(a), meta(b)
print("only in export:", sorted(set(ma) - set(mb)), "only in release:", sorted(set(mb) - set(ma)))
print("changed:", sorted(k for k in set(ma) & set(mb) if ma[k] != mb[k]))
print("name:", b.fields["general.name"].contents())
