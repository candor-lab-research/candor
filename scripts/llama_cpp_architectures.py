"""Regenerate src/candor/data/llama_cpp_architectures.json from a llama.cpp checkout.

    python -I scripts/llama_cpp_architectures.py PATH_TO_LLAMA_CPP

Maps each Hugging Face architecture that llama.cpp's converter registers for a text model to the
architecture name it writes into the GGUF, and lists the architectures for which the converter
ignores text_config.architectures. The converter is only parsed, never imported or run. Run it
against the same release that candor pins (see pyproject.toml).
"""

import ast
import json
import sys
from pathlib import Path

OUTPUT = Path(__file__).resolve().parent.parent / "src/candor/data/llama_cpp_architectures.json"


def classes(source: Path) -> dict[str, tuple[list[str], list[str], str | None]]:
    """name -> (base class names, registered Hugging Face architectures, MODEL_ARCH member or None)"""
    found = {}
    files = [source / "convert_hf_to_gguf.py", *sorted((source / "conversion").glob("*.py"))]
    for path in files:
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.ClassDef):
                continue
            bases = [b.id if isinstance(b, ast.Name) else getattr(b, "attr", "") for b in node.bases]
            registered = []
            for decorator in node.decorator_list:
                if isinstance(decorator, ast.Call) and getattr(decorator.func, "attr", "") == "register":
                    registered += [a.value for a in decorator.args if isinstance(a, ast.Constant)]
            arch = None
            for statement in node.body:
                if (
                    isinstance(statement, ast.Assign)
                    and any(getattr(t, "id", "") == "model_arch" for t in statement.targets)
                    and isinstance(statement.value, ast.Attribute)
                ):
                    arch = statement.value.attr
            found[node.name] = (bases, registered, arch)
    return found


def top_level_only(source: Path) -> list[str]:
    """The architectures get_model_architecture keeps even when text_config names another."""
    tree = ast.parse((source / "conversion" / "base.py").read_text())
    function = next(
        n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "get_model_architecture"
    )
    names = []
    for node in ast.walk(function):
        if (
            isinstance(node, ast.Compare)
            and isinstance(node.ops[0], ast.In)
            and isinstance(node.comparators[0], ast.Tuple)
        ):
            names += [e.value for e in node.comparators[0].elts if isinstance(e, ast.Constant)]
    return sorted(names)


def main() -> None:
    import gguf

    source = Path(sys.argv[1])
    found = classes(source)

    def ancestors(name: str) -> list[str]:
        chain, queue = [], [name]
        while queue:
            current = queue.pop(0)
            if current in found and current not in chain:
                chain.append(current)
                queue += found[current][0]
        return chain

    names = {member.name: gguf.MODEL_ARCH_NAMES[member] for member in gguf.MODEL_ARCH}
    table = {}
    for name, (_, registered, _) in found.items():
        chain = ancestors(name)
        if "TextModel" not in {base for c in chain for base in found[c][0]} | set(chain):
            continue
        arch = next((found[c][2] for c in chain if found[c][2]), None)
        if arch in names:
            for hf in registered:
                table.setdefault(hf, names[arch])
    data = {"top_level_only": top_level_only(source), "architectures": dict(sorted(table.items()))}
    OUTPUT.parent.mkdir(exist_ok=True)
    OUTPUT.write_text(json.dumps(data, indent=1) + "\n")
    print(f"{len(table)} architectures, {len(data['top_level_only'])} top-level exceptions -> {OUTPUT}")


if __name__ == "__main__":
    main()
