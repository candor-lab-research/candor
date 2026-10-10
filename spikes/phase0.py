"""Phase 0: can llama.cpp score abliteration edits as per-layer LoRA adapters?

Runs on a GPU pod after spikes/setup_pod.sh. Each stage saves its output under WORK and is skipped
when that output already exists, so a failed run can simply be started again.

Generated answers go to WORK/private and are never printed. Everything shown is a count or a
measurement.

The questions:
1. Does the base model plus adapters at the trial's strengths behave like the merged model?
2. Does that hold for Q4_K_M, where the adapters sit on a quantised base?
3. What do loaded and active adapters cost in startup time and generation speed?
4. How close is KL from the server's top-k log probabilities to the exact value?
"""

import gc
import json
import os
import subprocess
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from candor import kl, reference
from candor.adapters import architecture, layer_adapter_paths, write_layer_adapters
from candor.directions import after_layer, difference_of_means
from candor.edit import RowNorm, rank_one
from candor.engine import LlamaServer
from candor.judge import count_refusals
from candor.kernel import Kernel
from candor.merge import merge
from candor.prompts import HARMFUL_TEST, HARMFUL_TRAIN, HARMLESS_TEST, HARMLESS_TRAIN, encode
from candor.residuals import last_token_residuals
from candor.weights import DENSE, ModelFiles

MODEL = "unsloth/Llama-3.2-3B-Instruct"
WORK = Path(os.environ.get("WORK", "/root/work/phase0"))
LLAMA = Path(os.environ.get("LLAMA_BIN", "/root/llama/llama-b11524"))
CONVERTER = Path(os.environ.get("CONVERTER", "/root/llama.cpp/convert_hf_to_gguf.py"))
CONVERT_PYTHON = Path(os.environ.get("CONVERT_PYTHON", "/root/convenv/bin/python"))

# rude-llama's heretic Trial 170. rude-llama is a fine-tune of this model, so the shape transfers.
DIRECTION_LAYER = 12.038957433441261
KERNELS = {
    "attn": Kernel(1.3274761740029895, 16.526312731443696, 1.0459709836345665, 13.477533432535957),
    "mlp": Kernel(1.2617342942240766, 16.559180104404113, 0.5747798917065232, 12.769805326769387),
}
ROW_NORM = RowNorm.PRE
EXTRA_DIRECTION_LAYERS = range(8, 17)
TOP_K = 1000
MAX_TOKENS = 100
PARALLEL = 16


def save_json(name: str, data) -> None:
    (WORK / name).write_text(json.dumps(data, indent=1))


def load_json(name: str) -> dict:
    path = WORK / name
    return json.loads(path.read_text()) if path.exists() else {}


def keep_private(name: str, answers: list[str]) -> None:
    folder = WORK / "private"
    folder.mkdir(exist_ok=True)
    (folder / f"{name}.jsonl").write_text("".join(json.dumps(answer) + "\n" for answer in answers))


def load_model(folder: Path):
    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(folder, dtype=torch.bfloat16)
    return model.to("cuda").eval()  # pyright: ignore[reportArgumentType]


def free_gpu() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def find_directions(model, tokenizer: PreTrainedTokenizerBase, pad_id: int) -> torch.Tensor:
    harmful = last_token_residuals(model, encode(tokenizer, HARMFUL_TRAIN.load()), pad_id)
    harmless = last_token_residuals(model, encode(tokenizer, HARMLESS_TRAIN.load()), pad_id)
    return difference_of_means(harmful, harmless)


def measure_torch(
    name: str, model, tokenizer, harmful: list[list[int]], harmless: list[list[int]], pad_id: int
):
    results = load_json("results.json")
    if name in results:
        return
    torch.save(reference.first_token_logprobs(model, harmless, pad_id), WORK / f"{name}.logprobs.pt")
    answers = reference.greedy(model, tokenizer, harmful, MAX_TOKENS)
    keep_private(name, answers)
    results[name] = {"refusals": count_refusals(answers)}
    save_json("results.json", results)


def run_logged(args: list, log: Path) -> None:
    with log.open("ab") as out:
        subprocess.run([str(arg) for arg in args], stdout=out, stderr=subprocess.STDOUT, check=True)


def convert(folder: Path, name: str) -> tuple[Path, Path]:
    bf16, q4 = WORK / f"{name}-bf16.gguf", WORK / f"{name}-Q4_K_M.gguf"
    log = WORK / "logs" / f"convert-{name}.log"
    if not bf16.exists():
        partial = bf16.with_suffix(".partial")
        run_logged([CONVERT_PYTHON, CONVERTER, folder, "--outtype", "bf16", "--outfile", partial], log)
        partial.rename(bf16)
    if not q4.exists():
        partial = q4.with_suffix(".partial")
        run_logged([LLAMA / "llama-quantize", bf16, partial, "Q4_K_M"], log)
        partial.rename(q4)
    return bf16, q4


def truncate(tops: list[dict[int, float]], k: int) -> list[dict[int, float]]:
    return [dict(sorted(entry.items(), key=lambda item: -item[1])[:k]) for entry in tops]


def argmax_agreement(a: list[dict[int, float]], b: list[dict[int, float]]) -> int:
    return sum(max(x, key=x.__getitem__) == max(y, key=y.__getitem__) for x, y in zip(a, b, strict=True))


def main() -> None:
    (WORK / "logs").mkdir(parents=True, exist_ok=True)
    source = Path(snapshot_download(MODEL, allow_patterns=["*.json", "*.safetensors", "*.jinja"]))
    tokenizer = AutoTokenizer.from_pretrained(source)
    pad_id = reference.pad_token_id(tokenizer)
    harmful = encode(tokenizer, HARMFUL_TEST.load())
    harmless = encode(tokenizer, HARMLESS_TEST.load())

    directions_path = WORK / "directions.pt"
    if not directions_path.exists() or "torch-base" not in load_json("results.json"):
        print("Base model: directions and reference measurements", flush=True)
        model = load_model(source)
        if not directions_path.exists():
            torch.save(find_directions(model, tokenizer, pad_id), directions_path)
        measure_torch("torch-base", model, tokenizer, harmful, harmless, pad_id)
        del model
        free_gpu()
    directions = torch.load(directions_path)
    layers = len(directions) - 1
    trial_direction = after_layer(directions, DIRECTION_LAYER)
    strengths = {
        (t.component, layer): KERNELS[t.component].at(layer) for t in DENSE for layer in range(layers)
    }

    files = ModelFiles(source)
    merged = WORK / "merged"
    if not merged.exists():
        print("Merging the trial into a copy of the model", flush=True)
        deltas = {}
        for target in DENSE:
            for layer in range(layers):
                if strength := strengths[target.component, layer]:
                    hf_name, _ = target.names(layer)
                    b, a = rank_one(files.tensor(hf_name), trial_direction, ROW_NORM)
                    deltas[hf_name] = (strength * b, a)
        merge(source, merged, deltas)

    if "torch-merged" not in load_json("results.json"):
        print("Merged model: reference measurements", flush=True)
        model = load_model(merged)
        measure_torch("torch-merged", model, tokenizer, harmful, harmless, pad_id)
        del model
        free_gpu()

    print("Converting to GGUF", flush=True)
    base_bf16, base_q4 = convert(source, "base")
    merged_bf16, merged_q4 = convert(merged, "merged")

    print("Writing adapters", flush=True)
    arch = architecture(base_bf16)
    sets = {"trial": trial_direction.expand(layers, -1), "per-layer": directions[1:]}
    sets |= {f"after-{layer}": directions[layer + 1].expand(layers, -1) for layer in EXTRA_DIRECTION_LAYERS}
    adapters = {}
    for name, set_directions in sets.items():
        folder = WORK / "adapters" / name
        if (folder / "complete").exists():
            adapters[name] = layer_adapter_paths(folder, DENSE, layers)
        else:
            adapters[name] = write_layer_adapters(folder, arch, files, DENSE, set_directions, ROW_NORM)
            (folder / "complete").touch()
    trial_scales = {adapters["trial"][key]: strength for key, strength in strengths.items()}
    every_adapter = [path for paths in adapters.values() for path in paths.values()]

    results = load_json("results.json")
    tops: dict[str, list[dict[int, float]]] = {}
    speed: dict[str, float] = {}

    def measure(server: LlamaServer, name: str, scales=None) -> None:
        tops[name] = server.top_logprobs(harmless, TOP_K, scales)
        answers = server.generate(harmful, MAX_TOKENS, scales)
        keep_private(name, answers)
        results[name] = {"refusals": count_refusals(answers)}

    def server(model: Path, adapter_paths=()) -> LlamaServer:
        return LlamaServer(
            LLAMA / "llama-server", model, WORK / "logs" / "server.log", adapter_paths, PARALLEL, 512
        )

    print("Measuring with llama-server", flush=True)
    with server(base_bf16, list(trial_scales)) as s:
        measure(s, "llama-bf16-base")
        measure(s, "llama-bf16-lora", trial_scales)
    with server(merged_bf16) as s:
        measure(s, "llama-bf16-merged")
    with server(base_q4) as s:
        speed["startup, no adapters"] = s.startup_seconds
        speed["tokens/s, no adapters"] = s.throughput(harmful, MAX_TOKENS)
        measure(s, "llama-q4-base")
    with server(base_q4, list(trial_scales)) as s:
        measure(s, "llama-q4-lora-trial-only", trial_scales)
    with server(base_q4, every_adapter) as s:
        speed[f"startup, {len(every_adapter)} adapters"] = s.startup_seconds
        speed["tokens/s, all loaded, none active"] = s.throughput(harmful, MAX_TOKENS)
        speed[f"tokens/s, trial active ({sum(map(bool, trial_scales.values()))})"] = s.throughput(
            harmful, MAX_TOKENS, trial_scales
        )
        measure(s, "llama-q4-base-all-loaded")
        measure(s, "llama-q4-lora", trial_scales)
    with server(merged_q4) as s:
        measure(s, "llama-q4-merged")
    save_json("tops.json", tops)

    base_exact = torch.load(WORK / "torch-base.logprobs.pt")
    merged_exact = torch.load(WORK / "torch-merged.logprobs.pt")
    kl_results = {"exact, torch": kl.exact(base_exact, merged_exact)}
    for k in (20, 100, 1000):
        kl_results[f"top-{k} from torch"] = kl.mean_from_top(kl.top(base_exact, k), kl.top(merged_exact, k))
    for k in (20, 100, 1000):
        kl_results[f"top-{k}, llama bf16 base -> lora"] = kl.mean_from_top(
            truncate(tops["llama-bf16-base"], k), truncate(tops["llama-bf16-lora"], k)
        )
    kl_results["top-1000, llama bf16 base -> merged"] = kl.mean_from_top(
        tops["llama-bf16-base"], tops["llama-bf16-merged"]
    )
    kl_results["top-1000, llama Q4 base -> lora"] = kl.mean_from_top(
        tops["llama-q4-base"], tops["llama-q4-lora"]
    )
    kl_results["top-1000, llama Q4 base -> merged"] = kl.mean_from_top(
        tops["llama-q4-base"], tops["llama-q4-merged"]
    )
    pairs = {
        "bf16, trial adapters loaded": ("llama-bf16-merged", "llama-bf16-lora"),
        "Q4, trial adapters loaded": ("llama-q4-merged", "llama-q4-lora-trial-only"),
        f"Q4, all {len(every_adapter)} loaded": ("llama-q4-merged", "llama-q4-lora"),
        "Q4 base, none loaded vs all at 0": ("llama-q4-base", "llama-q4-base-all-loaded"),
    }
    agreement = {
        name: {
            "kl": kl.mean_from_top(tops[a], tops[b]),
            "same first token": argmax_agreement(tops[a], tops[b]),
        }
        for name, (a, b) in pairs.items()
    }
    results |= {"kl": kl_results, "should match": agreement, "speed": speed}
    save_json("results.json", results)
    report(results)


def report(results: dict) -> None:
    print(f"\nPhase 0 results ({MODEL}, rude-llama Trial 170 profile)\n")
    print("Refusals on 100 harmful prompts (greedy, 100 tokens)")
    for name, value in results.items():
        if name.startswith(("torch-", "llama-")):
            print(f"  {name:<28}{value['refusals']:>4}")
    print("\nKL on 100 harmless prompts (first answer token)")
    for name, value in results["kl"].items():
        print(f"  {name:<40}{value:>10.5f}")
    print("\nPairs that should match (100 harmless prompts)")
    for name, values in results["should match"].items():
        print(f"  {name:<36} KL {values['kl']:.6f}, same first token {values['same first token']}/100")
    print(f"\nSpeed (Q4_K_M, {PARALLEL} in parallel, {MAX_TOKENS} tokens each)")
    for name, value in results["speed"].items():
        print(f"  {name:<40}{value:>10.1f}")


if __name__ == "__main__":
    main()
