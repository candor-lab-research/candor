"""Rank base models candor could edit by how many people run them as GGUFs against how many
uncensored versions already exist.

    python3 -I results/evidence-2026-10-23/gaps.py > results/evidence-2026-10-23/gaps.tsv

Candidates are the 1,000 most downloaded and 1,000 most trending text-generation models on the
Hub, kept when candor can edit them: an original (not a quantisation, merge, adapter, test model or
someone else's fine-tune), a chat model (a chat template, no "base" in its name), an architecture
llama.cpp converts other than a draft model, dense by its config.json (no total-experts count above
1), and a known size of at most 36B parameters (the 27B fitted an A40), by the larger of the
safetensors index's total and its sum by type. An uncensored repo is one whose name or tags say so,
found by name, as a GGUF of the model or as a fine-tune of it. Downloads are Hugging Face's 30-day
counts. Reads only public metadata.
"""

import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

API = "https://huggingface.co/api/models"
ROOT = Path(__file__).resolve().parents[2]
ARCHITECTURES = json.loads((ROOT / "src/candor/data/llama_cpp_architectures.json").read_text())
ARCHITECTURES = ARCHITECTURES["architectures"]
UNCENSORED = ("uncensor", "abliterat", "heretic", "derestrict", "decensor", "refusal")
UNCENSORED_TAGS = {"uncensored", "abliterated"}
TEST_ORGS = ("nm-testing",)
TOTAL_EXPERTS = (
    "num_experts",
    "num_local_experts",
    "n_routed_experts",
    "moe_num_experts",
    "num_experts_total",
)
ACTIVE_EXPERTS = (
    "num_experts_per_tok",
    "num_experts_per_token",
    "moe_topk",
    "top_k_experts",
    "experts_per_token",
)
NOT_ORIGINAL = ("gguf", "awq", "gptq", "mlx", "fp8", "int4", "int8", "nvfp4", "bnb", "exl", "4bit", "8bit")
NOT_ORIGINAL += ("quantized", "w4a16")
MAX_PARAMS = 36e9
RELEASED = ["Qwen/Qwen3.8-27B", "IFM/K2-Horizon-7B", "Qwen/Qwen3.5-9B", "meta-llama/Llama-3.2-3B-Instruct"]
TOKEN = Path.home().joinpath(".cache/huggingface/token").read_text().strip()


def get(params, path="", url=None):
    url = url or API + path + "?" + urllib.parse.urlencode(params, doseq=True)
    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {TOKEN}"})
    for wait in (5, 15, 45, 120, 300, 600):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            if error.code != 429:
                raise
            time.sleep(wait)
    raise RuntimeError(f"rate limited: {url}")


def uncensored(model_id, tags=()):
    name = model_id.lower()
    said = any(word in name for word in UNCENSORED) or bool(UNCENSORED_TAGS & {t.lower() for t in tags})
    return said and "candor" not in name


def experts(model):
    """The model's total experts and experts used per token, by its config.json; (None, None) if dense."""
    try:
        config = get({}, url=f"https://huggingface.co/{model['id']}/resolve/main/config.json")
    except urllib.error.HTTPError:
        config = model.get("config") or {}
    for part in (config.get("text_config") or {}, config):
        total = next((part[k] for k in TOTAL_EXPERTS if isinstance(part.get(k), int) and part[k] > 1), None)
        if total:
            return total, next((part[k] for k in ACTIVE_EXPERTS if isinstance(part.get(k), int)), None)
    return None, None


def dense(model):
    return experts(model)[0] is None


def size(model):
    """Parameters, from the safetensors index: the larger of its total and its sum by type, as the total
    is sometimes wrong; None if unknown."""
    counts = model.get("safetensors") or {}
    known = [n for n in (counts.get("total"), sum((counts.get("parameters") or {}).values())) if n]
    return max(known) if known else None


def editable(model):
    name = model["id"].lower()
    tags = model.get("tags", [])
    config = model.get("config") or {}
    words = name.replace("_", "-").split("-")
    if uncensored(name, tags) or any(word in name for word in NOT_ORIGINAL) or "base" in words:
        return False
    org = name.split("/")[0]
    if org.endswith("internal-testing") or org in TEST_ORGS or "tiny-random" in name:
        return False
    if not (config.get("chat_template_jinja") or (config.get("tokenizer_config") or {}).get("chat_template")):
        return False
    if any(t.startswith(("base_model:quantized:", "base_model:merge:", "base_model:adapter:")) for t in tags):
        return False
    parents = [t.split(":", 2)[2] for t in tags if t.startswith("base_model:finetune:")]
    if any(parent.split("/")[0] != model["id"].split("/")[0] for parent in parents):
        return False
    architectures = config.get("architectures") or []
    params = size(model)
    known = bool(architectures) and architectures[0] in ARCHITECTURES and "Draft" not in architectures[0]
    return known and params is not None and params <= MAX_PARAMS


def names(model_id):
    """The names an uncensored version of the model is found by: its own, without its weights' type or
    chat suffix, and without a first word that repeats its org (NVIDIA-Nemotron-... of nvidia)."""
    org, name = model_id.split("/")
    short = name
    for suffix in ("-BF16", "-hf", "-Instruct", "-instruct", "-it", "-Chat", "-chat"):
        short = short.removesuffix(suffix)
    found = {name, short}
    if short.lower().startswith(org.lower() + "-"):
        found.add(short[len(org) + 1 :])
    return found


def gap(model):
    ggufs = get(
        {
            "filter": [f"base_model:quantized:{model['id']}", "gguf"],
            "limit": 1000,
            "expand[]": ["downloads", "tags"],
        }
    )
    found = {m["id"]: m["downloads"] for m in ggufs if uncensored(m["id"], m.get("tags", []))}
    tuned = get(
        {"filter": f"base_model:finetune:{model['id']}", "limit": 1000, "expand[]": ["downloads", "tags"]}
    )
    found |= {m["id"]: m["downloads"] for m in tuned if uncensored(m["id"], m.get("tags", []))}
    for name in names(model["id"]):
        for m in get({"search": name, "limit": 1000, "expand[]": ["downloads", "tags"]}):
            if name.lower() in m["id"].lower() and uncensored(m["id"], m.get("tags", [])):
                found[m["id"]] = m["downloads"]
    plain = [m["downloads"] for m in ggufs if not uncensored(m["id"], m.get("tags", []))]
    demand = sum(plain)
    return {
        "id": model["id"],
        "created": model.get("createdAt", "")[:10],
        "params_b": round((size(model) or 0) / 1e9, 1),
        "architecture": (model.get("config") or {}).get("architectures", ["?"])[0],
        "gguf_repos": len(ggufs),
        "gguf_downloads": demand,
        "top_gguf_downloads": max(plain, default=0),
        "uncensored_repos": len(found),
        "uncensored_downloads": sum(found.values()),
        "gap": round(demand / (1 + len(found))),
        "gap_without_top": round((demand - max(plain, default=0)) / (1 + len(found))),
        "capped": len(ggufs) == 1000,
    }


def main():
    expand = ["downloads", "createdAt", "safetensors", "tags", "config"]
    pool = {}
    for sort in ("downloads", "trendingScore"):
        for model in get(
            {"pipeline_tag": "text-generation", "sort": sort, "limit": 1000, "expand[]": expand}
        ):
            pool[model["id"]] = model
    for model_id in RELEASED:
        pool[model_id] = get({"expand[]": expand}, f"/{model_id}")
    candidates = [m for m in pool.values() if editable(m) or m["id"] in RELEASED]
    with ThreadPoolExecutor(8) as threads:
        candidates = [m for m, ok in zip(candidates, threads.map(dense, candidates), strict=True) if ok]
        print(f"{len(pool)} models, {len(candidates)} candor can edit", file=sys.stderr)
        rows = sorted(threads.map(gap, candidates), key=lambda r: r["gap"], reverse=True)
    print("\t".join(rows[0]))
    for row in rows:
        print("\t".join(str(v) for v in row.values()))


if __name__ == "__main__":
    main()
