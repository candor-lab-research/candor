"""Rank base models candor could edit by how many people run them as GGUFs against how many
uncensored versions already exist.

    python3 -I results/evidence-2026-10-23/gaps.py > results/evidence-2026-10-23/gaps.tsv

Candidates are the 1,000 most downloaded and 1,000 most trending text-generation models on the
Hub, kept when candor can edit them: an original (not a quantisation, merge, adapter or someone
else's fine-tune), a chat model (a chat template, no "base" in its name), an architecture llama.cpp
converts other than a draft model, dense by its config.json, and at most 36B parameters (the 27B
fitted an A40). Downloads are Hugging Face's 30-day counts. Reads only public metadata.
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
NOT_ORIGINAL = ("gguf", "awq", "gptq", "mlx", "fp8", "int4", "int8", "nvfp4", "bnb", "exl", "4bit", "8bit")
MAX_PARAMS = 36e9
RELEASED = ["Qwen/Qwen3.8-27B", "IFM/K2-Horizon-7B", "Qwen/Qwen3.5-9B", "meta-llama/Llama-3.2-3B-Instruct"]
TOKEN = Path.home().joinpath(".cache/huggingface/token").read_text().strip()


def get(params, path="", url=None):
    url = url or API + path + "?" + urllib.parse.urlencode(params, doseq=True)
    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {TOKEN}"})
    for wait in (5, 15, 45, 120):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            if error.code != 429:
                raise
            time.sleep(wait)
    raise RuntimeError(f"rate limited: {url}")


def uncensored(model_id):
    name = model_id.lower()
    return any(word in name for word in UNCENSORED) and "candor" not in name


def dense(model):
    try:
        config = get({}, url=f"https://huggingface.co/{model['id']}/resolve/main/config.json")
    except urllib.error.HTTPError:
        config = model.get("config") or {}
    for part in (config, config.get("text_config") or {}):
        if any("experts" in key and isinstance(value, int) and value > 1 for key, value in part.items()):
            return False
    return True


def editable(model):
    name = model["id"].lower()
    tags = model.get("tags", [])
    config = model.get("config") or {}
    words = name.replace("_", "-").split("-")
    if uncensored(name) or any(word in name for word in NOT_ORIGINAL) or "base" in words:
        return False
    if not (config.get("chat_template_jinja") or (config.get("tokenizer_config") or {}).get("chat_template")):
        return False
    if any(t.startswith(("base_model:quantized:", "base_model:merge:", "base_model:adapter:")) for t in tags):
        return False
    parents = [t.split(":", 2)[2] for t in tags if t.startswith("base_model:finetune:")]
    if any(parent.split("/")[0] != model["id"].split("/")[0] for parent in parents):
        return False
    architectures = config.get("architectures") or []
    params = (model.get("safetensors") or {}).get("total")
    known = bool(architectures) and architectures[0] in ARCHITECTURES and "Draft" not in architectures[0]
    return known and (params or 0) <= MAX_PARAMS


def names(model_id):
    name = model_id.split("/")[1]
    short = name
    for suffix in ("-Instruct", "-instruct", "-it", "-Chat", "-chat"):
        short = short.removesuffix(suffix)
    return {name, short}


def gap(model):
    ggufs = get(
        {"filter": [f"base_model:quantized:{model['id']}", "gguf"], "limit": 1000, "expand[]": ["downloads"]}
    )
    found = {m["id"]: m["downloads"] for m in ggufs if uncensored(m["id"])}
    for name in names(model["id"]):
        for m in get({"search": name, "limit": 1000, "expand[]": ["downloads"]}):
            if name.lower() in m["id"].lower() and uncensored(m["id"]):
                found[m["id"]] = m["downloads"]
    demand = sum(m["downloads"] for m in ggufs if not uncensored(m["id"]))
    return {
        "id": model["id"],
        "created": model.get("createdAt", "")[:10],
        "params_b": round(((model.get("safetensors") or {}).get("total") or 0) / 1e9, 1),
        "architecture": (model.get("config") or {}).get("architectures", ["?"])[0],
        "gguf_repos": len(ggufs),
        "gguf_downloads": demand,
        "uncensored_repos": len(found),
        "uncensored_downloads": sum(found.values()),
        "gap": round(demand / (1 + len(found))),
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
