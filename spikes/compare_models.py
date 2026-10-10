"""Score an edited model against its original the way heretic does, under both tokenizations.

    PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.7 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.6 HF_DEACTIVATE_ASYNC_LOAD=1 \\
        caffeinate -i uv run python spikes/compare_models.py ORIGINAL EDITED

Both models answer heretic's test prompts: refusals in the first 100 greedy tokens, and first-token
KL on the harmless ones. Each is measured once with the chat template's tokens alone, which is how
transformers, mlx-lm, Bionic and llama.cpp run a model, and once with the extra begin-of-text token
heretic's tokenizer call adds. Answers go to runs/<name>/private; only numbers are printed.
"""

import gc
import json
import os
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

from candor import kl, reference
from candor.judge import count_refusals
from candor.prompts import HARMFUL_TEST, HARMLESS_TEST, encode

RUNS = Path(__file__).resolve().parent.parent / "runs"
TOKENIZATIONS = {"template only": False, "heretic": True}
MAX_TOKENS = 100
BATCH = int(os.environ.get("BATCH", "32"))
TEMPLATE_KWARGS = {"date_string": os.environ["DATE_STRING"]} if "DATE_STRING" in os.environ else {}


def main() -> None:
    original, edited = Path(sys.argv[1]).expanduser(), Path(sys.argv[2]).expanduser()
    suffix = "".join(f"-{value.replace(' ', '')}" for value in TEMPLATE_KWARGS.values()) + f"-batch{BATCH}"
    run = RUNS / f"compare-{original.name}-vs-{edited.name}{suffix}"
    private = run / "private"
    private.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    device, dtype = reference.device_and_dtype()

    tokenizer = AutoTokenizer.from_pretrained(original)
    sample = [{"role": "user", "content": "Name three primary colours."}]
    same_tokenizer = tokenizer.apply_chat_template(sample) == AutoTokenizer.from_pretrained(
        edited
    ).apply_chat_template(sample)
    print(f"Both folders' tokenizers give the same tokens: {same_tokenizer}", flush=True)
    pad_id = reference.pad_token_id(tokenizer)
    harmful, harmless = HARMFUL_TEST.load(), HARMLESS_TEST.load()
    prompts = {
        name: (
            encode(tokenizer, harmful, add_special_tokens=extra, template_kwargs=TEMPLATE_KWARGS),
            encode(tokenizer, harmless, add_special_tokens=extra, template_kwargs=TEMPLATE_KWARGS),
        )
        for name, extra in TOKENIZATIONS.items()
    }

    logprobs: dict[tuple[str, str], torch.Tensor] = {}
    refusals: dict[tuple[str, str], int] = {}
    for label, folder in (("original", original), ("edited", edited)):
        model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(folder, dtype=dtype)
        model.to(device).eval()  # pyright: ignore[reportArgumentType]
        for name, (harmful_ids, harmless_ids) in prompts.items():
            logprobs[label, name] = reference.first_token_logprobs(model, harmless_ids, pad_id, BATCH)
            answers = reference.greedy(model, tokenizer, harmful_ids, MAX_TOKENS, BATCH)
            (private / f"{label}-{name}.jsonl").write_text("".join(json.dumps(a) + "\n" for a in answers))
            refusals[label, name] = count_refusals(answers)
            print(f"Measured the {label} model, {name} ({time.perf_counter() - started:.0f} s)", flush=True)
        del model
        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()

    results = {}
    for name in TOKENIZATIONS:
        first = prompts[name][0][0]
        results[name] = {
            "begin-of-text tokens": next(i for i, t in enumerate(first) if t != tokenizer.bos_token_id),
            "refusals, original": refusals["original", name],
            "refusals, edited": refusals["edited", name],
            "kl": kl.exact(logprobs["original", name], logprobs["edited", name]),
        }
    (run / "results.json").write_text(json.dumps(results, indent=1))

    conditions = f"{device}, {dtype}, batch {BATCH}" + "".join(
        f", {k} {v}" for k, v in TEMPLATE_KWARGS.items()
    )
    print(f"\n{edited.name} vs {original.name} ({conditions})\n")
    print(f"  {'tokenization':<16}{'BOS':>5}{'refusals before':>17}{'after':>7}{'KL':>9}")
    for name, values in results.items():
        print(
            f"  {name:<16}{values['begin-of-text tokens']:>5}{values['refusals, original']:>17}"
            f"{values['refusals, edited']:>7}{values['kl']:>9.4f}"
        )


if __name__ == "__main__":
    main()
