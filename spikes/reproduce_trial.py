"""Rebuild a heretic trial with candor's own code and measure it the way heretic does.

    PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.7 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.6 HF_DEACTIVATE_ASYNC_LOAD=1 \\
        caffeinate -i uv run python spikes/reproduce_trial.py MODEL_FOLDER JOURNAL TRIAL

The refusal directions are found again from the model, the trial's edit is applied to the weights
in memory, and both models are scored on heretic's test prompts: refusals in the first 100 greedy
tokens, and first-token KL. Answers go to runs/<name>/private; only numbers are printed.

HERETIC_TOKENIZATION=1 tokenizes the way heretic does, with the tokenizer's own special tokens added
in front of the ones the chat template writes.
"""

import json
import os
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from candor import kl, reference
from candor.directions import after_layer, difference_of_means
from candor.edit import rank_one
from candor.heretic import Journal, Trial, read_journal
from candor.judge import count_refusals
from candor.prompts import PromptSet, encode
from candor.residuals import last_token_residuals
from candor.weights import DENSE

RUNS = Path(__file__).resolve().parent.parent / "runs"


def device_and_dtype() -> tuple[str, torch.dtype]:
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    if torch.backends.mps.is_available():
        return "mps", torch.float16
    return "cpu", torch.float32


def load_prompts(tokenizer: PreTrainedTokenizerBase, settings: dict, name: str, heretic_tokens: bool):
    spec = settings[name]
    if spec["prefix"] or spec["suffix"]:
        raise SystemExit(f"{name} adds a prefix or suffix to each prompt, which isn't supported yet")
    prompts = PromptSet(spec["dataset"], spec["split"], spec["column"]).load()
    system = spec["system_prompt"] or settings["system_prompt"]
    return encode(tokenizer, prompts, system, settings["response_prefix"] or "", heretic_tokens)


def apply_trial(model: PreTrainedModel, directions: torch.Tensor, journal: Journal, trial: Trial) -> None:
    layers = len(directions) - 1
    with torch.no_grad():
        for layer in range(layers):
            if trial.direction_layer is None:
                direction = directions[layer + 1]
            else:
                direction = after_layer(directions, trial.direction_layer)
            for target in DENSE:
                strength = trial.kernels[target.component].at(layer)
                if strength:
                    weight = model.get_parameter(target.names(layer)[0])
                    b, a = rank_one(weight, direction, journal.row_norm)
                    weight.add_((strength * b @ a).to(weight.dtype))


def main() -> None:
    model_folder, journal_path, number = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
    heretic_tokens = os.environ.get("HERETIC_TOKENIZATION") == "1"
    journal = read_journal(journal_path)
    trial = journal.trials[number]
    settings = journal.settings
    if settings["winsorization_quantile"] < 1:
        raise SystemExit("this trial used winsorization, which isn't supported yet")

    run = RUNS / f"reproduce-{model_folder.name}-trial-{number}{'-heretic-tokens' if heretic_tokens else ''}"
    private = run / "private"
    private.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    def elapsed() -> str:
        return f"{time.perf_counter() - started:.0f} s"

    device, dtype = device_and_dtype()
    tokenizer = AutoTokenizer.from_pretrained(model_folder)
    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(model_folder, dtype=dtype)
    model.to(device).eval()  # pyright: ignore[reportArgumentType]
    pad_id = reference.pad_token_id(tokenizer)
    print(f"Loaded on {device} in {dtype} ({elapsed()})", flush=True)

    harmful_train = load_prompts(tokenizer, settings, "bad_prompts", heretic_tokens)
    harmless_train = load_prompts(tokenizer, settings, "good_prompts", heretic_tokens)
    harmful_test = load_prompts(tokenizer, settings, "bad_evaluation_prompts", heretic_tokens)
    harmless_test = load_prompts(tokenizer, settings, "good_evaluation_prompts", heretic_tokens)
    leading = next(i for i, token in enumerate(harmful_test[0]) if token != tokenizer.bos_token_id)
    print(f"Begin-of-text tokens in front of each prompt: {leading}", flush=True)

    directions = difference_of_means(
        last_token_residuals(model, harmful_train, pad_id),
        last_token_residuals(model, harmless_train, pad_id),
        orthogonalize=settings["orthogonalize_direction"],
    )
    print(f"Found the directions ({elapsed()})", flush=True)

    max_tokens = settings["max_response_length"]

    def measure(name: str) -> tuple[torch.Tensor, int]:
        logprobs = reference.first_token_logprobs(model, harmless_test, pad_id)
        answers = reference.greedy(model, tokenizer, harmful_test, max_tokens)
        (private / f"{name}.jsonl").write_text("".join(json.dumps(answer) + "\n" for answer in answers))
        print(f"Measured the {name} model ({elapsed()})", flush=True)
        return logprobs, count_refusals(answers)

    base_logprobs, base_refusals = measure("original")
    apply_trial(model, directions, journal, trial)
    edited_logprobs, edited_refusals = measure("edited")

    results = {
        "heretic": {
            "refusals, original": trial.base_refusals,
            "refusals, edited": trial.refusals,
            "kl": trial.kl_divergence,
        },
        "candor": {
            "refusals, original": base_refusals,
            "refusals, edited": edited_refusals,
            "kl": kl.exact(base_logprobs, edited_logprobs),
        },
        "heretic tokenization": heretic_tokens,
        "device": device,
        "dtype": str(dtype),
    }
    (run / "results.json").write_text(json.dumps(results, indent=1))

    print(f"\nTrial {number} of {journal_path.name}, rebuilt by candor\n")
    print(f"  {'':<22}{'heretic':>10}{'candor':>10}")
    for key in ("refusals, original", "refusals, edited"):
        print(f"  {key:<22}{results['heretic'][key]:>10}{results['candor'][key]:>10}")
    print(f"  {'KL':<22}{results['heretic']['kl']:>10.4f}{results['candor']['kl']:>10.4f}")


if __name__ == "__main__":
    main()
