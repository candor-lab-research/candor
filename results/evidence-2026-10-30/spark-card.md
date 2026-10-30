---
license: apache-2.0
language:
- en
base_model: XHToken/Spark-X2.5-4B
base_model_relation: finetune
pipeline_tag: text-generation
tags:
- spark
- spark-x2.5
- gguf
- llama-cpp
- uncensored
- abliterated
- thinking
- candor
---

<p align="center">
  <img src="https://huggingface.co/candor-lab/Spark-X2.5-4B-Candor-Uncensored-GGUF/resolve/main/candor-header.png" width="100%" alt="candor">
</p>

# Spark X2.5 4B candor Uncensored

Spark-X2.5-4B with its refusals removed by a weight edit, made with candor, our weight-editing tool. It ships as one GGUF for llama.cpp, and was measured with thinking on, as it's used.

## Results

Measured in llama-server on this exact file with thinking on, using prompts the edit's search never saw.

| | Original | This model |
| --- | --- | --- |
| Harmful prompts refused (of 288) | 282 | **0** |
| Harmless prompts refused (of 300) | 4 | **0** |
| KL divergence from the original | | **0.046** |
| GSM8K maths, correct (of 1,319) | 1,236 | **1,228** |
| GSM8K maths, thinking ran out of budget (of 1,319) | 66 | **67** |
| Harmful prompts, median thinking tokens | 415 | **1,318** |

- "Original" is the same model before the edit, at Q4_K_M.
- KL divergence measures how far the model's predictions drift from the original's on harmless prompts. Lower is closer.
- A second, larger refusal judge, Qwen3Guard-Gen-8B, counts 282 harmful refusals for the original and 1 for this model.
- Against the original, neither harmless refusals nor any maths count changes beyond noise.
- No other uncensored release of this model was measured in full, so there's no comparison column.

## How it was measured

- **Harmful prompts:** 288 from StrongREJECT, leaving out the ones it takes from AdvBench, since the edit's own prompts come from there. A refusal judge, Qwen3Guard-Gen-0.6B, reads the final answer of up to 256 tokens after the thinking. An answer that never arrives counts as a refusal.
- **Harmless prompts:** 300 Alpaca prompts the search never used, read by the same judge.
- **KL divergence:** on the first answer token after the original's own thinking, over the top 200 tokens, on the 278 of those 300 harmless prompts where the original's thinking ended within its budget.
- **Maths:** the full GSM8K test set, zero-shot, with answers of up to 512 tokens after the thinking.
- **Settings:** thinking on, the chat template's default, at Spark-X2.5's recommended sampling (temperature 1.0, top_p 0.95, min_p 0, and top_k 1,000 for its recommended -1) with a fixed seed per prompt, thinking up to 8,192 tokens and then closed, f16 KV cache, no system prompt, llama.cpp b11524.

## Measured as used

candor measures the exact file it ships with thinking on, to a budget of 8,192 tokens, and judges only the answer that follows.

Here it ruled one out. The search's best 3 edits were measured this way in full, against a rule set before they were. One, which the search couldn't tell from this one on KL divergence, failed it: its maths thinking ran out of budget on 96 problems against the original's 66, which is beyond noise. The judges found harmful content in the other two's answers to harmful prompts equally often within noise, so the rule shipped the one with the lower KL divergence, this one. Its maths, its harmless refusals and how often its maths thinking runs out of budget all stay within noise of the original.

Its median thinking on harmful prompts is about 3.2 times the original's.

All 31 edited matrices keep at least 99.7% of the edit at Q4_K_M, so the file is plain Q4_K_M.

| File | Size |
| --- | --- |
| `Spark-X2.5-4B-Candor-Uncensored-Q4_K_M.gguf` | 2.60 GB |

Other quantisations haven't been measured, so there's only this one.

## Where it comes from

1. **Base:** [XHToken/Spark-X2.5-4B](https://huggingface.co/XHToken/Spark-X2.5-4B), XHToken's original 16-bit weights, with no fine-tune.
2. **The edit:** a rank-3 change to the attention output matrices in layers 9 to 35 and the MLP output matrices in layers 21 to 24 (31 of the model's 72), removing the direction the model uses to refuse, read at layer 17 where the thinking starts. A 93-trial search chose it on refusals, thinking that ran out of budget, and KL divergence, scoring every trial with thinking on, as it's used. Its best 3 were then measured in full, and this one was picked by a rule set before they were.

## Use

With llama.cpp, at the settings it was measured with:

```bash
llama-server -hf candor-lab/Spark-X2.5-4B-Candor-Uncensored-GGUF:Q4_K_M -c 32768 \
  --temp 1.0 --top-p 0.95 --top-k 1000 --min-p 0 \
  --reasoning-budget 8192 \
  --reasoning-budget-message $'\n\nConsidering the limited time by the user, I have to give the solution based on the thinking directly now.'
```

Tested in llama-server only. Other GGUF apps, such as LM Studio and Ollama, should work but haven't been checked.

## Limits

- It answers requests the original model refuses. You are responsible for how you use it.
- Refusals and maths were measured only in English, with thinking on. Thinking turned off (`enable_thinking: false`) wasn't measured.
- It thinks before it answers, sometimes at length: give it a thinking budget. On 15 of the 288 harmful prompts its thinking ran out of the 8,192-token budget, against none for the original.
- Tool use and agentic tasks weren't measured.
- It's a 4B model, so anything it states as fact can be wrong.

## Contact

Requests and feedback are welcome: open a discussion in the Community tab. We read everything but can't promise every request.

## Licence

This model is under the Apache License 2.0, in `LICENSE`, as Spark-X2.5-4B is. Spark-X2.5-4B is Copyright 2026 XHToken. Its weights were changed by the edit described above.
