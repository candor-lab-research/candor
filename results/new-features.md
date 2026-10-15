# New features

Outside work the user wants kept for later: what it is, what it could give candor, and what is
unknown. Nothing here is in the plan until agreed. Ideas that change the method itself are in
`findings.md`.

## Uno: faster generation through diffusion (saved 2026-10-15)

[Unlocking Lossless Speedups in LLMs via Discrete Diffusion](https://arxiv.org/abs/2609.04010)
(Sahoo et al., 3 Sep 2026; IFM, K2-Horizon's makers, among the authors). Code:
[ifm-ai/uno](https://github.com/ifm-ai/uno).

- **What it is:** [Published] A model keeps its usual next-token weights and gains small "diffusion"
  adapter weights, trained by distillation, that propose several tokens per step. Its samplers
  (Ψ-Spec) draw those tokens from the original model's own distribution, so the output is claimed
  to be unchanged ("lossless"), with no separate draft model. Up to 3x the base model's throughput,
  and more than speculative decoding (DFlash, EAGLE-3) at every batch size tested.
- **For K2:** [Published, IFM's own numbers]
  [IFM/K2-Horizon-7B-Uno](https://huggingface.co/IFM/K2-Horizon-7B-Uno) is a 1.4 GB adapter on K2-Horizon-7B (Apache-2.0), averaging 2.71 tokens per forward pass with
  benchmark scores at the base model's level. Also released: K2-Horizon-0.9B-Uno and a Qwen3-8B Uno.
- **Why it matters to candor:** [Hypothesis] candor's cost is mostly thinking tokens, so 2-3 times
  more tokens per step could cut as-used scoring and measuring by about half. And since the answers
  come from the model's own weights, a refusal edit to those weights should carry over, though the
  adapter was trained on the unedited model, so it may propose fewer accepted tokens on edited
  answers (less speed, not different answers).
- **Unknown:** which engines run it (none is named; llama.cpp support would be needed for candor's
  measurements and your Mac); the speed on a Mac; how much an edit lowers the speedup. The full
  evaluation suite is "to be released soon".
- **Related:** geantendormi's "K2-Horizon-7B-Uno-Uncensored" may be built on it (its name, and its
  card mentions merging LoRA weights); unverified.
