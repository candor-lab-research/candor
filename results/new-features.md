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
  [IFM/K2-Horizon-7B-Uno](https://huggingface.co/IFM/K2-Horizon-7B-Uno) is a 1.4 GB adapter on
  K2-Horizon-7B (Apache-2.0), averaging 2.71 tokens per forward pass with benchmark scores at the
  base model's level. Also released: K2-Horizon-0.9B-Uno and a Qwen3-8B Uno.
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

## From the arXiv research of 2026-10-15

Four agents searched arXiv for speed, better edits for thinking models, cheaper evaluation and
search, and quantisation, MoE and deployment. Numbers, models and the cheap test for each are in
`evidence-2026-10-15/arxiv-new-features.md`. Ranked by what they would do for K2 and 27B, cheapest
to check first. All are [Published] on other models and a [Hypothesis] for candor until tested.

**Checks on data candor already has (no GPU):**

1. **Size measurements to the comparison, and stop early once it's settled** (2605.30315, SySRs
   2606.07726, CELEUS 2606.20820): best of many models found with 12-20% of the prompt-model pairs.
   Would have warned that 100 prompts couldn't separate the finalists. Replay steps 6-7's verdicts.
2. **Score the search on the prompts that tell edits apart** (item response theory: 2606.20626,
   2608.05086): the full ranking kept (Spearman above 0.90) at 80% or more less cost. Replay.
3. **Start the longest prompts first** (Seer 2511.14617): 4-10% more throughput, tail cut 89%. Aims
   at step 7's idle slots (busy 77-88%). Works with llama-server today; simulate from step 7's
   lengths.
4. **Check the judge** (2608.03201, 2606.25487): refusal wording makes guard models, Qwen3Guard
   named, call a harmful answer safe. Re-judge stored answers with a larger Qwen3Guard, counts only.
5. **Planned thinking length is stored at the think start** (2506.08390): if candor's direction
   there carries part of it, that would explain longer thinking. Compare on stored residuals.

**Small changes:**

6. Judge the answer's first tokens only, never cutting the thinking (2506.09996; 2510.06036 warns
   against judging partway through the thinking).
7. Identical GGUF files on the Mac and pods (2610.19498: one compiler flag made Q4_K_M
   bit-identical).
8. Detect weights that were quantised before and keep them at Q8_0 (2609.04526).
9. A small logit penalty on "keep thinking" tokens as a `serve` option (2606.00206, 2609.39801:
   thinking 5-51% shorter, accuracy -0.2 to +9.8 points).
10. Separate strengths for Qwen3.5's linear-attention and full-attention writers (2609.04721).
11. Choose the layer from activations without generating (COSMIC 2506.00085, 2601.19375), which
    could shorten the search.

**Bigger:**

12. Speculative decoding drafted from earlier trials' answers to the same prompt (Seer, DAS
    2511.13841, 2509.04474: about 1.5-2 times). First check that llama-server rolls back Qwen3.5's
    recurrent layers and that batched outputs stay exact (2510.22876).
13. DFlash (2602.06036): 4.0 times on Qwen3.5-9B one at a time, 1.9 at 32; vLLM, not llama.cpp.
14. A "stop thinking" direction built from the edited model's own answers (2610.19934).
15. A judge-weighted ridge direction (2512.16602), the one refusal study on a Qwen3.5-style hybrid.
16. Cheap labels on every prompt corrected by as-used labels on a few (2601.05420, 2607.08347).
17. Probes that read the verdict from activations (2605.28553, 2507.12428), with a circularity risk.
18. Spending as-used scoring on the leaders adaptively (2609.25645).
19. 27B at IQ3 on an 18 GB Mac, scored on that file (2608.06564: iq3_xxs flipped 6% of refusal
    decisions against 31% for plain rounding; 2606.25519: thinking 19-60% longer at 3 bits).
20. Editing mixture-of-experts models: every writer, or silencing refusal experts (2609.09793,
    2602.08741, 2610.02910).
