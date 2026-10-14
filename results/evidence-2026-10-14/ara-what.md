# ARA in heretic: what it is, from source

Research date 2026-10-14. Heretic master read at `66a952b` (2026-10-12), cloned to `scratchpad/ara/src/heretic`. PR branches fetched for reading only: #211 head `edc3b12` (the old `ara` branch), #332 `bf41e81`, #446 `15ad426`, #196 `444edec`. Nothing was installed or run. GitHub threads were read through a filter that hid code blocks, quotes, `<details>` and images (`scratchpad/ara/tools/strip.py`). Model cards, datasets and result images were not opened.

Every claim below names its source. Where I am reasoning from the code rather than quoting a source, I mark it **[inference]**.

---

## Plain-language explanation

Classic abliteration finds one "refusal direction": the average difference between the model's internal state on harmful prompts and on harmless ones. It then edits the weights so the model can no longer write along that direction. Heretic 1.x searches how strongly to do this, layer by layer.

ARA ("Arbitrary-Rank Ablation") drops the idea of a direction. For each weight matrix that writes into the residual stream (the attention output projection and the MLP down projection), it records what goes into and out of that matrix on 400 harmless and 400 harmful prompts, at the last token before the answer starts. Then it fits a small correction to the matrix with an ordinary numerical optimiser (L-BFGS) so that three things hold:

1. On harmless prompts, the matrix's output stays the same.
2. On harmful prompts, the output moves close to *some* harmless output (each harmful output is pulled towards its own nearest harmless neighbours, not towards the average).
3. On harmful prompts, the output also moves *away* from where it was ("overcorrection"). The author says this is what beats harder refusal mechanisms.

The correction is stored as a rank-50 LoRA adapter, which is just a fixed low-rank change to the weights. It can be merged into the model, so nothing extra is needed when the model runs. Heretic's outer search (Optuna TPE, as before) then tunes six numbers per trial: which layers to edit, how to weight the three goals, and how many neighbours count. It scores each trial on refusals and KL divergence, as in 1.x.

ARA was invented by heretic's maintainer, Philipp Emanuel Weidmann (p-e-w), in early 2026. It lived in an unmerged PR for six months and became heretic master's default on 2026-10-01. There is no paper or article yet. No tagged release ships it: heretic 1.4.0 does not have it. The maintainer says it must still "prove itself" before it stays the default in 2.0.

---

## 1. The maths

### Name
"Arbitrary-Rank Ablation (ARA) (Weidmann 2026)": `src/heretic/modifiers/ara.py:4-5`, which points to PR #211 for details. The display name is "Arbitrary-Rank Ablation (ARA)" (`ara.py:172-173`).

### What is captured, once per model
For every layer and every targeted module, `Model.get_module_io` puts forward hooks on the plain `nn.Linear` modules. One generation step runs (`max_new_tokens=1`), and the hooks keep the module's **input and output at the last prompt position** (`model.py:653-718`, slicing at `:691-692`). Tensors are kept on the CPU in the model dtype. `ARA.init` does this for the good and the bad prompt sets (`ara.py:196-201`) before the LoRA is attached (`:205`). The prompt is the chat template plus heretic's response prefix (`model.py:489-494`; see section 3.4 on thinking models).

For one module with weight `W` (d_out x d_in), that gives:
- `X_g` (n_g x d_in), `Y_g` (n_g x d_out): good-prompt inputs and outputs
- `X_b`, `Y_b`: the same for bad prompts

The defaults are n_g = n_b = 400, from `mlabonne/harmless_alpaca train[:400]` and `mlabonne/harmful_behaviors train[:400]` (`ara.py:54-70`, `config.default.toml:249-258`).

### Which matrices
These are the same "abliterable" modules as heretic's abliteration (`model.py:383-451`):
- `attn.o_proj`: `self_attn.o_proj`, and on Qwen3.5 hybrid layers also `linear_attn.out_proj`, the GatedDeltaNet output (`:400-407`)
- `mlp.down_proj`: dense `mlp.down_proj`, plus per-expert modules only when experts can be iterated as modules (`:409-445`)

Every module in every layer `l` with `start_layer_index <= l < end_layer_index` is optimised (`ara.py:263-268`). All components share one parameter set. The PR #211 TODO "Decide whether to have separate abliteration parameter sets for each component" was never ticked (PR #211 body).

### The correction and its parametrisation
`W' = W + B A`, with `B` (d_out x r) and `A` (r x d_in), and r = `lora_rank` = 50 (`ara.py:81-89`). PEFT is set up with `lora_alpha = r`, so the scaling is 1 ("Apply adapter at full strength", `model.py:217-226`). In each trial, `B` starts at 0 and `A` starts from a seeded Kaiming-uniform draw (`ara.py:258-305`, added by #476).

Inside the loss, with `preserve_row_magnitudes = true` (the default, `ara.py:72-79`), the effective weight is row-renormalised to the original row norms (`ara.py:320-327`):

```
W~ = diag(||w_i||) * rownormalize(W + B A)
Yhat_g = X_g W~^T        Yhat_b = X_b W~^T        (ara.py:330-331)
```

### The loss (`ara_loss`, `ara.py:126-161`; `mean_distances_to_knn`, `:119-122`)

```
knn_k(y; S) = mean of the k smallest Euclidean distances from y to the rows of S

L = alpha * mean((Yhat_g - Y_g)^2)
  + beta  * [ (1/n_b) sum_i knn_k(yhat_b,i ; Y_g)  -  rho * (1/n_b) sum_i knn_k(yhat_b,i ; Y_b) ]
```

| symbol | parameter | search range (`ara.py:207-253`) |
|---|---|---|
| alpha | `preserve_good_behavior_weight` | uniform [0, 1] |
| beta | `steer_bad_behavior_weight` | log-uniform [1e-4, 1] |
| rho | `overcorrect_relative_weight` | uniform [0, 1.3] |
| k | `neighbor_count` | int [1, min(15, n_g, n_b)] |
| s | `start_layer_index` | int [0, L//2] |
| e | `end_layer_index` (exclusive) | int [L//2 + 1, L] |

Each module is minimised separately with `torch.optim.LBFGS` over `[A, B]`: lr 1.0, `max_iter` 20, `history_size` 10, strong-Wolfe line search, run for `n_optimization_steps` = 5 outer steps (`ara.py:341-356`). PyTorch defaults `max_eval = max_iter*5//4 = 25` per step (PyTorch v2.9.1 `torch/optim/lbfgs.py:232-233`), so each module gets at most about 125 loss-and-gradient evaluations. Heretic turns gradients off globally (`main.py:292`). This still works because `LBFGS.step` wraps the closure in `torch.enable_grad()` (`lbfgs.py:301-312`).

### What the author says the terms do
From the PR #211 body (2026-03-04):
- (1) keeps harmless outputs unchanged
- (2) makes harmful outputs "as similar to those associated with harmless prompts as possible"
- (3) makes them "as dissimilar to those previously associated with harmful prompts as possible", which "overcorrects away from the original residuals, which results in stronger steering that can overcome more complex refusal mechanisms"

On the kNN choice (p-e-w, #211 comment 2026-03-04T16:55): difference-of-means "optimizes towards a configuration where the mean of the modified harmful outputs resembles the mean of the harmless outputs. This is an unnecessary constraint ... With ARA, every single harmful output is simply attracted towards somewhere in the harmless cluster."

### Rank, delta form, deployment
- **The shipped change is a fixed weight delta `dW = B A` of rank at most 50** on each edited `o_proj`/`out_proj`/`down_proj` matrix. Modules outside [s, e) keep B = A = 0 (`model.py:236-239`, `:315-338`).
- Export is either the PEFT adapter (`main.py:943-949`) or merged weights via `merge_and_unload` (`main.py:950-965`, `model.py:266-313`). With bnb 4-bit, the full-precision base is reloaded on the CPU and the adapter merged into it (`model.py:271-305`).
- **Nothing is needed at inference time**: no hooks and no runtime steering. A merged ARA model is an ordinary checkpoint, so it can go to GGUF exactly like heretic's current rank-3 "full" edit **[inference from the code]**. I did not check whether llama.cpp's LoRA tooling accepts this PEFT adapter directly.
- Why rank 50: "singular values tend to drop rapidly after a few dozen dimensions" (`ara.py:84-88`). p-e-w saw "effective ranks around 10-30 typically, with the corresponding singular values explaining 99+% of the variance" (#332 comment 2026-05-26). For scale, heretic's 1.4.0-style "full" abliteration uses rank 3 (`config.default.toml:276`).
- The one measured edit size I found is for the older full-rank, row-norm-preserving ARA (gpt-oss-20b v4, attention `o_proj` only, layers 2-23). There, the deltas had "relative Frobenius norm between 0.0060 and 0.0303, per-row norms within 0.8%" (hpnyaggerman, #211 comment 2026-07-27).

### A discrepancy that is still in master: row norms
In the current (LoRA) ARA, the loss is computed on the **row-renormalised** `W~` (`ara.py:324-327`). The model that is scored and exported, however, uses the **un-normalised** `W + B A`. Nothing writes the renormalisation back: `modify_model` ends after L-BFGS (`ara.py:355-366`), and PEFT adds `B A` at scale 1.

- The older full-rank ARA did write it back: `matrix.copy_(get_matrix())`, PR #211 head `edc3b12`, `src/heretic/model.py:681-682`.
- The LoRA port did not: `ara_lora_abliterate`, same file `:684-798`, normalisation only inside the objective at `:744`. Master's `ara.py` comes from that port ("the LoRA version of ARA ... is the one that made it into this pull request (with minor changes)", p-e-w, #446 comment 2026-10-01).
- red40maxxer reported exactly this ("evals a row-normalized effective weight for the objective, but the deployed model computes the un-normalized one", #211 comment 2026-07-20).
- p-e-w acknowledged it. He described a possible fix (merge, reparametrise, subtract, then truncate by SVD) as "very ugly, there might be a better solution" (#211 comment 2026-07-22). No later commit addresses it.

The heretic scores are still honest, because TPE scores the deployed `W + B A` model. It is the inner fit that targets a different matrix from the one shipped **[inference]**.

### My own observations on the maths **[inference, not from any source]**
- **Locality.** Every module is fitted against inputs captured from the *unedited* model. When layers below it are also edited in the same trial, its real inputs change, and ARA does not model this compounding. The outer TPE search is what absorbs it.
- **Position.** The fit uses only the last prompt position. The weight edit, though, applies at every position during generation. Abliteration's residual means are taken at the same position.
- **Overcorrection above 1 is unbounded without row norms.** For a far-away `yhat`, the distances to `Y_g` and to `Y_b` grow at the same rate, so the steering term behaves like `(1 - rho)*||yhat||`. That is unbounded below for rho > 1. The good-preservation term only constrains `W` on the span of 400 good inputs, which leaves most of the input space free. This fits p-e-w's description that the region rho > 1 "is necessarily unstable" and "chaotic (possibly even quasi-fractal)" (#211 comments 2026-03-11). The renormalised `W~` caps every row's norm *inside the loss*. Combined with the discrepancy above, this means L-BFGS could grow `B A` without limit while the loss stays bounded, and the shipped `W + B A` would carry that growth. That is a plausible extra cause of runaway trials, but I have not verified it.
- **Scale.** Only the ratio alpha/beta changes the minimiser. The L-BFGS stopping tolerances are absolute, though, so the overall scale is not fully irrelevant.
- **Convexity.** The PR body calls the objective "affine-convex". The `-rho * knn` term is the negative of a convex function, and k-nearest selection is a minimum over subsets, so the objective is not convex in general. p-e-w himself attributes instability to "small gradient discontinuities because of the top-k neighbor selection" (#211 comment 2026-03-15; also #332 comment 2026-05-26).

---

## 2. Origin and authorship

- **Author:** Philipp Emanuel Weidmann (p-e-w), heretic's maintainer. "I invented ARA from scratch just a few months ago, haven't described it in detail yet" (#332 comment 2026-05-26). The PR #211 body (2026-03-04) says he had been developing it "for the past two months or so". The first ARA commit is `ea7c59a` "feat(ara): add methods for obtaining module I/O", 2026-02-25, by Weidmann.
- **Contributors:** kabachuha wrote the LoRA variant (#332). joninco and coder3101 (Ashar) are credited as co-authors in the merge commit `662e4ba`.
- **No paper, article or blog post exists.** The PR #211 TODO "Write a proper article explaining the mathematics and motivation behind ARA" is unticked. On 2026-10-03 p-e-w wrote: "ARA will get its own article at some point; it's too complex to fit into the README" (#473). The README's "How Heretic works" still describes directional abliteration (#473 opening post; `README.md` has no mention of ARA). kabachuha found "absolutely no references" to ARA in the literature (#332 comment 2026-05-26).
- **What it claims to fix about abliteration**, all from p-e-w:
  - It does not assume a refusal direction or rank: it "doesn't use refusal directions at all ... doesn't assume a particular rank for the refusal manifold, or that the centroid of the outputs must shift in a specific manner" (PR #211 body). Also: "doesn't assume anything about the transformation that distinguishes the harmful and harmless manifolds, unlike virtually every method proposed previously" (#288 comment 2026-04-08).
  - Difference-of-means needlessly forces the means to line up (quoted above).
  - Overcorrection handles "more complex refusal mechanisms". On implicit refusals that redirect the topic: "The overcorrection mechanism indeed appears to substantially improve this issue, though ARA has not yet been tested with sufficiently many models" (#211 comment 2026-03-08).
  - It needs fewer trials: "ARA has fewer optimizable parameters than our current approach (meaning that fewer trials are needed for good results)" (PR #211 body). Abliteration in master searches 10 parameters (1 categorical, 1 direction index, and 4 per component times 2; `abliteration.py:227-306`), against ARA's 6.
- **Two-tier structure** (p-e-w, #288 comment 2026-04-08): "ARA itself optimizes for a certain residual vector alignment using gradient descent ..., while the global TPE-based optimizer optimizes for refusals using ARA hyperparameters. That optimizer is not part of ARA."

---

## 3. Implementation in heretic master

### 3.1 Code path
1. `main.py:478-572`: the response prefix is detected first (thinking handling, 3.4).
2. `main.py:596-603`: `load_and_init_modifiers` (`modifier.py:108-185`) loads exactly one modifier and calls `ARA.init` (`ara.py:175-205`). This loads the prompts, captures module I/O for good and bad prompts, and attaches a rank-50 LoRA to every targeted module in all layers (`model.py:192-234`). The LoRA A and B are zeroed (`model.py:232-239`, from #476).
3. Each trial (`main.py:613-655`):
   - `suggest_parameters` (`ara.py:207-253`)
   - `reset_model`, which zeroes A and B on the fast path (`ara.py:368-372`, `model.py:315-338`)
   - `modify_model`, the per-module L-BFGS (`ara.py:255-366`)
   - `evaluator.get_scores()`, which with the defaults is keyword refusals on `harmful_behaviors test[:100]` plus first-token KL on `harmless_alpaca test[:100]` (`config.default.toml:77-80, 196-207`)
4. Sampler: `TPESampler(n_startup_trials, n_ei_candidates=128, multivariate=True, seed)` (`main.py:670-676`), unchanged from 1.x.
5. A chosen trial is restored by re-running `modify_model` with the stored parameters (`main.py:863-873`). Reproducibility relies on the seeded A init (#476) and the fixed-date templates (#478).

### 3.2 Defaults (`config.default.toml`, `ara.py` Settings)

| setting | default | searched? |
|---|---|---|
| `modifiers` | `heretic.modifiers.ara.ARA` (`config.default.toml:86-88`) | - |
| `n_trials` / `n_startup_trials` | 100 / 30 (`:91, :94`); were 200 / 60 in v1.4.0 | - |
| `preserve_row_magnitudes` | true (`:225`) | no |
| `lora_rank` | 50 (`:231`) | no |
| `n_optimization_steps`, `learning_rate`, `max_iter`, `history_size` | 5, 1.0, 20, 10 (`:234-243`) | no |
| good / bad prompts | 400 + 400 train prompts (`:249-258`) | no |
| the 6 parameters in section 1 | see table | yes, TPE |

The trial reduction is commit `99be5a6` "feat: reduce default number of trials" (2026-10-01). Its message gives no reasoning; the only stated rationale is the "fewer parameters" claim above.

Range history:
- c76416f (2026-03-10) "expand parameter ranges" raised the rho maximum from 1.0 to 1.3 and lowered the beta minimum by a decade, following joninco's Qwen3.5-35B-A3B data (#211 comments 2026-03-08 to 2026-03-10).
- Raising rho to 1.3 took gpt-oss from "a best result of 30 refusals to 0 refusals" (p-e-w, #196 comment 2026-03-09).
- #478 (2026-10-06) moved the end-index range to start at L//2+1, so the layer range can never be empty, and capped k at the prompt count.

### 3.3 Compute and memory
**Once per model.** ARA runs the same one-token forward passes over 800 prompts that abliteration uses for its residual means, but it keeps per-module I/O instead of means. Storage is n x (d_in + d_out) per module, in bf16, on the CPU **[my arithmetic from configs]**:
- Qwen3.5-9B (hidden 4096, intermediate 12288, attention out input 4096, 32 layers; HF `config.json`): 24,576 values per prompt per layer, so about 1.57 MB per prompt and **about 1.26 GB of RAM** for 800 prompts.
- Qwen3.5-27B (hidden 5120, intermediate 17408, attention out input 6144, 64 layers): about 4.33 MB per prompt, **about 3.5 GB of RAM**.

**Per trial, ARA versus abliteration.**
- Abliteration builds a rank-1 delta per module, plus one randomised `svd_lowrank` for the "full" mode (`abliteration.py:440`).
- ARA, per edited module and per L-BFGS evaluation, forms `W + B A` at full size in fp32. It runs the two output matmuls over 800 prompts, two `cdist` + `topk`, and the backward pass. By my count that is about `d_in * d_out * (4n + 6r)` FLOP, or about 3,500 x (matrix size) with n = 800 and r = 50. At most about 125 evaluations per module.

**Unmeasured upper-bound estimates** (all 125 evaluations, fp32, no TF32 since heretic does not enable it):

| model | per layer | full range | typical random-phase range |
|---|---|---|---|
| Qwen3.5-9B | about 29 TFLOP | about 0.9 PFLOP (32 layers) | about 16 layers |
| Qwen3.5-27B | about 53 TFLOP | about 3.4 PFLOP (64 layers) | about 32 layers |

On a GPU that sustains about 50 TFLOPS in fp32, that is up to tens of seconds per trial for the ARA step. Early L-BFGS convergence would lower it; memory-bound operations and Python overhead would raise it. **Time 2-3 real trials before quoting anything.**

p-e-w's own claim, made for the full-rank version: "The abliteration process takes longer, but the time per trial is still dominated by counting refusals" (PR #211 body).

**VRAM.** Matrices are processed one at a time. Each module transiently needs a few fp32 copies of its matrix (W_base, W + BA, normalised, gradient). For Qwen3.5-9B's down_proj that is about 0.2 GB per copy, for 27B about 0.36 GB **[estimate]**. Gradient buffers are freed after each module (`ara.py:362-366`). The adapter has about 39M parameters for 9B and about 108M for 27B at rank 50 across all layers **[my arithmetic]**, about 17x heretic's rank-3 adapter.

**Quantisation.** ARA works with bnb 4-bit loading because it dequantises one matrix at a time (`ara.py:273-283`). Being able to run 4-bit was a main motivation for the LoRA port: "Gemma 4 31b it at only ~36 GB VRAM" (#332 body).

**Floating point.** L-BFGS "amplifies small floating-point differences", so outputs differ between CPUs (commit `8ae8526` message).

### 3.4 Thinking models
ARA has no thinking-specific code. It inherits heretic's response prefix:
- `main.py:478-572` decides `settings.response_prefix` **before** the modifier's `init`.
- `generate` appends the prefix to every prompt (`model.py:489-494`).
- So module I/O, KL and refusal scoring all happen at the position after the prefix.

The default `chain_of_thought_skips` pairs map `<think>` to `<think></think>`, plus pairs for gpt-oss, `<thought>` and `[THINK]` (`config.default.toml:48-69`). Since #423 (2026-09-05, after 1.4.0), if the rendered generation prompt ends in `<think>` plus whitespace, the prefix is set to the closed block. Heretic then regenerates and appends any further common prefix (`main.py:491-572`).

**[inference, not run]** Qwen3.5-9B's template ends the generation prompt with `<think>\n` when thinking is on (HF `chat_template.jinja:147-153`). Heretic renders without `enable_thinking`, so by my reading the prompt ARA fits on ends `...assistant\n<think>\n<think></think>` plus any common continuation. That is neither the template's own thinking-off form (`<think>\n\n</think>\n\n`) nor thinking-on. This should be checked by rendering it before any comparison.

Joninco reported, for Qwen3.5-35B-A3B: "Markers are unreliable on thinking models - the `<think></think>` prefix skip changes model behavior". Marker counts of 3-5/100 became 33+/100 under an LLM judge (#211 comment 2026-03-08).

### 3.5 Model-specific handling
- **Qwen3.5 hybrid layers:** `linear_attn.out_proj` counts as `attn.o_proj` (`model.py:404-407`).
- **Fused-expert MoE** (Qwen3.5 MoE with transformers 5.x): experts are a single `nn.Parameter`, so they are silently skipped and only attention is edited. "ARA will have this problem as well" (p-e-w, #339 comment 2026-05-31). The shared-expert path joninco patched locally is not in master (`get_layer_modules` has no `shared_expert`).
- **Gemma 4** (`Gemma4ClippableLinear`, #278) and gpt-oss work. Note that the gpt-oss "corrupted ARA model" was a transformers 5.4-5.5.1 MXFP4 save bug, not ARA (hpnyaggerman, #211 comment 2026-07-27, accepted by p-e-w 2026-07-28).

---

## 4. SOMA, briefly
- **What it is:** "Multi-directional refusal suppression with Self-Organizing Maps", kabachuha, PR #196 (opened 2026-02-26, closed 2026-07-07 unmerged, "in favor of the plugin system"). It ports Piras et al., "SOM Directions are Better than One: Multi-Directional Refusal Suppression in Language Models", arXiv 2511.08379 (Nov 2025, AAAI 2026), as suggested in #140.
- **How it works:** a MiniSom grid (default 4x4, 10,000 iterations) is trained on harmful-prompt residuals. The top-k most frequent winning neurons (default 4) give k directions (`pr196:src/heretic/som.py:30-85`). Each direction is applied as its own rank-1 heretic ablation, the deltas are summed, and each has its own weights. Users combined it with MPOA.
- **Relation to ARA:** SOMA is still direction-based abliteration, just with several directions. ARA uses no directions at all (PR #211 body). The maintainer treats them as peer modifiers ("Examples: Standard abliteration, ARA, SOMA", `modifier.py:41`). SOMA should be re-implemented as a modifier plugin, and p-e-w "would be happy to include it as a default plugin" (#332 comment 2026-05-28; #196 comment 2026-10-01). It is not on master yet.
- **On Qwen3.5-4B:** a user reported ARA 4/100 at KLD 0.1396 against SOMA 2/100 at KLD 0.1301. p-e-w called these "all indistinguishable from a practical standpoint", noting that keyword refusal counts carry "a '±10' after it, at minimum" (#196 comments 2026-03-11 and 2026-03-12).

---

## 5. Timeline

| date | event | source |
|---|---|---|
| ~Jan 2026 | development starts ("past two months or so") | PR #211 body |
| 2026-02-25 | first ARA commit `ea7c59a` (module I/O hooks) | git |
| 2026-02-27 | `154241f` matrix optimisation | git |
| 2026-03-04 | PR #211 "Arbitrary-Rank Ablation (ARA)" opened, demo `p-e-w/gpt-oss-20b-heretic-ara-v3` | PR #211 |
| 2026-03-10 | `c76416f` parameter ranges expanded | git, #211 |
| 2026-03-31 | `f7a456b` row-norm preservation (full rank, written back) | git |
| 2026-05-25 | PR #332 "ARA, but it's LoRA" (kabachuha) | #332 |
| 2026-06-14 | **v1.4.0 released, no ARA** (no ARA code in any tag v1.0.1-v1.4.0) | `gh release list`, `git grep` |
| 2026-06-18 | p-e-w: NaN/inf matrices are "a blocker for merging ARA" | #211 |
| 2026-07-05 | #332 merged into the `ara` branch (`25979ad`) | #332 |
| 2026-07-20/22 | row-norm objective/deployment mismatch reported and acknowledged | #211 |
| 2026-09-04 | PR #446 "modifier plugins + ARA" opened; #211 closed as superseded | #446, #211 |
| 2026-09-28 | `ce8b05c` ARA as a modifier plugin | git |
| 2026-10-01 | **#446 merged (`662e4ba`), ARA is the default**, trials 200/60 to 100/30 (`99be5a6`), first reproduced ARA model | #446, git |
| 2026-10-04 | #476: A re-initialised per trial (before this, trials inherited the previous trial's A, so restoring a trial did not rebuild the scored model) | git |
| 2026-10-06 | #478: empty layer range and k-cap fixes | git |
| now | master `66a952b`, version `2.0.0.dev0`; no release contains ARA | `pyproject.toml`, `git tag --contains` |

**Why it became the default.** #446 states no explicit reasoning beyond "Note that ARA is the default now" (2026-09-30). On the merge (2026-10-01): "ARA is the default on master now, though it will have to prove itself in manual testing for us to retain it as the default in the upcoming 2.0 release". The reasoning lives in #211: "it can replace all currently implemented methods in Heretic, including MPOA", plus community reports. In #404 (Qwen3.5-9B at 70/100 refusals after 200 trials with 1.x defaults), p-e-w answered only "ARA and SOMA give better results".

### Reported numbers relevant to Qwen3.5
All are community-reported and keyword-scored unless stated; none is a controlled comparison.
- **Qwen3.5-4B:** main branch best trials above 50/100. ARA 4/100 at KLD 0.1396, later 0/100 at 0.398 and 1/100 at 0.134 (#211 2026-03-06; #196 2026-03-11).
- **Qwen3.5-27B:** with the pre-c76416f ranges, "maybe 55/100 at best". Expanded ranges were "necessary" (erm14254, #211 2026-03-11; no post-expansion number given).
- **Qwen3.5-35B-A3B** (LLM judge, old ranges, 400 trials): best 34/100 at KLD 0.685; 43/100 at 0.026 (joninco, #211 2026-03-08).
- **Qwen3.5-9B:** no ARA number in the threads I read.

---

## 6. What this means for candor **[inference]**
- **Shipping:** an ARA edit ships exactly like candor's current one. It is a fixed low-rank delta on the same `o_proj`/`out_proj`/`down_proj` tensors, mergeable into the Q8_0 GGUF, with a rank-50 adapter as the fallback. Nothing is needed at inference time.
- **Reproducing ARA in candor needs new machinery:**
  - per-module I/O capture at the scored position
  - autograd L-BFGS per matrix, per trial
  - a 6-parameter search space instead of 10
  - a decision on the row-norm question: mirror master (renormalise in the loss only, ship `W + B A`), or follow the full-rank branch (ship the renormalised matrix, which is no longer low-rank and would need an SVD truncation like heretic's "full" mode)
- **Thinking models:** ARA fits at heretic's response-prefix position, so it inherits the thinking-off (or odd duplicated-tag) setup. It does not "measure as used".
- **Evidence:** the evidence that ARA beats abliteration on Qwen3.5 is anecdotal, small-model, and mostly keyword-scored, and the maintainer considers keyword differences of ±10 noise.

---

## 7. Open questions
1. Was the NaN/inf matrix problem (#211, 2026-06-18) ever solved for the LoRA version? The gpt-oss corruption was explained as a transformers bug, but p-e-w said "at least two users" reported NaNs. I found no statement that it is fixed and no NaN guard in `ara.py`.
2. Is the row-norm mismatch intended in master, and how large is the gap between the fitted `W~` and the shipped `W + B A` in practice? Does it contribute to the KLD > 10 trials?
3. Do the KLD > 10 runaway trials (rho > 1) persist in the LoRA version at the master ranges? No post-merge data was found.
4. What exact prefix does heretic append for Qwen3.5 thinking templates? My reading gives a duplicated `<think>`; render it to confirm.
5. MoE experts invoked as modules with 2D (tokens x dim) inputs would hit `inputs[0][:, -1, :]` in the hook (`model.py:691`). joninco hit this crash (#211 2026-03-08). Is it still reachable on any architecture in master?
6. Will ARA remain the default in 2.0? This is explicitly undecided (#446 2026-10-01).
7. Can llama.cpp load the rank-50 PEFT adapter for Qwen3.5 `linear_attn.out_proj` directly? Not checked.
8. There is no maths writeup, so the "affine-convex" description and the rank claims rest on the code and comments only.

## Sources (all opened)
- heretic master `66a952b`: `src/heretic/modifiers/ara.py`, `src/heretic/model.py`, `src/heretic/main.py`, `src/heretic/modifier.py`, `src/heretic/modifiers/abliteration.py`, `config.default.toml`, `pyproject.toml`
- Commits: `ea7c59a`, `154241f`, `c76416f`, `f7a456b`, `25979ad`, `662e4ba`, `99be5a6`, `ce8b05c`, `208c0ca` (#476), `8ae8526` (#478), `3521f86` (#423)
- PRs: #211 (body, 89 comments, 4 review comments), #332 (body, 40 comments, 13 review comments), #446 (body, 14 comments), #196 (body, comments)
- Issues: #404, #288, #339, #473, #140
- PR #211 head `edc3b12` `src/heretic/model.py:578-798`; PR #196 head `444edec` `src/heretic/som.py`
- PyTorch v2.9.1 `torch/optim/lbfgs.py`
- HF `Qwen/Qwen3.5-9B` and `Qwen/Qwen3.5-27B` `config.json`; `Qwen/Qwen3.5-9B` `chat_template.jinja`
- arXiv 2511.08379 abstract page
