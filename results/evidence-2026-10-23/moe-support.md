# Mixture-of-experts support: what it would take (2026-10-23)

Why: `gaps.md` ("Mixture-of-experts models") puts MoE bases of 36B or less at a third to a half of
the dense models' gap, led by poolside/Laguna-XS-2.1. Sources: arXiv abstracts and HTML tables,
GitHub source and API, Hugging Face configs, indexes and cards, llama.cpp b11524's source
(`~/.cache/candor/llama.cpp-b11524/source`), and `candor inspect` run today on five MoE models
(headers only). No weights were downloaded and no pod was started. Numbers marked (T) come from a
paper's HTML tables; an independent check matched 2609.09793, 2606.04160 and 2512.21008 against
their PDFs and confirmed every arXiv ID, llama.cpp line and candor file:line cited here.

## Bottom line

- A residual-stream difference-of-means direction still mediates refusal in MoE models, but the
  weight edit works only when every writer is edited together, every routed expert's down
  projection included. No paper's direction edit changes the router (SC117's Laguna-S-2.1 card
  scales router rows; L3, SAFEx and 2610.02910 change routing at inference).
- candor's scoring carries over. llama.cpp b11524 applies LoRA to stacked expert tensors at
  runtime with per-request scales, through the same server API. Shared experts and attention take
  ordinary 2D adapters. With `row_norm = "none"`, removing a direction from every expert, the shared
  expert and attention removes it exactly from the MoE block's output. Under "pre" (the default) and
  "full" (the 27B's route) each expert's rows are rescaled on their own: a per-expert edit along one
  direction, not a projection of the block.
- The work is in candor's code: about a dozen places assume one Hugging Face tensor is one 2D GGUF
  tensor. `inspect` already finds every Laguna-XS writer (10,064) and fails only on "Experts".
- Laguna-XS-2.1 needs no remote code (transformers 5.19 has it natively) and converts with b11524.
  Its residuals need an 80 GB card, as the 27B's did; its search fits an A40 at 32 slots.
- Order: the code with tests on tiny random MoE models on the Mac ($0), optionally a cheap real
  proof on OLMoE-1B-7B on one A40 (about $2-3), then Laguna-XS-2.1 (about $25-35 on the 27B's
  route). Laguna can't be scored on Metal in b11524, so its scoring runs on pods only.

## 1. Published work

**Published (measured)**

- **The edit needs every writer at once.** 2609.09793 removes one direction from GLM-5.3-Flash's
  (320B, 288 routed experts) attention output, dense and shared-expert down projections and every
  routed expert's down projection, router untouched: refusal 0.949 to 0.173 with inference-time
  hooks (T). Each group alone removes little (attention 0.039, dense plus shared 0.016, routed
  0.148), all together 0.776. Finding modules by name skips fused experts and removes only 0.066. Re-quantising to FP8 brought part of the
  refusal back (T). One model, direction from one layer.
- **One residual direction is still the right object.** 2606.04160 (GPT-OSS-20B, Mixtral,
  OLMoE): residual steering reaches 65-95% attack success. Per-expert directions recover 78% of it
  on average where free to choose their location (single runs up to 112%), 54-61% when fixed;
  routing-picked experts beat random only on GPT-OSS.
- **Routing follows topic, not refusal.** 2605.29708: refusal and compliance continuations of the
  same prompt share 7.18-7.92 of their top 8 experts (T); random pairs share 6.71 on Qwen3-30B, so
  there the margin is small.
- **Steering moves the router.** 2608.21236 (RARE): under direct injection, Mixtral's router-logit
  cosine distance grows with depth, 0.038 (layer 17) to 0.119 (layer 27) (T). Null-space projection
  plus a drift correction raises harmfulness-steering success from 14.6% to 53.3%, averaged over
  six models. This is steering, not a weight edit.
- **MoE may need strength above 1.** 2607.02714: down_proj and o_proj edits on 24 models (Kimi K2
  cyber refusal 100% to 7% at scale 1.0, T). It found a scale above 1.0 "impactful" for MoE;
  candor's 0.8-5.0 covers both.
- **Silencing a few experts is a separate lever** (routing at inference, not a direction edit).
  L3 2602.08741, router masks: 7.3% to 70.4% attack success over 8 MoE models. SAFEx 2506.17368:
  12 experts cut Qwen3-30B-A3B's refusal from 93.6% to 71.6% (T). 2610.02910, a negative router
  bias: OLMoE 34 to 9 refused of 100; router-gradient ranking beat frequency in 24 of 25 settings.
  GateBreaker 2512.21008, about 2.6% of gate/up neurons zeroed: 7.4% to 64.9% (T). SteerMoE
  2509.09660 overrides router scores for chosen experts.
- **Shared experts hold safety too.** SEAL 2609.02293, Qwen3.5-35B-A3B among its models: shared
  experts hold 0.63-0.89 times the routed experts' density of safety neurons (T).
- **No paper reports KL for an abliterated MoE.** 2512.13655 excluded Mixtral.

**Hypothesis (and the test)**

- **The router needs no edit, but the edit's routing drift may cost quality.** Test: KL and maths
  on the shipped file. If they suffer, also project the direction out of the router's input: a
  rank-1 LoRA on `ffn_gate_inp`, which llama.cpp also applies at runtime (`build_lora_mm`).
- **Q4_K_M brings refusal back as FP8 did in 2609.09793.** Test: the export already measures each
  edited tensor's kept share, and the final measurement uses the shipped file.
- **Editing "super experts" disturbs the model.** 2507.23279 finds 3 of Qwen3-30B-A3B's 6,144
  experts produce massive activations through down_proj. Test: KL with and without them edited.
- **Per-expert directions beat one per layer in a weight edit.** Steering is mixed (2606.04160);
  weight edits are untested. Not worth building first.

## 2. heretic and public MoE abliterations

**heretic master** (09271b6, 2026-10-20) edits only attention on fused-expert models.
- In its pinned transformers 5.18, Qwen3-MoE, Qwen3.5-MoE, gpt-oss, Mixtral, LFM2-MoE, Laguna and
  others hold a layer's experts as one fused 3D parameter. heretic's per-expert loop raises an
  error that `suppress(Exception)` swallows (`src/heretic/model.py` L414-482).
- It has no `shared_expert` path and no router edit. It doesn't use PEFT's `target_parameters`
  (0.17+, which can wrap a fused expert `nn.Parameter`), and the maintainer won't add direct weight
  edits (#339). On Laguna it would edit only the 40 `o_proj`
  and layer 0's dense MLP.

**Measured in heretic issues** (refusals of 100):
- **#342, Qwen3-30B-A3B.** With standard abliteration (5 seeds x 40 trials), adding experts
  lowered refusals at equal KL by 4.8 ± 4.2 (KL up to 0.02, within noise), 17.8 ± 4.5 (0.05),
  29.2 ± 6.0 (0.10) and 40.4 ± 6.3 (0.20); the maintainer judged 40 trials too short. Under LoRA-ARA
  (200 trials an arm) the experts were mixed, no gain beyond noise.
- **#198, Qwen3.5-122B-A10B.** Attention plus shared expert stayed at 98-100 after 48 trials;
  adding the fused experts reached 7 (one user's unreviewed report).
- **#218, Qwen3-30B-A3B, experts reachable.** 99 refusals at KL 0 on every trial: a broken
  measurement, not evidence about experts.

candor's edit is the standard kind, not ARA. The published case that experts matter is 2609.09793;
heretic's issues agree but are weak (#342 short, #198 one unreviewed report).

**Public cards** (claims only; prompt sets and KL definitions differ):
- huihui-ai cite remove-refusals-with-transformers with no numbers; their GLM-5.2, GLM-5.3-Flash
  and Ornith cards say experts or many layers were left unedited.
- Laguna-S-2.1 (97-98 refused unedited): SC117, abliterix plus router suppression (20 experts a
  layer, rows scaled 0.744), 8 at KL 0.0042; UnstableLlama, a heretic fork with ARA on each
  expert's down_proj, 3 at KL 0.036. Laguna-XS-2.1's one version is an abliterix NVFP4 (4, from
  98), no GGUF.
- Qwen3-30B-A3B-2507, heretic v1.2.0's per-expert LoRA: 11 of 100 (from 99), KL 0.0679. Qwen3.6-35B-A3B,
  attention only (tvall43): 5 of 86, KL 0.0097, so some hybrids may need less.

## 3. llama.cpp (b11524 = 86a2835, 2026-10-09; master 4231b23, 2026-10-23)

- **Expert LoRA runs at runtime.** `build_lora_mm_id` (`src/llama-graph.cpp:1590`, since PR #8332,
  2024-07-15) adds `mul_mat_id(b, mul_mat_id(a, x, ids), ids)` for each loaded adapter, and
  `build_moe_ffn` sends `down_exps` through it (`:2364`) before the router weights scale and sum
  the experts. Its scale is the per-request scale times alpha over rank (`:1625`), as for dense
  adapters; the per-request `lora` field (PR #10994) and `POST /lora-adapters` (PR #8857) ignore
  tensor type.
- **Layout for `blk.N.ffn_down_exps.weight`** (ne `[n_ff_exp, n_embd, n_expert]`): `lora_a` ne
  `[n_ff_exp, r, n_expert]`, `lora_b` ne `[r, n_embd, n_expert]`: under "none" the direction
  copied into every expert slice, under "pre" and "full" a `lora_b` of each expert's own (a shared
  2D `lora_b` isn't supported). The loader checks only `ne[0]`, `ne[1]`
  and the rank (`src/llama-adapter.cpp:358-370`): a wrong expert count loads, then writes out of
  bounds (#29128 and fix #29144, both open), so candor must check `ne[2]` itself.
- **Plain `build_lora_mm`** covers the shared expert (through `build_ffn`), attention output,
  `ssm_out` and the router in laguna, qwen35moe and nemotron-h (`src/models/laguna.cpp:258`).
- **Under "none" the edit is exact.** It is linear and router weights are per-token scalars applied
  after `down_exps` (Llama 4 scales the input instead, still exact), so sum_e g_e (I - s r r^T)
  W_e x = (I - s r r^T) sum_e g_e W_e x: with `row_norm = "none"`, the MoE block's output with the
  direction removed, whatever the routing. Under "pre" and "full" each expert's rows are
  normalised on their own: a per-expert edit along the same direction.
- **No alternative is needed.** Control vectors only add a vector (`llama_adapter_cvec::apply_to`),
  there is no runtime projection (#11029, closed as stale), and merging per trial would mean a
  requantise and reload every trial.
- **Conversion.** `convert_hf_to_gguf.py` stacks per-expert weights with `torch.stack(dim=0)`
  (`conversion/laguna.py`, `qwen.py:113-140`, `nemotron.py:475-505`); fused `experts.down_proj`
  passes through. `convert_lora_to_gguf.py` stacks expert LoRAs only when every projection of
  every expert is present, so candor keeps writing adapters directly with gguf-py.
- **Q4_K_M** (`src/llama-quant.cpp:620-668`): `ffn_down`, `_exps` and `_shexp` get Q6_K on
  `use_more_bits` layers, else Q4_K. One type per tensor, so candor's Q8_0 fallback takes a whole
  layer's experts. Rows not a multiple of 256 fall back to Q5_0 or Q8_0: all of Nemotron's (2688,
  1856, 3712); Qwen3-MoE, Qwen3.5/3.6-MoE and Laguna are clean.
- **Architectures** (b11524 and master): qwen3moe (#12828), qwen35moe (#19468), nemotron_h_moe
  (#18058), laguna (#25165, 2026-07-22; S-2.1 #26233).
- **Laguna caveats.** #30331, merged after b11524, reads XS-2.1's YaRN beta_fast of 64 from the
  file (b11524 uses 32; near-identical perplexity, per the PR). #30896 (open): on Metal, Laguna
  degenerates at ubatches of 32 or more because `ffn_moe_down`'s input exceeds F16 range in
  `mul_mm_id`. The issue's one-line fix doesn't survive an adapter in b11524: with a LoRA on
  `down_exps`, `experts` is an ADD node, so `ggml_prec_set_src` returns false (`ggml.c:3315`), and
  the adapter's own `mul_mat_id(lora_a, cur)` is unprotected. Master's `build_amax_src1_f16` runs
  before `build_lora_mm_id` and covers both. On CUDA the b11524 flag makes `supports_op` reject the
  MUL_MAT_ID (`ggml-cuda.cu:5354`), so the patch must stay out of pod builds.
- **Untested:** `MUL_MAT_ID` speed and correctness with rank 1-3 F32 adapters on CUDA. The first
  pod test is logits served with the adapter against the merged weights.

## 4. candor's code

**`inspect` today** fails only on "Experts" for Laguna-XS-2.1 (10,064 writers: 40 `o_proj`, 1
dense `down_proj`, 9,984 `experts.N.down_proj`, 39 `shared_expert.down_proj`), Ornith (120, fused)
and Qwen3-30B-A3B-Instruct-2507 (6,192). Nemotron-3.5-Lightning fails with "layer N has no mlp
writer" on all 52 layers; Granite-3.1-3B-A800M with `block_sparse_moe.input_linear` unclassified.

| Where | Assumes | Change |
| --- | --- | --- |
| `targets.py:144` | an expert's name contains `experts` | also any 3D writer (Granite's `block_sparse_moe.output_linear`; make it with the READERS change below, or Granite passes inspect with a 3D tensor read as 2D) |
| `targets.py:31` READERS | no `input_linear` | add it (Granite's fused gate/up) |
| `targets.py:94` `_component` | `mixer` means attention, checked first | `experts`/`shared_experts` mean mlp, checked first (NemotronH's `mixer.experts`) |
| `targets.py:152` missing | every layer has an attn and an mlp writer | every layer has a writer (a NemotronH layer is one block) |
| `targets.py:162` `gguf_architecture` | NemotronHForCausalLM is `nemotron_h` | `nemotron_h_moe` when the config has `num_experts_per_tok`, as the converter decides |
| `targets.py:179` `gguf_name`, `run.target_names` | one HF tensor is one GGUF tensor | per-expert names map to their layer's stacked `ffn_down_exps`; a target becomes a GGUF tensor holding its writers in expert order |
| `cli.py:140,151`, `run.py:395` | refuse experts | accept them; refuse only fused `[E, in, hidden]` (Llama 4, gpt-oss) and natively quantised experts (gpt-oss MXFP4) |
| `edit.py:13,29` `rank_one`, `full` | a 2D weight | the same maths over a leading expert axis `(E, out, in)`; `full` through batched `torch.svd_lowrank` |
| `adapters.py:14` | rank is `a.shape[0]` | `a.shape[-2]`; write 3D `lora_a`/`lora_b`; check the expert count against the base |
| `run.py:1009,1067,1090`, `export.py:70,346` | one weight per target | build a target's change from its stacked experts, reading a layer's experts in one shard open (`weights.py:24` opens the file per tensor: 9,984 opens a trial on Laguna) |
| `convert.py:127` `_column_order` | the linear-attention V reorder is `qwen35` only | also `qwen35moe`, whose converter shares the base class (Ornith) |
| `convert.py` `check_targets` | compares per writer | compares the stacked tensor, which checks orientation and expert order |
| `export.py:452` `kept` | one share per tensor | the least share over expert slices must reach 98% |
| `targets.py:45,186-242` | KV sizes for 5 architectures | add qwen3moe, qwen35moe, laguna (sliding window), nemotron_h_moe for inspect's memory row |
| `measure.py:84` `Edit.strength` | a kernel per component | unchanged: experts, shared expert and dense MLP take the mlp kernel, as in 2609.09793 and heretic |

`residuals.py` needs nothing: its decoder-layer hooks already take tuple outputs.

**Simplest robust design.** Keep the search, kernels and components. The one new idea is that an
edit target is a GGUF tensor, which may be a stack of a layer's experts. Every routed expert gets
the layer's direction at the mlp strength, built per expert and written as one 3D adapter. The
router, gate and up projections are never edited. Selecting experts, editing the router and
per-expert directions are later options, each to wait for evidence.

**Tests.**
- Unit tests on the real headers of Laguna-XS, Qwen3-30B-A3B, Ornith, Nemotron and Granite:
  targets, grouping, GGUF names, the remaining refusals.
- Batched `rank_one`/`full` equal a per-expert loop.
- On a tiny torch MoE with `row_norm = "none"`, editing all experts equals projecting the block's
  output.
- e2e (`CANDOR_E2E=1`, like the tiny Qwen3.5) through the real converter, on tiny random Qwen3-MoE
  and Laguna (per expert) and Qwen3.5-MoE (fused):
  - `check_targets` passes;
  - llama-server's logits with the 3D adapter at scale s match the merged export at s (KL about 0),
    on Metal and once on CUDA (tiny random weights don't overflow F16, so this can't show #30896);
  - Q4_K_M keeps at least 98% of the edit in every expert.

**Memory** (configs and poolside's GGUF; the 27B's numbers from `phase-5-qwen3.8-27b.md`, its
writers from its config):

| Model | Residuals need (2 x params x 1.1) | Q4_K_M | f16 KV, 32 slots x 8,504 tokens | Writers read per "full" trial |
| --- | --- | --- | --- | --- |
| Qwen3.8-27B (measured) | read on an A100 80GB | 16.81 GB | 16.5 GiB; A40 peak 37.1-39.6 GiB | 15.4 GB |
| Laguna-XS-2.1 | 73.6 GB | 20.27 GB | 11.2 GB full layers + 4.0 GB sliding = 15.2 GB | 22.3 GB |
| Qwen3-30B-A3B, thinking | 67.2 GB | about 18.6 GB | 26.8 GB | 20.1 GB |
| Ornith-1.5-35B-A3B | 79.1 GB | about 21.6 GB | 5.6 GB, plus linear-attention state | 22.2 GB |

What it means:
- **Residuals** for every 30-36B MoE need an 80 GB card, as the 27B's did. The 27B's stages up to
  the directions took about 14 min ($0.42).
- **Laguna-XS on an A40:** its weights and KV total 35.4 GB, against the 27B's 34.6 GB, so 32
  slots should fit. Qwen3-30B-A3B thinking would need 24 slots or a q8_0 KV cache.
- **Adapter memory:** rank-1 expert adapters are 2.6 MB a layer per direction set on Laguna. A
  "pre" search holding dozens of sets takes a few GB of GPU memory; a "full" search builds one
  adapter a trial.

**Time** [Hypothesis]: plan with the 27B's A40 times as a starting guess, not a ceiling (thinking
off 85-110 s a trial, as used 15-47 min): a "full" Laguna trial reads 22.3 GB, rebuilds 9,984
expert slices and restarts llama-server. A 3B-active model does about a ninth of the 27B's arithmetic a token,
but at 32 slots each step still reads about 64% of Laguna's experts (87% of Qwen3-30B's). The
expert adapters' cost and a "full" trial's 9,984 per-expert SVDs are untimed: time 2-3 trials
before quoting.

## 5. Laguna-XS-2.1

- **Architecture** (config, `modeling_laguna.py`, b11524's `src/models/laguna.cpp`): 33.44B
  parameters (3B active), 40 layers, hidden size 2048, context 262,144, vocabulary 100,352; bf16,
  66.89 GB in 14 shards.
  - **MLP:** 256 routed experts, top-8, size 512, stored per expert
    (`mlp.experts.E.down_proj.weight`), their sum scaled by 2.5; one shared SwiGLU expert of 512
    with no sigmoid output gate (`mlp.shared_expert`); layer 0 dense (8192).
  - **Router:** sigmoid (`mlp.gate`) with a selection-only bias stored at
    `mlp.experts.e_score_correction_bias`.
  - **Attention:** GQA, 8 KV heads of 128. Layers 0, 4, ..., 36 are full attention (48 query
    heads, YaRN); the other 30 use a 512-token window (64 query heads). A per-head softplus gate
    sits before `o_proj`, with q/k norms and no post-norms, so every writer is a plain matrix
    into the residual.
- **Chat template:** thinking in `<think>...</think>`, switched by `enable_thinking`: off in the
  template's default (also poolside's GGUF), but poolside's `generation_config.json` sets
  `enable_thinking: true` and the card's Transformers example says reasoning is on by default. `inspect` reads the
  default ("answers without thinking"). A thinking run sets `deployment.chat_template_kwargs =
  {enable_thinking = true}`, and the thinking checks need per-model work, as K2's step 1 did.
- **"Custom code":** the repo ships `configuration_laguna.py` and `modeling_laguna.py` through
  `auto_map`. But transformers has had Laguna natively since PR #45673 (2026-04-28); candor's
  5.19 has it (first shipped in 5.7.0), and `inspect` says "candor uses transformers' own code for
  it". 5.19's class has the per-head softplus gate (`modeling_laguna.py:354-408`, from XS.2) and
  fuses the experts into 3D tensors on load, which doesn't matter to candor (residual hooks sit on
  decoder layers; edits read the checkpoint's names). Still to check numerically against the
  shipped code on a tiny config.
- **llama.cpp:** b11524 converts and runs it (poolside's own Q4_K_M is 20.27 GB). It can't be
  scored on Metal in b11524, edited or not (#30896).
- **Licence:** OpenMDW-1.1, with no restriction on use or outputs; not gated. The card asks users
  to follow poolside's Acceptable Use Policy and advises against circumventing its safety
  guardrails without equivalent mitigations. The policy itself wasn't read: read it before a
  release.

## 6. Recommendation

1. **The code and tests, on the Mac ($0):** section 4's changes with their unit and e2e tests on
   tiny random MoE models, about 2-3 days plus review rounds. Also a check of transformers 5.19's
   Laguna class against poolside's shipped code on a tiny config. No refusing per-expert MoE fits the
   Mac for a real run (OLMoE needs 15.2 GB; microsoft/Phi-tiny-MoE-instruct, 3.76B, per-expert `w2`,
   MIT, 8.3 GB, fits but ships remote code and its refusals are unmeasured).
2. **Optional: a first real proof on OLMoE-1B-7B-Instruct on one A40** (about 3-5 h, $2-3), once a
   count of its unedited refusals shows enough to measure (2610.02910 had 34 of 100 refused). It has
   64 experts top-8 stored per expert like Laguna, Apache-2.0, and 2606.04160, 2610.02910 and SteerMoE
   use it. Its 4,096-token context needs smaller slots, so its timings won't carry over. Steps: the
   CUDA served-against-merged logit check, the unedited model, a short search. Pass: refusals cut at
   a KL like the dense models', every expert keeping 98% of its edit in Q4_K_M.
3. **Laguna-XS-2.1, the gap.** First the user decides whether it is used thinking on or off, and
   the llama.cpp build: b11524 lacks #30331 (beta_fast 64), and Laguna has no reference measurement
   yet, so moving to a newer build costs no comparison.
   - An A100 80GB for download, GGUF and directions (about 30-45 min, $1-1.5), then an A40 pilot
     with the served-against-merged logit check on Laguna itself, timing 2-3 trials ($3-5) before
     quoting the rest.
   - Thinking off: the 27B's thinking-off search (600 trials, about 15 h, $9), then finalists,
     export and measurement, about $25-35 in all.
   - Thinking on: the 27B's route with the proxy check; the 27B cost about $35 ($6.8 pilot, $23.97
     steps 2-5, about $3.9 step 6).
4. **Later, each for its own reason:** Granite-3.1-3B-A800M (the fused path Laguna doesn't use,
   Granite-only code, a template that writes the date so a search must finish the day it starts,
   and an unprotected Metal adapter path); Ornith-1.5-35B-A3B (qwen35moe, close to the released 27B;
   the `_column_order` and KV changes; 84 uncensored repos already); Nemotron-3.5-Lightning (the
   NemotronH target fixes, thinking work, Q5_0/Q8_0 in place of Q4_K); router edits only if KL or
   maths fail on Laguna. gpt-oss stays out: native MXFP4 loses edits on requantising (the
   2026-10-15 evidence, item 3).
