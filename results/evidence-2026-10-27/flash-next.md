# Qwen3.8-Flash-Next: what candor needs to edit it, and what running it takes (2026-10-27)

The user chose Qwen/Qwen3.8-Flash-Next as the first MoE (`history.md`, 2026-10-27). This report
gathers what its config, its tensors, transformers 5.19.0's `qwen4_exp` code and llama.cpp b11524's
source show, and what that means for candor and for the pods. Numbers only; no weights downloaded.
Hugging Face reads at about 13:00-14:00 UTC; Runpod prices from `list-gpu-types` (secure cloud) at
about 14:30 UTC.

## Bottom line

- **It fits candor's method with three changes.**
  - The residual is four streams, read as their mean, as heretic does since #519 (2026-10-26).
  - Its one PLE projection is a residual writer candor leaves unedited.
  - The directions run loads the 360 GB model across several GPUs.

  Every weight candor edits is one llama.cpp b11524 applies LoRA to, so the per-request adapter
  search works as it does for the 27B.
- **It is 180B parameters but scores like a small model.** 123.3B are routed experts (10 of 512
  active), 51.2B are n-gram tables that llama.cpp keeps in CPU memory and reads row by row, and 3.8B
  is everything else. About 3.5B are active per token. At Q4_K_M the GPU should hold about 78-80 GB
  (an estimate for the pilot to measure), so one 96 GB RTX PRO 6000 ($2.49/h, the best-stocked large
  card) may score it. KV is about 24 KiB a token, but the 36 linear-attention layers hold about
  113 MB of recurrent state per slot, so slots may have to drop below 32.
- **What costs more than the 27B:**
  - The directions need the model in 16 bits on several GPUs: four RTX PRO 6000 (384 GB, $9.96/h).
    They also need 240 GB of RAM or more: transformers keeps the n-gram tables (102 GB in 16 bits) in
    CPU memory and joins their 128 shards there, and candor asks for twice the tables and a tenth
    more (225 GB).
  - Disk is about 1.2 TB or more.
  - Each finalist's export writes a 360 GB 16-bit copy before quantising.
  - The search card costs 4.2 times an A40 an hour. Whether it is fast enough to make up for that is
    the pilot's main question.
- **Demand is very large and crowded.** 11.5M GGUF downloads and 234 uncensored repos in 30 days. The
  most downloaded are SC117's abliterated GGUF (2,786,304), orcarouter's (1,065,641) and
  huihui-ai's v1 and v2 (331,140; 206,587).

## 1. The model

| | |
| --- | --- |
| Released | 2026-08-24; 3,350,248 downloads in 30 days, 6,583 likes |
| Layers | 48, hidden 2560; 36 linear attention (Gated DeltaNet), 12 full attention with a QSA token indexer (every 4th) |
| Attention | 24 heads, 2 KV heads of 256; linear: 16 key and 48 value heads of 128 |
| MoE | 512 routed experts, 10 a token, `moe_intermediate_size` 640, stored stacked (`experts.down_proj` [512, 2560, 640]); a gated shared expert of 640 |
| Hyper-connections | 4 residual streams (`hc_count`), mixer rank 320 |
| PLE | one layer (`ple_layer_ids` [2]): hashed 2- and 3-gram embeddings, 128 shards of [2500012, 160] |
| MTP | one layer, outside the main stack |
| Context | 262,144; vocabulary 248,320 |
| Thinking | `enable_thinking` and `reasoning_effort` (xhigh, the default; medium; low); thinking ends with `\n</think>\n\n` |
| Card sampling, thinking | temperature 1.0, top_p 0.95, top_k 20, min_p 0, presence 0, as the 27B's |

Parameters by part, from the safetensors headers:

| Part | Parameters | Share |
| --- | --- | --- |
| Routed experts | 123.31B | 68.5% |
| PLE n-gram tables | 51.20B | 28.4% |
| Attention, shared experts, routers, hyper-connections, PLE projections | 3.77B | 2.1% |
| Token embedding and head | 1.27B | 0.7% |
| Vision tower | 0.45B | 0.2% |

## 2. How the residual works, and what candor edits

- **Four streams that never mix.** The embedding is repeated into 4 streams. Each block reads a gated
  mix of the normalised streams and writes its output `h` back into every stream, scaled per stream
  by a token-dependent weight (`2 * sigmoid(...)`). A final mixer combines the streams for the head.
  So each stream is the embedding plus scalar multiples of every block's `h`.
  - Removing a direction from a writer's output removes it from what that block adds to all four
    streams.
  - The streams' mean is the residual the directions are read from. heretic's #519 reads it the
    same way for DeepSeek V4 and GLM 5.3, whose streams are mixed by a doubly stochastic matrix.
    Here they aren't mixed at all.
- **What candor edits:**
  - 36 `linear_attn.out_proj` (`ssm_out`) and 12 `self_attn.o_proj` (`attn_output`);
  - 48 stacked `mlp.experts.down_proj` (`ffn_down_exps`), each layer's experts one target as for
    the MoE models built on 2026-10-23;
  - 48 `mlp.shared_expert.down_proj` (`ffn_down_shexp`).

  Routers and the hyper-connection weights don't write into the stream.
- **The PLE projection is left unedited.**
  - `ple.value_proj` [2560, 2560] writes n-gram features into every stream, gated per stream.
  - It also goes through a dilated depthwise convolution after a norm and a SiLU, so its output
    isn't linear in the projection, and removing a direction from it isn't an exact projection.
  - It is one layer of 48 and carries token n-gram features. It is reported as left unedited, as a
    router is.

## 3. llama.cpp b11524

- `src/models/qwen4exp.cpp` builds the graph and `conversion/` registers both
  `Qwen4ExpForConditionalGeneration` and `Qwen4ExpForCausalLM`, so the pinned release converts and
  runs it.
- LoRA is applied by `build_lora_mm` to `wo` (`:1008`), `ssm_out` (`:1133`) and `ple_value`
  (`:1377`), by `build_moe_ffn` to the experts (`:1145`) and by `build_ffn` to the shared expert
  (`:1164`). Every target candor edits can be scored as an adapter with per-request scales.
- The n-gram table is `per_layer_token_embd`, an input-layer tensor (`llama-arch.cpp:964`). Input
  tensors live on the CPU (`llama-model.cpp:1630`), and this one is `TENSOR_READ_LAZY`: rows are read
  on demand from the memory-mapped file. So its 51.2B parameters take neither GPU memory nor all
  their size in RAM.

## 4. Memory, cards and disk

Estimates, for the pilot to measure:
- **GPU at Q4_K_M:** experts about 74 GB (some `ffn_down` at Q6_K, as Q4_K_M does), the rest about
  3-4 GB: about 78-80 GB.
- **Per slot:** KV 24 KiB a token (12 layers x 2 KV heads x 256 x K and V in f16); recurrent state
  about 113 MB (36 layers x 48 heads x 128 x 128 in f32). At 32 slots of about 9k tokens, about
  7 GB of KV and 3.6 GB of state, which with the weights is about 90-93 GB.
- **Directions:**
  - transformers keeps the n-gram tables (102 GB in 16 bits, lookups only) in CPU memory
    (`_no_placement_params`). It joins their 128 shards there, peaking at about 205 GB of RAM.
  - transformers' `infer_auto_device_map` keeps the first GPU free for layer 1 (107.6 GB with the
    tables).
  - The rest of what the causal LM loads goes on three more GPUs: 251.5 GB (89.5, 88.1 and 73.9). That
    leaves at least 11 GB each for activations, which peak at 1.9-3.3 GiB for a batch of 32 prompts of
    128-256 tokens (measured in bf16 on the CPU on four Flash-Next-sized layers).
  - So the pod is 4 x 96 GB with 240 GB of RAM or more (independent review, 2026-10-27, simulating
    transformers 5.19's device map).

| Card (secure) | Memory | $/h | Stock | For |
| --- | --- | --- | --- | --- |
| RTX PRO 6000 Blackwell | 96 GB | 2.49 | HIGH, 8 data centres | scoring, one card; directions, four |
| H200 SXM | 141 GB | 5.29 | LOW, 8 | scoring if 96 GB is too tight |
| H100 NVL | 94 GB | 3.19 | LOW, 1 | |
| A100 SXM 80GB | 80 GB | 1.79 | MEDIUM, 6 | directions, six |
| B200 | 180 GB | 7.99 | LOW, 3 | |
| A40 | 48 GB | 0.59 | LOW, 2 | doesn't fit |

**Disk.** About 1.2 TB or more: the safetensors (360 GB), the 16-bit GGUF (360 GB), the Q4_K_M
(about 80 GB) and each export's 16-bit copy (360 GB) while it is quantised.

## 5. What the pilot must settle

1. The build on a 4-GPU pod: download, conversion and quantisation times; the directions across
   the GPUs, finite and hidden-wide. Each stream's difference of means against the mean's, by
   cosine per layer, from the same prompts (the review's check that the mean loses nothing).
2. That b11524 runs it correctly: the unedited model's refusals, maths and thinking on the first 64
   of each set, as used.
3. The scoring card: whether the Q4_K_M and 32 slots fit 96 GB, or how many do; tokens a second; and
   GPU memory under load.
4. Trial times, as used and thinking off, with the experts' adapters. How long a "full" edit's
   adapters take to build for 48 x 512 experts, or whether this model should search with
   `row_norm` "pre" or "none". An MoE block is edited exactly only under "none"
   (`evidence-2026-10-23/moe-support.md`).
5. The adapters against the merged weights on CUDA (the MoE e2e tests on the pod, and the new
   qwen4exp one).
6. Which reasoning effort is used, and whether thinking off ranks edits as thinking on does, before
   any thinking-off search (`candor proxy`, as for the 27B).

## 6. References to measure against, later

The most downloaded uncensored versions in 30 days are mostly GGUF-only:
- SC117's GSQ-RCO abliterated (2,786,304);
- orcarouter's Uncensored (1,065,641, and an FP8 copy of 110,691);
- huihui-ai's abliterated v1 and v2 (331,140; 206,587).

How each can be measured as candor ships its own is a question for the decision rule, before the
search.
