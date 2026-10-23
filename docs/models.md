# Models

What candor needs from a model, and the settings that change with it. Every setting lives in the
run's `candor.toml`, each explained by a comment; [`upgrade`](commands.md#upgrade) adds new ones.

- [What fits where](#what-fits-where)
- [Slots](#slots)
- [Thinking models](#thinking-models)
- [Sampling](#sampling)
- [Mixture of experts](#mixture-of-experts)

## What fits where

Reading residuals needs the model in 16 bits on the GPU. `run` checks this before downloading or
converting anything.

| Model | Where it runs |
| :--- | :--- |
| rude-llama (Llama 3.2 3B) | An 18 GB Mac |
| Qwen3.5-9B, K2-Horizon-7B | A [pod](pods.md) |

The 8B [second judge](commands.md#the-second-judge) needs a pod too.

Qwen3.5's fast linear-attention kernels aren't installed: transformers runs Qwen3.5 only to read
residuals for short prompts, the answers come from llama.cpp, and the judge (a Qwen3) has no linear
attention. If reading Qwen3.5's residuals turns out slow, `flash-linear-attention` is the fix.

## Slots

`[slots]` sets how many answers llama-server works on at once:

| Setting | Used by | Default |
| :--- | :--- | :--- |
| `thinking` | the unedited model thinking, for `thinking/` | `init` sets 32 for a thinking model, 8 otherwise |
| `search` | `search`, `score` and `proxy` | unset: 32 for a thinking model, 8 otherwise |
| `measure` | `measure` (`--slots` overrides it) | `init` sets 32 for a thinking model, 8 otherwise |

A measurement made at other than 8 slots has `-slots-N` in its name.

The server holds every slot's context in GPU memory from its start, so the slots times each one's
context, plus a hybrid's state for each slot, must fit. The search gives each slot room for the
longer of an answer and a KL read. When they don't fit, candor says how many slots of how many tokens
asked for how much, and which setting to lower.

## Thinking models

A model that thinks before it answers (its chat template opens the thinking, as Qwen3.5's does) is
scored thinking, as you run it.

### Budget and close

- `init` finds the text that closes the model's thinking and gives it a budget of 8192 tokens
  (`deployment.thinking_budget`). Past it, llama-server makes the model stop thinking and answer.
- Only the answer is judged. Each answer's thinking length is kept as a number.
- The search can score at a shorter budget (`scoring.thinking_budget`). The edits it suggests are
  measured at the deployment's.
- A template that can't turn thinking off, as K2-Horizon's can't, is closed by the end tag of the tag
  its prompts open with (`</ifm|think>`).

### Templates candor refuses or warns about

- `inspect` fails a model whose template turns thinking off but whose prompts don't open it, as
  Qwen3's hybrid models' don't, since candor would judge the thinking the model opens itself as its
  answer.
- It warns where a template names a think tag but its prompts neither open nor close one.

### Chat template settings

`deployment.chat_template_kwargs` holds settings the chat template is rendered with, such as
`{ reasoning_effort = "high" }`. Every prompt candor builds is rendered with them, and `serve` passes
them to llama-server.

- `init` sets a template's reasoning effort at its own default, and lists the values it takes at the
  top of `candor.toml`.
- K2-Horizon opens its thinking with another tag at each effort, so changing the effort means
  changing `think_close` too, which every command checks.

### Where directions and KL are read

- **Directions** are read both where the thinking starts and where the answer starts after an empty
  thinking block. The search chooses between them for each trial.
- **KL** is read where the answer starts after the unedited model's own thinking, leaving out prompts
  whose thinking ran out its budget.
- That thinking is sampled once for each set of settings and kept, never sampled over, since sampling
  it again wouldn't give the same (`thinking/`; the thinking itself only as token ids in `private/`).
- `measure` also reads KL after an empty thinking block, as heretic read it, and where the thinking
  starts. The search reads neither, since none of its choices use them.

### Scoring with thinking off

Scoring every trial thinking is costly. [`candor proxy`](commands.md#proxy) checks, on the run's own
model, whether the search may score with thinking off instead.

## Sampling

**Greedy by default.** Answers are greedy unless `[sampling]` says otherwise. Set it to the settings
you chat with, such as a thinking model card's recommended ones: the search, measurements and `serve`
then all sample that way.

**Seeds.** Each prompt gets a seed of its own that is the same for every edit, so two edits are still
compared prompt by prompt. A thinking model's answer, asked for after its thinking, gets a seed of its
own made from its prompt's, since llama-server starts its random draws again at every request.

**Noise.** `measure --seed N` samples with other draws, to see how much sampling alone moves the
counts. A greedy run has no draws to change, so it refuses `--seed`.

**On the GPU.** A sampled run samples on the GPU (`--backend-sampling`), many times faster for a
large vocabulary. llama-server can't do that for a request with a thinking budget, so candor holds the
model to its budget itself, token for token as the server's own budget does (checked against it).
Draws for a seed differ from sampling on the CPU, so a measurement made before 2026-10-14 pairs with
one made up to 2026-10-18 prompt by prompt, not draw by draw.

## Mixture of experts

A mixture-of-experts model whose experts are stored apart, as Laguna's, Qwen3-MoE's and OLMoE's are,
or held [experts, out, in] in one tensor, is edited with every routed expert's down projection along
its layer's direction at the MLP strength, as are a shared expert and any dense layer.

- A layer's experts are one edit target, stacked as llama.cpp's converter stacks them, scored as one
  3D adapter and checked in the export by the least any expert keeps.
- The router isn't edited.
- `inspect` fails experts held [experts, in, hidden] in one tensor, as Llama 4's are, or square,
  [experts, hidden, hidden], whose layout can't be told.
