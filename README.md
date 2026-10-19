# candor

Removes a language model's refusal direction. It is being built to score every candidate edit the
way you actually run the model: in llama.cpp, on the quantised GGUF you ship, at your settings.

How it works:

1. Read the residual stream for harmful and harmless prompts with transformers, once, and take
   the difference of means per layer. The prompts are tokenized by llama-server itself, so the
   directions see exactly the input that scoring and your apps use.
2. Turn the edit into rank-1 LoRA adapters, one per edit target (each weight that writes into the
   residual stream) for each candidate direction, in the layout llama.cpp's converter gives them.
3. Load them all into llama-server and score each candidate by setting per-layer strengths per
   request, so nothing is re-converted between trials. A "full" edit (`edit.row_norm`) isn't linear
   in strength, so each candidate instead gets one adapter of its own and the server is restarted.
4. Write the chosen edit into a copy of the model's GGUF, every other tensor untouched, quantise
   it as you run it, and measure that exact file.

Answers to harmful prompts are only ever written to a run's `private/` folder. Commands print
counts and measurements, never generated text.

Status: early development; the plan and what is done are in [STATUS.md](STATUS.md).

## Use

```
uv sync
uv run candor inspect ~/Desktop/tune/models/rude-llama-bf16   # a local folder or a Hugging Face id
uv run candor init rude --model ~/Desktop/tune/models/rude-llama-bf16   # writes runs/rude/candor.toml
uv run candor run runs/rude                                    # every stage built so far
uv run candor upgrade runs/rude                                # after updating candor: add new settings
uv run candor measure runs/rude                                # the unedited model, on held-out prompts
uv run candor measure runs/rude --heretic STUDY.jsonl --trial 170   # a heretic trial, rebuilt
uv run candor compare runs/rude base heretic-170               # prompt by prompt
uv run candor proxy runs/qwen                                  # a thinking model: may the search think off?
uv run candor proxy runs/qwen --trial 37 --trial 41            # the search's picks, scored thinking too
uv run candor search runs/rude                                 # search for a better edit, overnight
uv run candor measure runs/rude --trial 37                     # one of its trials, on held-out prompts
uv run candor score runs/rude --heretic STUDY.jsonl --trial 170     # where it would sit in the search
uv run candor export runs/rude --trial 37                      # write it into the GGUF you run
uv run candor measure runs/rude --trial 37 --exported          # measure that exact file
uv run candor measure runs/qwen --seed 1                       # sampled again with other draws: the noise
uv run candor export runs/rude --outside ~/heretic-save --name heretic-saved   # another tool's edit
uv run candor measure runs/rude --outside heretic-saved        # measured like candor's own
uv run candor serve runs/rude --trial 37                       # chat with it in llama-server
uv run candor                                                  # or all of it from menus
```

- `candor` with no command, in a terminal, opens menus: each run in `runs/` with how far it has got,
  then a run's panel (its model, the stages built, the search's best trials, its measurements and
  exports) and what it can do next, starting on the step it is ready for. Trials lists those worth
  measuring, fewest refused first, and any already measured or exported. Each choice runs the same
  command you could type, shown first; one that can take hours can show instead the `scripts/pod.sh`
  commands that run it on a pod. Arrows or j/k move, enter chooses, Ctrl-C goes back.
- `inspect` reads only the config, tokenizer and file headers, so it is quick and needs no GPU. For an
  architecture whose KV cache candor knows, such as Llama's, Qwen3.5's and K2-Horizon's, it shows how
  much a token of context takes at f16, how many tokens fit in a GiB, and for a hybrid such as
  Qwen3.5 the state its linear-attention layers hold for each slot.
- `[slots]` sets how many answers llama-server works on at once: `thinking` as the unedited model
  thinks for `thinking/`, `search` as the search, `score` and `proxy` score edits (unset: 32 for a
  thinking model, 8 otherwise), and `measure` for `measure`, which `--slots` overrides. `init` sets
  `thinking` and `measure` to 32 for a thinking model, 8 otherwise. The
  server holds every slot's context in GPU memory from its start, so the slots times each one's
  context, plus a hybrid's state for each slot, must fit: the search gives each slot room for the
  longer of an answer and a KL read. When they don't fit, candor says how many slots of how many
  tokens asked for how much, and which setting to lower.
- A model that thinks before it answers (its chat template opens the thinking, as Qwen3.5's does) is
  scored thinking, as you run it. `init` finds the text that closes its thinking and gives it a
  budget of 8192 tokens (`deployment.thinking_budget`): past it, llama-server makes the model stop
  thinking and answer. Only the answer is judged, and each answer's thinking length is kept as a
  number. The search can score at a shorter budget (`scoring.thinking_budget`) and the edits it
  suggests are measured at the deployment's. A template that can't turn thinking off, as
  K2-Horizon's can't, is closed by the end tag of the tag its prompts open with (`</ifm|think>`).
  `inspect` fails a model whose template turns thinking off but whose prompts don't open it, as
  Qwen3's hybrid models' don't, since candor would judge the thinking the model opens itself as its
  answer; and warns where a template names a think tag but its prompts neither open nor close one.
- `deployment.chat_template_kwargs` holds settings the chat template is rendered with, such as
  `{ reasoning_effort = "high" }`: every prompt candor builds is rendered with them, and `serve` passes
  them to llama-server. `init` sets a template's reasoning effort at its own default and lists the
  values it takes at the top of `candor.toml`. K2-Horizon opens its thinking with another tag at each
  effort, so changing it means changing `think_close` too, which every command checks.
- Scoring every trial thinking costs many times more than answering after an empty thinking block,
  greedily (`scoring.thinking = "off"`, answers `scoring.closed_answer_tokens` long). That cheap
  score is only worth having if it ranks edits as thinking does, which no published work shows and
  which changes by model. `candor proxy` checks it on the run's own model, after a short search
  scored with thinking off (`--trials 150`; random edits mostly change nothing, so they can't show an
  order): its 3 best trials of each kind and each place directions are read, each also at 1.4 times
  its strength, scored both ways on the test prompts and their orders compared, with the edit
  refusing fewest scored again with another seed to show the noise. If the cheap score holds, the
  search carries on from those trials. It says the cheap score holds, holds but thinking needs stronger edits (the search's
  suggestions must then also be scored thinking at more strength), or can't be trusted. Set
  `scoring.thinking = "off"` only when it holds; edits are chosen on numbers measured thinking
  either way. A model whose chat template can't turn thinking off is never used answering after an
  empty block, so candor refuses to score it that way, which `proxy` needs; so too where turning it
  off changes more than the end of the prompt, as Qwen3.8's does at every reasoning effort but
  medium. Its scores are kept in
  `proxy/`, so a stopped check carries on. `--trial`, repeated,
  scores the trials named instead (the search's best, once it may score thinking off), each also at
  1.4 times its strength, kept apart in `proxy/trials.json`, with counts but no verdict; naming other
  trials starts that afresh.
- `edit.row_norm = "full"` makes heretic's "full" edit: each edited weight row is scaled back to its
  own length, then the change is kept at heretic's rank of 3, so rows keep their length only roughly,
  built as heretic builds it (checked against its code; its randomised SVD's seed moves the change
  by under 0.1%). On Qwen3.5-9B heretic's Trial 161 used it, and a candor search with it did better
  than one with "pre" (`findings.md`). A "full"
  edit isn't linear in strength, so it can't be one adapter per weight dialled per request: each
  trial gets an adapter built at its strengths and llama-server is restarted with it, which adds
  seconds to every trial.
- A sampled run samples on the GPU (`--backend-sampling`), many times faster for a large
  vocabulary. llama-server can't do that for a request with a thinking budget, so candor holds the
  model to its budget itself, token for token as the server's own budget does (checked against it).
  Draws for a seed differ from sampling on the CPU, so a measurement made before 2026-10-14 pairs
  with one made up to 2026-10-18 prompt by prompt, not draw by draw.
- For a thinking model, refusal directions are read both where the thinking starts and where the
  answer starts after an empty thinking block, and the search chooses between them for each trial.
  KL is read where the answer starts after the unedited model's own thinking, leaving out prompts
  whose thinking ran out its budget. That thinking is sampled once for each set of settings and
  kept, never sampled over, since sampling it again wouldn't give the same (`thinking/`; the
  thinking itself only as token ids in `private/`). KL after an empty thinking block, as heretic
  read it, and where the thinking starts are kept beside it.
- Answers are greedy unless `[sampling]` says otherwise. Set it to the settings you chat with, such as
  a thinking model card's recommended ones: the search, measurements and `serve` then all sample
  that way, each prompt with a seed of its own that is the same for every edit, so two edits are
  still compared prompt by prompt. A thinking model's answer, asked for after its thinking, gets a
  seed of its own made from its prompt's, since llama-server starts its random draws again at every
  request. `measure --seed N` samples with other draws, to see how much sampling alone moves the
  counts; a greedy run has no draws to change, so it refuses `--seed`.
- `init` runs the same checks and refuses a model candor can't edit. `candor.toml` holds every
  setting the run uses, each explained by a comment; edit it freely, and every later command checks
  it. Hugging Face models and the prompt datasets are pinned to the commits that were checked, and
  local models are stored by their full path. A model that only loads with the Python files it ships
  (K2-Horizon) gets `trust_remote_code = true`, which runs that code at the pinned commit.
- `run` resumes where it stopped. Each result in the run folder records what it was built from,
  including the version of candor's code for its stage, so changing a setting or upgrading candor
  rebuilds what it affects and everything after it, and says so.
- `upgrade` adds the settings a newer candor has and a run's `candor.toml` lacks, at their defaults,
  and lists them. candor never fills in a missing setting by itself.
- The held-out harmful prompts, which the search never sees, come from StrongREJECT without the
  rows it took from AdvBench. It is gated: accept its terms on its Hugging Face page, then
  `hf auth login`.
- `measure` answers the held-out prompts and GSM8K's 1,319 maths questions with the deployed GGUF
  in llama-server. Qwen3Guard-Gen-0.6B (pinned) labels each held-out answer as a refusal or not, with
  keyword counts beside it, and the first-token probabilities on the harmless prompts give KL
  divergence. It keeps one number per prompt in `measurements/NAME/`; answers and the judge's
  replies go only to `private/`. `--limit N` measures the first N prompts of each set and estimates
  the full run's time; a stopped measurement resumes from the last answer it saved; the Mac is kept
  awake. `--name` names it, but never as candor names its own (`base`, `trial-N`, `heretic-N`).
- `--heretic STUDY.jsonl --trial N` (for `measure`, `score`, `export` and `serve`) rebuilds one of
  heretic's trials with the run's own directions, read with the run's system prompt, not heretic's;
  one that read its directions after the run's own `think_close` is rebuilt with those read where the
  answer starts. candor reads every layer before the model's final norm and heretic its last one after
  it, so a per-layer trial's last layer is rebuilt with a direction heretic didn't read. candor finds
  a study's finished trials by the refusals heretic 1.x records for each; heretic 2.x records none,
  so its studies can't be read.
- `compare A B` pairs two measurements prompt by prompt: each count, the change with a 95% interval
  (Bonett and Price's for paired counts; for KL a studentised bootstrap over the prompts), how many
  prompts went each way, and the exact McNemar p-value; and the slots and seeds where they differ.
  Measuring the same edit twice with different `--slots` shows what batching noise alone does. It
  refuses two measurements made by different versions of candor's measuring: those made before
  2026-10-19 drew a thinking model's answer with its thinking's seed and misread maths answers written
  `\boxed{18}` or `-$5`. Measuring again under the name of one made by older code keeps the old one
  as `NAME-measuring-N`. It shows the GPUs, and the judge's device, where they differ, and reads no KL
  between measurements made on different backends (Metal against CUDA, or a `CANDOR_LLAMA_BIN`
  build), where KL moves by a few percent whatever the edit does.
- `search` tries edits of the shape heretic searches, in two kinds searched apart that take turns:
  every layer reading one layer's refusal direction, and each layer reading its own
  (`search.trials_per_scope`, 300 each). Each kind starts from 20 evenly spread trials, then
  Optuna's TPE sampler aims at the least KL among edits refusing at most `search.max_refusals` (2)
  of the 100 test prompts. Every adapter is loaded into one llama-server once, so a trial only sets
  strengths: it answers the test prompts `slots.search` at a time and the judge counts refusals,
  stopping once the trial has refused more than the cap and more than 25 fully scored trials of its
  kind, which leaves every trial the sampler draws unchanged; KL is read from the 100 harmless test
  prompts, except for a trial stopped so, which can't win. For a thinking model
  `search.max_budget_hits` (unset) can also cap the test prompts whose thinking reaches the budget:
  an edit over it doesn't count, and its trial stops there, which unlike the refusal stop can change
  later draws. Changing it carries a search on, its finished trials judged
  again by the new number. Nothing a model writes is kept. Strength starts at heretic's 0.8-1.5. From
  a kind's 60th trial, and every 30th after, if its best edit within the cap presses an end of that range
  (or none is within the cap yet: the ceiling), that end moves out to `search.widest_strength`
  (0.5-3.0), once; the kind's trials carry on and 10 evenly spread trials start the wider range.
  For a thinking model the 20 start points are split evenly between the two places directions are
  read. It resumes where it stopped, refuses to carry on a search made under other settings before it
  rebuilds anything, and `--trials N` stops early to time it. The unedited model's first tokens, which
  KL is read against, are kept in `search/base-first-tokens/` for each set of inputs and machine, so
  every session sets its trials against the same ones; a search carried on on another machine says
  so, as KL read there can differ by a few percent. It ends with the fully scored trials no other
  beats on both refusals and KL; for each kind, any strength it widened to, and the least-KL trial at
  each count within the cap unless one refusing fewer has less (or, failing that, the one that refused
  fewest), each flagged if it presses on the edge of a range; and the commands to measure each on the
  held-out prompts, since 100 test prompts can't tell 0 refused from 2. Every `search.report_every`
  trials (25; 0 for never) it shows its pace and the time left, and for each kind its trials, how
  many stopped early, its best so far as the end picks it (least KL within the cap, or else fewest
  refused) with that trial's strengths against the kind's range, and the trials since that best
  improved, keeping the latest report in `search/progress.json`. The setting is left out of the
  search's record, so changing it never stops a search carrying on. The results files explain the
  design: `results/phase-2-rude-llama.md` and, for widening and the two places,
  `results/phase-3-qwen3.5-9b.md`.
- `score` scores one edit exactly as the search scores its trials, without adding it to the search.
  A heretic trial is scored as rebuilt, and at the nearest whole layer as the search could draw it;
  it shows which of the search's best trials beat it on both refusals and KL, and whether the search
  could have drawn it at all. The search's best trial is scored again beside it as a check, and
  `--trial N` alone scores one of the search's trials. A trial the search scored in full should
  repeat its score within noise (a refusal either way, and 3% of its KL), or `score` says how it
  differs; the unedited model is scored again only when no such trial is among those scored. Like the
  search, it is refused before anything is rebuilt when the search was made from other inputs.
- `export` writes a trial's edit, as its adapters apply it, into a copy of the run's unquantised
  GGUF, each edited weight as llama.cpp's converter writes it (checked to give the same file as
  editing the safetensors and converting them), then quantises it as the run deploys. Quantising can
  round part of a small edit away (on rude-llama, whose weights were 4-bit before, Q4_K_M kept 55-98%
  of trial 535's and its refusals came back), and so can writing it in 16 bits, as an edit under half
  a 16-bit step rounds back to the weight. So it measures how much of each edited weight's change the
  quantised file holds. A weight that kept less than 98% is written again unrounded, in F32, and
  quantised as Q8_0 from that, or left in F32 where Q8_0 still falls short
  (straight away for a run deploying Q8_0 or finer); every other tensor keeps its plain type. The
  file is refused if one still falls short. Only `exports/EDIT/RUN-EDIT-QUANTISATION.gguf` is kept.
  `measure --exported` measures that exact file, and `compare` sets it against the unedited model
  and against the trial measured with adapters.
  `export --outside FOLDER --name NAME` ships a copy of the model edited elsewhere, such as heretic's
  own save of one of its trials, the same way (the run's model with each edited weight taken from
  the copy, so it is written, quantised and checked alike), and `measure --outside NAME` measures it.
  `--outside` also takes one unquantised GGUF (F32, F16 or BF16), for an edit published only that way;
  a weight that differs from the run's only by rounding to the file's type counts as unedited.
- `serve` runs the unedited model, or a trial's export, in llama-server as candor measures it: the
  run's KV cache type, chat template settings, sampling and thinking budget, with today's date as
  apps write it. A trial's export only has to be the one built from that trial's edit, and a Hub
  model's own GGUF is found from the commit the run pins and its file headers, so a machine that only
  serves needs none of the model's full-size files. llama-server puts as many layers on the GPU as
  fit and runs the rest on the CPU. Before
  serving it shows the command, the settings and how the server's own template settings render a
  first message of Hello, so a thinking model's mode is plain before you chat. The chat page can
  override both the system message and the template settings, and remembers them, so serve says what
  to set there: the measured system message, every tool off (the page offers its own, and templates
  such as Llama 3's turn any tool into a request for a function call), and the reasoning menu on
  Default (its Low, Medium and High are token budgets for thinking, not the model's effort levels), and
  its sampling settings empty, so the server's apply. It serves llama-server's
  chat page (`--port`, 8080) until Ctrl-C, and fails if the server stops by itself with an error;
  the server's log goes to `private/`.
  `--mcp FILE` gives the model the tools of the MCP servers in FILE (llama-server's
  `--mcp-servers-config`, in Cursor's format), such as a sandbox or web search: serve lists the tools
  that started and says to keep them on, and the page asks before each call. `CANDOR_MCP` set to a
  file in your shell gives every serve its tools; `CANDOR_MCP=` before a command serves without them.
  Their descriptions reach the model too, so it then no longer runs exactly as measured. With tools
  the context is 32,768 tokens rather than 16,384, as tool calls and the pages they return fill it
  (at most the model's own; `--context` sets it).
- Reading residuals needs the model in 16 bits on the GPU. `run` checks this before downloading or
  converting anything: rude-llama fits on an 18 GB Mac; Qwen3.5-9B and K2-Horizon need a pod.

A run folder holds `candor.toml`, `inspect.json`, `gguf/` (the model as an unquantised GGUF of its
own float type, the deployment quantisation, and `convert.log`), `prompts.json`, `directions.pt`
(and for a thinking model `directions_answering.pt` and `thinking/`), `adapters/`,
`measurements/`, `search/` (Optuna's journal, a record of the settings it was made with,
`progress.json` and `base-first-tokens/`), `proxy/` (`candor proxy`'s scores), `scores/` (adapters
built only to score edits: `score`'s for directions the search has none for, and a "full" edit's
own), `exports/` (the GGUFs `export` makes), `private/` and `logs/`, each result beside a
`.inputs.json` record. llama.cpp itself (release b11524: the
binaries, checked by sha256, and its converter in its own environment, pinned by
`src/candor/data/converter-constraints.txt`) is fetched once into `~/.cache/candor`, or
`CANDOR_CACHE`; `CANDOR_LLAMA_BIN` points at an existing b11524 build instead.

## On a pod

Models the Mac can't hold, and jobs over a couple of hours, run on a Runpod pod. `scripts/pod.sh`
does everything but create and delete it (its header lists every command):

1. `scripts/pod.sh args HOURS` prints the container command to create the pod with (its `entrypoint`
   and `cmd`): Runpod's own start, and a watchdog that deletes the pod HOURS from now (at most 48)
   whatever happens on this side, even if the pod's own start fails. Create it from
   `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` with port `22/tcp`, SSH on, a CUDA floor of
   12.8 and enough container disk for the run: about 7.5 bytes a parameter for the run and its
   first export, 0.6 for each more export kept, and about 10 GB of tools (about 85 GB for
   Qwen3.5-9B, 260 GB for a 27B with four exports kept). A container disk can't grow without a
   restart, which wipes it, so `candor run` warns when an export won't fit.
2. `export CANDOR_POD=root@IP:PORT` (the pod's direct SSH address), then `scripts/pod.sh setup`
   checks the watchdog is armed, copies this working tree and your Hugging Face token, installs
   candor and checks that torch and llama-server both see the GPU. On Linux torch is the CUDA 12.8
   build, as llama.cpp's release is. `scripts/pod.sh code` copies the code again after a change
   and brings the pod's packages in line with its `uv.lock`.
3. `push`, `start NAME candor ...`, `log NAME`, `stop NAME` and `pull runs/NAME` move files, run a
   command detached and bring a run's results back, leaving out its GGUFs unless named
   (`pull runs/NAME/exports/trial-37`). Paths are the same as here, under `/root/candor` on the pod;
   anything there that git doesn't ignore is replaced by the next `code`, which refuses while a
   command is running, as it would change the code under it. `wait NAME runs/RUN` waits for the
   command to end and keeps this Mac awake until it does, so nothing needs a timed `caffeinate`. It
   pulls `runs/RUN` every 30 minutes and once more at the end, and exits 0 only if the command did.
   It stops early, with an error, if the pod stops answering for about five minutes or the
   watchdog's deadline is less than 6 hours away: move the deadline and wait again.
4. `scripts/pod.sh deadline` checks the watchdog and says when it deletes the pod, and
   `deadline HOURS` moves that to HOURS from now. Everything lives on the container disk, which
   Runpod wipes when a pod is stopped or restarted and which a stopped pod's watchdog can't
   delete, so pull the results, then delete the pod, never stop it.

Qwen3.5's fast linear-attention kernels aren't installed: transformers runs Qwen3.5 only to read
residuals for short prompts, the answers come from llama.cpp, and the judge (a Qwen3) has no linear
attention. If reading Qwen3.5's residuals turns out slow, `flash-linear-attention` is the fix.

## Development

```
uv sync
uv run pytest -n 6
uv run ruff check . && uv run ruff format --check . && uv run pyright
```

`CANDOR_E2E=1 uv run pytest tests/test_e2e.py` also runs small real and tiny random models end to
end: conversion with llama.cpp's converter (including a tokenizer saved by transformers 5 and a
tiny Qwen3.5), the edit-target and tokenization checks, `init`, `run` (resuming and a settings
change), `measure`, `compare`, `search`, `score` and `export` on SmolLM2-135M, a real thinking
model's budget on Qwen3.5-0.8B, Qwen3.8-27B's `init` and the judge. The first run downloads about
3.6 GB (Qwen3.5-0.8B 1.8 GB, the judge 1.5 GB, SmolLM2-135M 0.27 GB), and llama.cpp and its
converter environment if candor hasn't fetched them yet. Nothing else runs it; STATUS says when.

The engine tests run a tiny random model in the llama-server candor uses: `CANDOR_LLAMA_BIN`'s, or
the one in candor's cache. Run candor once to install it; without it those tests are skipped.

`tools/misscheck/` is the miss-rate simulator the search's design is judged on; its README says how
to run each part and which candor commit each result used.

The one-off experiments behind the log in `results/history.md` (phase 0 on a pod, rebuilding a
heretic trial, scoring heretic's saved model) were in `spikes/`, last in commit `2bb6bba`.
