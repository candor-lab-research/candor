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
   request, so nothing is re-converted between trials.
4. Merge the chosen edit straight into the safetensors files, copying every other file untouched,
   then convert and quantise it to the GGUF you run, and measure that exact file.

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
uv run candor export runs/rude --trial 37                      # merge it into the GGUF you run
uv run candor measure runs/rude --trial 37 --exported          # measure that exact file
uv run candor measure runs/rude --seed 1                       # sampled again with other draws: the noise
uv run candor export runs/rude --outside ~/heretic-save --name heretic-saved   # another tool's edit
uv run candor measure runs/rude --outside heretic-saved        # measured like candor's own
uv run candor serve runs/rude --trial 37                       # chat with it in llama-server
```

- `inspect` reads only the config, tokenizer and file headers, so it is quick and needs no GPU.
- A model that thinks before it answers (its chat template opens the thinking, as Qwen3.5's does) is
  scored thinking, as you run it. `init` finds the text that closes its thinking and gives it a
  budget of 8192 tokens (`deployment.thinking_budget`): past it, llama-server makes the model stop
  thinking and answer. Only the answer is judged, and each answer's thinking length is kept as a
  number. The search can score at a shorter budget (`scoring.thinking_budget`) and the edits it
  suggests are measured at the deployment's.
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
  either way. Its scores are kept in `proxy/`, so a stopped check carries on. `--trial`, repeated,
  scores the trials named instead (the search's best, once it may score thinking off), each also at
  1.4 times its strength, kept apart in `proxy/trials.json`, with counts but no verdict; naming other
  trials starts that afresh.
- `edit.row_norm = "full"` makes heretic's "full" edit: each edited weight row keeps its own length, at
  heretic's rank of 3, built as heretic builds it (checked against its code; its randomised SVD's seed
  moves the result by about 1e-5). On Qwen3.5-9B
  heretic's Trial 161 used it and candor's "pre" edits stalled near 70 of 100 refusals. A "full"
  edit isn't linear in strength, so it can't be one adapter per weight dialled per request: each
  trial gets an adapter built at its strengths and llama-server is restarted with it, which adds
  seconds to every trial. `measure --heretic` and `score --heretic` rebuild a heretic trial that read
  its directions after the run's own `think_close` with the directions read where the answer starts,
  this run's own: read with the run's system prompt, not heretic's.
- A sampled run samples on the GPU (`--backend-sampling`), many times faster for a large
  vocabulary. llama-server can't do that for a request with a thinking budget, so candor holds the
  model to its budget itself, token for token as the server's own budget does (checked against it).
  Draws for a seed differ from sampling on the CPU, so a measurement made before 2026-10-14 pairs
  with a later one prompt by prompt, not draw by draw.
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
  still compared prompt by prompt. `measure --seed N` samples with other draws, to see how much
  sampling alone moves the counts.
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
  the full run's time; a stopped measurement resumes; the Mac is kept awake.
- `compare A B` pairs two measurements prompt by prompt: each count, how many prompts went each
  way, and the exact McNemar p-value. Measuring the same edit twice with different `--slots` shows
  what batching noise alone does.
- `search` tries edits of the shape heretic searches, in two kinds searched apart that take turns:
  every layer reading one layer's refusal direction, and each layer reading its own
  (`search.trials_per_scope`, 300 each). Each kind starts from 20 evenly spread trials, then
  Optuna's TPE sampler aims at the least KL among edits refusing at most `search.max_refusals` (2)
  of the 100 test prompts. Every adapter is loaded into one llama-server once, so a trial only sets
  strengths: it answers the test prompts eight at a time and the judge counts refusals, stopping
  once the trial has refused more than the cap and more than 25 fully scored trials of its kind,
  which leaves every trial the sampler draws unchanged; KL is read from the 100 harmless test
  prompts. Nothing a model writes is kept. Strength starts at heretic's 0.8-1.5. From a kind's
  60th trial, and every 30th after, if its best edit within the cap presses an end of that range
  (or none is within the cap yet: the ceiling), that end moves out to `search.widest_strength`
  (0.5-3.0), once; the kind's trials carry on and 10 evenly spread trials start the wider range.
  For a thinking model the 20 start points are split evenly between the two places directions are
  read. It resumes where it
  stopped, refuses to carry on a search made under other settings, and `--trials N` stops early to
  time it. It ends with the fully scored trials no other beats on both refusals and KL; for each
  kind, any strength it widened to, and the least-KL trial at each count within the cap unless one
  refusing fewer has less (or, failing that, the one that refused fewest), each flagged if it
  presses on the edge of a range; and the commands to measure each on the held-out prompts, since
  100 test prompts can't tell 0 refused from 2. The results files explain the design:
  `results/phase-2-rude-llama.md` and, for widening and the two places, `results/phase-3-qwen3.5-9b.md`.
  `candor upgrade` keeps a run whose search was made before strength could widen from widening it.
- `score` scores one edit exactly as the search scores its trials, without adding it to the search.
  A heretic trial is scored as rebuilt, and at the nearest whole layer as the search could draw it;
  it shows which of the search's best trials beat it on both refusals and KL, and whether the search
  could have drawn it at all. The search's best trial is scored again beside it and should repeat its
  score, which shows the two can be compared. `--trial N` alone scores one of the search's trials.
- `export` merges a trial's edit into a copy of the model's weights as its adapters apply it, then
  converts and quantises the copy as the run deploys, checking the edited weights reach the GGUF
  unchanged. Quantising can round part of a small edit away (on rude-llama, whose weights were 4-bit
  before, Q4_K_M kept 55-98% of trial 535's and its refusals came back), so it measures how much of
  each edited weight's change the quantised file holds, quantises any weight that kept less than 98%
  as Q8_0 instead, and refuses the file if one still falls short. Only
  `exports/EDIT/RUN-EDIT-QUANTISATION.gguf` is kept. `measure --exported` measures that exact file,
  and `compare` sets it against the unedited model and against the trial measured with adapters.
  `export --outside FOLDER --name NAME` ships a copy of the model edited elsewhere, such as heretic's
  own save of one of its trials, the same way (the run's model with each edited weight taken from
  the copy, so it is converted, quantised and checked alike), and `measure --outside NAME` measures it.
- `serve` runs the unedited model, or a trial's current export, in llama-server as candor measures
  it: the run's KV cache type, chat template settings, sampling and thinking budget, with today's
  date as apps write it. Before
  serving it shows the command, the settings and how the server's own template settings render a
  first message of Hello, so a thinking model's mode is plain before you chat. The chat page can
  override both the system message and the template settings, and remembers them, so serve says what
  to set there: the measured system message, every tool off (the page offers its own, and templates
  such as Llama 3's turn any tool into a request for a function call), and the reasoning menu on
  Default (its Low, Medium and High are token budgets for thinking, not the model's effort levels), and
  its sampling settings empty, so the server's apply. It serves llama-server's
  chat page (`--port`, 8080) until Ctrl-C; the server's log goes to `private/`.
- Reading residuals needs the model in 16 bits on the GPU. `run` checks this before downloading or
  converting anything: rude-llama fits on an 18 GB Mac; Qwen3.5-9B and K2-Horizon need a pod.

A run folder holds `candor.toml`, `inspect.json`, `gguf/` (the model as an unquantised GGUF of its
own float type, the deployment quantisation, and `convert.log`), `prompts.json`, `directions.pt`
(and for a thinking model `directions_answering.pt` and `thinking/`), `adapters/`,
`measurements/`, `search/` (Optuna's journal and a record of the settings it was made with), `scores/` (adapters `score` builds for directions the search has none for), `exports/` (the
GGUFs `export` makes), `private/` and
`logs/`, each result beside a `.inputs.json` record. llama.cpp itself (release b11524: the
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
   12.8 and enough container disk for the run (about 50 GB for rude-llama, 100 GB for Qwen3.5-9B).
2. `export CANDOR_POD=root@IP:PORT` (the pod's direct SSH address), then `scripts/pod.sh setup`
   checks the watchdog is armed, copies this working tree and your Hugging Face token, installs
   candor and checks that torch and llama-server both see the GPU. On Linux torch is the CUDA 12.8
   build, as llama.cpp's release is. `scripts/pod.sh code` copies the code again after a change.
3. `push`, `start NAME candor ...`, `log NAME`, `stop NAME` and `pull runs/NAME` move files, run a
   command detached and bring a run's results back, leaving out its GGUFs unless named
   (`pull runs/NAME/exports/trial-37`). Paths are the same as here, under `/root/candor` on the pod;
   anything there that git doesn't ignore is replaced by the next `code`.
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
uv run pytest
uv run ruff check . && uv run ruff format --check . && uv run pyright
```

`CANDOR_E2E=1 uv run pytest tests/test_e2e.py` also runs small real and tiny random models end to
end: conversion with llama.cpp's converter (including a tokenizer saved by transformers 5 and a
tiny Qwen3.5), the edit-target and tokenization checks, and `candor init` and `candor run`,
including resuming and a settings change. The first run downloads about 270 MB of SmolLM2-135M, and
llama.cpp and its converter environment if candor hasn't fetched them yet.

The engine tests run a tiny random model in the llama-server candor uses: `CANDOR_LLAMA_BIN`'s, or
the one in candor's cache. Run candor once to install it; without it those tests are skipped.

The one-off experiments behind the log in `results/history.md` (phase 0 on a pod, rebuilding a
heretic trial, scoring heretic's saved model) were in `spikes/`, last in commit `2bb6bba`.
