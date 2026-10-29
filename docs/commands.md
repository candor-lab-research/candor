# Commands

Every command but `inspect` and `init` takes a run folder that `init` made. Run in a terminal with no
command, `candor` opens [menus](#the-menus) that run these same commands for you.

The usual order:

```
inspect → init → run → search → measure → export → measure --exported → serve
```

| Command | What it does |
| :--- | :--- |
| [`inspect`](#inspect) | Check a model before spending anything on it |
| [`init`](#init) | Start a run: check the model and write its settings |
| [`run`](#run) | Build every stage the search needs, reusing what is built |
| [`upgrade`](#upgrade) | Bring a run's `candor.toml` up to this version's settings |
| [`search`](#search) | Find the smallest edit that leaves almost no refusals |
| [`proxy`](#proxy) | Check if a thinking model may be searched with thinking off |
| [`score`](#score) | Score one edit as the search scores its trials |
| [`measure`](#measure) | Measure the unedited model or a trial, on prompts and maths |
| [`compare`](#compare) | Compare two measurements prompt by prompt |
| [`export`](#export) | Write a trial's edit into the GGUF you run |
| [`serve`](#serve) | Chat with the unedited model or a trial's export, as measured |

Also on this page: [heretic's trials](#heretics-trials) and [what a run folder holds](#the-run-folder).
Settings that depend on the model (thinking, sampling, slots) are in [models.md](models.md).

## The menus

```
uv run candor
```

- The first screen lists each run in `runs/` and how far it has got.
- Opening a run shows its panel: the model, the stages built, the search's best trials, its
  measurements and exports. The menu below it starts on the step the run is ready for.
- **Trials** lists the trials worth measuring, fewest refused first, and any already measured or
  exported.
- Each choice runs the same command you could type, and shows it first. A choice that can take hours
  can show the `scripts/pod.sh` commands that run it on a [pod](pods.md) instead.
- Arrows or j/k move, enter chooses, Ctrl-C goes back.

## inspect

```
uv run candor inspect Qwen/Qwen3.5-9B      # a Hugging Face id or a local folder
```

- Reads only the config, the tokenizer, the file headers and the first rows of four edit targets, so
  it is quick and needs no GPU.
- It warns when those weights look quantised to 4 bits before and saved unquantised, as rude-llama's
  were: most blocks of 32 values hold 16 or fewer, where weights never quantised hold nearly all 32.
  Quantising them to 4 bits again rounds most of an edit away, so [export](#export) keeps them as Q8_0
  or F32 and the file is larger.
- For an architecture whose KV cache candor knows (Llama, Qwen3.5, K2-Horizon) it shows how much a
  token of context takes at f16 and how many tokens fit in a GiB. For a hybrid such as Qwen3.5 it
  also shows the state its linear-attention layers hold for each slot.
- It fails models candor can't edit; [models.md](models.md) lists the cases.

## init

```
uv run candor init qwen --model Qwen/Qwen3.5-9B     # writes runs/qwen/candor.toml
```

- Runs the same checks as `inspect` and refuses a model candor can't edit.
- Writes `candor.toml`, which holds every setting the run uses, each explained by a comment. Edit it
  freely: every later command checks it.
- Pins Hugging Face models and the prompt datasets to the commits that were checked. Local models are
  stored by their full path.
- A model that only loads with the Python files it ships (K2-Horizon) gets
  `trust_remote_code = true`, which runs that code at the pinned commit.
- A new run gets a [second judge](#the-second-judge) and reads KL over the 200 likeliest first
  tokens (`scoring.kl_tokens`). Runs made before keep one judge and 20.
- For a thinking model it also sets the thinking's close and budget
  ([models.md](models.md#thinking-models)).

## run

```
uv run candor run runs/qwen
```

- Builds every stage the search needs: the model as a GGUF, the prompts, the directions and the
  adapters (and for a thinking model, its thinking).
- Resumes where it stopped. Each result in the run folder records what it was built from, including
  the version of candor's code for its stage. So changing a setting or upgrading candor rebuilds what
  it affects and everything after it, and says so.
- Before downloading or converting anything, checks that the model fits in 16 bits on the GPU, which
  reading residuals needs ([models.md](models.md#what-fits-where)).

## upgrade

```
uv run candor upgrade runs/qwen
```

Adds the settings a newer candor has and a run's `candor.toml` lacks, at their defaults, and lists
them. candor never fills in a missing setting by itself.

## search

```
uv run candor search runs/qwen               # overnight on a small model
uv run candor search runs/qwen --trials 30   # stop early, to time it
```

**What it searches.** Edits of the shape heretic searches, in two kinds that are searched apart and
take turns (`search.trials_per_scope`, 300 each):

- **one layer:** every layer reads one layer's refusal direction;
- **per layer:** each layer reads its own.

**How it picks.** Each kind starts from 20 evenly spread trials. Then Optuna's TPE sampler aims at the
least KL among edits refusing at most `search.max_refusals` (2) of the 100 test prompts. For a
thinking model the 20 start points are split evenly between the
[two places directions are read](models.md#where-directions-and-kl-are-read).

**How a trial is scored.**

- Every adapter is loaded into one llama-server once, so a trial only sets strengths. It answers the
  test prompts `slots.search` at a time and the judge counts refusals.
- A trial stops once it has refused more than the cap and its kind has more than 25 fully scored
  trials. This leaves every trial the sampler draws unchanged.
- KL is read from the 100 harmless test prompts, except for a trial stopped early, which can't win.
- Nothing a model writes is kept.

**Thinking budget cap.** For a thinking model, `search.max_budget_hits` (unset) can also cap the test
prompts whose thinking reaches the budget. An edit over it doesn't count and its trial stops there,
which, unlike the refusal stop, can change later draws. Changing it carries a search on, its finished
trials judged again by the new number.

**Strength range.** Strength starts at heretic's 0.8-1.5. From a kind's 60th trial, and every 30th
after, if its best edit within the cap presses an end of that range (or none is within the cap yet:
the ceiling), that end moves out to `search.widest_strength` (0.5-3.0), once. The kind's trials carry
on, and 10 evenly spread trials start the wider range.

**The "full" edit.** `edit.row_norm = "full"` makes heretic's "full" edit: each edited weight row is
scaled back to its own length, then the change is kept at heretic's rank of 3, so rows keep their
length only roughly. It is built as heretic builds it (checked against its code; its randomised SVD's
seed moves the change by under 0.1%). On Qwen3.5-9B heretic's Trial 161 used it, and a candor search
with it did better than one with "pre" (`results/findings.md`). A "full" edit isn't linear in
strength, so it can't be one adapter per weight dialled per request: each trial gets an adapter built
at its strengths and llama-server restarts with it, which adds seconds to every trial.

**Resuming.**

- It resumes where it stopped, and refuses to carry on a search made under other settings before it
  rebuilds anything.
- The unedited model's first tokens, which KL is read against, are kept in `search/base-first-tokens/`
  for each set of inputs and machine, so every session sets its trials against the same ones. They are
  kept with the sha256 of the GGUF they were read from, hashed again only when its size or time
  changes, and read again from a GGUF with other bytes, which the same recipe quantised on another pod
  can give. Those kept before candor recorded the sha256 are read again once.
- A search carried on on another machine says so, as KL read there can differ by a few percent.

**Progress.** Every `search.report_every` trials (25; 0 for never) it shows its pace and the time
left, and for each kind: its trials, how many stopped early, its best so far (least KL within the
cap, or else fewest refused) with that trial's strengths against the kind's range, and the trials
since that best improved. The latest report is kept in `search/progress.json`. The setting is left
out of the search's record, so changing it never stops a search carrying on.

**What it ends with.**

- The fully scored trials no other beats on both refusals and KL.
- For each kind, any strength it widened to, and the least-KL trial at each count within the cap
  unless one refusing fewer has less (or, failing that, the one that refused fewest). Each is flagged
  if it presses on the edge of a range, and shown with how much more KL it has than its kind's least,
  paired by prompt, as a tie when within 2 standard errors. Each trial keeps its KL at every prompt KL
  is read on; one scored before candor kept them shows none. The ties only guide: they change none of
  the trials named.
- The commands to [measure](#measure) each on the held-out prompts, since 100 test prompts can't tell
  0 refused from 2.

The design is explained in `results/phase-2-rude-llama.md` and, for widening and the two places
directions are read, `results/phase-3-qwen3.5-9b.md`.

## proxy

For a thinking model only.

```
uv run candor proxy runs/qwen                            # may the search think off?
uv run candor proxy runs/qwen --trial 37 --trial 41      # the search's picks, scored thinking too
```

**Why.** Scoring every trial thinking costs many times more than answering greedily after an empty
thinking block (`scoring.thinking = "off"`, answers `scoring.closed_answer_tokens` long). That cheap
score is only worth having if it ranks edits as thinking does, which no published work shows and
which changes by model.

**How it checks.** On the run's own model, after a short search scored with thinking off
(`search --trials 150`; random edits mostly change nothing, so they can't show an order):

1. It takes the 3 best trials of each kind and each place directions are read, each also at 1.4
   times its strength.
2. It scores them both ways on the test prompts and compares their orders.
3. It scores the edit refusing fewest again with another seed, to show the noise.

**What it says.** One of: the cheap score holds; it holds but thinking needs stronger edits (the
search's suggestions must then also be scored thinking, at more strength); or it can't be trusted. If
it holds, the search carries on from those trials. Set `scoring.thinking = "off"` only when it holds.
Edits are chosen on numbers measured thinking either way.

**When it refuses.** A model whose chat template can't turn thinking off is never used answering
after an empty block, so candor refuses to score it that way, which `proxy` needs. So too where
turning thinking off changes more than the end of the prompt, as Qwen3.8's does at every reasoning
effort but medium.

**Good to know.**

- Its scores, the unedited model's too, are kept in `proxy/` as each is done, so a stopped check
  carries on without scoring again what it finished.
- `--trial`, repeated, scores the trials named instead (the search's best, once it may score thinking
  off), each also at 1.4 times its strength. They are kept apart in `proxy/trials.json`, with counts
  but no verdict. Naming other trials starts that afresh.

## score

```
uv run candor score runs/rude --heretic STUDY.jsonl --trial 170   # where it would sit in the search
uv run candor score runs/rude --trial 37                          # one of the search's own trials
```

- Scores one edit exactly as the search scores its trials, without adding it to the search.
- A [heretic trial](#heretics-trials) is scored as rebuilt, and at the nearest whole layer, as the
  search could draw it. It shows which of the search's best trials beat it on both refusals and KL,
  and whether the search could have drawn it at all.
- The search's best trial is scored again beside it, as a check. `--trial N` alone scores one of the
  search's trials.
- A trial the search scored in full should repeat its score within noise (a refusal either way, and
  a KL that ties with the search's paired by prompt, as trials are compared), or `score` says how it
  differs. The unedited model is scored again only when no such trial is among those scored.
- Like the search, it is refused before anything is rebuilt when the search was made from other
  inputs.

## measure

```
uv run candor measure runs/qwen                              # the unedited model
uv run candor measure runs/qwen --trial 37                   # a trial, as adapters
uv run candor measure runs/qwen --trial 37 --exported        # the exact file export wrote
uv run candor measure runs/qwen --heretic STUDY.jsonl --trial 161
uv run candor measure runs/qwen --outside heretic-saved      # another tool's edit
uv run candor measure runs/qwen --seed 1                     # other draws: the sampling noise
```

**What it measures.** It answers the held-out prompts and GSM8K's 1,319 maths questions with the
deployed GGUF in llama-server.

- The held-out harmful prompts, which the search never sees, come from StrongREJECT without the rows
  it took from AdvBench. It is gated: accept its terms on its Hugging Face page, then
  `hf auth login`.
- Qwen3Guard-Gen-0.6B (pinned) labels each held-out answer as a refusal or not, with keyword counts
  beside it.
- The first-token probabilities on the harmless prompts give KL divergence. For a thinking model,
  see [where KL is read](models.md#where-directions-and-kl-are-read).

**What it keeps.** One number per prompt, in `measurements/NAME/`. Answers and the judge's replies go
only to `private/`. Each set's results also hold the sha256 of the GGUF that answered it
(`gguf_sha256`), read once before answering, so a file exported again can be matched to the
measurement; `measure` shows its first 16 digits.

**Options.**

| Option | What it does |
| :--- | :--- |
| `--trial N` | A trial of the search, or with `--heretic`, of heretic's |
| `--exported` | Measure the trial's exported GGUF, not its adapters |
| `--outside NAME` | A model edited elsewhere, by the name `export --outside` gave it |
| `--limit N` | Only the first N prompts of each set, and an estimate of the full run's time |
| `--name NAME` | Name the measurement, but never as candor names its own (`base`, `trial-N`, `heretic-N`) |
| `--slots N` | Override `slots.measure` |
| `--seed N` | Sample with other draws ([models.md](models.md#sampling)) |
| `--set SET` | Only `harmful_held_out`, `harmless_held_out` or `maths`; repeat it for more. Kept as a full measurement of the same name keeps it, so `measure` without it later answers only the rest |

A stopped measurement resumes from the last answer it saved, and the Mac is kept awake while it runs.

### The second judge

`measure.second_judge` (Qwen3Guard-Gen-8B for a new run, pinned) judges each held-out answer again,
once the first judge is done and freed, so the harmful-refusal bar can be held under both. The search
keeps to the first judge.

- Its labels, record and replies go in files of their own (`SET.second_judge.json`), so adding it to
  a finished measurement runs only it.
- With it set, a measurement isn't finished until it has judged.
- The 8B takes about 16 GB at 16 bits, so it judges on a [pod](pods.md). `measure` and `search` stop
  before loading any judge the GPU can't hold, such as the 8B on an 18 GB Mac. They size one already
  downloaded from the Hub's cache, so offline runs still judge.

## compare

```
uv run candor compare runs/rude base heretic-170
```

Pairs two measurements prompt by prompt and shows, for each count:

- the change, with a 95% interval (Bonett and Price's for paired counts; for KL, a studentised
  bootstrap over the prompts);
- how many prompts went each way, and the exact McNemar p-value;
- the slots and seeds, where they differ.

**Two judges.** The second judge's harmful and harmless refusals get rows of their own beside the
first judge's when both measurements have them. Labels by two different judges, first or second, are
never paired.

**Noise.** Measuring the same edit twice with different `--slots` shows what batching noise alone
does.

**What it refuses or flags.**

- Two measurements made by different versions of candor's measuring. Those made before 2026-10-19
  drew a thinking model's answer with its thinking's seed, and misread maths answers written
  `\boxed{18}` or `-$5`. Measuring again under the name of one made by older code keeps the old one
  as `NAME-measuring-N`.
- It shows the GPUs, and each judge's device, where they differ.
- It reads no KL between measurements made on different backends (Metal against CUDA, or a
  `CANDOR_LLAMA_BIN` build), where KL moves by a few percent whatever the edit does.

## export

```
uv run candor export runs/qwen --trial 37
uv run candor export runs/qwen --outside ~/heretic-save --name heretic-saved   # another tool's edit
```

**What it writes.** A trial's edit, as its adapters apply it, into a copy of the run's unquantised
GGUF. Each edited weight is written as llama.cpp's converter writes it (checked to give the same file
as editing the safetensors and converting them). Then it quantises the file as the run deploys.

**Why it checks every edited weight.** Quantising can round part of a small edit away: on
rude-llama, whose weights were 4-bit before, Q4_K_M kept 55-98% of trial 535's change and its
refusals came back. Writing it in 16 bits can too, as an edit under half a 16-bit step rounds back to
the weight. So export measures how much of each edited weight's change the quantised file holds:

1. A weight that kept less than 98% is written again unrounded, in F32, and quantised as Q8_0 from
   that (straight away for a run deploying Q8_0 or finer).
2. Where Q8_0 still falls short, it is left in F32.
3. If one still falls short, the file is refused.

Every other tensor keeps its plain type. The file's name still gives the run's quantisation, so
export says how many tensors it holds as Q8_0 or F32 instead, and its record lists them.

**What it keeps.** Only `exports/EDIT/RUN-EDIT-QUANTISATION.gguf`. `measure --exported` measures that
exact file, and `compare` sets it against the unedited model and against the trial measured with
adapters.

**Another tool's edit.** `export --outside FOLDER --name NAME` ships a copy of the model edited
elsewhere, such as heretic's own save of one of its trials, the same way: the run's model with each
edited weight taken from the copy, so it is written, quantised and checked alike. `measure --outside
NAME` measures it. `--outside` also takes one unquantised GGUF (F32, F16 or BF16), for an edit
published only that way. A weight that differs from the run's only by rounding to the file's type
counts as unedited. Only candor's edit targets are taken: building the file, export names the tensors
outside them that differ from the run's model, such as an edited `embed_tokens`, and leaves those as
the run's.

## serve

```
uv run candor serve runs/qwen                 # the unedited model
uv run candor serve runs/qwen --trial 37      # a trial's export
```

**As measured.** It runs the model in llama-server as candor measures it: the run's KV cache type,
chat template settings, sampling and thinking budget, with today's date as apps write it.

- A trial's export only has to be the one built from that trial's edit. A Hub model's own GGUF is
  found from the commit the run pins and its file headers, so a machine that only serves needs none
  of the model's full-size files.
- llama-server puts as many layers on the GPU as fit and runs the rest on the CPU.
- Before serving it shows the command, the settings, and how the server's own template settings
  render a first message of Hello, so a thinking model's mode is plain before you chat.

**The chat page.** It serves llama-server's chat page (`--port`, 8080) until Ctrl-C, and fails if the
server stops by itself with an error. The server's log goes to `private/`. The page can override both
the system message and the template settings, and remembers them, so `serve` says what to set there:

- the measured system message;
- every tool off (the page offers its own, and templates such as Llama 3's turn any tool into a
  request for a function call);
- the reasoning menu on Default (its Low, Medium and High are token budgets for thinking, not the
  model's effort levels);
- its sampling settings empty, so the server's apply.

**Tools.** `--mcp FILE` gives the model the tools of the MCP servers in FILE (llama-server's
`--mcp-servers-config`, in Cursor's format), such as a sandbox or web search.

- `serve` lists the tools that started and says to keep them on. The page asks before each call.
- `CANDOR_MCP` set to a file in your shell gives every serve its tools; `CANDOR_MCP=` before a command
  serves without them.
- Their descriptions reach the model too, so it then no longer runs exactly as measured.
- With tools the context is 32,768 tokens rather than 16,384, as tool calls and the pages they return
  fill it (at most the model's own; `--context` sets it).

## heretic's trials

`--heretic STUDY.jsonl --trial N`, for `measure`, `score`, `export` and `serve`, rebuilds one of
heretic's trials so candor can measure it like its own.

- It is rebuilt with the run's own directions, read with the run's system prompt, not heretic's. One
  that read its directions after the run's own `think_close` is rebuilt with those read where the
  answer starts.
- candor reads every layer before the model's final norm, and heretic its last one after it, so a
  per-layer trial's last layer is rebuilt with a direction heretic didn't read.
- candor finds a study's finished trials by the refusals heretic 1.x records for each. heretic 2.x
  records none, so its studies can't be read.

## The run folder

```
runs/NAME/
├── candor.toml        every setting the run uses, each explained
├── inspect.json
├── gguf/              the model as an unquantised GGUF of its own float type,
│                      the deployment quantisation, and convert.log
├── prompts.json
├── directions.pt      (a thinking model also has directions_answering.pt and thinking/)
├── adapters/
├── search/            Optuna's journal, the settings it was made with,
│                      progress.json and base-first-tokens/
├── proxy/             candor proxy's scores
├── scores/            adapters built only to score edits: score's for directions
│                      the search has none for, and a "full" edit's own
├── measurements/
├── exports/           the GGUFs export makes
├── private/           answers, the judge's replies and server logs; never printed
└── logs/
```

Each result sits beside a `.inputs.json` record of what it was built from.
