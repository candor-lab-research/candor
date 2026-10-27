# misscheck: the search's miss-rate simulator

candor's search is judged by how often it misses the best edit, which a real model can't tell us.
These scripts build synthetic score landscapes over candor's search settings, calibrated to real
searches' numbers, and run candor's search (or a candidate design) on them many times against a
known true best. Numbers only: nothing here reads prompts or answers or starts a model.

Only source and small numeric inputs are kept here. Run outputs (`cache/`, `sims/`, `runs/`, logs and
tables) are not; the results files hold the numbers.

## Inputs

- `candor_search_rude.csv`, `heretic_search_rude.csv`: rude-llama's two 200-trial searches as
  numbers (settings, refusals, KL, seconds), heretic's on its own scale.
- `harness/fitted.json`, `harness/fitted_mapping.json`: the landscape shape `harness/fit.py` fitted
  to them, and the map between the two scales. `misscheck/fitted.json` is the same file.
- `misscheck/k2/trials.csv`: K2-Horizon-7B's search trials as numbers, taken from its Optuna journal
  by `misscheck/k2/extract.py`.

## Parts

| Folder | What it does | Main scripts |
| --- | --- | --- |
| `space/` | Why the phase 2 search missed: how much of each kernel parametrisation's prior lands near good shapes, and TPE replays of the real run | `volume.py`, `compare.py`, `replay.py`, `sources.py` |
| `sim/` | The first version of the test: GP surrogates fitted to the two searches, search designs run on them, stop-rule and repeat analyses | `optimum.py`, `designs.py`, `traces.py`, `report.py` |
| `cheaper/` | Stopping trials early: which stop rules leave TPE's draws unchanged, and the time they save | `invariance.py`, `replay.py`, `replay2.py`, `budget2.py` |
| `harness/` | The landscape model fitted to both searches, its variants (base, per-layer, outside, needle, rand1-rand6), the strategies tested and their evaluation | `fit.py`, `variants.py`, `evaluate.py`, `analyze.py`, `replay_check.py` |
| `misscheck/` | Scenario landscapes built on the harness (strength stretched, two places to read directions), driven by candor's own search code; the stall-stop replay, the budget-limit check and the range ends | `rerun.sh`, `summary.py`, `stall_sim.py`, `budget_run.sh`, `budget_report.py`, `k2/fit.py`, `ends_run.sh` |

`misscheck/landscape.py`, `optimum.py` and `variants.py` are unchanged copies of the harness's. Each
script's docstring says what it does and how to call it.

## Which candor each result used

| Result | Code | candor commit |
| --- | --- | --- |
| Phase 2, the search fix (`results/phase-2-rude-llama.md`): why it missed, GP and early-stop studies | `space/`, `sim/`, `cheaper/` | none imported |
| Phase 2, the design comparison and the rand1-rand6 check | `harness/` | 742a69d (its search code is the same from 72777c6 to d46efb3); `replay_check.py` draws all 200 real trials with it |
| Phase 3 step 3, the miss rate with the thinking-model changes (`results/phase-3-qwen3.5-9b.md`) | `misscheck/rerun.sh` | 49fc36e |
| Phase 4 step 5, the stall stop (`results/phase-4-k2-horizon-7b.md`) | `misscheck/stall_sim.py`, on phase 3's runs | 49fc36e |
| Phase 4 step 5, the budget limit | `misscheck/budget_run.sh`, `misscheck/k2/` | cb0fa99 |
| The range ends, 2026-10-27 (`results/evidence-2026-10-27/ranges.md`) | `misscheck/ends_run.sh` | d2043bc |

## Running

Run everything with `uv run` from inside the repo: numpy, torch, optuna and pandas come with candor.
The GP samplers (`sim/designs.py`'s GP designs, the harness's `gp*` strategies) also need scipy,
which candor doesn't depend on, so run those with `uv run --with scipy python ...`.

Scripts that import candor need its search code at the commit above on `PYTHONPATH`. The two shell
scripts fetch it themselves with `git archive`.

```sh
# Phase 2 harness, from tools/misscheck/harness
mkdir -p candor-742a69d && git archive 742a69d src | tar -x -C candor-742a69d
export PYTHONPATH=candor-742a69d/src:.
uv run python replay_check.py        # draws candor's 200 real trials exactly, in seconds
uv run python fit.py                 # refits, rewriting fitted.json and fitted_mapping.json
uv run python variants.py            # each landscape's true best into cache/ (about 9 min for base)
uv run --with scipy python evaluate.py --seeds 30 --runs-only
uv run python evaluate.py --report

# Phases 3 and 4, from tools/misscheck/misscheck
./rerun.sh                           # phase 3 step 3, about 1.5 h on 12 cores; tables in summary.txt
PYTHONPATH=candor-head/src:. uv run python stall_sim.py   # after rerun.sh
./budget_run.sh                      # the budget limit, about 6 h on 10 cores; tables in budget-tables.txt
uv run python k2/fit.py              # K2's budget-hit model, in seconds
./ends_run.sh                        # the range ends, about 1.5 h on 10 cores; tables in ends-tables.txt

# sim/, from tools/misscheck/sim
uv run python optimum.py             # each surrogate's best, into optima.json
uv run python designs.py today,split_constrained union,heretic 400 20 runs.jsonl
uv run python report.py runs.jsonl
```

`space/` and `cheaper/` scripts run from anywhere, for example `uv run python space/volume.py`.
`sim/below_set.py` reads the CSV from the working directory, so run it from `tools/misscheck/`.

`cache/` is keyed by a landscape's settings, not by the code: clear it after changing
`landscape.py` or `optimum.py`.
