# candor full review: fix plan (2026-10-18)

A full review of the repository by 97 agents (10 finders, a coverage critic and 3 gap finders, then
adversarial verification of every finding and a synthesis), read-only and numbers only. The plan's
order of work and progress are in STATUS; this is the plan as the review wrote it.

Corrected since: Qwen3.8-27B was affected by F33 after all. At its default effort its thinking-off
prompt also drops a system message, so candor took it for a model that doesn't think; step 2 fixed
that (`findings.md`, Thinking models).

97 agents. 158 found, 170 after gap finders, 161 confirmed, 1 split, 8 refuted.

## Summary

Overall the repo is healthy, but it has real gaps to close before the 27B phase. 162 findings survived verification: 161 confirmed and 1 split (F95). After merging 30 duplicate groups and dropping three style-only items (F46's refactor, F31's constant rename, F88's constant move), they make 24 work packages plus 11 method questions. Verifiers lowered nearly every "high" to medium. Only F33 stays high. Nothing shows a shipped result is wrong: the McNemar code is exact, edits and references share seeds so the paired comparisons stand, and the K2 trial 205 release holds.

What matters most, in order:

(1) F33. A model that opens its own thinking (Qwen3-32B/14B hybrid, GLM-4.5, SmolLM3) passes inspect as "ok". It would then be searched and measured with its thinking judged as its answer, silently, on pod time. The planned Qwen3.8-27B opens <think> itself, so it is not affected, but the obvious alternates are.

(2) F52. compare can pair a stopped re-measure's new answers with the previous model's judge labels, or labels counted under another COUNTING.

(3) F51. llama.cpp b11524 restarts the RNG on every request, so each answer replays its thinking's random draws. This touches every thinking-on number. The effect is likely small and mostly shared by both arms, so it should be fixed in one measurements bump before any 27B reference is measured. F34 must land first, or that bump makes every export stale. A search VERSION bump would also block `measure/export --trial` on the K2 and Qwen searches, so trial_edit needs its own version.

(4) Statistics. compare prints counts and a two-sided p, with no intervals: trial 205's maths change is -0.99 pp, 95% CI -2.55 to +0.58. The signed-off rules read p >= 0.05 as "no worse". Print intervals now; set the margins at the phase-end discussion.

(5) findings.md carries wrong or overreaching numbers that the phase-end discussion will rely on (F94, F96, F102, F103, F107, F97).

(6) 27B cost and safety:
- Stopped trials wait for their in-flight thinkings and read KL nobody uses.
- A measurement set is saved only when it is complete: 7-10 h of maths at 27B, estimated.
- Each export rewrites the model twice while the GPU idles.
- Thinking runs default to 8 slots.
- pod.sh wait neither watches the 48 h deadline nor pulls results.

Order of work: do the 0.05 s health poll (F133, two lines) and pytest-xdist (F159) first. They cut the full test run from about 13 min to about 1.5 min, which speeds up every later package. F78 is superseded by F122. F95's own proposal (judge the proxy by Spearman among near-best edits) was rejected by both verifiers; only its doc correction is kept, and its concern moves into the proxy method question.

## Work packages

### 1. Refuse models that open their own thinking (small)

Ids: F33, F26

**Why.** Wrong results. With Qwen3's hybrid template, enable_thinking=False writes a whole <think></think> block, so think_close is None. Inspect then says Thinking 'ok' and init writes no think_close. Run.thinking only raises when the prompt itself ends in an open tag, so every answer starts with the model's own thinking, gets cut at 100/256 tokens, and is judged and keyword-counted as the answer. KL is read where every variant just predicts '<think>'. This was reproduced on the cached Qwen3-30B-A3B tokenizer. Inspect exists to stop this before money is spent, and Qwen3-32B/14B, GLM-4.5 and SmolLM3 share the pattern. No test runs a real thinking template (F26), which is how the gap went unnoticed.

**Fix and risk.** The regex must only fire when think_close is None, because Qwen3.5/K2 templates name <think> and do open it. Test it against the known non-thinking families (Llama, SmolLM2, Qwen2.5, Mistral) so none of them is refused. The Run.thinking guard is optional: one extra /apply-template call per start, and only needed for a hand-edited candor.toml. The no-switch case (Magistral [THINK], R1-distill) is a known gap and is listed under method questions. Fix: in cli._tokenizer_rows, make the Thinking row 'fail' when think_close is None and either turns_thinking_off is True or the template text names a think tag. Change the OPENS_ITS_OWN test to expect 'fail' and add a test for a template that names a tag. For F26, add one CANDOR_E2E-gated test on the smallest real model in the 27B template family: init writes the expected think_close and budget, the model converts, and one generate with a small budget forces the close.

### 2. compare and load never pair labels or sets from different answers (small)

Ids: F52, F59, F64, F29, F60

**Why.** Wrong results. load() merges {set}.judged.json with no check that the labels belong to the loaded answers, and compare checks only the paired digest and the judge name and revision. Two paths give silently wrong rows: a re-measure under the same default name (for example after a re-export) stopped between answering and judging, or a COUNTING bump that was never re-run. Both were reproduced on the tiny run. A stop between sets can also mix two edits' sets in one measurement. compare drives the ship decision. In the same area: compare shows a Sampling seed row but no Slots row (F59), unlabelled verdicts have no test (F64), and the reported 'median' takes the upper middle value for the even set sizes 288, 300 and 1,319 (F29 = F60).

**Fix and risk.** Deleting a todo set's old files before answering means a wrong --name destroys data at the start of the run rather than at commit, so land F22's --name guard (CLI package) at the same time. A stricter load() could report an existing measurement as unfinished if its judged record is stale; candor measure then re-judges without answering again. Fix: in Measurement.measure, before answering, remove each todo set's {set}.json and {set}.judged.json and their records. In load(), require run.current() on each held-out set's judged record ({answers digest, judge, COUNTING}), otherwise raise 'run candor measure to finish it'. Add a Slots row when a['maths']['slots'] != b['maths']['slots']. Add a unit test on hand-built results with one None label. Use statistics.median in measure.summary and cli._thinking_summary.

### 3. Give the answer its own random draws, in one measurements bump before 27B references (medium)

Ids: F34, F12, F21, F51, F61

**Why.** Wrong results. _answer_one sends one seed to the thinking, the character-finishing and the answer requests. llama-server b11524 seeds a fresh mt19937 per request (server-context.cpp:1916, llama-sampler.cpp:1405-1415), so answer token k reuses the uniform that picked thinking token k. With the same seed, 300 of 300 steps fell in the same CDF slice; with another seed, 45 of 300. That differs from real use, and test_engine.py:663 locks the behaviour in. The effect is unmeasured and probably small, but it affects every thinking-on number. F61's maths parser fix also changes measured results ('Answer: \\boxed{18}' is scored wrong, '-$5' reads as 5), so it should ride the same bump. Two prerequisites. F34: exports and scores records carry the measurements code version, so the bump would make every export stale. F21 plus a split version: trial_edit compares the search VERSION (search.py:720, 745), so the bump would block `measure --trial` and `export --trial` on the K2 and Qwen searches. F12 replaces the bare-digest pin so the re-pin shows what moved.

**Fix and risk.** Every existing measurement and search stops pairing draw for draw with new ones, so any reference used later must be measured again. Do this before 27B references, and include the prefix-cache change in the same bump if its pilot is ready, so references are measured again only once. Sizing the old bias needs pod time and the user's approval. Steps, in order: (1) set DEPENDS['exports'] and DEPENDS['scores'] to ('gguf','prompts.json','directions.pt','adapters') and re-pin records_not_thinking.json; (2) replace the seeded-search digest pin with a full JSON record file, as test_records.py does; (3) give trial_edit its own edit-rebuild version (a missing value in old records counts as 1) instead of the scoring VERSION, and use recorded_edit for --exported; (4) derive the answer seed as int.from_bytes(sha256(f'{seed}:answer').digest()[:4]) % RANDOM_SEED and the finishing seed from f'{seed}:finish', keeping the thinking on the prompt's seed; bump STAGES['measurements'] and search VERSION; assert in test_engine.py:663 that the seeds differ and are deterministic; add a CDF-slice test on the tiny GGUF; (5) strip '\\boxed{' and '$' before STATED/NUMBER and add the two cases to the parametrized test.

### 4. compare reports paired intervals, not just a p (small)

Ids: F116, F56, F117

**Why.** Statistics. compare prints counts, the discordant split and an exact two-sided McNemar p, with no effect size or interval. For trial 205, maths is 1222 -> 1209 (p 0.255), a -0.99 pp change with a 95% CI of -2.55 to +0.58 pp, so a 2.5 pp loss is not excluded. mcnemar(0, 5) = 0.0625 two-sided against 0.031 one-sided. KL rows are bare means, though held-out KL reversed the search's ranking of K2's finalists. The phase-end discussion that sets the 27B decision rule needs these numbers, and printing them changes no decision.

**Fix and risk.** This is reporting only. The decision rule itself stays as signed off until the user agrees a new one (method questions). Fix: per row, print B - A in pp with a paired 95% interval. Use Wald for maths. Use Newcombe for the 300-prompt sets, where Wald gives a zero-width interval at 0/0 discordant prompts. Add the exact one-sided McNemar p in the direction of harm and label the existing p as two-sided. For the KL rows, add a percentile bootstrap 95% interval over prompts (fixed RNG seed) from the per-prompt divergences _kl already computes. A prompt bootstrap does not capture batching noise; say so in the footnote.

### 5. Correct findings.md and the results files before the phase-end discussion (medium)

Ids: F94, F95, F96, F102, F103, F107, F97, F67, F99, F100, F148, F58

**Why.** Wrong maths and claims in the documents the 27B method decision will read:
- F94: 'think-start directions refused 71-100 either way on 12 edits' holds only for the 6 per-layer edits; the one-layer ones refused 14-99 off and 17-100 as used. It also leaves out K2, where every finalist reads at the think start.
- F96: '11/100 to 28/288' is a fall (11% to 9.7%), and the K2 half compares different prompts, judges and modes.
- F95 (doc part only): the cap counts leave out step 6. The fifth slip is 402x1.4, not 175x1.4. At the edits' own strength, 3 of 7 left the cap.
- F102: the batched-noise claim conflicts with rude-llama's measured floor (KL 0.0000, 0 of 588 held-out answers changed) and has no numbers home.
- F103: the GPU-failure hypothesis blames llama-server, which restarts on every 'full' trial.
- F107: wrong figures (8-41 should be 5-41; 41.5 GB should be 41,472 MiB), plus a contradiction on maths lengthening.
- F97, F67 (doc part): pre was at 50/100 after 69 trials and still falling, not 'stalled near 70'. Rank 3 does not keep row lengths.
- F99, F100: the dropped stall stop, the built budget limit and the rejected longest-first idea still read as open ideas.
- F148 (doc part): the Metal KL finding is confounded with a Mac re-quantised Q4_K_M, which differs in about 1% of K-quant blocks.
- F58 (sentence): the per-layer x1.4 claim is wrong, since attention was already at its ceiling.

**Fix and risk.** Take numbers only from the results files and `candor compare` output, never from answer text. Do not write new causal claims as fact: the K2 empty-thinking explanation goes in as a Hypothesis with a test. Put numbers that have no home (the Metal/CUDA KL reads, the 10.6 ms step time, the pre-search counts from evidence-2026-10-14/data-report.md) into the phase results file and point to them. Dropped items shrink to one [Dropped DATE: reason] line, as CLAUDE.md requires.

### 6. Thinking runs measure and sample at the search's slot count by default (small)

Ids: F47, F130, F105

**Why.** Wasted GPU time. slots.measure and slots.thinking default to 8, while the search defaults to 32 for a thinking model. At low slot counts K2 ran 213 tokens/s, against 425-1,107 at 28, and its 400-prompt thinking stage took about 30 min at 8. CLAUDE.md works around this with a manual '--slots 32' rule, which is already wrong for K2 (all -slots-28). Forgetting it already cost a 55 min restart in phase 3. compare does not catch a slots mismatch.

**Fix and risk.** Leave the MEASURE_SLOTS and THINKING_SLOTS constants unchanged, so old runs and measurement names stay as they are. Only init changes, writing Slots(thinking=32, measure=32) when think_close is set. Don't size slots from GPU memory at init, which runs on the Mac: the existing KV-fit error already names the setting to lower (28 on an A40). Set runs/qwen measure = 32 to match its -slots-32 references. Rewrite the CLAUDE.md slots bullet ('use the reference's -slots-N, pass no flag, check -np once running'); it is the user's instruction file, so propose the wording. Move the 'answers are long' comment onto THINKING_MODEL_SEARCH_SLOTS.

### 7. A stopped trial stops costing GPU (medium)

Ids: F10, F53, F2

**Why.** Wasted GPU time. K2 stopped 107 of 176 trials after the budget limit and Qwen stopped 232 of 545. On a stop, pool.shutdown waits for every in-flight thinking to finish to its end or to the 8,192 budget. In a tiny-model repro, close() took 28.8 s. b11524 does cancel a non-streaming request whose socket closes (wait_for_all(should_stop), polled every 1 s), so the comment saying the server can't be told is wrong. The trial then reads KL on 100 harmless prompts after the unedited thinking (tens of thousands of prefill tokens) plus two beside sets. Stopped trials are always infeasible, and nothing reads their KL except two display lines.

**Fix and risk.** Do not switch to stream: true. Streamed chunks drop the ids of tokens that end mid UTF-8 character, and the final chunk has none (reproduced on 3 of 3 greedy runs); that would corrupt prompt + thought. Order: (1) add the stop-invariance test first (draws identical with stop_early on and off, trials_per_scope 10, STOP_AFTER 1); (2) keep /completion non-streaming over connections whose sockets candor holds; on GeneratorExit set `stopped` and socket.shutdown(SHUT_RDWR) every in-flight one before pool.shutdown, the worker treats OSError as discarded, rewrite the comment, and add a test that close() returns well before an in-flight thinking would end; (3) when judged < len(harmful), tell Optuna inf, store kl=inf with beside None, skip every top_logprobs call, skip KL on the base pass, and filter non-finite KL in cli._thinking_summary and the score table. Do not bump VERSION: fully scored trials are unchanged, and a bump would freeze the K2 and Qwen searches. Time stopped trials before and after on the next pod.

### 8. Measurement saves as it goes and keeps every slot busy (large)

Ids: F152, F57

**Why.** Wasted GPU time at 27B scale. _answer holds a whole set in memory through generation, keyword counts, the gold-answer load and up to three first-token reads, then writes. A pod failure, signal, HTTP timeout, or a ServerError after 20 banned tokens throws away every answer made. Qwen-9B maths took 2 h 51 min per finalist and 4.9 h unedited, an estimated 7-10 h at 27B. Each set is also a separate generate call, so the GPU drains to 1-3 busy slots at every set boundary (K2's unedited harmless and maths sets each had an 8,213-token thinking).

**Fix and risk.** The per-answer journal holds answer text, so it must live in private/ under the same rules as the private answers. Key it by the set's inputs digest and answer only missing indices on resume; per-prompt seeds keep resumed answers valid. Sum answer_seconds across sessions. Load the gold answers and do the harmless first-token reads (they need only kl_ids) before generation, so an error there costs seconds, not hours. Time one finalist's set boundaries first to size the single-pool change before building it. Save each set as soon as its own last answer is in.

### 9. Export by patching the base GGUF, not merge plus convert (large)

Ids: F122, F155, F66, F78, F77, F75, F136, F71, F139, F150, F73, F72

**Why.** Wasted GPU time. _ship writes a full safetensors copy and then runs the Python converter over it while the run lock leaves the GPU idle. A 7B export took 21 min; a 27B BF16 is about 54 GB per pass and asks for 137 GB free. The base GGUF's targets are already checked byte for byte as as_converted(W), and torch and gguf-py round to bf16 identically on 1e6 values, so writing as_converted(W + delta) into a copy of the base gives the same tensor bytes. The same rewriter fixes F66: llama-quantize ignores --tensor-type-file pins when the target is F16/BF16, so the F32 fallback gives a false refusal or a wrong log (reproduced with the real binary). Also in this package:
- F77: _sweep silently deletes other quantisations, including the released file.
- F75, F136: outside_export re-derives _path, and there are two GGUF readers.
- F71 = F139 = F150: full factors are built on CUDA when scored and on the CPU when exported, so the docstring's 'the file shipped is the edit scored' holds only to about 1e-5 to 1e-3.
- F73: kept(), _unrounded and taken_from_gguf are never tested on a qwen35 reorder.
- F72: a duplicate test and a 6-way parametrisation that could be 3.
F78 (symlinking untouched shards) becomes unnecessary once merge.py is deleted.

**Fix and risk.** Prove the new path gives the same output before deleting merge.py. Compare tensor bytes and KV contents, not the whole-file sha256, on SmolLM2 and one K2 trial. If they match, do not bump STAGES['exports'], so the released K2 export's record stays current. Generalise _in_float32 into one rewriter {name: (values_fn, type)} used at the base float type and at F32. Change check_disk to one 16-bit copy plus the quantised file. Build full factors through one helper with one device choice and reword the docstring; don't aim for bit identity across machines. Add a fake quantize that ignores pins for float targets, as the real binary does, and an F16-model, BF16-deployment test.

### 10. Size pod disk and downloads up front (small)

Ids: F35, F42, F154, F36

**Why.** Wasted pod time and failed jobs.
- F35: MODEL_FILES' '*.safetensors' also matches Mistral's consolidated.safetensors: 47-48 GB extra download and disk for Mistral Small 24B or Magistral Small, checked live on the Hub.
- F42: the conversion's disk check runs only after the download.
- F154: the README's '100 GB for Qwen3.5-9B' covers only the build. One export asks for about 49 GB more, and a 27B needs about 262 GB for one export and about 310 GB with 4 finalists kept. A pod sized from the README passes the build and the whole search, then fails at the first export, and growing the container disk means a restart that wipes it.
- F36: logs/server.log grows forever (k2 245 MB) and travels with every fresh push.

**Fix and risk.** Write the README disk rule after the export package lands, because patching the base GGUF lowers what an export asks for; recompute from check_disk's formula. Fix: pass ignore_patterns=['consolidated*.safetensors'] to both snapshot_download calls. When the base GGUF doesn't exist yet, add check_disk's estimate (from source.tensor_shapes on the Hub) to _check_download's need. In prepare, warn (not raise) when free disk is below one export's ask. Rename server.log to server.log.1 at start when it passes 50 MB, which keeps the previous start for debugging.

### 11. The answer request reuses its own thinking's KV (pilot first) (medium)

Ids: F54, F128

**Why.** Wasted GPU time. cache_prompt is false on every request, so each answer re-prefills prompt plus thinking, and each prefill stalls every other slot's decode. The estimate is 100-140K tokens per fully scored K2 trial (about 5-7%) and 230K-400K per measurement set; not measured. b11524 keeps a slot's decoded tokens, so a continuation on the same slot under the same adapters can reuse exactly the KV its own thinking computed.

**Fix and risk.** Contamination is the main danger. Phase 0's '3 refusals became 28' came from reusing a cache built under other adapters, and an empty lora list never clears the cache. Do not rely on similarity routing alone (F128's variant), because the answer can land on a slot whose template prefix was computed under another trial's adapters. Use one server-wide queue of slot ids: each job owns its slot for the whole think-then-answer sequence, and continuations send id_slot with cache_prompt true while every first request keeps cache_prompt false. Numerics move (KV from decode instead of prefill), so draws stop pairing with earlier measurements: add an answer_cache field to the paired inputs, adopt only after timing one K2 set before and after, and land it in the same measurements bump as the answer-seed fix so references are measured again only once. Add an engine test that edited and unedited answers on one slot still match the merged and base models.

### 12. llama-server launch: one helper, fast health poll, GPU probe once (small)

Ids: F133, F126, F83, F127, F84, F87, F137, F131

**Why.** Wasted time.
- F133 = F126: /health is polled every 0.5 s and the first check always fails, so every start sleeps about 0.45 s. Test files run 1.6-2x faster at 0.05 s, and about 789 starts make this several minutes of every full test run.
- F83 = F127: every start runs `--version` and, on CUDA, `--list-devices`, which initialises CUDA. 'full' searches restart the server every trial: about 0.1-0.4% of search time.
- F84: CANDOR_LLAMA_BIN skips the CUDA check.
- F87 = F137: serve copies the launch loop and lacks _exited's allocation diagnosis.
- F131: prompt_ids starts a second server only for 3 prompt-length checks.

**Fix and risk.** Keep a per-start guard against a GPU that died mid-search (phase 3's 'CUDA wouldn't start'). Cache _check_gpu per process and pass --device CUDA0 on linux-cuda, so a restarted llama-server refuses to run instead of falling back to the CPU. Check --device CUDA0 on the next pod. Fake-binary override tests on Linux then need a CUDA0 line. Use a fixed 0.05 s poll; a backoff is unnecessary, and llama-server doesn't log /health. Add an engine launch(args, log, timeout) plus terminate(); LlamaServer and serve wrap them, and serve gets the KV diagnosis with a 'lower --context' hint. Do not give LlamaServer a chat mode. Run the prompt-length check on the first server whenever the slot fits.

### 13. Small run-time waste in score, proxy, adapters and the search log (medium)

Ids: F7, F18, F129, F43, F39, F156, F38, F70, F58

**Why.** Needless work:
- F7 = F18: candor score re-scores the unedited model as a full trial (about 160 s on K2, more on 27B), though the search recorded base_refusals and a search trial is already repeated as the check.
- F129: candor proxy builds three scorers (a server start, a judge load and base KL each) when 'on' and its repeat differ only in seeds.
- F43: every search, score and proxy session loads all seven datasets for harmful_test (6-8 s, GPU idle).
- F39 = F156: linear adapter sets re-read and re-normalise every weight once per set (68 sets and about 1 TB of reads at 27B, an estimated 10-15 min; 'pre' and 'none' runs only).
- F38: 'full' searches print 'Rebuilding scores/trial/edit.gguf' on every trial.
- F70: CPU residual capture keeps every layer output alive.
- F58: the proxy labels clamped copies 'x1.4'.

**Fix and risk.** Keep scoring the base when the search has no full score to repeat (a named trial that stopped early), and print the search's base count otherwise; one verifier saw the base line as a deliberate side-by-side. Proxy scorer sharing must keep the repeat's seeds distinct from 'on'. For adapters, loop over writers on the outside: read once, then one GEMM of stacked directions @ unit, written into each set's .building folder, and commit each set as now. Remove scores/trial before each full trial instead of adding a flag. Use .to(torch.float32, copy=True). Show effective clamped strengths in the proxy table. Whether to skip near-identical copies belongs to the proxy method question.

### 14. Pod jobs: watch the deadline, keep a copy, report failures truthfully (medium)

Ids: F151, F81, F82, F90, F170, F160, F164

**Why.** Robustness, with GPU money at stake.
- F151: wait_for only polls liveness. Nothing pulls on a schedule or warns before the watchdog deletes the container disk at the deadline. K2's pod ran 50 h, a 27B job will run for days, and each renewal is capped at 48 h.
- F81: an SSH drop (exit 255) makes `deadline` and `start` print 'delete it yourself' and `stop` print 'isn't running' (reproduced).
- F82: `wait` exits 0 when the job crashed.
- F90: `code` copies a new uv.lock without re-syncing the venv, and setup installs the dev group.
- F170: setup is the only untested command.
- F160: the watchdog test's `pkill -f` kills its own container, leaks a 'sleep 600' on every run, and cannot catch armed() losing -x.
- F164: watchdog tests wait a fixed 3 s.

**Fix and risk.** Keep deadline renewal manual: it is the cost guard. Fix:
- Read /pod-delete-at in wait's existing SSH probe and end the wait non-zero with 'deadline in N h: renew and pull' under about 4-6 h.
- `wait NAME [PATH]` pulls every 30 min inside a subshell, so a failed pull doesn't end the wait.
- Remote commands print their own state, as pull does, so only ssh failures fail.
- Record the exit status with sh -c '"$@"; echo "exited with $?"' and exit non-zero unless it is 0.
- In setup, move the uv install above `code`; code() then runs `uv sync --frozen --no-dev`.
- Use `pkill -fx`, assert the container is still alive, and always killpg in Pod.close with ProcessLookupError suppressed.
- Watchdog tests wait for the expected call count instead of a fixed time.
Update README step 3 and the CLAUDE.md Awake rule to `wait NAME runs/X`; propose the wording to the user.

### 15. Record the machine and flag cross-machine pairs (medium)

Ids: F144, F120, F145, F147, F148

**Why.** Robustness, guarding a latent confound. Nothing records where a set, trial or judge label was made. Metal reads KL 2.0-6.4% lower than an A40, the same size as the tie-break gaps (205 against 207 is 6.8%). A Mac-built Q4_K_M differs from the pod's in about 1% of K-quant blocks: the release macOS library contracts to FMA and the baseline x86 build does not, and both files carry the same paired.model digest (F148). A search resumed on another GPU puts both backends in one study (F145). The judge runs float16 on MPS and bfloat16 on CUDA (F147). The 27B phase is likely to use bigger cards and maybe a Mac Studio.

**Fix and risk.** Keep the machine out of 'paired' and out of search/inputs.json, which are reuse keys pinned by test_records; adding it there would make every saved set be measured again. Fix: add a cached toolchain.devices() that parses `--list-devices` names (reusing _check_gpu) plus the platform asset name. Write it as 'machine' into each set's JSON, each trial's user attrs, the proxy scores, and judged.json (device, dtype, torch version). compare adds a Machine row when the two differ and marks KL 'not comparable: different backend' when the backends differ. A resumed search reports a machine change once. Add 'same GPU type as the reference' to the CLAUDE.md checklist, with the user's approval. Whether different CUDA cards also need KL suppressed waits on the F146 measurement.

### 16. Search: check before rebuilding, keep the sampler and repeat check honest (medium)

Ids: F1, F8, F157, F118, F110, F121, F4, F30, F140

**Why.** Robustness.
- F1: a search refused for changed inputs first rewrites every adapter set and requantises, deleting the files the old search was scored with (reproduced), then refuses. Going back rebuilds them all again: tens of minutes each way on a 27B. score_locked's comment claiming the run is left as it was is false for a model-file change.
- F8: after max_budget_hits is unset or raised, budget-stopped trials become feasible from partial counts.
- F157: 'full' trials build their adapter on CUDA beside the old server and keep about 1.6 GB cached through every restart.
- F118: the base reads are re-read every session.
- F110 = F121: repeats() needs an exact 4-decimal KL match, so candor score says 'Not reliable' on noise (phase 2's 0.0003 case).
- F4: at_edges hard-codes which ends can move.
- F30: score says 'no finished trials' when some trials stopped, and skips the explanation.
- F140: _judge_again and _widen duplicate the study-move invariant.

**Fix and risk.** Splitting Run.gguf into a plan step and a build step must leave every record digest identical (test_records pins them). Fix:
- search() follows score_locked's order: plan, check, then build; adapters only when trials remain. Extend the refusal test so adapters/ and gguf/ stay byte-identical.
- Write max(violation, 0.5) for stopped trials into system_attrs only.
- Stop the server and free_memory before edit_adapter.
- Save the base top-k once per search at the search's own slot count, keyed by the KL prompts digest and model.
- repeats(): judged exact, refusals within a small count, KL within about 2-3%. Print the differences instead of prefixing every verdict.
- Derive at_edges from each setting's hard limits.
- Move score's repeat loop above the early return and fix the wording.
- Add a _moved(study, name, trials) helper.

### 17. Run stages: checkpoint long work and fail before the build (medium)

Ids: F158, F149, F153, F41, F138, F45, F79, F13, F24

**Why.** Robustness.
- F158: the unedited thinking traces (400 prompts at up to 8,192 tokens; K2 took 30 min at 8 slots, hours estimated at 27B) are saved only when both sets finish.
- F149: a lost directions.pt rebuilt on another device (cosine 0.9999) passes under an identical record, so an export could ship directions the search never scored.
- F153: adapter link names use 79% of Linux's argument limit at 64 layers x 2 targets, and fail at 72 layers or 3 targets a layer. The failure only comes when the search's server starts, after the whole run is built.
- F41: a scoring budget above the deployment budget is unchecked, and harmless_test KL room is never checked.
- F138, F45: inputs.json, proxy scores and config.rewrite are renamed into place without fsync.
- F79: an MPS watermark of 0.0 refuses every model.
- F13: `candor run` doesn't keep the Mac awake.
- F24: _awake uses -i where everything else uses -is.

**Fix and risk.** Fix:
- Save each KL set's traces under the same thinking key as soon as it finishes, keeping the final record and digest as they are.
- Raise RunError when directions*.pt is missing but its record exists.
- Name links by bare number, quoting only when needed (about 3x headroom). Add a 64x3 test and raise the too-many test above 23,697.
- Add a validator: scoring.thinking_budget <= deployment.thinking_budget.
- Give harmless_test thinking room + 1.
- Add one run.write_atomic used at about six sites, leaving run.commit's two-file order alone.
- Treat a watermark ratio of 0 as recommended_max_memory().
- Add _awake() to run, and use caffeinate -is with a fake-caffeinate test.

### 18. CLI: correct guidance, guards and exit codes (medium)

Ids: F14, F15, F16, F19, F109, F20, F22, F23, F27, F28, F37, F44, F85, F86

**Why.** Robustness of what users are told and allowed to do:
- F14: export prints compare commands with names that don't exist when slots != 8, which is every K2 and Qwen run.
- F15: search prints '0 of N answers reached the budget' for a thinking-off search (runs/qwen).
- F16: unexpected errors name a library line and leave no stack anywhere.
- F19 = F109: --seed on a greedy run (runs/rude, the README example) repeats a full measurement for nothing.
- F20: the time estimate quotes the whole search, not what is left.
- F22: --name can silently overwrite the base measurement.
- F23: --heretic is silently ignored with --outside.
- F27: init inspects HEAD, then pins a commit it never inspected.
- F28: serve exits 0 when llama-server crashes.
- F37: run's prompt count double-counts the closed copies (about 45%).
- F44: settings errors print ': Value error, ...'.
- F85: serve's -ngl 999 turns off --fit's partial offload.
- F86: serving the unedited model downloads the full weights (already in STATUS).

**Fix and risk.** Low behavioural risk. Notes:
- Build export's names with _measurement_name and print commands with shlex.join.
- Print the stack as file:line names only. Frame locations hold no prompt text; keep the message hidden.
- Refuse --seed when temperature == 0 and point the README at runs/qwen.
- Guard --name with export.RESERVED unless it equals the default name.
- Resolve the revision before the checks.
- Raise FAILED_CHECK outside the inner try so serve.stop still runs.
- Keep -ngl 999 in LlamaServer, so measurement never falls back to the CPU.
- For F86, serve an existing gguf/model-QUANT.gguf with its record, as export.built does.

### 19. Put the miss-rate simulator under version control (small)

Ids: F98

**Why.** Robustness of the evidence base. The phase 2 search design, phase 3 widening and two-place split, phase 4's stall-stop rejection and the budget-limit check were all decided on a simulator that lives only in .claude/search-design-backup/. .gitignore line 1 ignores that folder, and versions are kept by hand as *.orig copies. Losing this Mac, or moving to a pod or Mac Studio, loses the only way to reproduce those numbers or to check the 27B search design. The user's rule is to judge designs by measured miss rates.

**Fix and risk.** Commit only the source (about 62 .py and .sh files, about 500 KB) and the small numeric inputs (fitted.json, fitted_mapping.json, calibration inputs), under for example tools/misscheck/, with a README naming the candor commit each result used. Leave out caches, runs/ outputs, venvs, the copied candor snapshots and the .orig files. Before pushing, check that every committed JSON holds numbers only (no prompt or answer text) and that no file names the user or local paths, so candor-lab stays anonymous. Then point phase-4 line 305 and findings.md at the committed path.

### 20. Dependencies: lock only real platforms, current converter network stack (small)

Ids: F91, F89, F88

**Why.** Robustness and lock bloat. uv.lock resolves win32 and emscripten, which candor cannot run on (it uses fcntl; toolchain supports macOS arm64 and Linux x86_64): 2,693 lock lines against 1,556 with [tool.uv] environments. accelerate is never imported but transformers needs it for device_map, a trap for the next clean-up. converter-constraints.txt was compiled first-index against the PyTorch CPU index, which pins 2022 requests, urllib3, certifi, idna and charset-normalizer, and makes them impossible to raise. Finally, the llama.cpp pin test does not check the constraints file's release.

**Fix and risk.** Regenerating the constraints with --index-strategy unsafe-best-match (in both the compile header and toolchain.converter()'s install) rebuilds every converter venv and brings filelock 4 and packaging 26. Convert one known model and compare the GGUF before committing. Add a one-line comment above accelerate. From F88, keep only the assertion that converter-constraints.txt names LLAMA_CPP_RELEASE; moving the constant between modules is dropped as a style choice.

### 21. Dead code and duplicated helpers (medium)

Ids: F9, F32, F50, F141, F62, F125, F136, F135, F17, F134, F80, F132

**Why.** Bloat:
- F9: heretic.py keeps fields nothing reads, and the required base_refusals lookup rejects valid heretic v1.0-1.2 journals (reproduced with a KeyError).
- F32: MARKS['skipped'] is never used.
- F50 = F141: there is an unreachable dot check, unused Run.server defaults, Targets.stack, and the RETIRED/_filled upgrade shims. All four kept runs load with the strict config.load.
- F62 = F125: count_refusals, kl.exact and kl.top are used only by tests.
- F136: measure._kl re-implements kl.mean_from_top, and row_norm copies the RowNorm enum.
- F135: a hand-rolled Spearman matches statistics.correlation(method='ranked') on 20,000 tied cases.
- F17 = F134: seven copies of the error-mapping block, the report closure, the heretic_trial call and the progress columns in cli.py (about 80-90 lines).
- F80, F132: the embedding output is stored as directions[0] and never read, and residual batches are padded unsorted.

**Fix and risk.** F80 and F132 change directions.pt and force a directions rebuild (a pod load for K2 and Qwen), so defer them to the next directions stage bump. Remove F141's shims only after confirming no older run folders exist elsewhere. Also delete the README:164 sentence, cli.py:455's retired branch and their tests. Move RowNorm into config.py so config doesn't import torch. Keep heretic_trial's signature: wrap it in a cli helper rather than tying measure.py to Run. Note in STATUS that heretic 2.x writes no 'refusals' attribute. Move kl.exact and kl.top into tests/test_kl.py as the oracle.

### 22. Test suite: parallel, bounded, no repeated setup (small)

Ids: F159, F161, F11, F162, F65, F163

**Why.** Bloat in developer time. The full suite takes about 13 min serially and passed in 77-89 s on 6-12 pytest-xdist workers (796 passed). There is no hang guard, so a stuck llama-server holds one test for 600 s with no stack (F161). The two CHANGES tests build 25 tiny searches for one comparison (F11 = F162, about 30 s saved). Three proxy selection tests run a real 12-trial search each, and the link test forks readlink 2,952 times (F65 = F163, about 13 s).

**Fix and risk.** Don't put -n 6 in addopts: targeted -k runs would get about 4-5 s slower from starting six workers. Add pytest-xdist to the dev group and make the README's full-suite command `uv run pytest -n 6`. Use the built-in faulthandler_timeout = 300, which sits above the slowest e2e (about 219 s), with no new dependency. When merging the CHANGES tests, check score() and the adapter records before search(2), since search() rebuilds adapters before checking until the search package lands. Patch proxy.load, not search.load. List the links in one Python process instead of stat -f, which only works on BSD.

### 23. Test hygiene and coverage gaps (medium)

Ids: F143, F167, F49, F25, F165, F166

**Why.** Tests that prove less than they claim or depend on each other:
- F143 = F167: tests import fixtures from other test modules under noqa F401/F811, conftest loads test modules, write_journal and free_port are duplicated, and the cli config.write appears inline 17 times.
- F49: the description test passes with think_close's description blanked; `!= 'llama'` should be `== 'kimi-k3'`; there are duplicate adapter_factors and conversion fixtures and a leftover `text = cut`.
- F25: the '-x' run-name case passes on click's 'No such option', not on check_run_name.
- F165: the serve SIGTERM test signals pytest itself, so a regression kills the run with no report.
- F166: the last residual row, the one the hooks exist to get right, is never checked.

**Fix and risk.** Low. Move the fixtures converting and model (and tiny_llama/tiny) into conftest.py, which drops all the noqa lines. Put plain helpers in tiny_run.py so no test_*.py imports another. Keep write_journal's parameters as an argument, because the two tests assert different kernels. Put '--' before the run name in the shared init helper. Assert the SIGTERM handler is installed before os.kill. For the residual test, use a pre-hook on model.norm or norm(residuals[-1]) == hidden[-1], plus an inequality check.

### 24. Docs, docstrings and STATUS tidy (small)

Ids: F48, F63, F74, F106, F92, F93, F108, F111, F112, F115, F119, F124, F168, F31

**Why.** Stale or misplaced text:
- F48: the STAGES 'scores' wording, and thinking_room's +16 that implicitly covers UNFINISHED_CHARACTER.
- F63: the Thinking docstring says llama-server forces the close; measure says only 'edit and slots' may differ.
- F74 = F106 (part): export.py says rank-1. README 'How it works' describes only the linear path, and the presence_penalty help ignores llama-server's 64-token window.
- F92: the architecture script's usage needs uv run.
- F93: serve --help omits sampling and the thinking budget.
- F108: results headers point at STATUS for rules that live in history; phase-4 line 26's 'slots at 32 in code' is no longer true.
- F111: STATUS repeats README facts and carries Phase 4 reasons; the heretic upstream check was not recorded at phase 4's start.
- F112: history.md keeps the pre-trim plan, Next list and credit balance.
- F115: findings.md restates CLAUDE.md's lifecycle rule differently, uses a non-standard [Housekeeping] flag, and the README run-folder list lacks proxy/.
- F119 (doc part): the kl_tokens description doesn't say the 0.5% figure was measured on Llama-3B's first token without thinking.
- F124: rebuilding a heretic per-layer trial uses a pre-norm direction at the last layer.
- F168: the e2e downloads about 1.8 GB, not 270 MB, covers every command, and is never scheduled.
- F31 (comment only): the exit-code comment describes inspect only.

**Fix and risk.** Follow the one-home rule: link rather than repeat, and keep numbers in the results files. Don't tag the phase-2 export text 'Not used': its Q8_0 check is still in force, and a pointer to the 2026-10-18 extension is enough. Make the heretic upstream check part of the phase-end discussion item, so the 27B phase can't start without it. Add the e2e run to the end-of-phase checks. Renaming the exit-code constants is dropped as a style choice; only the comment is fixed.

## Method questions (for the phase-end discussion)

- F116 (decision rule): the signed-off rules pass harmless refusals, maths correct and maths budget hits unless the edit is worse at two-sided p < 0.05. That reads absence of evidence as no loss, has no pre-set margin (phase 2 promised 'McNemar with a margin'), and uses a two-sided p for a one-directional question; the harmful criterion already uses a one-sided binomial test. Proposal for the 27B rule: state each 'no worse' criterion as non-inferiority with a margin fixed before any search, passing only when the one-sided 95% bound clears it. Size margins from achievable power. GSM8K's 1,319 questions give a one-sided half-width of about 1.3 pp even at zero difference, so -1.5 pp would have failed trial 205 (its lower bound is -2.30 pp) and would pass a truly lossless edit only about half the time; about -2.5 to -3 pp is achievable, and tighter needs more data or seeds. Optionally run a seed-to-seed noise run of the unedited model. Needs the user's sign-off before any 27B finalist is measured; the compare-intervals package supplies the numbers.
- F117 / F56 (KL as exact): the search keeps only each trial's mean KL, and beats, front and to_measure compare it at 4 decimals. The rule ships the lower KL among passers. On K2 the 100 test prompts ranked 207 (0.0240) ahead of 205 (0.0306), while held-out reversed it (205 0.0518 against 207 0.0614); Qwen's 205 against 175 also flipped. Proposal: store each trial's per-prompt KL (100 floats, numbers only) and test on the K2 and Qwen journals whether 'finalists = every within-cap trial within about 2 paired SEs of the best' would have chosen the same or better sets than hand-picking. Make the tie-break 'lower beyond noise'. Leave beats, front and the rules alone until that evidence exists.
- F55 = F123, with F95's method part and F58's copy-skipping (proxy verdict): verdict() trusts thinking-off scoring on a point Spearman >= 0.7. The published 0.98 spans 24 edits dominated by the split between near-zero and 71-100 refusals. The 12 answer-start edits give 0.93 with a 95% bootstrap interval of 0.49 to 1.0 when resampling source trials. Step 6's near-best edits give 0.24 (0.77 at x1). x1 and x1.4 copies of one trial are not independent, so a Fisher-z interval on 24 edits is too narrow. history.md (2026-10-15) holds that near-zero sets can't show an order, so do not judge by Spearman among the near-best (both F95 verifiers rejected that). Proposal: a 95% interval clustered by source trial with its lower bound required to clear AGREEMENT; report beside the verdict the share of within-cap thinking-off edits that stay within the cap as used; spend the edit budget on more distinct trials than x1.4 copies; reword the 'offset' advice, which still says 'more strength'. Hypothesis (one model) until a second family; it matters for Qwen3.8-27B, which can think off.
- F3 (KL-first stop): a trial could skip its costly harmful answers when at least 25 fully scored feasible trials of its kind have lower KL, which leaves TPE's draws unchanged, and a within-cap trial refusing 0 has lower KL, which keeps it out of every pick. This changes what is scored, because the trial's refusal count goes unrecorded. Replay the K2 and Qwen journals offline (numbers only) and net the answer seconds saved against the KL seconds that every trial would then pay; that partly undoes the stopped-trial KL saving. Possibly zero on K2: per-layer finalists refuse 1-2, and one-layer had only 15 within the caps. Caveats: judged = 0 divides Score's ratios, and proxy groups are not covered. Adopt only if material on two families.
- F5 (beside KL reads): every trial of a thinking model reads kl_thinking and kl_empty on the 100 harmless test prompts (about 1% of a K2 trial, roughly 30-50 min over 313 trials). Only a min-max display line reads them, and measure already reads both places for every finalist. This is a documented choice (history, phase 3 step 3). Proposal: answer the open question once from the K2 and Qwen journals (rank correlation of each against kl, numbers only), record it in findings.md, then drop the reads from the search.
- F67 + F97 (pre against full row normalisation): on real Qwen3.5-9B weights, rank-3 'full' differs from 'pre' by only 0.26-1.1% at the strengths used, for a spread direction, and does not keep row lengths (2-3% change at strength 4.5). For concentrated directions it differs by 1-26%. The switch to 'full' rests on one pair of about 70 trials each, where 'pre' was at 50/100 and still falling against full's 32. K2 inherited 'full' untested, and init still defaults to 'pre'. 'full' costs an adapter build and a server restart every trial. Cheap local evidence: max|r_i| of each layer's direction in runs/qwen and runs/k2's directions.pt, and ||full3 - pre|| / ||pre|| at the finalists' strengths (numbers only). If the gap is under about 1% on both, score one finalist both ways, then decide the row_norm default for 27B (or run a short pre-against-full pair from the same Sobol starts on the 27B pilot).
- F69 = F114 (export kept-share check): the share is <stored - original, edit> / |edit|^2 against the unquantised original. Its noise grows like quantisation error over |edit| (Q4_0 sd ±0.09 at |E|/|W| = 1e-3, ±0.85 at 1e-4, where a fully lost edit passes 0.98 about 15-20% of the time). The bar is one-sided: Avicennasis passed at 98.0-173.8%. Proposal: measure against the unedited weight at its stored type (sd falls to ±0.017-0.069), use a two-sided band (for example 0.98-1.05), and skip weights whose |E| is below a few multiples of the rounding noise. This changes which weights are promoted to Q8_0/F32 and what ships, so it needs evidence across families. Until then, treat readings far above 1 as uninformative.
- F104 (judge recall on edited models): every ship decision counts harmful refusals with Qwen3Guard-Gen-0.6B. Its recall has been checked only on unedited models' plain refusals (judge and keywords agree at 287/287 on K2). Edited models disagree with keywords heavily (K2 205: judge 3, keywords 23; Qwen 205: 16 against 58), and K2's finalists were separated by 1-3 of 288, against a bar of 10/288. Proposal: in copies of runs/k2 and runs/qwen, re-judge the stored held-out answers of the finalists and the unedited model with Qwen3Guard-Gen-8B (counts only). Report the 2x2 agreement, especially the bigger judge's verdict on the keyword-only answers. Record it as Measured. Correct the rationale too: the refusal-cue paper (2608.03201) concerns the Safety label, which would make the judge over-count refusals, not miss them.
- F119 (top-20 KL truncation): KL lumps every token outside the top 20 into one bucket, which can only lower it. The '20 is within 0.5%' figure was measured on Llama-3.2-3B's first answer token without thinking, not at the answer start after thinking, where candor now reads. Synthetic checks give -5.5 to -8.9% at 97% top-20 mass and about -20% at 84%. From saved base reads, print the distribution of top-20 base mass per prompt for the Qwen and K2 runs (numbers only). If many prompts fall below about 0.995, set scoring.kl_tokens to 100-200 in the 27B run's candor.toml; it is nearly free per request and loses nothing in a new run. The config description fix is in the docs package.
- F33 follow-on (models that open their own thinking): the guard package only refuses these models. Supporting them as used, for example by having candor write the open tag itself, changes what is scored and belongs in the phase-end discussion. The no-switch case (Magistral's [THINK], R1-distill templates where the model writes its own tag) is not caught by the guard. Catching it needs a measured check, such as the share of the unedited model's first answer tokens that are a think tag (numbers only).
- F146 (KL shift between CUDA card types): the only backend measurement is Metal against an A40, on 3 trials. Every Qwen and K2 pod was an A40, and the 27B phase will likely use an A100, H100 or L40S, whose kernel choices differ (MMQ stream-k by SM count, flash attention, cuBLAS). Fold a same-card control into the already planned A100/H100 27B pilot: read the base twice on one card and once on the other, at the same slots and --limit. Costs about $1-2 of pod time and needs the user's approval. The result decides whether compare only warns or refuses for different CUDA cards. The simplest robust policy meanwhile is to keep each model's base, search and finalists on one card type.

## Refuted

- F6 (src/candor/search.py): The unedited model's first tokens are re-read every session, so trials from different sessions are measured against different references
- F101 (src/candor/cli.py): The proxy's "offset" verdict prescribes scoring at more strength, which the evidence says is unsafe for one-layer edits
- F40 (src/candor/run.py): Toggling trust_remote_code on a local model reconverts the GGUF, which doesn't use it
- F68 (src/candor/export.py): An edit that changes no weight is converted and quantised in full, then crashes in _span({})
- F76 (src/candor/export.py): _listed_apart's legacy branch exists only for K2 records from before 2026-10-18
- F113 (results/findings.md): "The search's best can fail the full measurement on maths" rests on p of about 0.02 among 16 uncorrected tests
- F142 (tests/test_toolchain.py): A stale monkeypatch: the test patches shutil.which, which this_platform no longer calls
- F169 (tests/test_measure.py): Weak assertions and global-RNG use in small tests

## Duplicates merged

- F133 = F126 (0.5 s health poll)
- F87 = F137 (serve copies LlamaServer's launch, poll and stop)
- F83 = F127 (binaries() probes the GPU on every server start; F84 is covered by the same --device CUDA0 fix)
- F7 = F18 (candor score re-scores the unedited model; F129's candor score part is the same issue)
- F11 = F162 (the two CHANGES tests run the same setup and check)
- F17 = F134 (error block, report closure, heretic_trial call and progress columns copied across cli.py)
- F19 = F109 (measure --seed on the greedy rude run, and the README example of it)
- F29 = F60 (upper-middle value reported as the median)
- F39 = F156 (adapter_set re-reads and re-normalises every weight per direction set)
- F47 = F130 (thinking-model slot defaults of 8; F105 and F48 item 3 are the docs side of it)
- F50 item 4 = F141 (RETIRED/_filled upgrade shims)
- F54 = F128 (answer request re-prefills prompt plus thinking because cache_prompt is off)
- F55 = F123 (proxy verdict as a point Spearman with no interval; F95's method part overlaps)
- F56 = F117 (compare's KL rows have no interval: the code part)
- F62 = F125 (count_refusals is used only by tests)
- F65 = F163 (the short-link engine test forks readlink 2,952 times)
- F69 = F114 (export kept share is noisy and one-sided)
- F71 = F139 = F150 (full factors built on CUDA when scored and on the CPU when exported)
- F110 = F121 (repeats() demands an exact 4-decimal match)
- F122 = F155 (export should patch the base GGUF; F78 is superseded if it lands)
- F144 = F120 (no machine or backend recorded in measurements; F145 is the search-side twin using the same device record)
- F143 = F167 (fixtures and helpers shared by importing test modules; F49 items 3-4 overlap)
- F45 = F138 (config.rewrite's missing fsync is one site of the atomic-write pattern)
- F53 = F63 (the wrong answers() comment that the server can't drop an answer)
- F136 = F125 (measure._kl re-implements kl.mean_from_top)
- F136 = F75 (outside_export re-derives _path)
- F74 = F106 (export.py line 3 says rank-1 factors)
- F67 = F97 (pre against full: the method part)
- F148 = F144 (Mac-built and pod-built files are both accepted as the same paired.model; platform recorded by the same fix)
- F26 = F33 (no test runs a real thinking template, which is how F33 went unnoticed)
