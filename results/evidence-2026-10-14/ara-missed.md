# How candor missed ARA

Written 2026-10-14. Research only: no code, commits or pods. heretic facts come from the GitHub API and PyPI, read on 2026-10-14. Our facts come from the candor repo, its git log, and the two memory folders. Inference is marked **(inference)**.

## Summary

We did not miss ARA completely, but we found it late and filed it under the wrong heading.

- ARA has been public since 2026-03-04 (PR #211). That is seven months before we started.
- heretic's maintainer pointed users with our exact model to it on 2026-07-09: issue #404, "Qwen3.5-9B --- 70/100 Refusals after 200 trials", reply "ARA and SOMA give better results".
- ARA became master's default on 2026-10-01 (PR #446).
- Our first record of ARA is 2026-10-13 (bf45412). It called ARA "unreleased" and made it a fallback "only if the phase 3 pilot fails".
- We first recorded that ARA is now the default on 2026-10-14 (b4f8e93). That is 13 days after the merge, and 2 hours after we built heretic 1.4.0's "full" edit into candor (f440194).

The cause: everything we knew about heretic came from the released package (heretic-llm 1.4.0) and from arXiv. heretic has not released since June, and ARA has no paper, so neither source could show it. Missing it was understandable before 2026-10-13. After that it was a near miss: the PR we cited said on its closing line that it was "Superseded by #446".

## Timeline

| Date | heretic (public) | candor (our records) |
| --- | --- | --- |
| 2025-11-16 | v1.0.0 / v1.0.1 on PyPI | |
| 2026-02-08 | PR #138 "make parameter ranges configurable" opened (still open) | |
| 2026-02-11 | Issue #140: the SOM multi-direction paper (2511.08379) | |
| 2026-02-14 | v1.2.0 | |
| 2026-02-26 | **SOMA first appears**: PR #196, a "[PoC]" of multi-directional refusal suppression with self-organising maps (kabachuha). Closed unmerged 2026-07-07. PR #446 says it "Will also revive #196", so SOMA is still not on master | |
| 2026-03-04 | **ARA first appears**: PR #211, opened by the maintainer. He says ARA "can replace all currently implemented methods in Heretic, including MPOA" and that its only serious competitor is SOMA (#196). One TODO item is unchecked: "Write a proper article explaining the mathematics" | |
| 2026-03-06 | Comment on PR #211: Qwen 3.5 4B on the main branch refused "more than 50 of 100"; with ARA, 4 of 100 at KL 0.1396 | |
| 2026-03 to 2026-08 | 11 more issues and PRs with ARA in the title (#214, #256, #288, #313, #332, #368, #400, #420, #421, #426) | |
| 2026-05-05 | v1.3.0. Its release notes don't mention ARA or SOMA | |
| 2026-06-14 | v1.4.0, the last release on GitHub and PyPI. Release notes and README don't mention ARA or SOMA | |
| 2026-06-18 | Maintainer on PR #211: "We can't make ARA the default while it has such instabilities that we don't understand" | |
| 2026-07-07 | PR #53, a generic plugin system, merged | |
| 2026-07-09 | Issue #404 (Qwen3.5-9B, 70/100, 4 runs). Maintainer: "Running further trials won't magically bring the refusals down ... ARA and SOMA give better results." | |
| 2026-08-17 | Master's version bumped to 2.0.0.dev0 (#428) | |
| 2026-09-04 | PR #211 closed: "Superseded by #446". PR #446 "feat: modifier plugins + ARA" opened | |
| 2026-09-05 | PR #423 merged: better detection of the thinking prefix, for templates that end the prompt with `<think>` | |
| **2026-10-01** | **PR #446 merged (662e4ba2)**. Master's `config.default.toml` now uses the modifier `heretic.modifiers.ara.ARA`, with `n_trials = 100` and `n_startup_trials = 30` (v1.4.0: abliteration, 200 and 60). The PR also "removes Heretic's research features. The intended replacement is Lophius" | |
| 2026-10-02 | Issue #473: "the default mode of Heretic version 2.0 is now going to be ARA/LoRA"; asks for ARA to be explained in the README. Still open; master's README doesn't mention ARA | |
| 2026-10-08 | | tune project files first dated (file times) |
| 2026-10-09 | | `pip install heretic-llm==1.4.0` (`tune/CLOUD_GPU.md:42`, `heretic-kit/setup_main.sh:6`, tune memory `heretic-runpod-procedure.md:22`). heretic runs on rude-llama, K2-Horizon and Qwen3.5-9B (Trial 161). We hit the Qwen `<think>` prefix gotcha by hand (`heretic-runpod-procedure.md:33-38`). We patched the hard-coded 1.5 strength ceiling ourselves (`heretic-kit/patch_ceiling.sh`). Guide and kit "tested ... with heretic 1.4.0" (`heretic-guide.md:11`). We checked other projects' trackers for tooling: mlx-lm#1876 and llama.cpp PR #29535 (`local-model-candidates.md:19,35`) |
| 2026-10-10 | | The user decides to replace heretic with candor (`candor-library.md:11`). Paper review: 3 agents, about 30 arXiv papers. It cites the SOM paper 2511.08379 but nothing from heretic's tracker (`candor-research-review.md:11,29`). STATUS 6cf9cf9 says "heretic and `tune/heretic-kit`: frozen, no new features", and its research table is "From the 2026-10-10 paper review", with multi-direction edits "Optional, only if phase 4 falls short". The full review checked that "The maths matches heretic's source" (STATUS:521) |
| 2026-10-12 | | Goal set: edits that "measurably score and perform better than existing tools (heretic first)" (`candor-library.md:18`) |
| 2026-10-13 16:52 | | **First ARA record** (bf45412), STATUS:254: "heretic's unreleased ARA, a rank-50 edit ... took Qwen3.5-4B from over 50/100 refused to 4/100 at KL 0.14", with the change "3-5 directions or ARA only if the phase 3 pilot fails". This pass also cites heretic #404, #253 and PR #52 |
| 2026-10-14 | | Phase 3 step 4: Trial 161 re-saved with heretic 1.4.0 (`results/phase-3-qwen3.5-9b.md:100`) |
| 2026-10-14 13:26 | | f440194 builds heretic's "full" edit, the 1.4.0 default, into candor ("checked against its code", README:67-68) |
| 2026-10-14 15:47 | | **First record that ARA is the default** (b4f8e93): `results/evidence-2026-10-14/lit-search.md:13`, "Master after PR #446 (Oct 2026) ... the default modifier is now ARA". STATUS:50-51 makes ARA a candidate: "find out what it is first" (still citing PR #211) |

## What our records show

- **The version we tracked: heretic-llm 1.4.0 from PyPI**, everywhere: the guide, the kit, the pod procedure, Trial 161's re-save on 2026-10-14. That is still PyPI's latest. Master has been 2.0.0.dev0 since 2026-08-17. No record shows us checking master before 2026-10-14.
- **When we last researched heretic:** 2026-10-14, the evidence pass (`lit-search.md:9-29`), the first to read master's code and its merged PRs.
- **Did any pass look at heretic's PRs, issues or master?**
  - 2026-10-09: no. Our heretic facts came from its installed code (for example "max_weight is hardcoded to 0.8-1.5 in heretic/main.py (~line 581)", `heretic-runpod-procedure.md:65`). We did check the mlx-lm and llama.cpp trackers that day.
  - 2026-10-10: no. arXiv only.
  - 2026-10-13: yes, issues and PRs (#404, #253, PR #52, PR #211), but not master.
  - 2026-10-14: yes, master, PRs and issues.
- **The closest pass before the find: 2026-10-13 (bf45412).**
  - It had PR #211 in hand, and #211's last comment, dated 2026-09-04, is "Superseded by #446".
  - It cited #404 for the 70/100 figure but not the maintainer's answer in that thread, which recommends ARA and SOMA. **(inference)** It read the issue for data, not advice.
  - It called ARA "unreleased". That is literally true (no release since v1.4.0), but ARA had been the default on master for 12 days.

## The gap

1. **We tracked the release, not the project.** heretic's last release is four months old. Since then master has gained ARA, the plugin system, the MLP floor change (#387), the thinking-prefix fix (#423) and new trial defaults. Tracking the release gave us a stable picture of a tool that had moved on. Two costs are visible in our records:
   - On 2026-10-09 we worked around a Qwen `<think>` prefix problem and the 1.5 ceiling by hand. Master and open PRs were already dealing with both (#423 merged 2026-09-05; #138 open). **(inference)** We have not checked whether #423 fixes Qwen3.5's case; #487 suggests it is still imperfect.
   - On 2026-10-14 we spent a step rebuilding the 1.4.0 default ("full") just before learning the default had changed.
2. **Research meant papers.** The 2026-10-10 review, the only rule for re-checking ("re-check any newer papers", `candor-research-review.md:36`) and STATUS's "Each paper gets read before its change is built" all point at arXiv. ARA has no paper (the article TODO in #211 is unchecked), no release-note entry, no README section and no discussion thread. It exists only in PRs, issues and master, so a paper-only process could not find it.
3. **"Frozen" covered the wrong thing.** The decision on 2026-10-10 froze *our* kit ("no new features"). In practice it also froze our view of heretic at 1.4.0. Yet two days later the goal became beating "existing tools (heretic first)", which needs heretic's current method.
4. **A closed PR was read as finished.** On 2026-10-13 we read PR #211 as an unreleased experiment without checking how it closed. "Closed" on GitHub covers merged, superseded and abandoned.
5. **No check runs at phase starts.** Nothing in STATUS or memory asks "what has heretic (or abliterix) changed since we last looked?" before a phase or a decision rule that names heretic as the reference. The phase 2 and 3 rules compare against Trial 170 and Trial 161, both from 1.4.0.

## Was missing it reasonable?

Partly.

- **For it:**
  - Until 2026-10-01, ARA lived on an unmerged branch, and the maintainer said on 2026-06-18 it couldn't be the default yet.
  - It is still in no release, and in neither README.
  - No paper describes it.
  - Phase 2 (rude-llama) didn't need it: candor beat heretic's real Trial 170 there.
- **Against it:**
  - #404 carries our exact model in its title, and the maintainer's one-line answer there named ARA and SOMA, three months before we started.
  - We searched other projects' trackers on 2026-10-09 but not heretic's.
  - On 2026-10-13 the merge was one click from the PR we cited.

## The fix (light-weight)

Add one short check, run by hand or by one agent in about 10 minutes, and a STATUS line that records it. Nothing automated.

**When:** at the start of each phase, and before writing any decision rule that names another tool's edit as the reference.

**What** (for heretic, and for any tool STATUS names as a comparison, such as abliterix):

```
gh api repos/p-e-w/heretic/commits/master --jq '.sha[0:8]+" "+.commit.committer.date'
gh api "repos/p-e-w/heretic/contents/config.default.toml?ref=master" --jq .content | base64 -d | grep -A2 '^modifiers'
gh pr list -R p-e-w/heretic --state merged --search "merged:>=LAST_CHECK" --json number,title
gh search issues --repo p-e-w/heretic "MODEL_FAMILY" --sort created --limit 20   # e.g. Qwen3.5, K2
```

Read the maintainer's replies in any thread about the model we are about to work on, and read every closed PR we cite to its last comment (merged, superseded or abandoned).

**Record:** one line in STATUS, replaced each time, for example: "Tools checked 2026-10-14: heretic master 66a952b4 (2.0.0.dev0, default ARA; last release 1.4.0), abliterix ...; new since last check: ...".

**Two wording fixes:**
- STATUS "Not in this plan": "heretic-kit frozen; heretic upstream checked at each phase start".
- The research rule: "papers and the trackers of the tools we compare against".

That is the whole change: four commands, one line, two edits to wording. A scheduled watcher or a dashboard would be over-engineering for a project that moves phase by phase.

## Sources opened

- GitHub API (2026-10-14): PRs #211, #446, #196, #423, #138; issues #404, #473; PR #211's comments (filtered); master's commits; `config.default.toml` at master and v1.4.0; README at v1.3.0, v1.4.0 and master; release notes for v1.2.0-v1.4.0; issue and discussion search for "ARA", "Arbitrary-Rank" and "SOMA".
- PyPI JSON for heretic-llm.
- candor: `STATUS.md` (current and at 6cf9cf9), `README.md`, `results/*.md`, `results/evidence-2026-10-14/*.md`, and the git log including `-S` and `-G` searches.
- Memory: the tune folder (`candor-library.md`, `candor-research-review.md`, `heretic-guide.md`, `heretic-runpod-procedure.md`, `heretic-memory-and-dtype.md`, `local-model-candidates.md`, `project-status.md`, `review-before-done.md`) and the candor folder.
- tune: `README.md`, `CLOUD_GPU.md`, `heretic-kit/` (script names, install lines, `patch_ceiling.sh`).
- Nothing under `candor/runs` or `tune/checkpoints`, and no datasets or logs.
- One slip: the first, unfiltered read of PR #211's comments printed the opening lines of a 2026-03-08 comment that quotes a prompt and an answer from a third-party model. Nothing from it is used or repeated here. Every later comment read was filtered to dates, authors and keyword flags.
