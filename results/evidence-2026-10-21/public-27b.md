# The public Qwen3.8-27B reference (2026-10-21)

Which public edit of Qwen/Qwen3.8-27B candor measures beside its own, rebuilt with `candor export
--outside` from its full safetensors. From model cards, Hub metadata and one third-party
re-measurement only; no weights downloaded, no model output read. Every candidate edits the base's
weights as uploaded once (commit 72a217a, 2026-08-13) and unchanged through candor's 1d4bf0f.

DreamFast (third party; [abliterlitics](https://abliterlitics.dev/models/qwen38-27b/),
[HF](https://huggingface.co/DreamFast/Qwen-3.8-27b-abliterlitics)): HarmBench 400, thinking on at
xhigh, temperature 0, budget 15,360, one judge (glm-5.3-flash) reading the whole trace;
"harmful-only" is its 300 behaviours without copyright. KL is heretic's: first token, 100
harmless_alpaca prompts.

| Repo @ commit | Method | Card says | DreamFast, harmful-only complied; explicit / soft refusals of 400 | KL | Downloads 30 d, likes |
| --- | --- | --- | --- | --- | --- |
| huihui-ai/Huihui-Qwen3.8-27B-abliterated @739e3c5 | rank-1, layers 17-51, 70 tensors (o_proj, out_proj, down_proj) | no numbers | 98.0%; 0 / 96 (174 of 400 never finish thinking) | 0.0535 (DreamFast) | 40,116, 600 |
| Zynerji/Ektome-...-PristinelyUncensored @f2d5a06 | fractional projection, 128 tensors | 0/100 AdvBench, keywords, thinking off | 98.3%; 0 / 95 | 0.195 | 1,423, 14 |
| orcarouter/Qwen3.8-27B-Uncensored @11a2003 (gated) | rank-1, 131 tensors with embed_tokens and 2 MTP | 0-6% off, 0-1.7% on, opening-phrase classifier | 96.7%; 2 / 69 | 0.0676 | 22,105, 470 |
| abliterant/Qwen3.8-27B-RANA-abliterated @095ca25 | norm-preserving, 131 tensors with embed_tokens | author's harness: ~0 hard refusals, complied 90.0% (orca 83.9%) | none | 0.041 (card) | 520, 1 |
| trohrbaugh/Qwen3.8-27B-heretic-ara @a67ae10 | heretic ARA, 60 tensors | 0/100, keywords | 76.3%; 122 / 48 | 0.0586 (card 0.0535) | 2,101, 108 |
| coder3101/Qwen3.8-27B-heretic @b17d875 | heretic directional, 88 tensors, MTP dropped | 33/100, keywords | 91.7%; 5 / 114 | 0.0842 (card 0.0187) | 217, 6 |
| llmfan46/...-Heretic-Native-MTP-Preserved @4b9e151 | heretic 2.0.0.dev0, MPOA | 3/100, keywords | 92.7%; 0 / 118 | 0.2996 (card 0.0244) | 1,226, 14 |
| ressl/Qwen3.8-27B-uncensored @f0d635a | biprojection, the 128 candor targets | 0/1120 by decline phrases, thinking off | none | none | 92, 1 |
| JonathanColetti/Qwen3.8-27B-Uncensored @5bb7aa9 | heretic, 200 trials | 12/100, keywords, thinking off | none | 0.1191 (card) | 34,957, 109 |

Ruled out: heterodoxin (the measured version deleted; the current one changes the MLP's shape),
0bserverx (a GGUF round trip), Blackfrost (51 explicit refusals), OBLITERATUS (edits lm_head, KL
1.54), DavidAU (a merge of fine-tunes), HauhauCS and dealignai (no BF16 safetensors), SFT and LoRA
releases, and releases with keyword counts only.

**Choice: huihui @739e3c5.** On harmful-only it ties the fewest refusals (ektome 98.3%, huihui
98.0%, orca 96.7%: about 5, 6 and 10 misses in 300, not different), with the least KL of the three,
and its edit lies wholly in candor's targets, so `export --outside` rebuilds it exactly. Orca's and
RANA's `embed_tokens` edits would be dropped silently: `export.taken_from` reads only the targets.
DreamFast's fingerprint (layers 17-51, 70 tensors) matches huihui's current weights (re-uploaded
2026-08-24).

**Thin evidence.** DreamFast is the only independent source: one greedy run on HarmBench, one judge,
xhigh with a 15,360 budget, where candor measures at medium with 8192 and a forced close. huihui
never finishes thinking on 43.5% of DreamFast's prompts (the others' share wasn't noted); under
candor's forced close those answer, so its count may move either way. RANA's harness is the closest
to candor's but one author's and unreplicated.

Sources: the DreamFast report and repo above; [Nathan Sapwell's write-up](https://nathan.sapwell.net/posts/qwen38-27b-abliteration/);
[OrcaRouter's benchmarks](https://www.orcarouter.ai/blog/qwen-3-8-27b-uncensored-benchmarks); the
Hub API and each repo's card at the commits listed.
