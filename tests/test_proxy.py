import json
import math

import pytest
from typer.testing import CliRunner

from candor import config, proxy, search
from candor.cli import app
from candor.run import RunError
from candor.search import Score, Scores, Search

from .test_search import flat
from .tiny_run import thinking, thinks_through


def test_spearman_shares_tied_ranks_and_is_none_for_a_constant():
    assert proxy.spearman([1, 2, 3], [10, 20, 30]) == pytest.approx(1)
    assert proxy.spearman([1, 2, 3], [30, 20, 10]) == pytest.approx(-1)
    assert proxy._ranks([5, 1, 5, 0]) == [2.5, 1, 2.5, 0]
    assert math.isnan(proxy.spearman([1, 2, 3], [4, 4, 4]))


def test_the_edits_cover_each_kind_and_position_at_every_strength_within_the_widest(tiny):
    thinking(tiny)
    tiny.settings.search.widest_strength = (0.6, 2.0)
    edits, factors = proxy.edits(tiny)
    shapes = len(search.SCOPES) * len(search.POSITIONS) * proxy.SHAPES
    assert len(edits) == len(factors) == shapes * len(proxy.FACTORS)
    assert factors == list(proxy.FACTORS) * shapes
    assert len({edit.name for edit in edits}) == len(edits)
    assert {edit.answering for edit in edits} == {False, True}
    assert {edit.direction_layer is None for edit in edits} == {False, True}
    strengths = [kernel.max_weight for edit in edits for kernel in edit.kernels.values()]
    assert min(strengths) >= 0.6 and max(strengths) <= 2.0
    # Each shape's strength rises with its factor until held at the widest range.
    for i in range(0, len(edits), len(proxy.FACTORS)):
        shape = edits[i : i + len(proxy.FACTORS)]
        peaks = [max(k.max_weight for k in edit.kernels.values()) for edit in shape]
        assert peaks == sorted(peaks)


def faked(monkeypatch, refusals: dict[bool, list[int]], die_after: int | None = None) -> list[tuple]:
    """Search.score_locked scoring each edit with the refusals given for its mode, by edit name, stopped
    once as if killed before edit die_after of its first call; records each call's edit names, thinks,
    seed and base."""
    calls = []
    dying = [die_after]

    def score_locked(
        self, edits, thinks=None, seed=None, scored=lambda i, s: None, base=True, beside_search=True
    ):
        assert not beside_search
        calls.append(([edit.name for edit in edits], thinks, seed, base))
        scores = []
        for i, edit in enumerate(edits):
            if i == dying[0]:
                dying[0] = None
                raise KeyboardInterrupt
            number = int(edit.name.split("-")[1])
            score = Score(refusals[bool(thinks)][number - 1], 0, 0, 0.01 * number, 3, 5.0, 0)
            scored(i, score)
            scores.append(score)
        return Scores(3, (7 if thinks else 6) if base else None, scores)

    monkeypatch.setattr(Search, "score_locked", score_locked)
    return calls


def test_the_check_scores_both_ways_carries_on_after_a_stop_and_repeats_the_best_thinking(tiny, monkeypatch):
    thinks_through(monkeypatch)
    thinking(tiny)
    count = len(proxy.edits(tiny)[0])
    off = list(range(count))
    on = [n + 1 for n in range(count)]
    on[4] = 0
    calls = faked(monkeypatch, {False: off, True: on}, die_after=3)
    with pytest.raises(KeyboardInterrupt):
        proxy.check(tiny)
    calls.clear()
    checked = proxy.check(tiny)
    # The three edits scored before the stop aren't scored again.
    assert calls[0][0] == [f"edit-{n}" for n in range(4, count + 1)] and calls[0][1:] == (False, None, True)
    assert calls[1][0] == [f"edit-{n}" for n in range(1, count + 1)] and calls[1][1:] == (True, None, True)
    assert calls[2] == (["edit-5"], True, tiny.settings.sampling.seed + 1, False)
    assert [s.refusals for s in checked.off] == off and [s.refusals for s in checked.on] == on
    assert checked.base == {"off": 6, "on": 7} and checked.repeated == 4
    assert checked.agreement < 1 and checked.verdict() == "holds"
    calls.clear()
    assert proxy.check(tiny) == checked and calls == []


def test_the_check_starts_afresh_when_what_it_depends_on_changes(tiny, monkeypatch):
    thinks_through(monkeypatch)
    thinking(tiny)
    count = len(proxy.edits(tiny)[0])
    calls = faked(monkeypatch, {False: [1] * count, True: [1] * count})
    proxy.check(tiny)
    tiny.settings.scoring.closed_answer_tokens = 20
    calls.clear()
    checked = proxy.check(tiny)
    assert len(calls[0][0]) == count and math.isnan(checked.agreement) and checked.verdict() == "fails"
    # Which way the search scores its trials doesn't change what the check scored.
    tiny.settings.scoring.thinking = "off"
    calls.clear()
    proxy.check(tiny)
    assert calls == []


def test_the_check_is_only_for_a_model_that_thinks(tiny):
    with pytest.raises(RunError, match="for a model that thinks"):
        proxy.check(tiny)


def test_the_check_scores_a_real_thinking_model_both_ways(tiny, monkeypatch):
    thinks_through(monkeypatch)
    thinking(tiny)
    tiny.settings.scoring.closed_answer_tokens = 16
    monkeypatch.setattr(proxy, "SHAPES", 1)
    # A search made from other settings doesn't stop the check, which never sets its scores beside it.
    tiny.path("search").mkdir()
    tiny.path("search", "inputs.json").write_text(json.dumps({"slots": 1}))
    tiny.path("search", "study.jsonl").write_text("")
    checked = proxy.check(tiny)
    assert len(checked.off) == len(checked.on) == len(search.SCOPES) * len(search.POSITIONS) * 3
    assert all(score.thinking == 0 for score in checked.off)
    assert sum(score.thinking for score in checked.on) > 0
    assert checked.repeat.judged == checked.on[checked.repeated].judged
    saved = json.loads(tiny.path("proxy", "scores.json").read_text())
    assert set(saved) == {"inputs", "base", "off", "on", "repeat"}


def test_proxy_shows_both_counts_and_whether_the_cheap_score_may_be_used(tiny, monkeypatch):
    thinks_through(monkeypatch)
    thinking(tiny)
    count = len(proxy.edits(tiny)[0])
    faked(monkeypatch, {False: list(range(count)), True: [n + 2 for n in range(count)]})
    folder = config.write(tiny.folder.parent, "cli", tiny.settings, "made by a test").parent
    result = CliRunner(env={"COLUMNS": "200"}).invoke(app, ["proxy", str(folder)])
    assert result.exit_code == 0, result.output
    shown = flat(result.output)
    assert "The unedited model refuses 6 with thinking off and 7 with it on." in shown
    assert f"Thinking refuses more on {count} of {count} edits." in shown
    assert "edit-1 refused 2 thinking, and 2 with another seed." in shown
    assert "Within the cap of 2 with thinking off: edit-1, edit-2, edit-3." in shown
    assert "Within the cap of 2 with thinking on: edit-1." in shown
    assert "Rank agreement (Spearman) between the two: 1.00." in shown
    # Ranked alike, but more lenient where an edit nearly works: edit-1, refusing none of 3 thinking off.
    assert (
        "Where thinking off refuses at most 25% of the prompts, thinking refuses a median of 2 more." in shown
    )
    assert "the edits the search suggests must also be scored thinking at more strength" in shown


def scored(refusals: list[int], judged: int = 100) -> list[Score]:
    return [Score(n, 0, 0, 0.0, judged) for n in refusals]


@pytest.mark.parametrize(
    ("off", "on", "verdict"),
    [
        # Ranked alike, and no more lenient where edits nearly work: may be used.
        ([0, 2, 10, 40, 90], [1, 2, 12, 60, 95], "holds"),
        # Ranked alike, but thinking refuses more where the cheap score says edits nearly work.
        ([0, 2, 10, 40, 90], [7, 9, 30, 60, 95], "offset"),
        # Thinking refusing less there is no harm: the cheap score is the stricter.
        ([5, 8, 10, 40, 90], [0, 1, 4, 30, 95], "holds"),
        # Ranked differently.
        ([0, 2, 10, 40, 90], [60, 2, 30, 0, 95], "fails"),
        # No edit came near working with thinking off, so whether it is lenient there can't be told.
        ([40, 50, 60, 70, 90], [41, 52, 63, 74, 95], "fails"),
    ],
)
def test_the_verdict_needs_alike_ranks_and_no_leniency_where_edits_nearly_work(off, on, verdict):
    edits = [proxy.Edit(f"edit-{i}", None, {}) for i in range(len(off))]
    checked = proxy.Check(edits, [1.0] * len(off), scored(off), scored(on), {}, 0, scored([0])[0])
    assert checked.verdict() == verdict
