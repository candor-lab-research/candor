import json
import math

import pytest
from typer.testing import CliRunner

from candor import config, proxy, search
from candor.cli import app
from candor.measure import Edit
from candor.run import RunError
from candor.search import Score, Scores, Search

from .test_search import flat, small
from .tiny_run import thinking, thinks_through


def test_spearman_shares_tied_ranks_and_is_none_for_a_constant():
    assert proxy.spearman([1, 2, 3], [10, 20, 30]) == pytest.approx(1)
    assert proxy.spearman([1, 2, 3], [30, 20, 10]) == pytest.approx(-1)
    assert proxy._ranks([5, 1, 5, 0]) == [2.5, 1, 2.5, 0]
    assert math.isnan(proxy.spearman([1, 2, 3], [4, 4, 4]))


def searched_thinking_off(tiny, monkeypatch, trials: int = 12):
    """The tiny run with a model that thinks, and a short search of it scored with thinking off."""
    thinks_through(monkeypatch)
    small(thinking(tiny), trials=trials // 2)
    tiny.settings.scoring.thinking = "off"
    tiny.settings.scoring.closed_answer_tokens = 12
    Search(tiny).search(trials)
    return tiny


def test_the_edits_are_each_kind_and_places_best_of_a_thinking_off_search(tiny, monkeypatch):
    searched_thinking_off(tiny, monkeypatch)
    tiny.settings.search.widest_strength = (0.5, 1.6)
    edits, trials, factors = proxy.edits(tiny)
    found = search.load(tiny)
    assert found is not None
    full = search.scored_in_full(found)
    expected = []
    for scope in search.SCOPES:
        for position in search.POSITIONS:
            group = [
                c for c in full if search.scope_of(c.params) == scope and c.params["position"] == position
            ]
            expected += [c.trial for c in sorted(group, key=lambda c: (c.refusals, c.kl))[: proxy.BEST]]
    assert trials == [trial for trial in expected for _ in proxy.FACTORS] and len(set(expected)) > 2
    assert factors == list(proxy.FACTORS) * len(expected)
    assert [edit.name for edit in edits] == [f"edit-{n}" for n in range(1, len(edits) + 1)]
    _, layers, _, _ = tiny.shape
    for edit, trial, factor in zip(edits, trials, factors, strict=True):
        found_edit = search.edit_from(found.trials[trial].params, trial, layers)
        assert edit.direction_layer == found_edit.direction_layer and edit.answering == found_edit.answering
        for component, kernel in edit.kernels.items():
            assert kernel.max_weight == pytest.approx(
                min(found_edit.kernels[component].max_weight * factor, 1.6)
            )


def test_the_check_needs_a_search_scored_with_thinking_off(tiny):
    thinking(tiny)
    with pytest.raises(RunError, match="run candor search with --trials 150 first"):
        proxy.edits(tiny)
    tiny.path("search").mkdir()
    tiny.path("search", "inputs.json").write_text(json.dumps({"trials think": "on"}))
    with pytest.raises(RunError, match=r'scoring\.thinking = "off"'):
        proxy.edits(tiny)


def fixed(monkeypatch, count: int = 24) -> int:
    """proxy.edits giving count edits named by their place, as if from search trials 101 on."""
    edits = [Edit(f"edit-{n}", None, {}) for n in range(1, count + 1)]
    trials = [101 + n // len(proxy.FACTORS) for n in range(count)]
    factors = list(proxy.FACTORS) * (count // len(proxy.FACTORS))
    monkeypatch.setattr(proxy, "edits", lambda run: (edits, trials, factors))
    return count


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
    count = fixed(monkeypatch)
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
    count = fixed(monkeypatch)
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
    searched_thinking_off(tiny, monkeypatch, trials=8)
    monkeypatch.setattr(proxy, "BEST", 1)
    # Settings changed since the search don't stop the check, which never sets its scores beside it.
    tiny.settings.scoring.closed_answer_tokens = 16
    checked = proxy.check(tiny)
    assert len(checked.off) == len(checked.on) == len(checked.trials) >= 2 * len(proxy.FACTORS)
    assert all(score.thinking == 0 for score in checked.off)
    assert sum(score.thinking for score in checked.on) > 0
    assert checked.repeat.judged == checked.on[checked.repeated].judged
    saved = json.loads(tiny.path("proxy", "scores.json").read_text())
    assert set(saved) == {"inputs", "base", "off", "on", "repeat"}


def test_proxy_shows_both_counts_and_whether_the_cheap_score_may_be_used(tiny, monkeypatch):
    thinks_through(monkeypatch)
    thinking(tiny)
    count = fixed(monkeypatch)
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
    edits = [Edit(f"edit-{i}", None, {}) for i in range(len(off))]
    ones = [1.0] * len(off)
    checked = proxy.Check(edits, [1] * len(off), ones, scored(off), scored(on), {}, 0, scored([0])[0])
    assert checked.verdict() == verdict
