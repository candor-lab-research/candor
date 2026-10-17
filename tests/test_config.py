import pytest
from pydantic import ValidationError

from candor import config
from candor.config import ConfigError, Model, RunConfig


def settings(**model) -> RunConfig:
    return RunConfig(model=Model.model_validate({"source": "org/model"} | model))


def write_text(tmp_path, text: str):
    (tmp_path / config.FILE).write_text(text, encoding="utf-8")
    return tmp_path


@pytest.mark.parametrize(
    "source",
    [
        "/models/rude llama",
        'C:\\Users\\me\\"quoted"\\model',
        "line\nbreak\ttab",
        "caf\u00e9 \u6a21\u578b \U0001f600",
        "delete\u007fcharacter",
    ],
)
def test_settings_read_back_exactly_as_written(tmp_path, source):
    written = settings(source=source, revision="abc123", date="10 Oct 2026", trust_remote_code=True)
    exclude = {source: [source], "plain_key": ["a", "b"]}
    written.prompts.harmful = config.PromptSet(
        dataset=source, split="train[:4]", revision="def456", exclude=exclude
    )
    config.write(tmp_path, "run", written, "A header\nover two lines")
    assert config.load(tmp_path / "run") == written


def test_a_character_utf8_cannot_hold_is_refused_before_writing():
    with pytest.raises(ValidationError, match="unicode"):
        settings(source="bad\ud800")


def test_every_setting_and_section_has_an_explanation():
    lines = config.render(settings(), "header").splitlines()
    for i, line in enumerate(lines):
        is_setting = " = " in line or line.endswith(" =")
        if (
            line.startswith("[")
            or (is_setting and not line.startswith("# "))
            or line in ("# date =", "# revision =", "# judge_revision =", "# search =", "# max_budget_hits =")
        ):
            assert lines[i - 1].startswith("# ") and not lines[i - 1].endswith(" ="), (
                f"{line!r} has no comment above it"
            )


def test_unset_optional_settings_are_left_as_comments(tmp_path):
    rendered = config.render(settings(), "header")
    assert "# date =" in rendered and "# revision =" in rendered
    loaded = config.load(write_text(tmp_path, rendered))
    assert loaded.model.date is None and loaded.model.revision is None


def test_a_misspelt_setting_is_named(tmp_path):
    text = config.render(settings(), "h").replace("trials_per_scope = 300", "trails_per_scope = 300")
    with pytest.raises(ConfigError, match=r"search\.trails_per_scope"):
        config.load(write_text(tmp_path, text))


@pytest.mark.parametrize(
    ("line", "name"),
    [
        ("startup_per_scope = 20", "search.startup_per_scope"),
        ("[search]", "search"),
        ("thinking = 8", "slots.thinking"),
    ],
)
def test_a_missing_setting_is_an_error_not_a_default(tmp_path, line, name):
    text = config.render(settings(), "h")
    cut = text[: text.index("# The search over edits.")] if line == "[search]" else text.replace(line, "")
    text = cut
    with pytest.raises(ConfigError, match=f"missing settings.*{name}"):
        config.load(write_text(tmp_path, text))


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("strength = [0.8, 1.5]", "strength = [1.5, 0.8]", "first number"),
        ("strength = [0.8, 1.5]", "strength = [0.0, 1.5]", "above 0"),
        ("strength = [0.8, 1.5]", "strength = [0.8, inf]", "finite"),
        ("strength = [0.8, 1.5]", 'strength = ["1", "3"]', "valid number"),
        ("widest_strength = [0.5, 3.0]", "widest_strength = [0.9, 3.0]", "must hold all of strength"),
        ("widest_strength = [0.5, 3.0]", "widest_strength = [0.5, 1.4]", "must hold all of strength"),
        ("widest_strength = [0.5, 3.0]", "widest_strength = [0.0, 3.0]", "above 0"),
        ("widest_strength = [0.5, 3.0]", "widest_strength = [3.0, 0.5]", "first number"),
        ("reach = [0.01, 0.6]", "reach = [0.01, 1.6]", "between 0 and 1"),
        ("trials_per_scope = 300", "trials_per_scope = true", "valid integer"),
        ("trials_per_scope = 300", 'trials_per_scope = "300"', "valid integer"),
        ("max_refusals = 2", "max_refusals = -1", "greater than or equal to 0"),
        ("# max_budget_hits =", "max_budget_hits = -1", "greater than or equal to 0"),
        ("# max_budget_hits =", "max_budget_hits = 2.5", "valid integer"),
        ("stop_early = true", 'stop_early = "yes"', "valid boolean"),
        ("seed = 0", "seed = -1", "greater than or equal to 0"),
        ("seed = 0", "seed = 4294967296", "less than"),
        ('source = "org/model"', 'source = ""', "at least 1 character"),
        ('quantization = "Q4_K_M"', 'quantization = "Q3"', "must be one of"),
        ('kv_cache = "f16"', 'kv_cache = "fp16"', "must be one of"),
        ("startup_per_scope = 20", "startup_per_scope = 600", "startup_per_scope"),
        ("answer_tokens = 100", "answer_tokens = true", "valid integer"),
        ("kl_tokens = 20", 'kl_tokens = "20"', "valid integer"),
        ('row_norm = "pre"', 'row_norm = "post"', '"none", "pre" or "full"'),
        ("trust_remote_code = false", 'trust_remote_code = "no"', "valid boolean"),
        ('split = "train[:400]"', "split = 400", "valid string"),
        ("thinking = 8", "thinking = 0", "greater than or equal to 1"),
        ("# search =", "search = 0", "greater than or equal to 1"),
        ("measure = 8", "measure = 2.5", "valid integer"),
    ],
)
def test_bad_values_are_explained(tmp_path, old, new, message):
    text = config.render(settings(), "h")
    assert old in text
    with pytest.raises(ConfigError, match=message):
        config.load(write_text(tmp_path, text.replace(old, new)))


def test_chat_template_settings_read_back_and_a_file_from_before_them_loads(tmp_path):
    written = settings()
    written.deployment.chat_template_kwargs = {"reasoning_effort": "high", "a key": True, "n": 2, "x": 0.5}
    config.write(tmp_path, "run", written, "h")
    assert config.load(tmp_path / "run") == written
    text = config.render(settings(), "h")
    assert "# chat_template_kwargs =" in text
    before = config.load(write_text(tmp_path, text.replace("# chat_template_kwargs =\n", "")))
    assert before == settings() and before.deployment.chat_template_kwargs is None


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ('{ date_string = "1 Jan 2020" }', "model.date"),
        ("{ when = 1979-05-27 }", "valid"),
        ("{ levels = [1, 2] }", "valid"),
        ('{ "" = 1 }', "at least 1 character"),
    ],
)
def test_chat_template_settings_that_cant_be_passed_on_as_they_are_are_refused(tmp_path, value, message):
    text = config.render(settings(), "h")
    text = text.replace("# chat_template_kwargs =", f"chat_template_kwargs = {value}")
    with pytest.raises(ConfigError, match=message):
        config.load(write_text(tmp_path, text))


def test_a_full_edit_can_be_chosen(tmp_path):
    text = config.render(settings(), "h").replace('row_norm = "pre"', 'row_norm = "full"')
    assert config.load(write_text(tmp_path, text)).edit.row_norm == "full"


def test_whole_numbers_in_ranges_are_fine(tmp_path):
    text = config.render(settings(), "h").replace("strength = [0.8, 1.5]", "strength = [1, 3]")
    assert config.load(write_text(tmp_path, text)).search.strength == (1.0, 3.0)


def test_missing_unreadable_and_broken_files(tmp_path):
    with pytest.raises(ConfigError, match="candor init"):
        config.load(tmp_path)
    (tmp_path / config.FILE).write_bytes("source = 'caf\u00e9'".encode("latin-1"))
    with pytest.raises(ConfigError, match="couldn't read"):
        config.load(tmp_path)
    with pytest.raises(ConfigError, match="isn't valid TOML"):
        config.load(write_text(tmp_path, "[model\nsource ="))


def test_an_existing_run_is_never_overwritten(tmp_path):
    config.write(tmp_path, "run", settings(), "h")
    with pytest.raises(FileExistsError):
        config.write(tmp_path, "run", settings(source="other/model"), "h")
    assert config.load(tmp_path / "run").model.source == "org/model"


@pytest.mark.parametrize("name", ["../x", "a/b", "", ".", "..", "/abs", "-flag", "~home"])
def test_run_names_must_be_one_plain_folder(tmp_path, name):
    with pytest.raises(ConfigError, match="can't be a run name"):
        config.write(tmp_path / "runs", name, settings(), "h")
    assert not (tmp_path / "runs").exists()


def test_extra_files_are_written_with_the_settings(tmp_path):
    config.write(tmp_path, "run", settings(), "h", {"inspect.json": "[]"})
    assert (tmp_path / "run" / "inspect.json").read_text() == "[]"


def test_a_failed_write_leaves_nothing_behind(tmp_path):
    with pytest.raises(FileNotFoundError):
        config.write(tmp_path, "run", settings(), "h", {"inspect.json": "[]", "missing/notes.txt": "x"})
    assert list(tmp_path.iterdir()) == []
    config.write(tmp_path, "run", settings(), "h")


def test_settings_that_dont_read_back_are_never_written(tmp_path, monkeypatch):
    render = config.render
    monkeypatch.setattr(
        config,
        "render",
        lambda *args: render(*args).replace("trials_per_scope = 300", "trials_per_scope = 301"),
    )
    with pytest.raises(ConfigError, match="read back"):
        config.write(tmp_path, "run", settings(), "h")
    assert list(tmp_path.iterdir()) == []


def test_a_missing_setting_points_to_upgrade(tmp_path):
    text = config.render(settings(), "h").replace("startup_per_scope = 20", "")
    with pytest.raises(ConfigError, match="candor upgrade adds them at their defaults"):
        config.load(write_text(tmp_path, text))


def test_an_upgrade_fills_in_missing_settings_and_keeps_the_rest(tmp_path):
    text = config.render(settings(source="/models/mine"), "Made by init\nfor a run")
    held_out = next(line for line in text.splitlines() if line.startswith("harmful_held_out = "))
    text = text.replace(held_out + "\n", "").replace("trials_per_scope = 300", "trials_per_scope = 123")
    upgraded, added, retired, header = config.load_for_upgrade(write_text(tmp_path, text))
    assert added == ["prompts.harmful_held_out"] and retired == []
    assert header == "Made by init\nfor a run"
    assert upgraded.prompts.harmful_held_out == config.Prompts().harmful_held_out
    assert upgraded.search.trials_per_scope == 123 and upgraded.model.source == "/models/mine"
    config.rewrite(tmp_path, upgraded, header)
    assert config.load(tmp_path) == upgraded
    assert [p.name for p in tmp_path.iterdir()] == [config.FILE]


def test_a_rewrite_that_fails_leaves_the_old_file(tmp_path, monkeypatch):
    write_text(tmp_path, "old")
    monkeypatch.setattr(config, "render", lambda *args: "not = [valid")
    with pytest.raises(ConfigError):
        config.rewrite(tmp_path, settings(), "h")
    assert (tmp_path / config.FILE).read_text() == "old"
    assert [p.name for p in tmp_path.iterdir()] == [config.FILE]


def test_an_unpinned_prompt_set_is_left_floating(tmp_path):
    rendered = config.render(settings(), "h")
    unpinned = 'harmful = { dataset = "mlabonne/harmful_behaviors", split = "train[:400]", column = "text" }'
    assert unpinned in rendered
    assert config.load(write_text(tmp_path, rendered)).prompts.harmful.revision is None


def test_settings_candor_no_longer_uses_are_refused_and_upgrade_removes_them(tmp_path):
    text = config.render(settings(), "h")
    old = "trials = 200\nstartup_trials = 60\nkl_target = 0.01\n"
    text = text.replace("trials_per_scope = 300\n", "").replace("startup_per_scope = 20\n", "")
    text = text.replace("[search]\n", "[search]\n" + old)
    path = write_text(tmp_path, text)
    with pytest.raises(
        ConfigError,
        match=r"no longer uses: search\.trials \(.*search\.trials_per_scope\).*candor upgrade removes them",
    ):
        config.load(path)
    upgraded, added, retired, _ = config.load_for_upgrade(path)
    assert retired == ["search.trials", "search.startup_trials", "search.kl_target"]
    assert added == ["search.trials_per_scope", "search.startup_per_scope"]
    assert upgraded.search == config.Search()


def test_a_thinking_model_needs_a_budget_and_one_end_tag():
    thinking = config.Deployment(think_close="\n</think>\n\n", thinking_budget=8192)
    assert thinking.tags() == ("<think>", "</think>")
    assert config.Deployment(think_close="</ifm|think>", thinking_budget=5).tags() == (
        "<ifm|think>",
        "</ifm|think>",
    )
    with pytest.raises(ValidationError, match="needs a thinking_budget"):
        config.Deployment(think_close="</think>")
    with pytest.raises(ValidationError, match="the model doesn't think"):
        config.Deployment(thinking_budget=5)
    for close in ["<think>", "</think> answer", "</think></think>", "think"]:
        with pytest.raises(ValidationError, match="one end tag"):
            config.Deployment(think_close=close, thinking_budget=5)


def test_a_search_budget_needs_a_model_that_thinks():
    with pytest.raises(ValidationError, match="doesn't think"):
        RunConfig(model=Model(source="org/model"), scoring=config.Scoring(thinking_budget=100))
    thinking = config.Deployment(think_close="</think>", thinking_budget=8192)
    RunConfig(
        model=Model(source="org/model"), deployment=thinking, scoring=config.Scoring(thinking_budget=100)
    )


def test_thinking_and_sampling_settings_read_back(tmp_path):
    written = settings()
    written.deployment = config.Deployment(think_close="\n</think>\n\n", thinking_budget=8192)
    written.sampling = config.Sampling(temperature=1.0, top_k=20, min_p=0.0, presence_penalty=1.5, seed=7)
    written.scoring.thinking_budget = 1024
    folder = config.write(tmp_path, "run", written, "made by a test").parent
    assert config.load(folder) == written


def test_slots_read_back_and_the_searchs_may_be_left_unset(tmp_path):
    written = settings()
    written.slots = config.Slots(thinking=4, search=28, measure=16)
    folder = config.write(tmp_path, "run", written, "made by a test").parent
    assert config.load(folder) == written
    text = config.render(settings(), "h")
    assert "[slots]\n" in text and "\n# search =\n" in text
    assert config.load(write_text(tmp_path, text)).slots == config.Slots(thinking=8, search=None, measure=8)


def test_scoring_with_thinking_off_needs_a_model_that_thinks():
    with pytest.raises(ValidationError, match=r'scoring\.thinking is "off" for a model that doesn'):
        config.RunConfig(model=config.Model(source="x"), scoring=config.Scoring(thinking="off"))
    thinks = config.Deployment(think_close="</think>", thinking_budget=8)
    off = config.Scoring(thinking="off")
    assert config.RunConfig(model=config.Model(source="x"), deployment=thinks, scoring=off).scoring == off


def test_max_budget_hits_needs_a_search_whose_trials_think(tmp_path):
    model, hits = config.Model(source="x"), config.Search(max_budget_hits=8)
    with pytest.raises(
        ValidationError, match=r"search\.max_budget_hits is set for a model that doesn't think"
    ):
        config.RunConfig(model=model, search=hits)
    thinks = config.Deployment(think_close="</think>", thinking_budget=8)
    off = config.Scoring(thinking="off")
    with pytest.raises(
        ValidationError, match=r"with scoring\.thinking \"off\" the search's trials don't think"
    ):
        config.RunConfig(model=model, deployment=thinks, scoring=off, search=hits)
    written = config.RunConfig(model=model, deployment=thinks, search=hits)
    assert config.load(config.write(tmp_path, "run", written, "made by a test").parent) == written
