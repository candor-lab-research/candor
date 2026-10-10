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
    written.prompts.harmful = config.PromptSet(dataset=source, split="train[:4]", revision="def456")
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
            or line in ("# date =", "# revision =")
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
    text = config.render(settings(), "h").replace("trials = 200", "trails = 200")
    with pytest.raises(ConfigError, match=r"search\.trails"):
        config.load(write_text(tmp_path, text))


@pytest.mark.parametrize(
    ("line", "name"), [("startup_trials = 60", "search.startup_trials"), ("[search]", "search")]
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
        ("reach = [0.01, 0.6]", "reach = [0.01, 1.6]", "between 0 and 1"),
        ("trials = 200", "trials = true", "valid integer"),
        ("trials = 200", 'trials = "200"', "valid integer"),
        ("seed = 0", "seed = -1", "greater than or equal to 0"),
        ("seed = 0", "seed = 4294967296", "less than"),
        ('source = "org/model"', 'source = ""', "at least 1 character"),
        ('quantization = "Q4_K_M"', 'quantization = "Q3"', "must be one of"),
        ('kv_cache = "f16"', 'kv_cache = "fp16"', "must be one of"),
        ("startup_trials = 60", "startup_trials = 600", "startup_trials"),
        ("answer_tokens = 100", "answer_tokens = true", "valid integer"),
        ("kl_tokens = 20", 'kl_tokens = "20"', "valid integer"),
        ('row_norm = "pre"', 'row_norm = "full"', '"none" or "pre"'),
        ("trust_remote_code = false", 'trust_remote_code = "no"', "valid boolean"),
        ('split = "train[:400]"', "split = 400", "valid string"),
    ],
)
def test_bad_values_are_explained(tmp_path, old, new, message):
    text = config.render(settings(), "h")
    assert old in text
    with pytest.raises(ConfigError, match=message):
        config.load(write_text(tmp_path, text.replace(old, new)))


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
    monkeypatch.setattr(config, "render", lambda *args: render(*args).replace("trials = 200", "trials = 201"))
    with pytest.raises(ConfigError, match="read back"):
        config.write(tmp_path, "run", settings(), "h")
    assert list(tmp_path.iterdir()) == []


def test_an_unpinned_prompt_set_is_left_floating(tmp_path):
    rendered = config.render(settings(), "h")
    unpinned = 'harmful = { dataset = "mlabonne/harmful_behaviors", split = "train[:400]", column = "text" }'
    assert unpinned in rendered
    assert config.load(write_text(tmp_path, rendered)).prompts.harmful.revision is None
