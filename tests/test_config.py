import pytest

from candor import config
from candor.config import ConfigError, Model, RunConfig


def write_text(tmp_path, text: str):
    (tmp_path / config.FILE).write_text(text)
    return tmp_path


def test_settings_read_back_exactly_as_written(tmp_path):
    settings = RunConfig(model=Model(source="/models/rude llama", date="10 Oct 2026"))
    config.write(tmp_path / "run", settings, "A header\nover two lines")
    assert config.load(tmp_path / "run") == settings


def test_every_setting_has_an_explanation(tmp_path):
    settings = RunConfig(model=Model(source="org/model"))
    lines = config.render(settings, "header").splitlines()
    for i, line in enumerate(lines):
        if line and not line.startswith(("#", "[")):
            assert lines[i - 1].startswith("#"), f"{line!r} has no comment above it"


def test_an_unpinned_date_is_left_as_a_comment(tmp_path):
    rendered = config.render(RunConfig(model=Model(source="org/model")), "header")
    assert "# date =" in rendered
    assert config.load(write_text(tmp_path, rendered)).model.date is None


def test_a_misspelt_setting_is_named(tmp_path):
    text = config.render(RunConfig(model=Model(source="org/model")), "h").replace(
        "trials = 200", "trails = 200"
    )
    with pytest.raises(ConfigError, match=r"search\.trails"):
        config.load(write_text(tmp_path, text))


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("strength = [0.8, 3.0]", "strength = [3.0, 0.8]", "first number"),
        ("strength = [0.8, 3.0]", "strength = [0.0, 3.0]", "above 0"),
        ("reach = [0.01, 0.6]", "reach = [0.01, 1.6]", "between 0 and 1"),
        ('quantization = "Q4_K_M"', 'quantization = "Q3"', "must be one of"),
        ('kv_cache = "f16"', 'kv_cache = "fp16"', "must be one of"),
        ("startup_trials = 60", "startup_trials = 600", "startup_trials"),
    ],
)
def test_bad_values_are_explained(tmp_path, old, new, message):
    text = config.render(RunConfig(model=Model(source="org/model")), "h")
    assert old in text
    with pytest.raises(ConfigError, match=message):
        config.load(write_text(tmp_path, text.replace(old, new)))


def test_missing_and_broken_files(tmp_path):
    with pytest.raises(ConfigError, match="candor init"):
        config.load(tmp_path)
    with pytest.raises(ConfigError, match="isn't valid TOML"):
        config.load(write_text(tmp_path, "[model\nsource ="))


def test_an_existing_run_is_never_overwritten(tmp_path):
    settings = RunConfig(model=Model(source="org/model"))
    config.write(tmp_path / "run", settings, "h")
    with pytest.raises(FileExistsError):
        config.write(tmp_path / "run", settings, "h")
