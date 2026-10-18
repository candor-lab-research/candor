import pytest

from candor.checks import (
    BeginOfText,
    Efforts,
    begin_of_text,
    chat_template,
    pinnable_date_format,
    reasoning_efforts,
    remote_code,
    think_close,
    turns_thinking_off,
    writes_the_date,
)

from .tiny_run import ALWAYS_THINKING_TEMPLATE
from .tiny_tokenizer import DATED, WITH_BOS, WITHOUT_BOS, tokenizer


def test_llama_3_style_doubles_up_in_heretic():
    result = begin_of_text(tokenizer(WITH_BOS, adds_bos=True))
    assert result == BeginOfText(template=1, tokenizer_adds=1)
    assert (result.transformers, result.llama_cpp, result.heretic) == (1, 1, 2)
    assert len(result.problems()) == 1
    assert "heretic" in result.problems()[0]


def test_template_without_bos_splits_the_runtimes():
    result = begin_of_text(tokenizer(WITHOUT_BOS, adds_bos=True))
    assert (result.transformers, result.llama_cpp, result.heretic) == (0, 1, 1)
    assert len(result.problems()) == 2


@pytest.mark.parametrize(("template", "adds_bos"), [(WITH_BOS, False), (WITHOUT_BOS, False)])
def test_consistent_tokenizers_have_no_problems(template, adds_bos):
    result = begin_of_text(tokenizer(template, adds_bos))
    assert result.problems() == []


def test_model_without_a_bos_token():
    assert begin_of_text(tokenizer(WITHOUT_BOS, adds_bos=False, bos=None)) == BeginOfText(0, 0)


def test_writes_the_date():
    assert writes_the_date(DATED)
    assert writes_the_date("{{ strftime_now ('%Y') }}")
    assert not writes_the_date(WITH_BOS)


LLAMA_3 = (
    "{%- if not date_string is defined %}{%- if strftime_now is defined %}"
    '{%- set date_string = strftime_now("%d %b %Y") %}{%- else %}{%- set date_string = "26 Jul 2024" %}'
    "{%- endif %}{%- endif %}Today Date: {{ date_string }}"
)


@pytest.mark.parametrize(
    ("template", "expected"),
    [
        (LLAMA_3, "%d %b %Y"),
        (LLAMA_3.replace("%d %b %Y", "%Y-%m-%d"), "%Y-%m-%d"),
        (
            LLAMA_3.replace("set date_string = strftime_now(", "set date_string  =  strftime_now ( "),
            "%d %b %Y",
        ),
        (LLAMA_3 + "{{ strftime_now('%A') }}", None),
        ("{{ strftime_now('%Y') }}" + LLAMA_3, None),
        ("{% set date_string = strftime_now('%d %b %Y') %}{{ date_string }}", None),
        (WITH_BOS, None),
    ],
)
def test_pinnable_date_format(template, expected):
    assert pinnable_date_format(template) == expected


def test_several_named_templates_resolve_to_the_default():
    named = tokenizer({"default": DATED, "tool_use": WITH_BOS}, adds_bos=False)
    assert chat_template(named) == DATED


def test_no_template():
    assert chat_template(tokenizer(None, adds_bos=False)) is None


def test_remote_code_from_both_configs():
    config = {"auto_map": {"AutoModelForCausalLM": "modeling_x.XForCausalLM"}}
    tokenizer_config = {"auto_map": {"AutoTokenizer": ["tokenization_x.XTokenizer", None]}}
    assert remote_code(config, tokenizer_config) == ["modeling_x.py", "tokenization_x.py"]
    assert remote_code({}, None) == []


THINKING = WITHOUT_BOS + (
    "{% if add_generation_prompt %}<think>\n{% if enable_thinking is defined and not enable_thinking %}"
    "\n</think>\n\n{% endif %}{% endif %}"
)
# Qwen3's own: the model opens its thinking itself, so turning it off writes a whole empty block.
OPENS_ITS_OWN = WITHOUT_BOS + (
    "{% if add_generation_prompt %}A:{% if enable_thinking is defined and not enable_thinking %}"
    "<think>\n\n</think>\n\n{% endif %}{% endif %}"
)


# Qwen3.8's: thinking adds a system message, at every reasoning effort but medium, that turning it
# off leaves out.
MESSAGE_WHEN_THINKING = (
    "{% if enable_thinking is not defined or enable_thinking %}Think hard.\n{% endif %}" + THINKING
)


def test_a_thinking_models_close_is_what_turning_thinking_off_adds():
    assert think_close(tokenizer(THINKING, False)) == "</think>\n\n"


def test_a_close_is_what_turning_thinking_off_writes_after_the_tag_that_opens_it():
    assert think_close(tokenizer(MESSAGE_WHEN_THINKING, False)) == "</think>\n\n"


@pytest.mark.parametrize("template", [WITHOUT_BOS, OPENS_ITS_OWN])
def test_models_that_dont_think_or_dont_open_their_thinking_have_no_close(template):
    assert think_close(tokenizer(template, False)) is None


def test_a_template_that_cant_turn_thinking_off_is_closed_by_the_tag_it_opens():
    always = tokenizer(ALWAYS_THINKING_TEMPLATE, False)
    assert not turns_thinking_off(always)
    assert think_close(always) == think_close(always, reasoning_effort="high") == "</think>"
    assert think_close(always, reasoning_effort="low") == "</think_less>"
    # A template that turns thinking off keeps the close it had, whatever settings it ignores.
    switch = tokenizer(THINKING, False)
    assert turns_thinking_off(switch)
    assert think_close(switch, reasoning_effort="high") == think_close(switch) == "</think>\n\n"


def test_the_reasoning_efforts_a_template_takes_and_the_one_it_takes_when_given_none():
    assert reasoning_efforts(tokenizer(ALWAYS_THINKING_TEMPLATE, False)) == Efforts(("low", "high"), "high")
    # One that writes whatever it is given takes only the values it names.
    anything = WITHOUT_BOS + "Effort: {{ reasoning_effort | default('medium') }}"
    assert reasoning_efforts(tokenizer(anything, False)) == Efforts(("medium",), "medium")
    named_elsewhere = WITHOUT_BOS + "{% if reasoning_effort is defined %}{{ reasoning_effort }}{% endif %}"
    assert reasoning_efforts(tokenizer(named_elsewhere, False)) is None
    assert reasoning_efforts(tokenizer(THINKING, False)) is None


def test_a_value_a_template_names_elsewhere_and_treats_as_unknown_isnt_taken():
    # It names "none" for its tools, and renders an effort of "none" as one it doesn't know: no effort.
    template = WITHOUT_BOS + (
        "{% if tool_choice == 'none' %}{% endif %}{% if reasoning_effort == 'low' %} briefly{% endif %}"
    )
    assert reasoning_efforts(tokenizer(template, False)) == Efforts(("low",), None)
