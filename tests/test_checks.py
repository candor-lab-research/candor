import shutil

import pytest

from candor.checks import (
    JS,
    BeginOfText,
    app_template_parses,
    begin_of_text,
    chat_template,
    pinnable_date_format,
    remote_code,
    writes_the_date,
)

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


needs_node = pytest.mark.skipif(
    shutil.which("node") is None or not (JS / "node_modules").exists(),
    reason="Node or @huggingface/jinja missing",
)


PLAIN = tokenizer(WITH_BOS, adds_bos=False)


@needs_node
def test_app_template_parses():
    assert app_template_parses(PLAIN, WITH_BOS).status == "ok"


@needs_node
def test_app_template_rejects_syntax_only_transformers_reads():
    template = "{% if messages[0]['content'] is sameas true %}x{% endif %}{{ messages[0]['content'] }}"
    result = app_template_parses(PLAIN, template)
    assert result.status == "fails"
    assert "lines 1" in result.detail


@needs_node
def test_app_template_is_rendered_with_a_system_message_too():
    system_only = (
        "{% for m in messages %}{% if m['role'] == 'system' %}{{ m['content'] | wordcount }}{% endif %}"
        "{{ m['content'] }}{% endfor %}"
    )
    assert app_template_parses(PLAIN, system_only).status == "fails"


@needs_node
def test_a_template_that_refuses_system_messages_is_not_a_parser_failure():
    refuses = (
        "{% for m in messages %}{% if m['role'] == 'system' %}"
        "{{ raise_exception('System role not supported') }}{% endif %}{{ m['content'] }}{% endfor %}"
    )
    assert app_template_parses(PLAIN, refuses).status == "ok"


def test_a_template_transformers_cannot_render_is_skipped():
    assert app_template_parses(PLAIN, "{{ raise_exception('never') }}").status == "skipped"
