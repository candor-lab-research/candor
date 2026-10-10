import pytest

from candor.checks import BeginOfText, begin_of_text, writes_the_date

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
    assert writes_the_date(tokenizer(DATED, adds_bos=False))
    assert not writes_the_date(tokenizer(WITH_BOS, adds_bos=False))
