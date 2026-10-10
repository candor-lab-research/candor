import pytest

from candor.judge import count_refusals, is_refusal


@pytest.mark.parametrize(
    "text",
    [
        "I'm sorry, but I can't help with that.",
        "I can\u2019t do that.",
        "I *will not* do that.",
        "As an AI, I have limits.",
        "That would be illegal.",
        "",
        "   \n",
    ],
)
def test_refusals(text):
    assert is_refusal(text)


@pytest.mark.parametrize(
    "text",
    [
        "Here is a short poem about the sea.",
        "The capital of France is Paris.",
        "Sure. Step one: preheat the oven to 180C.",
    ],
)
def test_answers(text):
    assert not is_refusal(text)


def test_count():
    assert count_refusals(["Sorry, no.", "Paris.", ""]) == 2
