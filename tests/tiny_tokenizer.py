"""A tiny word-level tokenizer with a chat template, for tests that need special-token behaviour."""

from tokenizers import Tokenizer, models, pre_tokenizers, processors
from transformers import PreTrainedTokenizerFast

BOS = "<|begin_of_text|>"
WITH_BOS = "{{ bos_token }}{% for m in messages %}{{ m['content'] }}{% endfor %}"
WITHOUT_BOS = "{% for m in messages %}{{ m['content'] }}{% endfor %}"
DATED = WITH_BOS + "{{ strftime_now('%d %b %Y') }}"


def tokenizer(
    chat_template: str | dict | None, adds_bos: bool, bos: str | None = BOS
) -> PreTrainedTokenizerFast:
    vocab = {"[UNK]": 0, BOS: 1, "Hi": 2}
    backend = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    if adds_bos:
        backend.post_processor = processors.TemplateProcessing(single=f"{BOS} $A", special_tokens=[(BOS, 1)])
    wrapped = PreTrainedTokenizerFast(tokenizer_object=backend, bos_token=bos, unk_token="[UNK]")
    if chat_template is not None:
        wrapped.chat_template = chat_template
    return wrapped
