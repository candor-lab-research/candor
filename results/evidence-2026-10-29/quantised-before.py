"""inspect's quantised-before statistic on Hub models, by ranged reads of 4 edit targets each."""

import sys

import numpy as np
from huggingface_hub import model_info

from candor import cli, source, targets

MODELS = sys.argv[1:]


def distinct(rows: np.ndarray) -> np.ndarray:
    blocks = rows[:, : rows.shape[1] // source.BLOCK * source.BLOCK].reshape(-1, source.BLOCK)
    zero = (blocks & (np.iinfo(blocks.dtype).max >> 1)) == 0
    values = np.sort(np.where(zero, 0, blocks), axis=1)
    found = 1 + np.count_nonzero(np.diff(values, axis=1), axis=1) - zero.any(axis=1)
    return found[np.count_nonzero(~zero, axis=1) > source.FEWEST]


for model in MODELS:
    try:
        sha = model_info(model).sha
        text = source.text_config(source.read_config(model, sha))
        hidden, layers = text["hidden_size"], text["num_hidden_layers"]
        found = targets.find_targets(source.tensor_shapes(model, sha), hidden, layers)
        writers = found.writers
        last = len(writers) - 1
        picked = [writers[i * last // (cli.SAMPLED_WEIGHTS - 1)].name for i in range(cli.SAMPLED_WEIGHTS)]
        headers = source.tensor_headers(model, sha)
        read = source.first_rows(model, list(dict.fromkeys(picked)), cli.SAMPLED_VALUES, sha)
        for name in dict.fromkeys(picked):
            dtype = headers[name]["dtype"]
            if name not in read:
                print(f"{model}\t{sha[:8]}\t{name}\t{dtype}\tnot read")
                continue
            told = distinct(read[name])
            flagged = source.quantised_before(read[name])
            print(
                f"{model}\t{sha[:8]}\t{name}\t{dtype}\t{len(told)}\t{np.median(told):g}\t{told.min()}\t{flagged}"
            )
    except Exception as error:
        print(f"{model}\terror\t{type(error).__name__}: {str(error)[:120]}")
