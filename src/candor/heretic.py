"""Read trials from heretic's Optuna journals, so its results can be rebuilt and checked here."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .edit import RowNorm
from .kernel import Kernel

SET_STUDY_USER_ATTR = 2
SET_TRIAL_USER_ATTR = 8
COMPONENTS = {"attn.o_proj": "attn", "mlp.down_proj": "mlp"}


@dataclass(frozen=True)
class Trial:
    number: int
    direction_layer: float | None
    kernels: dict[str, Kernel]
    refusals: int
    base_refusals: int
    kl_divergence: float


@dataclass(frozen=True)
class Journal:
    settings: dict[str, Any]
    trials: dict[int, Trial]

    @property
    def row_norm(self) -> RowNorm:
        return RowNorm(self.settings["row_normalization"])


def read_journal(path: Path) -> Journal:
    """Trials are keyed by the number heretic displays, which is the journal's trial_id + 1.

    direction_layer is None for trials where every layer uses its own direction.
    """
    settings: dict[str, Any] = {}
    attrs: dict[int, dict[str, Any]] = {}
    with path.open() as f:
        for line in f:
            op = json.loads(line)
            if op["op_code"] == SET_STUDY_USER_ATTR and "settings" in op["user_attr"]:
                settings = json.loads(op["user_attr"]["settings"])
            elif op["op_code"] == SET_TRIAL_USER_ATTR:
                attrs.setdefault(op["trial_id"], {}).update(op["user_attr"])
    trials = {}
    for attr in attrs.values():
        if "refusals" not in attr:
            continue
        trials[attr["index"]] = Trial(
            number=attr["index"],
            direction_layer=attr["direction_index"],
            kernels={COMPONENTS[name]: Kernel(**params) for name, params in attr["parameters"].items()},
            refusals=attr["refusals"],
            base_refusals=attr["base_refusals"],
            kl_divergence=attr["kl_divergence"],
        )
    return Journal(settings, trials)
