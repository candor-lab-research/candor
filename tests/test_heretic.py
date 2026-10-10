import json

import pytest

from candor.edit import RowNorm
from candor.heretic import read_journal
from candor.kernel import Kernel

PARAMETERS = {
    "attn.o_proj": {
        "max_weight": 1.3,
        "max_weight_position": 16.5,
        "min_weight": 1.0,
        "min_weight_distance": 13.5,
    },
    "mlp.down_proj": {
        "max_weight": 1.2,
        "max_weight_position": 16.0,
        "min_weight": 0.5,
        "min_weight_distance": 12.0,
    },
}


@pytest.fixture
def journal(tmp_path):
    ops = [
        {"op_code": 0, "study_name": "heretic", "directions": [1, 1]},
        {"op_code": 2, "study_id": 0, "user_attr": {"settings": json.dumps({"row_normalization": "pre"})}},
        {"op_code": 4, "study_id": 0},
        {"op_code": 8, "trial_id": 0, "user_attr": {"index": 1}},
        {"op_code": 8, "trial_id": 0, "user_attr": {"direction_index": 12.5}},
        {"op_code": 8, "trial_id": 0, "user_attr": {"parameters": PARAMETERS}},
        {"op_code": 8, "trial_id": 0, "user_attr": {"kl_divergence": 0.03}},
        {"op_code": 8, "trial_id": 0, "user_attr": {"refusals": 4}},
        {"op_code": 8, "trial_id": 0, "user_attr": {"base_refusals": 96}},
        {"op_code": 4, "study_id": 0},
        {"op_code": 8, "trial_id": 1, "user_attr": {"index": 2}},
        {"op_code": 8, "trial_id": 1, "user_attr": {"direction_index": None}},
    ]
    path = tmp_path / "journal.jsonl"
    path.write_text("".join(json.dumps(op) + "\n" for op in ops))
    return path


def test_reads_finished_trials_by_displayed_number(journal):
    result = read_journal(journal)
    assert list(result.trials) == [1]
    trial = result.trials[1]
    assert trial.direction_layer == 12.5
    assert trial.refusals == 4
    assert trial.base_refusals == 96
    assert trial.kl_divergence == 0.03
    assert trial.kernels["attn"] == Kernel(1.3, 16.5, 1.0, 13.5)
    assert trial.kernels["mlp"] == Kernel(1.2, 16.0, 0.5, 12.0)


def test_reads_settings(journal):
    assert read_journal(journal).row_norm is RowNorm.PRE
