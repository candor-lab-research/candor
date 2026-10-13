"""The records a model that doesn't think gets, pinned as candor wrote them before thinking models were
measured: a change to any of them would rebuild finished runs and stop their searches being carried on."""

import json
import os
from pathlib import Path

from candor import config, export
from candor.measure import Measurement
from candor.run import Run, record_of
from candor.search import Search

from .test_export import converting  # noqa: F401
from .test_measure import EDIT
from .test_run import fake_residuals, model  # noqa: F401
from .test_search import small

PINNED = Path(__file__).parent / "records_not_thinking.json"
# The model files' times, which a local model's fingerprint holds.
FIXED = 1_700_000_000_000_000_000


def test_the_records_of_a_model_that_doesnt_think_are_as_they_were(tiny, converting):  # noqa: F811
    for path in tiny.model_folder.rglob("*"):
        if path.is_file():
            os.utime(path, ns=(FIXED, FIXED))
    small(tiny)
    # As candor upgrade leaves a run whose search was made before strength could widen.
    tiny.settings.search.widest_strength = tiny.settings.search.strength
    tiny.adapters()
    Measurement(tiny, None, "base", limit=1).measure()
    Measurement(tiny, EDIT, "edited", limit=1).measure()
    Search(tiny).search(1)
    export.export(tiny, EDIT)
    root = tiny.folder
    found = {}
    for path in sorted(root.rglob("*.json")):
        relative = str(path.relative_to(root))
        if relative.startswith(("private", "logs")):
            continue
        if path.name.endswith(".inputs.json") or relative == "search/inputs.json":
            found[relative] = json.loads(path.read_text().replace(str(root.parents[1]), "TMP"))
    found["adapter_inputs"] = json.loads(json.dumps(tiny.adapter_inputs()))
    assert found == json.loads(PINNED.read_text())


def test_the_directions_record_of_a_model_that_doesnt_think_is_as_it_was(tmp_path, model, monkeypatch):  # noqa: F811
    # The tiny run above fakes its directions, so its adapters pin them only as "fixed".
    for path in model.rglob("*"):
        if path.is_file():
            os.utime(path, ns=(FIXED, FIXED))
    folder = tmp_path / "run"
    folder.mkdir()
    this = Run(folder, config.RunConfig(model=config.Model(source=str(model))), report=lambda _: None)
    monkeypatch.setattr(Run, "prompt_ids", lambda self: {"harmful": [[1, 2]], "harmless": [[3, 4]]})
    fake_residuals(monkeypatch)
    this.directions()
    assert json.loads(record_of(this.path("directions.pt")).read_text()) == {
        "code": {"directions.pt": 1, "gguf": 1, "prompts.json": 1},
        "ids": "dee4368a269e90de",
        "model": "20f561cf5e630ad8",
        "remote_code": False,
    }
    assert this.inputs["directions"] == "7cd1e675e542b346"
