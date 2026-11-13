import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(autouse=True)
def _isolate_plant_limits_state(tmp_path_factory, monkeypatch):
    """Point the cable-wrap state file at a per-test tmp path.

    ``CumulativeAzTracker.save()`` / ``load_or_fresh()`` default to
    ``device/plant_limits_state.json``, which live sessions trust on load.
    Without this, any test that runs a real ``track()`` overwrites the real
    install's cable-wrap state with fake values (and reads whatever state a
    previous run left behind). Both methods resolve ``_STATE_JSON`` at call
    time, so patching the module constant is enough. A fresh directory per
    test (separate from the test's own ``tmp_path``, so tests that inspect
    ``tmp_path`` contents are unaffected) keeps tests from sharing state.

    Only in-process code is covered: a subprocess (e.g. a simulator started
    by an integration test) still sees the real path.
    """
    import device.plant_limits as plant_limits

    state_dir = tmp_path_factory.mktemp("plant_limits_state")
    monkeypatch.setattr(
        plant_limits, "_STATE_JSON", str(state_dir / "plant_limits_state.json")
    )
    yield
