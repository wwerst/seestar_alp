"""Guard: unit tests must never read or write the real cable-wrap state file.

``device/plant_limits_state.json`` is trusted by live sessions on load
(cable-wrap planning). The autouse fixture in ``tests/conftest.py``
redirects it per test; these tests fail if that redirect stops working,
e.g. if someone reintroduces ``path=_STATE_JSON`` as a default argument
(captured at import time, so patching the module constant would no longer
reach it).
"""

import inspect
import json
import os
from pathlib import Path

import device.plant_limits as plant_limits
from device.plant_limits import CumulativeAzTracker

REAL_STATE = Path(plant_limits.__file__).resolve().parent / "plant_limits_state.json"
DEVICE_DIR = REAL_STATE.parent


def _snapshot(path: Path):
    if not path.exists():
        return None
    st = path.stat()
    return (st.st_mtime_ns, st.st_size, path.read_bytes())


def test_state_path_is_redirected_out_of_device_dir():
    redirected = Path(plant_limits._STATE_JSON).resolve()
    assert redirected != REAL_STATE
    assert DEVICE_DIR not in redirected.parents


def test_default_save_does_not_touch_real_state_file():
    before = _snapshot(REAL_STATE)

    t = CumulativeAzTracker()
    t.reset(cum_az_deg=123.0, wrapped_az_deg=123.0)
    t.save()  # default path

    assert _snapshot(REAL_STATE) == before
    redirected = Path(plant_limits._STATE_JSON)
    assert redirected.is_file()
    assert DEVICE_DIR not in redirected.resolve().parents
    assert json.loads(redirected.read_text(encoding="utf-8"))["cum_az_deg"] == 123.0

    # Default load reads the redirected file back, not the real one.
    loaded = CumulativeAzTracker.load_or_fresh(current_wrapped_az_deg=123.0)
    assert loaded.cum_az_deg == 123.0


def test_state_path_defaults_are_resolved_at_call_time():
    """A default of ``_STATE_JSON`` would be frozen at import time and bypass
    the conftest redirect; the defaults must be ``None``."""
    assert (
        inspect.signature(CumulativeAzTracker.save).parameters["path"].default is None
    )
    assert (
        inspect.signature(CumulativeAzTracker.load_or_fresh).parameters["path"].default
        is None
    )


def test_redirected_state_starts_empty():
    # Each test gets a fresh state dir, so no state leaks between tests.
    assert not os.path.exists(plant_limits._STATE_JSON)
