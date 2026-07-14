"""Tests for device.live_tracker_service — the fourth AppRunner.

Covers the lifecycle wiring without touching the network or a real
mount: start → monitor installed, reload → thresholds updated, stop →
monitor torn down.
"""

from __future__ import annotations

import pytest

from device.live_tracker_service import LiveTrackerMain
from device import sun_safety as ss


@pytest.fixture(autouse=True)
def _clear_monitor_singleton():
    prev = ss.get_sun_monitor()
    yield
    ss.set_sun_monitor(prev)


@pytest.fixture
def _stub_load_toml(monkeypatch):
    """Neutralise Config.load_toml so test-level monkeypatches on
    Config.sun_avoidance_* survive start/reload calls (otherwise the
    TOML reader overwrites them from disk)."""
    from device.config import Config

    monkeypatch.setattr(Config, "load_toml", lambda *a, **kw: None)


def test_start_installs_monitor_and_stop_tears_it_down(monkeypatch, _stub_load_toml):
    # Avoid spinning up the real monitor thread (it would try to reach
    # a non-running ALP server). Replace SunSafetyMonitor.start with a
    # no-op.
    monkeypatch.setattr(ss.SunSafetyMonitor, "start", lambda self: None)

    main = LiveTrackerMain()
    assert ss.get_sun_monitor() is None
    main.start()
    assert ss.get_sun_monitor() is not None
    assert isinstance(ss.get_sun_monitor(), ss.SunSafetyMonitor)
    main.stop()
    assert ss.get_sun_monitor() is None


def test_start_respects_sun_avoidance_disabled_flag(monkeypatch, _stub_load_toml):
    from device.config import Config

    monkeypatch.setattr(ss.SunSafetyMonitor, "start", lambda self: None)
    monkeypatch.setattr(Config, "sun_avoidance_enabled", False, raising=False)

    main = LiveTrackerMain()
    main.start()
    assert ss.get_sun_monitor() is None  # never installed


def test_reload_pushes_updated_thresholds_into_running_monitor(
    monkeypatch, _stub_load_toml
):
    from device.config import Config

    monkeypatch.setattr(ss.SunSafetyMonitor, "start", lambda self: None)

    main = LiveTrackerMain()
    main.start()
    m = ss.get_sun_monitor()
    assert m.min_separation_deg == 30.0  # default

    # Swap the config knob and reload.
    monkeypatch.setattr(Config, "sun_avoidance_min_sep_deg", 45.0, raising=False)
    main.reload()
    assert m.min_separation_deg == 45.0


def test_reload_spins_monitor_up_when_reenabled(monkeypatch, _stub_load_toml):
    from device.config import Config

    monkeypatch.setattr(ss.SunSafetyMonitor, "start", lambda self: None)
    monkeypatch.setattr(Config, "sun_avoidance_enabled", False, raising=False)

    main = LiveTrackerMain()
    main.start()
    assert ss.get_sun_monitor() is None

    monkeypatch.setattr(Config, "sun_avoidance_enabled", True, raising=False)
    main.reload()
    assert ss.get_sun_monitor() is not None


def test_abort_active_sessions_stops_every_mount_driver_per_seestar(monkeypatch):
    """The default abort_active callback iterates over Config.seestars and
    stops ALL six mount-driving managers — any survivor re-slews right after
    the emergency jog and re-trips the monitor in a loop. The visibility
    mapper must be stopped with force=True to override its 5-minute
    minimum-runtime gate."""
    from device.config import Config
    from device.live_tracker_service import _abort_active_sessions

    monkeypatch.setattr(
        Config,
        "seestars",
        [{"device_num": 1}, {"device_num": 2}],
        raising=False,
    )

    stops: list[tuple[str, int]] = []

    class _FakeMgr:
        def __init__(self, name):
            self._name = name

        def stop(self, tid):
            stops.append((self._name, tid))

    class _FakeVisMgr:
        def stop(self, tid, *, force=False):
            assert force is True, "emergency abort must force-stop visibility"
            stops.append(("visibility", tid))

    import device.calibrate_motion as cm
    import device.live_tracker as lt
    import device.nighttime_calibration as nc
    import device.rotation_calibration as rc
    import device.visibility_mapper as vm

    monkeypatch.setattr(lt, "get_manager", lambda: _FakeMgr("live"))
    monkeypatch.setattr(rc, "get_calibration_manager", lambda: _FakeMgr("rotcal"))
    monkeypatch.setattr(
        cm, "get_calibrate_motion_manager", lambda: _FakeMgr("calmotion")
    )
    monkeypatch.setattr(nc, "get_nighttime_manager", lambda: _FakeMgr("night"))
    monkeypatch.setattr(nc, "get_nighttime_auto_manager", lambda: _FakeMgr("auto"))
    monkeypatch.setattr(vm, "get_visibility_manager", lambda: _FakeVisMgr())

    _abort_active_sessions()
    names = {"live", "rotcal", "calmotion", "night", "auto", "visibility"}
    for tid in (1, 2):
        assert {(n, tid) for n in names} <= set(stops)
    assert len(stops) == 12  # 6 managers × 2 scopes, each exactly once


def test_altaz_reader_senses_from_encoder(monkeypatch):
    """The monitor's reader must come from scope_get_horiz_coord (raw
    encoder), not firmware RA/Dec — RA/Dec reads (0,0) until the first
    plate-solve, leaving the monitor blind during daytime work."""
    import device.alpaca_client as ac
    from device import live_tracker_service as svc

    calls: list[str] = []

    class _FakeCli:
        def __init__(self, *a, **kw):
            pass

        def method_sync(self, method, params=None):
            calls.append(method)
            assert method == "scope_get_horiz_coord"
            return {"result": [45.0, 100.0]}  # firmware order: [alt, az]

    monkeypatch.setattr(ac, "AlpacaClient", _FakeCli)
    reader = svc._make_altaz_reader()
    assert reader() == (100.0, 45.0)
    assert calls == ["scope_get_horiz_coord"]
