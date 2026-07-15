"""Tests for device.live_tracker — AtomicOffsets, TargetCatalog, session
lifecycle. Session-thread tests use the PositionLogger real implementation
and a FakeMountClient, but they don't drive a real Alpaca HTTP endpoint —
instead they rely on the session's dry-run + test-only cli override.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import numpy as np

from device.live_tracker import (
    ALONG_BOUND_DEG,
    AZ_BIAS_BOUND_DEG,
    AtomicOffsets,
    CachedTarget,
    EL_BIAS_BOUND_DEG,
    TIME_OFFSET_BOUND_S,
    LiveTrackManager,
    LiveTrackSession,
    TargetCatalog,
    load_session_mount_frame,
)
from device.reference_provider import JsonlECEFProvider, ReferenceSample
from device.streaming_controller import OffsetSnapshot
from device.target_frame import MountFrame
from scripts.trajectory.observer import build_site, lla_to_ecef


# --------- AtomicOffsets --------------------------------------------------


def test_atomic_offsets_defaults_are_zero():
    off = AtomicOffsets()
    snap = off.get()
    assert snap == OffsetSnapshot()


def test_atomic_offsets_clamps_to_bounds():
    off = AtomicOffsets()
    snap = off.set(
        az_bias_deg=100.0,
        el_bias_deg=-100.0,
        along_deg=100.0,
        cross_deg=-100.0,
        time_offset_s=100.0,
    )
    assert snap.az_bias_deg == AZ_BIAS_BOUND_DEG
    assert snap.el_bias_deg == -EL_BIAS_BOUND_DEG
    assert snap.along_deg == ALONG_BOUND_DEG
    assert snap.cross_deg == -ALONG_BOUND_DEG  # symmetric bound
    assert snap.time_offset_s == TIME_OFFSET_BOUND_S


def test_atomic_offsets_rejects_nan():
    """NaN values must raise before being stored — otherwise they
    propagate through the streaming controller and poison the mount
    command (NaN in az_bias → NaN in eff_ref_az → NaN velocity)."""
    import math

    import pytest

    off = AtomicOffsets()
    for field in (
        "az_bias_deg",
        "el_bias_deg",
        "along_deg",
        "cross_deg",
        "time_offset_s",
    ):
        with pytest.raises(ValueError):
            off.set(**{field: float("nan")})
    # Unchanged after failed set.
    assert off.get() == OffsetSnapshot()
    # +inf / -inf still clamp cleanly — not regressed by the NaN check.
    snap = off.set(az_bias_deg=float("inf"))
    assert math.isfinite(snap.az_bias_deg)
    assert snap.az_bias_deg == AZ_BIAS_BOUND_DEG


def test_atomic_offsets_partial_updates_preserve_other_fields():
    off = AtomicOffsets()
    off.set(az_bias_deg=0.2, el_bias_deg=-0.1, along_deg=0.05, cross_deg=0.02)
    snap = off.set(time_offset_s=0.5)
    assert snap.time_offset_s == 0.5
    assert snap.az_bias_deg == 0.2
    assert snap.el_bias_deg == -0.1
    assert snap.along_deg == 0.05
    assert snap.cross_deg == 0.02


def test_atomic_offsets_reset_scopes():
    off = AtomicOffsets()
    off.set(
        az_bias_deg=1.0,
        el_bias_deg=1.0,
        along_deg=1.0,
        cross_deg=1.0,
        time_offset_s=2.0,
    )
    after_azel = off.reset_azel()
    assert after_azel.az_bias_deg == 0.0 and after_azel.el_bias_deg == 0.0
    assert after_azel.along_deg == 1.0 and after_azel.cross_deg == 1.0
    assert after_azel.time_offset_s == 2.0
    after_ac = off.reset_alongcross()
    assert after_ac.along_deg == 0.0 and after_ac.cross_deg == 0.0
    assert after_ac.time_offset_s == 2.0
    after_all = off.reset_all()
    assert after_all == OffsetSnapshot()


def test_atomic_offsets_thread_safety_smoke():
    """Many threads writing should not produce inconsistent snapshots."""
    off = AtomicOffsets()

    def writer():
        for _ in range(200):
            off.set(az_bias_deg=0.1, el_bias_deg=-0.1)

    threads = [threading.Thread(target=writer) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    snap = off.get()
    assert snap.az_bias_deg == 0.1 and snap.el_bias_deg == -0.1


# --------- TargetCatalog --------------------------------------------------


def _offset_latlon(lat_deg, lon_deg, dnorth_m, deast_m):
    import math

    dlat = dnorth_m / 111320.0
    dlon = deast_m / (111320.0 * math.cos(math.radians(lat_deg)))
    return lat_deg + dlat, lon_deg + dlon


def _write_fixture(
    path: Path, t0_unix: float, duration_s: float = 5.0, dt: float = 0.5
) -> None:
    site = build_site()
    n = int(round(duration_s / dt)) + 1
    t_grid = t0_unix + np.arange(n) * dt
    east = 100.0 * (t_grid - t_grid[0]) - 100.0 * duration_s / 2.0
    header = {
        "kind": "header",
        "source": "test",
        "id": path.stem,
        "callsign": "FIXTURE",
        "duration_s": duration_s,
        "n_samples": n,
        "peak_el_deg": 5.0,
        "min_slant_m": 5000.0,
        "sample_rate_hz": 1.0 / dt,
    }
    with path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(header) + "\n")
        for i in range(n):
            lat, lon = _offset_latlon(
                site.lat_deg, site.lon_deg, 5000.0, float(east[i])
            )
            ex, ey, ez = lla_to_ecef(lat, lon, site.alt_m + 3000.0)
            f.write(
                json.dumps(
                    {
                        "kind": "sample",
                        "t_unix": float(t_grid[i]),
                        "ecef_x": ex,
                        "ecef_y": ey,
                        "ecef_z": ez,
                        "lat": lat,
                        "lon": lon,
                        "alt_m": site.alt_m + 3000.0,
                        "az_deg": 0.0,
                        "el_deg": 0.0,
                        "slant_m": 0.0,
                    }
                )
                + "\n"
            )


def test_target_catalog_lists_cached(tmp_path):
    root = tmp_path / "trajectories"
    (root / "aircraft").mkdir(parents=True)
    (root / "satellites").mkdir(parents=True)
    _write_fixture(root / "aircraft" / "a.jsonl", t0_unix=time.time() + 1.0)
    _write_fixture(root / "satellites" / "b.jsonl", t0_unix=time.time() + 1.0)
    catalog = TargetCatalog(root, live_enabled=False)
    cached = catalog.list_cached()
    assert len(cached) == 2
    kinds = {t.kind for t in cached}
    assert kinds == {"aircraft", "satellite"}
    assert all(isinstance(t, CachedTarget) for t in cached)
    assert all(t.n_samples > 0 for t in cached)


def test_target_catalog_make_provider_for_file(tmp_path):
    root = tmp_path / "trajectories"
    (root / "aircraft").mkdir(parents=True)
    p = root / "aircraft" / "a.jsonl"
    _write_fixture(p, t0_unix=time.time() + 1.0)
    catalog = TargetCatalog(root, live_enabled=False)
    provider = catalog.make_provider(
        "file",
        p.stem,
        MountFrame.from_identity_enu(),
    )
    assert isinstance(provider, JsonlECEFProvider)
    t0, t1 = provider.valid_range()
    assert t1 > t0


def test_target_catalog_missing_id_raises(tmp_path):
    import pytest

    catalog = TargetCatalog(tmp_path, live_enabled=False)  # no trajectories subdir
    with pytest.raises(KeyError):
        catalog.make_provider("file", "nope", MountFrame.from_identity_enu())


def test_target_catalog_live_missing_id_raises(tmp_path):
    import pytest

    catalog = TargetCatalog(tmp_path, live_enabled=False)
    with pytest.raises(KeyError):
        catalog.make_provider("live", "anything", MountFrame.from_identity_enu())


def test_target_catalog_defers_adsb_poller_until_list_live(tmp_path):
    """Spec: starting a TargetCatalog must not spin up the adsb.fi
    poller thread. Only `list_live()` should wake it up so the live
    tracker stays quiescent when not in use."""
    catalog = TargetCatalog(tmp_path, live_enabled=True)
    # Just after construction: no thread should exist.
    assert catalog._live_thread is None
    try:
        # Touching list_live() spins the poller up. Can't wait for real
        # network traffic, but we can verify the thread is alive within
        # a beat of the call returning.
        catalog.list_live()
        alive = False
        for _ in range(20):
            if catalog._live_thread is not None and catalog._live_thread.is_alive():
                alive = True
                break
            time.sleep(0.05)
        assert alive, "adsb.fi poller did not start after list_live()"
    finally:
        catalog.close()
        if catalog._live_thread is not None:
            catalog._live_thread.join(timeout=2.0)


def test_target_catalog_live_disabled_never_starts_poller(tmp_path):
    catalog = TargetCatalog(tmp_path, live_enabled=False)
    catalog.list_live()
    assert catalog._live_thread is None


# --------- load_session_mount_frame --------------------------------------


def test_load_session_mount_frame_returns_mountframe():
    mf = load_session_mount_frame()
    assert isinstance(mf, MountFrame)


# --------- LiveTrackSession + Manager (lifecycle) ------------------------


class _StationaryProvider:
    """Minimal ReferenceProvider used to verify session lifecycle without
    driving the real FF loop long enough to matter."""

    def __init__(self, t0: float, duration_s: float = 3.0):
        self._t0 = t0
        self._t1 = t0 + duration_s

    def sample(self, t):
        return ReferenceSample(
            t_unix=float(t),
            az_cum_deg=0.0,
            el_deg=45.0,
            v_az_degs=0.0,
            v_el_degs=0.0,
            a_az_degs2=0.0,
            a_el_degs2=0.0,
            stale=False,
            extrapolated=False,
        )

    def valid_range(self):
        return (self._t0, self._t1)


def test_live_track_manager_start_stop_roundtrip(tmp_path, monkeypatch):
    """End-to-end session lifecycle with a fake mount and a stationary
    provider. Verifies start → status.active → stop → status.exit_reason."""
    # Patch the mount-client factory inside the live_tracker module so the
    # session thread uses our fake instead of acquiring a real transport.
    import device.live_tracker as lt

    class _FakeCli:
        def method_sync(self, method, params=None):
            if method == "scope_get_horiz_coord":
                return {
                    "result": [45.0, 0.0],
                    "Timestamp": f"{time.time():.6f}",
                }
            return {"result": None}

    monkeypatch.setattr(lt, "get_mount_client", lambda *a, **kw: _FakeCli())

    offsets = AtomicOffsets()
    provider = _StationaryProvider(t0=time.time() + 0.2)
    session = LiveTrackSession(
        telescope_id=99,
        target_kind="file",
        target_id="fixture",
        target_display_name="Fixture",
        provider=provider,
        offsets=offsets,
        dry_run=True,
        log_dir=tmp_path / "logs",
    )
    mgr = LiveTrackManager()
    mgr.start(session)
    try:
        # Let at least one tick fire.
        time.sleep(1.0)
        st = mgr.status(99)
        assert st is not None
        assert st.active
    finally:
        mgr.stop(99)

    # Post-stop: thread should have joined and exit_reason populated.
    final = mgr.status(99)
    assert final is not None
    assert not final.active
    assert final.exit_reason in {"stop_signal", "end_of_track", "timeout"}


def test_live_track_manager_set_offsets_when_running(tmp_path, monkeypatch):
    import device.live_tracker as lt

    class _FakeCli:
        def method_sync(self, method, params=None):
            if method == "scope_get_horiz_coord":
                return {"result": [45.0, 0.0], "Timestamp": f"{time.time():.6f}"}
            return {"result": None}

    monkeypatch.setattr(lt, "get_mount_client", lambda *a, **kw: _FakeCli())

    session = LiveTrackSession(
        telescope_id=100,
        target_kind="file",
        target_id="fixture",
        target_display_name="Fixture",
        provider=_StationaryProvider(t0=time.time() + 0.2),
        offsets=AtomicOffsets(),
        dry_run=True,
        log_dir=tmp_path / "logs",
    )
    mgr = LiveTrackManager()
    mgr.start(session)
    try:
        snap = mgr.set_offsets(100, az_bias_deg=0.3)
        assert snap is not None
        assert snap.az_bias_deg == 0.3
        snap2 = mgr.reset_offsets(100, scope="azel")
        assert snap2.az_bias_deg == 0.0
    finally:
        mgr.stop(100)


# --------- LiveADSBProvider -----------------------------------------------


def test_live_adsb_provider_builds_from_buffer():
    import math

    from device.live_tracker import LiveADSBProvider, _LiveBuffer

    site = build_site()
    buf = _LiveBuffer(icao24="deadbe", callsign="TEST1")
    t0 = time.time() - 10.0
    n = 8
    for i in range(n):
        east_m = 100.0 * i
        lat = site.lat_deg + (5000.0 / 111320.0)
        lon = site.lon_deg + (
            east_m / (111320.0 * math.cos(math.radians(site.lat_deg)))
        )
        ecef = tuple(float(x) for x in lla_to_ecef(lat, lon, site.alt_m + 3000.0))
        buf.append(t0 + float(i), ecef)

    provider = LiveADSBProvider(buf, MountFrame.from_identity_enu(site), rebuild_s=0.0)
    t_start, t_end = provider.valid_range()
    assert t_end > t_start
    mid = 0.5 * (t_start + t_end)
    sample = provider.sample(mid)
    assert isinstance(sample, ReferenceSample)
    assert not sample.stale


def test_live_adsb_provider_stale_without_samples():
    import device.live_tracker as lt

    buf = lt._LiveBuffer(icao24="none", callsign="")
    provider = lt.LiveADSBProvider(buf, MountFrame.from_identity_enu())
    s = provider.sample(time.time())
    assert s.stale


def test_auto_slew_refuses_when_target_inside_sun_cone(monkeypatch):
    """Spec: LiveTrackSession._auto_slew must consult sun_safety.is_sun_safe
    against the provider's first (az, el) and refuse the pre-slew
    (exit_reason='sun_avoidance') rather than commanding the mount into
    the cone."""
    import device.live_tracker as lt
    from device import sun_safety as ss

    # Force is_sun_safe to always refuse. The session's cur_az/el
    # reading comes from the fake client.
    real_is_sun_safe = ss.is_sun_safe
    monkeypatch.setattr(
        ss,
        "is_sun_safe",
        lambda *a, **kw: (False, "sun_avoidance: forced by test"),
    )
    try:

        class _FakeCli:
            def method_sync(self, method, params=None):
                if method == "scope_get_horiz_coord":
                    return {
                        "result": [45.0, 0.0],
                        "Timestamp": f"{time.time():.6f}",
                    }
                return {"result": None}

        monkeypatch.setattr(lt, "get_mount_client", lambda *a, **kw: _FakeCli())

        session = LiveTrackSession(
            telescope_id=77,
            target_kind="file",
            target_id="fix",
            target_display_name="Fix",
            provider=_StationaryProvider(t0=time.time() + 0.2),
            offsets=AtomicOffsets(),
            dry_run=True,
        )
        cli = _FakeCli()
        loc = None  # loc is unused by measure_altaz_timed under the fake
        session._auto_slew(cli, loc)
    finally:
        monkeypatch.setattr(ss, "is_sun_safe", real_is_sun_safe)

    assert session._exit_reason == "sun_avoidance"
    assert session._phase == "refused"
    assert any("sun_avoidance" in e for e in session._errors)


def test_stop_requested_during_preslew_records_exit_reason():
    """Direct unit of the pre-slew stop helper: flipping the stop event
    makes the helper return True and stamp the session with
    exit_reason='stop_signal', phase='stopped'."""
    session = LiveTrackSession(
        telescope_id=200,
        target_kind="file",
        target_id="fix",
        target_display_name="Fix",
        provider=_StationaryProvider(t0=time.time() + 1.0),
        offsets=AtomicOffsets(),
        dry_run=True,
    )
    assert session._stop_requested_during_preslew() is False
    session._stop_evt.set()
    assert session._stop_requested_during_preslew() is True
    assert session._exit_reason == "stop_signal"
    assert session._phase == "stopped"


def test_live_track_manager_double_start_refuses(tmp_path, monkeypatch):
    import pytest
    import device.live_tracker as lt

    class _FakeCli:
        def method_sync(self, method, params=None):
            if method == "scope_get_horiz_coord":
                return {"result": [45.0, 0.0], "Timestamp": f"{time.time():.6f}"}
            return {"result": None}

    monkeypatch.setattr(lt, "get_mount_client", lambda *a, **kw: _FakeCli())

    def make_session():
        return LiveTrackSession(
            telescope_id=101,
            target_kind="file",
            target_id="fix",
            target_display_name="Fix",
            provider=_StationaryProvider(t0=time.time() + 0.2),
            offsets=AtomicOffsets(),
            dry_run=True,
            log_dir=tmp_path / "logs",
        )

    mgr = LiveTrackManager()
    s1 = make_session()
    mgr.start(s1)
    try:
        with pytest.raises(RuntimeError):
            mgr.start(make_session())
    finally:
        mgr.stop(101)


def test_live_track_manager_start_atomic_under_lock():
    """Regression for a TOCTOU race: session.start() must run while the
    manager lock is held. Otherwise two concurrent mgr.start() calls
    for the same telescope id can both pass the is_alive() check (the
    first session is registered but its thread hasn't been spawned yet,
    so is_alive() returns False) and spawn duplicate tracking threads
    that both drive the mount."""
    mgr = LiveTrackManager()
    captured: dict[str, bool] = {}

    class _FakeSession:
        telescope_id = 300

        def __init__(self) -> None:
            self._alive = False

        def start(self) -> None:
            # Observe the manager lock from inside session.start(). If
            # mgr.start() dropped the lock before calling us, a racing
            # caller could observe is_alive()=False and overwrite the
            # registry — exactly the bug this test guards against.
            captured["lock_held_during_session_start"] = mgr._lock.locked()
            self._alive = True

        def is_alive(self) -> bool:
            return self._alive

        def stop(self, timeout: float = 5.0) -> None:
            pass

        def status(self):
            return None

    mgr.start(_FakeSession())
    assert captured["lock_held_during_session_start"] is True


# --------- _poll_once backoff / seen_pos (findings 2, 3) -----------------


def test_poll_once_empty_response_is_success(monkeypatch):
    """A successful-but-empty adsb.fi response (clear sky) must return True
    so the poller keeps its 5 s cadence. Only a genuine fetch failure
    (poll_once_adsbfi -> None) returns False to drive the backoff. Before the
    fix the success path fell off the end returning None → every good poll
    looked like a failure and backed off 5 s → 60 s."""
    import device.live_tracker as lt

    catalog = lt.TargetCatalog(live_enabled=False)

    monkeypatch.setattr(lt, "poll_once_adsbfi", lambda *a, **k: [])
    assert catalog._poll_once() is True

    monkeypatch.setattr(lt, "poll_once_adsbfi", lambda *a, **k: None)
    assert catalog._poll_once() is False


def test_poll_once_populated_response_is_success(monkeypatch):
    import device.live_tracker as lt

    site = build_site()
    ac = {
        "hex": "abc123",
        "lat": site.lat_deg + 0.05,
        "lon": site.lon_deg,
        "alt_geom": 10000.0,  # ft → ~3048 m, under MAX_ALT_M
        "flight": "TEST123",
        "gs": 300.0,
        "track": 90.0,
    }
    catalog = lt.TargetCatalog(live_enabled=False)
    monkeypatch.setattr(lt, "poll_once_adsbfi", lambda *a, **k: [ac])
    assert catalog._poll_once() is True
    assert catalog._live_buffers.get("abc123") is not None


def test_poll_once_backdates_sample_by_seen_pos(monkeypatch):
    """ADS-B samples must be stamped at time.time() - seen_pos (the feed's
    per-aircraft position age), not the poll wall time — otherwise the mount
    trails the target by degrees."""
    import device.live_tracker as lt

    site = build_site()
    seen_pos = 8.0
    ac = {
        "hex": "def456",
        "lat": site.lat_deg + 0.05,
        "lon": site.lon_deg,
        "alt_geom": 10000.0,
        "flight": "AGED1",
        "gs": 300.0,
        "track": 90.0,
        "seen_pos": seen_pos,
    }
    catalog = lt.TargetCatalog(live_enabled=False)
    monkeypatch.setattr(lt, "poll_once_adsbfi", lambda *a, **k: [ac])

    t_before = time.time()
    assert catalog._poll_once() is True
    t_after = time.time()

    buf = catalog._live_buffers.get("def456")
    assert buf is not None
    assert (
        (t_before - seen_pos) - 0.1 <= buf.last_sample_t <= (t_after - seen_pos) + 0.1
    )


def test_live_buffer_drops_equal_and_backdated_stamps():
    """The buffer must stay strictly monotonic in t_unix: a re-poll of a
    stale position (seen_pos yields an equal or earlier stamp) is dropped so
    np.interp / unwrap downstream never see a non-sorted time axis."""
    from device.live_tracker import _LiveBuffer

    buf = _LiveBuffer(icao24="x", callsign="")
    buf.append(100.0, (1.0, 2.0, 3.0))
    buf.append(100.0, (4.0, 5.0, 6.0))  # equal → dropped
    buf.append(95.0, (7.0, 8.0, 9.0))  # backdated → dropped
    buf.append(105.0, (1.0, 1.0, 1.0))  # newer → kept
    _, t, _ecef = buf.snapshot()
    assert list(t) == [100.0, 105.0]


# --------- guarded unwrap (finding 8) ------------------------------------


def test_live_adsb_provider_guarded_unwrap_handles_non_finite(monkeypatch):
    """LiveADSBProvider._rebuild uses the guarded device.geometry
    unwrap_az_series; a non-finite az reading is rejected at the boundary
    (ValueError handled) instead of poisoning the interpolation table."""
    import device.live_tracker as lt
    from device.target_frame import MountFrame

    site = build_site()
    buf = lt._LiveBuffer(icao24="deadbe", callsign="T")
    # Exactly 3 samples → _rebuild takes the linear/else branch that calls
    # unwrap_az_series directly (≥4 would go through ecef_traj_to_mount).
    t0 = time.time() - 5.0
    ecef = tuple(
        float(x)
        for x in lla_to_ecef(site.lat_deg + 0.05, site.lon_deg, site.alt_m + 3000.0)
    )
    for i in range(3):
        buf.append(t0 + i, ecef)

    def _bad_azel(self, ecef_arr):
        n = len(ecef_arr)
        az = np.full(n, 10.0)
        az[0] = np.inf
        return az, np.full(n, 45.0), np.full(n, 1000.0)

    monkeypatch.setattr(MountFrame, "ecef_array_to_mount", _bad_azel)

    # Construction must not raise even though the unwrap rejects the input.
    provider = lt.LiveADSBProvider(
        buf, MountFrame.from_identity_enu(site), rebuild_s=0.0
    )
    s = provider.sample(time.time())
    assert s.stale  # empty tables → graceful stale sample, not a crash


# --------- feasibility pre-check + cable limits (finding 4) --------------


class _OverheadProvider:
    """Provider whose elevation exceeds the mount's usable band, so
    pre_check must mark it infeasible."""

    is_live = False
    extrapolation_s = 1.0

    def __init__(self, t0: float, duration_s: float = 3.0):
        self._t0 = t0
        self._t1 = t0 + duration_s

    def sample(self, t):
        return ReferenceSample(
            t_unix=float(t),
            az_cum_deg=0.0,
            el_deg=95.0,  # above the 85° usable ceiling
            v_az_degs=0.0,
            v_el_degs=0.0,
            a_az_degs2=0.0,
            a_el_degs2=0.0,
            stale=False,
            extrapolated=False,
        )

    def valid_range(self):
        return (self._t0, self._t1)


def test_live_track_manager_rejects_infeasible_trajectory(tmp_path, monkeypatch):
    """LiveTrackManager.start runs a feasibility pre-check and raises
    PreCheckFailed (a ValueError, mapped to HTTP 400) for a trajectory that
    would leave the elevation band before any motion is commanded."""
    import pytest
    import device.live_tracker as lt
    from device.live_tracker import PreCheckFailed

    class _FakeCli:
        def method_sync(self, method, params=None):
            return {"result": None}

    monkeypatch.setattr(lt, "get_mount_client", lambda *a, **kw: _FakeCli())

    session = LiveTrackSession(
        telescope_id=120,
        target_kind="file",
        target_id="overhead",
        target_display_name="Overhead",
        provider=_OverheadProvider(t0=time.time() + 0.2),
        offsets=AtomicOffsets(),
        dry_run=True,
        auto_slew=False,
        log_dir=tmp_path / "logs",
    )
    mgr = LiveTrackManager()
    with pytest.raises(PreCheckFailed):
        mgr.start(session)
    # Nothing was started.
    assert mgr.get(120) is None


def test_auto_slew_passes_cable_limits_to_move_to_ff(monkeypatch):
    """The web pre-slew must thread AzimuthLimits + CumulativeAzTracker into
    move_to_ff so its cumulative-aware planner keeps inside the ±450° cable
    hard stop (the CLI track path already does this)."""
    import device.live_tracker as lt
    from device import sun_safety as ss
    from device.plant_limits import AzimuthLimits, CumulativeAzTracker

    monkeypatch.setattr(ss, "is_sun_safe", lambda *a, **k: (True, ""))
    monkeypatch.setattr(lt, "ensure_scenery_mode", lambda cli: None)

    captured: dict = {}

    def _fake_move(cli, **kwargs):
        captured.update(kwargs)
        return (kwargs["target_el_deg"], kwargs["target_az_deg"], {"converged": True})

    monkeypatch.setattr(lt, "move_to_ff", _fake_move)

    class _FakeCli:
        def method_sync(self, method, params=None):
            if method == "scope_get_horiz_coord":
                return {"result": [45.0, 0.0], "Timestamp": f"{time.time():.6f}"}
            return {"result": None}

    session = LiveTrackSession(
        telescope_id=130,
        target_kind="file",
        target_id="fix",
        target_display_name="Fix",
        provider=_StationaryProvider(t0=time.time() + 0.2),
        offsets=AtomicOffsets(),
        dry_run=True,
    )
    limits = AzimuthLimits.load()
    tracker = CumulativeAzTracker()
    session._auto_slew(_FakeCli(), None, az_limits=limits, az_tracker=tracker)

    assert captured.get("az_limits") is limits
    assert captured.get("az_tracker") is tracker


# --------- sun-refusal terminal / direct motor stop (findings 5, 7) ------


def test_run_preslew_sun_refusal_is_terminal(tmp_path, monkeypatch):
    """On a sun refusal the pre-slew is terminal: _run must not fall through
    into track() and clobber phase/exit_reason. The refusal also sets the
    stop event."""
    import device.live_tracker as lt
    from device import sun_safety as ss

    monkeypatch.setattr(
        ss, "is_sun_safe", lambda *a, **k: (False, "sun_avoidance: forced")
    )
    monkeypatch.setattr(lt, "ensure_scenery_mode", lambda cli: None)

    class _FakeCli:
        def method_sync(self, method, params=None):
            if method == "scope_get_horiz_coord":
                return {"result": [45.0, 0.0], "Timestamp": f"{time.time():.6f}"}
            return {"result": None}

    monkeypatch.setattr(lt, "get_mount_client", lambda *a, **kw: _FakeCli())

    session = LiveTrackSession(
        telescope_id=140,
        target_kind="file",
        target_id="fix",
        target_display_name="Fix",
        provider=_StationaryProvider(t0=time.time() + 0.2),
        offsets=AtomicOffsets(),
        dry_run=False,
        auto_slew=True,
        log_dir=tmp_path / "logs",
    )
    session._run()

    assert session._exit_reason == "sun_avoidance"
    assert session._phase == "refused"
    assert session._stop_evt.is_set()


def test_run_direct_motor_stop_bypasses_sun_lockout(tmp_path, monkeypatch):
    """During a sun-safety emergency lockout the wrapper-based exit stop is
    refused; _run must issue a direct scope_speed_move(0) so the motor still
    halts on session exit."""
    import device.live_tracker as lt
    from device.sun_safety import SunSafetyMonitor, get_sun_monitor, set_sun_monitor
    from tests.fakes.fake_mount import FakeMountClient

    cli = FakeMountClient()
    cli.set_position(az_deg=0.0, el_deg=45.0)
    monkeypatch.setattr(lt, "get_mount_client", lambda *a, **kw: cli)

    session = LiveTrackSession(
        telescope_id=150,
        target_kind="file",
        target_id="fix",
        target_display_name="Fix",
        provider=_StationaryProvider(t0=time.time(), duration_s=0.3),
        offsets=AtomicOffsets(),
        dry_run=False,
        auto_slew=False,
        log_dir=tmp_path / "logs",
    )

    prev = get_sun_monitor()
    m = SunSafetyMonitor(
        altaz_reader=lambda: None,
        jog_command=lambda *a, **k: None,
        lat_deg=0.0,
        lon_deg=0.0,
    )
    m._emergency_lockout.set()
    set_sun_monitor(m)
    try:
        session._run()
    finally:
        set_sun_monitor(prev)

    # The only scope_speed_move reaching the mount under lockout is the
    # direct bypass in _run's finally.
    assert cli.state.commands_received >= 1
    assert cli.state.last_cmd is not None
    assert cli.state.last_cmd[0] == 0


# --------- manager cross-exclusion (finding 9) ---------------------------


def _make_stationary_session(tid: int, tmp_path) -> LiveTrackSession:
    return LiveTrackSession(
        telescope_id=tid,
        target_kind="file",
        target_id="fix",
        target_display_name="Fix",
        provider=_StationaryProvider(t0=time.time() + 0.2),
        offsets=AtomicOffsets(),
        dry_run=True,
        auto_slew=False,
        log_dir=tmp_path / "logs",
    )


def test_live_track_manager_refuses_when_visibility_map_running(tmp_path, monkeypatch):
    import pytest
    import device.visibility_mapper as vm

    class _FakeVisMgr:
        def is_running(self, tid):
            return True

    monkeypatch.setattr(vm, "get_visibility_manager", lambda: _FakeVisMgr())

    mgr = LiveTrackManager()
    with pytest.raises(RuntimeError):
        mgr.start(_make_stationary_session(160, tmp_path))
    assert mgr.get(160) is None


def test_live_track_manager_refuses_when_nighttime_auto_running(tmp_path, monkeypatch):
    import pytest
    import device.nighttime_calibration as nc

    class _FakeAutoMgr:
        def is_running(self, tid):
            return True

        def get(self, tid):
            return None

    monkeypatch.setattr(nc, "get_nighttime_auto_manager", lambda: _FakeAutoMgr())

    mgr = LiveTrackManager()
    with pytest.raises(RuntimeError):
        mgr.start(_make_stationary_session(161, tmp_path))
    assert mgr.get(161) is None


# --------- M3: split feasibility gate (cable-wrap deferred to runtime) ----


class _WoundAzProvider:
    """Provider whose topocentric az_cum sits far outside any plausible
    cable-wrap budget, but whose elevation is in-band. Used to prove the
    pre-launch gate no longer rejects on the frame mismatch."""

    is_live = False
    extrapolation_s = 1.0

    def __init__(
        self, t0: float, az_cum_deg: float = 100_000.0, duration_s: float = 3.0
    ):
        self._t0 = t0
        self._t1 = t0 + duration_s
        self._az = float(az_cum_deg)

    def sample(self, t):
        return ReferenceSample(
            t_unix=float(t),
            az_cum_deg=self._az,
            el_deg=45.0,
            v_az_degs=0.0,
            v_el_degs=0.0,
            a_az_degs2=0.0,
            a_el_degs2=0.0,
            stale=False,
            extrapolated=False,
        )

    def valid_range(self):
        return (self._t0, self._t1)


def test_pre_check_feasibility_ignores_cable_wrap(tmp_path):
    """M3: pre_check_feasibility is frame-independent. A trajectory whose raw
    topocentric az_cum lies far outside the cable-wrap budget (but with
    elevation in-band) must NOT be rejected pre-launch — the provider's
    az_cum only reconciles with the mount's cumulative encoder budget once
    the runtime anchor offset is applied. (Before the fix this produced a
    false 400.)"""
    session = LiveTrackSession(
        telescope_id=170,
        target_kind="file",
        target_id="wound",
        target_display_name="Wound",
        provider=_WoundAzProvider(t0=time.time() + 0.2),
        offsets=AtomicOffsets(),
        dry_run=True,
        auto_slew=False,
        log_dir=tmp_path / "logs",
    )
    # Sanity: real cable limits are on disk, so the *old* behavior would have
    # compared az_cum=100000 against them and rejected.
    assert session._resolve_az_limits() is not None
    pre = session.pre_check_feasibility()
    assert pre.feasible is True
    # Elevation gating still works (frame-independent).
    assert pre.el_limit_violations == 0


def test_pre_check_feasibility_still_rejects_out_of_band_elevation(tmp_path):
    """M3: the frame-independent elevation gate survives the split — an
    overhead trajectory is still infeasible pre-launch."""
    session = LiveTrackSession(
        telescope_id=171,
        target_kind="file",
        target_id="overhead",
        target_display_name="Overhead",
        provider=_OverheadProvider(t0=time.time() + 0.2),
        offsets=AtomicOffsets(),
        dry_run=True,
        auto_slew=False,
        log_dir=tmp_path / "logs",
    )
    pre = session.pre_check_feasibility()
    assert pre.feasible is False
    assert pre.el_limit_violations > 0


def _cable_wrap_session(tid: int, tmp_path) -> LiveTrackSession:
    return LiveTrackSession(
        telescope_id=tid,
        target_kind="file",
        target_id="fix",
        target_display_name="Fix",
        provider=_StationaryProvider(t0=time.time() + 0.2),  # az_cum=0 flat
        offsets=AtomicOffsets(),
        dry_run=True,
        auto_slew=False,
        log_dir=tmp_path / "logs",
    )


def test_runtime_cable_wrap_check_fires_on_wound_mount(tmp_path):
    """M3: with the mount wound past the usable CW hard stop, the deferred
    runtime gate (evaluated against the measured anchor) refuses the track
    and ends the session with exit_reason='cable_wrap'."""
    from device.plant_limits import AzimuthLimits, CumulativeAzTracker

    session = _cable_wrap_session(170, tmp_path)

    class _FakeCli:
        def method_sync(self, method, params=None):
            if method == "scope_get_horiz_coord":
                # wrapped 85° ≡ cumulative 445° for a mount wound +445°
                return {"result": [45.0, 85.0], "Timestamp": f"{time.time():.6f}"}
            return {"result": None}

    tracker = CumulativeAzTracker()
    tracker.reset(cum_az_deg=445.0, wrapped_az_deg=85.0)  # past +435° usable CW
    limits = AzimuthLimits(
        ccw_hard_stop_cum_deg=-450.0,
        cw_hard_stop_cum_deg=450.0,
        padding_deg=15.0,
    )

    abort = session._runtime_cable_wrap_check(_FakeCli(), None, tracker, limits)
    assert abort is True
    assert session._exit_reason == "cable_wrap"
    assert session._phase == "refused"
    assert session._stop_evt.is_set()
    assert any("cable-wrap" in e for e in session._errors)


def test_runtime_cable_wrap_check_passes_within_budget(tmp_path):
    """M3: a mount near home with an in-budget trajectory is not aborted by
    the runtime gate."""
    from device.plant_limits import AzimuthLimits, CumulativeAzTracker

    session = _cable_wrap_session(172, tmp_path)

    class _FakeCli:
        def method_sync(self, method, params=None):
            if method == "scope_get_horiz_coord":
                return {"result": [45.0, 0.0], "Timestamp": f"{time.time():.6f}"}
            return {"result": None}

    tracker = CumulativeAzTracker()
    tracker.reset(cum_az_deg=0.0, wrapped_az_deg=0.0)  # home
    limits = AzimuthLimits(
        ccw_hard_stop_cum_deg=-450.0,
        cw_hard_stop_cum_deg=450.0,
        padding_deg=15.0,
    )

    abort = session._runtime_cable_wrap_check(_FakeCli(), None, tracker, limits)
    assert abort is False
    assert session._exit_reason is None
    assert not session._stop_evt.is_set()


def test_runtime_cable_wrap_check_noop_without_limits(tmp_path):
    """M3: with no cable limits configured, the runtime gate is a no-op (it
    never measures the mount or aborts)."""
    from device.plant_limits import CumulativeAzTracker

    session = _cable_wrap_session(173, tmp_path)
    measured = {"n": 0}

    class _FakeCli:
        def method_sync(self, method, params=None):
            measured["n"] += 1
            return {"result": [45.0, 0.0], "Timestamp": f"{time.time():.6f}"}

    tracker = CumulativeAzTracker()
    abort = session._runtime_cable_wrap_check(_FakeCli(), None, tracker, None)
    assert abort is False
    assert measured["n"] == 0  # short-circuited before touching the mount


# --------- M5: pre-slew sun check keys off sky, not mount frame -----------


def test_auto_slew_sun_check_transforms_mount_to_sky(monkeypatch):
    """M5: the pre-slew sun-avoidance check must key off SKY coordinates, not
    the raw mount-frame (az, el). With a yawed session frame the mount az
    differs from the sky az, so is_sun_safe must be queried with the
    un-rotated sky coordinates (mirroring the per-tick net in
    streaming_controller.track)."""
    import pytest

    import device.live_tracker as lt
    from device import sun_safety as ss
    from device.streaming_controller import _mount_azel_to_sky
    from device.target_frame import MountFrame

    monkeypatch.setattr(lt, "ensure_scenery_mode", lambda cli: None)

    captured: dict = {}

    def _spy_is_sun_safe(az, el, *a, **k):
        captured["az"] = float(az)
        captured["el"] = float(el)
        return (False, "sun_avoidance: spy")

    monkeypatch.setattr(ss, "is_sun_safe", _spy_is_sun_safe)

    yaw_frame = MountFrame.from_euler_deg(yaw_deg=30.0, pitch_deg=0.0, roll_deg=0.0)

    class _FramedProvider:
        def __init__(self, t0, mount_frame):
            self.mount_frame = mount_frame
            self._t0 = t0
            self._t1 = t0 + 3.0

        def sample(self, t):
            return ReferenceSample(
                t_unix=float(t),
                az_cum_deg=10.0,
                el_deg=40.0,
                v_az_degs=0.0,
                v_el_degs=0.0,
                a_az_degs2=0.0,
                a_el_degs2=0.0,
                stale=False,
                extrapolated=False,
            )

        def valid_range(self):
            return (self._t0, self._t1)

    class _FakeCli:
        def method_sync(self, method, params=None):
            if method == "scope_get_horiz_coord":
                return {"result": [45.0, 0.0], "Timestamp": f"{time.time():.6f}"}
            return {"result": None}

    session = LiveTrackSession(
        telescope_id=180,
        target_kind="file",
        target_id="fix",
        target_display_name="Fix",
        provider=_FramedProvider(time.time() + 0.2, yaw_frame),
        offsets=AtomicOffsets(),
        dry_run=True,
    )
    session._auto_slew(_FakeCli(), None)

    assert session._exit_reason == "sun_avoidance"
    mount_az = 10.0  # provider az_cum_deg wrapped into [0, 360)
    expected_az, expected_el = _mount_azel_to_sky(yaw_frame, mount_az, 40.0)
    assert captured["az"] == pytest.approx(expected_az, abs=1e-6)
    assert captured["el"] == pytest.approx(expected_el, abs=1e-6)
    # The transform actually moved the az off the raw mount value (yaw=30°).
    delta = ((captured["az"] - mount_az + 180.0) % 360.0) - 180.0
    assert abs(delta) > 1.0


def test_run_direct_motor_stop_skipped_while_sun_jog_in_progress(tmp_path, monkeypatch):
    """The session-exit direct stop must NOT cancel an in-flight emergency
    jog-away: with the monitor's jog window open, no scope_speed_move may
    reach the mount from the session at all (the jog's own firmware dur_sec
    bounds the motion)."""
    import time as _time

    import device.live_tracker as lt
    from device.sun_safety import SunSafetyMonitor, get_sun_monitor, set_sun_monitor
    from tests.fakes.fake_mount import FakeMountClient

    cli = FakeMountClient()
    cli.set_position(az_deg=0.0, el_deg=45.0)
    monkeypatch.setattr(lt, "get_mount_client", lambda *a, **kw: cli)

    session = LiveTrackSession(
        telescope_id=151,
        target_kind="file",
        target_id="fix",
        target_display_name="Fix",
        provider=_StationaryProvider(t0=time.time(), duration_s=0.3),
        offsets=AtomicOffsets(),
        dry_run=False,
        auto_slew=False,
        log_dir=tmp_path / "logs",
    )

    prev = get_sun_monitor()
    m = SunSafetyMonitor(
        altaz_reader=lambda: None,
        jog_command=lambda *a, **k: None,
        lat_deg=0.0,
        lon_deg=0.0,
    )
    m._emergency_lockout.set()
    with m._lock:
        m._jog_until_ts = _time.time() + 30.0
    set_sun_monitor(m)
    try:
        session._run()
    finally:
        set_sun_monitor(prev)

    # Wrapper stops are refused by the lockout AND the direct bypass is
    # skipped by the jog window — nothing reaches the mount.
    assert cli.state.last_cmd is None


def test_runtime_cable_wrap_check_fails_closed_on_measure_error(tmp_path):
    """The deferred gate must fail CLOSED: a transient anchor-measurement
    error refuses the track (origin/main refused synchronously before any
    motion; deferral must not turn an error into an unchecked launch)."""
    from device.plant_limits import AzimuthLimits, CumulativeAzTracker

    session = _cable_wrap_session(174, tmp_path)

    class _BrokenCli:
        def method_sync(self, method, params=None):
            raise RuntimeError("scope unreachable")

    tracker = CumulativeAzTracker()
    tracker.reset(cum_az_deg=0.0, wrapped_az_deg=0.0)
    limits = AzimuthLimits(
        ccw_hard_stop_cum_deg=-450.0,
        cw_hard_stop_cum_deg=450.0,
        padding_deg=15.0,
    )

    abort = session._runtime_cable_wrap_check(_BrokenCli(), None, tracker, limits)
    assert abort is True
    assert session._exit_reason == "cable_wrap_unverified"
    assert session._phase == "refused"
    assert session._stop_evt.is_set()
    assert any("could not be verified" in e for e in session._errors)
