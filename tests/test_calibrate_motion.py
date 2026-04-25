"""Tests for the calibrate-motion primitive.

Covers:
- ``StaticReferenceProvider`` — sample/jog/nudge/set_target semantics.
- ``CalibrateMotionSession`` — lifecycle against a FakeMountClient (real
  streaming controller, real plant simulation), convergence on
  ``set_target``, jog kinematics, settled detection, stop event.
- ``CalibrateMotionManager`` — start/stop/status registry and the
  cross-manager mutex against ``LiveTrackManager``.

The session-thread tests use ``dry_run=False`` against a fake mount
because the streaming controller's anchoring and command path is what
we're verifying. ``dry_run=True`` would skip ``speed_move`` and the test
couldn't observe convergence.
"""

from __future__ import annotations

import time

import pytest

from device.calibrate_motion import (
    CalibrateMotionManager,
    CalibrateMotionSession,
    DEFAULT_SETTLED_THRESHOLD_DEG,
    MotionStatus,
    StaticReferenceProvider,
    get_calibrate_motion_manager,
)
from device.reference_provider import ReferenceSample
from tests.fakes.fake_mount import FakeMountClient


# ---------- StaticReferenceProvider ---------------------------------


def test_static_provider_sample_returns_initial_target():
    p = StaticReferenceProvider(initial_az_deg=10.0, initial_el_deg=30.0)
    s = p.sample(time.time())
    assert isinstance(s, ReferenceSample)
    assert s.az_cum_deg == pytest.approx(10.0, abs=1e-9)
    assert s.el_deg == pytest.approx(30.0, abs=1e-9)
    assert s.v_az_degs == 0.0
    assert s.v_el_degs == 0.0
    assert not s.stale
    assert not s.extrapolated


def test_static_provider_nudge_target_stacks():
    p = StaticReferenceProvider(initial_az_deg=0.0, initial_el_deg=45.0)
    p.nudge_target(0.5, -0.25)
    p.nudge_target(0.1, 0.05)
    s = p.sample(time.time())
    assert s.az_cum_deg == pytest.approx(0.6, abs=1e-9)
    assert s.el_deg == pytest.approx(44.8, abs=1e-9)


def test_static_provider_set_target_resets_jog():
    p = StaticReferenceProvider(initial_az_deg=0.0, initial_el_deg=0.0)
    p.set_jog(2.0, 0.0)
    p.set_target(7.0, 35.0)
    az, el = p.get_jog()
    assert az == 0.0 and el == 0.0
    s = p.sample(time.time() + 0.1)
    # Just after set_target, jog is zero, so the sample at any time
    # should equal the target itself within a tick.
    assert s.az_cum_deg == pytest.approx(7.0, abs=1e-6)
    assert s.el_deg == pytest.approx(35.0, abs=1e-6)


def test_static_provider_jog_advances_target():
    """At 2 °/s for 0.5 s the target should have advanced by 1°.

    set_jog snaps the curve to ``time.time()`` (not a parameter), so we
    use a small abs tolerance to absorb the call-overhead delta between
    the snap timestamp and our reference t0.
    """
    p = StaticReferenceProvider(initial_az_deg=0.0, initial_el_deg=0.0)
    # set_jog runs first so its internal snap timestamp is the reference
    # the next sample uses.
    p.set_jog(2.0, -1.0)
    t_snap = time.time()
    s = p.sample(t_snap + 0.5)
    assert s.az_cum_deg == pytest.approx(1.0, abs=1e-3)
    assert s.el_deg == pytest.approx(-0.5, abs=1e-3)
    assert s.v_az_degs == pytest.approx(2.0, abs=1e-9)
    assert s.v_el_degs == pytest.approx(-1.0, abs=1e-9)


def test_static_provider_jog_change_bakes_progress_into_target():
    """Setting a new jog rate snaps the curve: the target's value at the
    moment of ``set_jog`` is preserved (baked into the anchor) and only
    the slope changes from there. So freezing after running at 2 °/s for
    Δt should leave ~``2·Δt`` baked into the target."""
    p = StaticReferenceProvider()
    p.set_jog(2.0, 0.0)
    # Wait long enough that the snap delta is measurable but short
    # enough that the test stays fast. 50 ms at 2 °/s → 0.1° baked.
    time.sleep(0.05)
    p.freeze_jog()
    # After freeze, the target stays put — sample at any later t gives
    # the same value as sample now.
    s_now = p.sample(time.time())
    s_later = p.sample(time.time() + 0.5)
    assert s_later.az_cum_deg == pytest.approx(s_now.az_cum_deg, abs=1e-9)
    # The baked-in advance must be roughly proportional to the elapsed
    # jog window. We deliberately allow a wide range because the actual
    # elapsed time depends on scheduler quirks.
    assert s_now.az_cum_deg > 0.05
    assert s_now.az_cum_deg < 1.0


def test_static_provider_freeze_jog_zeroes_velocity():
    p = StaticReferenceProvider()
    p.set_jog(3.0, -2.0)
    p.freeze_jog()
    az, el = p.get_jog()
    assert az == 0.0 and el == 0.0


def test_static_provider_extrapolation_s_attribute_present():
    """streaming_controller looks up ``provider.__dict__.get('extrapolation_s')``;
    instances must expose the attribute (large value so stale never trips)."""
    p = StaticReferenceProvider()
    assert getattr(p, "extrapolation_s", None) >= 1e6


def test_static_provider_valid_range_wide():
    p = StaticReferenceProvider()
    t0, t1 = p.valid_range()
    now = time.time()
    assert t0 <= now <= t1
    # At least a year of validity so the controller's "past tail" exit
    # never fires for a calibrate session that lives a few hours.
    assert t1 - now > 365 * 86400 - 1


# ---------- CalibrateMotionSession lifecycle (against FakeMountClient) ----


def _install_fake_cli(monkeypatch, cli):
    """Patch device.calibrate_motion.AlpacaClient so the session thread
    uses our in-process fake instead of hitting a real Alpaca endpoint."""
    import device.alpaca_client as ac

    monkeypatch.setattr(ac, "AlpacaClient", lambda *a, **kw: cli)


def test_session_start_then_stop_clean(monkeypatch, tmp_path):
    cli = FakeMountClient()
    cli.set_position(az_deg=10.0, el_deg=30.0)
    _install_fake_cli(monkeypatch, cli)
    session = CalibrateMotionSession(
        telescope_id=99,
        log_dir=tmp_path,
        max_duration_s=5.0,
    )
    session.start()
    try:
        # Allow at least one tick (default tick_dt = 0.5 s).
        time.sleep(1.0)
        assert session.is_alive()
        st = session.status()
        assert isinstance(st, MotionStatus)
        assert st.active
    finally:
        session.stop(timeout=3.0)
    assert not session.is_alive()
    final = session.status()
    assert final.exit_reason in {"stop_signal", "end_of_track", "timeout"}


def test_session_set_target_converges(monkeypatch, tmp_path):
    """Calling set_target should drive the FakeMount within
    DEFAULT_SETTLED_THRESHOLD_DEG of the requested target within a few seconds.

    The streaming controller anchors at the start position so target
    deltas (not absolutes) are what gets commanded — set_target(2, 32)
    when the fake starts at (0, 30) commands a +2°/+2° move.
    """
    cli = FakeMountClient()
    cli.set_position(az_deg=0.0, el_deg=30.0)
    _install_fake_cli(monkeypatch, cli)
    session = CalibrateMotionSession(
        telescope_id=99,
        log_dir=tmp_path,
        max_duration_s=20.0,
        settled_threshold_deg=0.05,
        settled_ticks=3,
    )
    session.start()
    try:
        # Wait one tick so anchor is established before set_target.
        time.sleep(0.6)
        session.set_target(2.0, 2.0)
        deadline = time.time() + 12.0
        while time.time() < deadline:
            if session.is_settled():
                break
            time.sleep(0.2)
        st = session.status()
        # The fake mount integrates a first-order plant; we expect to be
        # very close to (anchor_az + 2, anchor_el + 2). Anchor was
        # (0, 30); target frame anchored to that, so encoder ≈ (2, 32).
        assert abs(cli.state.az_wrapped_deg - 2.0) < 0.1
        assert abs(cli.state.el_deg - 32.0) < 0.1
        assert st.is_settled
    finally:
        session.stop(timeout=3.0)


def test_session_jog_moves_target(monkeypatch, tmp_path):
    """Set a jog rate and verify the target field of /state advances."""
    cli = FakeMountClient()
    cli.set_position(az_deg=5.0, el_deg=40.0)
    _install_fake_cli(monkeypatch, cli)
    session = CalibrateMotionSession(
        telescope_id=99,
        log_dir=tmp_path,
        max_duration_s=10.0,
    )
    session.start()
    try:
        time.sleep(0.6)
        session.set_jog(1.0, 0.0)
        time.sleep(1.0)
        # The provider's get_target should reflect ~+1° on az from where
        # it started after 1 s at 1 °/s.
        az, _el = session.provider.get_target()
        # Allow generous slop on timing — the sample function just needs
        # to be advancing.
        assert az > 0.5
        session.freeze_jog()
        # After freeze, the target value must stay roughly stable.
        az_frozen, _ = session.provider.get_target()
        time.sleep(0.5)
        az_after, _ = session.provider.get_target()
        assert abs(az_after - az_frozen) < 0.05
    finally:
        session.stop(timeout=3.0)


def test_session_double_start_raises(monkeypatch, tmp_path):
    cli = FakeMountClient()
    _install_fake_cli(monkeypatch, cli)
    session = CalibrateMotionSession(
        telescope_id=99,
        log_dir=tmp_path,
        max_duration_s=2.0,
    )
    session.start()
    try:
        with pytest.raises(RuntimeError):
            session.start()
    finally:
        session.stop(timeout=3.0)


def test_session_outer_motor_stop_bypasses_sun_safety_lockout(monkeypatch, tmp_path):
    """If the streaming controller's inner ``speed_move(0, 0, ...)`` cleanup
    is refused mid-lockout (``SunSafetyLocked``), the session's outer
    finally must still issue a direct ``method_sync('scope_speed_move',
    {0,0,1})`` so the motor halts. Without the outer guard the previous
    tick's command keeps running until its firmware ``dur_sec`` TTL.
    """
    import device.streaming_controller as sc
    from device.sun_safety import SunSafetyLocked

    cli = FakeMountClient()
    cli.set_position(az_deg=0.0, el_deg=30.0)
    _install_fake_cli(monkeypatch, cli)

    # Tick command observed by the fake before stop (non-zero so we can
    # distinguish a successful zero-cleanup from "no cleanup ran").
    cli.method_sync(
        "scope_speed_move",
        {"speed": 100, "angle": 0, "dur_sec": 5},
    )
    assert cli.state.last_cmd == (100, 0, 5)

    # Refuse all subsequent ``speed_move`` calls to simulate sun-safety
    # lockout. ``streaming_controller.track``'s tick loop and inner
    # ``finally`` both go through this wrapper, so neither will issue a
    # zero command. Only the outer ``method_sync`` we added to
    # ``CalibrateMotionSession._run`` can clear the motor.
    def _refused(*_a, **_kw):
        raise SunSafetyLocked("test: lockout active")

    monkeypatch.setattr(sc, "speed_move", _refused)

    session = CalibrateMotionSession(
        telescope_id=99,
        log_dir=tmp_path,
        max_duration_s=5.0,
    )
    session.start()
    try:
        time.sleep(0.6)
    finally:
        session.stop(timeout=3.0)
    assert not session.is_alive()
    assert cli.state.last_cmd == (0, 0, 1), (
        "outer motor-stop did not bypass speed_move lockout; "
        f"last_cmd={cli.state.last_cmd}"
    )


def test_is_settled_false_until_history_full(monkeypatch, tmp_path):
    cli = FakeMountClient()
    _install_fake_cli(monkeypatch, cli)
    session = CalibrateMotionSession(
        telescope_id=99,
        log_dir=tmp_path,
        max_duration_s=5.0,
        settled_ticks=10,
    )
    # Before start: no history → not settled.
    assert not session.is_settled()
    session.start()
    try:
        # Just one tick → still under the 10-tick window.
        time.sleep(0.6)
        assert not session.is_settled(threshold_deg=10.0, ticks=10)
    finally:
        session.stop(timeout=3.0)


# ---------- CalibrateMotionManager ----------------------------------


def test_manager_singleton_returns_same_instance():
    a = get_calibrate_motion_manager()
    b = get_calibrate_motion_manager()
    assert a is b


def test_manager_start_stop_roundtrip(monkeypatch, tmp_path):
    cli = FakeMountClient()
    _install_fake_cli(monkeypatch, cli)
    mgr = CalibrateMotionManager()
    session = CalibrateMotionSession(
        telescope_id=77,
        log_dir=tmp_path,
        max_duration_s=3.0,
    )
    mgr.start(session)
    try:
        time.sleep(0.6)
        st = mgr.status(77)
        assert st is not None and st.active
        # Same session is returned by get().
        assert mgr.get(77) is session
    finally:
        mgr.stop(77)
    final = mgr.status(77)
    assert final is not None and not final.active


def test_manager_double_start_refuses(monkeypatch, tmp_path):
    cli = FakeMountClient()
    _install_fake_cli(monkeypatch, cli)
    mgr = CalibrateMotionManager()
    session_a = CalibrateMotionSession(
        telescope_id=88,
        log_dir=tmp_path,
        max_duration_s=3.0,
    )
    mgr.start(session_a)
    try:
        session_b = CalibrateMotionSession(
            telescope_id=88,
            log_dir=tmp_path,
            max_duration_s=3.0,
        )
        with pytest.raises(RuntimeError, match="already in calibrate-motion mode"):
            mgr.start(session_b)
    finally:
        mgr.stop(88)


def test_manager_refuses_when_live_tracker_running(monkeypatch, tmp_path):
    """Cross-manager mutex: refuse to start a motion session if the live
    tracker is already running on the same telescope."""
    import device.live_tracker as lt

    # Stand up a fake live-tracker session on telescope 66 just so the
    # cross-check finds an alive() session. The real session would call
    # streaming_controller.track; we just need is_alive() to return True
    # so the mutex check trips. Patching get_manager to a fake manager
    # is cleaner than spinning a real one.
    class _FakeAliveSession:
        def is_alive(self) -> bool:
            return True

    class _FakeTrackerMgr:
        def __init__(self):
            self.session = _FakeAliveSession()

        def get(self, tid):
            return self.session

    fake_mgr = _FakeTrackerMgr()
    monkeypatch.setattr(lt, "get_manager", lambda: fake_mgr)

    mgr = CalibrateMotionManager()
    cli = FakeMountClient()
    _install_fake_cli(monkeypatch, cli)
    session = CalibrateMotionSession(
        telescope_id=66,
        log_dir=tmp_path,
        max_duration_s=2.0,
    )
    with pytest.raises(RuntimeError, match="live-tracking"):
        mgr.start(session)
    # Session was never started, so no thread to clean up.
    assert not session.is_alive()


def test_live_tracker_refuses_when_motion_running(monkeypatch, tmp_path):
    """Reverse cross-manager mutex: live tracker must refuse if motion
    session is already alive on the same telescope."""
    import device.calibrate_motion as cm
    import device.live_tracker as lt

    class _FakeAliveMotion:
        def is_alive(self) -> bool:
            return True

    class _FakeMotionMgr:
        def __init__(self):
            self.s = _FakeAliveMotion()

        def is_running(self, tid):
            return True

        def get(self, tid):
            return self.s

    monkeypatch.setattr(cm, "get_calibrate_motion_manager", lambda: _FakeMotionMgr())

    # Build a minimal LiveTrackSession against a fake provider.
    from device.live_tracker import (
        AtomicOffsets,
        LiveTrackManager,
        LiveTrackSession,
    )
    from device.reference_provider import ReferenceSample

    class _StationaryProvider:
        def __init__(self):
            self._t0 = time.time() + 0.2
            self._t1 = self._t0 + 3.0

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

    monkeypatch.setattr(lt, "AlpacaClient", lambda *a, **kw: object())
    session = LiveTrackSession(
        telescope_id=66,
        target_kind="file",
        target_id="x",
        target_display_name="X",
        provider=_StationaryProvider(),
        offsets=AtomicOffsets(),
        dry_run=True,
        log_dir=tmp_path,
    )
    mgr = LiveTrackManager()
    with pytest.raises(RuntimeError, match="calibrate-motion"):
        mgr.start(session)
    assert not session.is_alive()


# ---------- CalibrationSession motion delegation -----------------------


def test_calibration_session_nudge_delegates_to_motion(monkeypatch, tmp_path):
    """When a calibrate-motion session is alive on this telescope,
    CalibrationSession.nudge must update the motion target rather than
    issuing move_to_ff. Otherwise the small UI increments are truncated
    by move_to_ff's 0.1° arrive-tolerance."""
    import device.calibrate_motion as cm

    # Build a stub motion session that records nudge_target calls but
    # doesn't run a streaming thread. The CalibrationSession code path
    # only calls is_alive(), nudge_target(), set_target(), is_settled(),
    # status() — we provide just those.
    class _StubMotion:
        def __init__(self):
            self.az = 0.0
            self.el = 0.0
            self.calls = []
            self._alive = True

        def is_alive(self):
            return self._alive

        def nudge_target(self, daz, del_):
            self.az += daz
            self.el += del_
            self.calls.append(("nudge", daz, del_))

        def set_target(self, az, el):
            self.az = az
            self.el = el
            self.calls.append(("set_target", az, el))

        def is_settled(self, threshold_deg=None, ticks=None):
            return True

        def status(self):
            return MotionStatus(
                active=True,
                phase="track",
                elapsed_s=1.0,
                exit_reason=None,
                target_az_deg=self.az,
                target_el_deg=self.el,
                cur_cum_az_deg=self.az,
                cur_el_deg=self.el,
                err_az_deg=0.0,
                err_el_deg=0.0,
                jog_az_degs=0.0,
                jog_el_degs=0.0,
                is_settled=True,
                tick=10,
            )

    stub_motion = _StubMotion()

    class _StubMgr:
        def get(self, tid):
            return stub_motion

    monkeypatch.setattr(cm, "get_calibrate_motion_manager", lambda: _StubMgr())

    # Now exercise CalibrationSession._on_nudge directly.
    from device.rotation_calibration import CalibrationSession

    # Build a minimal session — the targets list must have at least one
    # entry; we only call _on_nudge directly so the worker thread does
    # not need to start.
    from scripts.trajectory.faa_dof import HYPERION_06_000301
    from scripts.trajectory.observer import build_site

    site = build_site(lat_deg=33.96, lon_deg=-118.46, alt_m=2.0)
    az, el, slant = 280.0, 1.0, 5500.0
    session = CalibrationSession(
        telescope_id=42,
        targets=[(HYPERION_06_000301, az, el, slant)],
        site=site,
        out_path=tmp_path / "calibration.json",
        dry_run=False,
    )
    # Seed pending target so _on_nudge has a baseline.
    session._target_az = 100.0
    session._target_el = 30.0

    # No mount connection needed when motion is delegated.
    class _NoMount:
        def method_sync(self, *a, **kw):
            return {"result": None}

    session._on_nudge(_NoMount(), 0.005, -0.002)
    # The motion stub should have received the nudge.
    assert stub_motion.calls, "motion delegation did not receive nudge_target"
    kind, daz, del_ = stub_motion.calls[-1]
    assert kind == "nudge"
    assert daz == pytest.approx(0.005)
    assert del_ == pytest.approx(-0.002)


def test_motion_session_threshold_default_below_smallest_increment():
    """The default settled threshold must be tighter than the smallest UI
    nudge so a single 0.005° nudge actually shows ``settled=False`` until
    the mount has moved through it."""
    assert DEFAULT_SETTLED_THRESHOLD_DEG <= 0.0025 + 1e-9
