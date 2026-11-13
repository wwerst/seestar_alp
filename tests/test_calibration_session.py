"""Tests for device.rotation_calibration.CalibrationSession.

Thread-based lifecycle tests against a fake Alpaca client. The fake
captures commands and fabricates plausible scope_get_horiz_coord /
speed_move responses so the session can drive the full nudge→sight
→solve→commit flow without a mount or a network.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from device.rotation_calibration import (
    CalibrationManager,
    CalibrationSession,
    MAX_NUDGE_PER_CMD_DEG,
    get_calibration_manager,
)
from scripts.trajectory.faa_dof import (
    HYPERION_06_000301,
    LA_BROADCAST_06_000177,
    filter_visible,
)
from scripts.trajectory.observer import build_site


DOCKWEILER = dict(lat_deg=33.9615051, lon_deg=-118.4581361, alt_m=2.0)


def _site():
    return build_site(**DOCKWEILER)


def _targets(site):
    """The two visible defaults, as ``(lm, az, el, slant)`` tuples —
    same shape the CLI and web backends pass to the session.

    Uses ``preserve_order=True`` to mirror the production caller in
    ``front/app.py``: DEFAULT_LANDMARKS is hand-curated with Hyperion
    first (closer, lit, primary calibration target). Without this
    flag, ``filter_visible`` height-sorts and the taller LA broadcast
    tower leads instead.
    """
    return filter_visible(
        [HYPERION_06_000301, LA_BROADCAST_06_000177],
        site,
        min_el_deg=0.0,
        preserve_order=True,
    )


class _FakeCli:
    """Minimal AlpacaClient stand-in. Tracks az/el in an internal
    'encoder' state and responds to ``scope_get_horiz_coord``. Accepts
    method calls from ``move_to_ff``'s internals by returning benign
    empty results; the session patches ``move_to_ff`` itself."""

    def __init__(self):
        self.encoder_az = 0.0
        self.encoder_el = 0.0
        self.calls: list[tuple[str, object]] = []

    def method_sync(self, method, params=None):
        self.calls.append((method, params))
        if method == "scope_get_horiz_coord":
            return {
                "result": [float(self.encoder_el), float(self.encoder_az)],
                "Timestamp": f"{time.time():.6f}",
            }
        return {"result": None}


def _install_fakes(monkeypatch, cli: _FakeCli, *, neutralise_sun=False):
    """Replace the per-call imports inside CalibrationSession with
    lightweight fakes. ``move_to_ff`` updates the cli's encoder state
    to ``(target_az, target_el)`` and returns (new_el, new_az, {}).

    When ``neutralise_sun`` is true, ``device.sun_safety.is_sun_safe`` is
    monkeypatched to always return ``(True, '')`` so a celestial /
    plate-solve test pinned to a specific (az, el) doesn't fail when
    the wall-clock places the sun there.
    """
    import device.rotation_calibration as rc

    monkeypatch.setattr(
        "device.alpaca_client.AlpacaClient",
        lambda *a, **kw: cli,
    )

    def fake_move_to_ff(
        client,
        *,
        target_az_deg,
        target_el_deg,
        cur_az_deg,
        cur_el_deg,
        loc,
        tag="",
        arrive_tolerance_deg=0.3,
    ):
        # Teleport the fake mount to the requested target, mimicking
        # perfect convergence.
        client.encoder_az = float(target_az_deg)
        client.encoder_el = float(target_el_deg)
        return float(target_el_deg), float(target_az_deg), {"converged": True}

    def fake_measure_altaz_timed(client, loc):
        return float(client.encoder_el), float(client.encoder_az), time.time()

    def fake_ensure_scenery_mode(client):
        return None

    def fake_set_tracking(client, enabled):
        return None

    monkeypatch.setattr("device.velocity_controller.move_to_ff", fake_move_to_ff)
    monkeypatch.setattr(
        "device.velocity_controller.measure_altaz_timed",
        fake_measure_altaz_timed,
    )
    monkeypatch.setattr(
        "device.velocity_controller.ensure_scenery_mode",
        fake_ensure_scenery_mode,
    )
    monkeypatch.setattr(
        "device.velocity_controller.set_tracking",
        fake_set_tracking,
    )
    if neutralise_sun:
        from device import sun_safety as ss

        monkeypatch.setattr(ss, "is_sun_safe", lambda *a, **kw: (True, ""))
    # Make Config.port resolution cheap.
    import device.config as cfg

    monkeypatch.setattr(cfg.Config, "port", 5555, raising=False)
    return rc


# --------- lifecycle ----------------------------------------------


def _wait_for_phase(session: CalibrationSession, phase: str, timeout_s: float = 2.0):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if session.status().phase == phase:
            return
        time.sleep(0.02)
    raise AssertionError(
        f"phase never reached {phase!r} (got {session.status().phase!r}); "
        f"errors={session.status().errors}"
    )


def test_session_starts_and_slews_to_first_target(monkeypatch, tmp_path):
    site = _site()
    targets = _targets(site)
    assert len(targets) >= 2
    cli = _FakeCli()
    _install_fakes(monkeypatch, cli)

    session = CalibrationSession(
        telescope_id=1,
        targets=targets,
        site=site,
        out_path=tmp_path / "cal.json",
    )
    session.start()
    try:
        _wait_for_phase(session, "nudging")
        st = session.status()
        assert st.target_idx == 0
        # With preserve_order=True (the production path), Hyperion
        # leads — it is the closer, lit primary calibration target.
        assert st.current_landmark["oas"] == "06-000301"
        assert "Hyperion" in st.current_landmark["name"]
        # Commit 4: status surfaces aiming_hint + FAA σ_az / σ_el so the
        # UI doesn't have to re-derive them.
        assert "aiming_hint" in st.current_landmark
        assert isinstance(st.current_landmark["aiming_hint"], str)
        assert st.current_landmark["aiming_hint"]
        assert "sigma_az_deg" in st.current_landmark
        assert "sigma_el_deg" in st.current_landmark
        # Hyperion is 1A — both σ should be finite small floats.
        assert 0 < st.current_landmark["sigma_az_deg"] < 0.5
        assert 0 < st.current_landmark["sigma_el_deg"] < 0.5
        # After the first slew, encoder should match the target (fake
        # teleports perfectly).
        assert st.encoder_az_deg == pytest.approx(st.target_az_deg, abs=1e-6)
        assert st.encoder_el_deg == pytest.approx(st.target_el_deg, abs=1e-6)
    finally:
        session.stop()


def test_slew_to_target_refuses_when_landmark_inside_sun_cone(monkeypatch, tmp_path):
    """Spec: CalibrationSession._slew_to_target must call
    device.sun_safety.is_sun_safe against the landmark's true
    topocentric (az, el) and refuse — phase='error', sun_avoidance
    recorded in errors — rather than commanding the mount toward it."""
    site = _site()
    cli = _FakeCli()
    _install_fakes(monkeypatch, cli)

    from device import sun_safety as ss

    monkeypatch.setattr(
        ss,
        "is_sun_safe",
        lambda *a, **kw: (False, "sun_avoidance: forced by test"),
    )

    session = CalibrationSession(
        telescope_id=1,
        targets=_targets(site),
        site=site,
        out_path=tmp_path / "cal.json",
    )
    session.start()
    try:
        deadline = time.time() + 2.0
        while time.time() < deadline:
            if session.status().phase == "error":
                break
            time.sleep(0.02)
        st = session.status()
        assert st.phase == "error", f"phase={st.phase} errors={st.errors}"
        assert any("sun_avoidance" in e for e in st.errors)
    finally:
        session.stop()


def test_run_finally_skips_raw_stop_during_sun_safety_jog(monkeypatch, tmp_path):
    """The worker's ``finally`` normally issues a raw
    ``scope_speed_move(speed=0)`` to halt lingering motion. While the
    sun-safety monitor's emergency jog-away is executing it must skip
    that stop — issuing it would cancel the jog-away and strand the mount
    inside the sun cone. The jog's firmware ``dur_sec`` bounds the motion,
    so skipping cannot leave the motor running."""
    site = _site()
    cli = _FakeCli()
    _install_fakes(monkeypatch, cli, neutralise_sun=True)
    monkeypatch.setattr(
        "device.sun_safety.sun_safety_jog_in_progress", lambda tid=None: True
    )

    session = CalibrationSession(
        telescope_id=1,
        targets=_targets(site),
        site=site,
        out_path=tmp_path / "cal.json",
    )
    # Pre-set the stop event so ``_run`` returns straight after the first
    # slew and drops into the ``finally`` we're exercising, no thread
    # timing involved.
    session._stop_evt.set()
    session._run()

    speed_moves = [p for (m, p) in cli.calls if m == "scope_speed_move"]
    assert speed_moves == [], (
        "worker finally issued a raw motor stop during the sun-safety jog: "
        f"{speed_moves}"
    )


def test_run_finally_issues_raw_stop_when_no_jog(monkeypatch, tmp_path):
    """Companion to the jog-skip test: with no sun-safety jog in flight the
    worker's ``finally`` must still issue the raw
    ``scope_speed_move(speed=0)`` so any lingering motion halts on exit."""
    site = _site()
    cli = _FakeCli()
    _install_fakes(monkeypatch, cli, neutralise_sun=True)
    monkeypatch.setattr(
        "device.sun_safety.sun_safety_jog_in_progress", lambda tid=None: False
    )

    session = CalibrationSession(
        telescope_id=1,
        targets=_targets(site),
        site=site,
        out_path=tmp_path / "cal.json",
    )
    session._stop_evt.set()
    session._run()

    speed_moves = [p for (m, p) in cli.calls if m == "scope_speed_move"]
    assert speed_moves, "worker finally did not issue the session-exit motor stop"
    assert speed_moves[-1] == {"speed": 0, "angle": 0, "dur_sec": 0}


def test_nudge_updates_target_and_encoder(monkeypatch, tmp_path):
    site = _site()
    cli = _FakeCli()
    _install_fakes(monkeypatch, cli)
    session = CalibrationSession(
        telescope_id=1,
        targets=_targets(site),
        site=site,
        out_path=tmp_path / "cal.json",
    )
    session.start()
    try:
        _wait_for_phase(session, "nudging")
        before = session.status()
        session.nudge(0.5, -0.3)
        time.sleep(0.3)
        after = session.status()
        assert after.target_az_deg == pytest.approx(
            before.target_az_deg + 0.5, abs=1e-6
        )
        assert after.target_el_deg == pytest.approx(
            before.target_el_deg - 0.3, abs=1e-6
        )
        assert after.encoder_az_deg == pytest.approx(after.target_az_deg, abs=1e-6)
    finally:
        session.stop()


def test_nudge_per_command_is_clamped(monkeypatch, tmp_path):
    site = _site()
    cli = _FakeCli()
    _install_fakes(monkeypatch, cli)
    session = CalibrationSession(
        telescope_id=1,
        targets=_targets(site),
        site=site,
        out_path=tmp_path / "cal.json",
    )
    session.start()
    try:
        _wait_for_phase(session, "nudging")
        before = session.status()
        # Request an absurd +45° nudge; should clamp to MAX_NUDGE_PER_CMD_DEG.
        session.nudge(45.0, 0.0)
        time.sleep(0.3)
        after = session.status()
        assert after.target_az_deg == pytest.approx(
            before.target_az_deg + MAX_NUDGE_PER_CMD_DEG,
            abs=1e-6,
        )
    finally:
        session.stop()


def test_full_flow_sight_advance_commit(monkeypatch, tmp_path):
    """Full happy path: sight two landmarks, commit, verify JSON."""
    site = _site()
    targets = _targets(site)
    cli = _FakeCli()
    _install_fakes(monkeypatch, cli)
    out = tmp_path / "cal.json"
    session = CalibrationSession(
        telescope_id=1,
        targets=targets,
        site=site,
        out_path=out,
    )
    session.start()
    try:
        _wait_for_phase(session, "nudging")
        session.sight()
        # After sighting target 0 (Hyperion), session auto-slews to
        # target 1 (LA broadcast).
        time.sleep(0.4)
        st1 = session.status()
        assert st1.target_idx == 1
        assert st1.current_landmark["oas"] == "06-000177"  # LA broadcast second
        assert st1.solution is not None
        assert len(st1.solution["per_landmark"]) == 1
        session.sight()
        _wait_for_phase(session, "review")
        st2 = session.status()
        assert st2.solution is not None
        assert len(st2.solution["per_landmark"]) == 2
        session.commit()
        _wait_for_phase(session, "committed")
    finally:
        session.stop()

    assert out.exists()
    payload = json.loads(out.read_text())
    assert payload["calibration_method"] == "rotation_landmarks"
    assert payload["n_landmarks"] == 2
    assert set(payload["observer"].keys()) >= {"lat_deg", "lon_deg", "alt_m"}


def test_commit_refuses_with_fewer_than_two_sightings(monkeypatch, tmp_path):
    site = _site()
    cli = _FakeCli()
    _install_fakes(monkeypatch, cli)
    out = tmp_path / "cal.json"
    session = CalibrationSession(
        telescope_id=1,
        targets=_targets(site),
        site=site,
        out_path=out,
    )
    session.start()
    try:
        _wait_for_phase(session, "nudging")
        session.sight()
        time.sleep(0.3)
        # Only one sighting recorded; commit should refuse.
        session.commit()
        time.sleep(0.2)
        errors = session.status().errors
        assert any("need ≥ 2 sightings" in e for e in errors)
        assert not out.exists()
    finally:
        session.stop()


def test_skip_refused_when_would_leave_under_two(monkeypatch, tmp_path):
    """With 2 targets and 0 sightings, skipping target 0 would leave
    only 1 remaining → projected total < 2 → refused."""
    site = _site()
    cli = _FakeCli()
    _install_fakes(monkeypatch, cli)
    session = CalibrationSession(
        telescope_id=1,
        targets=_targets(site),
        site=site,
        out_path=tmp_path / "cal.json",
    )
    session.start()
    try:
        _wait_for_phase(session, "nudging")
        session.skip()
        time.sleep(0.2)
        errs = session.status().errors
        assert any("fewer than 2 sightings" in e for e in errs)
        assert session.status().target_idx == 0  # did not advance
    finally:
        session.stop()


def test_cancel_terminates_worker(monkeypatch, tmp_path):
    site = _site()
    cli = _FakeCli()
    _install_fakes(monkeypatch, cli)
    session = CalibrationSession(
        telescope_id=1,
        targets=_targets(site),
        site=site,
        out_path=tmp_path / "cal.json",
    )
    session.start()
    try:
        _wait_for_phase(session, "nudging")
        session.cancel()
        _wait_for_phase(session, "cancelled")
    finally:
        session.stop()
    assert not session.is_alive()


# --------- manager cross-check ------------------------------------


def test_manager_refuses_concurrent_calibration(monkeypatch, tmp_path):
    site = _site()
    cli = _FakeCli()
    _install_fakes(monkeypatch, cli)
    mgr = CalibrationManager()

    def _new():
        return CalibrationSession(
            telescope_id=99,
            targets=_targets(site),
            site=site,
            out_path=tmp_path / "cal.json",
        )

    s1 = mgr.start(_new())
    try:
        _wait_for_phase(s1, "nudging")
        with pytest.raises(RuntimeError, match="already calibrating"):
            mgr.start(_new())
    finally:
        mgr.stop(99)


def test_manager_refuses_while_tracker_running(monkeypatch, tmp_path):
    """Live Tracker session on the same telescope must block a new
    calibration session."""
    from device.live_tracker import (
        AtomicOffsets,
        LiveTrackManager,
        LiveTrackSession,
    )
    from device.reference_provider import ReferenceSample

    site = _site()
    cli = _FakeCli()
    _install_fakes(monkeypatch, cli)

    class _StationaryProvider:
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
            return (time.time(), time.time() + 5.0)

    tracker = LiveTrackSession(
        telescope_id=7,
        target_kind="file",
        target_id="fix",
        target_display_name="Fix",
        provider=_StationaryProvider(),
        offsets=AtomicOffsets(),
        dry_run=True,
        log_dir=tmp_path / "logs",
    )
    tracker_mgr = LiveTrackManager()
    # Patch the module-global get_manager to return *this* manager so
    # CalibrationManager's cross-check finds the running tracker.
    import device.live_tracker as lt

    monkeypatch.setattr(lt, "_MANAGER", tracker_mgr)
    monkeypatch.setattr(lt, "AlpacaClient", lambda *a, **kw: cli)
    tracker_mgr.start(tracker)
    try:
        cal_mgr = CalibrationManager()
        with pytest.raises(RuntimeError, match="is live-tracking"):
            cal_mgr.start(
                CalibrationSession(
                    telescope_id=7,
                    targets=_targets(site),
                    site=site,
                    out_path=tmp_path / "cal.json",
                )
            )
    finally:
        tracker_mgr.stop(7)


def test_get_calibration_manager_is_singleton():
    assert get_calibration_manager() is get_calibration_manager()


# ---------- unified target-spec session ----------------------------


def _faa_specs(site):
    """Helper: build a list of FAA target specs from the visible
    defaults. Mirrors what CalibrationStartResource passes through
    its translation step."""
    from device.calibration_targets import CalibrationTargetSpec

    hits = _targets(site)
    return [
        CalibrationTargetSpec.from_landmark(lm, slant_m=float(slant))
        for (lm, _az, _el, slant) in hits
    ]


def test_from_landmarks_factory_produces_working_faa_session(monkeypatch, tmp_path):
    """The legacy factory must round-trip the existing FAA-only
    targets list through the unified session unchanged."""
    site = _site()
    cli = _FakeCli()
    _install_fakes(monkeypatch, cli)
    session = CalibrationSession.from_landmarks(
        telescope_id=10,
        targets=_targets(site),
        site=site,
        out_path=tmp_path / "cal.json",
    )
    session.start()
    try:
        _wait_for_phase(session, "nudging")
        st = session.status()
        assert st.n_targets == 2
        assert st.current_landmark["kind"] == "faa"
        assert st.current_landmark["oas"] == "06-000301"
        # The new ``targets`` list mirrors the spec pool.
        assert st.targets is not None
        assert all(t["kind"] == "faa" for t in st.targets)
    finally:
        session.stop()


def test_targets_and_target_specs_are_mutually_exclusive(monkeypatch, tmp_path):
    site = _site()
    _install_fakes(monkeypatch, _FakeCli())

    specs = _faa_specs(site)
    with pytest.raises(ValueError, match="not both"):
        CalibrationSession(
            telescope_id=11,
            targets=_targets(site),
            target_specs=specs,
            site=site,
            out_path=tmp_path / "cal.json",
        )


def test_session_requires_target_specs_or_targets(monkeypatch, tmp_path):
    site = _site()
    _install_fakes(monkeypatch, _FakeCli())
    with pytest.raises(ValueError, match="need either target_specs or targets"):
        CalibrationSession(
            telescope_id=12,
            site=site,
            out_path=tmp_path / "cal.json",
        )


def test_mixed_faa_and_celestial_session_runs_through_sighting(monkeypatch, tmp_path):
    """A mixed (FAA, celestial) session must drive both targets and
    fit a 3-DOF rotation. The celestial truth is mocked so the test is
    deterministic regardless of wall-clock."""
    from device.calibration_targets import CalibrationTargetSpec, TargetKind

    site = _site()
    cli = _FakeCli()
    _install_fakes(monkeypatch, cli, neutralise_sun=True)
    # Pin celestial resolution to a known answer so the fit is
    # deterministic. Patch resolve_true_altaz on the spec via
    # ``_resolve_celestial`` — the simplest deterministic stub.
    fixed_celestial = (
        45.0,  # az
        60.0,  # el
        None,
    )

    def _fake_celestial(self, site_, when_utc):
        return fixed_celestial

    monkeypatch.setattr(CalibrationTargetSpec, "_resolve_celestial", _fake_celestial)
    faa_specs = _faa_specs(site)
    celestial_spec = CalibrationTargetSpec.celestial(
        "Vega", ra_hours=18.6157, dec_deg=38.7837, vmag=0.03
    )
    session = CalibrationSession(
        telescope_id=21,
        target_specs=[faa_specs[0], celestial_spec],
        site=site,
        out_path=tmp_path / "cal.json",
    )
    session.start()
    try:
        _wait_for_phase(session, "nudging")
        # Sight the FAA target first.
        session.sight()
        time.sleep(0.4)
        st = session.status()
        assert st.target_idx == 1
        assert st.current_landmark["kind"] == TargetKind.CELESTIAL.value
        # Sight the celestial target.
        session.sight()
        _wait_for_phase(session, "review")
        st2 = session.status()
        assert st2.solution is not None
        assert len(st2.solution["per_landmark"]) == 2
        kinds = {r["kind"] for r in st2.solution["per_landmark"]}
        assert kinds == {"faa", "celestial"}
    finally:
        session.stop()


def test_celestial_only_session_writes_unified_or_celestial_method(
    monkeypatch, tmp_path
):
    """Three celestial sightings produce a 3-DOF fit and the
    written calibration tags ``calibration_method`` based on the
    record set's kinds."""
    from device.calibration_targets import CalibrationTargetSpec, TargetKind

    site = _site()
    cli = _FakeCli()
    _install_fakes(monkeypatch, cli, neutralise_sun=True)

    # Spec-keyed canned truth so slew + sight for the same target
    # always agree (encoder == truth → residual ≈ 0). Ordering makes
    # sightings well-separated on the sky so the 3-DOF LM fit isn't
    # degenerate.
    truth_by_label = {
        "Vega": (45.0, 60.0, None),
        "Capella": (135.0, 50.0, None),
        "Arcturus": (225.0, 45.0, None),
    }

    def _fake_celestial(self, site_, when_utc):
        return truth_by_label[self.label]

    monkeypatch.setattr(CalibrationTargetSpec, "_resolve_celestial", _fake_celestial)
    specs = [
        CalibrationTargetSpec.celestial(
            "Vega", ra_hours=18.6157, dec_deg=38.7837, vmag=0.03
        ),
        CalibrationTargetSpec.celestial(
            "Capella", ra_hours=5.2782, dec_deg=45.998, vmag=0.08
        ),
        CalibrationTargetSpec.celestial(
            "Arcturus", ra_hours=14.261, dec_deg=19.1825, vmag=-0.05
        ),
    ]
    out = tmp_path / "cal.json"
    session = CalibrationSession(
        telescope_id=22,
        target_specs=specs,
        site=site,
        out_path=out,
    )
    session.start()
    try:
        _wait_for_phase(session, "nudging")
        for _ in range(3):
            session.sight()
            time.sleep(0.3)
        _wait_for_phase(session, "review")
        st = session.status()
        assert st.solution is not None
        assert len(st.solution["per_landmark"]) == 3
        for rec in st.solution["per_landmark"]:
            assert rec["kind"] == TargetKind.CELESTIAL.value
        session.commit()
        _wait_for_phase(session, "committed")
    finally:
        session.stop()
    assert out.exists()
    payload = json.loads(out.read_text())
    assert payload["calibration_method"] == "rotation_celestial"
    assert payload["n_landmarks"] == 3


def test_platesolve_session_requires_solver(monkeypatch, tmp_path):
    """A session with a PLATESOLVE spec must be constructed with a
    plate solver — otherwise the session refuses at construction."""
    from device.calibration_targets import CalibrationTargetSpec

    site = _site()
    _install_fakes(monkeypatch, _FakeCli())
    with pytest.raises(ValueError, match="plate_solver required"):
        CalibrationSession(
            telescope_id=31,
            target_specs=[
                CalibrationTargetSpec.from_landmark(HYPERION_06_000301, slant_m=5523.0),
                CalibrationTargetSpec.platesolve("free aim 1"),
            ],
            site=site,
            out_path=tmp_path / "cal.json",
        )


def test_platesolve_failure_does_not_append_sighting(monkeypatch, tmp_path):
    """If the inner plate-solve fails, the session must surface the
    error and leave the sightings list untouched so the operator can
    retry without losing prior good sightings."""
    from device.calibration_targets import CalibrationTargetSpec
    from device.plate_solver import FakePlateSolver

    site = _site()
    cli = _FakeCli()
    _install_fakes(monkeypatch, cli, neutralise_sun=True)
    solver = FakePlateSolver(results={"/tmp/fake-cap.fits": None})

    def _capture():
        return Path("/tmp/fake-cap.fits")

    spec_faa = CalibrationTargetSpec.from_landmark(HYPERION_06_000301, slant_m=5523.0)
    spec_lab = CalibrationTargetSpec.from_landmark(
        LA_BROADCAST_06_000177, slant_m=8500.0
    )
    spec_ps = CalibrationTargetSpec.platesolve("free aim 1")
    session = CalibrationSession(
        telescope_id=32,
        target_specs=[spec_faa, spec_ps, spec_lab],
        site=site,
        out_path=tmp_path / "cal.json",
        plate_solver=solver,
        capture_image_fn=_capture,
    )
    session.start()
    try:
        _wait_for_phase(session, "nudging")
        # Sight the first FAA target → success.
        session.sight()
        time.sleep(0.4)
        st = session.status()
        assert st.target_idx == 1
        assert st.current_landmark["kind"] == "platesolve"
        assert st.solution is not None
        assert len(st.solution["per_landmark"]) == 1
        # Sight the plate-solve target → fake solver returns None →
        # PlateSolverFailed → the sight attempt fails.
        session.sight()
        # Allow the worker to process.
        time.sleep(0.4)
        st2 = session.status()
        # Sighting count unchanged; idx still pointing at the
        # plate-solve target (no auto-advance on failure).
        assert st2.target_idx == 1
        assert st2.solution is not None
        assert len(st2.solution["per_landmark"]) == 1
        assert any("plate-solve" in e.lower() for e in st2.errors)
        assert st2.current_landmark.get("last_platesolve_error") is not None
    finally:
        session.stop()


def test_status_response_includes_kind_metadata(monkeypatch, tmp_path):
    from device.calibration_targets import CalibrationTargetSpec

    site = _site()
    cli = _FakeCli()
    _install_fakes(monkeypatch, cli, neutralise_sun=True)
    fixed = (45.0, 60.0, None)

    def _fake_celestial(self, site_, when_utc):
        return fixed

    monkeypatch.setattr(CalibrationTargetSpec, "_resolve_celestial", _fake_celestial)
    specs = [
        CalibrationTargetSpec.from_landmark(HYPERION_06_000301, slant_m=5523.0),
        CalibrationTargetSpec.celestial(
            "Vega", ra_hours=18.6157, dec_deg=38.7837, vmag=0.03
        ),
    ]
    session = CalibrationSession(
        telescope_id=33,
        target_specs=specs,
        site=site,
        out_path=tmp_path / "cal.json",
    )
    session.start()
    try:
        _wait_for_phase(session, "nudging")
        st = session.status()
        assert st.targets is not None
        assert [t["kind"] for t in st.targets] == ["faa", "celestial"]
        assert st.targets[1]["label"] == "Vega"
        # current_landmark on first target should carry FAA fields.
        assert st.current_landmark["kind"] == "faa"
        assert st.current_landmark["oas"] == "06-000301"
    finally:
        session.stop()


# --------- regression: motion delegate, seeding, ordering, timing --------


def _wrap_diff(a: float, b: float) -> float:
    return ((a - b + 180.0) % 360.0) - 180.0


def _wait_until(predicate, timeout_s: float = 3.0) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


class _FakeMotionDelegate:
    """Calibrate-motion stand-in modelling the streaming controller's
    start anchor: the mount sits at ``provider_target + offset`` (the
    offset carries the encoder origin), and converges instantly. When
    ``cli`` is given, ``set_target`` also moves that fake mount's encoder,
    so the session's idle encoder poll reads the same position."""

    def __init__(self, off_az: float, off_el: float, cli=None):
        self.off_az = float(off_az)
        self.off_el = float(off_el)
        self.cli = cli
        self.target = [0.0, 0.0]
        self.set_calls: list[tuple[float, float]] = []

    def mount(self) -> tuple[float, float]:
        return self.target[0] + self.off_az, self.target[1] + self.off_el

    def status(self):
        from types import SimpleNamespace

        cur_az, cur_el = self.mount()
        return SimpleNamespace(
            target_az_deg=self.target[0],
            target_el_deg=self.target[1],
            cur_cum_az_deg=cur_az,
            cur_el_deg=cur_el,
            err_az_deg=0.0,
            err_el_deg=0.0,
        )

    def set_target(self, az_deg, el_deg):
        self.set_calls.append((float(az_deg), float(el_deg)))
        self.target = [float(az_deg), float(el_deg)]
        if self.cli is not None:
            self.cli.encoder_az, self.cli.encoder_el = self.mount()

    def nudge_target(self, d_az_deg, d_el_deg):
        self.target[0] += float(d_az_deg)
        self.target[1] += float(d_el_deg)

    def is_settled(self, threshold_deg=None, ticks=None):
        return True


def test_pre_slew_through_motion_delegate_lands_mount_on_prediction(
    monkeypatch, tmp_path
):
    """The delegate's provider frame is offset from the encoder by its
    start anchor. The pre-slew must convert the absolute mount-frame
    prediction into that frame so the MOUNT lands on the prediction (not
    on prediction + anchor offset)."""
    site = _site()
    cli = _FakeCli()
    _install_fakes(monkeypatch, cli, neutralise_sun=True)
    fake = _FakeMotionDelegate(off_az=37.0, off_el=4.0, cli=cli)
    session = CalibrationSession(
        telescope_id=41,
        targets=_targets(site),
        site=site,
        out_path=tmp_path / "cal.json",
    )
    monkeypatch.setattr(session, "_motion_delegate", lambda: fake)
    session.start()
    try:
        _wait_for_phase(session, "nudging")
        st = session.status()
        assert fake.set_calls, "pre-slew did not go through the delegate"
        mount_az, mount_el = fake.mount()
        assert abs(_wrap_diff(mount_az, st.target_az_deg)) < 1e-6
        assert mount_el == pytest.approx(st.target_el_deg, abs=1e-6)
        # The session's encoder readout follows the delegate's mount.
        assert abs(_wrap_diff(st.encoder_az_deg, st.target_az_deg)) < 1e-6
    finally:
        session.stop()


def _count_move_to_ff(monkeypatch) -> list[tuple[float, float]]:
    """Wrap the installed fake move_to_ff so a test can assert whether
    (and where) a direct pre-slew was commanded."""
    import device.velocity_controller as vc

    inner = vc.move_to_ff
    calls: list[tuple[float, float]] = []

    def counting(client, **kw):
        calls.append((kw["target_az_deg"], kw["target_el_deg"]))
        return inner(client, **kw)

    monkeypatch.setattr(vc, "move_to_ff", counting)
    return calls


def test_delegate_without_tick_fails_preslew_without_move_to_ff(monkeypatch, tmp_path):
    """No tick yet → the anchor offset is unknown, so the delegate is not
    given an absolute target. The delegate is alive (its track() loop owns
    the mount), so a direct move_to_ff would fight it: the pre-slew must
    fail with a clear error instead, commanding nothing."""
    site = _site()
    cli = _FakeCli()
    _install_fakes(monkeypatch, cli, neutralise_sun=True)
    move_calls = _count_move_to_ff(monkeypatch)
    fake = _FakeMotionDelegate(off_az=37.0, off_el=4.0)
    from types import SimpleNamespace

    fake.status = lambda: SimpleNamespace(
        target_az_deg=0.0,
        target_el_deg=0.0,
        cur_cum_az_deg=None,
        cur_el_deg=None,
        err_az_deg=None,
        err_el_deg=None,
    )
    session = CalibrationSession(
        telescope_id=42,
        targets=_targets(site),
        site=site,
        out_path=tmp_path / "cal.json",
    )
    monkeypatch.setattr(session, "DELEGATE_FIRST_TICK_WAIT_S", 0.0)
    monkeypatch.setattr(session, "_motion_delegate", lambda: fake)
    session.start()
    try:
        _wait_for_phase(session, "error")
        st = session.status()
        assert fake.set_calls == []
        assert move_calls == []
        assert any("did not respond" in e for e in st.errors), st.errors
        # The fake mount never moved.
        assert (cli.encoder_az, cli.encoder_el) == (0.0, 0.0)
    finally:
        session.stop()


def _first_target_prediction(site, monkeypatch, tmp_path) -> tuple[float, float]:
    """Predicted mount (az, el) for target 0, via a dry-run slew."""
    dry = CalibrationSession(
        telescope_id=45,
        targets=_targets(site),
        site=site,
        out_path=tmp_path / "dry.json",
        dry_run=True,
    )
    dry._slew_to_target(None, 0)
    st = dry.status()
    return float(st.target_az_deg), float(st.target_el_deg)


def _az_dist(a: float, b: float) -> float:
    return abs(((a - b + 180.0) % 360.0) - 180.0)


def _path_session(site, monkeypatch, tmp_path, cone_az):
    """Non-dry-run session with no motion delegate whose sun cone (stubbed
    in sun_safety.is_sun_safe) is a 5° az band around ``cone_az``
    (``None`` = no cone)."""
    from device import sun_safety as ss

    def fake_is_sun_safe(az, el, **kw):
        if cone_az is not None and _az_dist(float(az), cone_az) < 5.0:
            return False, "sun_avoidance: stubbed cone"
        return True, ""

    monkeypatch.setattr(ss, "is_sun_safe", fake_is_sun_safe)
    session = CalibrationSession(
        telescope_id=46,
        targets=_targets(site),
        site=site,
        out_path=tmp_path / "cal.json",
    )
    monkeypatch.setattr(session, "_motion_delegate", lambda: None)
    return session


def test_preslew_refused_when_path_crosses_sun_cone(monkeypatch, tmp_path):
    """Start and end are outside the cone but the straight-line path
    between them crosses it: refuse with no motion command issued."""
    site = _site()
    tgt_az, tgt_el = _first_target_prediction(site, monkeypatch, tmp_path)
    cli = _FakeCli()
    _install_fakes(monkeypatch, cli)
    move_calls = _count_move_to_ff(monkeypatch)
    cli.encoder_az = ((tgt_az + 60.0 + 180.0) % 360.0) - 180.0
    cli.encoder_el = tgt_el
    start = (cli.encoder_az, cli.encoder_el)
    session = _path_session(site, monkeypatch, tmp_path, cone_az=tgt_az + 30.0)

    session._slew_to_target(cli, 0)

    st = session.status()
    assert st.phase == "error"
    assert session._stop_evt.is_set()
    assert any("sun_avoidance" in e and "path point" in e for e in st.errors), st.errors
    assert move_calls == []
    assert (cli.encoder_az, cli.encoder_el) == start


def test_preslew_proceeds_when_path_is_clear(monkeypatch, tmp_path):
    site = _site()
    tgt_az, tgt_el = _first_target_prediction(site, monkeypatch, tmp_path)
    cli = _FakeCli()
    _install_fakes(monkeypatch, cli)
    move_calls = _count_move_to_ff(monkeypatch)
    cli.encoder_az = ((tgt_az + 60.0 + 180.0) % 360.0) - 180.0
    cli.encoder_el = tgt_el
    # Cone on the far side of the sky: the short-way path never nears it.
    session = _path_session(site, monkeypatch, tmp_path, cone_az=tgt_az + 200.0)

    session._slew_to_target(cli, 0)

    st = session.status()
    assert st.phase == "slewing", st.errors
    assert not session._stop_evt.is_set()
    assert len(move_calls) == 1
    assert _az_dist(cli.encoder_az, tgt_az) < 1e-6
    assert cli.encoder_el == pytest.approx(tgt_el, abs=1e-6)


def test_preslew_refused_when_encoder_unreadable(monkeypatch, tmp_path):
    """Without a start point the path cannot be verified: refuse (same as
    the live tracker's pre-slew) rather than skip the check."""
    site = _site()
    cli = _FakeCli()
    _install_fakes(monkeypatch, cli)
    move_calls = _count_move_to_ff(monkeypatch)

    def broken_measure(client, loc):
        raise RuntimeError("no encoder")

    monkeypatch.setattr(
        "device.velocity_controller.measure_altaz_timed", broken_measure
    )
    session = _path_session(site, monkeypatch, tmp_path, cone_az=None)

    session._slew_to_target(cli, 0)

    st = session.status()
    assert st.phase == "error"
    assert any("could not be verified" in e for e in st.errors), st.errors
    assert move_calls == []


def test_two_sighting_commit_keeps_kept_prior_pitch_and_roll(monkeypatch, tmp_path):
    """With a kept prior, the yaw-only fit of a two-sighting commit must
    keep the prior's pitch/roll, not overwrite them with 0."""
    from device.target_frame import MountFrame

    site = _site()
    cli = _FakeCli()
    _install_fakes(monkeypatch, cli, neutralise_sun=True)
    prior = MountFrame.from_euler_deg(
        yaw_deg=5.0, pitch_deg=1.0, roll_deg=-0.5, site=site
    )
    out = tmp_path / "cal.json"
    session = CalibrationSession(
        telescope_id=43,
        targets=_targets(site),
        site=site,
        out_path=out,
        prior_frame=prior,
    )
    session.start()
    try:
        _wait_for_phase(session, "nudging")
        session.sight()
        assert _wait_until(lambda: session.status().target_idx == 1)
        _wait_for_phase(session, "nudging")
        session.sight()
        _wait_for_phase(session, "review")
        session.commit()
        _wait_for_phase(session, "committed")
    finally:
        session.stop()

    payload = json.loads(out.read_text())
    assert payload["n_landmarks"] == 2
    assert payload["pitch_offset_deg"] == pytest.approx(1.0, abs=1e-9)
    assert payload["roll_offset_deg"] == pytest.approx(-0.5, abs=1e-9)
    assert payload["yaw_offset_deg"] == pytest.approx(5.0, abs=0.1)


def test_nudge_coalescing_keeps_sight_ahead_of_later_nudge(monkeypatch, tmp_path):
    """Queued [nudge, sight, nudge]: the sight must record the position
    after the first nudge only. Re-queuing the sight behind the second
    nudge (the old behaviour) recorded a position moved by both."""
    import threading

    site = _site()
    _install_fakes(monkeypatch, _FakeCli(), neutralise_sun=True)
    session = CalibrationSession(
        telescope_id=44,
        targets=_targets(site),
        site=site,
        out_path=tmp_path / "cal.json",
        dry_run=True,
    )
    session._slew_to_target(None, 0)
    base_az = session.status().target_az_deg
    assert base_az is not None

    session.nudge(1.0, 0.0)
    session.sight()
    session.nudge(1.0, 0.0)

    worker = threading.Thread(target=session._process_loop, args=(None,), daemon=True)
    worker.start()
    try:
        assert _wait_until(lambda: len(session._sightings) >= 1)
    finally:
        session._stop_evt.set()
        worker.join(2.0)

    assert session._sightings[0].encoder_az_deg == pytest.approx(
        base_az + 1.0, abs=1e-9
    )


def test_platesolve_sighting_uses_capture_time_not_solve_completion(
    monkeypatch, tmp_path
):
    """The solved RA/Dec must be converted to alt/az at the exposure time
    (tracking is off, so the sky drifts while the solve runs), and the
    sighting carries that same timestamp."""
    import device.nighttime_calibration as nc
    from device.calibration_targets import CalibrationTargetSpec
    from device.plate_solver import SolveResult

    site = _site()
    _install_fakes(monkeypatch, _FakeCli(), neutralise_sun=True)

    capture_times: list[float] = []
    transform_times: list[float] = []

    def _capture():
        capture_times.append(time.time())
        return Path("cap.fits")

    class _SlowSolver:
        def solve(self, image_path):
            time.sleep(0.2)
            return SolveResult(
                ra_deg=10.0,
                dec_deg=20.0,
                fov_x_deg=1.27,
                fov_y_deg=0.71,
                position_angle_deg=0.0,
            )

    def _fake_transform(ra, dec, t_unix, site_):
        transform_times.append(float(t_unix))
        return 100.0, 40.0

    monkeypatch.setattr(nc, "radec_to_topocentric_azel", _fake_transform)
    specs = [
        CalibrationTargetSpec.platesolve("free aim"),
        CalibrationTargetSpec.from_landmark(HYPERION_06_000301, slant_m=5523.0),
    ]
    session = CalibrationSession(
        telescope_id=45,
        target_specs=specs,
        site=site,
        out_path=tmp_path / "cal.json",
        dry_run=True,
        plate_solver=_SlowSolver(),
        capture_image_fn=_capture,
    )
    session._encoder_az = 10.0
    session._encoder_el = 40.0
    t_before = time.time()
    session._on_sight(None)

    assert len(session._sightings) == 1, session.status().errors
    assert transform_times and capture_times
    t_used = transform_times[0]
    assert t_before <= t_used <= capture_times[0]
    assert session._sightings[0].t_unix == pytest.approx(t_used)


def _seeded_platesolve_session(site, monkeypatch, tmp_path, prior_frame, cone_sky_az):
    """Dry-run session whose only target is a plate-solve seed at mount
    (az 0, el 30); the stubbed sun cone is a 5° az band at ``cone_sky_az``."""
    from device import sun_safety as ss
    from device.calibration_targets import CalibrationTargetSpec
    from device.plate_solver import FakePlateSolver

    def fake_is_sun_safe(az, el, **kw):
        if _az_dist(float(az), cone_sky_az) < 5.0:
            return False, "sun_avoidance: stubbed cone"
        return True, ""

    monkeypatch.setattr(ss, "is_sun_safe", fake_is_sun_safe)
    return CalibrationSession(
        telescope_id=47,
        target_specs=[
            CalibrationTargetSpec.platesolve(
                "seeded", seed_az_deg=0.0, seed_el_deg=30.0
            )
        ],
        site=site,
        out_path=tmp_path / "cal.json",
        plate_solver=FakePlateSolver(results={}),
        capture_image_fn=lambda: Path("/tmp/unused.fits"),
        prior_frame=prior_frame,
        dry_run=True,
    )


def test_seeded_platesolve_refused_when_encoder_seed_lands_on_sun(
    monkeypatch, tmp_path
):
    """The seed is commanded as an encoder target. With a calibrated prior
    whose encoder zero is not north, encoder az 0 is a different sky az;
    if the sun is there the slew must be refused even though the raw seed
    numbers, read as sky, are far from it."""
    from device.streaming_controller import _mount_azel_to_sky
    from device.target_frame import MountFrame

    site = _site()
    prior = MountFrame.from_euler_deg(90.0, 0.0, 0.0, site)
    sky_az, _ = _mount_azel_to_sky(prior, 0.0, 30.0)
    assert _az_dist(sky_az, 0.0) > 45.0  # the two readings really differ
    session = _seeded_platesolve_session(site, monkeypatch, tmp_path, prior, sky_az)

    session._slew_to_target(None, 0)

    st = session.status()
    assert st.phase == "error"
    assert session._stop_evt.is_set()
    assert any("sun_avoidance" in e for e in st.errors), st.errors


def test_seeded_platesolve_allowed_with_identity_prior(monkeypatch, tmp_path):
    """Same cone, uncalibrated (identity) prior: both readings of the seed
    are sky az 0, well clear of it, so the slew goes ahead."""
    from device.streaming_controller import _mount_azel_to_sky
    from device.target_frame import MountFrame

    site = _site()
    cone_az, _ = _mount_azel_to_sky(
        MountFrame.from_euler_deg(90.0, 0.0, 0.0, site), 0.0, 30.0
    )
    session = _seeded_platesolve_session(site, monkeypatch, tmp_path, None, cone_az)

    session._slew_to_target(None, 0)

    st = session.status()
    assert st.phase == "slewing", st.errors
    assert not session._stop_evt.is_set()
