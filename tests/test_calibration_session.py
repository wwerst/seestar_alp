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
