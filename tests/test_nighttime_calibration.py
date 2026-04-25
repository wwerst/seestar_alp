"""Tests for nighttime calibration (plate-solve workflow).

The session uses a background thread for the solver call. To keep tests
deterministic without timing dependencies, the FakePlateSolver returns
canned ``SolveResult`` values keyed by image path; the session's solve
worker is a daemon thread, so we wait briefly with a short polling loop
before asserting on status.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from device.nighttime_calibration import (
    MIN_SIGHTINGS_FOR_APPLY,
    NighttimeCalibrationManager,
    NighttimeCalibrationSession,
    radec_to_topocentric_azel,
)
from device.plate_solver import (
    FakePlateSolver,
    PlateSolverFailed,
    PlateSolverNotAvailable,
    SolveResult,
    UnavailablePlateSolver,
    _parse_solve_field_stdout,
)
from device.rotation_calibration import solve_rotation_from_pairs
from scripts.trajectory.observer import build_site


DOCKWEILER = dict(lat_deg=33.9615051, lon_deg=-118.4581361, alt_m=2.0)


def _site():
    return build_site(**DOCKWEILER)


# ---------- PlateSolver layer ---------------------------------------


def test_unavailable_plate_solver_raises():
    s = UnavailablePlateSolver()
    assert not s.is_available()
    with pytest.raises(PlateSolverNotAvailable):
        s.solve(Path("/tmp/whatever"))


def test_fake_plate_solver_canned_results(tmp_path):
    canned = SolveResult(
        ra_deg=80.0,
        dec_deg=-10.0,
        fov_x_deg=1.27,
        fov_y_deg=0.71,
        position_angle_deg=12.0,
        stars_used=42,
    )
    img = tmp_path / "frame.jpg"
    img.write_text("dummy")
    fake = FakePlateSolver({str(img): canned})
    result = fake.solve(img)
    assert result == canned
    assert fake.calls == [img]


def test_fake_plate_solver_failure_raises(tmp_path):
    img = tmp_path / "frame.jpg"
    img.write_text("dummy")
    fake = FakePlateSolver({str(img): None})
    with pytest.raises(PlateSolverFailed):
        fake.solve(img)


def test_solve_field_stdout_parse_valid_centre_and_size():
    """Smoke-test the regex against a representative astrometry.net
    stdout fragment so a future solver-version bump shows up here
    instead of as a silent garbage solve."""
    sample = """
Reading input file 1 of 1: "frame.jpg"...
Field 1: solved with index index-4205-09.fits.
Field 1 solved: matched (10 match(es)) of 50 stars
Field 1: solved with index index-4205-09.fits.
Field center: (RA,Dec) = (80.123, -10.456) deg.
Field size: 1.27 x 0.71 degrees
Field rotation angle: up is 12.5 degrees E of N
"""
    parsed = _parse_solve_field_stdout(sample)
    assert parsed.ra_deg == pytest.approx(80.123)
    assert parsed.dec_deg == pytest.approx(-10.456)
    assert parsed.fov_x_deg == pytest.approx(1.27)
    assert parsed.fov_y_deg == pytest.approx(0.71)
    assert parsed.position_angle_deg == pytest.approx(12.5)
    assert parsed.stars_used == 10


def test_solve_field_stdout_parse_arcminute_units():
    """Solver may emit field size in arcminutes for narrow FOV; parser
    must convert."""
    sample = """
Field center: (RA,Dec) = (10.0, 20.0) deg.
Field size: 76.2 x 42.6 arcminutes
"""
    parsed = _parse_solve_field_stdout(sample)
    # 76.2 arcmin / 60 = 1.27°
    assert parsed.fov_x_deg == pytest.approx(1.27, abs=1e-3)
    assert parsed.fov_y_deg == pytest.approx(0.71, abs=1e-3)


def test_solve_field_stdout_parse_missing_centre_raises():
    with pytest.raises(PlateSolverFailed):
        _parse_solve_field_stdout("Field 1 unsolved.")


def test_get_default_plate_solver_returns_unavailable_without_solve_field(monkeypatch):
    import device.plate_solver as ps

    monkeypatch.setattr(ps.shutil, "which", lambda *a, **kw: None)
    s = ps.get_default_plate_solver()
    assert isinstance(s, UnavailablePlateSolver)


# ---------- solve_rotation_from_pairs ---------------------------------


def test_solve_rotation_from_pairs_recovers_known_rotation():
    """Generate synthetic sightings under a known rotation and verify
    the solver recovers the same yaw/pitch/roll."""
    # Truth rotation; sightings are synthesised by applying it to
    # randomly-distributed sky positions, then the solver should
    # invert it.
    from device.rotation_calibration import _predict_mount_azel_from_topo

    truth_yaw, truth_pitch, truth_roll = 12.5, -1.2, 0.7
    true_pairs = []
    test_directions = [
        (45.0, 30.0),
        (135.0, 50.0),
        (225.0, 60.0),
        (315.0, 25.0),
    ]
    for true_az, true_el in test_directions:
        enc_az, enc_el = _predict_mount_azel_from_topo(
            truth_yaw, truth_pitch, truth_roll, true_az, true_el
        )
        true_pairs.append((enc_az, enc_el, true_az, true_el))
    sol = solve_rotation_from_pairs(true_pairs)
    assert sol.yaw_deg == pytest.approx(truth_yaw, abs=0.001)
    assert sol.pitch_deg == pytest.approx(truth_pitch, abs=0.001)
    assert sol.roll_deg == pytest.approx(truth_roll, abs=0.001)
    assert sol.residual_rms_deg < 1e-3
    assert len(sol.per_landmark) == 4
    # Per-record dicts use the platesolve schema.
    assert sol.per_landmark[0]["kind"] == "platesolve"


def test_solve_rotation_from_pairs_yaw_only_one_sighting():
    """With one sighting the solver should fit yaw only (auto mode)."""
    from device.rotation_calibration import _predict_mount_azel_from_topo

    enc_az, enc_el = _predict_mount_azel_from_topo(8.0, 0.0, 0.0, 100.0, 30.0)
    sol = solve_rotation_from_pairs([(enc_az, enc_el, 100.0, 30.0)])
    assert sol.yaw_deg == pytest.approx(8.0, abs=0.001)
    assert sol.pitch_deg == 0.0
    assert sol.roll_deg == 0.0


# ---------- NighttimeCalibrationSession --------------------------------


def _wait_for(predicate, timeout_s: float = 5.0, poll_s: float = 0.05):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(poll_s)
    return False


def _make_session(tmp_path, plate_solver):
    return NighttimeCalibrationSession(
        telescope_id=99,
        site=_site(),
        out_path=tmp_path / "mount_calibration.json",
        plate_solver=plate_solver,
    )


def test_session_capture_below_floor_refuses(tmp_path):
    session = _make_session(tmp_path, FakePlateSolver())
    with pytest.raises(ValueError, match="below.*altitude"):
        session.capture_sighting(
            image_path=tmp_path / "img.jpg",
            encoder_az_deg=180.0,
            encoder_el_deg=5.0,
        )


def test_session_capture_above_ceiling_refuses(tmp_path):
    session = _make_session(tmp_path, FakePlateSolver())
    with pytest.raises(ValueError, match="above"):
        session.capture_sighting(
            image_path=tmp_path / "img.jpg",
            encoder_az_deg=180.0,
            encoder_el_deg=85.0,
        )


def test_session_solver_failure_records_in_status(tmp_path, monkeypatch):
    img = tmp_path / "img.jpg"
    img.write_text("dummy")
    fake = FakePlateSolver({str(img): None})  # None → PlateSolverFailed
    session = _make_session(tmp_path, fake)
    session.capture_sighting(img, encoder_az_deg=180.0, encoder_el_deg=40.0)
    assert _wait_for(lambda: session.status().last_failed is not None)
    st = session.status()
    assert st.last_failed["status"] == "fail"
    assert "fake failure" in (st.last_failed["error"] or "")
    assert st.n_accepted == 0


def test_session_solver_returns_bad_fov_records_failure(tmp_path):
    img = tmp_path / "img.jpg"
    img.write_text("dummy")
    bad = SolveResult(
        ra_deg=90.0,
        dec_deg=20.0,
        fov_x_deg=10.0,  # outside [0.5, 3.0]
        fov_y_deg=10.0,
        position_angle_deg=0.0,
    )
    fake = FakePlateSolver({str(img): bad})
    session = _make_session(tmp_path, fake)
    session.capture_sighting(img, encoder_az_deg=180.0, encoder_el_deg=40.0)
    assert _wait_for(lambda: session.status().last_failed is not None)
    st = session.status()
    assert "FOV" in (st.last_failed["error"] or "")
    assert st.n_accepted == 0


def test_session_three_sightings_fit_succeeds(tmp_path, monkeypatch):
    """Build 3 captures with canned plate solves whose true az/el are
    known to invert under a chosen rotation. The session should accept
    all three and produce a fit. We monkey-patch radec_to_topocentric_azel
    so we don't depend on astropy + the real wall clock."""
    import device.nighttime_calibration as nc
    from device.rotation_calibration import _predict_mount_azel_from_topo

    truth_yaw, truth_pitch, truth_roll = 5.0, 1.0, -0.5
    test_dirs = [(60.0, 30.0), (180.0, 45.0), (300.0, 35.0)]

    # Map (ra, dec) → fake sky direction. Use ra/dec as direct (true_az, true_el)
    # for simplicity — we don't care that they're not realistic celestial
    # coordinates for the purposes of the rotation solver.
    def fake_radec_to_topo(ra, dec, t_unix, site):
        return ra, dec

    monkeypatch.setattr(nc, "radec_to_topocentric_azel", fake_radec_to_topo)

    canned = {}
    captures = []
    for i, (true_az, true_el) in enumerate(test_dirs):
        enc_az, enc_el = _predict_mount_azel_from_topo(
            truth_yaw, truth_pitch, truth_roll, true_az, true_el
        )
        img = tmp_path / f"img{i}.jpg"
        img.write_text("dummy")
        canned[str(img)] = SolveResult(
            ra_deg=true_az,
            dec_deg=true_el,
            fov_x_deg=1.27,
            fov_y_deg=0.71,
            position_angle_deg=0.0,
        )
        captures.append((img, enc_az, enc_el))

    fake = FakePlateSolver(canned)
    session = _make_session(tmp_path, fake)

    for img, enc_az, enc_el in captures:
        session.capture_sighting(img, enc_az, enc_el)
        assert _wait_for(lambda: session.status().pending is None, timeout_s=5.0)
    st = session.status()
    assert st.n_accepted == 3
    assert st.fit is not None
    assert st.fit["yaw_deg"] == pytest.approx(truth_yaw, abs=0.05)
    assert st.fit["pitch_deg"] == pytest.approx(truth_pitch, abs=0.05)
    assert st.fit["roll_deg"] == pytest.approx(truth_roll, abs=0.05)
    assert st.fit["residual_rms_deg"] < 0.1


def test_session_apply_writes_atomic_json(tmp_path, monkeypatch):
    import device.nighttime_calibration as nc

    monkeypatch.setattr(
        nc, "radec_to_topocentric_azel", lambda ra, dec, t, site: (ra, dec)
    )
    # Three minimal sightings with deterministic results.
    canned = {}
    captures = []
    for i, (az, el) in enumerate([(80.0, 35.0), (160.0, 40.0), (250.0, 55.0)]):
        img = tmp_path / f"img{i}.jpg"
        img.write_text("dummy")
        canned[str(img)] = SolveResult(
            ra_deg=az,
            dec_deg=el,
            fov_x_deg=1.27,
            fov_y_deg=0.71,
            position_angle_deg=0.0,
        )
        captures.append((img, az, el))

    fake = FakePlateSolver(canned)
    session = _make_session(tmp_path, fake)
    out = tmp_path / "mount_calibration.json"
    session.out_path = out

    for img, az, el in captures:
        session.capture_sighting(img, az, el)
        assert _wait_for(lambda: session.status().pending is None, timeout_s=5.0)

    session.apply()
    assert out.exists()
    payload = json.loads(out.read_text())
    assert payload["calibration_method"] == "rotation_platesolve"
    assert payload["n_sightings"] == 3
    assert "yaw_offset_deg" in payload
    assert "observer" in payload
    assert payload["observer"]["lat_deg"] == pytest.approx(DOCKWEILER["lat_deg"])
    assert len(payload["sightings"]) == 3
    # Records carry kind=platesolve in the fit per-record list.
    assert payload["fit_per_record"][0]["kind"] == "platesolve"


def test_session_apply_refuses_below_minimum(tmp_path, monkeypatch):
    import device.nighttime_calibration as nc

    monkeypatch.setattr(
        nc, "radec_to_topocentric_azel", lambda ra, dec, t, site: (ra, dec)
    )
    img = tmp_path / "img.jpg"
    img.write_text("dummy")
    fake = FakePlateSolver(
        {
            str(img): SolveResult(
                ra_deg=100.0,
                dec_deg=30.0,
                fov_x_deg=1.27,
                fov_y_deg=0.71,
                position_angle_deg=0.0,
            )
        }
    )
    session = _make_session(tmp_path, fake)
    session.capture_sighting(img, encoder_az_deg=100.0, encoder_el_deg=30.0)
    assert _wait_for(lambda: session.status().pending is None, timeout_s=5.0)
    # Only 1 sighting; apply should refuse.
    with pytest.raises(ValueError, match=f"need .{MIN_SIGHTINGS_FOR_APPLY}"):
        session.apply()


def test_session_skip_pending_clears_failure(tmp_path):
    img = tmp_path / "img.jpg"
    img.write_text("dummy")
    fake = FakePlateSolver({str(img): None})
    session = _make_session(tmp_path, fake)
    session.capture_sighting(img, encoder_az_deg=180.0, encoder_el_deg=40.0)
    assert _wait_for(lambda: session.status().last_failed is not None)
    session.skip_pending()
    st = session.status()
    assert st.pending is None


def test_session_remove_sighting_refits(tmp_path, monkeypatch):
    import device.nighttime_calibration as nc

    monkeypatch.setattr(
        nc, "radec_to_topocentric_azel", lambda ra, dec, t, site: (ra, dec)
    )
    canned = {}
    captures = []
    for i, (az, el) in enumerate([(60.0, 35.0), (180.0, 45.0), (300.0, 35.0)]):
        img = tmp_path / f"img{i}.jpg"
        img.write_text("dummy")
        canned[str(img)] = SolveResult(
            ra_deg=az,
            dec_deg=el,
            fov_x_deg=1.27,
            fov_y_deg=0.71,
            position_angle_deg=0.0,
        )
        captures.append((img, az, el))
    fake = FakePlateSolver(canned)
    session = _make_session(tmp_path, fake)
    for img, az, el in captures:
        session.capture_sighting(img, az, el)
        assert _wait_for(lambda: session.status().pending is None, timeout_s=5.0)
    assert session.status().n_accepted == 3
    session.remove_sighting(1)
    st = session.status()
    assert st.n_accepted == 2


# ---------- NighttimeCalibrationManager ------------------------------


def test_manager_singleton():
    from device.nighttime_calibration import get_nighttime_manager

    a = get_nighttime_manager()
    b = get_nighttime_manager()
    assert a is b


def test_manager_refuses_when_live_tracker_running(tmp_path, monkeypatch):
    import device.live_tracker as lt

    class _FakeAlive:
        def is_alive(self):
            return True

    class _FakeMgr:
        def get(self, tid):
            return _FakeAlive()

    monkeypatch.setattr(lt, "get_manager", lambda: _FakeMgr())
    mgr = NighttimeCalibrationManager()
    session = NighttimeCalibrationSession(
        telescope_id=66,
        site=_site(),
        out_path=tmp_path / "out.json",
        plate_solver=FakePlateSolver(),
    )
    with pytest.raises(RuntimeError, match="live-tracking"):
        mgr.start(session)


def test_radec_to_topocentric_azel_roundtrip():
    """Sanity test the astropy-backed conversion against a known
    target. We pick a star at zenith from the equator at the right
    sidereal time so we can compute the expected AltAz analytically.

    The exact value depends on astropy's IERS table; we just check the
    function returns a finite (az, el) tuple in the expected ranges.
    """
    site = _site()
    # Use current time; we just need finite output, not an exact match.
    az, el = radec_to_topocentric_azel(0.0, 0.0, time.time(), site)
    assert -360.0 <= az <= 720.0
    assert -90.0 <= el <= 90.0
