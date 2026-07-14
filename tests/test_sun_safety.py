"""Tests for device.sun_safety pure helpers (Phase 1).

Monitor-thread / jog-angle behavior is covered in a follow-up file once
those are added. These tests deliberately pin lat/lon and time so the
ephem path is deterministic and runs the same anywhere.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone

import pytest

from device.sun_safety import (
    DEFAULT_ALT_THRESHOLD_DEG,
    SafetyTrip,
    SunSafetyMonitor,
    angular_separation,
    compute_jog_angle,
    compute_sun_altaz,
    is_sun_safe,
    make_scope_altaz_reader,
)


# El Segundo, CA — matches the user's site (also config.toml defaults).
SITE_LAT = 33.96
SITE_LON = -118.46


# --- angular_separation ---------------------------------------------------


def test_separation_identical_is_zero():
    assert angular_separation(123.0, 45.0, 123.0, 45.0) == pytest.approx(0.0)


def test_separation_quarter_turn_in_az_at_horizon():
    assert angular_separation(0.0, 0.0, 90.0, 0.0) == pytest.approx(90.0)


def test_separation_antipodal_horizons():
    # (0,0) and (180,0) lie on opposite sides of the horizon ring.
    assert angular_separation(0.0, 0.0, 180.0, 0.0) == pytest.approx(180.0)


def test_separation_same_az_pure_elevation():
    assert angular_separation(45.0, 10.0, 45.0, 40.0) == pytest.approx(30.0)


def test_separation_zenith_to_horizon_is_90():
    # +el=90 is the zenith, irrespective of azimuth.
    assert angular_separation(0.0, 90.0, 273.0, 0.0) == pytest.approx(90.0)


def test_separation_handles_az_wraparound():
    # 350° and 10° are 20° apart on the unit circle.
    assert angular_separation(350.0, 0.0, 10.0, 0.0) == pytest.approx(20.0)


def test_separation_clamps_floating_point_overflow():
    # Should not raise on a near-identical pointing where naive cos can
    # exceed 1.0 by FP rounding.
    sep = angular_separation(12.345678, 67.890123, 12.345678, 67.890123)
    assert math.isfinite(sep)
    assert sep == pytest.approx(0.0, abs=1e-9)


# --- compute_sun_altaz ----------------------------------------------------


def test_sun_below_horizon_at_local_midnight():
    # 09:00 UTC at lon -118.46 ≈ 01:00 local Pacific Standard Time.
    when = datetime(2026, 1, 1, 9, 0, tzinfo=timezone.utc)
    _, alt = compute_sun_altaz(lat_deg=SITE_LAT, lon_deg=SITE_LON, when=when)
    assert alt < DEFAULT_ALT_THRESHOLD_DEG


def test_sun_above_horizon_at_local_noon():
    # 20:00 UTC at lon -118.46 ≈ 12:00 local Pacific Standard Time.
    when = datetime(2026, 1, 1, 20, 0, tzinfo=timezone.utc)
    _, alt = compute_sun_altaz(lat_deg=SITE_LAT, lon_deg=SITE_LON, when=when)
    assert alt > 25.0  # winter sun in LA at noon ~ 33° — give margin


def test_sun_altitude_consistent_across_naive_and_aware():
    naive = datetime(2026, 6, 21, 20, 0)
    aware = naive.replace(tzinfo=timezone.utc)
    a = compute_sun_altaz(lat_deg=SITE_LAT, lon_deg=SITE_LON, when=naive)
    b = compute_sun_altaz(lat_deg=SITE_LAT, lon_deg=SITE_LON, when=aware)
    assert a[0] == pytest.approx(b[0], abs=1e-6)
    assert a[1] == pytest.approx(b[1], abs=1e-6)


# --- is_sun_safe ----------------------------------------------------------


def _midnight_utc():
    return datetime(2026, 1, 1, 9, 0, tzinfo=timezone.utc)


def _noon_utc():
    return datetime(2026, 1, 1, 20, 0, tzinfo=timezone.utc)


def test_is_safe_when_sun_below_threshold_for_any_pointing():
    # Even pointing straight at where the sun will be at noon must be
    # safe at midnight, because the sun is below -10°.
    safe, reason = is_sun_safe(
        180.0,
        33.0,
        lat_deg=SITE_LAT,
        lon_deg=SITE_LON,
        when=_midnight_utc(),
    )
    assert safe is True
    assert reason == ""


def test_is_unsafe_when_pointing_at_sun_during_day():
    sun_az, sun_alt = compute_sun_altaz(
        lat_deg=SITE_LAT,
        lon_deg=SITE_LON,
        when=_noon_utc(),
    )
    safe, reason = is_sun_safe(
        sun_az,
        sun_alt,
        lat_deg=SITE_LAT,
        lon_deg=SITE_LON,
        when=_noon_utc(),
    )
    assert safe is False
    assert "sun_avoidance" in reason
    assert "separation" in reason


def test_is_safe_when_pointing_well_away_from_sun_during_day():
    sun_az, sun_alt = compute_sun_altaz(
        lat_deg=SITE_LAT,
        lon_deg=SITE_LON,
        when=_noon_utc(),
    )
    # Point opposite the sun — separation should be ~180°.
    opp_az = (sun_az + 180.0) % 360.0
    opp_alt = -sun_alt
    safe, _ = is_sun_safe(
        opp_az,
        opp_alt,
        lat_deg=SITE_LAT,
        lon_deg=SITE_LON,
        when=_noon_utc(),
    )
    assert safe is True


def test_unsafe_at_just_inside_cone_edge():
    sun_az, sun_alt = compute_sun_altaz(
        lat_deg=SITE_LAT,
        lon_deg=SITE_LON,
        when=_noon_utc(),
    )
    # 29° away in pure elevation → exact 29° great-circle separation.
    safe, _ = is_sun_safe(
        sun_az,
        sun_alt + 29.0,
        lat_deg=SITE_LAT,
        lon_deg=SITE_LON,
        when=_noon_utc(),
    )
    assert safe is False


def test_safe_at_just_outside_cone_edge():
    sun_az, sun_alt = compute_sun_altaz(
        lat_deg=SITE_LAT,
        lon_deg=SITE_LON,
        when=_noon_utc(),
    )
    # 31° away in pure elevation → exact 31° great-circle separation.
    safe, _ = is_sun_safe(
        sun_az,
        sun_alt + 31.0,
        lat_deg=SITE_LAT,
        lon_deg=SITE_LON,
        when=_noon_utc(),
    )
    assert safe is True


def test_custom_cone_angle_overrides_default():
    sun_az, sun_alt = compute_sun_altaz(
        lat_deg=SITE_LAT,
        lon_deg=SITE_LON,
        when=_noon_utc(),
    )
    # 31° away (pure elevation) — outside default 30° cone, inside 60°.
    target_alt = sun_alt + 31.0
    safe_default, _ = is_sun_safe(
        sun_az,
        target_alt,
        lat_deg=SITE_LAT,
        lon_deg=SITE_LON,
        when=_noon_utc(),
    )
    safe_wider, _ = is_sun_safe(
        sun_az,
        target_alt,
        lat_deg=SITE_LAT,
        lon_deg=SITE_LON,
        when=_noon_utc(),
        min_separation_deg=60.0,
    )
    assert safe_default is True
    assert safe_wider is False


def test_custom_alt_threshold_overrides_default():
    # Sun at -5° (above -10° default → cone enforced; above -3° →
    # disabled). Use a small cone so we can flip behavior.
    when = datetime(2026, 1, 1, 14, 5, tzinfo=timezone.utc)  # ~near civil dawn
    sun_az, sun_alt = compute_sun_altaz(
        lat_deg=SITE_LAT,
        lon_deg=SITE_LON,
        when=when,
    )
    # Pick a time close to civil dawn; verify behavior changes if we
    # raise the threshold above the actual sun altitude.
    target_az, target_alt = sun_az, sun_alt  # pointing right at sun
    safe_default, _ = is_sun_safe(
        target_az,
        target_alt,
        lat_deg=SITE_LAT,
        lon_deg=SITE_LON,
        when=when,
    )
    safe_disabled, _ = is_sun_safe(
        target_az,
        target_alt,
        lat_deg=SITE_LAT,
        lon_deg=SITE_LON,
        when=when,
        alt_threshold_deg=sun_alt + 1.0,
    )
    # With default threshold (-10°): if sun_alt > -10°, unsafe (pointing at sun).
    # With threshold above current sun_alt: always safe.
    if sun_alt >= DEFAULT_ALT_THRESHOLD_DEG:
        assert safe_default is False
    assert safe_disabled is True


# --- SafetyTrip dataclass -------------------------------------------------


def test_safety_trip_is_immutable():
    trip = SafetyTrip(
        when_utc=datetime(2026, 1, 1, tzinfo=timezone.utc),
        sun_az_deg=180.0,
        sun_alt_deg=33.0,
        mount_az_deg=181.0,
        mount_el_deg=34.0,
        separation_deg=1.4,
        cone_deg=30.0,
        jog_angle_deg=90,
        jog_speed=1440,
        jog_duration_s=3,
    )
    with pytest.raises(Exception):
        trip.cone_deg = 15.0  # frozen dataclass — must raise


def test_safety_trip_default_message():
    trip = SafetyTrip(
        when_utc=datetime(2026, 1, 1, tzinfo=timezone.utc),
        sun_az_deg=0,
        sun_alt_deg=0,
        mount_az_deg=0,
        mount_el_deg=0,
        separation_deg=0,
        cone_deg=30.0,
        jog_angle_deg=0,
        jog_speed=1440,
        jog_duration_s=3,
    )
    assert "Sun safety triggered" in trip.message


# --- compute_jog_angle ----------------------------------------------------


def _apply_jog(mount_az, mount_el, angle_deg, jog_speed=1440, jog_duration_s=6.0):
    """Forward-simulate exactly what the monitor does: motion in (daz, del).

    Default ``jog_duration_s`` matches the operational default in
    ``compute_jog_angle`` and ``SunSafetyMonitor`` (6 s × ~6°/s = 36°
    step against a 30° cone with 5° margin).
    """
    rate = jog_speed / 237.0
    step = rate * jog_duration_s
    rad = math.radians(angle_deg)
    new_az = (mount_az + step * math.cos(rad)) % 360.0
    new_el = max(-90.0, min(90.0, mount_el + step * math.sin(rad)))
    return new_az, new_el


def _sep_after_jog(mount_az, mount_el, sun_az, sun_alt, angle):
    new_az, new_el = _apply_jog(mount_az, mount_el, angle)
    return angular_separation(new_az, new_el, sun_az, sun_alt)


def test_jog_increases_separation_sun_east_mount_west():
    sun_az, sun_alt = 90.0, 30.0
    mount_az, mount_el = 100.0, 30.0  # 10° east of mount; inside 30° cone
    sep_before = angular_separation(mount_az, mount_el, sun_az, sun_alt)
    angle = compute_jog_angle(mount_az, mount_el, sun_az, sun_alt)
    sep_after = _sep_after_jog(mount_az, mount_el, sun_az, sun_alt, angle)
    assert sep_after > sep_before, (
        f"angle={angle} sep before={sep_before} after={sep_after}"
    )


def test_jog_increases_separation_sun_west_mount_east():
    sun_az, sun_alt = 270.0, 30.0
    mount_az, mount_el = 260.0, 30.0
    sep_before = angular_separation(mount_az, mount_el, sun_az, sun_alt)
    angle = compute_jog_angle(mount_az, mount_el, sun_az, sun_alt)
    sep_after = _sep_after_jog(mount_az, mount_el, sun_az, sun_alt, angle)
    assert sep_after > sep_before


def test_jog_increases_separation_sun_above_mount():
    # Sun at 40° alt, mount at 30° alt — optical axis pointing low at same az.
    sun_az, sun_alt = 180.0, 40.0
    mount_az, mount_el = 180.0, 30.0
    sep_before = angular_separation(mount_az, mount_el, sun_az, sun_alt)
    angle = compute_jog_angle(mount_az, mount_el, sun_az, sun_alt)
    sep_after = _sep_after_jog(mount_az, mount_el, sun_az, sun_alt, angle)
    assert sep_after > sep_before
    # Should be moving downward (angle near 270° = -el).
    assert 200 < angle < 340


def test_jog_increases_separation_sun_below_mount():
    sun_az, sun_alt = 180.0, 20.0
    mount_az, mount_el = 180.0, 30.0
    sep_before = angular_separation(mount_az, mount_el, sun_az, sun_alt)
    angle = compute_jog_angle(mount_az, mount_el, sun_az, sun_alt)
    sep_after = _sep_after_jog(mount_az, mount_el, sun_az, sun_alt, angle)
    assert sep_after > sep_before
    # Should be moving up (near 90°).
    assert 20 < angle < 160


def test_jog_direction_is_opposite_from_sun_in_az_el_space():
    # 45° diagonal from sun in (daz, del) space → jog should be ~225°
    # (i.e. 45° + 180°).
    sun_az, sun_alt = 100.0, 40.0
    mount_az, mount_el = 110.0, 50.0  # +10° in az, +10° in el from sun
    angle = compute_jog_angle(mount_az, mount_el, sun_az, sun_alt)
    # direction-to-sun has atan2(-10, -10) = -135° → -135 + 360 = 225°.
    # away-from-sun direction: atan2(10, 10) = 45° ≈ the answer.
    assert abs(angle - 45) < 2


def test_jog_clears_cone_with_default_params_starting_at_sun_center():
    """P1-6: with the operational default 6s × ~6°/s ≈ 36° jog, the
    function must drive the mount *out* of a 30° cone with 5° margin
    even from the worst case sep ≈ 0° (mount aligned with sun)."""
    sun_az, sun_alt = 100.0, 30.0
    # Place the mount basically at the sun (sep ≈ 0.1°). Strictly zero
    # would short-circuit the primary direction (norm < 1e-6) and force
    # the +el/−el axial fallbacks; keep a tiny offset to exercise the
    # primary path.
    mount_az, mount_el = 100.1, 30.0
    angle = compute_jog_angle(mount_az, mount_el, sun_az, sun_alt)
    new_az, new_el = _apply_jog(mount_az, mount_el, angle)
    new_sep = angular_separation(new_az, new_el, sun_az, sun_alt)
    assert new_sep >= 35.0 - 1e-6, (
        f"jog must clear cone+margin from sep≈0; got new_sep={new_sep:.2f} "
        f"(angle={angle}, new_pos=({new_az:.2f}, {new_el:.2f}))"
    )


def test_jog_clears_cone_for_any_starting_point_inside_cone():
    """P1-6: across a grid of start positions inside the 30° cone, the
    default jog must always clear cone + 5° margin (or come within the
    spherical distortion budget at high elevations)."""
    sun_az, sun_alt = 180.0, 35.0
    failures: list[str] = []
    for daz in (-25, -15, -5, 0, 5, 15, 25):
        for del_ in (-25, -15, -5, 0, 5, 15, 25):
            mount_az = (sun_az + daz) % 360.0
            mount_el = max(-85.0, min(85.0, sun_alt + del_))
            sep_before = angular_separation(mount_az, mount_el, sun_az, sun_alt)
            if sep_before > 30.0:
                continue  # outside cone — not in scope of test
            angle = compute_jog_angle(mount_az, mount_el, sun_az, sun_alt)
            new_az, new_el = _apply_jog(mount_az, mount_el, angle)
            new_sep = angular_separation(new_az, new_el, sun_az, sun_alt)
            if new_sep < 35.0 - 1e-6:
                failures.append(
                    f"daz={daz} del={del_} sep_before={sep_before:.2f} "
                    f"angle={angle} new_sep={new_sep:.2f}"
                )
    assert not failures, "jog failed cone+margin for:\n" + "\n".join(failures)


def test_jog_falls_back_to_max_separation_when_jog_too_short():
    """If jog_duration_s is too short to clear cone+margin from any
    direction, the function still returns a non-refusal — it picks the
    candidate with the largest predicted separation (best-effort)."""
    sun_az, sun_alt = 180.0, 30.0
    mount_az, mount_el = 180.5, 30.0  # nearly at sun, sep ≈ 0.5°
    # 0.1 s jog at 1440 → step = 0.6° per axis; cannot reach cone+margin.
    angle = compute_jog_angle(
        mount_az,
        mount_el,
        sun_az,
        sun_alt,
        jog_duration_s=0.1,
    )
    # Function must not refuse — it returns SOME angle in [0, 360).
    assert isinstance(angle, int)
    assert 0 <= angle < 360
    # And the chosen direction must increase separation (best of 5
    # candidates beats the starting separation in this geometry).
    new_az, new_el = _apply_jog(mount_az, mount_el, angle, jog_duration_s=0.1)
    new_sep = angular_separation(new_az, new_el, sun_az, sun_alt)
    sep_before = angular_separation(mount_az, mount_el, sun_az, sun_alt)
    assert new_sep >= sep_before - 1e-6


def test_jog_never_decreases_separation_over_random_inputs():
    """Property check: the function must not pick a direction that
    brings the mount closer to the sun."""
    rng = __import__("random").Random(1234)
    failures = []
    for _ in range(200):
        sun_az = rng.uniform(0, 360)
        sun_alt = rng.uniform(-5, 75)
        # Mount somewhere in the 30° cone around the sun.
        daz = rng.uniform(-20, 20)
        del_ = rng.uniform(-20, 20)
        mount_az = (sun_az + daz) % 360.0
        mount_el = max(-85.0, min(85.0, sun_alt + del_))
        if math.hypot(daz, del_) < 0.1:
            continue  # degenerate: mount coincides with sun
        sep_before = angular_separation(mount_az, mount_el, sun_az, sun_alt)
        angle = compute_jog_angle(mount_az, mount_el, sun_az, sun_alt)
        sep_after = _sep_after_jog(mount_az, mount_el, sun_az, sun_alt, angle)
        if sep_after < sep_before - 1e-3:
            failures.append(
                f"sun=({sun_az:.1f},{sun_alt:.1f}) mount=({mount_az:.1f},{mount_el:.1f})"
                f" angle={angle} sep {sep_before:.2f}→{sep_after:.2f}"
            )
    assert not failures, "jog decreased separation:\n" + "\n".join(failures[:5])


# --- SunSafetyMonitor: lockout + jog behavior ----------------------------


class _FakeJog:
    def __init__(self) -> None:
        self.calls: list[tuple[int, int, int]] = []
        self.was_locked_during_call: list[bool] = []
        self._monitor: SunSafetyMonitor | None = None

    def bind(self, m: SunSafetyMonitor) -> None:
        self._monitor = m

    def __call__(self, speed: int, angle: int, dur: int) -> None:
        self.calls.append((speed, angle, dur))
        if self._monitor is not None:
            self.was_locked_during_call.append(self._monitor.is_locked_out())


def test_monitor_trips_and_calls_abort_then_jog_then_releases():
    # Mount pointing RIGHT at a fixed sun position; monitor should trip.
    sun_pos = (180.0, 30.0)

    # Patch compute_sun_altaz via the module so tick() sees a known sun.
    from device import sun_safety as ss

    real = ss.compute_sun_altaz
    ss.compute_sun_altaz = lambda **kw: sun_pos
    try:
        aborts: list[int] = []
        jog = _FakeJog()

        def reader():
            # In the cone before the jog; clear of it afterwards — the
            # post-jog verify re-jogs only when the jog was truncated.
            return (181.0, 31.0) if not jog.calls else (90.0, 10.0)

        m = SunSafetyMonitor(
            altaz_reader=reader,
            jog_command=jog,
            abort_active=lambda: aborts.append(1),
            lat_deg=33.96,
            lon_deg=-118.46,
            jog_duration_s=0,  # make the post-jog sleep fast
        )
        jog.bind(m)
        # Drive one tick in-line; don't bother with the thread loop.
        m._tick()
    finally:
        ss.compute_sun_altaz = real

    assert len(aborts) == 1, "abort_active should be called exactly once"
    assert len(jog.calls) == 1, "jog_command should be called exactly once"
    speed, angle, dur = jog.calls[0]
    assert speed == 1440
    assert dur == 0
    assert 0 <= angle < 360
    # Lockout should have been set during the jog call.
    assert jog.was_locked_during_call == [True]
    # After _trigger_emergency returns, lockout should be cleared.
    assert m.is_locked_out() is False
    trip = m.last_trip()
    assert trip is not None
    assert trip.cone_deg == 30.0
    assert trip.separation_deg < 30.0
    assert trip.jog_angle_deg == angle


def test_monitor_skips_when_sun_below_threshold():
    from device import sun_safety as ss

    real = ss.compute_sun_altaz
    ss.compute_sun_altaz = lambda **kw: (180.0, -15.0)  # below -10° default
    try:
        jog = _FakeJog()
        m = SunSafetyMonitor(
            altaz_reader=lambda: (180.0, -15.0),  # pointing RIGHT at sun
            jog_command=jog,
            lat_deg=33.96,
            lon_deg=-118.46,
            jog_duration_s=0,
        )
        m._tick()
    finally:
        ss.compute_sun_altaz = real
    assert jog.calls == []
    assert m.last_trip() is None


def test_monitor_does_not_trip_when_altaz_reader_returns_none():
    # Simulates "mount not plate-solved; RA/Dec unreliable".
    from device import sun_safety as ss

    real = ss.compute_sun_altaz
    ss.compute_sun_altaz = lambda **kw: (180.0, 30.0)
    try:
        jog = _FakeJog()
        m = SunSafetyMonitor(
            altaz_reader=lambda: None,
            jog_command=jog,
            lat_deg=33.96,
            lon_deg=-118.46,
            jog_duration_s=0,
        )
        m._tick()
    finally:
        ss.compute_sun_altaz = real
    assert jog.calls == []
    assert m.last_trip() is None


def test_monitor_skip_when_disabled():
    from device import sun_safety as ss

    real = ss.compute_sun_altaz
    ss.compute_sun_altaz = lambda **kw: (180.0, 30.0)
    try:
        jog = _FakeJog()
        m = SunSafetyMonitor(
            altaz_reader=lambda: (180.0, 30.0),
            jog_command=jog,
            lat_deg=33.96,
            lon_deg=-118.46,
            jog_duration_s=0,
            enabled=False,
        )
        m._tick()
    finally:
        ss.compute_sun_altaz = real
    assert jog.calls == []


def test_monitor_dismiss_hides_last_trip():
    from device import sun_safety as ss

    real = ss.compute_sun_altaz
    ss.compute_sun_altaz = lambda **kw: (180.0, 30.0)
    try:
        m = SunSafetyMonitor(
            altaz_reader=lambda: (181.0, 31.0),
            jog_command=_FakeJog(),
            lat_deg=33.96,
            lon_deg=-118.46,
            jog_duration_s=0,
        )
        m._tick()
    finally:
        ss.compute_sun_altaz = real
    assert m.last_trip() is not None
    m.dismiss_last_trip()
    assert m.last_trip() is None


def test_reload_updates_thresholds_without_restart():
    m = SunSafetyMonitor(
        altaz_reader=lambda: None,
        jog_command=_FakeJog(),
        lat_deg=33.96,
        lon_deg=-118.46,
    )
    assert m.min_separation_deg == 30.0
    assert m.enabled is True
    m.reload(min_separation_deg=15.0, enabled=False)
    assert m.min_separation_deg == 15.0
    assert m.enabled is False


# --- module-level singleton helpers --------------------------------------


def test_sun_safety_is_locked_out_without_monitor():
    from device.sun_safety import (
        get_sun_monitor,
        set_sun_monitor,
        sun_safety_is_locked_out,
    )

    prev = get_sun_monitor()
    set_sun_monitor(None)
    try:
        assert sun_safety_is_locked_out() is False
    finally:
        set_sun_monitor(prev)


def test_set_and_get_sun_monitor_roundtrip():
    from device.sun_safety import get_sun_monitor, set_sun_monitor

    prev = get_sun_monitor()
    m = SunSafetyMonitor(
        altaz_reader=lambda: None,
        jog_command=_FakeJog(),
        lat_deg=0.0,
        lon_deg=0.0,
    )
    set_sun_monitor(m)
    try:
        assert get_sun_monitor() is m
    finally:
        set_sun_monitor(prev)


# --- speed_move wrapper: honors emergency lockout ------------------------


class _DummyCli:
    """Minimal MountClient double — just records method_sync calls."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def method_sync(self, method: str, params=None):
        self.calls.append((method, dict(params or {})))
        return {"result": {}}


def test_speed_move_passes_through_when_not_locked():
    from device.sun_safety import get_sun_monitor, set_sun_monitor
    from device.velocity_controller import speed_move

    prev = get_sun_monitor()
    set_sun_monitor(None)
    try:
        cli = _DummyCli()
        speed_move(cli, speed=100, angle=45, dur_sec=1)
        assert cli.calls == [
            ("scope_speed_move", {"speed": 100, "angle": 45, "dur_sec": 1}),
        ]
    finally:
        set_sun_monitor(prev)


def test_speed_move_refuses_while_monitor_locked_out():
    from device.sun_safety import SunSafetyLocked, get_sun_monitor, set_sun_monitor
    from device.velocity_controller import speed_move

    prev = get_sun_monitor()
    m = SunSafetyMonitor(
        altaz_reader=lambda: None,
        jog_command=_FakeJog(),
        lat_deg=0.0,
        lon_deg=0.0,
    )
    # Simulate the monitor mid-jog.
    m._emergency_lockout.set()
    set_sun_monitor(m)
    try:
        cli = _DummyCli()
        with pytest.raises(SunSafetyLocked):
            speed_move(cli, speed=100, angle=45, dur_sec=1)
        assert cli.calls == []  # firmware never touched
    finally:
        set_sun_monitor(prev)


# --- P1: fail CLOSED when the observer location is unset -------------------


def test_is_sun_safe_fails_closed_when_location_unset():
    # Explicit 0,0 sentinel: the sun position at Null Island cannot be
    # trusted to represent the real site, so refuse rather than fail open.
    safe, reason = is_sun_safe(180.0, 33.0, lat_deg=0.0, lon_deg=0.0)
    assert safe is False
    assert "location is not set" in reason


def test_is_sun_safe_fails_closed_when_config_location_unset(monkeypatch):
    # lat/lon omitted → resolved from Config; the fresh-install default is
    # 0/0, which must fail closed for every pointing (even at night).
    from device import config as _config

    monkeypatch.setattr(_config.Config, "init_lat", 0.0)
    monkeypatch.setattr(_config.Config, "init_long", 0.0)
    safe, reason = is_sun_safe(10.0, 5.0)
    assert safe is False
    assert "location is not set" in reason


def test_is_sun_safe_ok_when_location_is_set():
    # Sanity: a real site at night is still allowed (not tripped by the
    # unset-location guard).
    safe, reason = is_sun_safe(
        180.0,
        33.0,
        lat_deg=SITE_LAT,
        lon_deg=SITE_LON,
        when=_midnight_utc(),
    )
    assert safe is True
    assert reason == ""


def test_monitor_warns_and_skips_when_location_unset(caplog):
    jog = _FakeJog()
    m = SunSafetyMonitor(
        altaz_reader=lambda: (180.0, 30.0),  # pointing right at where sun is
        jog_command=jog,
        lat_deg=0.0,
        lon_deg=0.0,
        jog_duration_s=0,
    )
    with caplog.at_level(logging.ERROR, logger="device.sun_safety"):
        m._tick()
    assert jog.calls == [], "blind monitor must not jog on an untrusted site"
    assert m.last_trip() is None
    assert any("SUN SAFETY BLIND" in r.message for r in caplog.records)


# --- P1: encoder-based sensing (make_scope_altaz_reader) ------------------


def test_make_scope_altaz_reader_returns_encoder_azel():
    # scope_get_horiz_coord → [alt, az]; reader yields (az, alt).
    reader = make_scope_altaz_reader(lambda *_a, **_k: {"result": [33.0, 180.0]})
    assert reader() == (180.0, 33.0)


def test_make_scope_altaz_reader_wraps_az():
    reader = make_scope_altaz_reader(lambda *_a, **_k: {"result": [12.0, 370.0]})
    az, alt = reader()
    assert az == pytest.approx(10.0)
    assert alt == pytest.approx(12.0)


def test_make_scope_altaz_reader_none_on_missing_or_bad():
    assert make_scope_altaz_reader(lambda *_a, **_k: {"ok": 1})() is None
    assert make_scope_altaz_reader(lambda *_a, **_k: "nope")() is None
    assert make_scope_altaz_reader(lambda *_a, **_k: {"result": [1.0]})() is None
    assert (
        make_scope_altaz_reader(lambda *_a, **_k: {"result": [float("nan"), 1.0]})()
        is None
    )

    def _boom(*_a, **_k):
        raise RuntimeError("socket down")

    assert make_scope_altaz_reader(_boom)() is None


def test_monitor_trips_for_uncalibrated_scope_via_encoder():
    # Regression: RA/Dec reads ra==dec==0 (pre-plate-solve) during the day,
    # but the encoder reader still senses a real pointing near the sun. The
    # monitor must trip on the encoder reading.
    from device import sun_safety as ss

    sun_pos = (180.0, 30.0)
    real = ss.compute_sun_altaz
    ss.compute_sun_altaz = lambda **kw: sun_pos
    try:
        jog = _FakeJog()

        # Encoder reports (alt=31, az=181) → 1.4° from the sun before the
        # jog; clear of the cone once the jog has run (so the post-jog
        # verify doesn't re-jog).
        def _horiz(*_a, **_k):
            return {"result": [31.0, 181.0] if not jog.calls else [10.0, 90.0]}

        reader = make_scope_altaz_reader(_horiz)
        m = SunSafetyMonitor(
            altaz_reader=reader,
            jog_command=jog,
            lat_deg=33.96,
            lon_deg=-118.46,
            jog_duration_s=0,
        )
        m._tick()
    finally:
        ss.compute_sun_altaz = real
    assert len(jog.calls) == 1, "uncalibrated (encoder-sensed) scope must trip"
    trip = m.last_trip()
    assert trip is not None
    assert trip.separation_deg < 30.0


# --- P1: emergency jog is issued BEFORE abort_active ----------------------


def test_trip_jogs_before_aborting():
    from device import sun_safety as ss

    sun_pos = (180.0, 30.0)
    real = ss.compute_sun_altaz
    ss.compute_sun_altaz = lambda **kw: sun_pos
    try:
        order: list[str] = []

        def _jog(_s, _a, _d):
            order.append("jog")

        def _abort():
            order.append("abort")

        def _reader():
            # In-cone before the jog, clear afterwards — keeps the
            # post-jog verify from re-jogging in this ordering test.
            return (181.0, 31.0) if "jog" not in order else (90.0, 10.0)

        m = SunSafetyMonitor(
            altaz_reader=_reader,
            jog_command=_jog,
            abort_active=_abort,
            lat_deg=33.96,
            lon_deg=-118.46,
            jog_duration_s=0,
        )
        m._tick()
    finally:
        ss.compute_sun_altaz = real
    assert order == ["jog", "abort"], (
        "mount must jog out of the cone before slow abort_active teardown"
    )


# --- P1: geometry consolidation -------------------------------------------


def test_angular_separation_delegates_to_geometry():
    from device import geometry as geo

    for args in [
        (12.0, 34.0, 200.0, -5.0),
        (0.0, 0.0, 90.0, 0.0),
        (350.0, 10.0, 10.0, -10.0),
    ]:
        assert angular_separation(*args) == geo.angular_separation_deg(*args)


def test_sun_safety_reuses_geometry_wrap_pm180():
    from device import geometry as geo
    from device import sun_safety as ss

    # The local (-180,180] duplicate was removed in favour of the shared
    # [-180,180) helper.
    assert ss.wrap_pm180 is geo.wrap_pm180
    assert not hasattr(ss, "_wrap_pm180")


def test_monitor_rejogs_when_teardown_truncates_jog():
    """A session teardown's direct motor-stop can cancel the in-flight
    emergency jog. The monitor must verify separation after each jog and
    re-jog (uncontested, sessions now aborted) until it is out of the cone."""
    sun_pos = (180.0, 30.0)
    from device import sun_safety as ss

    real = ss.compute_sun_altaz
    ss.compute_sun_altaz = lambda **kw: sun_pos
    try:
        jog = _FakeJog()

        def reader():
            # Still inside the cone until the second jog lands (jog #1
            # "truncated" by a teardown stop), then well clear.
            return (181.0, 31.0) if len(jog.calls) < 2 else (90.0, 10.0)

        m = SunSafetyMonitor(
            altaz_reader=reader,
            jog_command=jog,
            abort_active=lambda: None,
            lat_deg=33.96,
            lon_deg=-118.46,
            jog_duration_s=0,
        )
        jog.bind(m)
        m._tick()
    finally:
        ss.compute_sun_altaz = real

    assert len(jog.calls) == 2, "truncated jog must be re-issued once"
    assert m.is_locked_out() is False
    assert m.is_jog_in_progress() is False


def test_sun_safety_jog_in_progress_helper():
    """Module helper reflects the monitor's jog window; False without a
    monitor installed (test mode / CLI)."""
    import time as _time

    from device import sun_safety as ss

    assert ss.sun_safety_jog_in_progress() is False
    m = SunSafetyMonitor(
        altaz_reader=lambda: None,
        jog_command=lambda *a: None,
        lat_deg=33.96,
        lon_deg=-118.46,
    )
    prev = ss.get_sun_monitor()
    ss.set_sun_monitor(m)
    try:
        assert ss.sun_safety_jog_in_progress() is False
        with m._lock:
            m._jog_until_ts = _time.time() + 5.0
        assert ss.sun_safety_jog_in_progress() is True
    finally:
        ss.set_sun_monitor(prev)


def test_jog_in_progress_is_scoped_to_the_jogged_telescope():
    """The monitor jogs exactly one scope; a jog on the primary must not
    suppress motor stops on OTHER mounts. Unknown ids match conservatively."""
    import time as _time

    from device import sun_safety as ss

    m = SunSafetyMonitor(
        altaz_reader=lambda: None,
        jog_command=lambda *a: None,
        lat_deg=33.96,
        lon_deg=-118.46,
        jog_telescope_id=1,
    )
    prev = ss.get_sun_monitor()
    ss.set_sun_monitor(m)
    try:
        with m._lock:
            m._jog_until_ts = _time.time() + 5.0
        assert ss.sun_safety_jog_in_progress(1) is True
        assert ss.sun_safety_jog_in_progress(2) is False, (
            "stop on a different mount must not be suppressed"
        )
        assert ss.sun_safety_jog_in_progress(None) is True  # unknown caller
    finally:
        ss.set_sun_monitor(prev)


def test_jog_in_progress_unknown_jog_scope_matches_all():
    """Without a configured jog_telescope_id the window conservatively
    matches every telescope (pre-scoping behavior)."""
    import time as _time

    from device import sun_safety as ss

    m = SunSafetyMonitor(
        altaz_reader=lambda: None,
        jog_command=lambda *a: None,
        lat_deg=33.96,
        lon_deg=-118.46,
    )
    prev = ss.get_sun_monitor()
    ss.set_sun_monitor(m)
    try:
        with m._lock:
            m._jog_until_ts = _time.time() + 5.0
        assert ss.sun_safety_jog_in_progress(7) is True
    finally:
        ss.set_sun_monitor(prev)
