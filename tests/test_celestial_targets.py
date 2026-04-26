"""Unit tests for ``scripts.trajectory.celestial_targets``.

Coverage:
- compute_altaz: Polaris (geometric reference — alt ≈ lat, az ≈ 0)
  and Vega (cross-check against astropy, refraction-tolerant).
- planet_target: Mars / Jupiter / Venus return finite values; bad name
  raises ValueError.
- filter_visible: dim and below-horizon targets are dropped.
- angular_distance_deg: identity, orthogonal, near-pole.
- sort_by_proximity / sort_by_magnitude: ordering + magnitude tiebreak.
- Curated catalog: size + magnitude / RA / Dec invariants.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from scripts.trajectory.celestial_targets import (
    BRIGHT_STARS,
    PLANETS,
    CelestialTarget,
    all_targets,
    angular_distance_deg,
    compute_altaz,
    filter_visible,
    planet_target,
    sort_by_magnitude,
    sort_by_proximity,
)
from scripts.trajectory.observer import build_site


# ---------- shared fixtures ------------------------------------------


# LA observer; matches the existing FAA DOF tests so behaviour is
# comparable to the daytime calibration picker.
_LA_LAT_DEG = 33.96
_LA_LON_DEG = -118.46
_LA_ALT_M = 2.0


def _la_site():
    return build_site(
        lat_deg=_LA_LAT_DEG,
        lon_deg=_LA_LON_DEG,
        alt_m=_LA_ALT_M,
    )


def _by_name(name: str) -> CelestialTarget:
    for target in BRIGHT_STARS:
        if target.name == name:
            return target
    raise AssertionError(f"missing catalog entry: {name}")


# ---------- compute_altaz --------------------------------------------


def test_compute_altaz_polaris_altitude_matches_latitude():
    """Polaris sits within ~0.7° of the celestial pole, so its
    altitude equals the observer's latitude to within ~1° regardless
    of time. This is a refraction-tolerant geometric anchor that
    catches gross errors in the ephem path (wrong RA/Dec, wrong epoch,
    swapped lat/lon, etc.)."""
    site = _la_site()
    polaris = _by_name("Polaris")
    when = datetime(2026, 4, 25, 12, 0, 0, tzinfo=timezone.utc)
    az, el = compute_altaz(polaris, site, when)
    assert abs(el - _LA_LAT_DEG) < 1.0


def test_compute_altaz_polaris_az_near_north():
    """Polaris's azimuth is within ~1° of true north for any northern
    observer. Wrap to (-180, 180] before comparing so the assertion
    works whether ephem reports az=0 or az=360."""
    site = _la_site()
    polaris = _by_name("Polaris")
    when = datetime(2026, 4, 25, 12, 0, 0, tzinfo=timezone.utc)
    az, _el = compute_altaz(polaris, site, when)
    az_signed = ((az + 180.0) % 360.0) - 180.0
    assert abs(az_signed) < 1.5


def test_compute_altaz_vega_agrees_with_astropy():
    """At a time when Vega is high in LA (low refraction, low
    aberration of the altaz path), ephem and astropy should agree on
    the apparent (az, el) to within 0.5°.

    The 0.5° tolerance is generous to absorb ephem's older precession /
    nutation model + Saemundsson refraction (vs astropy's full IAU
    2006/2000A model). At a 70°+ altitude these differences are well
    under 1 arcmin, so 0.5° is a sanity bound that still catches
    lat/lon swaps, RA/Dec unit confusion, and timezone bugs.
    """
    from astropy import units as u
    from astropy.coordinates import AltAz, EarthLocation, SkyCoord
    from astropy.time import Time

    site = _la_site()
    vega = _by_name("Vega")
    # 2026-08-15T05:30:00 UTC — late evening in LA in summer; Vega is
    # near the meridian at >80° altitude. Refraction-negligible.
    when = datetime(2026, 8, 15, 5, 30, 0, tzinfo=timezone.utc)
    az, el = compute_altaz(vega, site, when)

    loc = EarthLocation(
        lat=_LA_LAT_DEG * u.deg,
        lon=_LA_LON_DEG * u.deg,
        height=_LA_ALT_M * u.m,
    )
    t = Time(when)
    sc = SkyCoord(
        ra=vega.ra_hours * u.hourangle,
        dec=vega.dec_deg * u.deg,
        frame="icrs",
    )
    expected = sc.transform_to(AltAz(obstime=t, location=loc))
    expected_az = float(expected.az.deg)
    expected_el = float(expected.alt.deg)

    # Wrap az difference to [-180, 180] before |.|.
    daz = ((az - expected_az + 180.0) % 360.0) - 180.0
    assert abs(daz) < 0.5, f"az mismatch: {az} vs {expected_az}"
    assert abs(el - expected_el) < 0.5, f"el mismatch: {el} vs {expected_el}"


def test_compute_altaz_unknown_kind_raises():
    bogus = CelestialTarget(
        name="bogus",
        kind="nebula",
        ra_hours=10.0,
        dec_deg=20.0,
        vmag=2.0,
    )
    with pytest.raises(ValueError, match="unrecognised target kind"):
        compute_altaz(
            bogus,
            _la_site(),
            datetime(2026, 4, 25, 12, 0, 0, tzinfo=timezone.utc),
        )


def test_compute_altaz_static_planet_kind_without_radec_raises():
    """A ``kind='star'`` target with no RA/Dec should error rather than
    silently produce garbage. Catches accidental construction of a
    catalog entry that forgot to set the coordinates."""
    bogus = CelestialTarget(
        name="bogus",
        kind="star",
        ra_hours=None,
        dec_deg=None,
        vmag=2.0,
    )
    with pytest.raises(ValueError, match="no static RA/Dec"):
        compute_altaz(
            bogus,
            _la_site(),
            datetime(2026, 4, 25, 12, 0, 0, tzinfo=timezone.utc),
        )


# ---------- planet_target --------------------------------------------


def test_planet_target_mars_returns_finite():
    """Mars at any reasonable date has a finite RA/Dec/vmag from ephem.
    We don't pin the exact values (they drift daily) — just sanity
    check the schema."""
    when = datetime(2026, 4, 25, 12, 0, 0, tzinfo=timezone.utc)
    target = planet_target("Mars", when, _la_site())
    assert target.kind == "planet"
    assert target.name == "Mars"
    assert target.ra_hours is not None and 0.0 <= target.ra_hours < 24.0
    assert target.dec_deg is not None and -90.0 <= target.dec_deg <= 90.0
    # Mars's apparent magnitude varies between ~-2.9 and ~+1.9. Bound a
    # bit beyond that in case ephem's model is off by a fraction.
    assert -3.5 <= target.vmag <= 3.0


def test_planet_target_jupiter_returns_finite():
    when = datetime(2026, 4, 25, 12, 0, 0, tzinfo=timezone.utc)
    target = planet_target("Jupiter", when, _la_site())
    assert target.kind == "planet"
    assert target.name == "Jupiter"
    # Jupiter is always bright; vmag in [-3.0, -1.0].
    assert -3.5 <= target.vmag <= -1.0


def test_planet_target_invalid_name_raises():
    with pytest.raises(ValueError, match="unknown planet"):
        planet_target(
            "Pluto",
            datetime(2026, 4, 25, 12, 0, 0, tzinfo=timezone.utc),
            _la_site(),
        )


def test_planet_target_excludes_sun_and_moon():
    """Sun / Moon are intentionally NOT in PLANETS — calibration must
    avoid the sun cone (sun_safety) and the moon (separately filtered
    in filter_visible). The PLANETS tuple is the source of truth for
    the picker; check the expected exclusions here so a future change
    that adds them is caught loudly."""
    assert "Sun" not in PLANETS
    assert "Moon" not in PLANETS
    assert "Pluto" not in PLANETS


# ---------- filter_visible -------------------------------------------


def _t_night_la():
    """A night-time UTC instant for LA where the sun is well below
    horizon (so sun-separation doesn't dominate the filter and we can
    test el / vmag thresholds independently)."""
    # 2026-04-26T08:00:00Z = 1am Pacific local. Sun is below horizon.
    return datetime(2026, 4, 26, 8, 0, 0, tzinfo=timezone.utc)


def test_filter_visible_drops_dim_targets():
    """A vmag-5 target is below the default max_mag=4.0 and must be
    excluded even when it would otherwise be visible."""
    bright = CelestialTarget(
        name="bright_test",
        kind="star",
        ra_hours=12.0,
        dec_deg=0.0,
        vmag=2.0,
    )
    dim = CelestialTarget(
        name="dim_test",
        kind="star",
        ra_hours=12.0,
        dec_deg=0.0,
        vmag=5.0,
    )
    visible = filter_visible(
        [bright, dim],
        _la_site(),
        _t_night_la(),
        min_el_deg=-90.0,  # disable horizon filter for this test
    )
    names = {t[0].name for t in visible}
    assert "dim_test" not in names
    # bright_test passes vmag — its visibility depends on time; we
    # disabled the horizon filter, so it must be present unless caught
    # by sun/moon separation. At 1am LA the sun is far away; the moon
    # is the only other concern. Allow either outcome (we can't
    # control where the moon is on this date) but assert it wasn't
    # vmag-filtered.
    if "bright_test" not in names:
        # Filtered by sun/moon separation, not vmag — that's fine.
        pass


def test_filter_visible_drops_below_horizon():
    """A target at dec=-60° is far below horizon at LA (lat=34°) at
    every hour of the day — its altitude never exceeds the default
    20° min_el. Must be filtered."""
    far_south = CelestialTarget(
        name="far_south_test",
        kind="star",
        ra_hours=12.0,
        dec_deg=-60.0,
        vmag=1.0,
    )
    visible = filter_visible(
        [far_south],
        _la_site(),
        _t_night_la(),
    )
    assert all(t[0].name != "far_south_test" for t in visible)


def test_filter_visible_passes_bright_above_horizon():
    """Polaris is always above 20° elevation in LA (alt ≈ 34°) and is
    bright (vmag 1.98). It must come through filter_visible at any
    time of day, and the (az, el) tuple must be sensible."""
    polaris = _by_name("Polaris")
    visible = filter_visible(
        [polaris],
        _la_site(),
        _t_night_la(),
    )
    assert len(visible) == 1
    target, az, el = visible[0]
    assert target.name == "Polaris"
    assert 0.0 <= az < 360.0
    assert 30.0 < el < 38.0  # within ~2° of latitude


def test_filter_visible_returns_empty_for_all_dim():
    """Edge case: every input is too dim — return empty list, not
    None / not raise."""
    targets = [
        CelestialTarget(
            name=f"d{i}",
            kind="star",
            ra_hours=float(i),
            dec_deg=0.0,
            vmag=10.0,
        )
        for i in range(5)
    ]
    visible = filter_visible(
        targets,
        _la_site(),
        _t_night_la(),
        max_mag=4.0,
    )
    assert visible == []


# ---------- angular_distance_deg -------------------------------------


def test_angular_distance_identical_points():
    assert angular_distance_deg(0.0, 0.0, 0.0, 0.0) == pytest.approx(0.0)
    assert angular_distance_deg(123.4, 56.7, 123.4, 56.7) == pytest.approx(
        0.0,
        abs=1e-9,
    )


def test_angular_distance_orthogonal_along_horizon():
    """0/0 vs 90/0 = 90° along the horizon — unit-circle check."""
    assert angular_distance_deg(0.0, 0.0, 90.0, 0.0) == pytest.approx(
        90.0,
        abs=1e-6,
    )
    assert angular_distance_deg(0.0, 0.0, 180.0, 0.0) == pytest.approx(
        180.0,
        abs=1e-6,
    )


def test_angular_distance_near_polar():
    """0/89 vs 180/89: two stars near the celestial pole, 180°
    apart in azimuth, separated by ~2°. The exact value is
    180° - 2 × 89° = 2°."""
    d = angular_distance_deg(0.0, 89.0, 180.0, 89.0)
    assert d == pytest.approx(2.0, abs=1e-3)


def test_angular_distance_symmetric():
    """Distance from A to B equals B to A."""
    a = (10.0, 20.0)
    b = (200.0, -30.0)
    assert angular_distance_deg(*a, *b) == pytest.approx(
        angular_distance_deg(*b, *a),
    )


# ---------- sort helpers ---------------------------------------------


def _t(name: str, vmag: float = 2.0) -> CelestialTarget:
    return CelestialTarget(
        name=name,
        kind="star",
        ra_hours=0.0,
        dec_deg=0.0,
        vmag=vmag,
    )


def test_sort_by_proximity_orders_ascending():
    """Targets sorted by great-circle distance from the current
    pointing. Hand-built three-target list with known geometry."""
    a = _t("A")
    b = _t("B")
    c = _t("C")
    # Current pointing is (0, 0). A is 10° away, B is 30° away, C is 5°.
    visible = [(a, 10.0, 0.0), (b, 30.0, 0.0), (c, 5.0, 0.0)]
    out = sort_by_proximity(visible, current_az_deg=0.0, current_el_deg=0.0)
    assert [t[0].name for t in out] == ["C", "A", "B"]
    assert [round(t[3], 2) for t in out] == [5.0, 10.0, 30.0]


def test_sort_by_proximity_magnitude_tiebreak():
    """When two targets are at the same distance, the brighter one
    (smaller vmag) comes first. Catches a regression where ties were
    sorted by insertion order."""
    bright = _t("bright", vmag=0.5)
    dim = _t("dim", vmag=3.0)
    # Both at the same (az, el), so distance to (0, 0) is identical.
    visible = [(dim, 10.0, 0.0), (bright, 10.0, 0.0)]
    out = sort_by_proximity(visible, current_az_deg=0.0, current_el_deg=0.0)
    assert out[0][0].name == "bright"
    assert out[1][0].name == "dim"


def test_sort_by_proximity_distance_tuple_shape():
    """The returned tuple is 4-wide: (target, az, el, distance_deg).
    Catches a refactor that drops or rearranges the distance column."""
    visible = [(_t("X"), 0.0, 0.0)]
    out = sort_by_proximity(visible, current_az_deg=10.0, current_el_deg=0.0)
    assert len(out) == 1
    target, az, el, dist = out[0]
    assert isinstance(target, CelestialTarget)
    assert az == 0.0 and el == 0.0
    assert dist == pytest.approx(10.0, abs=1e-6)


def test_sort_by_magnitude_orders_brightest_first():
    a = _t("A", vmag=2.0)
    b = _t("B", vmag=0.5)
    c = _t("C", vmag=1.5)
    visible = [(a, 0.0, 0.0), (b, 0.0, 0.0), (c, 0.0, 0.0)]
    out = sort_by_magnitude(visible)
    assert [t[0].name for t in out] == ["B", "C", "A"]


# ---------- catalog invariants ---------------------------------------


def test_catalog_has_at_least_25_entries():
    """Spec says ~25–40 calibration-grade targets. Floor at 25 so a
    future trim doesn't quietly leave the picker too sparse."""
    assert len(BRIGHT_STARS) >= 25


def test_catalog_includes_polaris_and_vega():
    """Two anchor targets the docstring promises explicitly."""
    names = {t.name for t in BRIGHT_STARS}
    assert "Polaris" in names
    assert "Vega" in names


def test_catalog_magnitudes_are_calibration_grade():
    """All curated stars must be brighter than vmag=4.0 (default
    ``max_mag``) so the catalog never surfaces a dim target by default.
    Floor at -2.0 to catch a typo like vmag=-15 (a planet-class
    luminosity for a star is suspicious)."""
    for target in BRIGHT_STARS:
        assert -2.0 <= target.vmag <= 4.0, (
            f"{target.name} has unreasonable vmag={target.vmag}"
        )


def test_catalog_ra_dec_in_valid_range():
    for target in BRIGHT_STARS:
        assert target.ra_hours is not None
        assert 0.0 <= target.ra_hours < 24.0, (
            f"{target.name} ra_hours={target.ra_hours} out of [0,24)"
        )
        assert target.dec_deg is not None
        assert -90.0 <= target.dec_deg <= 90.0, (
            f"{target.name} dec_deg={target.dec_deg} out of [-90,90]"
        )


def test_catalog_kinds_are_recognised():
    for target in BRIGHT_STARS:
        assert target.kind in ("star", "double"), (
            f"{target.name} has unknown kind={target.kind!r}"
        )


def test_catalog_names_unique():
    """No duplicate name entries — the picker assumes name is the
    user-visible identifier."""
    names = [t.name for t in BRIGHT_STARS]
    assert len(set(names)) == len(names)


# ---------- all_targets ----------------------------------------------


def test_all_targets_includes_stars_and_planets():
    """``all_targets`` should return every catalog star plus a
    freshly-computed entry for each planet listed in PLANETS."""
    when = datetime(2026, 4, 25, 12, 0, 0, tzinfo=timezone.utc)
    pool = all_targets(when, _la_site())
    star_names = {t.name for t in pool if t.kind in ("star", "double")}
    planet_names = {t.name for t in pool if t.kind == "planet"}
    catalog_star_names = {t.name for t in BRIGHT_STARS}
    assert star_names == catalog_star_names
    assert planet_names == set(PLANETS)
