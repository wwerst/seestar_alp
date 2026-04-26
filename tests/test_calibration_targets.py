"""Tests for :mod:`device.calibration_targets`.

Covers the tagged-union spec, per-kind ``resolve_true_altaz`` paths,
the plate-solve handoff, and the sigma plumbing the unified session
threads through to the residual table.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

import pytest

from device.calibration_targets import (
    DEFAULT_OPERATOR_CENTERING_SIGMA_DEG,
    DEFAULT_PLATESOLVE_SIGMA_DEG,
    CalibrationTargetSpec,
    PlateSolveOutcome,
    TargetKind,
)
from scripts.trajectory.faa_dof import (
    HYPERION_06_000301,
    LA_BROADCAST_06_000177,
)
from scripts.trajectory.observer import build_site


DOCKWEILER = dict(lat_deg=33.9615051, lon_deg=-118.4581361, alt_m=2.0)
LA = dict(lat_deg=34.0522, lon_deg=-118.2437, alt_m=89.0)


def _site():
    return build_site(**DOCKWEILER)


# ---------- factories + dataclass shape ------------------------------


def test_from_landmark_builds_faa_spec():
    spec = CalibrationTargetSpec.from_landmark(HYPERION_06_000301, slant_m=5523.0)
    assert spec.kind is TargetKind.FAA
    assert spec.landmark is HYPERION_06_000301
    assert spec.slant_m == pytest.approx(5523.0)
    assert "Hyperion" in spec.label
    assert spec.ra_hours is None
    assert spec.dec_deg is None


def test_celestial_factory_normalises_floats():
    spec = CalibrationTargetSpec.celestial(
        "Vega", ra_hours=18.6157, dec_deg=38.7837, vmag=0.03, bayer="alpha Lyr"
    )
    assert spec.kind is TargetKind.CELESTIAL
    assert spec.label == "Vega"
    assert spec.ra_hours == pytest.approx(18.6157)
    assert spec.dec_deg == pytest.approx(38.7837)
    assert spec.vmag == pytest.approx(0.03)
    assert spec.bayer == "alpha Lyr"
    assert spec.landmark is None


def test_platesolve_factory_accepts_optional_seed():
    spec_free = CalibrationTargetSpec.platesolve("free aim 1")
    assert spec_free.kind is TargetKind.PLATESOLVE
    assert spec_free.seed_az_deg is None
    assert spec_free.seed_el_deg is None
    spec_seeded = CalibrationTargetSpec.platesolve(
        "Polaris region", seed_az_deg=359.0, seed_el_deg=33.5
    )
    assert spec_seeded.seed_az_deg == pytest.approx(359.0)
    assert spec_seeded.seed_el_deg == pytest.approx(33.5)


# ---------- resolve_true_altaz: FAA ----------------------------------


def test_resolve_faa_returns_known_topocentric_position():
    """Hyperion from Dockweiler: roughly south-east, low elevation."""
    site = _site()
    spec = CalibrationTargetSpec.from_landmark(HYPERION_06_000301)
    az, el, slant = spec.resolve_true_altaz(site, datetime.now(timezone.utc))
    # Az is wrapped to [-180, 180); Hyperion sits roughly south of
    # Dockweiler, so the wrapped az should be close to -180/+180 area
    # but we don't pin it tight here. Check it's a finite number.
    assert math.isfinite(az)
    # Elevation is small (ground landmark), positive (above horizon).
    assert -1.0 < el < 5.0
    assert slant is not None
    assert 4000.0 < slant < 7000.0


def test_resolve_faa_is_time_independent():
    """FAA truth depends only on landmark + site. Two times → same
    answer to within sub-arcsec."""
    site = _site()
    spec = CalibrationTargetSpec.from_landmark(LA_BROADCAST_06_000177)
    t1 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    t2 = datetime(2026, 7, 1, 0, 0, 0, tzinfo=timezone.utc)
    az1, el1, slant1 = spec.resolve_true_altaz(site, t1)
    az2, el2, slant2 = spec.resolve_true_altaz(site, t2)
    assert az1 == pytest.approx(az2, abs=1e-10)
    assert el1 == pytest.approx(el2, abs=1e-10)
    assert slant1 == pytest.approx(slant2, abs=1e-6)


def test_resolve_faa_without_landmark_raises():
    spec = CalibrationTargetSpec(kind=TargetKind.FAA, label="bogus", landmark=None)
    with pytest.raises(ValueError, match="missing landmark"):
        spec.resolve_true_altaz(_site(), datetime.now(timezone.utc))


# ---------- resolve_true_altaz: celestial ----------------------------


def test_resolve_celestial_vega_from_la():
    """Vega from Los Angeles, 2026-04-26 11:00 UTC (≈ 4 AM PT, pre-dawn
    local). At that time Vega is high in the northeast — verify the az/el
    are in the expected octant. Cross-check with astropy: at LA on
    2026-04-26 11:00 UTC, Vega is at az ≈ 66.6°, el ≈ 74.4°."""
    site = build_site(**LA)
    spec = CalibrationTargetSpec.celestial(
        "Vega", ra_hours=18.6157, dec_deg=38.7837, vmag=0.03
    )
    when = datetime(2026, 4, 26, 11, 0, 0, tzinfo=timezone.utc)
    az_wrapped, el, slant = spec.resolve_true_altaz(site, when)
    # Az is wrapped to [-180, 180); add 360 to get [0, 360).
    az = az_wrapped + 360.0 if az_wrapped < 0 else az_wrapped
    assert 60.0 < az < 75.0
    assert 70.0 < el < 80.0
    assert slant is None  # no slant for celestial


def test_resolve_celestial_polaris_near_pole():
    """Polaris at LA: el ≈ observer latitude (34°), az near 0°/360°."""
    site = build_site(**LA)
    spec = CalibrationTargetSpec.celestial(
        "Polaris", ra_hours=2.5303, dec_deg=89.2641, vmag=2.0
    )
    when = datetime(2026, 4, 26, 6, 0, 0, tzinfo=timezone.utc)
    az_wrapped, el, slant = spec.resolve_true_altaz(site, when)
    assert el == pytest.approx(LA["lat_deg"], abs=1.0)
    assert slant is None
    # Polaris az should be near 0/360 — wrapping to [-180, 180) means
    # somewhere in [-5, 5].
    assert abs(az_wrapped) < 5.0


def test_resolve_celestial_drifts_in_time():
    """Vega's altaz is time-dependent. 6 h later, az has shifted by
    roughly 90° (Earth rotates 15°/hour)."""
    site = build_site(**LA)
    spec = CalibrationTargetSpec.celestial(
        "Vega", ra_hours=18.6157, dec_deg=38.7837, vmag=0.03
    )
    t1 = datetime(2026, 4, 26, 6, 0, 0, tzinfo=timezone.utc)
    t2 = datetime(2026, 4, 26, 12, 0, 0, tzinfo=timezone.utc)
    az1, _, _ = spec.resolve_true_altaz(site, t1)
    az2, _, _ = spec.resolve_true_altaz(site, t2)
    assert az1 != pytest.approx(az2, abs=10.0)


def test_resolve_celestial_without_radec_raises():
    spec = CalibrationTargetSpec(kind=TargetKind.CELESTIAL, label="ghost")
    with pytest.raises(ValueError, match="missing RA/Dec"):
        spec.resolve_true_altaz(_site(), datetime.now(timezone.utc))


def test_resolve_celestial_naive_datetime_treated_as_utc():
    """A naive datetime should be coerced to UTC, not error."""
    site = build_site(**LA)
    spec = CalibrationTargetSpec.celestial(
        "Vega", ra_hours=18.6157, dec_deg=38.7837, vmag=0.03
    )
    when = datetime(2026, 4, 26, 6, 0, 0)  # no tzinfo
    az, el, _ = spec.resolve_true_altaz(site, when)
    assert math.isfinite(az)
    assert math.isfinite(el)


# ---------- resolve_true_altaz: platesolve ---------------------------


def test_resolve_platesolve_uses_outcome_directly():
    spec = CalibrationTargetSpec.platesolve("free aim 1")
    outcome = PlateSolveOutcome(true_az_deg=72.5, true_el_deg=45.0, sigma_deg=0.005)
    az_wrapped, el, slant = spec.resolve_true_altaz(
        _site(), datetime.now(timezone.utc), plate_solve_result=outcome
    )
    # Wrapped az: 72.5 stays in [-180, 180).
    assert az_wrapped == pytest.approx(72.5, abs=1e-9)
    assert el == pytest.approx(45.0, abs=1e-9)
    assert slant is None


def test_resolve_platesolve_wraps_az_into_pm180():
    """Outcomes with az > 180 should be wrapped to [-180, 180)."""
    spec = CalibrationTargetSpec.platesolve("east horizon")
    outcome = PlateSolveOutcome(true_az_deg=359.0, true_el_deg=10.0, sigma_deg=None)
    az_wrapped, _, _ = spec.resolve_true_altaz(
        _site(), datetime.now(timezone.utc), plate_solve_result=outcome
    )
    assert az_wrapped == pytest.approx(-1.0, abs=1e-9)


def test_resolve_platesolve_without_outcome_raises():
    spec = CalibrationTargetSpec.platesolve("free aim 2")
    with pytest.raises(ValueError, match="requires a plate-solve result"):
        spec.resolve_true_altaz(_site(), datetime.now(timezone.utc))


# ---------- per-target sigma -----------------------------------------


def test_sigma_faa_carries_through_pointing_uncertainty():
    """For an FAA spec with a known accuracy class (1A) + slant,
    ``sigma_az_el_deg`` matches ``pointing_uncertainty_deg``."""
    spec = CalibrationTargetSpec.from_landmark(HYPERION_06_000301, slant_m=5523.0)
    saz, sel = spec.sigma_az_el_deg()
    # Hyperion is 1A (±50ft horiz, ±3ft vert); should yield sub-degree
    # σ given the 5.5 km slant.
    assert saz is not None and 0.001 < saz < 0.5
    assert sel is not None and 0.001 < sel < 0.5


def test_sigma_celestial_is_operator_centering_default():
    spec = CalibrationTargetSpec.celestial(
        "Vega", ra_hours=18.6157, dec_deg=38.7837, vmag=0.03
    )
    saz, sel = spec.sigma_az_el_deg()
    assert saz == DEFAULT_OPERATOR_CENTERING_SIGMA_DEG
    assert sel == DEFAULT_OPERATOR_CENTERING_SIGMA_DEG


def test_sigma_platesolve_default_when_outcome_unavailable():
    spec = CalibrationTargetSpec.platesolve("free aim 1")
    saz, sel = spec.sigma_az_el_deg()
    assert saz == DEFAULT_PLATESOLVE_SIGMA_DEG
    assert sel == DEFAULT_PLATESOLVE_SIGMA_DEG


def test_sigma_faa_without_slant_returns_none():
    """Spec without cached slant can't compute σ; return None,
    not NaN."""
    spec = CalibrationTargetSpec.from_landmark(HYPERION_06_000301)
    saz, sel = spec.sigma_az_el_deg()
    assert saz is None
    assert sel is None


# ---------- to_dict --------------------------------------------------


def test_to_dict_faa_carries_landmark_fields():
    spec = CalibrationTargetSpec.from_landmark(HYPERION_06_000301, slant_m=5523.0)
    d = spec.to_dict()
    assert d["kind"] == "faa"
    assert d["oas"] == "06-000301"
    assert d["lat_deg"] == pytest.approx(HYPERION_06_000301.lat_deg)
    assert d["accuracy_class"] == HYPERION_06_000301.accuracy_class
    assert d["slant_m"] == pytest.approx(5523.0)


def test_to_dict_celestial_carries_radec():
    spec = CalibrationTargetSpec.celestial(
        "Vega", ra_hours=18.6157, dec_deg=38.7837, vmag=0.03, bayer="alpha Lyr"
    )
    d = spec.to_dict()
    assert d["kind"] == "celestial"
    assert d["label"] == "Vega"
    assert d["ra_hours"] == pytest.approx(18.6157)
    assert d["dec_deg"] == pytest.approx(38.7837)
    assert d["vmag"] == pytest.approx(0.03)
    assert d["bayer"] == "alpha Lyr"


def test_to_dict_platesolve_only_includes_seed_when_present():
    free = CalibrationTargetSpec.platesolve("free aim 1").to_dict()
    assert free["kind"] == "platesolve"
    assert free["label"] == "free aim 1"
    assert "seed_az_deg" not in free
    assert "seed_el_deg" not in free

    seeded = CalibrationTargetSpec.platesolve(
        "Polaris region", seed_az_deg=359.0, seed_el_deg=33.5
    ).to_dict()
    assert seeded["seed_az_deg"] == pytest.approx(359.0)
    assert seeded["seed_el_deg"] == pytest.approx(33.5)


# ---------- TargetKind enum + str round-trip -------------------------


def test_targetkind_value_strings_match_wire_format():
    assert TargetKind.FAA.value == "faa"
    assert TargetKind.CELESTIAL.value == "celestial"
    assert TargetKind.PLATESOLVE.value == "platesolve"


def test_targetkind_constructor_accepts_string_values():
    """The REST resource constructs ``TargetKind(kind_str)`` from the
    JSON payload — it should accept the wire strings directly."""
    assert TargetKind("faa") is TargetKind.FAA
    assert TargetKind("celestial") is TargetKind.CELESTIAL
    assert TargetKind("platesolve") is TargetKind.PLATESOLVE
    with pytest.raises(ValueError):
        TargetKind("nonsense")
