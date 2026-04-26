"""Celestial calibration-target catalog + altaz computation.

Parallel to :mod:`scripts.trajectory.faa_dof` but for nighttime
calibration. Provides a curated list of bright stars + a planet
ephemeris so the calibrate-rotation page can offer a nighttime picker
that mirrors the daytime FAA DOF picker: filter by horizon visibility,
sort by proximity to the current pointing, surface a top-N list.

Design choices:

- The bright-star catalog is **hand-curated** (~30 entries), not pulled
  from Hipparcos / Yale BSC. The list is deliberately small so an
  operator scrolling it on a touch screen sees only candidates that are
  bright enough for a single-frame centroid lock and well-spread across
  declination. Each entry is a Python literal — no runtime data fetch.
- Planet positions come from :mod:`ephem` (already a project dep —
  ``device/sun_safety.py`` uses it for the sun). Apparent magnitude is
  computed at runtime since it varies by phase / elongation.
- Altaz computation goes through ``ephem.FixedBody`` for stars and
  ``ephem.<Planet>`` for planets, so the math path is one library —
  consistent with ``sun_safety.compute_sun_altaz``. Refraction is
  ephem's default (Saemundsson model at standard temperature/pressure),
  which is fine for calibration-grade pointing where we want apparent
  altitude.
- All angles in **degrees**. Az is measured east of north in [0, 360).
  El (altitude) is in [-90, 90].

Catalog provenance: the magnitudes and ICRS J2000 positions are taken
from each star's Wikipedia infobox / SIMBAD lookup. Sub-arcsecond
precision is overkill for a calibration aim point — 4 decimals on
``ra_hours`` (15 arcsec) and ``dec_deg`` (4 arcsec) are well within the
mount's pointing tolerance. If a star's catalog position is off by a
few tens of arcsec, the centroid lock still works.

The module is dependency-light (just ``ephem`` + the in-repo
``ObserverSite``). Importable in test contexts without the full Alpaca
stack.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable

import ephem

from scripts.trajectory.observer import ObserverSite


# ---------- data model -----------------------------------------------


@dataclass(frozen=True)
class CelestialTarget:
    """A calibration-grade celestial target: bright star, planet, or
    well-known double star. Used by the calibrate page's nighttime
    picker.

    ``ra_hours`` / ``dec_deg`` are ICRS J2000 for stars and doubles. For
    planets they're populated at runtime by :func:`planet_target` from
    the apparent-place computation (epoch-of-date) and are ``None`` on
    the static catalog entry.
    """

    name: str
    kind: str  # "star" | "planet" | "double"
    ra_hours: float | None
    dec_deg: float | None
    vmag: float
    bayer: str = ""
    notes: str = ""


# ---------- planet list ----------------------------------------------


# Planets we expose for nighttime calibration. Notable exclusions:
#   - Moon: the moon itself is not a calibration target (extended,
#     scattered light), and we filter out targets too close to it.
#   - Sun: handled by sun_safety; calibration must avoid the sun cone.
#   - Uranus / Neptune: vmag ~5.7 / 7.8, too dim for a fast centroid
#     lock at the typical calibration exposure.
PLANETS: tuple[str, ...] = (
    "Mercury",
    "Venus",
    "Mars",
    "Jupiter",
    "Saturn",
)


# ---------- curated bright-star catalog ------------------------------


# Hand-curated list of calibration-grade bright stars. See the module
# docstring for the rationale and provenance. Magnitudes are mean V
# (apparent visual). Well-known doubles get ``kind="double"`` so the UI
# can flag them — the position is the brighter component.
#
# Coverage is biased toward northern declinations (most users observe
# from the northern hemisphere) but includes a few mid-southern entries
# for SoCal / lower-latitude observers.
BRIGHT_STARS: tuple[CelestialTarget, ...] = (
    # Polaris — the obvious north-pole reference.
    CelestialTarget(
        name="Polaris",
        kind="star",
        ra_hours=2.5303,
        dec_deg=89.2641,
        vmag=1.98,
        bayer="alpha UMi",
        notes="Circumpolar at most northern latitudes; el ≈ observer lat.",
    ),
    # Alpha-class (mag < 1.5) — well-distributed across the sky.
    CelestialTarget(
        name="Sirius",
        kind="star",
        ra_hours=6.7525,
        dec_deg=-16.7161,
        vmag=-1.46,
        bayer="alpha CMa",
        notes="Brightest star; winter sky for northern observers.",
    ),
    CelestialTarget(
        name="Canopus",
        kind="star",
        ra_hours=6.3992,
        dec_deg=-52.6957,
        vmag=-0.74,
        bayer="alpha Car",
        notes="Mid-southern decl; only briefly above horizon at lat<37N.",
    ),
    CelestialTarget(
        name="Arcturus",
        kind="star",
        ra_hours=14.2610,
        dec_deg=19.1825,
        vmag=-0.05,
        bayer="alpha Boo",
        notes="Spring/summer; orange giant.",
    ),
    CelestialTarget(
        name="Vega",
        kind="star",
        ra_hours=18.6157,
        dec_deg=38.7837,
        vmag=0.03,
        bayer="alpha Lyr",
        notes="Summer Triangle; reference standard for vmag=0.",
    ),
    CelestialTarget(
        name="Capella",
        kind="star",
        ra_hours=5.2782,
        dec_deg=45.9980,
        vmag=0.08,
        bayer="alpha Aur",
        notes="Circumpolar at lat>44N.",
    ),
    CelestialTarget(
        name="Rigel",
        kind="star",
        ra_hours=5.2423,
        dec_deg=-8.2017,
        vmag=0.13,
        bayer="beta Ori",
        notes="Orion; bright blue-white supergiant.",
    ),
    CelestialTarget(
        name="Procyon",
        kind="star",
        ra_hours=7.6553,
        dec_deg=5.2250,
        vmag=0.34,
        bayer="alpha CMi",
        notes="Winter sky; near Sirius + Betelgeuse.",
    ),
    CelestialTarget(
        name="Achernar",
        kind="star",
        ra_hours=1.6286,
        dec_deg=-57.2367,
        vmag=0.46,
        bayer="alpha Eri",
        notes="Far-southern; below horizon for most US observers.",
    ),
    CelestialTarget(
        name="Betelgeuse",
        kind="star",
        ra_hours=5.9195,
        dec_deg=7.4070,
        vmag=0.42,
        bayer="alpha Ori",
        notes="Variable (~0.0–1.3); use only when above mag 1.0.",
    ),
    CelestialTarget(
        name="Altair",
        kind="star",
        ra_hours=19.8464,
        dec_deg=8.8683,
        vmag=0.77,
        bayer="alpha Aql",
        notes="Summer Triangle.",
    ),
    CelestialTarget(
        name="Aldebaran",
        kind="star",
        ra_hours=4.5987,
        dec_deg=16.5092,
        vmag=0.85,
        bayer="alpha Tau",
        notes="Eye of Taurus; orange giant.",
    ),
    CelestialTarget(
        name="Spica",
        kind="star",
        ra_hours=13.4199,
        dec_deg=-11.1614,
        vmag=0.97,
        bayer="alpha Vir",
        notes="Spring; close to ecliptic.",
    ),
    CelestialTarget(
        name="Antares",
        kind="star",
        ra_hours=16.4901,
        dec_deg=-26.4320,
        vmag=1.09,
        bayer="alpha Sco",
        notes="Heart of Scorpius; red supergiant, slightly variable.",
    ),
    CelestialTarget(
        name="Pollux",
        kind="star",
        ra_hours=7.7553,
        dec_deg=28.0262,
        vmag=1.14,
        bayer="beta Gem",
        notes="Twin of Castor; orange giant.",
    ),
    CelestialTarget(
        name="Fomalhaut",
        kind="star",
        ra_hours=22.9608,
        dec_deg=-29.6222,
        vmag=1.16,
        bayer="alpha PsA",
        notes="Autumn; only star of mag<2 in that southern sky region.",
    ),
    CelestialTarget(
        name="Deneb",
        kind="star",
        ra_hours=20.6905,
        dec_deg=45.2803,
        vmag=1.25,
        bayer="alpha Cyg",
        notes="Summer Triangle apex; far supergiant.",
    ),
    CelestialTarget(
        name="Regulus",
        kind="star",
        ra_hours=10.1395,
        dec_deg=11.9672,
        vmag=1.35,
        bayer="alpha Leo",
        notes="Spring; close to ecliptic, occasional planet conjunction.",
    ),
    CelestialTarget(
        name="Adhara",
        kind="star",
        ra_hours=6.9770,
        dec_deg=-28.9722,
        vmag=1.50,
        bayer="epsilon CMa",
        notes="Below Sirius; bright southern winter star.",
    ),
    CelestialTarget(
        name="Castor",
        kind="star",
        ra_hours=7.5766,
        dec_deg=31.8883,
        vmag=1.58,
        bayer="alpha Gem",
        notes="Sextuple system; centroid is on the bright A+B pair.",
    ),
    CelestialTarget(
        name="Bellatrix",
        kind="star",
        ra_hours=5.4188,
        dec_deg=6.3497,
        vmag=1.64,
        bayer="gamma Ori",
        notes="Right shoulder of Orion.",
    ),
    CelestialTarget(
        name="Elnath",
        kind="star",
        ra_hours=5.4382,
        dec_deg=28.6075,
        vmag=1.65,
        bayer="beta Tau",
        notes="Tip of Taurus's northern horn.",
    ),
    CelestialTarget(
        name="Alnilam",
        kind="star",
        ra_hours=5.6036,
        dec_deg=-1.2019,
        vmag=1.69,
        bayer="epsilon Ori",
        notes="Middle star of Orion's belt.",
    ),
    CelestialTarget(
        name="Alnitak",
        kind="star",
        ra_hours=5.6793,
        dec_deg=-1.9426,
        vmag=1.74,
        bayer="zeta Ori",
        notes="East star of Orion's belt; triple system.",
    ),
    CelestialTarget(
        name="Dubhe",
        kind="star",
        ra_hours=11.0621,
        dec_deg=61.7510,
        vmag=1.79,
        bayer="alpha UMa",
        notes="Lip of the Big Dipper; a pointer to Polaris.",
    ),
    CelestialTarget(
        name="Mirfak",
        kind="star",
        ra_hours=3.4054,
        dec_deg=49.8612,
        vmag=1.79,
        bayer="alpha Per",
        notes="Brightest in Perseus; circumpolar at lat>40N.",
    ),
    CelestialTarget(
        name="Alkaid",
        kind="star",
        ra_hours=13.7923,
        dec_deg=49.3133,
        vmag=1.85,
        bayer="eta UMa",
        notes="End of the Big Dipper handle.",
    ),
    CelestialTarget(
        name="Mintaka",
        kind="star",
        ra_hours=5.5334,
        dec_deg=-0.2991,
        vmag=2.23,
        bayer="delta Ori",
        notes="West star of Orion's belt; sits ≈ on the celestial equator.",
    ),
    # Doubles — useful for verifying focus / centroid behaviour.
    CelestialTarget(
        name="Mizar+Alcor",
        kind="double",
        ra_hours=13.3990,
        dec_deg=54.9254,
        vmag=2.04,  # Mizar A
        bayer="zeta UMa",
        notes="Naked-eye double 11.8' from Alcor (mag 4.0).",
    ),
    CelestialTarget(
        name="Albireo",
        kind="double",
        ra_hours=19.5120,
        dec_deg=27.9597,
        vmag=3.05,  # Albireo A (yellow)
        bayer="beta Cyg",
        notes='Beautiful blue/orange pair, separation 35".',
    ),
)


# ---------- ephem helpers --------------------------------------------


def _make_observer(site: ObserverSite, when_utc: datetime) -> ephem.Observer:
    obs = ephem.Observer()
    obs.lat = str(site.lat_deg)
    obs.lon = str(site.lon_deg)
    obs.elevation = float(site.alt_m)
    if when_utc.tzinfo is None:
        when_utc = when_utc.replace(tzinfo=timezone.utc)
    # ephem expects a naive UTC datetime.
    obs.date = when_utc.astimezone(timezone.utc).replace(tzinfo=None)
    return obs


def _star_body(target: CelestialTarget) -> ephem.FixedBody:
    body = ephem.FixedBody()
    if target.ra_hours is None or target.dec_deg is None:
        raise ValueError(
            f"target {target.name!r} (kind={target.kind!r}) "
            "has no static RA/Dec — use planet_target() for ephemerides"
        )
    # ephem.hours / ephem.degrees take string arguments in DMS or decimal.
    body._ra = ephem.hours(str(target.ra_hours))
    body._dec = ephem.degrees(str(target.dec_deg))
    body._epoch = ephem.J2000
    body.name = target.name
    return body


def _planet_body(name: str) -> ephem.PlanetMoon:
    """Return an uninitialised ephem planet body by capitalised name.

    Raises ValueError if ``name`` isn't in :data:`PLANETS`. Kept tight
    so a typo in the API parameter can't accidentally summon ``Sun`` or
    ``Moon`` (which have their own safety-cone treatment).
    """
    if name not in PLANETS:
        raise ValueError(f"unknown planet {name!r}; expected one of {PLANETS}")
    return getattr(ephem, name)()


# ---------- core API -------------------------------------------------


def compute_altaz(
    target: CelestialTarget,
    site: ObserverSite,
    when_utc: datetime,
) -> tuple[float, float]:
    """Return ``(az_deg, el_deg)`` for ``target`` from ``site`` at
    ``when_utc``.

    Az in [0, 360) measured east of north; el in [-90, 90]. Refraction
    is ephem's default (Saemundsson at standard atmosphere) so the
    elevation is *apparent* — what the operator sees through the eyepiece.

    Raises ``ValueError`` if ``target.kind`` is unrecognised.
    """
    obs = _make_observer(site, when_utc)
    if target.kind in ("star", "double"):
        body = _star_body(target)
    elif target.kind == "planet":
        body = _planet_body(target.name)
    else:
        raise ValueError(f"unrecognised target kind: {target.kind!r}")
    body.compute(obs)
    az = math.degrees(float(body.az)) % 360.0
    el = math.degrees(float(body.alt))
    return az, el


def planet_target(
    name: str,
    when_utc: datetime,
    site: ObserverSite,
) -> CelestialTarget:
    """Compute a :class:`CelestialTarget` for a planet at ``when_utc``.

    ``ra_hours`` / ``dec_deg`` come from the apparent-place computation
    (epoch-of-date), so the values are useful for slew commands but not
    interchangeable with the ICRS J2000 entries in
    :data:`BRIGHT_STARS`. ``vmag`` reflects the current phase /
    elongation per ephem's built-in magnitude model.
    """
    body = _planet_body(name)
    obs = _make_observer(site, when_utc)
    body.compute(obs)
    ra_rad = float(body.ra)
    dec_rad = float(body.dec)
    return CelestialTarget(
        name=name,
        kind="planet",
        ra_hours=math.degrees(ra_rad) / 15.0,
        dec_deg=math.degrees(dec_rad),
        vmag=float(body.mag),
        bayer="",
        notes="apparent place (epoch-of-date)",
    )


# ---------- geometry helpers -----------------------------------------


def angular_distance_deg(
    az1_deg: float,
    el1_deg: float,
    az2_deg: float,
    el2_deg: float,
) -> float:
    """Spherical great-circle distance between two altaz directions.

    Pure function — no observer, no time. Az is treated as longitude,
    el as latitude on the celestial sphere. Returns degrees in [0, 180].
    """
    a_az = math.radians(az1_deg)
    a_el = math.radians(el1_deg)
    b_az = math.radians(az2_deg)
    b_el = math.radians(el2_deg)
    cos_sep = math.sin(a_el) * math.sin(b_el) + math.cos(a_el) * math.cos(
        b_el
    ) * math.cos(a_az - b_az)
    cos_sep = max(-1.0, min(1.0, cos_sep))
    return math.degrees(math.acos(cos_sep))


def sun_separation_deg(
    target_az_deg: float,
    target_el_deg: float,
    site: ObserverSite,
    when_utc: datetime,
) -> float:
    """Great-circle separation between the target and the sun (deg).

    Uses ephem directly to keep the dependency surface tight. The
    sun_safety module's ``compute_sun_altaz`` would also work but that
    pulls in :mod:`device.config` for its lat/lon fallback path, which
    we want to avoid in pure-trajectory code.
    """
    obs = _make_observer(site, when_utc)
    sun = ephem.Sun()
    sun.compute(obs)
    sun_az = math.degrees(float(sun.az)) % 360.0
    sun_alt = math.degrees(float(sun.alt))
    return angular_distance_deg(target_az_deg, target_el_deg, sun_az, sun_alt)


def moon_separation_deg(
    target_az_deg: float,
    target_el_deg: float,
    site: ObserverSite,
    when_utc: datetime,
) -> float:
    """Great-circle separation between the target and the moon (deg).

    Filters out targets too close to the moon — scattered moonlight
    swamps the centroid for a calibration-grade lock even when both are
    above horizon.
    """
    obs = _make_observer(site, when_utc)
    moon = ephem.Moon()
    moon.compute(obs)
    moon_az = math.degrees(float(moon.az)) % 360.0
    moon_alt = math.degrees(float(moon.alt))
    return angular_distance_deg(
        target_az_deg,
        target_el_deg,
        moon_az,
        moon_alt,
    )


# ---------- visibility filter ----------------------------------------


def filter_visible(
    targets: Iterable[CelestialTarget],
    site: ObserverSite,
    when_utc: datetime,
    *,
    min_el_deg: float = 20.0,
    max_mag: float = 4.0,
    sun_min_sep_deg: float = 30.0,
    moon_min_sep_deg: float = 5.0,
) -> list[tuple[CelestialTarget, float, float]]:
    """Return [(target, az_deg, el_deg), ...] for the visible subset.

    A target is **visible** if ``el >= min_el_deg`` (above horizon with
    margin), ``vmag <= max_mag`` (bright enough to centroid), at least
    ``sun_min_sep_deg`` from the sun (daytime calibration must skip),
    and at least ``moon_min_sep_deg`` from the moon (avoid scattered
    moonlight washing out the centroid).

    Order is **unsorted** — the caller picks the sort key
    (proximity vs magnitude). The list is the filtered subset only.
    """
    obs = _make_observer(site, when_utc)
    sun = ephem.Sun()
    sun.compute(obs)
    sun_az = math.degrees(float(sun.az)) % 360.0
    sun_alt = math.degrees(float(sun.alt))
    moon = ephem.Moon()
    moon.compute(obs)
    moon_az = math.degrees(float(moon.az)) % 360.0
    moon_alt = math.degrees(float(moon.alt))

    out: list[tuple[CelestialTarget, float, float]] = []
    for target in targets:
        if target.vmag > max_mag:
            continue
        try:
            az, el = compute_altaz(target, site, when_utc)
        except ValueError:
            # Static catalog entry without RA/Dec — skip.
            continue
        if el < min_el_deg:
            continue
        if angular_distance_deg(az, el, sun_az, sun_alt) < sun_min_sep_deg:
            continue
        if angular_distance_deg(az, el, moon_az, moon_alt) < moon_min_sep_deg:
            continue
        out.append((target, az, el))
    return out


# ---------- ranking --------------------------------------------------


def sort_by_proximity(
    visible: list[tuple[CelestialTarget, float, float]],
    current_az_deg: float,
    current_el_deg: float,
) -> list[tuple[CelestialTarget, float, float, float]]:
    """Rank ``visible`` by ascending great-circle distance from
    ``(current_az_deg, current_el_deg)``.

    Each output tuple is ``(target, az_deg, el_deg, distance_deg)``.
    Magnitude is the secondary key so two equally-near targets prefer
    the brighter (smaller vmag).
    """
    decorated: list[tuple[CelestialTarget, float, float, float]] = [
        (
            t,
            az,
            el,
            angular_distance_deg(az, el, current_az_deg, current_el_deg),
        )
        for (t, az, el) in visible
    ]
    decorated.sort(key=lambda r: (r[3], r[0].vmag))
    return decorated


def sort_by_magnitude(
    visible: list[tuple[CelestialTarget, float, float]],
) -> list[tuple[CelestialTarget, float, float]]:
    """Sort ``visible`` ascending by ``vmag`` (brightest first).

    Used when no current pointing is supplied and the picker should
    surface the showy candidates first.
    """
    out = list(visible)
    out.sort(key=lambda r: r[0].vmag)
    return out


def all_targets(
    when_utc: datetime,
    site: ObserverSite,
) -> list[CelestialTarget]:
    """Return the full pool of targets for the picker: every star in
    the curated catalog plus a freshly-computed entry for each planet.

    Planets that ephem can't compute (e.g. a corrupt mid-month run-time
    state) are silently skipped — the static stars still come through.
    """
    out: list[CelestialTarget] = list(BRIGHT_STARS)
    for name in PLANETS:
        try:
            out.append(planet_target(name, when_utc, site))
        except Exception:
            continue
    return out


__all__ = [
    "BRIGHT_STARS",
    "PLANETS",
    "CelestialTarget",
    "all_targets",
    "angular_distance_deg",
    "compute_altaz",
    "filter_visible",
    "moon_separation_deg",
    "planet_target",
    "sort_by_magnitude",
    "sort_by_proximity",
    "sun_separation_deg",
]
