"""Unified calibration-target abstraction.

The calibrate-rotation page can now mix three classes of target in a
single calibration session:

- **FAA DOF landmarks** (terrestrial, daytime). Truth comes from the
  landmark's published position + the observer site, time-independent.
- **Celestial targets** (bright stars + planets). Truth comes from
  ephem at sighting time.
- **Plate-solve aim points** (operator-chosen sky areas). Truth comes
  from a plate-solver run on a captured image, available only after
  the operator clicks "sight".

This module defines the tagged-union type the session, REST resources,
and UI all share, plus a resolver that turns a spec + (site, time,
plate-solve) into the topocentric (az, el) the fit consumes.

The dataclass is deliberately frozen + JSON-friendly so the same shape
flows from picker → /calibration/start → in-process session →
mount_calibration.json → audit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from scripts.trajectory.faa_dof import Landmark
from scripts.trajectory.observer import ObserverSite


class TargetKind(str, Enum):
    """Discriminator for :class:`CalibrationTargetSpec`. The string
    values are the wire format used by the REST resources and the
    on-disk audit JSON; do not rename without a payload migration."""

    FAA = "faa"
    CELESTIAL = "celestial"
    PLATESOLVE = "platesolve"


# ---------- 1σ uncertainties used when the spec doesn't carry one ----
#
# The fit is unweighted; these are display-only / per-sighting metadata
# so the UI can show "your residual is within noise" tooltips. Numbers
# are deliberately conservative.
#
# - FAA: handled by ``pointing_uncertainty_deg`` from ECEF + slant +
#   FAA accuracy class — much smaller than the ~0.05° operator centring
#   error for nearby landmarks. The defaults below apply only when a
#   FAA target's accuracy class is missing/malformed.
# - CELESTIAL: pure-ephemeris pointing is sub-arcsecond; the bottleneck
#   is the operator centring the bright star in the eyepiece — quote
#   that as the dominant noise source.
# - PLATESOLVE: the WCS solve typically reports its own RMS in
#   arcseconds. When the runtime result doesn't carry one, fall back
#   to a conservative 0.01° (~36").
DEFAULT_OPERATOR_CENTERING_SIGMA_DEG: float = 0.05
DEFAULT_PLATESOLVE_SIGMA_DEG: float = 0.01


# ---------- plate-solve handoff --------------------------------------


@dataclass(frozen=True)
class PlateSolveOutcome:
    """The handful of fields :meth:`CalibrationTargetSpec.resolve_true_altaz`
    needs from a plate-solve run.

    Decoupled from ``device.plate_solver.SolveResult`` so this module
    stays trajectory-pure (no ``subprocess`` / catalog imports at
    module load time). The session converts a real ``SolveResult`` to
    this shape after invoking the solver.
    """

    true_az_deg: float
    true_el_deg: float
    sigma_deg: float | None = None
    ra_deg: float | None = None
    dec_deg: float | None = None


# ---------- spec -----------------------------------------------------


@dataclass(frozen=True)
class CalibrationTargetSpec:
    """Operator-selected calibration target.

    The session does not care what kind of target it is until sighting
    time; it just slews to a seed (az, el), lets the operator nudge,
    then asks the spec to resolve the truth.

    Field semantics by ``kind``:

    - ``FAA`` — ``landmark`` is required. ``slant_m`` is cached at
      session start so the per-sighting metadata is consistent even if
      ``site`` drifts.
    - ``CELESTIAL`` — ``ra_hours`` and ``dec_deg`` are required (ICRS
      J2000 for stars; epoch-of-date for planets). ``vmag`` and
      ``bayer`` are nice-to-have for the UI banner.
    - ``PLATESOLVE`` — only ``label`` is required. ``seed_az_deg`` /
      ``seed_el_deg`` are optional initial slew targets; without them,
      the session leaves the mount where the operator points it
      (free-aim).
    """

    kind: TargetKind
    label: str

    # FAA-specific.
    landmark: Landmark | None = None
    slant_m: float | None = None

    # Celestial-specific.
    ra_hours: float | None = None
    dec_deg: float | None = None
    vmag: float | None = None
    bayer: str | None = None

    # Plate-solve-specific. Truth comes from the WCS at sighting time;
    # these are only the seed slew (optional).
    seed_az_deg: float | None = None
    seed_el_deg: float | None = None

    # ---------- factories ----------

    @classmethod
    def from_landmark(
        cls,
        landmark: Landmark,
        slant_m: float | None = None,
    ) -> "CalibrationTargetSpec":
        return cls(
            kind=TargetKind.FAA,
            label=f"{landmark.oas} {landmark.name}",
            landmark=landmark,
            slant_m=slant_m,
        )

    @classmethod
    def celestial(
        cls,
        name: str,
        ra_hours: float,
        dec_deg: float,
        *,
        vmag: float | None = None,
        bayer: str | None = None,
    ) -> "CalibrationTargetSpec":
        return cls(
            kind=TargetKind.CELESTIAL,
            label=name,
            ra_hours=float(ra_hours),
            dec_deg=float(dec_deg),
            vmag=float(vmag) if vmag is not None else None,
            bayer=bayer,
        )

    @classmethod
    def platesolve(
        cls,
        label: str,
        *,
        seed_az_deg: float | None = None,
        seed_el_deg: float | None = None,
    ) -> "CalibrationTargetSpec":
        return cls(
            kind=TargetKind.PLATESOLVE,
            label=str(label),
            seed_az_deg=(float(seed_az_deg) if seed_az_deg is not None else None),
            seed_el_deg=(float(seed_el_deg) if seed_el_deg is not None else None),
        )

    # ---------- truth resolution ----------

    def resolve_true_altaz(
        self,
        site: ObserverSite,
        when_utc: datetime,
        plate_solve_result: PlateSolveOutcome | None = None,
    ) -> tuple[float, float, float | None]:
        """Return ``(true_az_deg, true_el_deg, slant_m_or_none)``.

        - FAA: deterministic. Uses the landmark's ECEF + the local ENU
          rotation, with terrestrial-refraction correction baked in so
          ``true_el`` matches the apparent altitude the operator sees.
          Time-independent; ``when_utc`` is ignored.
        - CELESTIAL: depends on ``when_utc`` (Earth rotation). Refraction
          is ephem's default (Saemundsson at standard atmosphere) so
          ``true_el`` is the apparent altitude.
        - PLATESOLVE: pulls ``(az, el)`` from ``plate_solve_result``;
          raises ``ValueError`` when the result is missing.
        """
        if self.kind == TargetKind.FAA:
            return self._resolve_faa(site)
        if self.kind == TargetKind.CELESTIAL:
            return self._resolve_celestial(site, when_utc)
        if self.kind == TargetKind.PLATESOLVE:
            return self._resolve_platesolve(plate_solve_result)
        raise ValueError(f"unknown TargetKind: {self.kind!r}")

    def _resolve_faa(self, site: ObserverSite) -> tuple[float, float, float | None]:
        if self.landmark is None:
            raise ValueError("FAA target spec missing landmark")
        # Lazy imports keep this module free of optional-dep + heavy
        # path costs at import time.
        from device.rotation_calibration import terrestrial_refraction_deg
        from device.target_frame import MountFrame

        mf = MountFrame.from_identity_enu(site)
        az, el, slant = mf.ecef_to_mount_azel(self.landmark.ecef())
        # Add the terrestrial-refraction lift so the truth matches the
        # apparent altitude the operator nudges to.
        el_app = float(el) + terrestrial_refraction_deg(float(slant))
        # Wrap az to the same [-180, 180) convention the encoder uses.
        az_wrapped = ((float(az) + 180.0) % 360.0) - 180.0
        return az_wrapped, el_app, float(slant)

    def _resolve_celestial(
        self, site: ObserverSite, when_utc: datetime
    ) -> tuple[float, float, float | None]:
        if self.ra_hours is None or self.dec_deg is None:
            raise ValueError(f"celestial target {self.label!r} missing RA/Dec")
        if when_utc.tzinfo is None:
            when_utc = when_utc.replace(tzinfo=timezone.utc)
        # Planets drift quickly in (RA, Dec). When the spec carries a
        # planet name, recompute the apparent place at sighting time so
        # the truth is current to within seconds. The session's pickers
        # populate (ra_hours, dec_deg) at session start, but a slow
        # operator could be 10 min behind by sighting — Mars moves
        # ~0.005°/min, which compounds with operator centring noise.
        from scripts.trajectory.celestial_targets import (
            BRIGHT_STARS,
            PLANETS,
            CelestialTarget,
            compute_altaz,
            planet_target,
        )

        if self.label in PLANETS:
            ct = planet_target(self.label, when_utc, site)
        else:
            # Doubles get the same star path; the picker resolves the
            # bright component's position so the centroid lock works.
            ct = CelestialTarget(
                name=self.label,
                kind="star",
                ra_hours=float(self.ra_hours),
                dec_deg=float(self.dec_deg),
                vmag=(
                    float(self.vmag)
                    if self.vmag is not None
                    else _bright_star_vmag(self.label, BRIGHT_STARS)
                ),
                bayer=self.bayer or "",
            )
        az, el = compute_altaz(ct, site, when_utc)
        # Match FAA's [-180, 180) az convention.
        az_wrapped = ((float(az) + 180.0) % 360.0) - 180.0
        return az_wrapped, float(el), None

    def _resolve_platesolve(
        self, plate_solve_result: PlateSolveOutcome | None
    ) -> tuple[float, float, float | None]:
        if plate_solve_result is None:
            raise ValueError("platesolve target requires a plate-solve result")
        az = float(plate_solve_result.true_az_deg)
        el = float(plate_solve_result.true_el_deg)
        az_wrapped = ((az + 180.0) % 360.0) - 180.0
        return az_wrapped, el, None

    # ---------- per-spec uncertainty ----------

    def sigma_az_el_deg(self) -> tuple[float | None, float | None]:
        """Return ``(sigma_az, sigma_el)`` 1σ in degrees, or ``(None,
        None)`` if the spec doesn't have enough info to compute one.

        FAA combines the published landmark accuracy with the observer
        GPS to get a per-axis pointing uncertainty (handled by
        :func:`device.rotation_calibration.pointing_uncertainty_deg`).
        Celestial / plate-solve sigmas come from the operator centring
        / WCS RMS respectively — not really a function of the spec
        itself, so we surface a default the session can override.
        """
        if self.kind == TargetKind.FAA:
            if self.landmark is None or self.slant_m is None:
                return (None, None)
            from device.rotation_calibration import pointing_uncertainty_deg
            from scripts.trajectory.faa_dof import faa_accuracy_ft

            h_ft, v_ft = faa_accuracy_ft(self.landmark.accuracy_class)
            saz, sel = pointing_uncertainty_deg(float(self.slant_m), h_ft, v_ft)
            saz_ok = saz if math.isfinite(saz) else None
            sel_ok = sel if math.isfinite(sel) else None
            return (saz_ok, sel_ok)
        if self.kind == TargetKind.CELESTIAL:
            return (
                DEFAULT_OPERATOR_CENTERING_SIGMA_DEG,
                DEFAULT_OPERATOR_CENTERING_SIGMA_DEG,
            )
        if self.kind == TargetKind.PLATESOLVE:
            return (DEFAULT_PLATESOLVE_SIGMA_DEG, DEFAULT_PLATESOLVE_SIGMA_DEG)
        return (None, None)

    # ---------- JSON helpers ----------

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dict suitable for the status response and
        the on-disk audit blob. Every ``kind`` writes its discriminator
        and label; kind-specific fields are included when present."""
        out: dict[str, Any] = {"kind": self.kind.value, "label": self.label}
        if self.kind == TargetKind.FAA and self.landmark is not None:
            out["oas"] = self.landmark.oas
            out["name"] = self.landmark.name
            out["lat_deg"] = self.landmark.lat_deg
            out["lon_deg"] = self.landmark.lon_deg
            out["height_amsl_m"] = self.landmark.height_amsl_m
            out["accuracy_class"] = self.landmark.accuracy_class
            out["lit"] = bool(self.landmark.lit)
            if self.slant_m is not None:
                out["slant_m"] = float(self.slant_m)
        if self.kind == TargetKind.CELESTIAL:
            if self.ra_hours is not None:
                out["ra_hours"] = float(self.ra_hours)
            if self.dec_deg is not None:
                out["dec_deg"] = float(self.dec_deg)
            if self.vmag is not None:
                out["vmag"] = float(self.vmag)
            if self.bayer:
                out["bayer"] = self.bayer
        if self.kind == TargetKind.PLATESOLVE:
            if self.seed_az_deg is not None:
                out["seed_az_deg"] = float(self.seed_az_deg)
            if self.seed_el_deg is not None:
                out["seed_el_deg"] = float(self.seed_el_deg)
        return out


# ---------- helpers ---------------------------------------------------


def _bright_star_vmag(name: str, catalog) -> float:
    """Return the catalog vmag for ``name`` if it's a bright-star
    entry; else fall back to a placeholder. Used only to satisfy
    :class:`CelestialTarget`'s required ``vmag`` field when the spec
    didn't carry one — magnitude isn't load-bearing for the altaz
    computation, but the dataclass requires a float."""
    for ct in catalog:
        if ct.name == name:
            return float(ct.vmag)
    return 0.0


__all__ = [
    "CalibrationTargetSpec",
    "DEFAULT_OPERATOR_CENTERING_SIGMA_DEG",
    "DEFAULT_PLATESOLVE_SIGMA_DEG",
    "PlateSolveOutcome",
    "TargetKind",
]
