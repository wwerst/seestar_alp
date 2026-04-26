"""Reusable rotation-calibration science and session plumbing.

Shared by the REPL CLI (`scripts/trajectory/calibrate_rotation.py`)
and the web front end. The CLI keeps only its `input()` glue + menu
rendering; everything below is I/O-free enough to test without a
mount or a network.

Exposed API:

- Dataclasses: :class:`Sighting`, :class:`RotationSolution`,
  :class:`PriorInfo`, :class:`CalibrationStatus`.
- Constants: :data:`KEEP_MAX_AGE_S`, :data:`KEEP_MAX_DISTANCE_M` — the
  two thresholds behind the "clear or keep" heuristic.
- Pure helpers:
    - :func:`terrestrial_refraction_deg`
    - :func:`predict_mount_azel`
    - :func:`solve_rotation`, :func:`write_calibration`
    - :func:`parse_calibrated_at`, :func:`inspect_prior`,
      :func:`decide_clear_or_keep`
- Session:
    - :class:`CalibrationSession` — thread-based mount driver for the
      browser calibration UI, modelled on `LiveTrackSession`.
    - :class:`CalibrationManager` — per-process singleton,
      telescope-keyed. Cross-checks with the live-tracker manager so
      the two flows can't drive the mount concurrently.
"""

from __future__ import annotations

import json
import math
import queue
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from astropy.coordinates import EarthLocation
from scipy.optimize import least_squares

from device.calibration_targets import (
    CalibrationTargetSpec,
    PlateSolveOutcome,
    TargetKind,
)
from device.target_frame import MountFrame
from scripts.trajectory.faa_dof import Landmark
from scripts.trajectory.observer import ObserverSite, haversine_m


# ---------- data model ----------------------------------------------


class Sighting:
    """One target → encoder (az, el) record.

    Carries a :class:`CalibrationTargetSpec` (FAA / celestial /
    platesolve) plus the operator-aligned encoder reading and the
    truth resolved at sighting time. ``slant_m`` is populated only for
    FAA targets; ``sigma_*`` carry per-sighting 1σ display metadata.

    Accepts either ``target=`` (new tagged-union API) or
    ``landmark=`` (legacy FAA-only API). The latter wraps the landmark
    into an FAA-kind spec so existing callers + tests keep working
    without code changes.
    """

    __slots__ = (
        "target",
        "encoder_az_deg",
        "encoder_el_deg",
        "true_az_deg",
        "true_el_deg",
        "slant_m",
        "t_unix",
        "sigma_az_deg",
        "sigma_el_deg",
    )

    def __init__(
        self,
        *,
        encoder_az_deg: float,
        encoder_el_deg: float,
        true_az_deg: float,
        true_el_deg: float,
        slant_m: float | None,
        t_unix: float,
        target: CalibrationTargetSpec | None = None,
        landmark: Landmark | None = None,
        sigma_az_deg: float | None = None,
        sigma_el_deg: float | None = None,
    ) -> None:
        if target is not None and landmark is not None:
            raise ValueError("Sighting accepts target= OR landmark=, not both")
        if target is None and landmark is None:
            raise ValueError("Sighting requires target= or landmark=")
        if target is None:
            target = CalibrationTargetSpec.from_landmark(
                landmark,
                slant_m=float(slant_m) if slant_m is not None else None,
            )
        # Use object.__setattr__ to support __slots__ without dataclass.
        object.__setattr__(self, "target", target)
        object.__setattr__(self, "encoder_az_deg", float(encoder_az_deg))
        object.__setattr__(self, "encoder_el_deg", float(encoder_el_deg))
        object.__setattr__(self, "true_az_deg", float(true_az_deg))
        object.__setattr__(self, "true_el_deg", float(true_el_deg))
        object.__setattr__(
            self,
            "slant_m",
            float(slant_m) if slant_m is not None else None,
        )
        object.__setattr__(self, "t_unix", float(t_unix))
        object.__setattr__(
            self,
            "sigma_az_deg",
            float(sigma_az_deg) if sigma_az_deg is not None else None,
        )
        object.__setattr__(
            self,
            "sigma_el_deg",
            float(sigma_el_deg) if sigma_el_deg is not None else None,
        )

    def __repr__(self) -> str:  # pragma: no cover — debug-only
        return (
            f"Sighting(target={self.target!r}, "
            f"encoder=({self.encoder_az_deg:.3f}, {self.encoder_el_deg:.3f}), "
            f"true=({self.true_az_deg:.3f}, {self.true_el_deg:.3f}), "
            f"slant_m={self.slant_m}, t_unix={self.t_unix})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Sighting):
            return NotImplemented
        return all(getattr(self, k) == getattr(other, k) for k in self.__slots__)

    def __hash__(self) -> int:
        return hash(
            (
                self.target,
                self.encoder_az_deg,
                self.encoder_el_deg,
                self.true_az_deg,
                self.true_el_deg,
                self.slant_m,
                self.t_unix,
            )
        )

    @property
    def landmark(self) -> Landmark | None:
        """Return the FAA landmark for FAA targets, ``None`` otherwise.

        Compatibility shim for callers and tests that predate the
        tagged-union spec. New code should use ``self.target`` directly.
        """
        if self.target.kind == TargetKind.FAA:
            return self.target.landmark
        return None

    @property
    def kind(self) -> TargetKind:
        return self.target.kind


@dataclass
class RotationSolution:
    yaw_deg: float
    pitch_deg: float
    roll_deg: float
    residual_rms_deg: float
    per_landmark: list[dict]


# ---------- constants ------------------------------------------------


KEEP_MAX_AGE_S = 6 * 3600
KEEP_MAX_DISTANCE_M = 10.0

_EARTH_R_M = 6_371_000.0


# ---------- prior inspection ----------------------------------------


@dataclass(frozen=True)
class PriorInfo:
    """Minimum of what we need to know about the on-disk calibration
    to decide whether to keep it as a seed."""

    path: Path
    observer_lat_deg: float | None
    observer_lon_deg: float | None
    observer_alt_m: float | None
    calibrated_at: datetime | None
    age_s: float | None
    distance_from_current_m: float | None

    @property
    def should_default_keep(self) -> bool:
        """Default state of the clear-or-keep prompt: keep when the
        prior is both fresh and local."""
        if self.age_s is None or self.distance_from_current_m is None:
            return False
        return (
            self.age_s < KEEP_MAX_AGE_S
            and self.distance_from_current_m < KEEP_MAX_DISTANCE_M
        )


def parse_calibrated_at(raw: str | None) -> datetime | None:
    """Parse the ``calibrated_at`` string emitted by the calibration
    writers. Handles both the legacy dash-tz form
    (``%Y-%m-%dT%H-%M-%S%z``) and standard ISO 8601. Returns an
    aware UTC datetime, or ``None`` if missing / malformed."""
    if not isinstance(raw, str) or not raw:
        return None
    candidates = (
        "%Y-%m-%dT%H-%M-%S%z",  # legacy: 2026-04-21T23-28-52-0700
        "%Y-%m-%dT%H:%M:%S%z",  # standard ISO 8601 with colons
        "%Y-%m-%dT%H:%M:%S.%f%z",
    )
    for fmt in candidates:
        try:
            dt = datetime.strptime(raw, fmt)
            return dt.astimezone(timezone.utc)
        except ValueError:
            continue
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            return None
        return dt.astimezone(timezone.utc)
    except ValueError:
        return None


def inspect_prior(
    path: Path,
    current_lat: float,
    current_lon: float,
) -> PriorInfo | None:
    """Parse the prior calibration JSON (if any) and return age +
    distance metadata the clear-or-keep prompt uses. Returns ``None``
    when the file doesn't exist; returns a ``PriorInfo`` with mostly-
    ``None`` fields when the file exists but is unreadable / legacy."""
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return PriorInfo(path, None, None, None, None, None, None)
    obs = payload.get("observer") if isinstance(payload, dict) else None
    lat = lon = alt = None
    if isinstance(obs, dict):
        try:
            lat = float(obs.get("lat_deg")) if obs.get("lat_deg") is not None else None
            lon = float(obs.get("lon_deg")) if obs.get("lon_deg") is not None else None
            alt = float(obs.get("alt_m")) if obs.get("alt_m") is not None else None
        except (TypeError, ValueError):
            lat = lon = alt = None
    dt = parse_calibrated_at(
        payload.get("calibrated_at") if isinstance(payload, dict) else None,
    )
    now = datetime.now(timezone.utc)
    age = (now - dt).total_seconds() if dt is not None else None
    dist = (
        haversine_m(lat, lon, current_lat, current_lon)
        if lat is not None and lon is not None
        else None
    )
    return PriorInfo(
        path=path,
        observer_lat_deg=lat,
        observer_lon_deg=lon,
        observer_alt_m=alt,
        calibrated_at=dt,
        age_s=age,
        distance_from_current_m=dist,
    )


def decide_clear_or_keep(prior: PriorInfo | None) -> bool:
    """Return True if the prior should be kept by default, False if
    the smart-default is to clear it. Missing / unreadable / legacy
    priors default to clear (False) so an accidental run against an
    old compass calibration doesn't silently poison the seed."""
    if prior is None:
        return False
    return prior.should_default_keep


# ---------- geometry helpers ----------------------------------------


def _wrap_pm180(deg: float) -> float:
    d = (deg + 180.0) % 360.0 - 180.0
    return 180.0 if d == -180.0 else d


def pointing_uncertainty_deg(
    slant_m: float,
    horizontal_ft: float,
    vertical_ft: float,
    *,
    observer_sigma_m: float = 10.0,
) -> tuple[float, float]:
    """Propagate FAA landmark + observer GPS uncertainties to
    predicted (az, el) 1σ in degrees.

    Methodology (analytic, first-order small-angle):

        σ_az ≈ hypot(σ_h, σ_obs) / slant   [rad]
        σ_el ≈ hypot(σ_v, σ_obs) / slant   [rad]

    FAA DOF bounds are conventionally ~95% confidence, so we divide
    the published ± ft value by 2 to get a 1σ before combining with
    the observer GPS term (given as 1σ). The result is a true 1σ
    angular uncertainty suitable for ± display in the UI.

    The small-angle approximation holds to ≪ 1% for ground landmarks
    (σ/slant ≈ 3 × 10⁻³ for Hyperion); a Monte-Carlo cross-check
    lives in ``tests/test_calibrate_rotation.py`` and agrees with
    the analytic output within ~2% on 10 000 draws.

    ``nan`` ft inputs propagate to ``nan`` outputs — callers show
    those as "unknown" in the UI.
    """
    if slant_m <= 0.0 or not math.isfinite(slant_m):
        return (float("nan"), float("nan"))
    ft_to_m = 0.3048
    # Treat FAA bounds as 2σ → divide by 2 to get 1σ.
    sigma_h_m = (
        (horizontal_ft * ft_to_m) / 2.0
        if math.isfinite(horizontal_ft)
        else float("nan")
    )
    sigma_v_m = (
        (vertical_ft * ft_to_m) / 2.0 if math.isfinite(vertical_ft) else float("nan")
    )
    obs = float(observer_sigma_m)
    if math.isfinite(sigma_h_m):
        sigma_az_rad = math.hypot(sigma_h_m, obs) / slant_m
        sigma_az_deg = math.degrees(sigma_az_rad)
    else:
        sigma_az_deg = float("nan")
    if math.isfinite(sigma_v_m):
        sigma_el_rad = math.hypot(sigma_v_m, obs) / slant_m
        sigma_el_deg = math.degrees(sigma_el_rad)
    else:
        sigma_el_deg = float("nan")
    return (sigma_az_deg, sigma_el_deg)


def terrestrial_refraction_deg(slant_m: float, k: float = 0.13) -> float:
    """Apparent el lift from atmospheric bending over a ground path.

    Standard terrestrial-refraction coefficient ``k ≈ 0.13`` (over land;
    higher over water). The geometric Earth-curvature drop over slant
    ``d`` is ``d² / (2R)`` metres; refraction cancels ``k`` of it, so the
    apparent angular lift above a straight-line line-of-sight is
    approximately ``k · d / (2R)`` radians.

    For the Dockweiler ground landmarks: Hyperion @ 5.5 km → 0.003°;
    Culver City @ 9.2 km → 0.005°. Below FAA 1E accuracy (~0.04°), so
    numerically tiny — applied here for correctness rather than
    measurable improvement.
    """
    if slant_m <= 0.0:
        return 0.0
    return math.degrees(k * slant_m / (2.0 * _EARTH_R_M))


def predict_mount_azel(
    yaw_deg: float,
    pitch_deg: float,
    roll_deg: float,
    site: ObserverSite,
    landmark: Landmark,
    *,
    apply_refraction: bool = True,
) -> tuple[float, float, float]:
    """Predict (az, el, slant) in the mount frame for ``landmark``
    under the given rotation. With ``apply_refraction=True`` the el
    is lifted by the terrestrial-refraction correction so the
    prediction matches what the scope actually sees."""
    mf = MountFrame.from_euler_deg(
        yaw_deg=yaw_deg,
        pitch_deg=pitch_deg,
        roll_deg=roll_deg,
        site=site,
    )
    az, el, slant = mf.ecef_to_mount_azel(landmark.ecef())
    if apply_refraction:
        el = el + terrestrial_refraction_deg(slant)
    return az, el, slant


def _topo_to_mount_azel(
    az_deg: float,
    el_deg: float,
    topo_to_mount: np.ndarray,
) -> tuple[float, float]:
    """Rotate a topocentric (az, el) direction through a mount-frame
    rotation matrix. Used by the unified slew dispatch for celestial /
    plate-solve targets where we already have ENU truth and just need
    to express it in the mount frame.
    """
    az_r = math.radians(az_deg)
    el_r = math.radians(el_deg)
    cos_el = math.cos(el_r)
    east = math.sin(az_r) * cos_el
    north = math.cos(az_r) * cos_el
    up = math.sin(el_r)
    enu = np.array([east, north, up], dtype=np.float64)
    mount = topo_to_mount @ enu
    e, n, u = float(mount[0]), float(mount[1]), float(mount[2])
    slant = math.sqrt(e * e + n * n + u * u)
    if slant == 0.0:
        return (0.0, 90.0)
    az = (math.degrees(math.atan2(e, n)) + 360.0) % 360.0
    el = math.degrees(math.asin(u / slant))
    return az, el


# ---------- solver ---------------------------------------------------


def solve_rotation(
    sightings: list[Sighting],
    site: ObserverSite,
    *,
    yaw_seed_deg: float = 0.0,
    pitch_seed_deg: float = 0.0,
    roll_seed_deg: float = 0.0,
    dof: str = "auto",
) -> RotationSolution:
    """Least-squares fit of a mount-frame rotation to the sightings.

    ``dof``:
      - ``"auto"`` (default): fit yaw only when exactly one sighting
        is available (enables seeding landmark #2 without pitch/roll
        ambiguity), otherwise fit full 3-DOF (yaw, pitch, roll).
      - ``"yaw"``: force yaw-only. Useful as a sanity check.
      - ``"full"``: force 3-DOF even from a single sighting (under-
        determined but occasionally useful for regression tests).

    Residuals combine az and el errors with equal weight. The encoder
    az reported by the mount is in [-180, 180) — we wrap-diff the
    prediction against it to avoid 359° vs -1° issues. If the fit is
    ill-conditioned the solver still returns; the caller should check
    ``residual_rms_deg`` before trusting the result.
    """
    if len(sightings) < 1:
        raise ValueError("need at least 1 sighting to solve")
    if dof not in ("auto", "yaw", "full"):
        raise ValueError(f"unknown dof mode: {dof!r}")

    yaw_only = (dof == "yaw") or (dof == "auto" and len(sightings) == 1)

    def _predict(
        s: Sighting, yaw: float, pitch: float, roll: float
    ) -> tuple[float, float]:
        """Predict mount-frame ``(az, el)`` for one sighting under the
        trial rotation. FAA targets predict from the landmark's ECEF
        + the local observer (preserving the legacy refraction-after-
        rotation behaviour); celestial/plate-solve sightings predict
        from the (already-apparent) topocentric truth recorded at
        sighting time."""
        if s.target.kind == TargetKind.FAA and s.target.landmark is not None:
            pred_az, pred_el, _ = predict_mount_azel(
                yaw,
                pitch,
                roll,
                site,
                s.target.landmark,
            )
            return pred_az, pred_el
        return _predict_mount_azel_from_topo(
            yaw,
            pitch,
            roll,
            s.true_az_deg,
            s.true_el_deg,
        )

    def _resid(yaw: float, pitch: float, roll: float) -> np.ndarray:
        out = np.empty(2 * len(sightings), dtype=np.float64)
        for i, s in enumerate(sightings):
            pred_az, pred_el = _predict(s, yaw, pitch, roll)
            d_az = _wrap_pm180(_wrap_pm180(pred_az) - _wrap_pm180(s.encoder_az_deg))
            d_el = pred_el - s.encoder_el_deg
            out[2 * i] = d_az
            out[2 * i + 1] = d_el
        return out

    if yaw_only:

        def residuals(x: np.ndarray) -> np.ndarray:
            return _resid(float(x[0]), pitch_seed_deg, roll_seed_deg)

        x0 = np.array([yaw_seed_deg], dtype=np.float64)
        result = least_squares(residuals, x0, method="lm")
        yaw = float(result.x[0])
        pitch, roll = pitch_seed_deg, roll_seed_deg
    else:

        def residuals(x: np.ndarray) -> np.ndarray:
            return _resid(float(x[0]), float(x[1]), float(x[2]))

        x0 = np.array(
            [yaw_seed_deg, pitch_seed_deg, roll_seed_deg],
            dtype=np.float64,
        )
        result = least_squares(residuals, x0, method="lm")
        yaw, pitch, roll = [float(v) for v in result.x]

    per_landmark: list[dict] = []
    sq_sum = 0.0
    n = 0
    for s in sightings:
        pred_az, pred_el = _predict(s, yaw, pitch, roll)
        r_az = _wrap_pm180(_wrap_pm180(pred_az) - _wrap_pm180(s.encoder_az_deg))
        r_el = pred_el - s.encoder_el_deg
        per_landmark.append(
            sighting_to_record(
                s,
                predicted_az_deg=float(pred_az),
                predicted_el_deg=float(pred_el),
                residual_az_deg=float(r_az),
                residual_el_deg=float(r_el),
            )
        )
        sq_sum += r_az * r_az + r_el * r_el
        n += 2
    rms = float(np.sqrt(sq_sum / n)) if n else 0.0
    return RotationSolution(
        yaw_deg=yaw,
        pitch_deg=pitch,
        roll_deg=roll,
        residual_rms_deg=rms,
        per_landmark=per_landmark,
    )


def sighting_to_record(
    s: Sighting,
    *,
    predicted_az_deg: float,
    predicted_el_deg: float,
    residual_az_deg: float,
    residual_el_deg: float,
) -> dict:
    """Build the per-landmark record dict written to the calibration
    JSON and surfaced through the status response.

    FAA records preserve the legacy schema (``oas``, ``name``,
    ``lat_deg``, …) so existing consumers don't break. Celestial /
    plate-solve records carry their own kind-specific fields. Every
    record carries ``kind``, the encoder + truth + predicted + residual
    floats, and the per-target ``sigma_*`` for tooltip display.
    """
    common = {
        "kind": s.target.kind.value,
        "label": s.target.label,
        "encoder_az_deg": float(s.encoder_az_deg),
        "encoder_el_deg": float(s.encoder_el_deg),
        "true_az_deg": float(s.true_az_deg),
        "true_el_deg": float(s.true_el_deg),
        "predicted_az_deg": float(predicted_az_deg),
        "predicted_el_deg": float(predicted_el_deg),
        "residual_az_deg": float(residual_az_deg),
        "residual_el_deg": float(residual_el_deg),
        "sigma_az_deg": _none_or_finite(s.sigma_az_deg),
        "sigma_el_deg": _none_or_finite(s.sigma_el_deg),
    }
    if s.target.kind == TargetKind.FAA and s.target.landmark is not None:
        lm = s.target.landmark
        common["oas"] = lm.oas
        common["name"] = lm.name
        common["lat_deg"] = lm.lat_deg
        common["lon_deg"] = lm.lon_deg
        common["height_amsl_m"] = lm.height_amsl_m
        common["accuracy_class"] = lm.accuracy_class
        if s.slant_m is not None:
            common["slant_m"] = float(s.slant_m)
    elif s.target.kind == TargetKind.CELESTIAL:
        if s.target.ra_hours is not None:
            common["ra_hours"] = float(s.target.ra_hours)
        if s.target.dec_deg is not None:
            common["dec_deg"] = float(s.target.dec_deg)
        if s.target.vmag is not None:
            common["vmag"] = float(s.target.vmag)
        if s.target.bayer:
            common["bayer"] = s.target.bayer
    elif s.target.kind == TargetKind.PLATESOLVE:
        if s.target.seed_az_deg is not None:
            common["seed_az_deg"] = float(s.target.seed_az_deg)
        if s.target.seed_el_deg is not None:
            common["seed_el_deg"] = float(s.target.seed_el_deg)
    return common


def _none_or_finite(x: float | None) -> float | None:
    if x is None:
        return None
    return float(x) if math.isfinite(float(x)) else None


def _predict_mount_azel_from_topo(
    yaw_deg: float,
    pitch_deg: float,
    roll_deg: float,
    true_az_deg: float,
    true_el_deg: float,
) -> tuple[float, float]:
    """Predict mount-frame (az, el) for a celestial direction given a
    rotation hypothesis. The true (az, el) is the topocentric direction
    to a sky target (e.g. plate-solved (RA, Dec) → AltAz). This function
    rotates the unit vector through ``topo_to_mount`` (the same rotation
    the daytime path uses via :class:`MountFrame`) and reads back the
    mount-frame az/el.
    """
    az_r = math.radians(true_az_deg)
    el_r = math.radians(true_el_deg)
    # Topocentric ENU unit vector. `az` measured east-of-north so x=east,
    # y=north, z=up.
    cos_el = math.cos(el_r)
    east = math.sin(az_r) * cos_el
    north = math.cos(az_r) * cos_el
    up = math.sin(el_r)
    # Apply topo_to_mount (yaw → pitch → roll); inline so we don't import
    # MountFrame and its observer-site dependency for what is otherwise
    # a pure rotation. Matches `MountFrame.from_euler_deg` exactly.
    cy, sy = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    cp, sp = math.cos(math.radians(pitch_deg)), math.sin(math.radians(pitch_deg))
    cr, sr = math.cos(math.radians(roll_deg)), math.sin(math.radians(roll_deg))
    # yaw about up
    e1 = cy * east - sy * north
    n1 = sy * east + cy * north
    u1 = up
    # pitch about east
    e2 = e1
    n2 = cp * n1 - sp * u1
    u2 = sp * n1 + cp * u1
    # roll about north
    e3 = cr * e2 + sr * u2
    n3 = n2
    u3 = -sr * e2 + cr * u2
    slant = math.sqrt(e3 * e3 + n3 * n3 + u3 * u3)
    if slant == 0.0:
        return (0.0, 90.0)
    pred_az = (math.degrees(math.atan2(e3, n3)) + 360.0) % 360.0
    pred_el = math.degrees(math.asin(u3 / slant))
    return pred_az, pred_el


def solve_rotation_from_pairs(
    pairs: list[tuple[float, float, float, float]],
    *,
    yaw_seed_deg: float = 0.0,
    pitch_seed_deg: float = 0.0,
    roll_seed_deg: float = 0.0,
    dof: str = "auto",
) -> RotationSolution:
    """Least-squares fit of yaw/pitch/roll to ``(encoder_az, encoder_el,
    true_az, true_el)`` pairs.

    Mirrors :func:`solve_rotation` but skips the landmark-ECEF round-trip
    so the celestial nighttime path can plug in topocentric (az, el) it
    already converted from plate-solved (RA, Dec). The math is identical:
    rotate the true direction through the trial rotation, compare against
    the encoder reading, fit by Levenberg-Marquardt.

    The returned ``per_landmark`` list is repurposed for celestial
    sightings — the dicts have a ``kind: "platesolve"`` key plus the
    encoder + true az/el plus residuals. Records do **not** include
    landmark-specific fields (``oas``, ``height_amsl_m``, ``slant_m``);
    callers should treat the field set as a superset variant.
    """
    if len(pairs) < 1:
        raise ValueError("need at least 1 sighting to solve")
    if dof not in ("auto", "yaw", "full"):
        raise ValueError(f"unknown dof mode: {dof!r}")

    yaw_only = (dof == "yaw") or (dof == "auto" and len(pairs) == 1)

    def _resid(yaw: float, pitch: float, roll: float) -> np.ndarray:
        out = np.empty(2 * len(pairs), dtype=np.float64)
        for i, (enc_az, enc_el, true_az, true_el) in enumerate(pairs):
            pred_az, pred_el = _predict_mount_azel_from_topo(
                yaw, pitch, roll, true_az, true_el
            )
            d_az = _wrap_pm180(_wrap_pm180(pred_az) - _wrap_pm180(enc_az))
            d_el = pred_el - enc_el
            out[2 * i] = d_az
            out[2 * i + 1] = d_el
        return out

    if yaw_only:

        def residuals(x: np.ndarray) -> np.ndarray:
            return _resid(float(x[0]), pitch_seed_deg, roll_seed_deg)

        x0 = np.array([yaw_seed_deg], dtype=np.float64)
        result = least_squares(residuals, x0, method="lm")
        yaw = float(result.x[0])
        pitch, roll = pitch_seed_deg, roll_seed_deg
    else:

        def residuals(x: np.ndarray) -> np.ndarray:
            return _resid(float(x[0]), float(x[1]), float(x[2]))

        x0 = np.array(
            [yaw_seed_deg, pitch_seed_deg, roll_seed_deg],
            dtype=np.float64,
        )
        result = least_squares(residuals, x0, method="lm")
        yaw, pitch, roll = [float(v) for v in result.x]

    per_record: list[dict] = []
    sq_sum = 0.0
    n = 0
    for enc_az, enc_el, true_az, true_el in pairs:
        pred_az, pred_el = _predict_mount_azel_from_topo(
            yaw, pitch, roll, true_az, true_el
        )
        r_az = _wrap_pm180(_wrap_pm180(pred_az) - _wrap_pm180(enc_az))
        r_el = pred_el - enc_el
        per_record.append(
            {
                "kind": "platesolve",
                "encoder_az_deg": float(enc_az),
                "encoder_el_deg": float(enc_el),
                "true_az_deg": float(true_az),
                "true_el_deg": float(true_el),
                "predicted_az_deg": float(pred_az),
                "predicted_el_deg": float(pred_el),
                "residual_az_deg": float(r_az),
                "residual_el_deg": float(r_el),
            }
        )
        sq_sum += r_az * r_az + r_el * r_el
        n += 2
    rms = float(np.sqrt(sq_sum / n)) if n else 0.0
    return RotationSolution(
        yaw_deg=yaw,
        pitch_deg=pitch,
        roll_deg=roll,
        residual_rms_deg=rms,
        per_landmark=per_record,
    )


# ---------- JSON writer ----------------------------------------------


def write_calibration(
    path: Path,
    sol: RotationSolution,
    site: ObserverSite,
    landmark_records: list[dict],
    *,
    calibration_method: str | None = None,
) -> None:
    """Write the calibration JSON every consumer reads.

    Schema keeps backward compatibility with the compass-tool format:
    yaw/pitch/roll + origin_offset_ecef_m are the minimum every loader
    honours. ``observer`` ties the calibration to the site that
    produced it; ``landmarks`` records per-point residuals for audit.

    ``calibration_method`` defaults to the legacy ``rotation_landmarks``
    when every record is a FAA landmark, ``rotation_unified`` when the
    record set mixes target kinds, and ``rotation_celestial`` /
    ``rotation_platesolve`` when the records are all of one non-FAA kind.
    Callers can override explicitly when they want a specific tag.
    """
    if calibration_method is None:
        calibration_method = _infer_calibration_method(landmark_records)
    payload = {
        "calibration_method": calibration_method,
        "calibrated_at": time.strftime("%Y-%m-%dT%H-%M-%S%z"),
        "yaw_offset_deg": sol.yaw_deg,
        "pitch_offset_deg": sol.pitch_deg,
        "roll_offset_deg": sol.roll_deg,
        "origin_offset_ecef_m": [0.0, 0.0, 0.0],
        "residual_rms_deg": sol.residual_rms_deg,
        "n_landmarks": len(landmark_records),
        "observer": {
            "lat_deg": site.lat_deg,
            "lon_deg": site.lon_deg,
            "alt_m": site.alt_m,
            "source": "telescope_get_device_state",
        },
        "landmarks": landmark_records,
    }
    # Atomic write: every consumer (MountFrame.from_calibration_json,
    # the live-tracker boot path) reads this file unconditionally on
    # session start. A non-atomic open(..., "w") leaves a truncated file
    # on SIGKILL/power-loss mid-write and blocks the next session.
    from device._atomic_json import write_atomic_json

    write_atomic_json(path, payload, indent=2)


def _infer_calibration_method(records: list[dict]) -> str:
    """Pick a ``calibration_method`` tag from a record set's kinds.

    Mixed-kind sessions get ``rotation_unified``; single-kind sessions
    get a kind-specific tag so existing audit tooling can quickly
    grep for daytime-only calibrations.
    """
    kinds = {r.get("kind", "faa") for r in records}
    if len(kinds) <= 1:
        only = next(iter(kinds), "faa")
        if only == TargetKind.FAA.value:
            return "rotation_landmarks"
        if only == TargetKind.CELESTIAL.value:
            return "rotation_celestial"
        if only == TargetKind.PLATESOLVE.value:
            return "rotation_platesolve"
    return "rotation_unified"


# ---------- CalibrationSession --------------------------------------


# Maximum per-command nudge, in degrees. Guards against a typo in the
# web UI driving the mount tens of degrees in one request.
MAX_NUDGE_PER_CMD_DEG = 5.0

# Arrive tolerance for the pre-slew to each landmark (coarse) vs.
# the nudge-to-beacon move (fine). Matches the CLI values.
ARRIVE_TOL_SLEW_DEG = 0.3
ARRIVE_TOL_NUDGE_DEG = 0.1


def _now_utc() -> datetime:
    """Aware-UTC datetime. Wrapped so tests can monkeypatch the time
    source without touching ``datetime.now``."""
    return datetime.now(timezone.utc)


@dataclass
class _Command:
    """Worker-thread queue entry."""

    kind: str  # "slew" | "nudge" | "sight" | "skip" | "commit" | "cancel"
    payload: dict[str, Any] = field(default_factory=dict)


class _PlateSolveSightingFailure(Exception):
    """Raised inside :meth:`CalibrationSession._capture_and_solve` to
    signal that the plate-solve sub-flow failed and the sighting must
    not be appended. Caught one level up so the operator can retry."""


@dataclass
class CalibrationStatus:
    """JSON-serialisable session snapshot for the browser."""

    active: bool
    # init / slewing / nudging / plate_solving / sighting / review /
    # committed / cancelled / error. ``plate_solving`` is set while a
    # plate-solve sighting's inner loop is running so the UI can show
    # a spinner and disable Sight ✓ until the result lands.
    phase: str
    target_idx: int
    n_targets: int
    current_landmark: (
        dict | None
    )  # kind-aware: {kind, label, true_az_deg, true_el_deg, slant_m?, ...}
    target_az_deg: float | None  # pending encoder target (drives the mount)
    target_el_deg: float | None
    encoder_az_deg: float | None  # last-read encoder (polled each cycle)
    encoder_el_deg: float | None
    solution: dict | None  # {yaw, pitch, roll, rms, per_landmark}
    errors: list[str]
    # Heterogeneous list of all selected targets, in the order the
    # session will visit them. Each entry is the same kind-aware dict
    # ``current_landmark`` uses (without truth fields). The UI uses
    # this to render the "FAA / ★ / 🎯 plate" tag next to each pending
    # row.
    targets: list[dict] | None = None


class CalibrationSession:
    """Thread-backed calibration run. Mirrors LiveTrackSession: spawn
    a daemon worker, expose `start/stop/is_alive/status`, accept
    command posts (``nudge``, ``sight`` …) that flow through a
    thread-safe queue so HTTP handlers can return immediately.

    Workflow per target:
      1. Pre-slew to the landmark's predicted encoder (az, el) under
         any prior rotation supplied at construction time.
      2. Operator nudges via the HTTP `/nudge` endpoint; each nudge
         re-issues `move_to_ff` against the updated encoder target
         and reads the encoder back.
      3. Operator posts `/sight`; the session records the current
         encoder as a Sighting, refits (yaw-only for 1, full 3-DOF
         for ≥2), and auto-advances to the next landmark.

    The session ignores repeat commands that are queued faster than
    the mount can execute them (nudge coalescing keeps the latest
    pending target rather than backing up a queue of moves).
    """

    # Polling cadence for encoder reads when the mount is idle (to
    # keep the browser KPI strip responsive).
    IDLE_POLL_DT_S = 0.5

    def __init__(
        self,
        telescope_id: int,
        targets: list[tuple[Landmark, float, float, float]] | None = None,
        site: ObserverSite | None = None,
        *,
        out_path: Path,
        prior_frame: MountFrame | None = None,
        alpaca_host: str = "127.0.0.1",
        alpaca_port: int | None = None,
        dry_run: bool = False,
        target_specs: list[CalibrationTargetSpec] | None = None,
        plate_solver: Any | None = None,
        capture_image_fn: Any | None = None,
    ) -> None:
        """Construct a calibration session.

        Two ways to specify targets, mutually exclusive:

        - ``target_specs``: the unified path. A list of
          :class:`CalibrationTargetSpec` of any kind (FAA, celestial,
          plate-solve). Used by the unified picker.
        - ``targets``: legacy FAA-only path. List of
          ``(landmark, true_az, true_el, slant)`` tuples — each is
          wrapped into an FAA-kind ``CalibrationTargetSpec``. Existing
          callers + tests keep working without code changes.

        ``plate_solver`` and ``capture_image_fn`` are required iff any
        spec has ``kind=PLATESOLVE``. ``capture_image_fn`` is an
        ``() -> Path`` that captures from the imager and returns the
        resulting image path; the session calls it when the operator
        clicks Sight on a plate-solve target.
        """
        if site is None:
            raise ValueError("site is required")
        if target_specs is None and targets is None:
            raise ValueError("need either target_specs or targets")
        if target_specs is not None and targets is not None:
            raise ValueError("pass target_specs or targets, not both")
        if target_specs is None:
            target_specs = [
                CalibrationTargetSpec.from_landmark(lm, slant_m=float(slant))
                for (lm, _az, _el, slant) in targets
            ]
        if not target_specs:
            raise ValueError("need at least 1 target")
        # Sanity-check plate-solve targets have a solver wired up.
        if (
            any(ts.kind == TargetKind.PLATESOLVE for ts in target_specs)
            and plate_solver is None
        ):
            raise ValueError(
                "plate_solver required when target_specs contain PLATESOLVE"
            )
        self.telescope_id = int(telescope_id)
        self.target_specs: list[CalibrationTargetSpec] = list(target_specs)
        # Pre-resolved truth + sigma cache for FAA targets so the status
        # response can render the active-target banner before sighting
        # (matches the legacy ``targets`` tuple shape). Celestial /
        # plate-solve entries are resolved on demand.
        self._target_meta: list[dict] = self._resolve_target_meta()
        # Backwards-compat alias kept so internal consumers that relied
        # on ``self.targets`` (and tests / call sites) keep working.
        # Each entry mirrors the legacy 4-tuple shape for FAA targets;
        # for non-FAA targets, ``true_az/el/slant`` are None / NaN.
        self.targets = self._legacy_targets_view()
        self.site = site
        self.out_path = Path(out_path)
        self._prior_frame = prior_frame
        self._alpaca_host = alpaca_host
        self._alpaca_port = alpaca_port
        self.dry_run = bool(dry_run)
        self._plate_solver = plate_solver
        self._capture_image_fn = capture_image_fn

        self._queue: queue.Queue[_Command] = queue.Queue()
        self._stop_evt = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

        # Mutable state protected by _lock.
        self._phase = "init"
        self._target_idx = 0
        self._sightings: list[Sighting] = []
        self._solution: RotationSolution | None = None
        self._target_az: float | None = None
        self._target_el: float | None = None
        self._encoder_az: float | None = None
        self._encoder_el: float | None = None
        self._errors: list[str] = []
        # Per-target plate-solve diagnostics carried into the status
        # response so the UI can render a "Solving…" spinner / failure
        # message without polling its own endpoint.
        self._last_platesolve_error: str | None = None

    @classmethod
    def from_landmarks(
        cls,
        telescope_id: int,
        targets: list[tuple[Landmark, float, float, float]],
        site: ObserverSite,
        *,
        out_path: Path,
        prior_frame: MountFrame | None = None,
        alpaca_host: str = "127.0.0.1",
        alpaca_port: int | None = None,
        dry_run: bool = False,
    ) -> "CalibrationSession":
        """Backwards-compat factory for FAA-only sessions. Wraps each
        ``(landmark, az, el, slant)`` tuple into an FAA-kind
        :class:`CalibrationTargetSpec` and forwards to ``__init__``.
        Use this in callers that predate the unified picker."""
        return cls(
            telescope_id=telescope_id,
            targets=targets,
            site=site,
            out_path=out_path,
            prior_frame=prior_frame,
            alpaca_host=alpaca_host,
            alpaca_port=alpaca_port,
            dry_run=dry_run,
        )

    def _resolve_target_meta(self) -> list[dict]:
        """Pre-resolve display metadata for each target. Called once
        at construction; FAA targets resolve their truth + slant + σ
        from the spec; celestial / plate-solve targets fill placeholder
        fields the sighting cycle will overwrite."""
        meta: list[dict] = []
        for ts in self.target_specs:
            entry: dict[str, Any] = {
                "kind": ts.kind.value,
                "label": ts.label,
                "spec": ts,
            }
            if ts.kind == TargetKind.FAA and ts.landmark is not None:
                # FAA truth is time-independent, so resolve eagerly.
                # ``site`` is needed; called before self.site is set,
                # so accept None ``slant_m`` and recompute lazily in
                # ``status()``.
                pass
            meta.append(entry)
        return meta

    def _legacy_targets_view(
        self,
    ) -> list[tuple[Landmark | None, float, float, float]]:
        """Compatibility shim for code that still expects the legacy
        4-tuple shape. FAA entries get the original landmark + truth;
        non-FAA entries get a placeholder so existing index iteration
        doesn't crash. New callers should use ``self.target_specs``."""
        out: list[tuple[Landmark | None, float, float, float]] = []
        for ts in self.target_specs:
            if ts.kind == TargetKind.FAA and ts.landmark is not None:
                slant = ts.slant_m if ts.slant_m is not None else float("nan")
                out.append((ts.landmark, float("nan"), float("nan"), slant))
            else:
                out.append((None, float("nan"), float("nan"), float("nan")))
        return out

    # ---------- public lifecycle ----------

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("calibration session already running")
        self._stop_evt.clear()
        self._thread = threading.Thread(
            target=self._run,
            name=f"CalibrationSession({self.telescope_id})",
            daemon=True,
        )
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop_evt.set()
        self._queue.put(_Command("cancel"))
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def status(self) -> CalibrationStatus:
        with self._lock:
            lm_info = None
            if 0 <= self._target_idx < len(self.target_specs):
                lm_info = self._target_status_dict(self._target_idx)
            sol_info = None
            if self._solution is not None:
                sol_info = {
                    "yaw_deg": self._solution.yaw_deg,
                    "pitch_deg": self._solution.pitch_deg,
                    "roll_deg": self._solution.roll_deg,
                    "residual_rms_deg": self._solution.residual_rms_deg,
                    "per_landmark": list(self._solution.per_landmark),
                }
            targets_list = [ts.to_dict() for ts in self.target_specs]
            return CalibrationStatus(
                active=self.is_alive(),
                phase=self._phase,
                target_idx=self._target_idx,
                n_targets=len(self.target_specs),
                current_landmark=lm_info,
                target_az_deg=self._target_az,
                target_el_deg=self._target_el,
                encoder_az_deg=self._encoder_az,
                encoder_el_deg=self._encoder_el,
                solution=sol_info,
                errors=list(self._errors),
                targets=targets_list,
            )

    def _target_status_dict(self, idx: int) -> dict | None:
        """Build the ``current_landmark`` payload for the status
        response. Kind-aware: FAA records preserve the legacy field set
        (``oas``, ``name``, ``slant_m``, ``aiming_hint``) so the existing
        UI banner keeps rendering; celestial / plate-solve add their own
        kind-specific fields. Every record carries ``kind`` so the UI
        can pick the right glyph and tooltip."""
        if not (0 <= idx < len(self.target_specs)):
            return None
        ts = self.target_specs[idx]
        out: dict[str, Any] = ts.to_dict()
        # Truth / sigma resolve. For FAA targets we resolve from the
        # cached spec; celestial uses a fresh now-UTC; plate-solve
        # surfaces the seed (or NaN) until a sighting fills in the
        # WCS-derived truth.
        if ts.kind == TargetKind.FAA and ts.landmark is not None:
            try:
                az, el, slant = ts.resolve_true_altaz(self.site, _now_utc())
            except Exception:
                az = el = float("nan")
                slant = None
            out["true_az_deg"] = _none_or_finite(az)
            out["true_el_deg"] = _none_or_finite(el)
            if slant is not None:
                out["slant_m"] = float(slant)
            from scripts.trajectory.faa_dof import aiming_hint as _aim

            out["aiming_hint"] = _aim(ts.landmark)
        elif ts.kind == TargetKind.CELESTIAL:
            try:
                az, el, _ = ts.resolve_true_altaz(self.site, _now_utc())
                out["true_az_deg"] = float(az)
                out["true_el_deg"] = float(el)
            except Exception:
                out["true_az_deg"] = None
                out["true_el_deg"] = None
            out["aiming_hint"] = (
                f"Centre {ts.label} in the eyepiece; bright stars "
                "click into focus crisply."
            )
        elif ts.kind == TargetKind.PLATESOLVE:
            out["true_az_deg"] = (
                float(ts.seed_az_deg) if ts.seed_az_deg is not None else None
            )
            out["true_el_deg"] = (
                float(ts.seed_el_deg) if ts.seed_el_deg is not None else None
            )
            out["aiming_hint"] = (
                "Free-aim: jog the mount toward the sky region you want, "
                "then Sight to capture + plate-solve."
            )
        # Per-target sigma from the spec helper.
        saz, sel = ts.sigma_az_el_deg()
        out["sigma_az_deg"] = _none_or_finite(saz) if saz is not None else None
        out["sigma_el_deg"] = _none_or_finite(sel) if sel is not None else None
        if self._last_platesolve_error and ts.kind == TargetKind.PLATESOLVE:
            out["last_platesolve_error"] = self._last_platesolve_error
        return out

    # ---------- command posts ----------

    def nudge(self, d_az_deg: float, d_el_deg: float) -> None:
        d_az = max(-MAX_NUDGE_PER_CMD_DEG, min(MAX_NUDGE_PER_CMD_DEG, float(d_az_deg)))
        d_el = max(-MAX_NUDGE_PER_CMD_DEG, min(MAX_NUDGE_PER_CMD_DEG, float(d_el_deg)))
        self._queue.put(_Command("nudge", {"d_az": d_az, "d_el": d_el}))

    def sight(self) -> None:
        self._queue.put(_Command("sight"))

    def skip(self) -> None:
        self._queue.put(_Command("skip"))

    def commit(self) -> None:
        self._queue.put(_Command("commit"))

    def cancel(self) -> None:
        self._queue.put(_Command("cancel"))

    # ---------- worker thread ----------

    def _run(self) -> None:
        cli = None
        try:
            cli = self._connect_mount()
            self._set_phase("slewing")
            self._slew_to_target(cli, 0)
            if self._stop_evt.is_set():
                return
            self._set_phase("nudging")
            self._process_loop(cli)
        except Exception as exc:  # noqa: BLE001 — surface any worker failure
            with self._lock:
                self._errors.append(f"worker crashed: {exc}")
                self._phase = "error"
        finally:
            # Best-effort stop of any lingering motion.
            if cli is not None and not self.dry_run:
                try:
                    cli.method_sync(
                        "scope_speed_move", {"speed": 0, "angle": 0, "dur_sec": 0}
                    )
                except Exception:
                    pass

    def _connect_mount(self):
        """Import lazily so unit tests that stub AlpacaClient via the
        module-level symbol pick up the stub without a prior import
        side-effect."""
        from device.alpaca_client import AlpacaClient
        from device.config import Config

        port = self._alpaca_port if self._alpaca_port is not None else int(Config.port)
        cli = AlpacaClient(self._alpaca_host, port, self.telescope_id)
        if self.dry_run:
            return cli
        try:
            from device.velocity_controller import (
                ensure_scenery_mode,
                set_tracking,
            )

            ensure_scenery_mode(cli)
            set_tracking(cli, False)
        except Exception as exc:
            with self._lock:
                self._errors.append(f"scenery/tracking setup: {exc}")
        return cli

    def _process_loop(self, cli) -> None:
        """Dispatch commands until the queue is empty, then idle-poll
        the encoder. Returns when a CMD_CANCEL is processed (already
        handled inside the dispatcher, which sets phase to cancelled)."""
        while not self._stop_evt.is_set():
            try:
                cmd = self._queue.get(timeout=self.IDLE_POLL_DT_S)
            except queue.Empty:
                # Idle tick: refresh encoder status so the UI polling
                # loop stays live even when nothing else is happening.
                self._poll_encoder_nonfatal(cli)
                continue
            if cmd.kind == "cancel":
                self._set_phase("cancelled")
                return
            self._dispatch(cli, cmd)
            if self._phase in ("committed", "cancelled", "error"):
                return

    def _dispatch(self, cli, cmd: _Command) -> None:
        if cmd.kind == "nudge":
            self._on_nudge(cli, float(cmd.payload["d_az"]), float(cmd.payload["d_el"]))
        elif cmd.kind == "sight":
            self._on_sight(cli)
        elif cmd.kind == "skip":
            self._on_skip(cli)
        elif cmd.kind == "commit":
            self._on_commit()

    def _on_nudge(self, cli, d_az: float, d_el: float) -> None:
        with self._lock:
            if self._target_az is None or self._target_el is None:
                # No pre-slew baseline yet; ignore.
                return
            self._target_az += d_az
            self._target_el += d_el
            target_az = self._target_az
            target_el = self._target_el
        # Coalesce: drain pending nudges queued behind this one, sum
        # their deltas, and issue a single move to the final target.
        coalesced_d_az = d_az
        coalesced_d_el = d_el
        while True:
            try:
                nxt = self._queue.get_nowait()
            except queue.Empty:
                break
            if nxt.kind != "nudge":
                # Put non-nudge command back on the queue front-ish; we
                # can't peek, so just enqueue. Order preservation
                # between the coalesced move and the subsequent command
                # is preserved by the move completing first.
                self._queue.put(nxt)
                break
            with self._lock:
                self._target_az += float(nxt.payload["d_az"])
                self._target_el += float(nxt.payload["d_el"])
                target_az = self._target_az
                target_el = self._target_el
            coalesced_d_az += float(nxt.payload["d_az"])
            coalesced_d_el += float(nxt.payload["d_el"])
        self._set_phase("nudging")
        if self.dry_run:
            with self._lock:
                self._encoder_az = target_az
                self._encoder_el = target_el
            return

        # Delegate to the calibrate-motion session if one is running on
        # this telescope. The continuous-control loop closes |err| below
        # 0.005° rather than truncating at ``move_to_ff``'s 0.1° arrive
        # tolerance, which is what makes the small-step UI buttons (down
        # to 0.005°) actually meaningful at the mount. We pass the
        # coalesced delta so the streaming target shifts by the same
        # amount the legacy path would have moved to in absolute terms.
        motion = self._motion_delegate()
        if motion is not None:
            try:
                motion.nudge_target(coalesced_d_az, coalesced_d_el)
            except Exception as exc:
                with self._lock:
                    self._errors.append(f"motion nudge_target failed: {exc}")
                return
            # Update self._encoder_* from the latest tick the motion
            # session has seen so the UI's KPI strip stays in sync.
            try:
                st = motion.status()
                if st.cur_cum_az_deg is not None and st.cur_el_deg is not None:
                    with self._lock:
                        self._encoder_az = float(st.cur_cum_az_deg)
                        self._encoder_el = float(st.cur_el_deg)
            except Exception:
                pass
            return

        try:
            from device.velocity_controller import move_to_ff

            loc = EarthLocation.from_geodetic(0, 0, 0)
            cur_el = self._encoder_el if self._encoder_el is not None else target_el
            cur_az = self._encoder_az if self._encoder_az is not None else target_az
            new_el, new_az, _ = move_to_ff(
                cli,
                target_az_deg=target_az,
                target_el_deg=target_el,
                cur_az_deg=cur_az,
                cur_el_deg=cur_el,
                loc=loc,
                tag="[calibrate_web]",
                arrive_tolerance_deg=ARRIVE_TOL_NUDGE_DEG,
            )
            with self._lock:
                self._encoder_az = new_az
                self._encoder_el = new_el
        except Exception as exc:
            with self._lock:
                self._errors.append(f"nudge move_to_ff failed: {exc}")

    def _on_sight(self, cli) -> None:
        """Record the current encoder as a Sighting and refit.

        Kind-aware truth resolution:

        - FAA / CELESTIAL: encoder is already at the operator-aligned
          position; resolve truth at sighting time and append.
        - PLATESOLVE: capture from the imager + run plate-solve. If the
          inner loop fails, the sight attempt fails — we don't append a
          sighting and surface ``last_platesolve_error`` so the UI can
          let the user retry.
        """
        self._poll_encoder_nonfatal(cli)
        with self._lock:
            if self._encoder_az is None or self._encoder_el is None:
                self._errors.append("cannot sight: no encoder read yet")
                return
            if not (0 <= self._target_idx < len(self.target_specs)):
                return
            ts = self.target_specs[self._target_idx]
            enc_az = float(self._encoder_az)
            enc_el = float(self._encoder_el)
        # Resolve truth + handle plate-solve outside the lock — celestial
        # ephem and the plate-solver both can take seconds.
        if ts.kind == TargetKind.PLATESOLVE:
            self._set_phase("plate_solving")
            try:
                outcome = self._capture_and_solve(ts, enc_az, enc_el)
            except _PlateSolveSightingFailure as exc:
                with self._lock:
                    self._last_platesolve_error = str(exc)
                    self._errors.append(f"plate-solve sight failed: {exc}")
                # Stay on the same target; don't auto-advance.
                self._set_phase("nudging")
                return
            true_az = outcome.true_az_deg
            true_el = outcome.true_el_deg
            slant = None
            sigma = outcome.sigma_deg if outcome.sigma_deg is not None else None
            sigma_az = sigma_el = sigma
        else:
            try:
                az, el, slant = ts.resolve_true_altaz(self.site, _now_utc())
            except Exception as exc:
                with self._lock:
                    self._errors.append(f"resolve {ts.label} truth: {exc}")
                return
            true_az = float(az)
            true_el = float(el)
            saz, sel = ts.sigma_az_el_deg()
            sigma_az = saz
            sigma_el = sel
        # Wrap az to match the encoder convention so residual math
        # doesn't trip on the ±180 boundary.
        true_az_wrapped = ((float(true_az) + 180.0) % 360.0) - 180.0
        s = Sighting(
            target=ts,
            encoder_az_deg=enc_az,
            encoder_el_deg=enc_el,
            true_az_deg=true_az_wrapped,
            true_el_deg=float(true_el),
            slant_m=(float(slant) if slant is not None else None),
            t_unix=time.time(),
            sigma_az_deg=sigma_az,
            sigma_el_deg=sigma_el,
        )
        # Pre-compute / refresh the per-sighting sigma if the spec
        # carries enough info (FAA accuracy class). This is also done
        # by ``ts.sigma_az_el_deg()`` above; carry through to Sighting.
        with self._lock:
            self._sightings.append(s)
            sightings = list(self._sightings)
            next_idx = self._target_idx + 1
            self._last_platesolve_error = None
        # Fit outside the lock.
        try:
            sol = solve_rotation(sightings, self.site)
            with self._lock:
                self._solution = sol
        except ValueError as exc:
            with self._lock:
                self._errors.append(f"solve_rotation failed: {exc}")

        with self._lock:
            self._target_idx = next_idx
        if next_idx >= len(self.target_specs):
            self._set_phase("review")
        else:
            self._slew_to_target(cli, next_idx)
            self._set_phase("nudging")

    def _capture_and_solve(
        self, ts: CalibrationTargetSpec, enc_az: float, enc_el: float
    ) -> PlateSolveOutcome:
        """Drive the plate-solve sub-flow for a PLATESOLVE sighting.

        Reuses :func:`device.nighttime_calibration.radec_to_topocentric_azel`
        so the celestial transform stays in one place. Raises
        :class:`_PlateSolveSightingFailure` for any of: missing solver,
        capture failure, solver error, FOV out of range, or transform
        failure. Caller catches it, surfaces the error, and lets the
        operator retry.
        """
        if self._plate_solver is None:
            raise _PlateSolveSightingFailure(
                "plate solver not configured; cannot sight a platesolve target"
            )
        if self._capture_image_fn is None:
            raise _PlateSolveSightingFailure(
                "no capture_image_fn configured; cannot sight a platesolve target"
            )
        try:
            image_path = self._capture_image_fn()
        except Exception as exc:
            raise _PlateSolveSightingFailure(f"image capture failed: {exc}") from exc
        try:
            solve_result = self._plate_solver.solve(Path(image_path))
        except Exception as exc:
            raise _PlateSolveSightingFailure(f"solver error: {exc}") from exc
        # Convert (RA, Dec) → topocentric (az, el) using the same path
        # the standalone NighttimeCalibrationSession uses.
        from device.nighttime_calibration import radec_to_topocentric_azel

        try:
            true_az, true_el = radec_to_topocentric_azel(
                solve_result.ra_deg,
                solve_result.dec_deg,
                time.time(),
                self.site,
            )
        except Exception as exc:
            raise _PlateSolveSightingFailure(
                f"radec→azel transform failed: {exc}"
            ) from exc
        # The solver's reported FOV is a sanity gate — outside of the
        # S50's expected range, the match is almost certainly wrong.
        from device.plate_solver import S50_FOV_MAX_DEG, S50_FOV_MIN_DEG

        fx = float(solve_result.fov_x_deg)
        fy = float(solve_result.fov_y_deg)
        if not (
            S50_FOV_MIN_DEG <= fx <= S50_FOV_MAX_DEG
            and S50_FOV_MIN_DEG <= fy <= S50_FOV_MAX_DEG
        ):
            raise _PlateSolveSightingFailure(
                f"solver FOV {fx:.2f}×{fy:.2f}° outside "
                f"[{S50_FOV_MIN_DEG}, {S50_FOV_MAX_DEG}]°"
            )
        return PlateSolveOutcome(
            true_az_deg=float(true_az),
            true_el_deg=float(true_el),
            ra_deg=float(solve_result.ra_deg),
            dec_deg=float(solve_result.dec_deg),
            sigma_deg=None,
        )

    def _on_skip(self, cli) -> None:
        with self._lock:
            remaining = len(self.target_specs) - (self._target_idx + 1)
            already_sighted = len(self._sightings)
            projected = already_sighted + remaining
        if projected < 2:
            with self._lock:
                self._errors.append("cannot skip: would leave fewer than 2 sightings")
            return
        with self._lock:
            next_idx = self._target_idx + 1
            self._target_idx = next_idx
        if next_idx >= len(self.target_specs):
            self._set_phase("review")
        else:
            self._slew_to_target(cli, next_idx)
            self._set_phase("nudging")

    def _on_commit(self) -> None:
        with self._lock:
            sol = self._solution
            sightings = list(self._sightings)
        if sol is None or len(sightings) < 2:
            with self._lock:
                self._errors.append("cannot commit: need ≥ 2 sightings")
            return
        try:
            write_calibration(self.out_path, sol, self.site, sol.per_landmark)
        except Exception as exc:
            with self._lock:
                self._errors.append(f"write_calibration failed: {exc}")
                self._phase = "error"
            return
        self._set_phase("committed")

    def _slew_to_target(self, cli, idx: int) -> None:
        """Drive the mount to the predicted encoder (az, el) for
        ``target_specs[idx]``. Updates pending target + current encoder.

        Dispatches by kind:

        - FAA: predict via the prior mount frame from the landmark's
          ECEF (legacy behaviour).
        - CELESTIAL: resolve fresh topocentric (az, el) from ephem at
          slew time, then rotate through the prior mount frame.
        - PLATESOLVE: use the spec's seed (az, el) when present;
          otherwise leave the mount where it is (free-aim — the
          operator jogs to a clearer-sky region before sighting).
        """
        if not (0 <= idx < len(self.target_specs)):
            return
        ts = self.target_specs[idx]
        prior_frame = self._prior_frame or MountFrame.from_identity_enu(self.site)
        # Plate-solve free-aim mode: no slew, no sun check (the operator
        # is already pointing somewhere they jogged to). Skip ahead.
        if ts.kind == TargetKind.PLATESOLVE and (
            ts.seed_az_deg is None or ts.seed_el_deg is None
        ):
            return
        # Resolve seed (mount-frame) + truth (sky-frame). For FAA /
        # CELESTIAL we resolve the truth from the spec; for seeded
        # PLATESOLVE the operator-supplied seed *is* the mount-frame
        # target and serves as the sky-frame proxy for sun safety.
        if ts.kind == TargetKind.PLATESOLVE:
            pred_az = float(ts.seed_az_deg)
            pred_el = float(ts.seed_el_deg)
            true_az = pred_az
            true_el = pred_el
        else:
            try:
                true_az, true_el, _ = ts.resolve_true_altaz(self.site, _now_utc())
            except Exception as exc:
                with self._lock:
                    self._errors.append(f"resolve {ts.label}: {exc}")
                    self._phase = "error"
                self._stop_evt.set()
                return
            if ts.kind == TargetKind.FAA and ts.landmark is not None:
                # FAA: predict mount-frame az/el via prior_frame from
                # the landmark's ECEF (legacy behaviour preserves the
                # refraction-after-rotation convention).
                pred_az, pred_el, _ = prior_frame.ecef_to_mount_azel(ts.landmark.ecef())
            else:
                # CELESTIAL — rotate the topocentric apparent (az, el)
                # through ``prior_frame.topo_to_mount`` to get
                # mount-frame (az, el).
                pred_az, pred_el = _topo_to_mount_azel(
                    true_az, true_el, prior_frame.topo_to_mount
                )
        pred_az_wrapped = ((pred_az + 180.0) % 360.0) - 180.0

        # Pre-flight sun-avoidance. Uses the *true* topocentric (az, el)
        # for the target so the check is in sky frame regardless of the
        # prior calibration's accuracy. Plate-solve seeded slews use the
        # mount-frame seed (operator-chosen) — sun-check that too.
        from device.sun_safety import is_sun_safe as _is_sun_safe

        sun_safe, sun_reason = _is_sun_safe(
            float(true_az) % 360.0,
            float(true_el),
        )
        if not sun_safe:
            with self._lock:
                self._errors.append(f"{sun_reason} (target {ts.label})")
                self._phase = "error"
            self._stop_evt.set()
            return

        with self._lock:
            self._target_az = pred_az_wrapped
            self._target_el = pred_el
        self._set_phase("slewing")
        if self.dry_run:
            with self._lock:
                self._encoder_az = pred_az_wrapped
                self._encoder_el = pred_el
            return

        # If a calibrate-motion session is running, hand the slew to it
        # so the same continuous-control primitive drives both pre-slew
        # and the operator's subsequent nudges. Wait briefly for the
        # mount to settle before transitioning to ``nudging``; the
        # motion loop never "arrives" so we cap the wait at ~30 s and
        # carry on regardless. The user can refine via nudges from
        # whatever the mount actually reached.
        motion = self._motion_delegate()
        if motion is not None:
            try:
                motion.set_target(pred_az_wrapped, pred_el)
            except Exception as exc:
                with self._lock:
                    self._errors.append(f"motion set_target failed: {exc}")
                return
            deadline = time.time() + 30.0
            while time.time() < deadline and not self._stop_evt.is_set():
                try:
                    if motion.is_settled(threshold_deg=0.05, ticks=4):
                        break
                except Exception:
                    break
                time.sleep(0.25)
            try:
                st = motion.status()
                if st.cur_cum_az_deg is not None and st.cur_el_deg is not None:
                    with self._lock:
                        self._encoder_az = float(st.cur_cum_az_deg)
                        self._encoder_el = float(st.cur_el_deg)
            except Exception:
                pass
            return

        try:
            from device.velocity_controller import move_to_ff

            loc = EarthLocation.from_geodetic(0, 0, 0)
            cur_el, cur_az = self._read_encoder_nonfatal(cli)
            if cur_el is None or cur_az is None:
                cur_el, cur_az = pred_el, pred_az_wrapped
            new_el, new_az, _ = move_to_ff(
                cli,
                target_az_deg=pred_az_wrapped,
                target_el_deg=pred_el,
                cur_az_deg=cur_az,
                cur_el_deg=cur_el,
                loc=loc,
                tag="[calibrate_web]",
                arrive_tolerance_deg=ARRIVE_TOL_SLEW_DEG,
            )
            with self._lock:
                self._encoder_az = new_az
                self._encoder_el = new_el
        except Exception as exc:
            with self._lock:
                self._errors.append(f"slew to {ts.label} failed: {exc}")

    # ---------- helpers ----------

    def _set_phase(self, phase: str) -> None:
        with self._lock:
            self._phase = phase

    def _motion_delegate(self):
        """Return the active calibrate-motion session for this telescope,
        or ``None`` if one is not running (or the module isn't importable
        from this test context). Lazy import keeps this module independent
        of ``device.calibrate_motion`` for unit tests that stub one but
        not the other.

        When non-None, motion is delegated through the live-tracker
        primitive's continuous-control loop (no arrive-tolerance), so
        sub-degree nudges actually reach the mount instead of being
        truncated by ``move_to_ff``'s 0.1° / 0.3° tolerances.
        """
        if self.dry_run:
            return None
        try:
            from device.calibrate_motion import get_calibrate_motion_manager
        except ImportError:
            return None
        try:
            mgr = get_calibrate_motion_manager()
        except Exception:
            return None
        try:
            session = mgr.get(self.telescope_id)
        except Exception:
            return None
        if session is None or not session.is_alive():
            return None
        return session

    def _read_encoder_nonfatal(self, cli) -> tuple[float | None, float | None]:
        if self.dry_run:
            with self._lock:
                return self._encoder_el, self._encoder_az
        try:
            from device.velocity_controller import measure_altaz_timed

            alt, az, _ = measure_altaz_timed(
                cli,
                EarthLocation.from_geodetic(0, 0, 0),
            )
            return float(alt), float(az)
        except Exception:
            return None, None

    def _poll_encoder_nonfatal(self, cli) -> None:
        el, az = self._read_encoder_nonfatal(cli)
        if el is None or az is None:
            return
        with self._lock:
            self._encoder_az = az
            self._encoder_el = el


# ---------- CalibrationManager ---------------------------------------


class CalibrationManager:
    """Process singleton keyed by telescope id. Mirrors
    :class:`device.live_tracker.LiveTrackManager`."""

    def __init__(self) -> None:
        self._sessions: dict[int, CalibrationSession] = {}
        self._lock = threading.Lock()

    def get(self, telescope_id: int) -> CalibrationSession | None:
        with self._lock:
            return self._sessions.get(int(telescope_id))

    def is_running(self, telescope_id: int) -> bool:
        s = self.get(telescope_id)
        return s is not None and s.is_alive()

    def start(self, session: CalibrationSession) -> CalibrationSession:
        tid = session.telescope_id
        # Hold the shared per-telescope start-lock across the entire
        # sequence (cross-check + registry write + session.start()) so
        # that concurrent CalibrationManager.start / LiveTrackManager.start
        # calls on the same scope cannot both pass their respective
        # cross-checks. Without this shared lock, each manager only
        # locks its own registry → TOCTOU between the two.
        from device._scope_start_lock import get_scope_start_lock

        with get_scope_start_lock(int(tid)):
            # Refuse if the live tracker is driving the same mount. The
            # import is lazy so tests that stub either module don't pull
            # the other unnecessarily.
            try:
                from device.live_tracker import get_manager as _get_tracker_mgr

                tracker = _get_tracker_mgr().get(tid)
                if tracker is not None and tracker.is_alive():
                    raise RuntimeError(f"telescope {tid} is live-tracking; stop first")
            except ImportError:
                pass
            with self._lock:
                existing = self._sessions.get(tid)
                if existing is not None and existing.is_alive():
                    raise RuntimeError(
                        f"telescope {tid} already calibrating; stop first"
                    )
                # Start the thread before publishing the session into the
                # registry, and keep both inside ``self._lock``. Otherwise a
                # concurrent ``stop(tid)`` can read the not-yet-started
                # session out of the registry, call ``session.stop()`` (which
                # only sets ``_stop_evt`` since no thread exists), and then
                # ``session.start()`` runs here, clears the stop event, and
                # spawns the thread anyway — the stop request is silently
                # dropped. Mirrors LiveTrackManager.start above.
                session.start()
                self._sessions[tid] = session
        return session

    def stop(self, telescope_id: int) -> CalibrationStatus | None:
        s = self.get(telescope_id)
        if s is None:
            return None
        s.stop()
        return s.status()

    def status(self, telescope_id: int) -> CalibrationStatus | None:
        s = self.get(telescope_id)
        return s.status() if s is not None else None


_MANAGER: CalibrationManager | None = None
_MANAGER_LOCK = threading.Lock()


def get_calibration_manager() -> CalibrationManager:
    """Process-level singleton. Matches the
    ``device.live_tracker.get_manager`` pattern."""
    global _MANAGER
    with _MANAGER_LOCK:
        if _MANAGER is None:
            _MANAGER = CalibrationManager()
        return _MANAGER
