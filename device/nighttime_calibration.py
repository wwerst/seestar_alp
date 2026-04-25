"""Nighttime calibration session — plate-solve a series of sky images
to fit the same 3-DOF mount rotation the daytime FAA-landmark workflow
fits.

Each sighting cycle:

1. Operator commands a slew to ``(commanded_az, commanded_el)`` (a
   pre-set sky position; the operator then nudges via the live-tracker
   continuous-control loop from PR #15 if needed).
2. Mount settles (motion session reports ``is_settled``).
3. Caller invokes :meth:`NighttimeCalibrationSession.capture_sighting`
   with an image path captured from the imager.
4. The session runs the plate solver in a background thread, converts
   the solved (RA, Dec) to topocentric (az, el) for the site + capture
   time, and stores the resulting ``(encoder_az_el, true_az_el)`` pair.
5. With ≥3 accepted sightings the session refits the rotation matrix
   via :func:`device.rotation_calibration.solve_rotation_from_pairs`.

If a plate solve fails (no solution / wildly-wrong FOV / timeout), the
caller can :meth:`skip_pending` to discard the latest cycle without
losing prior accepted sightings, then jog (PR #15 arrow keys / click-to-
go) to a clearer-sky neighbour and retry.

The session writes the same ``mount_calibration.json`` schema as the
daytime path, with ``calibration_method: "rotation_platesolve"`` so
downstream consumers (``MountFrame.from_calibration_json``, the live
tracker) pick it up unchanged.

Mutex: refuses to start while ``LiveTrackSession`` is alive on this
telescope. Allowed alongside ``CalibrateMotionSession``.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from device._atomic_json import write_atomic_json
from device.plate_solver import (
    PlateSolver,
    PlateSolverFailed,
    PlateSolverNotAvailable,
    S50_FOV_MAX_DEG,
    S50_FOV_MIN_DEG,
    SolveResult,
    get_default_plate_solver,
)
from device.rotation_calibration import (
    RotationSolution,
    solve_rotation_from_pairs,
)
from scripts.trajectory.observer import ObserverSite


# Minimum altitude (degrees) the mount may be asked to sight at. Below
# this, plate-solving the ground (or trees) is a waste; refuse the
# capture so the operator jogs to a clearer position.
MIN_SIGHTING_ALTITUDE_DEG = 10.0
# Maximum altitude. The az frame is degenerate near the pole, so we
# keep sightings out of the last few degrees.
MAX_SIGHTING_ALTITUDE_DEG = 80.0
# Need this many accepted sightings before ``apply()`` will write the
# calibration. The 3-DOF fit is ill-conditioned with fewer than 3 points
# spanning meaningful sky.
MIN_SIGHTINGS_FOR_APPLY = 3


# ---------- data model ------------------------------------------------


@dataclass(frozen=True)
class NighttimeSighting:
    """One plate-solved (commanded → true) sighting."""

    encoder_az_deg: float
    encoder_el_deg: float
    true_ra_deg: float
    true_dec_deg: float
    true_az_deg: float
    true_el_deg: float
    fov_x_deg: float
    fov_y_deg: float
    position_angle_deg: float
    image_path: str
    t_unix: float
    stars_used: int = 0


@dataclass
class PendingCapture:
    """A single capture currently being plate-solved. Polling the
    ``state`` endpoint shows ``status='solving'`` while this is active;
    success appends to ``sightings``, failure surfaces an error."""

    image_path: str
    encoder_az_deg: float
    encoder_el_deg: float
    t_started_unix: float
    status: str = "queued"  # "queued" | "solving" | "ok" | "fail" | "skipped"
    error: str | None = None


@dataclass
class NighttimeStatus:
    """JSON-serialisable session snapshot."""

    active: bool
    phase: str
    n_accepted: int
    min_required: int
    pending: dict | None
    last_failed: dict | None
    fit: dict | None
    sightings: list[dict] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


# ---------- coordinate conversion ------------------------------------


def radec_to_topocentric_azel(
    ra_deg: float,
    dec_deg: float,
    t_unix: float,
    site: ObserverSite,
) -> tuple[float, float]:
    """Convert ICRS (RA, Dec) → topocentric (az, el) for the site at the
    given UTC time. Uses astropy's AltAz transform — handles precession,
    nutation, atmospheric refraction (default sea-level NIST conditions),
    and aberration.

    Wrapped here so tests can monkey-patch this single function rather
    than mocking astropy's machinery.
    """
    from astropy import units as u
    from astropy.coordinates import AltAz, EarthLocation, SkyCoord
    from astropy.time import Time

    loc = EarthLocation.from_geodetic(
        lon=site.lon_deg * u.deg,
        lat=site.lat_deg * u.deg,
        height=site.alt_m * u.m,
    )
    sky = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
    altaz = sky.transform_to(AltAz(obstime=Time(t_unix, format="unix"), location=loc))
    return float(altaz.az.deg), float(altaz.alt.deg)


# ---------- session ---------------------------------------------------


class NighttimeCalibrationSession:
    """Stateful holder for one nighttime calibration run.

    Single-flight: one capture in flight at a time. The background
    solve thread is daemonic, joined on ``stop()``.
    """

    def __init__(
        self,
        telescope_id: int,
        site: ObserverSite,
        out_path: Path,
        *,
        plate_solver: PlateSolver | None = None,
        min_sightings: int = MIN_SIGHTINGS_FOR_APPLY,
    ):
        self.telescope_id = int(telescope_id)
        self.site = site
        self.out_path = Path(out_path)
        self.plate_solver = plate_solver or get_default_plate_solver()
        self.min_sightings = int(min_sightings)

        self._lock = threading.Lock()
        self._sightings: list[NighttimeSighting] = []
        self._solution: RotationSolution | None = None
        self._pending: PendingCapture | None = None
        self._last_failed: PendingCapture | None = None
        self._errors: list[str] = []
        self._phase = "idle"
        # Set when ``stop()`` is called; cancels any in-flight solve.
        self._stop_evt = threading.Event()
        self._solve_thread: threading.Thread | None = None
        self._active = True

    # ---------- lifecycle ----------

    def stop(self) -> None:
        with self._lock:
            self._active = False
            self._phase = "stopped"
        self._stop_evt.set()
        # Don't join here; the daemon thread will exit on its own when
        # the solve completes (and we cancel further work via the event
        # check). Caller should not invoke any further methods after stop.

    def is_active(self) -> bool:
        with self._lock:
            return self._active

    # ---------- snapshot ----------

    def status(self) -> NighttimeStatus:
        with self._lock:
            sol_dict = None
            if self._solution is not None:
                sol_dict = {
                    "yaw_deg": self._solution.yaw_deg,
                    "pitch_deg": self._solution.pitch_deg,
                    "roll_deg": self._solution.roll_deg,
                    "residual_rms_deg": self._solution.residual_rms_deg,
                    "per_record": list(self._solution.per_landmark),
                }
            pending_dict = None
            if self._pending is not None:
                pending_dict = {
                    "image_path": self._pending.image_path,
                    "encoder_az_deg": self._pending.encoder_az_deg,
                    "encoder_el_deg": self._pending.encoder_el_deg,
                    "t_started_unix": self._pending.t_started_unix,
                    "elapsed_s": time.time() - self._pending.t_started_unix,
                    "status": self._pending.status,
                    "error": self._pending.error,
                }
            last_failed_dict = None
            if self._last_failed is not None:
                last_failed_dict = {
                    "image_path": self._last_failed.image_path,
                    "encoder_az_deg": self._last_failed.encoder_az_deg,
                    "encoder_el_deg": self._last_failed.encoder_el_deg,
                    "status": self._last_failed.status,
                    "error": self._last_failed.error,
                }
            sightings_list = [
                {
                    "encoder_az_deg": s.encoder_az_deg,
                    "encoder_el_deg": s.encoder_el_deg,
                    "true_ra_deg": s.true_ra_deg,
                    "true_dec_deg": s.true_dec_deg,
                    "true_az_deg": s.true_az_deg,
                    "true_el_deg": s.true_el_deg,
                    "fov_x_deg": s.fov_x_deg,
                    "fov_y_deg": s.fov_y_deg,
                    "position_angle_deg": s.position_angle_deg,
                    "image_path": s.image_path,
                    "t_unix": s.t_unix,
                    "stars_used": s.stars_used,
                }
                for s in self._sightings
            ]
            return NighttimeStatus(
                active=self._active,
                phase=self._phase,
                n_accepted=len(self._sightings),
                min_required=self.min_sightings,
                pending=pending_dict,
                last_failed=last_failed_dict,
                fit=sol_dict,
                sightings=sightings_list,
                errors=list(self._errors),
            )

    # ---------- capture ----------

    def capture_sighting(
        self,
        image_path: Path | str,
        encoder_az_deg: float,
        encoder_el_deg: float,
    ) -> None:
        """Queue a plate solve on ``image_path`` for the given encoder
        position. Background-threaded. Caller polls :meth:`status` for
        completion.

        Raises if a previous solve is still in flight (single-flight) or
        if the encoder position is below the altitude floor.
        """
        if encoder_el_deg < MIN_SIGHTING_ALTITUDE_DEG:
            raise ValueError(
                f"encoder el {encoder_el_deg:.2f}° below "
                f"{MIN_SIGHTING_ALTITUDE_DEG:.0f}° altitude floor"
            )
        if encoder_el_deg > MAX_SIGHTING_ALTITUDE_DEG:
            raise ValueError(
                f"encoder el {encoder_el_deg:.2f}° above "
                f"{MAX_SIGHTING_ALTITUDE_DEG:.0f}° (az ill-conditioned at zenith)"
            )
        with self._lock:
            if self._pending is not None and self._pending.status in (
                "queued",
                "solving",
            ):
                raise RuntimeError("a plate-solve is already in flight; wait")
            self._pending = PendingCapture(
                image_path=str(image_path),
                encoder_az_deg=float(encoder_az_deg),
                encoder_el_deg=float(encoder_el_deg),
                t_started_unix=time.time(),
                status="queued",
            )
            self._phase = "solving"
        self._solve_thread = threading.Thread(
            target=self._solve_worker,
            name=f"NighttimePlateSolve({self.telescope_id})",
            daemon=True,
        )
        self._solve_thread.start()

    def skip_pending(self) -> None:
        """Discard a pending or recently-failed capture. The accepted
        sighting list is unchanged."""
        with self._lock:
            if self._pending is not None and self._pending.status in (
                "fail",
                "ok",
            ):
                # Keep last_failed in place so the UI can still show the
                # diagnostic; just clear the pending slot.
                if self._pending.status == "fail":
                    self._last_failed = self._pending
                self._pending = None
            elif self._pending is not None:
                # In-flight solve: mark cancelled. The worker will see
                # _stop_evt is set (or detect _pending=None on exit).
                self._pending.status = "skipped"
                self._last_failed = self._pending
                self._pending = None
            self._phase = "idle"

    def remove_sighting(self, idx: int) -> None:
        """Remove an accepted sighting and refit. Used when the operator
        spots a bad fit row and wants to drop it rather than re-shoot."""
        with self._lock:
            if not (0 <= idx < len(self._sightings)):
                raise IndexError(f"sighting idx {idx} out of range")
            del self._sightings[idx]
        self._refit_locked()

    def apply(self) -> None:
        """Persist the current fit to the calibration JSON. Refuses if
        we don't have ``min_sightings`` accepted records."""
        with self._lock:
            if len(self._sightings) < self.min_sightings:
                raise ValueError(
                    f"need ≥{self.min_sightings} sightings; have {len(self._sightings)}"
                )
            if self._solution is None:
                raise ValueError("no solution yet; capture more sightings")
            sol = self._solution
            sightings_snapshot = list(self._sightings)
        payload = self._build_payload(sol, sightings_snapshot)
        write_atomic_json(self.out_path, payload, indent=2)
        with self._lock:
            self._phase = "committed"

    # ---------- internals ----------

    def _solve_worker(self) -> None:
        with self._lock:
            pending = self._pending
            if pending is None:
                return
            pending.status = "solving"

        # Run the (possibly slow) solver outside the lock so other
        # methods like status() stay responsive.
        try:
            solve_result = self.plate_solver.solve(Path(pending.image_path))
        except PlateSolverNotAvailable as exc:
            self._record_failure(pending, str(exc))
            return
        except PlateSolverFailed as exc:
            self._record_failure(pending, str(exc))
            return
        except FileNotFoundError as exc:
            self._record_failure(pending, str(exc))
            return
        except Exception as exc:
            self._record_failure(pending, f"unexpected solver error: {exc}")
            return

        # FOV sanity check.
        fx = solve_result.fov_x_deg
        fy = solve_result.fov_y_deg
        if not (
            S50_FOV_MIN_DEG <= fx <= S50_FOV_MAX_DEG
            and S50_FOV_MIN_DEG <= fy <= S50_FOV_MAX_DEG
        ):
            self._record_failure(
                pending,
                f"solver returned FOV {fx:.2f}×{fy:.2f}° outside "
                f"[{S50_FOV_MIN_DEG}, {S50_FOV_MAX_DEG}]°",
            )
            return

        # Convert (RA, Dec) → topocentric (az, el).
        try:
            true_az, true_el = radec_to_topocentric_azel(
                solve_result.ra_deg,
                solve_result.dec_deg,
                pending.t_started_unix,
                self.site,
            )
        except Exception as exc:
            self._record_failure(pending, f"radec→azel failed: {exc}")
            return

        sighting = NighttimeSighting(
            encoder_az_deg=pending.encoder_az_deg,
            encoder_el_deg=pending.encoder_el_deg,
            true_ra_deg=solve_result.ra_deg,
            true_dec_deg=solve_result.dec_deg,
            true_az_deg=true_az,
            true_el_deg=true_el,
            fov_x_deg=fx,
            fov_y_deg=fy,
            position_angle_deg=solve_result.position_angle_deg,
            image_path=pending.image_path,
            t_unix=pending.t_started_unix,
            stars_used=solve_result.stars_used,
        )
        with self._lock:
            self._sightings.append(sighting)
            pending.status = "ok"
            self._pending = None
            self._phase = "fit_pending"
        self._refit_locked()

    def _record_failure(self, pending: PendingCapture, reason: str) -> None:
        with self._lock:
            pending.status = "fail"
            pending.error = reason
            self._last_failed = pending
            self._pending = None
            self._phase = "fail_pending"
            self._errors.append(reason)

    def _refit_locked(self) -> None:
        # Snapshot under lock, fit outside.
        with self._lock:
            sightings = list(self._sightings)
        if len(sightings) < 1:
            with self._lock:
                self._solution = None
                self._phase = "idle"
            return
        try:
            sol = solve_rotation_from_pairs(
                [
                    (
                        s.encoder_az_deg,
                        s.encoder_el_deg,
                        s.true_az_deg,
                        s.true_el_deg,
                    )
                    for s in sightings
                ],
            )
        except Exception as exc:
            with self._lock:
                self._errors.append(f"refit failed: {exc}")
            return
        with self._lock:
            self._solution = sol
            self._phase = (
                "ready_to_apply"
                if len(self._sightings) >= self.min_sightings
                else "fit_pending"
            )

    def _build_payload(
        self,
        sol: RotationSolution,
        sightings: list[NighttimeSighting],
    ) -> dict:
        """Build the same JSON schema the daytime path writes, with
        ``calibration_method`` flipped to ``rotation_platesolve`` and
        the per-record list extended with platesolve-specific fields."""
        return {
            "calibration_method": "rotation_platesolve",
            "calibrated_at": time.strftime("%Y-%m-%dT%H-%M-%S%z"),
            "yaw_offset_deg": sol.yaw_deg,
            "pitch_offset_deg": sol.pitch_deg,
            "roll_offset_deg": sol.roll_deg,
            "origin_offset_ecef_m": [0.0, 0.0, 0.0],
            "residual_rms_deg": sol.residual_rms_deg,
            "n_sightings": len(sightings),
            "observer": {
                "lat_deg": self.site.lat_deg,
                "lon_deg": self.site.lon_deg,
                "alt_m": self.site.alt_m,
                "source": "telescope_get_device_state",
            },
            # Per-sighting records carry both the solver output and the
            # fit residuals (solver's per-record list already has
            # encoder, true, predicted, residual).
            "sightings": [
                {
                    "encoder_az_deg": s.encoder_az_deg,
                    "encoder_el_deg": s.encoder_el_deg,
                    "true_ra_deg": s.true_ra_deg,
                    "true_dec_deg": s.true_dec_deg,
                    "true_az_deg": s.true_az_deg,
                    "true_el_deg": s.true_el_deg,
                    "fov_x_deg": s.fov_x_deg,
                    "fov_y_deg": s.fov_y_deg,
                    "position_angle_deg": s.position_angle_deg,
                    "image_path": s.image_path,
                    "t_unix": s.t_unix,
                    "stars_used": s.stars_used,
                }
                for s in sightings
            ],
            "fit_per_record": list(sol.per_landmark),
        }


# ---------- manager ---------------------------------------------------


class NighttimeCalibrationManager:
    """Singleton-per-process registry, telescope-keyed. Mirrors
    ``CalibrationManager`` and ``CalibrateMotionManager``."""

    def __init__(self) -> None:
        self._sessions: dict[int, NighttimeCalibrationSession] = {}
        self._lock = threading.Lock()

    def get(self, telescope_id: int) -> NighttimeCalibrationSession | None:
        with self._lock:
            return self._sessions.get(int(telescope_id))

    def is_running(self, telescope_id: int) -> bool:
        s = self.get(telescope_id)
        return s is not None and s.is_active()

    def start(
        self, session: NighttimeCalibrationSession
    ) -> NighttimeCalibrationSession:
        tid = int(session.telescope_id)
        # Refuse if the live tracker is driving the same mount.
        try:
            from device.live_tracker import get_manager as _get_tracker_mgr

            tracker = _get_tracker_mgr().get(tid)
            if tracker is not None and tracker.is_alive():
                raise RuntimeError(
                    f"telescope {tid} is live-tracking; stop the tracker first"
                )
        except ImportError:
            pass
        with self._lock:
            existing = self._sessions.get(tid)
            if existing is not None and existing.is_active():
                raise RuntimeError(
                    f"telescope {tid} is already in nighttime calibration mode"
                )
            self._sessions[tid] = session
        return session

    def stop(self, telescope_id: int) -> NighttimeStatus | None:
        s = self.get(telescope_id)
        if s is None:
            return None
        s.stop()
        return s.status()

    def status(self, telescope_id: int) -> NighttimeStatus | None:
        s = self.get(telescope_id)
        return s.status() if s is not None else None


_MANAGER: NighttimeCalibrationManager | None = None
_MANAGER_LOCK = threading.Lock()


def get_nighttime_manager() -> NighttimeCalibrationManager:
    global _MANAGER
    with _MANAGER_LOCK:
        if _MANAGER is None:
            _MANAGER = NighttimeCalibrationManager()
        return _MANAGER


# Silence unused-import warnings.
_ = (math, SolveResult)
