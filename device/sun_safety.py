"""Sun-avoidance safety primitives.

Pure helpers + dataclass used by the `SunSafetyMonitor` (see follow-up
module) and by the pre-flight guards in `live_tracker`,
`rotation_calibration`, and `seestar_device`. Kept dependency-free
beyond `ephem` so tests can run without a mount or Alpaca.

Conventions:
- Az is measured east of north, in degrees, in the range [0, 360).
- Altitude (el) is measured from the horizon, in degrees, in [-90, 90].
- "Pointing" is the (az, el) of the optical axis; "sun" is the (az, alt)
  of the sun's center as seen from the observer at a given instant.

The sun-altitude threshold defaults to -10° (sun must be at least 10°
below the horizon to short-circuit the check). The exclusion cone
defaults to 30° angular separation between the optical axis and the
sun's center.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Optional

import ephem

from device.geometry import angular_separation_deg, wrap_pm180


logger = logging.getLogger(__name__)


# Defaults match `[sun_avoidance]` in `device/config.toml`. Re-read via
# `Config` at startup; these constants exist so the helpers stay usable
# from contexts without a populated Config (tests, scripts).
DEFAULT_MIN_SEPARATION_DEG = 30.0
DEFAULT_ALT_THRESHOLD_DEG = -10.0


@dataclass(frozen=True)
class SafetyTrip:
    """Snapshot of a sun-safety violation, displayed in the UI banner."""

    when_utc: datetime
    sun_az_deg: float
    sun_alt_deg: float
    mount_az_deg: float
    mount_el_deg: float
    separation_deg: float
    cone_deg: float
    jog_angle_deg: int
    jog_speed: int
    jog_duration_s: int
    message: str = (
        "Sun safety triggered: mount jogged away from sun and tracking aborted."
    )


@dataclass
class _Site:
    lat_deg: float
    lon_deg: float


def _site_from_config_or(lat_deg: Optional[float], lon_deg: Optional[float]) -> _Site:
    """Resolve site lat/lon, falling back to `Config` if not supplied.

    Imported lazily so this module is importable in environments where
    `device.config` cannot be loaded (e.g. minimal test contexts).
    """
    if lat_deg is not None and lon_deg is not None:
        return _Site(float(lat_deg), float(lon_deg))
    from device.config import Config

    return _Site(float(Config.init_lat), float(Config.init_long))


# Both lat and lon within this many degrees of 0 is the "location not set"
# sentinel: Config.init_lat/init_long default to 0/0, and no real observer
# sits within ~0.1 m of Null Island. When the site is unset we cannot trust
# the computed sun position, so the guard must fail CLOSED rather than open
# (see `is_sun_safe` and `SunSafetyMonitor._tick`).
_UNSET_LOCATION_EPS_DEG = 1e-6


def _location_is_unset(lat_deg: float, lon_deg: float) -> bool:
    return (
        abs(lat_deg) < _UNSET_LOCATION_EPS_DEG
        and abs(lon_deg) < _UNSET_LOCATION_EPS_DEG
    )


def angular_separation(
    a_az_deg: float, a_el_deg: float, b_az_deg: float, b_el_deg: float
) -> float:
    """Great-circle angular separation between two (az, el) directions.

    Returns degrees in [0, 180]. Public name kept for existing callers;
    delegates to :func:`device.geometry.angular_separation_deg`, the single
    source of truth for this math.
    """
    return angular_separation_deg(a_az_deg, a_el_deg, b_az_deg, b_el_deg)


def compute_sun_altaz(
    *,
    lat_deg: Optional[float] = None,
    lon_deg: Optional[float] = None,
    when: Optional[datetime] = None,
) -> tuple[float, float]:
    """Sun (az, alt) in degrees as seen from the given site at `when`.

    `when` defaults to UTC now. `lat_deg` / `lon_deg` default to the
    Config-configured observer site. Uses the `ephem` library, which
    is already a project dependency (see `front/app.py`).
    """
    site = _site_from_config_or(lat_deg, lon_deg)
    obs = ephem.Observer()
    obs.lat = str(site.lat_deg)
    obs.lon = str(site.lon_deg)
    if when is None:
        when = datetime.now(tz=timezone.utc)
    elif when.tzinfo is None:
        # Treat naive datetimes as UTC for predictability.
        when = when.replace(tzinfo=timezone.utc)
    # ephem expects naive UTC; strip tzinfo after converting.
    obs.date = when.astimezone(timezone.utc).replace(tzinfo=None)
    sun = ephem.Sun()
    sun.compute(obs)
    return math.degrees(float(sun.az)), math.degrees(float(sun.alt))


def is_sun_safe(
    target_az_deg: float,
    target_el_deg: float,
    *,
    lat_deg: Optional[float] = None,
    lon_deg: Optional[float] = None,
    when: Optional[datetime] = None,
    min_separation_deg: float = DEFAULT_MIN_SEPARATION_DEG,
    alt_threshold_deg: float = DEFAULT_ALT_THRESHOLD_DEG,
) -> tuple[bool, str]:
    """Return ``(safe, reason)`` for pointing the optical axis at ``(az, el)``.

    Always safe when the sun is below ``alt_threshold_deg`` (default
    -10°). Otherwise returns False if the angular separation between
    the pointing and the sun is below ``min_separation_deg``.

    The ``reason`` string is empty when safe and includes the numbers
    (separation, cone, sun alt) when unsafe so callers can log it.

    Fails CLOSED when the observer site is the unset 0,0 sentinel: the sun
    position cannot be trusted there (the real sun may be well up while the
    Null-Island computation puts it below the threshold), so we refuse the
    motion and tell the operator to set their location rather than silently
    allowing a sun-pointing slew.
    """
    site = _site_from_config_or(lat_deg, lon_deg)
    if _location_is_unset(site.lat_deg, site.lon_deg):
        return False, (
            "sun_avoidance: observer location is not set (lat/long = 0,0); "
            "set your site location before slewing so the sun position can "
            "be computed"
        )
    sun_az, sun_alt = compute_sun_altaz(
        lat_deg=site.lat_deg,
        lon_deg=site.lon_deg,
        when=when,
    )
    if sun_alt < alt_threshold_deg:
        return True, ""
    sep = angular_separation(target_az_deg, target_el_deg, sun_az, sun_alt)
    if sep < min_separation_deg:
        return False, (
            f"sun_avoidance: separation {sep:.1f}° < cone {min_separation_deg:.1f}° "
            f"(sun alt {sun_alt:.1f}°, sun az {sun_az:.1f}°)"
        )
    return True, ""


class SunSafetyLocked(RuntimeError):
    """Raised by the speed_move wrapper while the emergency jog is in progress.

    Tracking / calibration loops should catch this, log it, and exit
    cleanly — do NOT keep retrying. The monitor owns the mount while the
    lockout event is set.
    """


# Firmware speed→rate constant (mirrors device.velocity_controller).
# Duplicated here so the monitor module has no import-time dependency on
# the velocity controller (which pulls in astropy and other heavy deps).
_SPEED_PER_DEG_PER_SEC = 237.0


def compute_jog_angle(
    mount_az_deg: float,
    mount_el_deg: float,
    sun_az_deg: float,
    sun_alt_deg: float,
    *,
    jog_speed: int = 1440,
    jog_duration_s: float = 6.0,
    min_separation_deg: float = DEFAULT_MIN_SEPARATION_DEG,
    safety_margin_deg: float = 5.0,
) -> int:
    """Pick the `speed_move` angle that drives the mount AWAY from the sun.

    Angle convention (matches firmware + `streaming_controller.track`):
    - 0°   = pure +azimuth motion
    - 90°  = pure +elevation motion
    - 180° = pure -azimuth motion
    - 270° = pure -elevation motion

    Algorithm — two-pass selection over candidate directions
    [primary-reverse-of-sun, +el (90°), −el (270°), +az (0°), −az (180°)]:
      1. First, return the highest-priority candidate whose forward-
         simulated separation from the sun reaches
         ``min_separation_deg + safety_margin_deg`` (absolute cone-exit
         with cushion). With the operational defaults (jog_speed=1440 ≈
         6°/s × 6 s = 36° step against a 30° cone + 5° margin), the
         primary direction satisfies this from anywhere inside the cone.
      2. If no candidate clears the cone (shorter jog or worse geometry,
         e.g. very near a pole), return the candidate with the LARGEST
         predicted separation. The function never refuses; the monitor
         re-trips on the next tick if still inside the cone.

    Returns an integer angle in [0, 360).
    """
    daz_diff = wrap_pm180(sun_az_deg - mount_az_deg)
    del_diff = sun_alt_deg - mount_el_deg
    norm = math.hypot(daz_diff, del_diff)

    rate = jog_speed / _SPEED_PER_DEG_PER_SEC  # deg/s
    step = rate * jog_duration_s  # total motion in degrees
    target_sep = min_separation_deg + safety_margin_deg

    def _new_sep(angle_deg: int) -> float:
        """Forward-sim this angle; return predicted separation in degrees."""
        rad = math.radians(angle_deg)
        new_az = (mount_az_deg + step * math.cos(rad)) % 360.0
        new_el = max(-90.0, min(90.0, mount_el_deg + step * math.sin(rad)))
        return angular_separation(new_az, new_el, sun_az_deg, sun_alt_deg)

    # Build candidate list in priority order. Primary first when the
    # direction-to-sun is well-defined; otherwise fall straight to the
    # axial fallbacks.
    candidates: list[int] = []
    if norm > 1e-6:
        primary = int(
            round((math.degrees(math.atan2(-del_diff, -daz_diff)) + 360.0) % 360.0)
        )
        candidates.append(primary)
    candidates.extend([90, 270, 0, 180])

    # Pass 1: first candidate that clears the cone with margin.
    for c in candidates:
        if _new_sep(c) >= target_sep:
            return c

    # Pass 2: best-effort — pick the candidate with the largest predicted
    # separation. Always at least matches every other candidate, so this
    # cannot decrease separation below the geometric maximum reachable
    # within one jog.
    best_angle = candidates[0]
    best_sep = _new_sep(best_angle)
    for c in candidates[1:]:
        s = _new_sep(c)
        if s > best_sep:
            best_angle = c
            best_sep = s
    return best_angle


# ---------- SunSafetyMonitor ---------------------------------------------


# Signature of a "raw mount altaz reader". Returns sky (az_deg, alt_deg) or
# None if the reading is unavailable / untrustworthy this tick (e.g. mount
# disconnected, not plate-solved). Factoring this out keeps the monitor
# independent of AlpacaClient/astropy at import time and testable with a
# fake.
AltazReader = Callable[[], Optional[tuple[float, float]]]

# Signature of a "raw jog commander". Receives (speed, angle, dur_sec)
# and commands the firmware to execute one `scope_speed_move` burst.
# Intentionally bypasses any lockout-aware wrapper — the monitor is the
# one source authorized to move during the emergency window.
RawJogCommand = Callable[[int, int, int], None]


class SunSafetyMonitor:
    """Always-on daemon that trips when the mount points inside the sun cone.

    Started once per process (see `device/live_tracker_service.py`). Two
    cadences: while the sun is below `alt_threshold_deg` it polls slowly
    (default 60 s); when the sun rises above the threshold it polls at
    the active cadence (default 2 s) and compares the mount's sky
    pointing against the sun.

    On a violation it:
      1. Sets the emergency lockout event (blocks the wrapped speed_move).
      2. Runs one jog at `jog_speed` / `jog_duration_s` in the direction
         picked by `compute_jog_angle` — issued FIRST so the mount leaves
         the cone immediately, before any slow teardown.
      3. Calls `abort_active()` to stop in-flight tracking/calibration
         (which may join worker threads and take several seconds).
      4. Sleeps for jog_duration_s + margin, then clears the lockout
         so the user can drive the mount again.
      5. Leaves `last_trip` populated until the UI POSTs dismiss.
    """

    def __init__(
        self,
        *,
        altaz_reader: AltazReader,
        jog_command: RawJogCommand,
        abort_active: Optional[Callable[[], None]] = None,
        lat_deg: Optional[float] = None,
        lon_deg: Optional[float] = None,
        min_separation_deg: float = DEFAULT_MIN_SEPARATION_DEG,
        alt_threshold_deg: float = DEFAULT_ALT_THRESHOLD_DEG,
        jog_speed: int = 1440,
        jog_duration_s: int = 6,
        tick_interval_active_s: float = 2.0,
        tick_interval_dormant_s: float = 60.0,
        enabled: bool = True,
        jog_telescope_id: Optional[int] = None,
    ) -> None:
        self._altaz_reader = altaz_reader
        self._jog_command = jog_command
        self._abort_active = abort_active
        # Which telescope the jog_command drives (the monitor senses and
        # jogs a single scope). None = unknown -> jog-window queries match
        # every telescope (conservative).
        self._jog_telescope_id = (
            int(jog_telescope_id) if jog_telescope_id is not None else None
        )
        self._lat_deg = lat_deg
        self._lon_deg = lon_deg

        self._min_separation_deg = float(min_separation_deg)
        self._alt_threshold_deg = float(alt_threshold_deg)
        self._jog_speed = int(jog_speed)
        self._jog_duration_s = int(jog_duration_s)
        self._tick_active = float(tick_interval_active_s)
        self._tick_dormant = float(tick_interval_dormant_s)
        self._enabled = bool(enabled)

        self._stop_evt = threading.Event()
        self._emergency_lockout = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._last_trip: Optional[SafetyTrip] = None
        self._trip_dismissed: bool = False
        self._lock = threading.Lock()
        # End of the currently-executing emergency jog window (0 = none).
        # Guarded by _lock; see is_jog_in_progress().
        self._jog_until_ts: float = 0.0
        # Bounded re-jog attempts when a session teardown's direct motor
        # stop truncates the jog; the periodic tick is the final backstop.
        self._max_jog_attempts: int = 3
        # Timestamp of the last "location not set" hard warning; rate-limits
        # the log so a blind monitor doesn't flood at the active cadence.
        self._last_unset_warn_ts: float = 0.0

    # ---------- lifecycle ----------

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_evt.clear()
        self._thread = threading.Thread(
            target=self._loop,
            name="SunSafetyMonitor",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "SunSafetyMonitor started: cone=%.1f° alt_thr=%.1f° jog=speed=%d dur=%ds",
            self._min_separation_deg,
            self._alt_threshold_deg,
            self._jog_speed,
            self._jog_duration_s,
        )

    def stop(self, timeout: float = 5.0) -> None:
        self._stop_evt.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def reload(
        self,
        *,
        min_separation_deg: Optional[float] = None,
        alt_threshold_deg: Optional[float] = None,
        jog_speed: Optional[int] = None,
        jog_duration_s: Optional[int] = None,
        enabled: Optional[bool] = None,
    ) -> None:
        """Update thresholds in place without stopping the monitor thread."""
        with self._lock:
            if min_separation_deg is not None:
                self._min_separation_deg = float(min_separation_deg)
            if alt_threshold_deg is not None:
                self._alt_threshold_deg = float(alt_threshold_deg)
            if jog_speed is not None:
                self._jog_speed = int(jog_speed)
            if jog_duration_s is not None:
                self._jog_duration_s = int(jog_duration_s)
            if enabled is not None:
                self._enabled = bool(enabled)

    # ---------- public state ----------

    def is_locked_out(self) -> bool:
        return self._emergency_lockout.is_set()

    def last_trip(self) -> Optional[SafetyTrip]:
        with self._lock:
            if self._trip_dismissed:
                return None
            return self._last_trip

    def dismiss_last_trip(self) -> None:
        with self._lock:
            self._trip_dismissed = True

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def min_separation_deg(self) -> float:
        return self._min_separation_deg

    # ---------- loop ----------

    def _loop(self) -> None:
        while not self._stop_evt.is_set():
            try:
                self._tick()
            except Exception:
                logger.exception("SunSafetyMonitor tick raised")
            # Sleep interval depends on whether sun is above the
            # activation threshold at the end of the tick. Cheap to
            # recompute.
            try:
                _, sun_alt = compute_sun_altaz(
                    lat_deg=self._lat_deg,
                    lon_deg=self._lon_deg,
                )
                wait = (
                    self._tick_active
                    if sun_alt >= self._alt_threshold_deg
                    else self._tick_dormant
                )
            except Exception:
                wait = self._tick_active
            self._stop_evt.wait(timeout=wait)

    def _warn_location_unset(self) -> None:
        """Log a rate-limited hard warning that the guard is blind."""
        now = time.time()
        if now - self._last_unset_warn_ts < 300.0:
            return
        self._last_unset_warn_ts = now
        logger.error(
            "SUN SAFETY BLIND: observer location is not set (lat/long = 0,0). "
            "The sun-avoidance monitor cannot compute the sun position and is "
            "NOT protecting the mount — set your site location in config."
        )

    def _tick(self) -> None:
        if not self._enabled or self._emergency_lockout.is_set():
            return
        site = _site_from_config_or(self._lat_deg, self._lon_deg)
        if _location_is_unset(site.lat_deg, site.lon_deg):
            # Fail closed: without a site we cannot trust the sun position.
            # We can't compute a jog direction either, so log hard and skip;
            # the is_sun_safe pre-flights refuse motion in the meantime.
            self._warn_location_unset()
            return
        sun_az, sun_alt = compute_sun_altaz(
            lat_deg=site.lat_deg,
            lon_deg=site.lon_deg,
        )
        if sun_alt < self._alt_threshold_deg:
            return
        try:
            altaz = self._altaz_reader()
        except Exception:
            logger.warning("altaz_reader failed this tick", exc_info=True)
            return
        if altaz is None:
            return
        mount_az, mount_el = altaz
        sep = angular_separation(mount_az, mount_el, sun_az, sun_alt)
        if sep >= self._min_separation_deg:
            return
        self._trigger_emergency(mount_az, mount_el, sun_az, sun_alt, sep)

    def _trigger_emergency(
        self,
        mount_az: float,
        mount_el: float,
        sun_az: float,
        sun_alt: float,
        sep: float,
    ) -> None:
        jog_angle = compute_jog_angle(
            mount_az,
            mount_el,
            sun_az,
            sun_alt,
            jog_speed=self._jog_speed,
            jog_duration_s=float(self._jog_duration_s),
        )
        trip = SafetyTrip(
            when_utc=datetime.now(timezone.utc),
            sun_az_deg=sun_az,
            sun_alt_deg=sun_alt,
            mount_az_deg=mount_az,
            mount_el_deg=mount_el,
            separation_deg=sep,
            cone_deg=self._min_separation_deg,
            jog_angle_deg=jog_angle,
            jog_speed=self._jog_speed,
            jog_duration_s=self._jog_duration_s,
        )
        logger.error(
            "SUN SAFETY TRIP: sep=%.1f° < cone=%.1f° "
            "(mount az=%.1f° el=%.1f°, sun az=%.1f° alt=%.1f°) — "
            "jogging at speed=%d angle=%d° for %ds",
            sep,
            self._min_separation_deg,
            mount_az,
            mount_el,
            sun_az,
            sun_alt,
            self._jog_speed,
            jog_angle,
            self._jog_duration_s,
        )
        with self._lock:
            self._last_trip = trip
            self._trip_dismissed = False

        # 1. Lock out lockout-aware speed_move calls from tracker/calibration.
        self._emergency_lockout.set()
        try:
            attempts = 0
            while True:
                attempts += 1
                # 2. Issue the jog FIRST (raw path — bypasses the wrapper) so
                #    the mount leaves the cone immediately. Doing this before
                #    abort_active() matters: abort joins worker threads
                #    (seconds each), and we must not leave the optics pointed
                #    at the sun for that whole window. Publish the jog window
                #    so session-exit direct stops know not to cancel it.
                with self._lock:
                    self._jog_until_ts = time.time() + self._jog_duration_s + 0.5
                try:
                    self._jog_command(
                        self._jog_speed,
                        jog_angle,
                        self._jog_duration_s,
                    )
                except Exception:
                    logger.exception("jog_command raised — NOT retrying")
                # 3. Stop any active session (first pass only). Its teardown
                #    stop commands run on the raw channel and may land AFTER
                #    our jog command, truncating it — that's why we re-sense
                #    and re-jog below once the sessions are gone.
                if attempts == 1 and self._abort_active is not None:
                    try:
                        self._abort_active()
                    except Exception:
                        logger.exception("abort_active callback failed")
                # 4. Wait for the jog to complete, plus a small margin so any
                #    caller that races back in can see us already done.
                time.sleep(self._jog_duration_s + 0.5)
                # 5. Verify we actually left the cone: a session teardown's
                #    direct motor-stop can have cancelled the in-flight jog.
                #    With sessions now aborted, a re-jog runs uncontested.
                if attempts >= self._max_jog_attempts:
                    logger.error(
                        "SUN SAFETY: still inside the cone after %d jog "
                        "attempt(s) — releasing lockout; monitor will re-trip "
                        "next tick",
                        attempts,
                    )
                    break
                sep_now, rejog_angle = self._resense_for_rejog()
                if sep_now is None or sep_now >= self._min_separation_deg:
                    break
                logger.error(
                    "SUN SAFETY: jog truncated (sep=%.1f° < cone=%.1f°) — "
                    "re-jogging (attempt %d)",
                    sep_now,
                    self._min_separation_deg,
                    attempts + 1,
                )
                if rejog_angle is not None:
                    jog_angle = rejog_angle
        finally:
            with self._lock:
                self._jog_until_ts = 0.0
            # 6. Release the lockout so the user can drive the mount.
            self._emergency_lockout.clear()
        logger.info("SUN SAFETY jog complete — user has control")

    def _resense_for_rejog(self) -> tuple[Optional[float], Optional[int]]:
        """Re-read mount pointing + sun position after a jog.

        Returns ``(separation_deg, next_jog_angle)``; ``(None, None)`` when
        either reading fails — the caller treats that as "cannot verify" and
        exits the jog loop (the periodic tick re-trips if we're still unsafe).
        """
        try:
            altaz = self._altaz_reader()
        except Exception:
            logger.warning("altaz_reader failed during jog verify", exc_info=True)
            return None, None
        if altaz is None:
            return None, None
        site = _site_from_config_or(self._lat_deg, self._lon_deg)
        try:
            sun_az, sun_alt = compute_sun_altaz(
                lat_deg=site.lat_deg,
                lon_deg=site.lon_deg,
            )
        except Exception:
            logger.warning("sun ephemeris failed during jog verify", exc_info=True)
            return None, None
        mount_az, mount_el = altaz
        sep = angular_separation(mount_az, mount_el, sun_az, sun_alt)
        angle = compute_jog_angle(
            mount_az,
            mount_el,
            sun_az,
            sun_alt,
            jog_speed=self._jog_speed,
            jog_duration_s=float(self._jog_duration_s),
        )
        return sep, angle

    def is_jog_in_progress(self, telescope_id: Optional[int] = None) -> bool:
        """True while an emergency jog command is executing on the mount.

        Session-exit direct motor-stops (which deliberately bypass the
        lockout-aware wrapper) consult this so they don't cancel the
        in-flight jog — the jog's own firmware dur_sec bounds it, so
        skipping the stop cannot leave the motor running.

        The monitor jogs exactly one scope (``jog_telescope_id``). Pass the
        telescope the stop targets so a jog on the primary does not suppress
        motor stops on OTHER mounts; ``None`` (caller's scope unknown) and an
        unknown jog scope both match conservatively.
        """
        with self._lock:
            if time.time() >= self._jog_until_ts:
                return False
            return (
                telescope_id is None
                or self._jog_telescope_id is None
                or int(telescope_id) == self._jog_telescope_id
            )


def make_scope_altaz_reader(method_sync: Callable[..., object]) -> AltazReader:
    """Build an :data:`AltazReader` that senses the mount's sky (az, el) from
    the raw motor **encoder** (``scope_get_horiz_coord``) rather than from
    plate-solved RA/Dec.

    The RA/Dec path reports ``ra==dec==0`` until the first plate-solve
    alignment — i.e. during *daytime, pre-alignment* operation (landmark
    calibration, manual jogging), which is exactly the window in which the
    mount can be swept toward the sun. Reading the encoder keeps the monitor
    sighted then instead of going blind.

    Encoder (alt, az) is used directly as sky (az, el): exact after rotation
    calibration and a conservative (wider-than-needed) cone otherwise — the
    same approximation the pre-flight guards use. Returns ``None`` only when
    the reading is genuinely missing / malformed / non-finite, so the monitor
    skips that tick rather than acting on bad data.

    ``method_sync`` is any callable with the ``method_sync(method[, params])``
    shape (AlpacaClient / Seestar).
    """

    def _read() -> Optional[tuple[float, float]]:
        try:
            resp = method_sync("scope_get_horiz_coord")
        except Exception:
            logger.debug("scope_get_horiz_coord failed", exc_info=True)
            return None
        if not isinstance(resp, dict) or "result" not in resp:
            return None
        result = resp["result"]
        try:
            enc_alt = float(result[0])
            enc_az = float(result[1])
        except (TypeError, ValueError, IndexError, KeyError):
            return None
        if not (math.isfinite(enc_alt) and math.isfinite(enc_az)):
            return None
        return enc_az % 360.0, enc_alt

    return _read


_MONITOR: Optional[SunSafetyMonitor] = None
_MONITOR_LOCK = threading.Lock()


def get_sun_monitor() -> Optional[SunSafetyMonitor]:
    """Return the process-singleton monitor, or None if not set up yet.

    The live_tracker_service wires this up at process startup. Callers
    that hit this path before startup (or in tests) get None and should
    treat "no monitor" as "no lockout active".
    """
    with _MONITOR_LOCK:
        return _MONITOR


def set_sun_monitor(monitor: Optional[SunSafetyMonitor]) -> None:
    """Install (or clear) the process-singleton monitor."""
    global _MONITOR
    with _MONITOR_LOCK:
        _MONITOR = monitor


def sun_safety_is_locked_out() -> bool:
    """Cheap convenience for the speed_move wrapper.

    Returns False when no monitor is installed (test mode, CLI tools,
    etc.). Never raises.
    """
    m = get_sun_monitor()
    return bool(m is not None and m.is_locked_out())


def sun_safety_jog_in_progress(telescope_id: Optional[int] = None) -> bool:
    """True while the monitor's emergency jog is executing on the mount.

    Session-exit direct motor-stops consult this before issuing their raw
    ``scope_speed_move(speed=0)`` so they don't cancel the jog-away. Pass
    the telescope id the stop targets — the jog only ever drives the
    monitor's scope, and a stop on a different mount must not be
    suppressed. Returns False when no monitor is installed. Never raises.
    """
    m = get_sun_monitor()
    return bool(m is not None and m.is_jog_in_progress(telescope_id))


__all__ = [
    "DEFAULT_ALT_THRESHOLD_DEG",
    "DEFAULT_MIN_SEPARATION_DEG",
    "AltazReader",
    "RawJogCommand",
    "SafetyTrip",
    "SunSafetyLocked",
    "SunSafetyMonitor",
    "angular_separation",
    "compute_jog_angle",
    "compute_sun_altaz",
    "get_sun_monitor",
    "is_sun_safe",
    "make_scope_altaz_reader",
    "set_sun_monitor",
    "sun_safety_is_locked_out",
    "sun_safety_jog_in_progress",
]
