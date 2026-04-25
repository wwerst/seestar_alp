"""Continuous-control motion primitive for the calibrate-rotation page.

Wraps :func:`device.streaming_controller.track` with a mutable
:class:`StaticReferenceProvider` so the calibrate UI can drive the mount
through one motion path for jog, click-to-go, and fine-position-for-capture.
The primitive avoids ``move_to_ff``'s arrival-tolerance behaviour, which
truncates nudges at ``arrive_tolerance_deg`` (0.3° default, 0.1° in the
calibrate-nudge path) — 60× the smallest 0.005° increment in the calibrate
UI. The streaming loop never "arrives"; it just keeps closing the residual
error until the page tears the session down.

Three knobs the UI uses:

- ``set_target(az, el)`` — replace the static target outright; jog rate is
  reset to 0. Used by click-to-go.
- ``nudge_target(daz, del)`` — delta. Preserves the current jog rate so a
  user can nudge mid-jog. Used by the existing daytime nudge buttons (via
  ``CalibrationSession.nudge`` delegation) and by per-tick advance during
  arrow-key jogging.
- ``set_jog(az_degs, el_degs)`` — set a constant velocity that the provider
  applies to the target each tick. Held arrow keys map to a non-zero rate;
  ``freeze_jog()`` resets to 0.

The session is singleton-per-telescope (``CalibrateMotionManager``), with a
mutex against ``LiveTrackManager`` so the two cannot both drive the same
mount. ``CalibrationManager`` (the daytime FAA-landmark workflow) is
allowed to coexist; when both are running, ``CalibrationSession`` delegates
its motion to the active ``CalibrateMotionSession`` instead of issuing
``move_to_ff`` directly.

Lifecycle: the calibrate page POSTs ``/calibrate_motion/start`` on load and
``/calibrate_motion/stop`` on ``pagehide`` / ``visibilitychange(hidden)``.
``stop()`` sets the streaming loop's stop event; the loop's ``finally``
issues ``speed_move(cli, 0, 0)`` so the motor halts even if the join hits
the 5 s timeout. Same try/finally pattern as PR #11's velocity-controller
hardening.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from astropy.coordinates import EarthLocation

from device.config import Config
from device.plant_limits import AzimuthLimits, CumulativeAzTracker
from device.reference_provider import ReferenceSample
from device.streaming_controller import (
    TickInfo,
    track,
)
from device.velocity_controller import PositionLogger
from scripts.trajectory.observer import build_site


# Default convergence threshold for ``is_settled`` checks. Half of the
# smallest UI nudge (0.005°) so a single nudge is only "settled" once the
# mount has actually moved through the increment.
DEFAULT_SETTLED_THRESHOLD_DEG = 0.0025
DEFAULT_SETTLED_TICKS = 5

# The session is meant to live as long as the calibrate page is open —
# operators may leave it up for hours during a long observation prep. The
# streaming controller's default 15-min cap is too tight; lift it to 8 h.
SESSION_MAX_DURATION_S = 8 * 3600.0

# Logging directory matches LiveTrackSession's so post-run analysis tools
# pick up calibrate-motion runs alongside live-tracker ones.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_LOG_DIR = _REPO_ROOT / "auto_level_logs"


# ---------- StaticReferenceProvider ----------------------------------


class StaticReferenceProvider:
    """ReferenceProvider whose target is mutable from the outside.

    The target moves at constant velocity ``(jog_az_degs, jog_el_degs)``
    starting from ``(target_az_at_t0, target_el_at_t0)`` at time ``t0``:

        az(t) = target_az_at_t0 + jog_az_degs * (t - t0)
        el(t) = target_el_at_t0 + jog_el_degs * (t - t0)

    Mutators (``set_jog``, ``nudge_target``, ``set_target``) snap the curve
    to ``time.time()`` so a control-loop sampling at any later ``t`` sees a
    continuous trajectory — no jumps when jog rates change mid-run.

    The provider expresses positions in whatever frame the streaming
    controller's anchor offset bridges. ``track()`` reads the encoder once
    at start and computes ``az_offset = cur_cum_az - ref_anchor.az_cum_deg``
    so ``eff_ref = ref + offset`` lands on the encoder's frame. From then
    on, deltas commanded through this provider are commanded at the mount
    encoder, regardless of whether the mount has been rotation-calibrated.
    """

    # Stored as a plain attribute so streaming_controller's
    # ``provider.__dict__.get("extrapolation_s", ...)`` lookup succeeds.
    extrapolation_s = 1e9

    def __init__(
        self,
        initial_az_deg: float = 0.0,
        initial_el_deg: float = 0.0,
    ) -> None:
        self._target_az_at_t0 = float(initial_az_deg)
        self._target_el_at_t0 = float(initial_el_deg)
        self._jog_az_degs = 0.0
        self._jog_el_degs = 0.0
        # ``_t0`` is set lazily on the first ``sample`` call so the first
        # tick after ``start()`` does not see a large dt from session-init
        # time. ``time.time()`` here would also work but the lazy form is
        # robust to test code that constructs a provider and only samples
        # later.
        self._t0: float | None = None
        self._lock = threading.Lock()

    # ---------- mutators (called from HTTP handlers) ----------

    def set_jog(self, az_degs: float, el_degs: float) -> None:
        """Set a constant velocity applied to the target each tick.

        Snaps the curve so the target value at the moment of the call is
        preserved — only the slope changes. ``az_degs=0, el_degs=0`` is
        equivalent to ``freeze_jog()``.
        """
        with self._lock:
            self._snap_locked()
            self._jog_az_degs = float(az_degs)
            self._jog_el_degs = float(el_degs)

    def freeze_jog(self) -> None:
        self.set_jog(0.0, 0.0)

    def nudge_target(self, d_az_deg: float, d_el_deg: float) -> None:
        """Add a fixed delta to the current target. Preserves jog rate."""
        with self._lock:
            self._snap_locked()
            self._target_az_at_t0 += float(d_az_deg)
            self._target_el_at_t0 += float(d_el_deg)

    def set_target(self, az_deg: float, el_deg: float) -> None:
        """Replace the target outright. Resets jog to zero."""
        with self._lock:
            self._target_az_at_t0 = float(az_deg)
            self._target_el_at_t0 = float(el_deg)
            self._jog_az_degs = 0.0
            self._jog_el_degs = 0.0
            self._t0 = time.time()

    def get_target(self) -> tuple[float, float]:
        """Return the *current* target position (after applying jog up to
        now). Useful to surface the moving target in /state polls."""
        with self._lock:
            t0 = self._t0 if self._t0 is not None else time.time()
            dt = time.time() - t0
            return (
                self._target_az_at_t0 + self._jog_az_degs * dt,
                self._target_el_at_t0 + self._jog_el_degs * dt,
            )

    def get_jog(self) -> tuple[float, float]:
        with self._lock:
            return (self._jog_az_degs, self._jog_el_degs)

    # ---------- ReferenceProvider protocol ----------

    def sample(self, t_unix: float) -> ReferenceSample:
        with self._lock:
            if self._t0 is None:
                self._t0 = float(t_unix)
            dt = float(t_unix) - self._t0
            return ReferenceSample(
                t_unix=float(t_unix),
                az_cum_deg=self._target_az_at_t0 + self._jog_az_degs * dt,
                el_deg=self._target_el_at_t0 + self._jog_el_degs * dt,
                v_az_degs=self._jog_az_degs,
                v_el_degs=self._jog_el_degs,
                a_az_degs2=0.0,
                a_el_degs2=0.0,
                stale=False,
                extrapolated=False,
            )

    def valid_range(self) -> tuple[float, float]:
        # A wide band centred on now means the controller's "before head"
        # wait branch never trips and the "past tail" exit never fires.
        now = time.time()
        return (now - 86400.0, now + 365.0 * 86400.0)

    # ---------- internal ----------

    def _snap_locked(self) -> None:
        """Bake the current jog into the anchor and reset the time origin.

        Must be called with ``self._lock`` held.
        """
        if self._t0 is None:
            self._t0 = time.time()
            return
        now = time.time()
        dt = now - self._t0
        if dt != 0.0:
            self._target_az_at_t0 += self._jog_az_degs * dt
            self._target_el_at_t0 += self._jog_el_degs * dt
        self._t0 = now


# ---------- MotionStatus -----------------------------------------------


@dataclass
class MotionStatus:
    """JSON-serialisable snapshot of a CalibrateMotionSession."""

    active: bool
    phase: str  # "init" | "track" | "stopped" | "error" | "done"
    elapsed_s: float
    exit_reason: str | None
    # Last-tick fields. ``None`` until the first tick fires.
    target_az_deg: float | None
    target_el_deg: float | None
    cur_cum_az_deg: float | None
    cur_el_deg: float | None
    err_az_deg: float | None
    err_el_deg: float | None
    jog_az_degs: float
    jog_el_degs: float
    is_settled: bool
    tick: int
    errors: list[str] = field(default_factory=list)


# ---------- CalibrateMotionSession ------------------------------------


class CalibrateMotionSession:
    """One streaming-controller run dedicated to interactive motion.

    Spawns a daemon thread that calls
    :func:`device.streaming_controller.track` with a
    :class:`StaticReferenceProvider`. Mutators on the provider are exposed
    here for HTTP handlers; ``status()`` reads the most-recent
    :class:`device.streaming_controller.TickInfo` under a short lock.
    """

    def __init__(
        self,
        telescope_id: int,
        *,
        initial_az_deg: float = 0.0,
        initial_el_deg: float = 0.0,
        alpaca_host: str = "127.0.0.1",
        alpaca_port: int | None = None,
        log_dir: Path | None = None,
        az_limits: AzimuthLimits | None = None,
        max_duration_s: float = SESSION_MAX_DURATION_S,
        settled_threshold_deg: float = DEFAULT_SETTLED_THRESHOLD_DEG,
        settled_ticks: int = DEFAULT_SETTLED_TICKS,
        dry_run: bool = False,
    ) -> None:
        self.telescope_id = int(telescope_id)
        self.provider = StaticReferenceProvider(
            initial_az_deg=initial_az_deg,
            initial_el_deg=initial_el_deg,
        )
        self._alpaca_host = alpaca_host
        self._alpaca_port = (
            int(alpaca_port) if alpaca_port is not None else int(Config.port)
        )
        self._log_dir = Path(log_dir) if log_dir else _LOG_DIR
        self._az_limits = az_limits
        self._max_duration_s = float(max_duration_s)
        self._settled_threshold_deg = float(settled_threshold_deg)
        self._settled_ticks = int(settled_ticks)
        self.dry_run = bool(dry_run)

        self._stop_evt = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._t_start = 0.0
        self._last_tick: TickInfo | None = None
        # Rolling window of |err| samples for is_settled. Length = settled_ticks.
        self._err_history: deque[tuple[float, float]] = deque(
            maxlen=self._settled_ticks
        )
        self._phase = "init"
        self._exit_reason: str | None = None
        self._errors: list[str] = []
        self._log_path: Path | None = None
        self._position_logger: PositionLogger | None = None

    # ---------- lifecycle ----------

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("calibrate-motion session already running")
        self._stop_evt.clear()
        self._t_start = time.time()
        self._phase = "starting"
        self._thread = threading.Thread(
            target=self._run,
            name=f"CalibrateMotionSession({self.telescope_id})",
            daemon=True,
        )
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop_evt.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ---------- mutators (delegated to provider) ----------

    def set_target(self, az_deg: float, el_deg: float) -> None:
        self.provider.set_target(az_deg, el_deg)

    def nudge_target(self, d_az_deg: float, d_el_deg: float) -> None:
        self.provider.nudge_target(d_az_deg, d_el_deg)

    def set_jog(self, az_degs: float, el_degs: float) -> None:
        self.provider.set_jog(az_degs, el_degs)

    def freeze_jog(self) -> None:
        self.provider.freeze_jog()

    def get_target(self) -> tuple[float, float]:
        return self.provider.get_target()

    # ---------- introspection ----------

    def is_settled(
        self,
        threshold_deg: float | None = None,
        ticks: int | None = None,
    ) -> bool:
        """Return True if the last ``ticks`` ticks had |err| under
        ``threshold_deg`` on both axes. ``None`` falls back to session
        defaults set at construction."""
        thr = (
            float(threshold_deg)
            if threshold_deg is not None
            else self._settled_threshold_deg
        )
        n = int(ticks) if ticks is not None else self._settled_ticks
        with self._lock:
            hist = list(self._err_history)
        if len(hist) < n:
            return False
        # Look at the trailing n entries.
        for eaz, eel in hist[-n:]:
            if abs(eaz) >= thr or abs(eel) >= thr:
                return False
        return True

    def status(self) -> MotionStatus:
        with self._lock:
            tick = self._last_tick
            elapsed = (time.time() - self._t_start) if self._t_start else 0.0
            phase = self._phase
            exit_reason = self._exit_reason
            errors = list(self._errors)
            jog = self.provider.get_jog()
        target = self.provider.get_target()
        if tick is None:
            return MotionStatus(
                active=self.is_alive(),
                phase=phase,
                elapsed_s=elapsed,
                exit_reason=exit_reason,
                target_az_deg=target[0],
                target_el_deg=target[1],
                cur_cum_az_deg=None,
                cur_el_deg=None,
                err_az_deg=None,
                err_el_deg=None,
                jog_az_degs=jog[0],
                jog_el_degs=jog[1],
                is_settled=False,
                tick=0,
                errors=errors,
            )
        return MotionStatus(
            active=self.is_alive(),
            phase=phase,
            elapsed_s=elapsed,
            exit_reason=exit_reason,
            target_az_deg=target[0],
            target_el_deg=target[1],
            cur_cum_az_deg=tick.cur_cum_az_deg,
            cur_el_deg=tick.cur_el_deg,
            err_az_deg=tick.err_az_deg,
            err_el_deg=tick.err_el_deg,
            jog_az_degs=jog[0],
            jog_el_degs=jog[1],
            is_settled=self.is_settled(),
            tick=tick.tick,
            errors=errors,
        )

    @property
    def log_path(self) -> Path | None:
        return self._log_path

    # ---------- internal ----------

    def _on_tick(self, info: TickInfo) -> None:
        with self._lock:
            self._last_tick = info
            self._err_history.append((info.err_az_deg, info.err_el_deg))
            self._phase = "track"

    def _connect_mount(self) -> Any:
        from device.alpaca_client import AlpacaClient

        return AlpacaClient(self._alpaca_host, self._alpaca_port, self.telescope_id)

    def _run(self) -> None:
        cli = self._connect_mount()
        try:
            site = build_site()
            loc = EarthLocation.from_geodetic(
                lon=site.lon_deg,
                lat=site.lat_deg,
                height=site.alt_m,
            )
        except Exception as exc:
            with self._lock:
                self._errors.append(f"site lookup failed: {exc}")
            site = None
            loc = EarthLocation.from_geodetic(0, 0, 0)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        run_tag = time.strftime("%Y-%m-%dT%H-%M-%S", time.localtime())
        self._log_path = self._log_dir / f"{run_tag}.calibrate_motion.jsonl"
        try:
            self._position_logger = PositionLogger(cli, loc, self._log_path)
            self._position_logger.start()
            self._position_logger.set_phase("calibrate_motion_init")
            self._position_logger.mark_event(
                "calibrate_motion_start",
                telescope_id=self.telescope_id,
                dry_run=self.dry_run,
            )
        except Exception as exc:
            with self._lock:
                self._errors.append(f"PositionLogger start failed: {exc}")
            self._position_logger = None

        # Outer try/finally guarantees a direct ``cli.method_sync`` motor
        # stop on session exit. ``streaming_controller.track`` already
        # issues ``speed_move(cli, 0, 0, ...)`` from its own ``finally``,
        # but ``speed_move`` is the lockout-aware wrapper that refuses
        # commands while ``SunSafetyMonitor`` holds the emergency lockout.
        # If lockout is active when we exit, that cleanup is silently
        # swallowed and the previous tick's command keeps running until
        # its firmware ``dur_sec`` TTL expires (bounded uncommanded motion
        # of up to ``v_max × TICK_CMD_DUR_S``). Calling ``method_sync``
        # directly bypasses the wrapper, mirroring the
        # ``_motor_stop_on_exit`` pattern in ``velocity_controller``.
        try:
            try:
                tracker: CumulativeAzTracker | None
                try:
                    tracker = CumulativeAzTracker.load_or_fresh()
                except Exception:
                    tracker = CumulativeAzTracker()

                with self._lock:
                    self._phase = "track"
                try:
                    result = track(
                        cli,
                        self.provider,
                        az_limits=self._az_limits,
                        az_tracker=tracker,
                        position_logger=self._position_logger,
                        stop_signal=self._stop_evt,
                        dry_run=self.dry_run,
                        max_duration_s=self._max_duration_s,
                        tick_callback=self._on_tick,
                    )
                except Exception as exc:
                    with self._lock:
                        self._exit_reason = "session_error"
                        self._errors.append(f"track() raised: {exc}")
                        self._phase = "error"
                else:
                    with self._lock:
                        self._exit_reason = result.exit_reason
                        if result.errors:
                            self._errors.extend(result.errors)
                        self._phase = (
                            "done"
                            if result.exit_reason in ("end_of_track", "stop_signal")
                            else "error"
                        )
            finally:
                if self._position_logger is not None:
                    try:
                        self._position_logger.mark_event(
                            "calibrate_motion_end",
                            exit_reason=self._exit_reason,
                        )
                        self._position_logger.stop()
                    except Exception:
                        pass
        finally:
            if not self.dry_run:
                try:
                    cli.method_sync(
                        "scope_speed_move",
                        {"speed": 0, "angle": 0, "dur_sec": 1},
                    )
                except Exception:
                    with self._lock:
                        self._errors.append("outer motor-stop on session exit failed")


# ---------- CalibrateMotionManager ------------------------------------


class CalibrateMotionManager:
    """Process-singleton registry of CalibrateMotionSessions, keyed by
    telescope id. Mirrors ``LiveTrackManager`` and ``CalibrationManager``.

    Mutex policy:
    - Refuses to start while a ``LiveTrackSession`` is alive on the same
      telescope (both drive the mount continuously; running both would
      have them fighting over commands).
    - Allowed alongside ``CalibrationSession`` — when both are alive, the
      calibration session delegates its motion to the motion session
      instead of issuing ``move_to_ff``.
    """

    def __init__(self) -> None:
        self._sessions: dict[int, CalibrateMotionSession] = {}
        self._lock = threading.Lock()

    def get(self, telescope_id: int) -> CalibrateMotionSession | None:
        with self._lock:
            return self._sessions.get(int(telescope_id))

    def is_running(self, telescope_id: int) -> bool:
        s = self.get(telescope_id)
        return s is not None and s.is_alive()

    def start(self, session: CalibrateMotionSession) -> CalibrateMotionSession:
        tid = int(session.telescope_id)
        # Hold the shared per-telescope start-lock across the entire
        # sequence (cross-checks + registry write + session.start()) so
        # that concurrent CalibrationManager / LiveTrackManager /
        # CalibrateMotionManager starts on the same scope cannot all pass
        # their respective cross-checks. Mirrors the pattern in
        # ``device.live_tracker.LiveTrackManager.start``.
        from device._scope_start_lock import get_scope_start_lock

        with get_scope_start_lock(tid):
            # Cross-manager mutex against LiveTrackManager. Lazy import keeps
            # this module from pulling in live_tracker at import time.
            try:
                from device.live_tracker import get_manager as _get_tracker_mgr

                tracker = _get_tracker_mgr().get(tid)
                if tracker is not None and tracker.is_alive():
                    raise RuntimeError(
                        f"telescope {tid} is live-tracking; stop the live tracker first"
                    )
            except ImportError:
                pass
            with self._lock:
                existing = self._sessions.get(tid)
                if existing is not None and existing.is_alive():
                    raise RuntimeError(
                        f"telescope {tid} already in calibrate-motion mode; "
                        "stop the existing session first"
                    )
                # Start under the lock so the cross-check + registry write +
                # thread spawn are atomic against another concurrent start
                # call. Same pattern as LiveTrackManager.start.
                session.start()
                self._sessions[tid] = session
        return session

    def stop(self, telescope_id: int) -> MotionStatus | None:
        with self._lock:
            s = self._sessions.get(int(telescope_id))
        if s is None:
            return None
        s.stop()
        return s.status()

    def status(self, telescope_id: int) -> MotionStatus | None:
        s = self.get(telescope_id)
        return s.status() if s is not None else None


_MANAGER: CalibrateMotionManager | None = None
_MANAGER_LOCK = threading.Lock()


def get_calibrate_motion_manager() -> CalibrateMotionManager:
    """Return the process-level singleton."""
    global _MANAGER
    with _MANAGER_LOCK:
        if _MANAGER is None:
            _MANAGER = CalibrateMotionManager()
        return _MANAGER
