"""Top-level LiveTracker / sun-safety service wired into `root_app.py`.

`LiveTrackerMain` is the fourth `AppRunner`-compatible entry point in
the process (alongside `DeviceMain`, `FrontMain`, and the imaging
Flask server). At boot it:

- Instantiates the `LiveTrackManager` singleton (cheap — no threads).
- Instantiates the `TargetCatalog` with `live_enabled=True` (cheap —
  the adsb.fi poller is deferred to the first `list_live()` call).
- Starts a `SunSafetyMonitor` whose altaz_reader senses the mount's
  pointing from the raw motor encoder (`scope_get_horiz_coord` via
  AlpacaClient — stays sighted during daytime, pre-alignment work),
  and whose jog_command bypasses the lockout-aware `speed_move`
  wrapper by hitting `cli.method_sync('scope_speed_move', ...)`
  directly.

The goal: the tracker infrastructure is always loaded so the
sun-safety guard is authoritative, but it stays quiescent (no
outbound traffic, no mount commands) until the user actually opens
the live-tracker UI or runs a calibration.
"""

from __future__ import annotations

import logging
from typing import Optional

from device import live_tracker as _lt
from device import sun_safety as _ss
from device.config import Config


logger = logging.getLogger(__name__)


class LiveTrackerMain:
    """AppRunner-compatible wrapper for the live-tracker service."""

    def __init__(self) -> None:
        self._monitor: Optional[_ss.SunSafetyMonitor] = None

    # ---------- lifecycle ----------

    def start(self) -> None:
        Config.load_toml()

        # Touch the lazy singletons so their state is ready when the
        # Front hits them. Neither call starts a thread.
        _lt.get_manager()
        _lt.get_catalog()

        if not getattr(Config, "sun_avoidance_enabled", True):
            logger.info("sun_avoidance disabled in config — monitor NOT started")
            return

        self._monitor = _ss.SunSafetyMonitor(
            altaz_reader=_make_altaz_reader(),
            jog_command=_make_jog_command(),
            abort_active=_abort_active_sessions,
            lat_deg=Config.init_lat,
            lon_deg=Config.init_long,
            min_separation_deg=Config.sun_avoidance_min_sep_deg,
            alt_threshold_deg=Config.sun_avoidance_alt_threshold_deg,
            jog_speed=Config.sun_avoidance_jog_speed,
            jog_duration_s=Config.sun_avoidance_jog_duration_s,
            enabled=Config.sun_avoidance_enabled,
        )
        _ss.set_sun_monitor(self._monitor)
        self._monitor.start()

    def reload(self) -> None:
        """Picked up by root_app's config watcher. Pushes the new
        thresholds into the running monitor without restarting."""
        Config.load_toml()
        if self._monitor is None:
            # Config might have re-enabled us; spin up now.
            if getattr(Config, "sun_avoidance_enabled", True):
                self.start()
            return
        self._monitor.reload(
            min_separation_deg=Config.sun_avoidance_min_sep_deg,
            alt_threshold_deg=Config.sun_avoidance_alt_threshold_deg,
            jog_speed=Config.sun_avoidance_jog_speed,
            jog_duration_s=Config.sun_avoidance_jog_duration_s,
            enabled=Config.sun_avoidance_enabled,
        )

    def stop(self) -> None:
        if self._monitor is not None:
            self._monitor.stop()
        _ss.set_sun_monitor(None)


# ---------- callbacks / adapters ----------


def _make_altaz_reader() -> _ss.AltazReader:
    """Build a callable that reads the first configured scope's sky
    (az, alt). Returns None on any failure so the monitor skips a
    tick rather than crashing.

    Senses from the raw motor encoder (``scope_get_horiz_coord``) via
    :func:`device.sun_safety.make_scope_altaz_reader` — NOT from firmware
    RA/Dec, which reads (0, 0) until the first plate-solve alignment and
    left the monitor blind during exactly the daytime, pre-alignment
    window where sun protection matters most."""

    def _method_sync(method: str, params: Optional[dict] = None) -> object:
        from device.alpaca_client import AlpacaClient

        tid = _primary_telescope_id()
        cli = AlpacaClient("127.0.0.1", int(Config.port), tid)
        return cli.method_sync(method, params)

    return _ss.make_scope_altaz_reader(_method_sync)


def _make_jog_command() -> _ss.RawJogCommand:
    """Build the privileged jog callable. Uses the raw AlpacaClient
    directly (bypasses device.velocity_controller.speed_move, which
    would short-circuit on the lockout event the monitor just set)."""

    def _jog(speed: int, angle: int, dur_sec: int) -> None:
        from device.alpaca_client import AlpacaClient

        tid = _primary_telescope_id()
        cli = AlpacaClient("127.0.0.1", int(Config.port), tid)
        cli.method_sync(
            "scope_speed_move",
            {"speed": int(speed), "angle": int(angle), "dur_sec": int(dur_sec)},
        )

    return _jog


def _abort_active_sessions() -> None:
    """Stop every mount-driving session on every configured scope: live
    tracker, rotation calibration, calibrate-motion, nighttime calibration
    (session + hands-free auto-runner), and the sky-visibility mapper.
    Covering all of them matters — any survivor re-slews right after the
    monitor's jog and re-trips it in a loop. Runs inside the monitor's
    trigger sequence; exceptions are caught per-manager so one failing
    teardown can't shield the rest."""
    import device.calibrate_motion as _cm
    import device.nighttime_calibration as _nc
    import device.rotation_calibration as _rc
    import device.visibility_mapper as _vm

    mgr = _lt.get_manager()
    stops = (
        ("live_tracker", lambda tid: mgr.stop(tid)),
        ("rotation_calibration", lambda tid: _rc.get_calibration_manager().stop(tid)),
        ("calibrate_motion", lambda tid: _cm.get_calibrate_motion_manager().stop(tid)),
        ("nighttime_calibration", lambda tid: _nc.get_nighttime_manager().stop(tid)),
        ("nighttime_auto", lambda tid: _nc.get_nighttime_auto_manager().stop(tid)),
        # force=True: the mapper's 5-minute minimum-runtime gate must not
        # delay an emergency abort.
        (
            "visibility_mapper",
            lambda tid: _vm.get_visibility_manager().stop(tid, force=True),
        ),
    )
    for dev in getattr(Config, "seestars", []) or []:
        tid = int(dev["device_num"])
        for name, stop in stops:
            try:
                stop(tid)
            except Exception:
                logger.debug("%s stop raised for %s", name, dev, exc_info=True)


def _primary_telescope_id() -> int:
    """Return the first configured seestar's device_num, defaulting to 1."""
    seestars = getattr(Config, "seestars", None) or []
    if seestars:
        try:
            return int(seestars[0]["device_num"])
        except (KeyError, TypeError, ValueError):
            pass
    return 1


__all__ = ["LiveTrackerMain"]
