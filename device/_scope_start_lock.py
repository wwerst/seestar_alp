"""Shared per-telescope start-lock for cross-manager coordination.

Both ``device.rotation_calibration.CalibrationManager.start`` and
``device.live_tracker.LiveTrackManager.start`` need to drive the same
mount and must not run concurrently on the same telescope. Each manager
holds its own ``self._lock`` for its own registry, but those two locks
do NOT coordinate with each other — leaving a TOCTOU window where:

  thread A: cal_mgr.start(...)  → cross-checks tracker_mgr (none) ✓
  thread B: tracker_mgr.start(...) → cross-checks cal_mgr (none yet) ✓
  thread A: cal_mgr._lock acquired, registry write, session.start()
  thread B: tracker_mgr._lock acquired, registry write, session.start()
  → both sessions running on the same physical mount.

This module exposes ``get_scope_start_lock(telescope_id)`` returning a
per-telescope ``threading.Lock``. Both managers acquire that single
lock around the entire start sequence (cross-check + registry write +
session.start()), so the whole "is anyone running on this scope?" →
"register me" critical section is atomic across managers.

The lock is keyed by ``int(telescope_id)``; per-scope granularity means
a calibration on scope 1 does not block a tracker start on scope 2.
"""

from __future__ import annotations

import threading


_REGISTRY_LOCK = threading.Lock()
_SCOPE_LOCKS: dict[int, threading.Lock] = {}


def get_scope_start_lock(telescope_id: int) -> threading.Lock:
    """Return the shared start-lock for ``telescope_id``.

    Lazily creates a new ``threading.Lock`` on first reference; subsequent
    calls for the same id return the same lock object. Locks are NEVER
    removed from the registry — operating systems handle a few thousand
    idle locks fine, and removing them safely would require a strong
    "no one is currently waiting" signal that ``threading.Lock`` does
    not expose.
    """
    tid = int(telescope_id)
    with _REGISTRY_LOCK:
        lock = _SCOPE_LOCKS.get(tid)
        if lock is None:
            lock = threading.Lock()
            _SCOPE_LOCKS[tid] = lock
        return lock


def raise_if_scope_busy(
    telescope_id: int,
    new_owner: str,
    *,
    check_tracker: bool = True,
    check_calibration: bool = True,
    check_calibrate_motion: bool = True,
    check_visibility: bool = True,
    check_nighttime_auto: bool = True,
) -> None:
    """Raise ``RuntimeError`` if another mount-driving manager already owns
    ``telescope_id``.

    A manager calls this while it holds this telescope's start-lock, right
    before it registers its own session, so the "is anyone else driving
    this mount?" check and the registration are atomic across managers.
    Every manager import is lazy + ``ImportError``-guarded so this helper
    stays usable from any ``device.*`` module and in unit tests that stub
    only a subset of managers. Pass ``check_<self>=False`` to skip the
    caller's own registry — the caller's duplicate-start check lives in
    its own manager.
    """
    tid = int(telescope_id)
    if check_tracker:
        try:
            from device.live_tracker import get_manager as _get_tracker_mgr

            tracker = _get_tracker_mgr().get(tid)
            if tracker is not None and tracker.is_alive():
                raise RuntimeError(
                    f"telescope {tid} is live-tracking; stop the tracker before "
                    f"starting {new_owner}"
                )
        except ImportError:
            pass
    if check_calibration:
        try:
            from device.rotation_calibration import get_calibration_manager

            if get_calibration_manager().is_running(tid):
                raise RuntimeError(
                    f"telescope {tid} is calibrating; stop the calibration before "
                    f"starting {new_owner}"
                )
        except ImportError:
            pass
    if check_calibrate_motion:
        try:
            from device.calibrate_motion import get_calibrate_motion_manager

            if get_calibrate_motion_manager().is_running(tid):
                raise RuntimeError(
                    f"telescope {tid} is in calibrate-motion mode; stop it before "
                    f"starting {new_owner}"
                )
        except ImportError:
            pass
    if check_visibility:
        try:
            from device.visibility_mapper import get_visibility_manager

            if get_visibility_manager().is_running(tid):
                raise RuntimeError(
                    f"telescope {tid} is running a visibility map; stop it before "
                    f"starting {new_owner}"
                )
        except ImportError:
            pass
    if check_nighttime_auto:
        try:
            from device.nighttime_calibration import get_nighttime_auto_manager

            if get_nighttime_auto_manager().is_running(tid):
                raise RuntimeError(
                    f"telescope {tid} has a nighttime auto-run in flight; stop it "
                    f"before starting {new_owner}"
                )
        except ImportError:
            pass
