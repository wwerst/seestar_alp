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
