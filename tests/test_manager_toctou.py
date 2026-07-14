"""Regression test for the cross-manager TOCTOU race between
``CalibrationManager.start`` and ``LiveTrackManager.start``.

Both managers refuse to start a session on a telescope id that the
*other* manager is already running, but each only holds its own
``self._lock`` around its own registry. Without a shared per-telescope
start-lock, two concurrent starts on the same scope can both pass their
respective cross-checks and then each register a session — leaving two
sessions driving the same physical mount.

The shared lock in ``device._scope_start_lock`` closes the window: the
whole "is anyone running on this scope?" → "register me" sequence runs
under one mutex shared by both managers.
"""

from __future__ import annotations

import threading
import time

import pytest


class _FakeCalSession:
    """Stand-in for :class:`CalibrationSession`. Mirrors the contract
    that ``CalibrationManager`` consumes: ``telescope_id`` attribute,
    ``start()`` / ``is_alive()`` / ``stop()`` / ``status()``."""

    def __init__(self, telescope_id: int) -> None:
        self.telescope_id = int(telescope_id)
        self._alive = False

    def start(self) -> None:
        # Brief sleep widens the TOCTOU window between
        # "registry write" and "session is_alive() == True" so the bug
        # (in pre-fix code) is reliably observable. Without the shared
        # lock, the racing thread's cross-check fires inside this window
        # and incorrectly sees ``is_alive() == False`` even though the
        # session is already registered, so it passes its check and
        # both managers end up holding sessions on the same telescope.
        time.sleep(0.005)
        self._alive = True

    def is_alive(self) -> bool:
        return self._alive

    def stop(self, timeout: float = 5.0) -> None:
        self._alive = False

    def status(self):
        return None


class _FakeTrackerSession:
    """Stand-in for :class:`LiveTrackSession`. Same contract as
    :class:`_FakeCalSession`."""

    def __init__(self, telescope_id: int) -> None:
        self.telescope_id = int(telescope_id)
        self._alive = False

    def start(self) -> None:
        time.sleep(0.005)
        self._alive = True

    def is_alive(self) -> bool:
        return self._alive

    def stop(self, timeout: float = 5.0) -> None:
        self._alive = False

    def status(self):
        return None


def test_cross_manager_start_lock_serializes_concurrent_starts(monkeypatch):
    """Spawn one thread starting a calibration and one thread starting
    a live-track session on the same telescope id. The calibration is
    given a small head start so it has registered its session in
    ``cal_mgr._sessions`` and is mid ``session.start()`` (the slow
    sleep below) at the moment the tracker thread's cross-check fires.
    Exactly one must win and the other must raise ``RuntimeError``;
    repeating 100 iterations makes any regression reliable.

    Without the shared per-telescope start-lock this fails
    deterministically: the tracker's cross-check calls
    ``cal_mgr.is_running(tid)`` which sees the registered session but
    its ``is_alive()`` returns False (the worker hasn't been spawned
    yet — that's still happening inside the slow ``session.start()``),
    so the tracker passes its check and both managers end up holding
    sessions on the same telescope.
    """
    import device.live_tracker as lt
    import device.rotation_calibration as rc
    from device.live_tracker import LiveTrackManager
    from device.rotation_calibration import CalibrationManager

    iterations = 100
    tid = 99
    bad_outcomes: list[tuple[int, dict]] = []

    for i in range(iterations):
        cal_mgr = CalibrationManager()
        track_mgr = LiveTrackManager()
        # Cross-checks resolve via get_calibration_manager() / get_manager()
        # — patch the module-globals so they return our fresh per-iteration
        # instances rather than the process singletons leaking state.
        monkeypatch.setattr(lt, "_MANAGER", track_mgr)
        monkeypatch.setattr(rc, "_MANAGER", cal_mgr)

        cal_session = _FakeCalSession(tid)
        track_session = _FakeTrackerSession(tid)

        results: dict[str, object] = {}

        def start_cal() -> None:
            try:
                cal_mgr.start(cal_session)
                results["cal"] = "ok"
            except RuntimeError as e:
                results["cal"] = e

        def start_track() -> None:
            try:
                track_mgr.start(track_session)
                results["track"] = "ok"
            except RuntimeError as e:
                results["track"] = e

        # daemon=True so a regression that wedges these threads can't keep
        # the pytest process alive past the test failure — CI would
        # otherwise hang waiting for a non-daemon thread instead of
        # surfacing the assertion failure.
        ta = threading.Thread(target=start_cal, daemon=True)
        tb = threading.Thread(target=start_track, daemon=True)
        try:
            ta.start()
            # Head start so cal's registry write happens before tracker's
            # cross-check. With the 5 ms sleep inside _FakeCalSession.start()
            # cal is reliably mid ``session.start()`` (registry written, but
            # not yet alive) when tracker's cross-check fires. The shared
            # lock turns this into "tracker waits, cal finishes, tracker's
            # check sees alive=True and raises". Without it, tracker's check
            # sees alive=False and both succeed.
            time.sleep(0.001)
            tb.start()
            ta.join(timeout=5.0)
            tb.join(timeout=5.0)
            assert not ta.is_alive(), "cal thread hung"
            assert not tb.is_alive(), "track thread hung"

            oks = [k for k, v in results.items() if v == "ok"]
            errs = [k for k, v in results.items() if isinstance(v, RuntimeError)]
            if not (len(oks) == 1 and len(errs) == 1):
                bad_outcomes.append((i, dict(results)))
        finally:
            # Cleanup so the next iteration starts clean, even if an
            # assertion above fails. Best-effort: a deadlock regression
            # may not be unblocked by stop(), but the daemon=True flag
            # above ensures pytest can still exit.
            cal_mgr.stop(tid)
            track_mgr.stop(tid)
            ta.join(timeout=1.0)
            tb.join(timeout=1.0)

    assert not bad_outcomes, (
        f"cross-manager TOCTOU: {len(bad_outcomes)}/{iterations} "
        f"iterations produced wrong outcome (sample: {bad_outcomes[:3]})"
    )


def test_scope_start_lock_is_per_telescope(monkeypatch):
    """The shared lock is keyed by telescope id, so a calibration on
    scope 1 must not block a live-track start on scope 2."""
    import device.live_tracker as lt
    import device.rotation_calibration as rc
    from device.live_tracker import LiveTrackManager
    from device.rotation_calibration import CalibrationManager

    cal_mgr = CalibrationManager()
    track_mgr = LiveTrackManager()
    monkeypatch.setattr(lt, "_MANAGER", track_mgr)
    monkeypatch.setattr(rc, "_MANAGER", cal_mgr)

    cal_session = _FakeCalSession(1)
    track_session = _FakeTrackerSession(2)

    cal_mgr.start(cal_session)
    try:
        # Different scope id → different lock; this must not deadlock
        # or raise.
        track_mgr.start(track_session)
        try:
            assert track_session.is_alive()
            assert cal_session.is_alive()
        finally:
            track_mgr.stop(2)
    finally:
        cal_mgr.stop(1)


def test_scope_start_lock_returns_same_lock_per_id():
    """Sanity: the lock registry must hand out the same lock object on
    every call for a given id. Otherwise managers wouldn't actually
    coordinate."""
    from device._scope_start_lock import get_scope_start_lock

    a = get_scope_start_lock(42)
    b = get_scope_start_lock(42)
    c = get_scope_start_lock(43)
    assert a is b
    assert a is not c
    # int-coercion: float-keyed lookups must hit the same lock.
    assert get_scope_start_lock(42.0) is a


def test_cal_then_tracker_same_scope_refused(monkeypatch):
    """Sequential (non-racing) sanity: calibration first → tracker on
    the same scope must be refused, even though the cross-check now
    runs under the shared lock. This guards against accidentally
    breaking the existing cross-check semantics while wiring up the
    shared lock."""
    import device.live_tracker as lt
    import device.rotation_calibration as rc
    from device.live_tracker import LiveTrackManager
    from device.rotation_calibration import CalibrationManager

    cal_mgr = CalibrationManager()
    track_mgr = LiveTrackManager()
    monkeypatch.setattr(lt, "_MANAGER", track_mgr)
    monkeypatch.setattr(rc, "_MANAGER", cal_mgr)

    cal_mgr.start(_FakeCalSession(7))
    try:
        with pytest.raises(RuntimeError, match="calibrating"):
            track_mgr.start(_FakeTrackerSession(7))
    finally:
        cal_mgr.stop(7)


def test_tracker_then_cal_same_scope_refused(monkeypatch):
    """Mirror of the above, in the opposite order."""
    import device.live_tracker as lt
    import device.rotation_calibration as rc
    from device.live_tracker import LiveTrackManager
    from device.rotation_calibration import CalibrationManager

    cal_mgr = CalibrationManager()
    track_mgr = LiveTrackManager()
    monkeypatch.setattr(lt, "_MANAGER", track_mgr)
    monkeypatch.setattr(rc, "_MANAGER", cal_mgr)

    track_mgr.start(_FakeTrackerSession(8))
    try:
        with pytest.raises(RuntimeError, match="live-tracking"):
            cal_mgr.start(_FakeCalSession(8))
    finally:
        track_mgr.stop(8)


# ---------- CalibrateMotion vs LiveTrack TOCTOU coverage --------------
#
# ``CalibrateMotionManager.start`` and ``LiveTrackManager.start`` have
# the same TOCTOU shape as the cal/tracker pair above: each manager
# holds its own ``self._lock`` for its own registry, but the
# cross-manager check + register sequence has to be atomic across both
# managers, otherwise concurrent starts can both register sessions on
# the same telescope. Both adopt the shared per-telescope start lock
# from :mod:`device._scope_start_lock` to close the window.


class _FakeMotionSession:
    """Stand-in for :class:`CalibrateMotionSession`. Same contract as
    :class:`_FakeCalSession`."""

    def __init__(self, telescope_id: int) -> None:
        self.telescope_id = int(telescope_id)
        self._alive = False

    def start(self) -> None:
        time.sleep(0.005)
        self._alive = True

    def is_alive(self) -> bool:
        return self._alive

    def stop(self, timeout: float = 5.0) -> None:
        self._alive = False

    def status(self):
        return None


def test_motion_vs_tracker_start_lock_serializes_concurrent_starts(monkeypatch):
    """Same TOCTOU shape as cal/tracker — calibrate-motion and live
    tracker must serialize on the shared per-telescope start lock so
    that exactly one wins when both start concurrently on the same
    telescope id."""
    import device.calibrate_motion as cm
    import device.live_tracker as lt
    from device.calibrate_motion import CalibrateMotionManager
    from device.live_tracker import LiveTrackManager

    iterations = 100
    tid = 199
    bad_outcomes: list[tuple[int, dict]] = []

    for i in range(iterations):
        motion_mgr = CalibrateMotionManager()
        track_mgr = LiveTrackManager()
        monkeypatch.setattr(lt, "_MANAGER", track_mgr)
        monkeypatch.setattr(cm, "_MANAGER", motion_mgr)

        motion_session = _FakeMotionSession(tid)
        track_session = _FakeTrackerSession(tid)

        results: dict[str, object] = {}

        def start_motion() -> None:
            try:
                motion_mgr.start(motion_session)
                results["motion"] = "ok"
            except RuntimeError as e:
                results["motion"] = e

        def start_track() -> None:
            try:
                track_mgr.start(track_session)
                results["track"] = "ok"
            except RuntimeError as e:
                results["track"] = e

        ta = threading.Thread(target=start_motion, daemon=True)
        tb = threading.Thread(target=start_track, daemon=True)
        try:
            ta.start()
            time.sleep(0.001)
            tb.start()
            ta.join(timeout=5.0)
            tb.join(timeout=5.0)
            assert not ta.is_alive(), "motion thread hung"
            assert not tb.is_alive(), "track thread hung"

            oks = [k for k, v in results.items() if v == "ok"]
            errs = [k for k, v in results.items() if isinstance(v, RuntimeError)]
            if not (len(oks) == 1 and len(errs) == 1):
                bad_outcomes.append((i, dict(results)))
        finally:
            motion_mgr.stop(tid)
            track_mgr.stop(tid)
            ta.join(timeout=1.0)
            tb.join(timeout=1.0)

    assert not bad_outcomes, (
        f"motion/tracker TOCTOU: {len(bad_outcomes)}/{iterations} "
        f"iterations produced wrong outcome (sample: {bad_outcomes[:3]})"
    )


def test_motion_then_tracker_same_scope_refused(monkeypatch):
    """Sequential sanity: motion first → tracker on the same scope must
    be refused."""
    import device.calibrate_motion as cm
    import device.live_tracker as lt
    from device.calibrate_motion import CalibrateMotionManager
    from device.live_tracker import LiveTrackManager

    motion_mgr = CalibrateMotionManager()
    track_mgr = LiveTrackManager()
    monkeypatch.setattr(lt, "_MANAGER", track_mgr)
    monkeypatch.setattr(cm, "_MANAGER", motion_mgr)

    motion_mgr.start(_FakeMotionSession(17))
    try:
        with pytest.raises(RuntimeError, match="calibrate-motion"):
            track_mgr.start(_FakeTrackerSession(17))
    finally:
        motion_mgr.stop(17)


def test_tracker_then_motion_same_scope_refused(monkeypatch):
    """Mirror: tracker first → motion on the same scope must be
    refused."""
    import device.calibrate_motion as cm
    import device.live_tracker as lt
    from device.calibrate_motion import CalibrateMotionManager
    from device.live_tracker import LiveTrackManager

    motion_mgr = CalibrateMotionManager()
    track_mgr = LiveTrackManager()
    monkeypatch.setattr(lt, "_MANAGER", track_mgr)
    monkeypatch.setattr(cm, "_MANAGER", motion_mgr)

    track_mgr.start(_FakeTrackerSession(18))
    try:
        with pytest.raises(RuntimeError, match="live-tracking"):
            motion_mgr.start(_FakeMotionSession(18))
    finally:
        track_mgr.stop(18)


# ---------- VisibilityMap / NighttimeAuto vs LiveTrack coverage ------
#
# The visibility mapper and the nighttime auto-runner both drive the
# mount (they slew between sampled cells / waypoints). Historically their
# managers only locked their own registries and cross-checked nothing, so
# they could start on a scope already owned by live-track / calibration /
# calibrate-motion (or each other). They now acquire the shared
# per-telescope start-lock and cross-check the other mount-driving
# managers via device._scope_start_lock.raise_if_scope_busy — the same
# atomic "is anyone else driving this scope?" → "register me" section the
# compliant managers use.


class _FakeVisibilityMapper:
    """Stand-in for :class:`VisibilityMapper`. Matches the contract
    ``VisibilityMapManager`` consumes: ``telescope_id`` attribute,
    ``start()`` / ``is_active()`` / ``request_stop()`` / ``status()``."""

    def __init__(self, telescope_id: int) -> None:
        self.telescope_id = int(telescope_id)
        self._active = False

    def start(self) -> None:
        time.sleep(0.005)
        self._active = True

    def is_active(self) -> bool:
        return self._active

    def request_stop(self, *, force: bool = False):
        self._active = False
        return (True, "stopping")

    def status(self):
        return {}


class _FakeAutoRunner:
    """Stand-in for :class:`NighttimeAutoRunner`. Matches the contract
    ``NighttimeAutoManager`` consumes: ``start()`` / ``is_alive()`` /
    ``stop()`` / ``status()``."""

    def __init__(self, telescope_id: int) -> None:
        self.telescope_id = int(telescope_id)
        self._alive = False

    def start(self) -> None:
        time.sleep(0.005)
        self._alive = True

    def is_alive(self) -> bool:
        return self._alive

    def stop(self) -> None:
        self._alive = False

    def status(self):
        return None


def _race_two_starts(start_a, start_b) -> dict:
    """Run two manager-start callables concurrently (a gets a small head
    start) and return the {name: 'ok' | RuntimeError} outcome map."""
    results: dict[str, object] = {}

    def run(name, fn):
        try:
            fn()
            results[name] = "ok"
        except RuntimeError as e:
            results[name] = e

    ta = threading.Thread(target=lambda: run("a", start_a), daemon=True)
    tb = threading.Thread(target=lambda: run("b", start_b), daemon=True)
    ta.start()
    time.sleep(0.001)
    tb.start()
    ta.join(timeout=5.0)
    tb.join(timeout=5.0)
    assert not ta.is_alive() and not tb.is_alive(), "a start thread hung"
    return results


def test_visibility_vs_tracker_start_lock_serializes(monkeypatch):
    """Visibility-map start and live-track start on the same scope must
    serialize on the shared start-lock: exactly one wins."""
    import device.live_tracker as lt
    import device.visibility_mapper as vm
    from device.live_tracker import LiveTrackManager
    from device.visibility_mapper import VisibilityMapManager

    bad: list[tuple[int, dict]] = []
    for i in range(60):
        vis_mgr = VisibilityMapManager()
        track_mgr = LiveTrackManager()
        monkeypatch.setattr(lt, "_MANAGER", track_mgr)
        monkeypatch.setattr(vm, "_MANAGER", vis_mgr)
        tid = 299
        vis = _FakeVisibilityMapper(tid)
        trk = _FakeTrackerSession(tid)
        res = _race_two_starts(
            lambda: vis_mgr.start(vis),
            lambda: track_mgr.start(trk),
        )
        oks = [k for k, v in res.items() if v == "ok"]
        errs = [k for k, v in res.items() if isinstance(v, RuntimeError)]
        if not (len(oks) == 1 and len(errs) == 1):
            bad.append((i, dict(res)))
        vis_mgr.stop(tid, force=True)
        track_mgr.stop(tid)
    assert not bad, f"visibility/tracker TOCTOU: {len(bad)}/60 bad (sample {bad[:3]})"


def test_auto_vs_tracker_start_lock_serializes(monkeypatch):
    """Nighttime auto-run start and live-track start on the same scope
    must serialize: exactly one wins."""
    import device.live_tracker as lt
    import device.nighttime_calibration as nc
    from device.live_tracker import LiveTrackManager
    from device.nighttime_calibration import NighttimeAutoManager

    bad: list[tuple[int, dict]] = []
    for i in range(60):
        auto_mgr = NighttimeAutoManager()
        track_mgr = LiveTrackManager()
        monkeypatch.setattr(lt, "_MANAGER", track_mgr)
        monkeypatch.setattr(nc, "_AUTO_MANAGER", auto_mgr)
        tid = 399
        runner = _FakeAutoRunner(tid)
        trk = _FakeTrackerSession(tid)
        res = _race_two_starts(
            lambda: auto_mgr.start(tid, runner),
            lambda: track_mgr.start(trk),
        )
        oks = [k for k, v in res.items() if v == "ok"]
        errs = [k for k, v in res.items() if isinstance(v, RuntimeError)]
        if not (len(oks) == 1 and len(errs) == 1):
            bad.append((i, dict(res)))
        auto_mgr.stop(tid)
        track_mgr.stop(tid)
    assert not bad, f"auto/tracker TOCTOU: {len(bad)}/60 bad (sample {bad[:3]})"


def test_visibility_vs_auto_start_lock_serializes(monkeypatch):
    """Two mount-driving managers that are NOT live-track — visibility map
    and nighttime auto — must also serialize on the shared start-lock."""
    import device.nighttime_calibration as nc
    import device.visibility_mapper as vm
    from device.nighttime_calibration import NighttimeAutoManager
    from device.visibility_mapper import VisibilityMapManager

    bad: list[tuple[int, dict]] = []
    for i in range(60):
        vis_mgr = VisibilityMapManager()
        auto_mgr = NighttimeAutoManager()
        monkeypatch.setattr(vm, "_MANAGER", vis_mgr)
        monkeypatch.setattr(nc, "_AUTO_MANAGER", auto_mgr)
        tid = 499
        vis = _FakeVisibilityMapper(tid)
        runner = _FakeAutoRunner(tid)
        res = _race_two_starts(
            lambda: vis_mgr.start(vis),
            lambda: auto_mgr.start(tid, runner),
        )
        oks = [k for k, v in res.items() if v == "ok"]
        errs = [k for k, v in res.items() if isinstance(v, RuntimeError)]
        if not (len(oks) == 1 and len(errs) == 1):
            bad.append((i, dict(res)))
        vis_mgr.stop(tid, force=True)
        auto_mgr.stop(tid)
    assert not bad, f"visibility/auto TOCTOU: {len(bad)}/60 bad (sample {bad[:3]})"


def test_visibility_refused_when_tracker_running(monkeypatch):
    """Sequential sanity: live-track first → visibility map on the same
    scope is refused."""
    import device.live_tracker as lt
    import device.visibility_mapper as vm
    from device.live_tracker import LiveTrackManager
    from device.visibility_mapper import VisibilityMapManager

    vis_mgr = VisibilityMapManager()
    track_mgr = LiveTrackManager()
    monkeypatch.setattr(lt, "_MANAGER", track_mgr)
    monkeypatch.setattr(vm, "_MANAGER", vis_mgr)
    track_mgr.start(_FakeTrackerSession(20))
    try:
        with pytest.raises(RuntimeError, match="live-tracking"):
            vis_mgr.start(_FakeVisibilityMapper(20))
    finally:
        track_mgr.stop(20)


def test_tracker_refused_when_visibility_running(monkeypatch):
    """Reverse: visibility map first → live-track on the same scope is
    refused (the reciprocal check lives in LiveTrackManager)."""
    import device.live_tracker as lt
    import device.visibility_mapper as vm
    from device.live_tracker import LiveTrackManager
    from device.visibility_mapper import VisibilityMapManager

    vis_mgr = VisibilityMapManager()
    track_mgr = LiveTrackManager()
    monkeypatch.setattr(lt, "_MANAGER", track_mgr)
    monkeypatch.setattr(vm, "_MANAGER", vis_mgr)
    vis_mgr.start(_FakeVisibilityMapper(21))
    try:
        with pytest.raises(RuntimeError, match="visibility map"):
            track_mgr.start(_FakeTrackerSession(21))
    finally:
        vis_mgr.stop(21, force=True)


def test_auto_refused_when_tracker_running(monkeypatch):
    """Sequential sanity: live-track first → nighttime auto-run refused."""
    import device.live_tracker as lt
    import device.nighttime_calibration as nc
    from device.live_tracker import LiveTrackManager
    from device.nighttime_calibration import NighttimeAutoManager

    auto_mgr = NighttimeAutoManager()
    track_mgr = LiveTrackManager()
    monkeypatch.setattr(lt, "_MANAGER", track_mgr)
    monkeypatch.setattr(nc, "_AUTO_MANAGER", auto_mgr)
    track_mgr.start(_FakeTrackerSession(22))
    try:
        with pytest.raises(RuntimeError, match="live-tracking"):
            auto_mgr.start(22, _FakeAutoRunner(22))
    finally:
        track_mgr.stop(22)


def test_visibility_refused_when_auto_running(monkeypatch):
    """Two non-tracker mount-drivers exclude each other: nighttime auto
    first → visibility map refused."""
    import device.nighttime_calibration as nc
    import device.visibility_mapper as vm
    from device.nighttime_calibration import NighttimeAutoManager
    from device.visibility_mapper import VisibilityMapManager

    vis_mgr = VisibilityMapManager()
    auto_mgr = NighttimeAutoManager()
    monkeypatch.setattr(vm, "_MANAGER", vis_mgr)
    monkeypatch.setattr(nc, "_AUTO_MANAGER", auto_mgr)
    auto_mgr.start(23, _FakeAutoRunner(23))
    try:
        with pytest.raises(RuntimeError, match="auto-run"):
            vis_mgr.start(_FakeVisibilityMapper(23))
    finally:
        auto_mgr.stop(23)


def test_tracker_refused_when_auto_running(monkeypatch):
    """Reciprocal of test_auto_refused_when_tracker_running — closes the
    live-track <-> nighttime-auto symmetry: nighttime auto first → live-track
    on the same scope is refused (the check lives in LiveTrackManager)."""
    import device.live_tracker as lt
    import device.nighttime_calibration as nc
    from device.live_tracker import LiveTrackManager
    from device.nighttime_calibration import NighttimeAutoManager

    auto_mgr = NighttimeAutoManager()
    track_mgr = LiveTrackManager()
    monkeypatch.setattr(lt, "_MANAGER", track_mgr)
    monkeypatch.setattr(nc, "_AUTO_MANAGER", auto_mgr)
    auto_mgr.start(24, _FakeAutoRunner(24))
    try:
        with pytest.raises(RuntimeError, match="auto-run"):
            track_mgr.start(_FakeTrackerSession(24))
    finally:
        auto_mgr.stop(24)


def test_auto_refused_when_visibility_running(monkeypatch):
    """Reciprocal of test_visibility_refused_when_auto_running — closes the
    visibility <-> nighttime-auto symmetry: visibility map first → nighttime
    auto on the same scope is refused."""
    import device.nighttime_calibration as nc
    import device.visibility_mapper as vm
    from device.nighttime_calibration import NighttimeAutoManager
    from device.visibility_mapper import VisibilityMapManager

    vis_mgr = VisibilityMapManager()
    auto_mgr = NighttimeAutoManager()
    monkeypatch.setattr(vm, "_MANAGER", vis_mgr)
    monkeypatch.setattr(nc, "_AUTO_MANAGER", auto_mgr)
    vis_mgr.start(_FakeVisibilityMapper(25))
    try:
        with pytest.raises(RuntimeError, match="visibility map"):
            auto_mgr.start(25, _FakeAutoRunner(25))
    finally:
        vis_mgr.stop(25, force=True)


# ---------- Calibration / CalibrateMotion / NighttimeInteractive ------
#
# The three interactive calibration managers (rotation CalibrationManager,
# CalibrateMotionManager, and the interactive NighttimeCalibrationManager)
# historically cross-checked ONLY the live tracker: a running visibility
# map or nighttime auto-run — or each other — did not block them. They now
# route the whole cross-check through ``raise_if_scope_busy`` so every other
# mount-driving manager blocks them, EXCEPT the documented allowed pairs:
#   - rotation calibration  <-> calibrate-motion   (cal delegates its motion)
#   - nighttime interactive <-> calibrate-motion   (operator nudges via motion)
# Both allowances are symmetric; the tests below assert refusal for every
# non-allowed ordering and success for the two allowed pairs.


class _FakeNighttimeInteractiveSession:
    """Stand-in for the interactive :class:`NighttimeCalibrationSession`.
    ``NighttimeCalibrationManager`` registers the session directly (it does
    not call ``start()``) and reports liveness via ``is_active()``."""

    def __init__(self, telescope_id: int) -> None:
        self.telescope_id = int(telescope_id)
        self._active = True

    def is_active(self) -> bool:
        return self._active

    def stop(self, timeout: float = 5.0) -> None:
        self._active = False

    def status(self):
        return None


# ----- rotation calibration (CalibrationManager) -----


def test_cal_refused_when_visibility_running(monkeypatch):
    """visibility map running -> rotation-cal start refused."""
    import device.rotation_calibration as rc
    import device.visibility_mapper as vm
    from device.rotation_calibration import CalibrationManager
    from device.visibility_mapper import VisibilityMapManager

    vis_mgr = VisibilityMapManager()
    cal_mgr = CalibrationManager()
    monkeypatch.setattr(vm, "_MANAGER", vis_mgr)
    monkeypatch.setattr(rc, "_MANAGER", cal_mgr)
    vis_mgr.start(_FakeVisibilityMapper(700))
    try:
        with pytest.raises(RuntimeError, match="visibility map"):
            cal_mgr.start(_FakeCalSession(700))
    finally:
        vis_mgr.stop(700, force=True)


def test_cal_refused_when_auto_running(monkeypatch):
    """nighttime auto-run running -> rotation-cal start refused."""
    import device.nighttime_calibration as nc
    import device.rotation_calibration as rc
    from device.nighttime_calibration import NighttimeAutoManager
    from device.rotation_calibration import CalibrationManager

    auto_mgr = NighttimeAutoManager()
    cal_mgr = CalibrationManager()
    monkeypatch.setattr(nc, "_AUTO_MANAGER", auto_mgr)
    monkeypatch.setattr(rc, "_MANAGER", cal_mgr)
    auto_mgr.start(701, _FakeAutoRunner(701))
    try:
        with pytest.raises(RuntimeError, match="auto-run"):
            cal_mgr.start(_FakeCalSession(701))
    finally:
        auto_mgr.stop(701)


def test_cal_refused_when_nighttime_interactive_running(monkeypatch):
    """interactive nighttime calibration running -> rotation-cal refused."""
    import device.nighttime_calibration as nc
    import device.rotation_calibration as rc
    from device.nighttime_calibration import NighttimeCalibrationManager
    from device.rotation_calibration import CalibrationManager

    night_mgr = NighttimeCalibrationManager()
    cal_mgr = CalibrationManager()
    monkeypatch.setattr(nc, "_MANAGER", night_mgr)
    monkeypatch.setattr(rc, "_MANAGER", cal_mgr)
    night_mgr.start(_FakeNighttimeInteractiveSession(702))
    try:
        with pytest.raises(RuntimeError, match="interactive nighttime"):
            cal_mgr.start(_FakeCalSession(702))
    finally:
        night_mgr.stop(702)


def test_cal_allowed_when_motion_running(monkeypatch):
    """Documented allowed pair: rotation calibration may start while a
    calibrate-motion session is running (it delegates its motion to it)."""
    import device.calibrate_motion as cm
    import device.rotation_calibration as rc
    from device.calibrate_motion import CalibrateMotionManager
    from device.rotation_calibration import CalibrationManager

    motion_mgr = CalibrateMotionManager()
    cal_mgr = CalibrationManager()
    monkeypatch.setattr(cm, "_MANAGER", motion_mgr)
    monkeypatch.setattr(rc, "_MANAGER", cal_mgr)
    motion_mgr.start(_FakeMotionSession(703))
    try:
        cal_mgr.start(_FakeCalSession(703))
        assert cal_mgr.is_running(703)
    finally:
        cal_mgr.stop(703)
        motion_mgr.stop(703)


# ----- calibrate-motion (CalibrateMotionManager) -----


def test_motion_refused_when_visibility_running(monkeypatch):
    """visibility map running -> calibrate-motion start refused."""
    import device.calibrate_motion as cm
    import device.visibility_mapper as vm
    from device.calibrate_motion import CalibrateMotionManager
    from device.visibility_mapper import VisibilityMapManager

    vis_mgr = VisibilityMapManager()
    motion_mgr = CalibrateMotionManager()
    monkeypatch.setattr(vm, "_MANAGER", vis_mgr)
    monkeypatch.setattr(cm, "_MANAGER", motion_mgr)
    vis_mgr.start(_FakeVisibilityMapper(710))
    try:
        with pytest.raises(RuntimeError, match="visibility map"):
            motion_mgr.start(_FakeMotionSession(710))
    finally:
        vis_mgr.stop(710, force=True)


def test_motion_refused_when_auto_running(monkeypatch):
    """nighttime auto-run running -> calibrate-motion start refused."""
    import device.calibrate_motion as cm
    import device.nighttime_calibration as nc
    from device.calibrate_motion import CalibrateMotionManager
    from device.nighttime_calibration import NighttimeAutoManager

    auto_mgr = NighttimeAutoManager()
    motion_mgr = CalibrateMotionManager()
    monkeypatch.setattr(nc, "_AUTO_MANAGER", auto_mgr)
    monkeypatch.setattr(cm, "_MANAGER", motion_mgr)
    auto_mgr.start(711, _FakeAutoRunner(711))
    try:
        with pytest.raises(RuntimeError, match="auto-run"):
            motion_mgr.start(_FakeMotionSession(711))
    finally:
        auto_mgr.stop(711)


def test_motion_allowed_when_cal_running(monkeypatch):
    """Documented allowed pair (reciprocal): calibrate-motion may start
    while a rotation CalibrationSession is running."""
    import device.calibrate_motion as cm
    import device.rotation_calibration as rc
    from device.calibrate_motion import CalibrateMotionManager
    from device.rotation_calibration import CalibrationManager

    cal_mgr = CalibrationManager()
    motion_mgr = CalibrateMotionManager()
    monkeypatch.setattr(rc, "_MANAGER", cal_mgr)
    monkeypatch.setattr(cm, "_MANAGER", motion_mgr)
    cal_mgr.start(_FakeCalSession(712))
    try:
        motion_mgr.start(_FakeMotionSession(712))
        assert motion_mgr.is_running(712)
    finally:
        motion_mgr.stop(712)
        cal_mgr.stop(712)


def test_motion_allowed_when_nighttime_interactive_running(monkeypatch):
    """Documented allowed pair: calibrate-motion may start while the
    interactive nighttime calibration session is running."""
    import device.calibrate_motion as cm
    import device.nighttime_calibration as nc
    from device.calibrate_motion import CalibrateMotionManager
    from device.nighttime_calibration import NighttimeCalibrationManager

    night_mgr = NighttimeCalibrationManager()
    motion_mgr = CalibrateMotionManager()
    monkeypatch.setattr(nc, "_MANAGER", night_mgr)
    monkeypatch.setattr(cm, "_MANAGER", motion_mgr)
    night_mgr.start(_FakeNighttimeInteractiveSession(713))
    try:
        motion_mgr.start(_FakeMotionSession(713))
        assert motion_mgr.is_running(713)
    finally:
        motion_mgr.stop(713)
        night_mgr.stop(713)


# ----- interactive nighttime calibration (NighttimeCalibrationManager) -----


def test_nighttime_interactive_refused_when_visibility_running(monkeypatch):
    """visibility map running -> interactive nighttime-cal start refused."""
    import device.nighttime_calibration as nc
    import device.visibility_mapper as vm
    from device.nighttime_calibration import NighttimeCalibrationManager
    from device.visibility_mapper import VisibilityMapManager

    vis_mgr = VisibilityMapManager()
    night_mgr = NighttimeCalibrationManager()
    monkeypatch.setattr(vm, "_MANAGER", vis_mgr)
    monkeypatch.setattr(nc, "_MANAGER", night_mgr)
    vis_mgr.start(_FakeVisibilityMapper(720))
    try:
        with pytest.raises(RuntimeError, match="visibility map"):
            night_mgr.start(_FakeNighttimeInteractiveSession(720))
    finally:
        vis_mgr.stop(720, force=True)


def test_nighttime_interactive_refused_when_auto_running(monkeypatch):
    """nighttime auto-run running -> interactive nighttime-cal refused."""
    import device.nighttime_calibration as nc
    from device.nighttime_calibration import (
        NighttimeAutoManager,
        NighttimeCalibrationManager,
    )

    auto_mgr = NighttimeAutoManager()
    night_mgr = NighttimeCalibrationManager()
    monkeypatch.setattr(nc, "_AUTO_MANAGER", auto_mgr)
    monkeypatch.setattr(nc, "_MANAGER", night_mgr)
    auto_mgr.start(721, _FakeAutoRunner(721))
    try:
        with pytest.raises(RuntimeError, match="auto-run"):
            night_mgr.start(_FakeNighttimeInteractiveSession(721))
    finally:
        auto_mgr.stop(721)


def test_nighttime_interactive_refused_when_cal_running(monkeypatch):
    """rotation calibration running -> interactive nighttime-cal refused."""
    import device.nighttime_calibration as nc
    import device.rotation_calibration as rc
    from device.nighttime_calibration import NighttimeCalibrationManager
    from device.rotation_calibration import CalibrationManager

    cal_mgr = CalibrationManager()
    night_mgr = NighttimeCalibrationManager()
    monkeypatch.setattr(rc, "_MANAGER", cal_mgr)
    monkeypatch.setattr(nc, "_MANAGER", night_mgr)
    cal_mgr.start(_FakeCalSession(722))
    try:
        with pytest.raises(RuntimeError, match="calibrating"):
            night_mgr.start(_FakeNighttimeInteractiveSession(722))
    finally:
        cal_mgr.stop(722)


def test_nighttime_interactive_allowed_when_motion_running(monkeypatch):
    """Documented allowed pair: the interactive nighttime session may start
    while a calibrate-motion session is running (it drives the nudges)."""
    import device.calibrate_motion as cm
    import device.nighttime_calibration as nc
    from device.calibrate_motion import CalibrateMotionManager
    from device.nighttime_calibration import NighttimeCalibrationManager

    motion_mgr = CalibrateMotionManager()
    night_mgr = NighttimeCalibrationManager()
    monkeypatch.setattr(cm, "_MANAGER", motion_mgr)
    monkeypatch.setattr(nc, "_MANAGER", night_mgr)
    motion_mgr.start(_FakeMotionSession(723))
    try:
        night_mgr.start(_FakeNighttimeInteractiveSession(723))
        assert night_mgr.is_running(723)
    finally:
        night_mgr.stop(723)
        motion_mgr.stop(723)


# ---------- Directed-pair matrix completion / symmetry ----------------
#
# The refusal tests above cover one direction of most manager pairs. The
# exclusion web must be *symmetric*: for every non-allowed pair, "X running
# -> Y refused" must hold in BOTH orderings. The tests below fill the
# remaining untested directed pairs so all 30 (X, Y) orderings across the six
# mount-driving managers are asserted.
#
# The load-bearing one is live-track <-> interactive nighttime calibration:
# ``NighttimeCalibrationManager.start`` already refused to start while the
# tracker was running, but ``LiveTrackManager.start`` hand-rolls its cross
# checks and originally omitted the interactive-nighttime manager — so the
# reverse ordering (interactive running -> tracker start) slipped through and
# let two controllers drive the same mount. The live tracker now cross-checks
# the interactive manager, and the pair of tests below pin the exclusion in
# both directions.


def test_nighttime_interactive_refused_when_tracker_running(monkeypatch):
    """live-track first -> interactive nighttime-cal start refused (the
    interactive manager cross-checks the tracker)."""
    import device.live_tracker as lt
    import device.nighttime_calibration as nc
    from device.live_tracker import LiveTrackManager
    from device.nighttime_calibration import NighttimeCalibrationManager

    track_mgr = LiveTrackManager()
    night_mgr = NighttimeCalibrationManager()
    monkeypatch.setattr(lt, "_MANAGER", track_mgr)
    monkeypatch.setattr(nc, "_MANAGER", night_mgr)
    track_mgr.start(_FakeTrackerSession(730))
    try:
        with pytest.raises(RuntimeError, match="live-tracking"):
            night_mgr.start(_FakeNighttimeInteractiveSession(730))
    finally:
        track_mgr.stop(730)


def test_tracker_refused_when_nighttime_interactive_running(monkeypatch):
    """Reciprocal (regression for the missing check): interactive nighttime
    calibration first -> live-track on the same scope must be refused. The
    live tracker now cross-checks the interactive nighttime manager, closing
    the interactive <-> live-track exclusion so it holds in both directions."""
    import device.live_tracker as lt
    import device.nighttime_calibration as nc
    from device.live_tracker import LiveTrackManager
    from device.nighttime_calibration import NighttimeCalibrationManager

    track_mgr = LiveTrackManager()
    night_mgr = NighttimeCalibrationManager()
    monkeypatch.setattr(lt, "_MANAGER", track_mgr)
    monkeypatch.setattr(nc, "_MANAGER", night_mgr)
    night_mgr.start(_FakeNighttimeInteractiveSession(731))
    try:
        with pytest.raises(RuntimeError, match="interactive nighttime"):
            track_mgr.start(_FakeTrackerSession(731))
    finally:
        night_mgr.stop(731)


def test_visibility_refused_when_cal_running(monkeypatch):
    """rotation calibration first -> visibility map start refused (reverse of
    test_cal_refused_when_visibility_running)."""
    import device.rotation_calibration as rc
    import device.visibility_mapper as vm
    from device.rotation_calibration import CalibrationManager
    from device.visibility_mapper import VisibilityMapManager

    cal_mgr = CalibrationManager()
    vis_mgr = VisibilityMapManager()
    monkeypatch.setattr(rc, "_MANAGER", cal_mgr)
    monkeypatch.setattr(vm, "_MANAGER", vis_mgr)
    cal_mgr.start(_FakeCalSession(732))
    try:
        with pytest.raises(RuntimeError, match="calibrating"):
            vis_mgr.start(_FakeVisibilityMapper(732))
    finally:
        cal_mgr.stop(732)


def test_auto_refused_when_cal_running(monkeypatch):
    """rotation calibration first -> nighttime auto-run start refused (reverse
    of test_cal_refused_when_auto_running)."""
    import device.nighttime_calibration as nc
    import device.rotation_calibration as rc
    from device.nighttime_calibration import NighttimeAutoManager
    from device.rotation_calibration import CalibrationManager

    cal_mgr = CalibrationManager()
    auto_mgr = NighttimeAutoManager()
    monkeypatch.setattr(rc, "_MANAGER", cal_mgr)
    monkeypatch.setattr(nc, "_AUTO_MANAGER", auto_mgr)
    cal_mgr.start(_FakeCalSession(733))
    try:
        with pytest.raises(RuntimeError, match="calibrating"):
            auto_mgr.start(733, _FakeAutoRunner(733))
    finally:
        cal_mgr.stop(733)


def test_visibility_refused_when_motion_running(monkeypatch):
    """calibrate-motion first -> visibility map start refused (reverse of
    test_motion_refused_when_visibility_running)."""
    import device.calibrate_motion as cm
    import device.visibility_mapper as vm
    from device.calibrate_motion import CalibrateMotionManager
    from device.visibility_mapper import VisibilityMapManager

    motion_mgr = CalibrateMotionManager()
    vis_mgr = VisibilityMapManager()
    monkeypatch.setattr(cm, "_MANAGER", motion_mgr)
    monkeypatch.setattr(vm, "_MANAGER", vis_mgr)
    motion_mgr.start(_FakeMotionSession(734))
    try:
        with pytest.raises(RuntimeError, match="calibrate-motion"):
            vis_mgr.start(_FakeVisibilityMapper(734))
    finally:
        motion_mgr.stop(734)


def test_auto_refused_when_motion_running(monkeypatch):
    """calibrate-motion first -> nighttime auto-run start refused (reverse of
    test_motion_refused_when_auto_running)."""
    import device.calibrate_motion as cm
    import device.nighttime_calibration as nc
    from device.calibrate_motion import CalibrateMotionManager
    from device.nighttime_calibration import NighttimeAutoManager

    motion_mgr = CalibrateMotionManager()
    auto_mgr = NighttimeAutoManager()
    monkeypatch.setattr(cm, "_MANAGER", motion_mgr)
    monkeypatch.setattr(nc, "_AUTO_MANAGER", auto_mgr)
    motion_mgr.start(_FakeMotionSession(735))
    try:
        with pytest.raises(RuntimeError, match="calibrate-motion"):
            auto_mgr.start(735, _FakeAutoRunner(735))
    finally:
        motion_mgr.stop(735)


def test_visibility_refused_when_nighttime_interactive_running(monkeypatch):
    """interactive nighttime calibration first -> visibility map refused
    (reverse of test_nighttime_interactive_refused_when_visibility_running)."""
    import device.nighttime_calibration as nc
    import device.visibility_mapper as vm
    from device.nighttime_calibration import NighttimeCalibrationManager
    from device.visibility_mapper import VisibilityMapManager

    night_mgr = NighttimeCalibrationManager()
    vis_mgr = VisibilityMapManager()
    monkeypatch.setattr(nc, "_MANAGER", night_mgr)
    monkeypatch.setattr(vm, "_MANAGER", vis_mgr)
    night_mgr.start(_FakeNighttimeInteractiveSession(736))
    try:
        with pytest.raises(RuntimeError, match="interactive nighttime"):
            vis_mgr.start(_FakeVisibilityMapper(736))
    finally:
        night_mgr.stop(736)


def test_auto_refused_when_nighttime_interactive_running(monkeypatch):
    """interactive nighttime calibration first -> nighttime auto-run refused
    (reverse of test_nighttime_interactive_refused_when_auto_running)."""
    import device.nighttime_calibration as nc
    from device.nighttime_calibration import (
        NighttimeAutoManager,
        NighttimeCalibrationManager,
    )

    night_mgr = NighttimeCalibrationManager()
    auto_mgr = NighttimeAutoManager()
    monkeypatch.setattr(nc, "_MANAGER", night_mgr)
    monkeypatch.setattr(nc, "_AUTO_MANAGER", auto_mgr)
    night_mgr.start(_FakeNighttimeInteractiveSession(737))
    try:
        with pytest.raises(RuntimeError, match="interactive nighttime"):
            auto_mgr.start(737, _FakeAutoRunner(737))
    finally:
        night_mgr.stop(737)
