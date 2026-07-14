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
