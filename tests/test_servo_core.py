"""Golden-trace + unit tests for the shared servo-tick control core.

Two layers of protection for the servo_core extraction (audited migration
step 1):

1. GOLDEN TRACES (this file's ``test_golden_*``). Each of the four
   control-tick sites — the azimuth mover, the elevation mover, the 2-axis
   mover, and the streaming ``track`` loop — is driven through a fully
   deterministic scenario and the exact ``(speed, angle, dur_sec)`` sequence
   it emits to the mount is captured at the ``method_sync`` boundary. The
   sequences are frozen as ``GOLDEN[...]`` constants generated from the
   pre-refactor code. The refactor onto ``device.servo_core.servo_tick`` must
   reproduce them byte-for-byte (integer tuple equality, no float tolerance).

   Determinism comes from a ``FakeClock``: the movers call
   ``time.monotonic``/``time.sleep``/``time.time`` and the ``FakeMountClient``
   integrates its plant on the *same* injected clock, so every firmware
   timestamp, measured position, and reference-time is reproducible. Nothing
   here touches the real wall clock, so the traces do not drift with the time
   of day (the streaming site's sun-safety net is pinned to "safe" for the
   same reason).

2. UNIT TESTS for ``servo_tick`` itself (``test_servo_tick_*``). These pin
   Layer B (the speed/angle quantization) directly, including the two
   deliberately-divergent conventions the golden traces would only exercise
   incidentally: the SIGNED_MOD360 (site 3) vs POS_ADD360_MOD (site 4) angle
   normalization at ``atan2 in (-0.5deg, 0deg)`` and the fine-speed floor that
   zeros ``speed`` while keeping the computed ``angle``.
"""

from __future__ import annotations

import contextlib
import threading

import pytest

from device.plant_limits import AzimuthLimits, CumulativeAzTracker
from device.reference_provider import ReferenceSample
from device.velocity_controller import (
    move_azimuth_to_ff,
    move_elevation_to_ff,
    move_to_ff,
)
from device.streaming_controller import track
from tests.fakes.fake_mount import FakeMountClient


# ---------------------------------------------------------------------------
# Deterministic clock + mount recorder
# ---------------------------------------------------------------------------


class FakeClock:
    """Monotonic clock whose ``now`` only advances on ``sleep``.

    The movers read ``monotonic()``/``time()`` many times per tick (never
    advancing) and pace with a single ``sleep``/``wait`` per tick (advancing
    by exactly the requested delta). The ``FakeMountClient`` integrates its
    plant from the same ``time()``, so the whole loop is reproducible.
    """

    def __init__(self, base: float = 1_000_000.0) -> None:
        self.now = float(base)

    def time(self) -> float:
        return self.now

    def monotonic(self) -> float:
        return self.now

    def sleep(self, dt: float) -> None:
        if dt and dt > 0:
            self.now += float(dt)


class ClockEvent(threading.Event):
    """A stop-signal ``Event`` whose ``wait(timeout)`` advances the clock.

    The streaming controller paces with ``stop_signal.wait(timeout)`` rather
    than ``time.sleep``; routing that through the shared clock keeps the
    streaming trace as deterministic as the velocity-controller traces.
    """

    def __init__(self, clock: FakeClock) -> None:
        super().__init__()
        self._clock = clock

    def wait(self, timeout: float | None = None) -> bool:
        if timeout and timeout > 0:
            self._clock.sleep(timeout)
        return self.is_set()


class RecordingMount:
    """Wraps a ``FakeMountClient`` and records every ``scope_speed_move``.

    Capturing at the ``method_sync`` boundary records BOTH the lockout-aware
    ``speed_move`` wrapper calls AND the raw ``_motor_stop_on_exit`` /
    streaming ``finally`` motor-stops, uniformly. The recorded list of
    ``(speed, angle, dur_sec)`` integer tuples IS the golden trace.
    """

    def __init__(self, inner: FakeMountClient) -> None:
        self._inner = inner
        self.commands: list[tuple[int, int, int]] = []

    def method_sync(self, method: str, params=None):
        if method == "scope_speed_move":
            p = params or {}
            self.commands.append(
                (
                    int(p.get("speed", 0)),
                    int(p.get("angle", 0)),
                    int(p.get("dur_sec", 0)),
                )
            )
        return self._inner.method_sync(method, params)

    def set_position(self, *args, **kwargs) -> None:
        self._inner.set_position(*args, **kwargs)

    @property
    def state(self):
        return self._inner.state


class _AnalyticProvider:
    """Deterministic constant-jerk-free reference for the streaming trace.

    Mirrors ``JsonlECEFProvider``'s in-buffer / extrapolate / stale contract
    (interpolate on ``[t0, t1]``; past ``t1`` extrapolate with the tail
    velocity/acceleration and mark ``stale`` once beyond ``extrapolation_s``;
    before ``t0`` raise ``ValueError``) but with closed-form az/el so the
    streaming trace has no scipy/ECEF dependency and no float platform drift.
    """

    is_live = False
    extrapolation_s = 1.0

    def __init__(
        self,
        t0: float,
        t1: float,
        az0: float,
        el0: float,
        v_az: float,
        v_el: float,
        a_az: float = 0.0,
        a_el: float = 0.0,
    ) -> None:
        self._t0 = float(t0)
        self._t1 = float(t1)
        self._az0 = float(az0)
        self._el0 = float(el0)
        self._v_az = float(v_az)
        self._v_el = float(v_el)
        self._a_az = float(a_az)
        self._a_el = float(a_el)

    def valid_range(self) -> tuple[float, float]:
        return (self._t0, self._t1)

    def _state_at(self, dt: float) -> tuple[float, float, float, float]:
        az = self._az0 + self._v_az * dt + 0.5 * self._a_az * dt * dt
        el = self._el0 + self._v_el * dt + 0.5 * self._a_el * dt * dt
        v_az = self._v_az + self._a_az * dt
        v_el = self._v_el + self._a_el * dt
        return az, el, v_az, v_el

    def sample(self, t_unix: float) -> ReferenceSample:
        t = float(t_unix)
        if t < self._t0:
            raise ValueError(f"query t={t:.3f} before head {self._t0:.3f}")
        if t <= self._t1:
            az, el, v_az, v_el = self._state_at(t - self._t0)
            return ReferenceSample(
                t_unix=t,
                az_cum_deg=az,
                el_deg=el,
                v_az_degs=v_az,
                v_el_degs=v_el,
                a_az_degs2=self._a_az,
                a_el_degs2=self._a_el,
                stale=False,
                extrapolated=False,
            )
        dt_past = t - self._t1
        az, el, v_az, v_el = self._state_at(self._t1 - self._t0)
        az += v_az * dt_past + 0.5 * self._a_az * dt_past * dt_past
        el += v_el * dt_past + 0.5 * self._a_el * dt_past * dt_past
        return ReferenceSample(
            t_unix=t,
            az_cum_deg=az,
            el_deg=el,
            v_az_degs=v_az + self._a_az * dt_past,
            v_el_degs=v_el + self._a_el * dt_past,
            a_az_degs2=self._a_az,
            a_el_degs2=self._a_el,
            stale=dt_past > self.extrapolation_s,
            extrapolated=True,
        )


# A location is required by the mover signatures but unused by the raw-encoder
# measurement path (measure_altaz_timed ``del loc``); any object works.
_LOC = object()


@contextlib.contextmanager
def _install_clock(clock: FakeClock, *, streaming: bool = False):
    """Point the controllers' ``time`` module at the fake clock.

    Restores the real module on exit so ordering between scenarios (and other
    tests in the session) is unaffected.
    """
    import device.velocity_controller as vc

    saved_vc = vc.time
    vc.time = clock
    saved_sc = None
    saved_sun = None
    if streaming:
        import device.streaming_controller as sc

        saved_sc = sc.time
        sc.time = clock
        # Pin the per-tick sun net to "safe" so the streaming trace is
        # independent of the wall-clock date/sun position.
        saved_sun = sc._is_sun_safe
        sc._is_sun_safe = lambda az, el: (True, "")
    try:
        yield
    finally:
        vc.time = saved_vc
        if streaming:
            import device.streaming_controller as sc

            sc.time = saved_sc
            sc._is_sun_safe = saved_sun


def _fresh_mount(clock: FakeClock) -> RecordingMount:
    return RecordingMount(FakeMountClient(time_fn=clock.time))


def _wide_limits() -> AzimuthLimits:
    # Wide enough that no scenario here clips against the cable range.
    return AzimuthLimits(
        ccw_hard_stop_cum_deg=-450.0,
        cw_hard_stop_cum_deg=+450.0,
        padding_deg=15.0,
    )


# ---------------------------------------------------------------------------
# Scenario runners — one per golden trace. Every non-default kwarg is pinned
# here so the fixtures stay stable across the refactor.
# ---------------------------------------------------------------------------


def _run_az_wrapped() -> list[tuple[int, int, int]]:
    clock = FakeClock()
    cli = _fresh_mount(clock)
    cli.set_position(az_deg=0.0, el_deg=20.0)
    with _install_clock(clock):
        move_azimuth_to_ff(
            cli,
            target_az_deg=30.0,
            cur_az_deg=0.0,
            loc=_LOC,
            target_alt_deg=20.0,
            v_max=5.0,
            a_max=4.0,
            j_max=12.0,
            tick_dt=0.5,
            settle_s=1.5,
            profile="scurve",
            kp_pos=0.5,
            v_corr_max=2.0,
            arrive_tolerance_deg=0.3,
            settle_max_s=5.0,
            converged_ticks_required=2,
        )
    return cli.commands


def _run_az_cumulative() -> list[tuple[int, int, int]]:
    clock = FakeClock()
    cli = _fresh_mount(clock)
    cli.set_position(az_deg=0.0, el_deg=20.0)
    tracker = CumulativeAzTracker()
    with _install_clock(clock):
        move_azimuth_to_ff(
            cli,
            target_az_deg=25.0,
            cur_az_deg=0.0,
            loc=_LOC,
            target_alt_deg=20.0,
            v_max=5.0,
            a_max=4.0,
            j_max=12.0,
            tick_dt=0.5,
            settle_s=1.5,
            profile="scurve",
            az_limits=_wide_limits(),
            az_tracker=tracker,
            kp_pos=0.5,
            v_corr_max=2.0,
            arrive_tolerance_deg=0.3,
            settle_max_s=5.0,
            converged_ticks_required=2,
        )
    return cli.commands


def _run_az_fine_floor() -> list[tuple[int, int, int]]:
    clock = FakeClock()
    cli = _fresh_mount(clock)
    cli.set_position(az_deg=0.0, el_deg=20.0)
    with _install_clock(clock):
        move_azimuth_to_ff(
            cli,
            target_az_deg=-0.15,
            cur_az_deg=0.0,
            loc=_LOC,
            target_alt_deg=20.0,
            v_max=5.0,
            a_max=4.0,
            j_max=12.0,
            tick_dt=0.5,
            settle_s=1.5,
            profile="scurve",
            kp_pos=0.5,
            v_corr_max=2.0,
            arrive_tolerance_deg=0.3,
            settle_max_s=2.0,
            converged_ticks_required=2,
        )
    return cli.commands


def _run_el_move() -> list[tuple[int, int, int]]:
    clock = FakeClock()
    cli = _fresh_mount(clock)
    cli.set_position(az_deg=0.0, el_deg=20.0)
    with _install_clock(clock):
        move_elevation_to_ff(
            cli,
            target_el_deg=45.0,
            cur_el_deg=20.0,
            loc=_LOC,
            v_max=5.0,
            a_max=4.0,
            j_max=12.0,
            tick_dt=0.5,
            settle_s=1.5,
            profile="scurve",
            kp_pos=0.5,
            v_corr_max=2.0,
            arrive_tolerance_deg=0.3,
            settle_max_s=5.0,
            converged_ticks_required=2,
        )
    return cli.commands


def _run_el_clamped() -> list[tuple[int, int, int]]:
    clock = FakeClock()
    cli = _fresh_mount(clock)
    cli.set_position(az_deg=0.0, el_deg=20.0)
    with _install_clock(clock):
        move_elevation_to_ff(
            cli,
            target_el_deg=80.0,
            cur_el_deg=20.0,
            loc=_LOC,
            v_max=5.0,
            a_max=4.0,
            j_max=12.0,
            tick_dt=0.5,
            settle_s=1.5,
            profile="scurve",
            el_max_deg=60.0,
            kp_pos=0.5,
            v_corr_max=2.0,
            arrive_tolerance_deg=0.3,
            settle_max_s=5.0,
            converged_ticks_required=2,
        )
    return cli.commands


def _run_el_noop() -> list[tuple[int, int, int]]:
    clock = FakeClock()
    cli = _fresh_mount(clock)
    cli.set_position(az_deg=0.0, el_deg=20.0)
    with _install_clock(clock):
        move_elevation_to_ff(
            cli,
            target_el_deg=20.005,
            cur_el_deg=20.0,
            loc=_LOC,
            profile="scurve",
        )
    return cli.commands


def _run_diag_2d() -> list[tuple[int, int, int]]:
    clock = FakeClock()
    cli = _fresh_mount(clock)
    cli.set_position(az_deg=0.0, el_deg=30.0)
    with _install_clock(clock):
        move_to_ff(
            cli,
            target_az_deg=20.0,
            target_el_deg=45.0,
            cur_az_deg=0.0,
            cur_el_deg=30.0,
            loc=_LOC,
            v_max=5.0,
            a_max=4.0,
            j_max=12.0,
            tick_dt=0.5,
            settle_s=1.5,
            profile="scurve",
            kp_pos=0.5,
            v_corr_max=2.0,
            arrive_tolerance_deg=0.3,
            settle_max_s=5.0,
            converged_ticks_required=2,
        )
    return cli.commands


def _run_diag_2d_cum() -> list[tuple[int, int, int]]:
    clock = FakeClock()
    cli = _fresh_mount(clock)
    cli.set_position(az_deg=0.0, el_deg=30.0)
    tracker = CumulativeAzTracker()
    with _install_clock(clock):
        move_to_ff(
            cli,
            target_az_deg=20.0,
            target_el_deg=45.0,
            cur_az_deg=0.0,
            cur_el_deg=30.0,
            loc=_LOC,
            v_max=5.0,
            a_max=4.0,
            j_max=12.0,
            tick_dt=0.5,
            settle_s=1.5,
            profile="scurve",
            az_limits=_wide_limits(),
            az_tracker=tracker,
            kp_pos=0.5,
            v_corr_max=2.0,
            arrive_tolerance_deg=0.3,
            settle_max_s=5.0,
            converged_ticks_required=2,
        )
    return cli.commands


def _run_unwind_force_cum() -> list[tuple[int, int, int]]:
    clock = FakeClock()
    cli = _fresh_mount(clock)
    wrapped_now = 30.0
    cli.set_position(az_deg=wrapped_now, el_deg=25.0)
    tracker = CumulativeAzTracker()
    tracker.reset(cum_az_deg=30.0, wrapped_az_deg=wrapped_now)
    with _install_clock(clock):
        move_to_ff(
            cli,
            target_az_deg=0.0,
            target_el_deg=25.0,
            cur_az_deg=wrapped_now,
            cur_el_deg=25.0,
            loc=_LOC,
            v_max=5.0,
            a_max=4.0,
            j_max=12.0,
            tick_dt=0.5,
            settle_s=1.5,
            profile="scurve",
            az_limits=_wide_limits(),
            az_tracker=tracker,
            kp_pos=0.5,
            v_corr_max=2.0,
            arrive_tolerance_deg=0.3,
            settle_max_s=5.0,
            converged_ticks_required=2,
            force_cum_az_target=0.0,
        )
    return cli.commands


def _run_stream() -> list[tuple[int, int, int]]:
    clock = FakeClock()
    cli = _fresh_mount(clock)
    t0 = clock.now + 1.0
    t1 = t0 + 3.0
    provider = _AnalyticProvider(
        t0=t0,
        t1=t1,
        az0=10.0,
        el0=45.0,
        v_az=1.0,
        v_el=0.3,
    )
    first = provider.sample(t0)
    cli.set_position(az_deg=first.az_cum_deg, el_deg=first.el_deg)
    tracker = CumulativeAzTracker()
    stop = ClockEvent(clock)
    with _install_clock(clock, streaming=True):
        track(
            cli,
            provider,
            tick_dt=0.5,
            latency_s=0.4,
            tau_s=0.348,
            kp_pos=0.5,
            v_corr_max=2.0,
            v_max=6.0,
            az_limits=None,
            az_tracker=tracker,
            stop_signal=stop,
            max_duration_s=60.0,
        )
    return cli.commands


_RUNNERS = {
    "az_wrapped": _run_az_wrapped,
    "az_cumulative": _run_az_cumulative,
    "az_fine_floor": _run_az_fine_floor,
    "el_move": _run_el_move,
    "el_clamped": _run_el_clamped,
    "el_noop": _run_el_noop,
    "diag_2d": _run_diag_2d,
    "diag_2d_cum": _run_diag_2d_cum,
    "unwind_force_cum": _run_unwind_force_cum,
    "stream": _run_stream,
}


# ---------------------------------------------------------------------------
# GOLDEN traces — generated from the pre-refactor movers. Frozen: only
# regenerate on an intentional, reviewed behavior change.
# ---------------------------------------------------------------------------

GOLDEN: dict[str, list[tuple[int, int, int]]] = {
    "az_wrapped": [
        (0, 0, 5),
        (688, 0, 5),
        (1216, 0, 5),
        (1358, 0, 5),
        (1312, 0, 5),
        (1285, 0, 5),
        (1258, 0, 5),
        (1237, 0, 5),
        (1222, 0, 5),
        (1211, 0, 5),
        (1203, 0, 5),
        (1199, 0, 5),
        (1195, 0, 5),
        (506, 0, 5),
        (24, 180, 5),
        (167, 180, 5),
        (121, 180, 5),
        (95, 180, 5),
        (69, 180, 5),
        (47, 180, 5),
        (32, 180, 5),
        (21, 180, 5),
        (0, 0, 1),
    ],
    "az_cumulative": [
        (0, 0, 5),
        (688, 0, 5),
        (1216, 0, 5),
        (1358, 0, 5),
        (1312, 0, 5),
        (1285, 0, 5),
        (1258, 0, 5),
        (1237, 0, 5),
        (1222, 0, 5),
        (1211, 0, 5),
        (1203, 0, 5),
        (511, 0, 5),
        (21, 180, 5),
        (165, 180, 5),
        (120, 180, 5),
        (94, 180, 5),
        (68, 180, 5),
        (47, 180, 5),
        (31, 180, 5),
        (21, 180, 5),
        (0, 0, 1),
    ],
    "az_fine_floor": [
        (0, 0, 5),
        (0, 180, 5),
        (0, 180, 5),
        (0, 0, 1),
    ],
    "el_move": [
        (0, 0, 5),
        (688, 90, 5),
        (1216, 90, 5),
        (1358, 90, 5),
        (1312, 90, 5),
        (1285, 90, 5),
        (1258, 90, 5),
        (1237, 90, 5),
        (1222, 90, 5),
        (1211, 90, 5),
        (1203, 90, 5),
        (511, 90, 5),
        (21, 270, 5),
        (165, 270, 5),
        (120, 270, 5),
        (94, 270, 5),
        (68, 270, 5),
        (47, 270, 5),
        (31, 270, 5),
        (21, 270, 5),
        (0, 0, 1),
    ],
    "el_clamped": [
        (0, 0, 5),
        (688, 90, 5),
        (1216, 90, 5),
        (1358, 90, 5),
        (1312, 90, 5),
        (1285, 90, 5),
        (1258, 90, 5),
        (1237, 90, 5),
        (1222, 90, 5),
        (1211, 90, 5),
        (1203, 90, 5),
        (1199, 90, 5),
        (1195, 90, 5),
        (1193, 90, 5),
        (1192, 90, 5),
        (1191, 90, 5),
        (1191, 90, 5),
        (503, 90, 5),
        (26, 270, 5),
        (168, 270, 5),
        (122, 270, 5),
        (95, 270, 5),
        (69, 270, 5),
        (48, 270, 5),
        (32, 270, 5),
        (21, 270, 5),
        (0, 0, 1),
    ],
    "el_noop": [],
    "diag_2d": [
        (0, 0, 5),
        (859, 37, 5),
        (1520, 37, 5),
        (1698, 37, 5),
        (1640, 37, 5),
        (1606, 37, 5),
        (1573, 37, 5),
        (1547, 37, 5),
        (1527, 37, 5),
        (654, 36, 5),
        (0, 236, 5),
        (202, 218, 5),
        (149, 218, 5),
        (118, 218, 5),
        (85, 218, 5),
        (59, 219, 5),
        (40, 219, 5),
        (26, 219, 5),
        (0, 0, 1),
    ],
    "diag_2d_cum": [
        (0, 0, 5),
        (859, 37, 5),
        (1520, 37, 5),
        (1698, 37, 5),
        (1640, 37, 5),
        (1606, 37, 5),
        (1573, 37, 5),
        (1547, 37, 5),
        (1527, 37, 5),
        (654, 36, 5),
        (0, 236, 5),
        (202, 218, 5),
        (149, 218, 5),
        (118, 218, 5),
        (85, 218, 5),
        (59, 219, 5),
        (40, 219, 5),
        (26, 219, 5),
        (0, 0, 1),
    ],
    "unwind_force_cum": [
        (0, 0, 5),
        (688, 180, 5),
        (1216, 180, 5),
        (1358, 180, 5),
        (1312, 180, 5),
        (1285, 180, 5),
        (1258, 180, 5),
        (1237, 180, 5),
        (1222, 180, 5),
        (1211, 180, 5),
        (1203, 180, 5),
        (1199, 180, 5),
        (1195, 180, 5),
        (506, 180, 5),
        (24, 0, 5),
        (167, 0, 5),
        (121, 0, 5),
        (95, 0, 5),
        (69, 0, 5),
        (47, 0, 5),
        (32, 0, 5),
        (21, 0, 5),
        (0, 0, 1),
    ],
    "stream": [
        (247, 17, 1),
        (280, 17, 1),
        (285, 17, 1),
        (279, 17, 1),
        (272, 16, 1),
        (265, 16, 1),
        (260, 17, 1),
        (256, 17, 1),
        (253, 17, 1),
        (0, 0, 1),
    ],
}


@pytest.mark.parametrize("name", sorted(_RUNNERS))
def test_golden(name: str) -> None:
    expected = GOLDEN[name]
    got = _RUNNERS[name]()
    assert got == expected, (
        f"{name}: servo trace drifted from golden.\n"
        f"  expected ({len(expected)}): {expected}\n"
        f"  got      ({len(got)}): {got}"
    )


# ---------------------------------------------------------------------------
# Direct unit tests for the pure core. These pin Layer B (quantization)
# independently of the movers, including the two deliberately-divergent
# conventions the golden traces only exercise incidentally.
# ---------------------------------------------------------------------------

import math  # noqa: E402 — kept with the servo_tick unit tests it supports

from device.servo_core import (  # noqa: E402
    AngleConv,
    AxisMode,
    MagFn,
    TickCommand,
    servo_tick,
)


def _tick(**overrides):
    """servo_tick with a benign default gain set; override per-case."""
    kw = dict(
        ref_vel_az=0.0,
        ref_acc_az=0.0,
        err_az=0.0,
        ref_vel_el=0.0,
        ref_acc_el=0.0,
        err_el=0.0,
        kp_pos=0.5,
        v_corr_max=2.0,
        tau_s=0.348,
        v_limit=6.0,
        axis_mode=AxisMode.AZ,
    )
    kw.update(overrides)
    return servo_tick(**kw)


def test_servo_tick_az_angle_sign():
    """AZ: positive command -> angle 0, negative -> angle 180."""
    pos, _ = _tick(ref_vel_az=1.0, axis_mode=AxisMode.AZ)
    neg, _ = _tick(ref_vel_az=-1.0, axis_mode=AxisMode.AZ)
    assert pos == TickCommand(speed=237, angle=0, dur_sec=5)
    assert neg == TickCommand(speed=237, angle=180, dur_sec=5)


def test_servo_tick_el_angle_sign():
    """EL: positive command -> angle 90, negative -> angle 270."""
    pos, _ = _tick(ref_vel_el=1.0, axis_mode=AxisMode.EL)
    neg, _ = _tick(ref_vel_el=-1.0, axis_mode=AxisMode.EL)
    assert pos == TickCommand(speed=237, angle=90, dur_sec=5)
    assert neg == TickCommand(speed=237, angle=270, dur_sec=5)


def test_servo_tick_feedforward_and_corr_compose():
    """v_cmd = (ref_vel + tau*ref_acc) + clamp(kp*err). Diagnostics expose
    each term the wrappers log."""
    cmd, diag = _tick(
        ref_vel_az=1.0,
        ref_acc_az=2.0,
        err_az=0.4,
        axis_mode=AxisMode.AZ,
    )
    assert diag.v_ff_az == pytest.approx(1.0 + 0.348 * 2.0)
    assert diag.v_corr_az == pytest.approx(0.5 * 0.4)
    assert diag.v_cmd_az == pytest.approx((1.0 + 0.348 * 2.0) + 0.2)
    assert cmd.speed == int(round(diag.v_cmd_az * 237))
    assert cmd.angle == 0


def test_servo_tick_v_corr_clamped():
    """The feedback term saturates at +/-v_corr_max regardless of error size."""
    _, hi = _tick(err_az=1000.0, v_corr_max=2.0, axis_mode=AxisMode.AZ)
    _, lo = _tick(err_az=-1000.0, v_corr_max=2.0, axis_mode=AxisMode.AZ)
    assert hi.v_corr_az == pytest.approx(2.0)
    assert lo.v_corr_az == pytest.approx(-2.0)


def test_servo_tick_v_limit_clamp_and_sat_flags():
    """The command saturates at +/-v_limit; sat flags mark the pre-clamp
    overflow (a streaming-only statistic)."""
    _, diag = _tick(
        ref_vel_az=10.0,
        ref_vel_el=-9.0,
        axis_mode=AxisMode.VECTOR,
        v_limit=6.0,
    )
    assert diag.v_cmd_az == pytest.approx(6.0)
    assert diag.v_cmd_el == pytest.approx(-6.0)
    assert diag.sat_az is True
    assert diag.sat_el is True
    # A within-limit command does not set the flag.
    _, ok = _tick(ref_vel_az=1.0, axis_mode=AxisMode.VECTOR)
    assert ok.sat_az is False


def test_servo_tick_deadband_full_stop():
    """|v_cmd| below the deadband -> a full stop (speed 0, angle 0)."""
    cmd, _ = _tick(axis_mode=AxisMode.AZ)  # all-zero inputs
    assert cmd == TickCommand(speed=0, angle=0, dur_sec=5)


def test_servo_tick_fine_floor_zeros_speed_but_keeps_angle():
    """Sites 1-3: below the fine-finish floor, speed drops to 0 but the
    computed angle is retained (here 180, a negative az command)."""
    # v_cmd_az = kp*err = 0.5 * -0.1 = -0.05 deg/s -> speed round(11.85)=12 < 20.
    on, _ = _tick(err_az=-0.1, axis_mode=AxisMode.AZ, fine_min_speed=20)
    off, _ = _tick(err_az=-0.1, axis_mode=AxisMode.AZ, fine_min_speed=None)
    assert on == TickCommand(speed=0, angle=180, dur_sec=5)
    assert off == TickCommand(speed=12, angle=180, dur_sec=5)


def test_servo_tick_angle_convention_divergence():
    """The reachable SIGNED_MOD360 (site 3) vs POS_ADD360_MOD (site 4)
    divergence: for v_cmd_az > 0 and atan2 landing in (-0.5deg, 0deg), site 3
    emits 0 and site 4 emits 360. This is why the flag stays per-caller."""
    v_el = math.tan(math.radians(-0.3))  # atan2(v_el, 1.0) == -0.3 deg
    kw = dict(
        ref_vel_az=1.0,
        ref_vel_el=v_el,
        axis_mode=AxisMode.VECTOR,
    )
    signed, _ = _tick(angle_conv=AngleConv.SIGNED_MOD360, mag_fn=MagFn.SUMSQ, **kw)
    pos_add, _ = _tick(angle_conv=AngleConv.POS_ADD360_MOD, mag_fn=MagFn.HYPOT, **kw)
    assert signed.angle == 0
    assert pos_add.angle == 360
    # Same magnitude either way — only the normalization differs.
    assert signed.speed == pos_add.speed


def test_servo_tick_sumsq_and_hypot_agree_on_speed():
    """SUMSQ (site 3) and HYPOT (site 4) magnitude paths resolve to the same
    firmware speed for an ordinary diagonal command."""
    kw = dict(ref_vel_az=1.0, ref_vel_el=0.3, axis_mode=AxisMode.VECTOR)
    sumsq, d1 = _tick(mag_fn=MagFn.SUMSQ, **kw)
    hyp, d2 = _tick(mag_fn=MagFn.HYPOT, **kw)
    assert sumsq.speed == hyp.speed
    assert sumsq.angle == hyp.angle


def test_servo_tick_dur_sec_passthrough():
    """dur_sec is echoed unchanged (firmware TTL: 5 s point-to-point, 1 s
    streaming)."""
    five, _ = _tick(ref_vel_az=1.0, dur_sec=5)
    one, _ = _tick(ref_vel_az=1.0, dur_sec=1)
    assert five.dur_sec == 5
    assert one.dur_sec == 1
