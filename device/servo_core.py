"""Shared servo-tick control core for the Seestar velocity movers.

Every closed-loop mover in this codebase re-issues a firmware
``scope_speed_move(speed, angle, dur_sec)`` each tick, computed from the
*same* feedforward-plus-feedback control law:

    v_corr = clamp(kp_pos * err,  ±v_corr_max)   # P feedback
    v_ff   = ref_vel + tau_s * ref_acc            # first-order plant inversion
    v_cmd  = clamp(v_ff + v_corr, ±v_limit)       # saturated command
    speed  = round(|v_cmd| * speed_per_degs)      # firmware speed unit
    (deadband: |v_cmd| < 1e-6 -> a full stop)

The four call sites — :func:`device.velocity_controller.move_azimuth_to_ff`
(AZ), :func:`~device.velocity_controller.move_elevation_to_ff` (EL),
:func:`~device.velocity_controller.move_to_ff` (2-axis vector), and
:func:`device.streaming_controller.track` (2-axis vector, live provider) —
share that law byte-for-byte. What legitimately differs per site is only the
*quantization convention* (how ``(speed, angle)`` are derived from
``v_cmd``), which this module captures as a handful of enum flags rather than
duplicated arithmetic. Those divergences are deliberate and verified (see the
golden traces in ``tests/test_servo_core.py``); they are parameters, never
bugs to unify:

* **axis_mode** — AZ emits angle 0/180, EL emits angle 90/270, VECTOR emits
  ``atan2(v_el, v_az)``.
* **angle_conv** — VECTOR sites differ in normalization. Site 3 (``move_to_ff``)
  uses ``round(deg(atan2)) % 360``; site 4 (``track``) uses
  ``round((deg(atan2) + 360) % 360)``. For ``v_cmd_az > 0`` with
  ``v_cmd_el in (-0.5deg, 0deg)`` these disagree (0 vs 360), so the flag must
  stay per-caller.
* **mag_fn** — site 3 uses ``sqrt(az^2 + el^2)``; site 4 uses ``hypot``. They
  agree to the emitted integer across the whole in-range rate domain but the
  flag is kept as zero-cost insurance.
* **fine_min_speed** — sites 1-3 zero the command when ``speed`` falls below
  the fine-finish floor (``VC_FINE_MIN_SPEED``) yet *keep* the computed angle;
  site 4 has no such floor (``None``).
* **dur_sec** — the firmware TTL: 5 s for the point-to-point movers, 1 s for
  the streaming loop.

Everything time-, I/O-, or safety-related (measurement, reference derivation,
error framing, the issue gate, ``speed_move`` / ``_motor_stop_on_exit``,
logging, settle/exit bookkeeping, deadline pacing) stays in each caller's thin
wrapper. This module is intentionally pure: only ``math`` + enums +
dataclasses, no time, no I/O, no logging, no numpy — which makes it trivially
unit-testable and deterministic, a requirement for the golden traces.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum


# Firmware calibration: rate deg/s -> firmware speed unit (speed / 237 = deg/s;
# verified +/-1% over speed 80..1440). Duplicated verbatim from
# ``device.velocity_controller.SPEED_PER_DEG_PER_SEC`` so this module stays
# free of any import from the I/O-heavy controller (and any import cycle).
SPEED_PER_DEG_PER_SEC = 237


class AxisMode(Enum):
    """Which axis (or vector) the command angle encodes."""

    AZ = "az"  # site 1: angle 0 (+az) / 180 (-az)
    EL = "el"  # site 2: angle 90 (+el) / 270 (-el)
    VECTOR = "vector"  # sites 3, 4: angle = atan2(v_el, v_az)


class AngleConv(Enum):
    """VECTOR-only atan2 normalization convention (see module docstring)."""

    SIGNED_MOD360 = "signed_mod360"  # site 3: round(deg(atan2)) % 360
    POS_ADD360_MOD = "pos_add360_mod"  # site 4: round((deg(atan2) + 360) % 360)


class MagFn(Enum):
    """VECTOR-only magnitude formula."""

    SUMSQ = "sumsq"  # site 3: sqrt(az*az + el*el)
    HYPOT = "hypot"  # site 4: hypot(az, el)  (== np.hypot for scalar doubles)


@dataclass(frozen=True)
class TickCommand:
    """The firmware command a tick resolves to."""

    speed: int
    angle: int
    dur_sec: int


@dataclass(frozen=True)
class TickDiag:
    """Intermediate quantities a wrapper needs for its own logs / stats.

    Pass-through so each caller keeps its exact ``position_logger`` event
    fields and stats accounting (e.g. ``max_v_corr_degs``, the streaming
    saturation counters) without re-deriving anything the core already
    computed. ``sat_az`` / ``sat_el`` mark that the pre-clamp command exceeded
    ``v_limit`` (a streaming-only statistic; the point-to-point movers ignore
    them).
    """

    v_ff_az: float
    v_ff_el: float
    v_corr_az: float
    v_corr_el: float
    v_cmd_az: float
    v_cmd_el: float
    v_mag: float
    sat_az: bool
    sat_el: bool


def _clip(x: float, lo: float, hi: float) -> float:
    """Clamp ``x`` to ``[lo, hi]`` — the shared saturation primitive."""
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


def _angle_for(
    axis_mode: AxisMode,
    angle_conv: AngleConv,
    v_cmd_az: float,
    v_cmd_el: float,
) -> int:
    """Reproduce each site's angle quantization EXACTLY.

    Only reached on a non-deadband command, so the relevant velocity
    component is non-zero and its sign is well-defined.
    """
    if axis_mode is AxisMode.AZ:
        return 0 if v_cmd_az > 0 else 180
    if axis_mode is AxisMode.EL:
        return 90 if v_cmd_el > 0 else 270
    deg = math.degrees(math.atan2(v_cmd_el, v_cmd_az))
    if angle_conv is AngleConv.POS_ADD360_MOD:
        return int(round((deg + 360.0) % 360.0))
    return int(round(deg)) % 360


def servo_tick(
    *,
    ref_vel_az: float,
    ref_acc_az: float,
    err_az: float,
    ref_vel_el: float,
    ref_acc_el: float,
    err_el: float,
    kp_pos: float,
    v_corr_max: float,
    tau_s: float,
    v_limit: float,
    axis_mode: AxisMode,
    angle_conv: AngleConv = AngleConv.SIGNED_MOD360,
    mag_fn: MagFn = MagFn.SUMSQ,
    fine_min_speed: int | None = None,
    dur_sec: int = 5,
    speed_per_degs: int = SPEED_PER_DEG_PER_SEC,
    deadband: float = 1e-6,
) -> tuple[TickCommand, TickDiag]:
    """Compute one tick's firmware command from the shared FF+FB law.

    The caller computes ``err_az`` / ``err_el`` in whatever frame it uses
    (az-wrapped, cumulative, raw-elevation, or anchored streaming) and passes
    the reference velocity/acceleration for each axis. For a single-axis mover
    the unused axis passes ``0.0, 0.0, 0.0``.

    Returns ``(TickCommand, TickDiag)``. Byte-for-byte identical to the
    open-coded law at each of the four sites — see ``tests/test_servo_core.py``
    for the golden traces and the direct convention unit tests.
    """
    # ---- Layer A: control law (identical across all four sites) ----
    v_corr_az = _clip(kp_pos * err_az, -v_corr_max, v_corr_max)
    v_corr_el = _clip(kp_pos * err_el, -v_corr_max, v_corr_max)
    v_ff_az = ref_vel_az + tau_s * ref_acc_az
    v_ff_el = ref_vel_el + tau_s * ref_acc_el
    raw_az = v_ff_az + v_corr_az
    raw_el = v_ff_el + v_corr_el
    sat_az = abs(raw_az) > v_limit
    sat_el = abs(raw_el) > v_limit
    v_cmd_az = _clip(raw_az, -v_limit, v_limit)
    v_cmd_el = _clip(raw_el, -v_limit, v_limit)

    # ---- Layer B: quantize to (speed, angle) per the caller's convention ----
    if axis_mode is AxisMode.AZ:
        v_mag = abs(v_cmd_az)
    elif axis_mode is AxisMode.EL:
        v_mag = abs(v_cmd_el)
    elif mag_fn is MagFn.HYPOT:
        v_mag = math.hypot(v_cmd_az, v_cmd_el)
    else:
        v_mag = math.sqrt(v_cmd_az * v_cmd_az + v_cmd_el * v_cmd_el)

    if v_mag < deadband:
        speed = 0
        angle = 0
    else:
        speed = int(round(v_mag * speed_per_degs))
        if fine_min_speed is not None and speed < fine_min_speed:
            # Fine-finish floor: drop below-stiction speed to a full stop but
            # KEEP the computed angle (sites 1-3).
            speed = 0
        angle = _angle_for(axis_mode, angle_conv, v_cmd_az, v_cmd_el)

    return (
        TickCommand(speed=speed, angle=angle, dur_sec=dur_sec),
        TickDiag(
            v_ff_az=v_ff_az,
            v_ff_el=v_ff_el,
            v_corr_az=v_corr_az,
            v_corr_el=v_corr_el,
            v_cmd_az=v_cmd_az,
            v_cmd_el=v_cmd_el,
            v_mag=v_mag,
            sat_az=sat_az,
            sat_el=sat_el,
        ),
    )
