"""Feasible velocity-profile trajectories for the Seestar velocity plant.

Produces a `PlannedTrajectory` — a sampled sequence of
(t, pos, vel, acc) points satisfying plant limits from Phase 1:

- rate cap `v_max` = 6 °/s (firmware clamp, `MAIN_RATE_DEGS`)
- accel cap `a_max` ~ 10 °/s² (Phase 1 fit: rate-limit bound was
  never reached with this cap during step-response training,
  suggesting it's within the plant's achievable envelope)
- jerk cap `j_max` (S-curve only)

Two profile forms:
- `trapezoidal_profile`: accel → cruise → decel. Simplest and
  fastest point-to-point for |delta| large enough to reach v_max.
- `scurve_profile`: jerk-limited accel/decel for smoother
  commanded-speed transitions, more forgiving on the firmware ramp.

Both correctly handle azimuth wrap-around via `wrap_pm180` on
`delta = p_target - p0`, so they always pick the shorter path
through the ±180° boundary.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from device.velocity_controller import wrap_pm180


@dataclass(frozen=True)
class TrajectoryPoint:
    t: float  # seconds from trajectory start
    pos: float  # signed azimuth position (may exceed [-180, +180); use
    # wrap_pm180 when comparing to measured)
    vel: float  # signed rate (deg/s)
    acc: float  # signed accel (deg/s^2)


@dataclass
class PlannedTrajectory:
    points: tuple[TrajectoryPoint, ...]
    total_duration: float

    def sample(self, t: float) -> TrajectoryPoint:
        """Linear-interpolated sample at arbitrary time.

        Before t=0 returns the first point; past total_duration returns
        the last point. Between samples, linearly interpolates pos/vel/acc
        against the surrounding two points.
        """
        if not self.points:
            raise ValueError("empty trajectory")
        if t <= self.points[0].t:
            return self.points[0]
        if t >= self.points[-1].t:
            return self.points[-1]
        # binary search for the bracketing interval
        lo, hi = 0, len(self.points) - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if self.points[mid].t <= t:
                lo = mid
            else:
                hi = mid
        a, b = self.points[lo], self.points[hi]
        if b.t == a.t:
            return a
        f = (t - a.t) / (b.t - a.t)
        return TrajectoryPoint(
            t=t,
            pos=a.pos + f * (b.pos - a.pos),
            vel=a.vel + f * (b.vel - a.vel),
            acc=a.acc + f * (b.acc - a.acc),
        )


_POS_EPS_DEG = 0.01  # below this, consider "already at target"


def _path_crosses_forbidden(
    p0: float, delta_signed: float, az_forbidden: float
) -> bool:
    """Does traveling from p0 by signed-delta cross az_forbidden (interior)?

    Crossing means the forbidden bearing is reached at some travel distance
    strictly inside ``(0, |delta_signed|)``. Endpoint-equal doesn't count
    (the trajectory stops before the forbidden angle), and the start point
    doesn't count either.

    Correct for multi-wrap paths (``|delta_signed| > 180``): rather than
    wrapping the offset into a single ±180° window — which silently misses
    forbidden bearings that sit in the far arc or on a second lap — we
    accumulate signed arc coverage across the full ``|delta_signed|``. The
    forbidden bearing (any representative mod 360) is hit at travel
    distances ``d, d + 360, d + 720, …`` where ``d`` is the offset measured
    in the direction of travel and reduced to ``[0, 360)``.
    """
    if delta_signed == 0:
        return False
    direction = 1.0 if delta_signed > 0 else -1.0
    # Offset from start to the forbidden bearing, measured in the direction
    # of travel and reduced to [0, 360).
    d = ((az_forbidden - p0) * direction) % 360.0
    abs_delta = abs(delta_signed)
    # A d of 0 means the forbidden bearing coincides with the start; the
    # first interior hit is then one full lap away (360), not at distance 0.
    first_hit = d if d > 1e-9 else 360.0
    return first_hit < abs_delta - 1e-9


def _select_delta(
    p0: float,
    p_target: float,
    az_forbidden: float | None,
) -> float:
    """Compute the signed delta from p0 to p_target, avoiding az_forbidden
    if set. Returns a delta in the shorter path whenever that's feasible;
    otherwise, the long way (|delta| > 180).
    """
    short_delta = wrap_pm180(p_target - p0)
    if az_forbidden is None:
        return short_delta
    if not _path_crosses_forbidden(p0, short_delta, az_forbidden):
        return short_delta
    # Short path crosses the forbidden azimuth — take the long way.
    long_delta = short_delta - 360.0 if short_delta >= 0 else short_delta + 360.0
    if _path_crosses_forbidden(p0, long_delta, az_forbidden):
        # Impossible geometrically (only occurs if p0 or p_target sits
        # exactly on the forbidden angle); prefer the short path and let
        # the caller decide.
        return short_delta
    return long_delta


def _apply_t_offset(
    traj: PlannedTrajectory,
    t_offset: float,
    tick_dt: float,
) -> PlannedTrajectory:
    """Prepend a hold phase of `t_offset` seconds at (p0, v=0, a=0) and shift
    every existing sample by `+t_offset`.

    Models the cold-start dead time between the FF controller starting its
    wall clock and the plant actually accepting its first motion command.
    During that window the trajectory reports a static hold at the starting
    position, so the FF controller issues a zero/idle command instead of
    chasing a reference that has already begun accelerating.
    """
    if t_offset <= 0.0 or not traj.points:
        return traj
    p0 = traj.points[0].pos
    hold_points: list[TrajectoryPoint] = []
    k = 0
    while k * tick_dt < t_offset - 1e-9:
        hold_points.append(
            TrajectoryPoint(
                t=k * tick_dt,
                pos=p0,
                vel=0.0,
                acc=0.0,
            )
        )
        k += 1
    shifted = tuple(
        TrajectoryPoint(
            t=p.t + t_offset,
            pos=p.pos,
            vel=p.vel,
            acc=p.acc,
        )
        for p in traj.points
    )
    return PlannedTrajectory(
        points=tuple(hold_points) + shifted,
        total_duration=traj.total_duration + t_offset,
    )


def _sample_phase(
    out: list,
    t_start: float,
    p_start: float,
    v_start: float,
    a: float,
    dur: float,
    tick_dt: float,
    include_endpoint: bool,
) -> tuple[float, float, float]:
    """Append `TrajectoryPoint`s from t_start for dur seconds under
    constant acceleration `a`, at `tick_dt` intervals.

    Returns (t_end, p_end, v_end). If `include_endpoint`, also appends
    the exact endpoint point even when the last tick doesn't line up.
    """
    if dur <= 0:
        if include_endpoint:
            out.append(TrajectoryPoint(t=t_start, pos=p_start, vel=v_start, acc=a))
        return t_start, p_start, v_start

    n_ticks = max(1, int(math.floor(dur / tick_dt)))
    for k in range(1, n_ticks + 1):
        dt = min(k * tick_dt, dur)
        pos = p_start + v_start * dt + 0.5 * a * dt * dt
        vel = v_start + a * dt
        out.append(TrajectoryPoint(t=t_start + dt, pos=pos, vel=vel, acc=a))
    # ensure we hit the exact endpoint
    if include_endpoint and (n_ticks * tick_dt) < dur - 1e-9:
        pos = p_start + v_start * dur + 0.5 * a * dur * dur
        vel = v_start + a * dur
        out.append(TrajectoryPoint(t=t_start + dur, pos=pos, vel=vel, acc=a))
    t_end = t_start + dur
    p_end = p_start + v_start * dur + 0.5 * a * dur * dur
    v_end = v_start + a * dur
    return t_end, p_end, v_end


def _plan_trapezoid_phases(
    v0: float,
    delta: float,
    v_max: float,
    a_max: float,
) -> list[tuple[float, float]]:
    """Bang-bang accel schedule that starts at velocity ``v0``, achieves a
    net signed displacement ``delta``, and ends at ``v = 0``, respecting
    ``|v| <= v_max`` and ``|a| <= a_max``.

    Returns a list of ``(accel, duration)`` phases (durations > 0). Works
    in a single continuous coordinate — no wrapping — so cumulative targets
    beyond ±180° and any pre-chosen (short/long) path are preserved by the
    caller passing the already-resolved signed ``delta``.

    Handles arbitrary ``v0``: over-speed (|v0| > v_max), same-direction
    (accelerate/cruise/decel straight to the target), opposing-direction or
    too-fast-to-stop-in-range (brake to rest, then a rest-to-rest move that
    may reverse to land exactly on the target). In every case the terminal
    velocity is zero on the target.
    """
    phases: list[tuple[float, float]] = []
    v = v0
    x = 0.0  # displacement accumulated so far

    # Pre-phase: shed excess speed down to v_max, keeping v0's direction.
    if abs(v) > v_max:
        dur = (abs(v) - v_max) / a_max
        a = -math.copysign(a_max, v)
        x += v * dur + 0.5 * a * dur * dur
        v = math.copysign(v_max, v)
        phases.append((a, dur))

    remaining = delta - x
    # Signed displacement if we braked to rest starting from the current v.
    d_stop = v * abs(v) / (2.0 * a_max)
    same_dir = (v > 0 and remaining > 0) or (v < 0 and remaining < 0)

    if v != 0.0 and same_dir and abs(d_stop) <= abs(remaining) + 1e-12:
        # Enough room to keep going in v's direction: accel (maybe) to a
        # peak, optional cruise, decel to 0, landing on the target.
        dir_sign = 1.0 if remaining > 0 else -1.0
        rem_abs = abs(remaining)
        v_in = abs(v)
        v_p = math.sqrt(max(0.0, (2.0 * a_max * rem_abs + v_in * v_in) / 2.0))
        if v_p <= v_max:
            t_accel = (v_p - v_in) / a_max
            if t_accel > 1e-12:
                phases.append((dir_sign * a_max, t_accel))
            phases.append((-dir_sign * a_max, v_p / a_max))
        else:
            t_accel = (v_max - v_in) / a_max
            t_decel = v_max / a_max
            d_accel = (v_max * v_max - v_in * v_in) / (2.0 * a_max)
            d_decel = (v_max * v_max) / (2.0 * a_max)
            d_cruise = rem_abs - d_accel - d_decel
            if t_accel > 1e-12:
                phases.append((dir_sign * a_max, t_accel))
            if d_cruise > 1e-12:
                phases.append((0.0, d_cruise / v_max))
            phases.append((-dir_sign * a_max, t_decel))
        return phases

    # Either at rest, moving the wrong way, or moving too fast to stop
    # within range: brake to rest first, then a rest-to-rest move covers
    # whatever displacement remains (reversing if the brake overshot).
    if v != 0.0:
        dur = abs(v) / a_max
        a = -math.copysign(a_max, v)
        x += v * dur + 0.5 * a * dur * dur
        v = 0.0
        phases.append((a, dur))

    remaining = delta - x
    if abs(remaining) > 1e-12:
        dir_sign = 1.0 if remaining >= 0 else -1.0
        rem_abs = abs(remaining)
        v_p = math.sqrt(max(0.0, a_max * rem_abs))  # rest-to-rest peak
        if v_p <= v_max:
            phases.append((dir_sign * a_max, v_p / a_max))
            phases.append((-dir_sign * a_max, v_p / a_max))
        else:
            t_ramp = v_max / a_max
            d_ramp = v_max * v_max / (2.0 * a_max)
            d_cruise = rem_abs - 2.0 * d_ramp
            phases.append((dir_sign * a_max, t_ramp))
            if d_cruise > 1e-12:
                phases.append((0.0, d_cruise / v_max))
            phases.append((-dir_sign * a_max, t_ramp))
    return phases


def trapezoidal_profile(
    p0: float,
    v0: float,
    p_target: float,
    v_max: float,
    a_max: float,
    tick_dt: float = 0.1,
    t_offset: float = 0.0,
    az_forbidden_deg: float | None = None,
    wrap_target: bool = True,
) -> PlannedTrajectory:
    """Accel → cruise → decel profile that reaches `p_target` with
    v_end = 0 under |v| ≤ v_max and |acc| ≤ a_max.

    When `wrap_target=True` (default), `delta = wrap_pm180(p_target - p0)`
    picks the shorter path through the ±180° boundary, and
    `az_forbidden_deg` (if set) forces the long way when the short path
    would cross a forbidden azimuth.

    When `wrap_target=False`, the planner treats `p0` and `p_target` as
    **cumulative / unwrapped** positions and uses `delta = p_target - p0`
    verbatim. This is the right mode for multi-turn cable-wrap planning
    where the caller has already picked which cumulative target to reach
    (see `device.plant_limits.pick_cum_target`). `az_forbidden_deg` is
    ignored in this mode.

    Non-zero `v0` is handled by `_plan_trapezoid_phases`: the profile
    always terminates at `v = 0` exactly on the target, whether `v0` is in
    the direction of travel, opposes it, or is too fast to stop within the
    remaining distance (in which case it brakes, overshoots, and returns).

    `t_offset > 0` prepends a hold at (p0, v=0, a=0) for that many seconds
    before the motion starts — for firmware cold-start compensation.
    """
    assert v_max > 0
    assert a_max > 0
    assert t_offset >= 0

    if wrap_target:
        delta_signed = _select_delta(p0, p_target, az_forbidden_deg)
    else:
        delta_signed = p_target - p0

    if abs(delta_signed) < _POS_EPS_DEG and abs(v0) < 1e-6:
        base = PlannedTrajectory(
            points=(TrajectoryPoint(t=0.0, pos=p0, vel=v0, acc=0.0),),
            total_duration=0.0,
        )
        return _apply_t_offset(base, t_offset, tick_dt)

    out: list[TrajectoryPoint] = [
        TrajectoryPoint(t=0.0, pos=p0, vel=v0, acc=0.0),
    ]
    t_cur, p_cur, v_cur = 0.0, p0, v0

    # Bang-bang phase schedule that starts at v0 and lands on the target at
    # v=0. Planned in continuous coordinates from the resolved signed delta,
    # so cumulative targets (|delta| > 180) and the chosen forbidden-avoiding
    # path are never re-wrapped away.
    for a_phase, dur_phase in _plan_trapezoid_phases(v0, delta_signed, v_max, a_max):
        if dur_phase <= 1e-12:
            continue
        t_cur, p_cur, v_cur = _sample_phase(
            out,
            t_cur,
            p_cur,
            v_cur,
            a=a_phase,
            dur=dur_phase,
            tick_dt=tick_dt,
            include_endpoint=True,
        )

    # Force the final point exactly (floating error can leave v_cur tiny).
    out[-1] = TrajectoryPoint(t=t_cur, pos=p_cur, vel=0.0, acc=0.0)

    # Deduplicate consecutive same-t points, which can appear when
    # phase durations are shorter than tick_dt.
    dedup: list[TrajectoryPoint] = []
    for pt in out:
        if dedup and abs(dedup[-1].t - pt.t) < 1e-9:
            dedup[-1] = pt
        else:
            dedup.append(pt)

    base = PlannedTrajectory(points=tuple(dedup), total_duration=t_cur)
    return _apply_t_offset(base, t_offset, tick_dt)


def scurve_profile(
    p0: float,
    v0: float,
    p_target: float,
    v_max: float,
    a_max: float,
    j_max: float,
    tick_dt: float = 0.1,
    t_offset: float = 0.0,
    az_forbidden_deg: float | None = None,
    wrap_target: bool = True,
) -> PlannedTrajectory:
    """Jerk-limited (S-curve) version of `trapezoidal_profile`.

    Each accel/decel phase is split into (ramp-up accel, constant accel,
    ramp-down accel) under jerk cap `j_max`. Phase durations:
        t_j = a_max / j_max   (ramp-up / ramp-down of accel)
        t_a = (v - t_j * a_max) / a_max   (constant-accel segment)

    When the distance is too short for full accel ramps, falls back to
    a pure-jerk triangular profile. Implementation uses 7 segments max
    (jerk up, const accel, jerk down, cruise, jerk up, const decel,
    jerk down) with zero-duration segments skipped.

    For v0 != 0 this delegates to `trapezoidal_profile` to first bring v
    to zero, then builds an S-curve from rest — simpler than a proper
    v0-aware S-curve, adequate for our Phase 2 needs.

    `t_offset > 0` behaves as in `trapezoidal_profile`: a lead-in hold at
    (p0, v=0, a=0) prepended to the final trajectory.

    `az_forbidden_deg` picks the non-crossing path (short or long) so the
    trajectory never traverses the forbidden azimuth.
    """
    assert t_offset >= 0

    if abs(v0) > 1e-6:
        # Bring to rest first via trapezoid (uses a_max without S-curving
        # the v0 decel), then append an S-curve from rest. The lead-in
        # hold for t_offset is applied only at the outer return.
        pre = trapezoidal_profile(
            p0=p0,
            v0=v0,
            p_target=p0,  # "rest in place"; trapezoid handles it
            v_max=v_max,
            a_max=a_max,
            tick_dt=tick_dt,
        )
        # Start the S-curve at pre's endpoint.
        pre_end = pre.points[-1]
        tail = scurve_profile(
            p0=pre_end.pos,
            v0=0.0,
            p_target=p_target,
            v_max=v_max,
            a_max=a_max,
            j_max=j_max,
            tick_dt=tick_dt,
            az_forbidden_deg=az_forbidden_deg,
            wrap_target=wrap_target,
        )
        # Concatenate (shift tail by pre.total_duration).
        shifted = tuple(
            TrajectoryPoint(
                t=p.t + pre.total_duration,
                pos=p.pos,
                vel=p.vel,
                acc=p.acc,
            )
            for p in tail.points
        )
        combined = PlannedTrajectory(
            points=pre.points + shifted[1:],  # skip duplicated boundary
            total_duration=pre.total_duration + tail.total_duration,
        )
        return _apply_t_offset(combined, t_offset, tick_dt)

    # Rest-to-rest S-curve.
    if wrap_target:
        delta_signed = _select_delta(p0, p_target, az_forbidden_deg)
    else:
        delta_signed = p_target - p0
    if abs(delta_signed) < _POS_EPS_DEG:
        base = PlannedTrajectory(
            points=(TrajectoryPoint(t=0.0, pos=p0, vel=0.0, acc=0.0),),
            total_duration=0.0,
        )
        return _apply_t_offset(base, t_offset, tick_dt)
    dir_sign = 1.0 if delta_signed >= 0 else -1.0
    abs_delta = abs(delta_signed)

    t_j = a_max / j_max
    # Velocity gained during each jerk ramp: v_j = 0.5 * a_max * t_j
    # Velocity gained during a full accel phase (j-up + const-a + j-down) to
    # reach v_peak: v_const_a = v_peak - 2*v_j (const-accel part).
    # If v_peak < 2*v_j, we can't reach max accel; use pure triangular jerk
    # profile where accel ramps up then down without a constant segment.

    v_j = 0.5 * a_max * t_j  # = a_max² / (2 * j_max)

    # Distance to reach v_max from rest via full accel ramp:
    # using symmetric accel/decel about peak accel a_max
    # t_full_accel = t_j + t_a + t_j, where t_a = (v_max - 2 * v_j) / a_max
    if v_max >= 2 * v_j:
        t_a_full = (v_max - 2 * v_j) / a_max
        d_full_accel = (
            # jerk-up phase
            (1.0 / 6.0) * j_max * (t_j**3)
            # const-accel phase (v at start of const-accel = v_j)
            + v_j * t_a_full
            + 0.5 * a_max * (t_a_full**2)
            # jerk-down phase (v at start = v_j + a_max*t_a; accel goes
            # from a_max to 0 linearly, integrated over t_j)
            + (v_j + a_max * t_a_full) * t_j
            + 0.5 * a_max * t_j * t_j
            - (1.0 / 6.0) * j_max * (t_j**3)
        )
        reach_full = 2 * d_full_accel <= abs_delta
    else:
        reach_full = False
        # d under pure-triangular-jerk (no const-accel) to reach v = 2*v_j:
        d_full_accel = 2 * (1.0 / 6.0) * j_max * (t_j**3) + v_j * t_j
        # Actually for rest→v_peak via symmetric j ramps of t_j each,
        # the distance is just v_j * t_j = (a_max²/(2 j_max)) * (a_max/j_max)
        # = a_max³ / (2 j_max²). That's a purely derived formula — fall
        # through to the simpler triangular planner below.

    if not reach_full:
        # Not enough distance for a full trapezoidal-in-accel profile.
        # Fall back to trapezoidal (no S-curve). This trades off smoothness
        # for keeping the planner simple; Phase 2 acceptance criteria
        # don't require S-curve in short-move cases. Pass t_offset through
        # so the lead-in hold still happens.
        return trapezoidal_profile(
            p0=p0,
            v0=0.0,
            p_target=p_target,
            v_max=v_max,
            a_max=a_max,
            tick_dt=tick_dt,
            t_offset=t_offset,
            az_forbidden_deg=az_forbidden_deg,
            wrap_target=wrap_target,
        )

    # Build the 7-segment profile at rest-to-rest.
    out: list[TrajectoryPoint] = [
        TrajectoryPoint(t=0.0, pos=p0, vel=0.0, acc=0.0),
    ]
    t_cur, p_cur, v_cur = 0.0, p0, 0.0

    # Cruise distance
    d_cruise = abs_delta - 2 * d_full_accel
    t_cruise = d_cruise / v_max
    t_a_full = (v_max - 2 * v_j) / a_max

    # Segment 1: jerk up (a: 0 → +a_max) during t_j
    for k in range(1, max(1, int(math.floor(t_j / tick_dt))) + 1):
        dt = min(k * tick_dt, t_j)
        a = dir_sign * j_max * dt
        v = v_cur + dir_sign * 0.5 * j_max * dt * dt
        p = p_cur + v_cur * dt + dir_sign * (1.0 / 6.0) * j_max * dt**3
        out.append(TrajectoryPoint(t=t_cur + dt, pos=p, vel=v, acc=a))
    # advance state by full t_j
    dt = t_j
    p_cur = p_cur + v_cur * dt + dir_sign * (1.0 / 6.0) * j_max * dt**3
    v_cur = v_cur + dir_sign * 0.5 * j_max * dt * dt
    t_cur = t_cur + dt
    a_cur = dir_sign * a_max

    # Segment 2: const accel at +a_max during t_a_full
    if t_a_full > 1e-9:
        t_cur, p_cur, v_cur = _sample_phase(
            out,
            t_cur,
            p_cur,
            v_cur,
            a=a_cur,
            dur=t_a_full,
            tick_dt=tick_dt,
            include_endpoint=True,
        )

    # Segment 3: jerk down (a: +a_max → 0) during t_j
    for k in range(1, max(1, int(math.floor(t_j / tick_dt))) + 1):
        dt = min(k * tick_dt, t_j)
        a = a_cur - dir_sign * j_max * dt
        v = v_cur + a_cur * dt - dir_sign * 0.5 * j_max * dt * dt
        p = (
            p_cur
            + v_cur * dt
            + 0.5 * a_cur * dt * dt
            - dir_sign * (1.0 / 6.0) * j_max * dt**3
        )
        out.append(TrajectoryPoint(t=t_cur + dt, pos=p, vel=v, acc=a))
    dt = t_j
    p_cur = (
        p_cur
        + v_cur * dt
        + 0.5 * a_cur * dt * dt
        - dir_sign * (1.0 / 6.0) * j_max * dt**3
    )
    v_cur = v_cur + a_cur * dt - dir_sign * 0.5 * j_max * dt * dt
    t_cur = t_cur + dt

    # Segment 4: cruise at v_max
    if t_cruise > 1e-9:
        t_cur, p_cur, v_cur = _sample_phase(
            out,
            t_cur,
            p_cur,
            v_cur,
            a=0.0,
            dur=t_cruise,
            tick_dt=tick_dt,
            include_endpoint=True,
        )

    # Segment 5: jerk up (a: 0 → -a_max) during t_j (decel begins)
    for k in range(1, max(1, int(math.floor(t_j / tick_dt))) + 1):
        dt = min(k * tick_dt, t_j)
        a = -dir_sign * j_max * dt
        v = v_cur - dir_sign * 0.5 * j_max * dt * dt
        p = p_cur + v_cur * dt - dir_sign * (1.0 / 6.0) * j_max * dt**3
        out.append(TrajectoryPoint(t=t_cur + dt, pos=p, vel=v, acc=a))
    dt = t_j
    p_cur = p_cur + v_cur * dt - dir_sign * (1.0 / 6.0) * j_max * dt**3
    v_cur = v_cur - dir_sign * 0.5 * j_max * dt * dt
    t_cur = t_cur + dt
    a_cur = -dir_sign * a_max

    # Segment 6: const decel at -a_max during t_a_full
    if t_a_full > 1e-9:
        t_cur, p_cur, v_cur = _sample_phase(
            out,
            t_cur,
            p_cur,
            v_cur,
            a=a_cur,
            dur=t_a_full,
            tick_dt=tick_dt,
            include_endpoint=True,
        )

    # Segment 7: jerk down (a: -a_max → 0) during t_j
    for k in range(1, max(1, int(math.floor(t_j / tick_dt))) + 1):
        dt = min(k * tick_dt, t_j)
        a = a_cur + dir_sign * j_max * dt
        v = v_cur + a_cur * dt + dir_sign * 0.5 * j_max * dt * dt
        p = (
            p_cur
            + v_cur * dt
            + 0.5 * a_cur * dt * dt
            + dir_sign * (1.0 / 6.0) * j_max * dt**3
        )
        out.append(TrajectoryPoint(t=t_cur + dt, pos=p, vel=v, acc=a))
    dt = t_j
    p_cur = (
        p_cur
        + v_cur * dt
        + 0.5 * a_cur * dt * dt
        + dir_sign * (1.0 / 6.0) * j_max * dt**3
    )
    v_cur = v_cur + a_cur * dt + dir_sign * 0.5 * j_max * dt * dt
    t_cur = t_cur + dt

    # Snap terminal (v should be ~0).
    out[-1] = TrajectoryPoint(t=t_cur, pos=p_cur, vel=0.0, acc=0.0)

    # Dedup same-t
    dedup: list[TrajectoryPoint] = []
    for pt in out:
        if dedup and abs(dedup[-1].t - pt.t) < 1e-9:
            dedup[-1] = pt
        else:
            dedup.append(pt)

    base = PlannedTrajectory(points=tuple(dedup), total_duration=t_cur)
    return _apply_t_offset(base, t_offset, tick_dt)


# ---------------------------------------------------------------------------
# 2-axis coordinated planners
# ---------------------------------------------------------------------------
#
# Plan a single straight-line path in (az, el) space so both axes arrive
# simultaneously. Path-length v/a/j caps are derived from the per-axis caps
# via 1 / max(|dir_az|, |dir_el|) scaling — pure-axis moves match 1-D
# throughput exactly; diagonals project back to per-axis rates equal to the
# per-axis cap along whichever axis is dominant.


def _plan_2d(
    p0_az: float,
    p0_el: float,
    p_target_az: float,
    p_target_el: float,
    v_max: float,
    a_max: float,
    j_max: float | None,
    tick_dt: float,
    wrap_az: bool,
    az_forbidden_deg: float | None,
) -> tuple[PlannedTrajectory, PlannedTrajectory]:
    """Shared implementation for the 2-D trapezoidal and S-curve planners.

    Picks the az delta (wrap-aware or cumulative), builds a 1-D profile on
    path length, then projects each sample back onto per-axis trajectories.
    Passing `j_max=None` selects the trapezoidal planner; otherwise S-curve.
    """
    if wrap_az:
        delta_az = _select_delta(p0_az, p_target_az, az_forbidden_deg)
    else:
        delta_az = p_target_az - p0_az
    delta_el = p_target_el - p0_el

    path_len = math.sqrt(delta_az * delta_az + delta_el * delta_el)

    if path_len < _POS_EPS_DEG:
        traj_az = PlannedTrajectory(
            points=(TrajectoryPoint(t=0.0, pos=p0_az, vel=0.0, acc=0.0),),
            total_duration=0.0,
        )
        traj_el = PlannedTrajectory(
            points=(TrajectoryPoint(t=0.0, pos=p0_el, vel=0.0, acc=0.0),),
            total_duration=0.0,
        )
        return traj_az, traj_el

    dir_az = delta_az / path_len
    dir_el = delta_el / path_len

    # Scale per-axis caps into path-length caps so per-axis rates stay
    # within their caps. For direction (dir_az, dir_el), the per-axis
    # rate along the path is v_path * |dir_i|; max(|dir_i|) gives the
    # tightest per-axis bound.
    max_abs_dir = max(abs(dir_az), abs(dir_el))
    scale = 1.0 / max_abs_dir
    v_max_path = v_max * scale
    a_max_path = a_max * scale

    if j_max is None:
        path_traj = trapezoidal_profile(
            p0=0.0,
            v0=0.0,
            p_target=path_len,
            v_max=v_max_path,
            a_max=a_max_path,
            tick_dt=tick_dt,
            wrap_target=False,
        )
    else:
        j_max_path = j_max * scale
        path_traj = scurve_profile(
            p0=0.0,
            v0=0.0,
            p_target=path_len,
            v_max=v_max_path,
            a_max=a_max_path,
            j_max=j_max_path,
            tick_dt=tick_dt,
            wrap_target=False,
        )

    az_points = tuple(
        TrajectoryPoint(
            t=p.t,
            pos=p0_az + dir_az * p.pos,
            vel=dir_az * p.vel,
            acc=dir_az * p.acc,
        )
        for p in path_traj.points
    )
    el_points = tuple(
        TrajectoryPoint(
            t=p.t,
            pos=p0_el + dir_el * p.pos,
            vel=dir_el * p.vel,
            acc=dir_el * p.acc,
        )
        for p in path_traj.points
    )
    return (
        PlannedTrajectory(points=az_points, total_duration=path_traj.total_duration),
        PlannedTrajectory(points=el_points, total_duration=path_traj.total_duration),
    )


def trapezoidal_profile_2d(
    p0_az: float,
    p0_el: float,
    p_target_az: float,
    p_target_el: float,
    v_max: float,
    a_max: float,
    tick_dt: float = 0.1,
    wrap_az: bool = True,
    az_forbidden_deg: float | None = None,
) -> tuple[PlannedTrajectory, PlannedTrajectory]:
    """Coordinated 2-D trapezoidal profile: straight line in (az, el) with
    both axes arriving simultaneously. `v_max` and `a_max` are per-axis caps.
    """
    return _plan_2d(
        p0_az,
        p0_el,
        p_target_az,
        p_target_el,
        v_max=v_max,
        a_max=a_max,
        j_max=None,
        tick_dt=tick_dt,
        wrap_az=wrap_az,
        az_forbidden_deg=az_forbidden_deg,
    )


def scurve_profile_2d(
    p0_az: float,
    p0_el: float,
    p_target_az: float,
    p_target_el: float,
    v_max: float,
    a_max: float,
    j_max: float,
    tick_dt: float = 0.1,
    wrap_az: bool = True,
    az_forbidden_deg: float | None = None,
) -> tuple[PlannedTrajectory, PlannedTrajectory]:
    """Coordinated 2-D S-curve: jerk-limited straight-line path in (az, el)
    with both axes arriving simultaneously. `v_max`, `a_max`, `j_max` are
    per-axis caps.
    """
    return _plan_2d(
        p0_az,
        p0_el,
        p_target_az,
        p_target_el,
        v_max=v_max,
        a_max=a_max,
        j_max=j_max,
        tick_dt=tick_dt,
        wrap_az=wrap_az,
        az_forbidden_deg=az_forbidden_deg,
    )
