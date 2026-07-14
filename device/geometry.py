"""Canonical angle/geometry helpers shared across the mount-control stack.

Single source of truth for the wrap/unwrap/separation math that was
previously copy-pasted (with drifting boundary conventions) across
velocity_controller, sun_safety, auto_level, sky_grid,
nighttime_calibration, celestial_targets and the scripts/ tree.

Convention: ``wrap_pm180`` maps into the half-open interval
[-180.0, +180.0) — -180 inclusive, +180 wraps back to -180, matching
Python's modulo on [0, 360). The rotation-calibration fit keeps its own
(-180, +180] variant internally; unify it here only together with a
recalibration-safe migration.
"""

import math


def wrap_pm180(deg: float) -> float:
    """Wrap an angle in degrees into the half-open interval [-180, +180).

    -180 is inclusive, +180 wraps back to -180 — same convention as Python's
    modulo on [0, 360).
    """
    return ((deg + 180.0) % 360.0) - 180.0


def unwrap_az_series(wrapped_samples: list[float]) -> list[float]:
    """Given an azimuth sample series wrapped to [-180, +180), return an
    unwrapped cumulative series suitable for fitting.

    Each per-step delta is interpreted modulo 360 via wrap_pm180: if two
    adjacent wrapped samples differ by more than 180, the true motion is
    assumed to be the shorter path (< 180). This is safe as long as the
    true per-sample delta is bounded below ±180 — at the S50's 6°/s
    firmware cap and sample intervals up to 30 s the implicit bound is
    ~180° per 30 s, which the caller must respect.

    The first element is returned as-is; subsequent elements accumulate
    wrap_pm180(sample[i] - sample[i-1]).

    Raises ``ValueError`` on any non-finite sample (NaN / Inf). Without
    this guard a single bad reading silently propagates NaN across the
    whole tail of the series and corrupts every downstream consumer
    (``CumulativeAzTracker``, fit_auto_level, JSON persistence). Reject
    at the boundary so the caller sees the failure.
    """
    if not wrapped_samples:
        return []
    out: list[float] = []
    for i, v in enumerate(wrapped_samples):
        if not math.isfinite(v):
            raise ValueError(f"unwrap_az_series: non-finite sample at index {i}: {v!r}")
    out.append(wrapped_samples[0])
    for i in range(1, len(wrapped_samples)):
        d = wrap_pm180(wrapped_samples[i] - wrapped_samples[i - 1])
        out.append(out[-1] + d)
    return out


def angular_separation_deg(
    a_az_deg: float, a_el_deg: float, b_az_deg: float, b_el_deg: float
) -> float:
    """Great-circle angular separation between two (az, el) directions.

    Returns degrees in [0, 180]. Pure function — no observer, no time.
    Az is treated as longitude, el as latitude on the celestial sphere.
    RA/Dec callers convert at their own boundary (same sphere math).
    """
    a_az = math.radians(a_az_deg)
    a_el = math.radians(a_el_deg)
    b_az = math.radians(b_az_deg)
    b_el = math.radians(b_el_deg)
    # Spherical law of cosines, clamped for numerical safety.
    cos_sep = math.sin(a_el) * math.sin(b_el) + math.cos(a_el) * math.cos(
        b_el
    ) * math.cos(a_az - b_az)
    cos_sep = max(-1.0, min(1.0, cos_sep))
    return math.degrees(math.acos(cos_sep))
