# Nighttime calibration via plate solving — plan

Date: 2026-04-25
Branch: `wwerst/calibrate-nighttime-platesolve` (depends on PR #15)

## Goal

Add a nighttime mode to the calibrate-rotation page that fits the same
3-DOF mount rotation matrix the daytime FAA-landmark workflow already
fits, but using **plate-solved camera images of the night sky** instead
of operator-aligned ground landmarks.

Per-sighting cycle:

1. Operator commands a slew to a `(commanded_az, commanded_el)` target.
2. Mount settles (the live-tracker continuous-control loop shipped in
   PR #15 closes residual error to ≤0.003° before capture).
3. Camera captures one image.
4. Plate solver computes the true `(RA, Dec)` of the image centre.
5. Site location + UTC convert `(RA, Dec)` → true `(az, el)`.
6. Pair `(commanded_az, commanded_el) → (true_az, true_el)` is one
   sighting, equivalent to one daytime FAA-landmark sighting.

After ≥3 sightings spanning ≥30° of sky, the existing
`RotationCalibrationSolver` (in `device/rotation_calibration.py`) fits
yaw/pitch/roll to the sightings. Same math, different inputs.

If a plate solve **fails** at a position (occluded sky, thin cloud,
no stars detected, ambiguous solution), the operator jogs to a clearer
neighbour using the arrow-key continuous jog (item 4) or click-to-go
(item 5) and retries — without losing prior accepted sightings.

## Why plate solving

Daytime FAA landmarks need:
- Daylight (some lit, some accuracy is night-only)
- An unobstructed line-of-sight to a tall structure
- The operator to manually align the crosshair on a feature ≤0.04° wide
  at multi-km slant

Nighttime plate solving needs:
- A clear patch of sky containing ≥10 stars
- 5–60 s of solver time
- No operator alignment — the solver matches the image to a star
  catalog and returns the centre's `(RA, Dec)` to ~arcsecond accuracy

For sites where Hyperion-class daytime targets are obstructed (anywhere
with trees, urban skyline, weather), nighttime calibration is the only
practical option.

## Plate-solver options

### Option A — astrometry.net `solve-field` (CLI)

Pro: industry standard, used by every astronomy app from KStars to
ASIAir. Robust against star-pattern variants, faint stars, vignetting.
Returns RA/Dec/FOV/position-angle plus diagnostic chunks.

Con: requires installing the binary plus index files (~5–500 GB
depending on which sky-resolution range you cover). Not pip-installable.

Recommended for production: install the `solve-field` Debian / brew
package and the appropriate index range for the S50's FOV (~1.27° ×
0.71° → use `index-4204..-4209` series, ~400 MB total).

### Option B — `astroquery.astrometry_net` (web service)

Pro: pure-pip. Pro: works without local index files.

Con: requires an astrometry.net API key, network access, and 30–120 s
typical latency. Pro/con: queues — slow under load. Subject to upstream
availability and rate limits.

Reasonable as a **fallback** when local `solve-field` is unavailable.

### Option C — minimal blind-solve via `astropy` + a bundled catalog

Pro: self-contained.

Con: large code surface, slower than A, less robust against the long
tail of edge cases (vignetting, defects, hot pixels, satellite trails).

**Skip for MVP.**

### Choice for this PR

Implement a **`PlateSolver` interface** with three concrete subclasses:

- `SolveFieldPlateSolver` — wraps the `solve-field` CLI. Default when
  available.
- `AstrometryNetWebPlateSolver` — Option B fallback, only enabled when
  `ASTROMETRY_API_KEY` env var is set. Stub for now; full
  implementation is a separate follow-up if anyone needs it.
- `FakePlateSolver` — used by tests. Returns canned solutions from a
  fixtures dictionary keyed by image path.

The interface plus `FakePlateSolver` are enough for this PR's tests.
`SolveFieldPlateSolver`'s shell-out is included with a graceful
"binary not on PATH" error path so the nighttime mode tab can be
disabled at runtime when the binary isn't available.

## Data model

### `NighttimeSighting` (new dataclass)

```python
@dataclass(frozen=True)
class NighttimeSighting:
    """One plate-solved (commanded_az, commanded_el) → (true_az, true_el)."""
    commanded_az_deg: float   # what the mount was told to point at
    commanded_el_deg: float
    encoder_az_deg: float     # what the mount actually reached (post-settle)
    encoder_el_deg: float
    true_ra_deg: float        # plate-solved RA (J2000)
    true_dec_deg: float       # plate-solved Dec (J2000)
    true_az_deg: float        # converted to topocentric az/el at t_unix
    true_el_deg: float
    fov_x_deg: float          # plate-solved FOV (sanity check)
    fov_y_deg: float
    position_angle_deg: float # plate-solved roll
    image_path: Path
    t_unix: float             # capture time
```

### Reuse: `Sighting` (existing daytime dataclass)

The existing `RotationCalibrationSolver.solve_rotation` takes a
`list[Sighting]`. `Sighting` has `landmark`, `encoder_az_deg`,
`encoder_el_deg`, `true_az_deg`, `true_el_deg`, `slant_m`. We synthesise
a "celestial landmark" stub for each nighttime sighting (the solver
only reads `encoder_az_deg`, `encoder_el_deg`, and the `landmark.ecef()`
return — for celestial we need to bypass that path).

**Decision:** add an alternate solver entry point
`solve_rotation_from_pairs(pairs, site)` that takes raw
`[(encoder_az, encoder_el, true_az, true_el)]` tuples and skips the
landmark-ECEF round-trip. Daytime and nighttime both ultimately call
this helper. Daytime still goes through `solve_rotation(sightings,
site)` which now calls `solve_rotation_from_pairs` internally.

This is a small refactor — keeps the daytime API unchanged, makes the
math reusable, lets tests exercise the solver without inventing fake
landmarks.

### Persistence

Reuses `device/_atomic_json.py:write_atomic_json`. The output JSON is
the same `mount_calibration.json` schema produced by daytime calibration,
with `calibration_method: "rotation_platesolve"` instead of
`"rotation_landmarks"`. The existing `MountFrame.from_calibration_json`
loader is method-agnostic — it only cares about yaw/pitch/roll/observer.

The per-landmark records in the JSON are extended with a `kind` field
("landmark" or "platesolve") and the platesolve records carry RA/Dec +
FOV diagnostics.

## UX flow on calibrate page

### Mode toggle

Two buttons at the top of the right-hand controls column:

- ☀️ **Daytime (FAA landmarks)** — the existing flow. Default if it's
  before sunset or if any FAA landmark is above-horizon.
- 🌙 **Nighttime (plate solve)** — the new flow. Default if it's after
  astronomical dusk and `solve-field` is on PATH.

Server endpoint `/api/{tid}/calibrate_nighttime/availability` returns:

```json
{
  "solver_available": true,
  "solver_kind": "solve-field",
  "solver_path": "/usr/bin/solve-field",
  "is_dark_now": true,
  "min_alt_deg": 10.0
}
```

If `solver_available: false`, the nighttime tab shows "Plate solver
not configured" and a link to this plan doc.

### Nighttime panel

```
┌──────────────────────────────────────────────────────────┐
│  Nighttime calibration (plate solve)                     │
├──────────────────────────────────────────────────────────┤
│  Sightings: 2/3 minimum  ·  fit RMS: 0.018°              │
├──────────────────────────────────────────────────────────┤
│  Current pointing:  az 142.31°  el 45.07°  ✓ settled     │
│                                                          │
│  [📷 Capture sighting]    [Skip]    [Cancel]             │
│                                                          │
│  Status: Solving image (12/60 s) …                       │
├──────────────────────────────────────────────────────────┤
│  Sightings:                                              │
│    ✓ #1   az 88°   el 35°   ra 5h 14m  dec +10°  ε 0.02° │
│    ✓ #2   az 200°  el 50°   ra 12h 03m dec +21°  ε 0.01° │
│    ⏳ #3  …solving                                       │
│    ✗ #4  failed (no solution; jog to clearer sky)        │
├──────────────────────────────────────────────────────────┤
│  [Apply calibration]                  (≥3 ok required)   │
└──────────────────────────────────────────────────────────┘
```

The "Capture sighting" button:

1. Reads current encoder pointing.
2. Triggers an image capture via the same firmware command the live
   view uses.
3. Posts the image path to the plate solver (background thread).
4. UI polls /state every 1 s; status shows "Solving (Δt s)…".
5. On success: row turns ✓, fit refreshes.
6. On failure: row turns ✗ with a hint ("no solution", "ambiguous",
   "below threshold"). Operator can tap the ✗ to remove it, then jog
   (arrow keys) or click-to-go to a clearer position and recapture.

The "Apply calibration" button writes `mount_calibration.json` via the
same atomic-write helper the daytime flow uses; the live tracker /
streaming controller pick it up on next session.

### Refusals

- Below 10° altitude: capture button disabled with "below altitude
  floor" tooltip. The mount can mechanically slew below 10°, but
  plate-solving the ground is a waste; the streaming loop's altitude
  limits handle the no-slew-below-horizon side.
- Cable-wrap violation: the motion session refuses; the capture button
  shows "outside cable-wrap range".
- Solver not configured: tab is disabled (see above).

## Threading

Plate solving takes 5–60 s typical. Run it on a background worker
thread per session (single-flight queue: only one solve at a time). The
HTTP `/capture` POST returns 202 Accepted immediately with a `job_id`;
the UI polls `/state` for completion. State machine:

```
idle ── capture ──→ awaiting_settle (~1 s) ──→ capturing (~1–3 s)
   ↑                                              │
   │                                              ↓
   │                                           solving (5–60 s)
   │                                              │
   └─── (success/fail/skip) ←────────────────────┘
```

## Edge cases

- **Plate solver returns wildly wrong solution.** The S50 FOV ≈ 1.27 ×
  0.71°. If the solver returns FOV outside [0.5°, 3°], reject the
  sighting (`solver_fov_out_of_range` error). Sanity threshold is wider
  than the spec to accommodate cropping.
- **Sighting at zenith.** Az/el is degenerate at the pole. We require
  `el ≤ 80°` for nighttime sightings — slightly tighter than the
  daytime band — to avoid the singularity.
- **Site time clock skew.** RA/Dec → az/el conversion uses `time.time()`
  + the site location. A clock skew of 60 s shifts az by ~0.25° at the
  equator, which would degrade fit RMS by ~0.25°. We log the diagnostic
  but don't auto-detect; future PR could compare consecutive sightings
  for self-consistency.
- **Atmospheric refraction at altitude.** astropy's `AltAz` frame
  applies refraction by default for altitudes above the horizon; we
  pass `pressure` and `temperature` from a sane default (NIST sea
  level) for now. Operator can override via env vars in the future.
- **Precession / nutation.** astropy's `transform_to(AltAz)` handles
  these natively given a `Time` and an `EarthLocation`. No special
  handling needed.

## Mutual exclusion

`NighttimeCalibrationSession` follows the same mutex pattern as the
daytime workflow:

- Refuses to start if `LiveTrackSession` is alive on this telescope.
- Allowed alongside `CalibrateMotionSession` (delegates motion to it).
- A single per-telescope `NighttimeCalibrationManager` allows only one
  session at a time.

## Persistence flow

```
operator: capture sighting
  ↓
session: snapshot encoder; trigger image capture; queue plate solve
  ↓ (background thread)
plate_solver.solve(image_path)
  ↓
session: convert ra/dec → az/el (using site + UTC)
  ↓
session.add_sighting(...)  # append to internal list
  ↓
session.refit()  # solve_rotation_from_pairs(pairs, site)
  ↓
session.status() reflects new fit RMS + sighting list
```

When operator clicks Apply:

```
session.apply()
  ↓
write_atomic_json(mount_calibration.json, payload)
  ↓
return MotionStatus(phase="committed")
```

The streaming controller / live tracker pick up the new
`mount_calibration.json` on next session start (existing
`load_session_mount_frame()` re-reads the file).

## Test plan

`tests/test_nighttime_calibration.py`:

1. **`FakePlateSolver`** returns canned `(RA, Dec, FOV)` from a
   fixtures dict keyed by image path. Used to drive the session
   without a real solver.

2. **3 sightings → fit succeeds** — Build session with FakePlateSolver,
   add 3 sightings spanning 30° of sky, assert solver runs and returns
   a `RotationSolution` with RMS < 0.05°.

3. **<3 sightings → fit refuses to apply** — Add 2 sightings, call
   `apply()`, assert ValueError-ish error in status.

4. **Failed sighting can be skipped** — FakePlateSolver returns `None`
   for one image; assert `add_sighting` returns failure, prior accepted
   sightings still in the list, `is_settled` returns false until skip.

5. **Solver returns wildly wrong FOV** — FakePlateSolver returns
   `fov_x=10°` (way larger than S50 spec). Assert sighting rejected
   with `solver_fov_out_of_range` error.

6. **Persistence is atomic** — Spy on `write_atomic_json`; assert
   `apply()` calls it with the correct payload schema (yaw/pitch/roll
   + observer + landmarks subarray with `kind: "platesolve"`).

7. **Below-altitude refusal** — Slew to el=5°, attempt capture, assert
   refusal with `below_altitude_floor` error. Capture button never
   reaches plate solver.

8. **Mutex against live tracker** — Stub a fake LiveTrackSession alive
   on the same telescope, attempt to start NighttimeCalibrationSession,
   assert RuntimeError.

9. **Solver not on PATH** — `solve-field` CLI not installed; the
   availability endpoint returns `solver_available: false`; capture
   POST returns 503.

10. **Background-threaded solve** — Assert `/capture` returns 202 +
    `job_id` immediately (within 100 ms even if solver takes 30 s),
    and `/state` polls reflect "solving" → "ok"/"fail".

## Out of scope for this PR

- Auto-blind-pointing (have the system pick its own sighting positions
  spanning sky to maximise fit info — operator-driven for now).
- Multi-band catalog support (Tycho-2 fallback when astrometry.net
  index data isn't installed).
- Cloud / cosmic-ray detection in captured images.
- Real-time refit during multi-sighting passes (currently refits after
  every accepted sighting; could be deferred to "operator clicks
  Apply" for slow sites).
- Migrate the `RotationCalibrationSolver`'s yaw-only special case from
  the daytime path to the new shared helper. Keeps the two flows
  separate at the function level for now; consolidate in a follow-up.

## Files to touch

- New `plans/nighttime_calibration.md` (this doc)
- New `device/plate_solver.py`
- New `device/nighttime_calibration.py`
- Modified `device/rotation_calibration.py` — extract
  `solve_rotation_from_pairs` so the celestial path can reuse it.
- Modified `front/app.py` — new resource classes + routes.
- Modified `front/templates/calibrate_rotation.html` — mode toggle +
  nighttime panel.
- New `tests/test_nighttime_calibration.py`.

## Estimated complexity

Largest chunk is the frontend mode-toggle + nighttime panel. Backend is
~300 lines (session + routes); plate solver wrapper is ~120 lines.
Tests ~250 lines. Plan doc ~250 lines (this doc).

Total ~1500 lines. Operator's directive estimates 400–500 turns; this
plan + scaffolding gets us most of the way before the implementation
intensifies.
