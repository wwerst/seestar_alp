# Operations runbook — Seestar S50 (seestar_alp fork)

A concise field guide for running the mount: what the process is, how to set up
a site, how to calibrate at dusk and at night, how to track aircraft, and how
to react when the telescope's wifi uplink saturates. Every route, UI label, and
config key below is quoted from the current tree.

---

## 1. Architecture in brief

The whole stack runs in **one process** (`root_app.py`) as **four runners**:

| Runner | Entry point | Listens on | Role |
|---|---|---|---|
| ALP device server | `DeviceMain` (`device/app.py`) | TCP `Config.port` (default **5555**) | Alpaca/Falcon device API; owns the RPC socket to the mount firmware. |
| Front UI | `FrontMain` (`front/app.py`) | TCP `Config.uiport` (default **5432**) | The web UI you point a browser at. |
| LiveTracker service | `LiveTrackerMain` (`device/live_tracker_service.py`) | no socket | Always-on background service: loads the live-track manager + target catalog (both lazy) and starts the **sun-safety monitor**. |
| Imaging web server | inline Flask/waitress in `root_app.py` | TCP `Config.imgport` (default **7556**) | Serves the MJPEG video (`/<dev>/vid`), the SSE status/event streams. |

The mount firmware itself exposes three telescope-side TCP ports over wifi:
**4700** (RPC/commands), **4800** (raw preview/stack frames), **4554** (RTSP
H.264 video). All three share the telescope's single wifi uplink — see §6.

**Motion sessions.** Anything that drives the mount runs as a *session* object
with its own worker thread, started/stopped through a per-feature manager:
live tracking (`device/live_tracker.py`), rotation calibration
(`device/rotation_calibration.py`), calibrate-motion
(`device/calibrate_motion.py`), nighttime calibration
(`device/nighttime_calibration.py`, including its hands-free auto-runner), and
the sky-visibility mapper (`device/visibility_mapper.py`, which slews between
sample cells). Only one should own the mount at a time.

**Sun-safety watchdog.** `SunSafetyMonitor` (`device/sun_safety.py`) is an
always-on daemon started once by the LiveTracker runner. It senses the mount's
sky pointing from the raw motor encoder (`scope_get_horiz_coord`, so it stays
sighted even before plate-solve alignment). When the sun is above
`alt_threshold_deg` and the optical axis enters the sun cone it: sets an
emergency lockout, **jogs the mount away first**, then aborts every active
session (`_abort_active_sessions` fans out a stop to all managers above),
waits, and clears the lockout. The S50 has no solar filter, so this is a
hard-safety system, not a convenience.

**Where the always-on "MountService" is headed.** Today mount ownership is
*split*: the device server owns the RPC socket, the LiveTracker runner owns the
sun-safety monitor, and each feature spins up its own session that drives the
mount directly. Coordinating them is why the watchdog has to fan a stop out to
six managers (`_abort_active_sessions`). The direction is to consolidate that
behind a **single always-on MountService** that owns the mount and arbitrates
motion + safety centrally, so sessions request motion from one authority
instead of each grabbing the mount and being torn down by callbacks. The
sun-safety monitor — already always-on and already the sole authority allowed
to move during an emergency jog — is the seed of that service.

---

## 2. Site setup (latitude / longitude)

Set your observing site in `config.toml` under `[seestar_initialization]`:

```toml
[seestar_initialization]
lat = 37.12      # decimal degrees; set to 0 to guess from IP address
long = -123.45   # decimal degrees; set to 0 to guess from IP address
```

The sun-safety monitor reads these (`Config.init_lat` / `Config.init_long`).

> **FAIL-CLOSED SAFETY NOTE.** If lat/long are left at `0,0` (Null Island), the
> monitor **cannot compute the sun's position** and logs
> `SUN SAFETY BLIND: observer location is not set (lat/long = 0,0) ... NOT
> protecting the mount`. It then **fails closed**: the `is_sun_safe`
> pre-flight checks refuse mount motion rather than risk pointing at an
> unknown-position sun. In other words, an unset site does not silently
> disable protection — it blocks motion. Always set a real site before daytime
> or dusk work.

The `[sun_avoidance]` section tunes the guard (all optional):
`enabled`, `min_separation_deg` (default 30°), `alt_threshold_deg`
(default −10°, i.e. active once the sun is within 10° of the horizon),
`jog_speed`, `jog_duration_s`.

---

## 3. Dusk calibration — FAA landmarks (3-DOF)

Route: **`/{telescope_id}/calibrate_rotation`** (e.g. `/1/calibrate_rotation`;
there is a "Calibrate Mount →" link from the Live Tracker page).

At dusk the sky is too bright to plate-solve but ground beacons are lit and
crisp, so calibrate against **FAA obstruction landmarks**:

1. Pick the **`☀️ Daytime (FAA landmarks)`** mode button.
2. Use the **unified target picker** — the page states it accepts *"any
   combination of FAA landmarks, celestial bright stars/planets, and
   plate-solve free-aim"* targets. FAA entries live under the
   **`📍 FAA landmarks`** heading.
3. The default FAA target is the **Hyperion primary beacon stack**
   (`HYPERION_06_000301`, the first entry in `DEFAULT_LANDMARKS`,
   `scripts/trajectory/faa_dof.py`) — the closest lit primary calibration
   target for the reference Dockweiler site. Substitute your own local
   landmarks if you observe elsewhere.
4. Sight each landmark by clicking it in the live view crosshair
   (`<img id="cal-vid">`), which streams from the imaging server.

**Why ≥3 sightings.** The mount rotation is a full 3-DOF (yaw / pitch / roll)
solve. With one or two sightings roll is unobservable, so the fit auto-degrades
to yaw-only; a stable 3-DOF solution needs at least
`MIN_SIGHTINGS_FOR_3DOF = 3` sightings from different directions
(`device/rotation_calibration.py`). Give it three or more well-separated
landmarks.

---

## 4. Night auto-calibrate — onboard plate solver

Same page, **`🌙 Nighttime (plate solve)`** mode. Once it's dark enough to
solve fields:

- Press **`🤖 Auto-calibrate (3 sightings)`** (`#cal-night-auto`). This runs the
  hands-free `NighttimeAutoRunner`: it slews the mount to a spread of sky
  waypoints, captures and **plate-solves each field on the Seestar's onboard
  solver**, and fits the same 3-DOF mount rotation as the daytime workflow.
- The header reads **`Sightings: 0/3`** — it needs
  `MIN_SIGHTINGS_FOR_APPLY = 3` solved points before it will apply a solution
  (the 3-DOF fit is ill-conditioned below three).
- Solutions are **refraction-corrected**: astropy applies atmospheric
  refraction in the AltAz transform (`device/nighttime_calibration.py`),
  which matters increasingly below ~60° elevation. The auto-runner keeps
  sightings above `MIN_SIGHTING_ALTITUDE_DEG = 10°`.
- You can also press **`📷 Capture sighting`** (`#cal-night-capture`) to add
  points manually.

---

## 5. Live plane tracking + AVI recording

Route: **`/{telescope_id}/live_tracker`** (e.g. `/1/live_tracker`).

**Pick a target.** The **Target** card has two tabs:

- **Live** — aircraft from the **adsb.fi** feed, polled in the background by
  `LiveADSBProvider` (`device/live_tracker.py`). Each row shows az/el, range,
  altitude, ground speed, and az/el rates.
- **Cached** — pre-recorded trajectories from `data/trajectories/*.jsonl`.

Filter by callsign/id, click a target, then **`Start track`**. Toggles:
**`auto-slew`** (drive the mount vs. compute-only) and **`dry-run`**.

**Offsets** (fine-aim while tracking; all live sliders that POST to
`/api/{id}/live_tracker/offsets`):

- **Time offset** — `Δt` lead/lag in seconds.
- **Spatial offset** — either the **Az / El** basis (`Az`, `El` sliders) or the
  **Along / Cross** basis (along-track / cross-track relative to the target's
  heading). The 2-D pad shows the combined bias; **Reset Az/El**,
  **Reset Along/Cross**, and **Reset all** zero them.

The live **Tracking status** card shows the phase pill, target, elapsed time,
reference az/el, and az/el error, plus a rolling error chart.

**AVI recording.** The **Recording** card (top-left of the page) hosts the
record control; it loads `partials/live_video_record.html` via
`{{ root }}/live/video` and toggles firmware recording with the
`start_record_avi` / `stop_record_avi` RPCs (`front/app.py`). Use it to capture
a pass to disk on the telescope while you track.

> The live video `<img id="lt-vid">` self-heals: if the MJPEG stream closes
> (e.g. the mount was idle at connect time) the page re-opens it on a 3 s
> throttle, and the page kicks the firmware into scenery view on load
> (guarded — it never tears down an active star-imaging session).

---

## 6. Network stats + stream-quality controls (this PR)

The telescope's **single wifi uplink carries both the video** (RTSP on 4554 /
preview on 4800) **and the mount RPCs** (4700). When live video is heavy, the
video can starve the command channel, so mount commands arrive late. Two new
pieces on the **Live Tracker** page help you see and manage this.

### 6a. Network readout (`partials/net_stats.html`)

The **Network** card polls `/api/{id}/netstats` every 3 s (plain fetch, no DOM
swap) and shows four numbers:

- **video** — bytes/s served to *this* browser (rolling ~10 s window).
- **fps** — distinct frames/s served.
- **rpc p50** / **rpc p95** — median / 95th-percentile **round-trip latency of
  mount RPCs** over the recent command history (`dev.rpc_rtt_stats()`).

**Reading it:** rising **rpc p95** *at the same time as* a heavy **video**
byte-rate is the fingerprint of the uplink saturating and delaying mount
commands. If RPC latency is fine, the uplink is not your problem.

### 6b. Stream-quality knobs (`Stream quality` card)

Two sliders, backed by `GET`/`POST /api/{id}/stream_quality` and persisted to
the **`[stream_quality]`** section of `config.toml`:

- **FPS** — `max_stream_fps`: cap on frames/s served to the browser
  (`0` = uncapped, the default).
- **JPEG** — `jpeg_quality`: JPEG encode quality of served frames
  (10–100; `95` = OpenCV default = historical behavior).

```toml
[stream_quality]
max_stream_fps = 0    # 0 = uncapped
jpeg_quality = 95     # 10-100
```

> **What these DO and DON'T do — read this before you reach for them.**
> They are **browser-hop only**, but **shared and persistent**: they shape the
> MJPEG stream this imaging server re-encodes for *every* viewer and live page
> of this telescope (fewer frames, smaller JPEGs), and the setting persists in
> `config.toml` across restarts.
> They do **not** change what the *telescope* pushes over its wifi uplink — the
> RTSP H.264 bitrate is fixed by firmware and there is **no** app-level or
> firmware knob for its bitrate/resolution/fps. So lowering FPS/JPEG helps a
> weak laptop, a slow laptop→browser link, or CPU on the imaging host, but it
> will **not** cut mount-command latency from a saturated telescope uplink.

**When wifi saturates and mount commands lag — what actually helps:**

1. **Stop live video.** The only application-level lever that relieves the
   *telescope* uplink is to reduce what the telescope sends: stop the live
   view / recording so the firmware stops pushing RTSP/preview frames over
   wifi. Then RPC latency (watch **rpc p95**) should recover.
2. **Move the telescope closer to the router / reduce interference.** The
   uplink is a physical-layer resource; the application cannot QoS or
   prioritize RPC over video on it.
3. Use the FPS / JPEG sliders only to fix a *browser-side* or *host-CPU*
   bottleneck — not an uplink one.

There is intentionally **no** "stream bitrate/resolution/codec" control here:
the firmware does not expose one, and this fork does not invent it.
