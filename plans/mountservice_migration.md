# MountService migration and post-PR-26 roadmap

Living plan and status for the mount-ownership refactor series, together
with the safety gaps and follow-ups found while reviewing it. **Update the
status of every item a PR touches, in that PR**, so this file stays the
single source of truth.

- **Status as of:** 2026-10-01, written into PR #26 just before it merged.
- **Line references** are to commit `d2d9130` (PR #26's code). The merge
  adds only this file and a runbook pointer, so they match `main` right
  after the merge, and drift after that.
- **Item IDs** (`S*`, `M*`, `F*`, `B*`, `H*`, `U*`) are stable; reference
  them in PR titles and commits, e.g. `fix(safety): S1 route joystick
  through move_scope`.
- **Priorities:** P0 = safety hazard today; P1 = blocks aircraft tracking
  or the migration; P2 = valuable; P3 = nice-to-have. **Effort:** S < ½
  day, M ≈ 1–2 days, L ≈ a week, XL = more.

---

## 1. Where this comes from

1. **Fork audit report, §3 "Migration plan (safety-first, one writer per
   PR)".** It was produced in the Claude Code session that built PRs
   #23–#26 (2026-07-13) and never committed to the repo until now. Its 12
   steps were labelled "PR1…PR12"; they are renamed **A1–A12** here to
   avoid confusion with GitHub PR numbers.
2. **Target architecture.** `docs/operations_runbook.md` §1 describes a
   single always-on **MountService** that owns the mount and arbitrates
   motion and safety centrally. It replaces today's split ownership: the
   device server owns the RPC socket, the LiveTracker runner owns
   `SunSafetyMonitor`, every feature session drives the mount directly,
   and the watchdog fans a stop out to six managers.
3. **Review of PR #26 (2026-09-30/10-01).**
   - Five independent sweeps: the original plan, device-side mount
     ownership, external entry points, PR #26 follow-ups, and the
     plans/TODO backlog.
   - Each finding was checked adversarially against `d2d9130`, followed by
     a completeness pass.
   - 102 findings survived and 3 were rejected; the survivors were merged
     into the items below.
   - Anything that was inferred rather than reproduced is marked
     *(inferred)*.

## 2. Status of the original audit plan (A1–A12)

| # | Step | Status | Where |
|---|---|---|---|
| A1 | Daytime sun sensing from the motor encoder; fail closed on the 0,0 site | **Done** | #23 |
| A2 | Live-track continuity (re-read `valid_range`, `_poll_once` returns, ADS-B `seen_pos` backdating) | **Done** | #23 |
| A3 | Cable and motion safety (`unwind_azimuth` forces cum=0, live-track `pre_check`, cable gate in session) | **Done** | #23, #24 |
| A4 | Boundary hardening (`isfinite` guards, unique atomic-write temp files, HEALPix key, refit clears solution) | **Done** | #23 |
| A5 | Calibration convention (apparent-elevation refraction, ≥3 sightings for 3-DOF) | **Done** | #23, #24 |
| A6 | Front server + templates (threaded server, vendored Chart.js) | **Done**, via threaded *WSGI*, not ASGI, so websockets are still blocked (see M5) | #23 |
| A7 | De-dup control math: pure `servo_tick` with golden traces | **Done** | #26 |
| A8 | Mount seam: `InProcessMount` + `HttpMount` (`device/mount_client.py`) + send lock | **Partial.** The seam merged in #26 and the send lock in #23. The plan's "verify tick latency drop on the simulator" was never done (M1). | #26 |
| A9 | EventHub + websocket/REST streaming ("architecture step 2") | **Open** | M5–M7 |
| A10 | Always-on MountService as sole `scope_speed_move` emitter: live-track first, then one writer per PR; a `MountOwnership` lease replaces `_scope_start_lock`; one sun gate in true sky az/el; MoveAxis and emergency jog on a privileged preempt channel, jog before aborts; sun safety stays an independent watchdog ("architecture step 3") | **Open** | M8–M13 |
| A11 | Consolidation: one `VelocityControllerConfig`, `logging` everywhere, promote `scripts/trajectory` modules into a package, terrain extras | **Partial.** Geometry module, logging, PD-path removal and terrain extra are done (#23). Still open: B7, B13, B14. | — |
| A12 | Satellites + streaming replanning | **Partial.** The trapezoid v0 fix and the forbidden-arc fix for \|Δ\|>180° are done (#23). `TLEProvider` is open (B6). | — |

## 3. Recommended order

1. **H0: Hardware A/B of the PR #26 transport.** PR #26 merged without it,
   and the merge made the in-process transport the default. If anything
   misbehaves, set `[mount_client] force_http = true` and restart the
   session. Answer H1–H4 in the same outing.
2. **Safety and correctness fixes on main.** Each is a small, separate PR:
   - S1, S2, S3+S4, S5, B1
   - S9+S10 together with **M2**
   - S6, S7, S8
   - then the P2 safety items
3. **U1: Sync with upstream** before any step that restructures
   upstream-owned motion code (M4c, M9, M11d, M12).
4. **Behaviour-preserving migration prep:** M1, M3, M4.
5. **Architecture step 2 (EventHub):**
   - M5 (transport decision), M6 (hub core), then M7 (endpoint and pages).
   - M6 needs only M1, so it can run in parallel with the M8 track.
6. **Architecture step 3 (MountService):** M8 → M9 → M10 → M11 → M12.
7. **M13** (single control loop), then **M14** (cleanup).
8. **Tuning, once hardware data exists:** F1, B3, B5 and B4, in parallel
   with steps 4–7.

## 4. Safety gaps on main (independent of the migration)

These all exist on `main` today, are small to fix, and do not need the
MountService. Every fix must ship with tests: `tests/test_front_app_state.py`
and `tests/integration/test_simulator_e2e.py` for any front change (see
AGENTS.md).

| ID | Pri | Eff | Gap | Status |
|---|---|---|---|---|
| S1 | P0 | S | Web joystick sends raw `scope_speed_move`, bypassing lockout, sun pre-flight and jog stop-suppression | Open |
| S2 | P0 | M | Sun monitor and `move_scope` pre-flight treat raw encoder azimuth as sky azimuth | Open |
| S3 | P0 | S | `track()` sun check tests the provider reference, not the commanded pointing; calibrate-motion is seeded at (0,0) | Open |
| S4 | P1 | S | Delegated rotation-calibration slews land at pred + start-encoder offset *(inferred)* | Open |
| S5 | P1 | S | Web sessions call `CumulativeAzTracker.load_or_fresh()` without its argument, so they discard the cable-wrap datum | Open |
| S6 | P1 | S | `Seestar.goto_target` has no sun pre-flight and no lockout check | Open |
| S7 | P1 | M | Watchdog abort leaves the scheduler and firmware gotos running; legacy routes ignore session ownership | Open |
| S8 | P1 | M | Sun monitor fails open when blind; only senses and jogs the primary scope | Open |
| S9 | P1 | S | Sun monitor freezes the site at startup | Open |
| S10 | P1 | S | Hard-coded `127.0.0.1` while the device server binds `Config.ip_address` | Open |
| S11 | P2 | S | Lockout gates only `speed_move`/`move_scope`; check-then-send is not atomic; no cool-down | Open |
| S12 | P2 | M | Sun reader and jog queue behind the single-threaded device HTTP server *(latent, inferred)* | Open (fixed by M2) |
| S13 | P2 | M | Rotation calibration and calibrate-motion can co-drive the mount *(inferred)* | Open |
| S14 | P2 | S | Federation (telescope 0) motion fans out with no per-scope checks | Open |
| S15 | P2 | M | `track()` has no runtime elevation-limit gate; el limits are hard-coded | Open |

**S1. Joystick bypasses sun safety.**
- `TelescopePositionResource.on_post` (`front/app.py:7745-7772`) sends
  `method_sync scope_speed_move` (speed up to 1440, `dur_sec` 3). The
  device action handler forwards it ungated (`device/telescope.py:205-211`).
- `front/public/liveview.js:111-125` re-posts every 250 ms while the
  stick is held.
- This path never reaches `Seestar.move_scope`
  (`device/seestar_device.py:1162-1208`), where the lockout refusal, the
  current-pointing sun check and the stop-suppression during a jog live.
- A held stick overrides each emergency jog. After 3 attempts the lockout
  releases (`device/sun_safety.py:366, 586-615`).
- **Fix:**
  - Call `get_seestar_device(tid).move_scope(int(angle), int(speed), 3)`.
  - Return 409 while a mount-driving session owns the scope.
  - No test covers this route today; add tests.

**S2. Encoder azimuth used as sky azimuth.**
- `make_scope_altaz_reader` returns `(enc_az % 360, enc_alt)` with no
  `MountFrame` (`device/sun_safety.py:700-713`). Encoder az is referenced
  to the cable midpoint at power-on, not to north
  (`plans/velocity_controller_research_plan.md:116-122`).
- So the cone is *displaced* by the tripod yaw, not widened. The docstring
  claims "conservative otherwise" (`:686-688`), which is wrong.
- The jog direction (`compute_jog_angle`, `:506-520`) and
  `Seestar._move_scope_sun_safe` (`device/seestar_device.py:1128-1147`)
  make the same assumption. The monitor never loads `mount_calibration.json`.
- **Fix:**
  - Un-rotate through `load_session_mount_frame`
    (`device/live_tracker.py:186-205`) in the reader, the pre-flight and
    the jog angle.
  - Choose a fail-closed rule for an uncalibrated mount. For example,
    guard the |el − sun_el| < cone band at any azimuth. This is a design
    decision.
  - Add tests with a non-identity yaw.
- This is A10's "one sun gate in true sky az/el", and it can land now.

**S3. `track()` sun check tests the wrong point.**
- `track()` commands `eff_ref = ref + anchor offset + bias`
  (`device/streaming_controller.py:372-373, 527-528`), but the per-tick
  sun check evaluates only `ref` (`:580-594`).
- The calibrate page starts calibrate-motion with body `{}`
  (`front/templates/calibrate_rotation.html:1208-1211`), and the route
  defaults az/el to 0.0 (`front/app.py:5930-5931`). The check therefore
  looks near (0,0) regardless of where the scope actually points.
- With auto-slew off, the live tracker's offset is arbitrary too
  (`device/live_tracker.py:1437-1441`).
- **Fix:**
  - Check `(eff_ref_az, eff_ref_el)`, ideally the measured pointing as
    well, un-rotated (S2).
  - Seed the calibrate-motion `StaticReferenceProvider` from the encoder
    at session start. This also fixes S4.

**S4. Delegated slews land off target** *(inferred, reproduce first)*.
- `CalibrationSession` calls `motion.set_target(pred_az, pred_el)` with
  absolute values (`device/rotation_calibration.py:1706-1712`).
  `StaticReferenceProvider.set_target` replaces the target
  (`device/calibrate_motion.py:152-159`), and `track()` adds the start
  anchor.
- The mount therefore goes to pred + (encoder position at page load),
  while the pre-flight (`:1676-1687`) checked pred.
- **Fix:** the same as S3. Add a test that starts the fake mount away from
  (0,0).

**S5. Cable-wrap datum discarded on every web session.**
- `load_or_fresh` requires `current_wrapped_az_deg`
  (`device/plant_limits.py:161-166`). `device/live_tracker.py:1433` and
  `device/calibrate_motion.py:483` call it with no arguments inside a bare
  `except`.
- The resulting TypeError means each session starts fresh and assumes the
  mount is within ±180° of cable centre (usable limits are ±435°).
- On exit the wrong datum overwrites the good one
  (`device/streaming_controller.py:689-691`), which the CLI tools use.
- **Fix:** measure the wrapped az, pass it in, log instead of swallowing,
  and add a regression test.

**S6. Firmware gotos are unguarded.**
- `goto_target` (`device/seestar_device.py:975-1007`) checks only
  `is_goto()`.
- It is reached from the UI `/goto`, ASCOM slew (`device/telescope.py:2153,
  2207, 2236, 2262`), the Alpaca action, the scheduler and mosaic,
  federation and INDI.
- **Fix:**
  - Factor `_slew_to_ra_dec`'s fail-closed pre-flight (`:1053-1076`) into
    a shared helper.
  - Add a `sun_safety_is_locked_out()` refusal to both `goto_target` and
    `_slew_to_ra_dec`.
  - Keep the `iscope_start_view` path for firmware compatibility.
- Raise to P0 if H1 shows the jog does not pre-empt a goto.

**S7. Watchdog abort is incomplete.**
- `_abort_active_sessions` (`device/live_tracker_service.py:137-172`) stops
  only the six managers. It never calls `stop_scheduler` (`device/seestar_device.py:2775-2791`),
  `stop_slew` (`:1107-1113`) or `stop_goto_target` (`:944-947`).
- The nighttime auto-runner never interrupts a firmware goto
  (`device/nighttime_calibration.py:926-930`).
- These front routes move the mount without checking session ownership:
  start_scheduler (`front/app.py:2749-2780`), startup/3PPA (`:7325-7355`),
  runpa (`:7942-7950`), `/goto`, park/horizon, `/live/goto`, `/position`.
- **Interim fix:**
  - Stop the scheduler and any goto in the abort. The ordering relative to
    the jog depends on H1.
  - Add a read-only `scope_owner(tid)` that legacy routes consult (409).

**S8. Monitor fails open when blind.**
- A failed or `None` read makes `_tick` do nothing
  (`device/sun_safety.py:494-501`), indefinitely.
- It senses and jogs only `_primary_telescope_id()`, but the lockout is
  process-global (`:739-746`).
- **Fix** *(inferred design)*:
  - After N blind ticks with the sun above threshold, lock out, abort and
    show it in the UI.
  - Run one monitor per configured scope.

**S9. Monitor site frozen at startup.**
- `LiveTrackerMain.start` passes `lat_deg=Config.init_lat,
  lon_deg=Config.init_long` (`device/live_tracker_service.py:59-60`).
  `_site_from_config_or` prefers those stored values over Config
  (`device/sun_safety.py:70-80`), and `reload()` never updates them.
- The `is_sun_safe` pre-flights read Config fresh. After the config form
  or the startup sequence's IP-geolocation (`device/seestar_device.py:1488-1505`)
  replaces 0,0, motion is allowed, but the monitor stays in its
  "location unset, skip" branch until restart.
- **Fix:**
  - Construct the monitor with `None` so it reads Config on each tick, or
    push the site through `reload()`.
  - Add a test.

**S10. Loopback host mismatch.**
- The device server binds `Config.ip_address` (`device/app.py:325-330`;
  the default is `127.0.0.1`).
- These paths hard-code `127.0.0.1`: the monitor reader and jog
  (`device/live_tracker_service.py:113, 128`), the `get_mount_client` HTTP
  fallback (`device/mount_client.py:144, 167`), and the nine front
  `AlpacaClient` sites.
- If the bind is changed to a LAN IP, the monitor goes blind, with only
  DEBUG logs. Since #26, the servo sessions keep moving in-process anyway.
- **Fix:**
  - One host helper (0.0.0.0 maps to 127.0.0.1, otherwise the bound IP).
  - WARNING-level logs on reader failures.
- Do this together with M2.

**S11. Lockout coverage.**
- `sun_safety_is_locked_out` is checked only at
  `device/velocity_controller.py:148` and `device/seestar_device.py:1187`.
  Manager starts, gotos and raw `method_sync` never check it.
- There is no cool-down after the lockout clears (`device/sun_safety.py:610-615`).
- `speed_move` checks, then sends, so the two steps are not atomic.
- **Interim fix:** refuse manager starts and gotos while locked out. Atomic
  check-and-send comes with M8.

**S12. Monitor behind the single-threaded device server** *(inferred)*.
- The device server is `make_server` with no `server_class`
  (`device/app.py:325-330`).
- `start_solve_sync` can hold it for up to 60 s
  (`device/telescope.py:215-220`).
- #26 moved the servo sessions off this queue but left the safety channel
  on it.
- Fixed by M2.

**S13. Co-drive race** *(inferred)*.
- The calibrate-motion and rotation-calibration managers skip each other
  in their busy checks (`device/calibrate_motion.py:600-606`,
  `device/rotation_calibration.py:1857-1861`).
- The fallback `move_to_ff` has no stop signal (`:1733-1749`).
- A page reload during that move can start a second loop.
- **Interim fix:** refuse or hand over. The structural fix is M9.

**S14. Federation (tid 0).**
- Unprefixed `/goto`, `/command`, `/mosaic` and `/startup` default to tid 0
  (`front/app.py:2198, 2234, 2280, 7325`), which fans out to every scope
  (`device/seestar_federation.py:51-63`).
- `get_mount_client(0)` returns the federation over HTTP.
- **Fix:** return 400 for tid 0 on session-start routes. Keep the
  federation nav entry (AGENTS.md).

**S15. Elevation limits.**
- `el_max_deg`/`el_min_deg` are used only in `pre_check`
  (`device/streaming_controller.py:157-158, 202, 255-256`).
- Usable el is hard-coded (`device/live_tracker.py:1139-1140, 1250-1251`),
  while `plant_limits.json` has hard stops of 85.779/−85.882 that are never
  loaded.
- **Fix:** add `ElevationLimits` with a runtime clamp or exit.

## 5. MountService migration steps (M1–M14)

Each step is one reviewable PR, like #26. Behaviour-preserving steps come
before behaviour-changing ones. "Author" means the step comes from A8–A11;
"inferred" means it was added by the review.

| ID | Eff | Step | Depends on | Source | Status |
|---|---|---|---|---|---|
| M1 | S | Transport observability + measured tick latency | — | Author (A8) | Open |
| M2 | M | Sun-monitor reader/jog on a privileged in-process client | — | Inferred (runbook: monitor is the seed) | Open |
| M3 | S | Front's 9 direct `AlpacaClient` sites → `get_mount_client` | — | Inferred | Open |
| M4 | M | Device-side motion primitives: shared stop, cancellable movers, bounded goto | M1, U1 for part (c) | Inferred | Open |
| M5 | S | Decide the front streaming transport (SSE+REST vs ASGI websocket) | — | Inferred (A9 says websocket) | Open |
| M6 | M | EventHub core: typed in-process pub/sub | M1 | Author (A9) | Open |
| M7 | L | Streaming endpoint + REST facade; move live-tracker and velocity-controller pages onto it | M5, M6 | Author (A9) | Open |
| M8 | L | MountService skeleton per telescope (behaviour-identical facade) | M2, M3, M4, M6, S2 | Author (A10) | Open |
| M9 | L | `MountOwnership` lease replaces `_scope_start_lock` / `raise_if_scope_busy` | M8 | Author (A10) | Open |
| M10 | M | Migrate live-track onto the service (first writer) | M9 | Author (A10) | Open |
| M11 | XL | Migrate the remaining writers, one per PR | M10, U1 | Author (A10) | Open |
| M12 | L | Privileged preempt channel; retire the six-manager abort fan-out | M9, M11 | Author (A10) | Open |
| M13 | XL | Single service control loop consuming `servo_core` | M12 | Inferred (#26 commit 32ca25a) | Open |
| M14 | S | Cleanup: delete JSONL tailing and polling, update docs | M7, M13 | Author (A9/A10) | Open |

**M1. Observability.**
- Log the chosen transport and the reason once per `get_mount_client`
  call. That covers `force_http`, device not registered or connected, and
  an import failure (silently swallowed today at `device/mount_client.py:120-123`).
- Expose the transport in the session status payloads.
- Record per-tick work time and overruns in `TickInfo`/`TrackResult`
  (`device/streaming_controller.py:97-131`). Keep this separate from
  `rpc_rtt_stats`, which measures only firmware round-trip.
- Add a simulator benchmark comparing in-process and HTTP.
- **Accept:** status JSON carries the transport (tests); golden traces
  unchanged.

**M2. Privileged in-process safety channel.**
- Build the monitor's reader and jog
  (`device/live_tracker_service.py:98-133`) on `get_mount_client`, keeping
  the HTTP fallback with the correct host (S10).
- Keep the jog as a raw `method_sync` so it bypasses the lockout-aware
  `speed_move`, and keep `is_jog_in_progress` keyed on the int id.
- Optionally add blind-tick escalation (S8).
- **Accept:**
  - The jog is still sent while locked out.
  - Exit-stop suppression still works during a jog.
  - The sun-trip drill passes on both transports.
- **Risk:** the monitor now contends on `Seestar._send_lock`.

**M3. Front transport consolidation.**
- Replace `AlpacaClient("127.0.0.1", …)` at `front/app.py:5044, 5107, 5241,
  5392, 6285, 6625, 6767, 6890, 7108`.
- Add a `_site_lonlat(tid)` helper with a short TTL cache, which removes
  the GPS round-trip on every slew (`:6890`).
- Re-check integration `test_06c`'s 503 expectation.

**M4. Motion primitives.**
- **(a)** One shared motor-stop helper to replace the four copies
  (`device/live_tracker.py:1516-1533`, `device/calibrate_motion.py:531-548`,
  `device/rotation_calibration.py:1250-1268`,
  `device/velocity_controller.py:159-201`). Choose `dur_sec` from H2.
- **(b)** A cancel token for `move_azimuth_to_ff` / `move_elevation_to_ff` /
  `move_to_ff`, which cannot be interrupted today
  (`device/live_tracker.py:1064-1068`). Wrap `goto_origin` stage 1 in
  `_motor_stop_on_exit`.
- **(c)** A device-side, bounded, cancellable, lockout-aware
  `goto_altaz(tid, az, el, timeout, stop_evt)` that accepts ScopeGoto or
  AutoGoto completion.
  - Move the nighttime-auto and visibility slew closures onto it
    (`front/app.py:6502-6599, 6906-6960`).
  - Give `_slew_to_ra_dec`'s `wait_end_op` a deadline
    (`device/seestar_device.py:2815-2821`).
- (a) and (b) are behaviour-preserving; (c) changes behaviour.

**M5. Transport decision.**
- The front is WSGI (`falcon.App` on a threaded `wsgiref` server), with no
  connection-upgrade path. Options:
  - **(a) SSE + REST.** No new dependency; SSE already works for the
    visibility stream (`front/app.py:7257`).
  - **(b)** `falcon.asgi` + uvicorn.
  - **(c)** A side websocket thread.
- Record the decision and the topic list. The audit's position was
  "no broker".

**M6. EventHub core.**
- Publish these from the existing emit points: `mount.telemetry`,
  `control.tick`, `trajectory.ref`, `target.feed`, `calibration.state`,
  `safety.event`.
- Adapt the `add_listener` hooks (`device/protocols/socket_base.py:138`,
  `device/visibility_mapper.py:604`).
- Publish today's swallowed tick-loop errors (B14).
- Bounded queues, drop-oldest; a slow subscriber must never block a tick.

**M7. Streaming endpoint.**
- Move the pages onto the stream, keeping polling as a fallback until it
  is verified on hardware.
- Delete the `_latest_trajectory_ref` JSONL tailing
  (`front/app.py:4244-4380`) only when unused.
- Follow the AGENTS.md HTMX rules: no destructive empty swaps, no
  whole-page overlays, bounded reconnects.

**M8. Service skeleton.**
- One `MountService` per `Config.seestars` entry. It owns:
  - the in-process client, which re-resolves the `Seestar` on reconnect
    (F7)
  - a per-mount command lock that makes lockout check + send atomic (S11)
  - the single sun gate in true sky az/el (S2)
  - one motion API: velocity, stop, bounded goto, view mode, tracking
  - command logging and timing published on the EventHub
- It starts as a facade over the existing functions.
- Sun safety stays an independent watchdog.
- Future service knobs are toml-only unless the config form gets
  `load_from_form` + `render_config_html` plumbing.

**M9. Ownership lease.**
- One acquire-or-409 lease, held for the session's lifetime.
- Write today's allowed concurrency as an explicit policy table:
  - rotation calibration and interactive nighttime delegate to
    calibrate-motion
  - nighttime-auto is a child of interactive nighttime (fixes B1 if not
    fixed earlier)
  - the scheduler, mosaic and gotos become lease holders
- Make preemption explicit; `front/app.py:6538-6550, 6799-6817` stop
  calibrate-motion outside the lock today.
- Port `tests/test_manager_toctou.py` to the policy table.

**M10–M11. Writer migration.**
- Live-track goes first.
- Then one PR each:
  - **(a)** calibrate-motion + rotation calibration's delegation
  - **(b)** rotation calibration's fallback `move_to_ff`, now with
    `az_limits`/`az_tracker` (omitted at
    `device/rotation_calibration.py:1397-1406, 1739-1748`)
  - **(c)** nighttime-auto and visibility gotos
  - **(d)** `goto_target`, `_slew_to_ra_dec`, scheduler/mosaic/spectra and
    the startup sequence
  - **(e)** auto-level, if it is added to the web UI
- The service becomes the owner of the cable-wrap datum.
- **Accept:** `scope_speed_move`/`scope_goto` appear only in the service,
  the privileged channel and CLI tools.

**M12. Preempt channel.**
- Route the emergency jog, ASCOM MoveAxis
  (`device/seestar_device.py:1162-1215`) and the joystick through it, with
  the jog issued before any abort.
- A trip then:
  - revokes the lease
  - cancels any firmware goto and stops the scheduler
  - requires a cool-down or acknowledgement before motion resumes
- Delete `_abort_active_sessions` and the jog-in-progress special cases.
- Add a motion-RPC classifier and gate at the one choke point below the
  transports. This also covers the Alpaca `method_sync` passthrough
  (`device/telescope.py:205-214`), INDI and CLI clients.
- Run one sensing instance per scope with a per-telescope lockout.

**M13. Single control loop.**
- Collapse the four `servo_tick` call sites
  (`device/velocity_controller.py:556, 986, 1319`,
  `device/streaming_controller.py:532`) into one loop per scope.
- Sessions submit targets or providers instead.
- Unify the gains in `VelocityControllerConfig` (B7). Keep the per-site
  convention flags.
- This is the largest behaviour change in the series and needs hardware
  retuning.

**M14. Cleanup.**
- Remove the JSONL-tail and polling readers and the dead start-lock module.
- Rewrite runbook §1 with the entry-point inventory.

## 6. PR #26 follow-ups (F*) and upstream sync (U1)

| ID | Pri | Eff | Item | Status |
|---|---|---|---|---|
| F0 | P2 | S | Correct the PR #26 text: transport is default-on, not "opt-in"; nighttime and visibility do not use the factory; the "~0.2°" claim depends on F1 | **Done** (PR body edited at merge; commit messages are immutable) |
| F1 | P1 | M | Re-tune `tick_dt`/`latency_s` so the in-process gain reaches tracking | Open |
| F2 | P2 | M | Integration test: real `Seestar` over the simulator socket through `InProcessMount` | Open |
| F3 | P2 | M | Simulator kinematics + end-to-end mount-driving session test | Open |
| F4 | P2 | S | Session unit tests through the real `get_mount_client` factory | Open |
| F5 | P2 | S | Document `mount_client`/`force_http`/`servo_core` in the runbook; note the user_hooks `action_*` change | Open |
| F6 | P3 | S | Harden `get_mount_client`: loopback-only shortcut, `force_http` string parsing, logged import failure | Open |
| F7 | P3 | S | `InProcessMount` pins the `Seestar` object; re-resolve on reconnect (fold into M8) | Open |
| F8 | P3 | S | Broaden `servo_core` coverage: more streaming golden traces, numpy-vs-math edge tests, 237-constant parity | Open |
| F9 | P3 | S | Delete the now-dead `_rate_to_speed` (`device/velocity_controller.py:131`) | Open |
| U1 | P2 | M | Sync with upstream `smart-underworld/seestar_alp` (last merged at `df14fe0`, 2026-04-22; upstream has shipped v3.2.2 since) | Open |

- **F1.** The tick rate is still `TICK_DT_S=0.5` / `LATENCY_S=0.4`
  (`device/streaming_controller.py:53-54`), and the movers default
  `tick_dt=0.5`.
  - At 5.7°/s that is a correction about every 2.85° on either transport.
    The ~0.2° figure in #26 needs roughly a 35 ms tick.
  - After H0: lower `tick_dt`, re-fit `latency_s`, re-check
    `TICK_CMD_DUR_S`/`VC_CMD_DUR_S`, and confirm the firmware tolerates the
    higher `scope_speed_move` rate.
- **F2.** Reuse the `two_device_federation` fixture
  (`tests/integration/test_simulator_e2e.py:1027-1060`).
  - Assert transport selection and identical replies on both transports.
  - Assert that concurrent callers stay correlated. `PositionLogger` shares
    the session client from its own thread (`device/live_tracker.py:1411`),
    so those calls now run concurrently where HTTP serialized them.
- **F3.** The simulator has no `scope_get_horiz_coord` handler
  (`simulator/src/seestar_simulator.py:613-624`), and its
  `scope_speed_move` is a bare ack.
  - Add a velocity integrator (speed/237 °/s along `angle`, with the
    `dur_sec` TTL).
  - Then run `CalibrateMotionSession` or `track()` end to end. This
    enables regression tests for S3, S4 and S5.
  - Keep the integration lane under about 5 minutes.
- **F5.** Runbook §1: sessions use `get_mount_client`; `force_http` takes
  effect at the next session start. user_hooks subscribed to `action_*`
  no longer see servo RPCs from the three migrated sessions.
- **U1.** Merge upstream before M4(c), M9, M11(d) and M12, which
  restructure upstream-owned `seestar_device.py` and scheduler code. Keep
  every firmware-compatibility fallback (AGENTS.md).

## 7. Backlog (B*)

| ID | Pri | Eff | Item | Status |
|---|---|---|---|---|
| B1 | P1 | S | **Nighttime auto-calibrate can never start.** The route requires an active interactive session (`front/app.py:6757-6765`), but `NighttimeAutoManager.start` refuses one via `raise_if_scope_busy` (`device/nighttime_calibration.py:1223`, `device/_scope_start_lock.py:133-142`), and `tests/test_manager_toctou.py` enforces the refusal. **Fix:** pass `check_nighttime_interactive=False`, invert that test, add a route test. | Open |
| B2 | P1 | L | Hardware regression pass of the motion stack (July fixes + #26), incl. the unchecked clean 6-setpoint sweep (`plans/velocity_controller_research_plan.md:1641-1644`) | Open |
| B3 | P1 | M | Tune the streaming FF+FB gains. The plan says τ 0.348→0.55 and kp 0.5→1.0 were "applied" (`plans/velocity_controller_research_plan.md:1838-1842`), but the code still has 0.348 / 0.5. Target: az RMS < 0.1°. | Open |
| B4 | P1 | M | Absolute-pointing mode for live tracking: the engagement anchor turns pre-slew residuals into a permanent bias (`device/streaming_controller.py:343-373`) | Open |
| B5 | P2 | M | Trajectory-replay tuning harness (grid over τ/kp/v_corr/latency; plan:1880-1885). Prerequisite for B3. | Open |
| B6 | P2 | L | Live satellite provider `TLEProvider` (A12); skyfield is already a dependency | Open |
| B7 | P2 | M | One `VelocityControllerConfig` with named per-mode values (A11) | Open |
| B8 | P2 | M | Auto pre-positioning / cable-direction planning before a track (plan:1892-1916) | Open |
| B9 | P2 | M | Cable-wrap coverage for non-streaming motion; web recenter / re-pin datum action | Open |
| B10 | P2 | M | Near-zenith (keyhole) handling; the web pre-check does not gate on saturation (`device/streaming_controller.py:216-220` vs the comment at `device/live_tracker.py:1632-1633`) | Open |
| B11 | P2 | S | Policy for intentional sun pointing (`/live/goto` Sun, `start_scan_planet`); whitelist the `/live/mode` params, which are forwarded verbatim (`front/app.py:3229-3238, 3252-3266`) | Open |
| B12 | P3 | S | Refresh stale docs: see §9 | Open |
| B13 | P3 | M | Move `observer`/`faa_dof`/`celestial_targets`/`fetch_aircraft` out of `scripts/trajectory` into a package (A11) | Open |
| B14 | P3 | M | Tick-loop exception discipline: about 9 `except Exception: pass` in `device/live_tracker.py` and 7 in `device/streaming_controller.py` (A11); publish via M6 | Open |
| B15 | P3 | S | `goto_origin` stage 1 has no guaranteed motor stop (no callers today; covered by M4) | Open |
| B16 | P3 | S | `move_to_ff` cable-stop convergence bug (plan bug #4): repro + test | Open |
| B17 | P3 | S | PyInstaller release builds do not bundle `device/plant_limits.json`, so the cable gate silently fails open (`.github/workflows/build_seestar_alp.yaml`, `device/live_tracker.py:1328`) *(inferred)* | Open |
| B18 | P3 | L | Grouped low-priority items (see below) | Open |

B18 groups:
- spiral-search acquisition (plan:1406-1420)
- in-pass pointing refinement
- controller experiments: capped kI, cold-start lag, MPC
- tune_vc handoff probe
- step-response panel on `/velocity_controller`
- auto-level from the web UI
- nighttime refraction P/T override and clock-skew check
- rotation-solver consolidation
- terrain-LoS multi-tile support
- the remaining RA/Dec `scope_goto` paths
- the `/console` route
- INDI port hard-coded to 5555
- upstream `SensorTempWatch` never fires (`device/event_callbacks.py:113`)

## 8. Hardware and firmware questions (H*)

**H0. A/B test of the #26 transport.**
- Run each of these with `force_http = true` and again with `false`,
  restarting the session after each change:
  - calibrate-motion hold and nudges
  - a live-tracker track (compare RMS, peak and saturated ticks)
  - a rotation-calibration `move_to_ff` slew
  - the same runs with live video and a second browser polling
  - Stop pressed mid-track
  - a wifi drop mid-session
  - a sun-trip drill with `min_separation_deg` raised
- **Telling which transport ran:** the Alpaca INFO line `request:
  method_sync` (`device/telescope.py:176-192`) appears only over HTTP,
  while `scope send:` appears on both.

Open firmware questions:
- **H1.** Is a `scope_speed_move` jog honoured, refused or queued while a
  firmware goto is in flight? This decides S6's priority and S7's ordering.
- **H2.** Does a stop with `dur_sec` 0 (rotation calibration) behave like
  `dur_sec` 1 (live tracker, calibrate-motion)?
- **H3.** Which firmware versions emit `ScopeGoto` and which emit
  `AutoGoto` (`front/app.py:6506-6511`)?
- **H4.** Is `angle=360` treated the same as `angle=0`? See `servo_core`'s
  `POS_ADD360_MOD` convention.

## 9. What other docs already track, and what is stale

*Pending: an item-by-item audit of which of the items above the existing
`plans/` and `docs/` files already track (and where they record a wrong
status) is being added in a follow-up commit.* What is already known: the
repo has no GitHub issues; before this file, the only mention of the
MountService anywhere in the docs was the one-paragraph "direction" in
`docs/operations_runbook.md` §1; and the 12-step audit plan existed only in
a chat transcript.

<!-- TRACKING-AUDIT -->

## 10. Resuming this plan in a new session

Paste this into a fresh Claude Code session at the repo root:

```text
We're continuing the MountService migration and safety roadmap for this
repo (seestar_alp fork). Before doing anything else:

1. git pull origin main. Read plans/mountservice_migration.md (the plan and
   its status: sections 3-8 are the work), then AGENTS.md and
   .claude/CLAUDE.md (test lanes, front/HTMX rules, agent model policy).
   Use this machine's Python environment for the test/ruff commands in
   AGENTS.md; the pyenv path written there belongs to another machine.
2. Re-verify, don't trust: line numbers in the plan are from commit
   d2d9130, so re-locate code by symbol. Before starting an item, confirm
   it is still open on main.
3. Tell me the current state in a few lines (which items are open, what's
   next per §3 "Recommended order"), and ask whether I'm at the telescope
   today. If I am, start with H0 (the PR #26 transport A/B test) and
   H1-H4, guiding me step by step and recording the answers in §8.
   Otherwise start with the next open safety item (S1 first).
4. Work rules: one item (or tightly coupled pair) per branch/PR; add the
   tests AGENTS.md requires; run ruff check, ruff format --check and both
   pytest lanes before pushing; and in the same PR update that item's
   Status in plans/mountservice_migration.md (and §2 if an audit step
   changes). Don't push to main directly, and ask me before merging.
```
