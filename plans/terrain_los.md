# Terrain line-of-sight MVP

Short status doc for the DEM-based LoS filter. The full design is in
`~/.openclaw/workspace/terrain-los-plan.md` (not committed to this
repo). This file captures what's implemented, what isn't, and the
decisions made for MVP.

## What this does

- `scripts/trajectory/terrain_los.check_los(observer, tgt_lat, tgt_lon,
  tgt_h_amsl, ...)` runs a refraction-corrected WGS-84 geodesic sample
  along the path and returns an `LosResult` with the worst clearance,
  the location/height of the worst blocker, and the `k` that was
  applied. Blocking is per-sample against a DEM tile.
- `scripts/trajectory/terrain_los.dem_lookup_elevation(lat, lon)`
  returns ground elevation from the same DEM — used as an observer
  altitude source in the calibration REPL.
- `scripts/trajectory/faa_dof.filter_visible(..., check_terrain=True)`
  adds a terrain-LoS pass on top of the existing horizon+slant filter
  so FAA landmarks behind Baldwin Hills / Hyperion bluff drop out of
  the calibration candidate list.
- `calibrate_rotation.py --altitude-source dem_lookup` and the menu's
  new `[1] DEM lookup (recommended)` option feed the DEM tile into the
  observer altitude, closing the "observer below ground" degenerate
  case by construction (the DEM that drives LoS is the same DEM that
  defines the observer's ground level).

## MVP scope (§I of the design doc)

Implemented here:

| Decision | Choice |
|---|---|
| DEM source | SRTM 1-arcsec global (SRTMGL1 v3) via OpenTopography |
| Tile cache | `~/.cache/seestar_alp/dem/srtm1/` (respects `XDG_CACHE_HOME`) |
| Geodesic path | `pyproj.Geod(ellps="WGS84")` |
| Bilinear interp | hand-rolled (numpy vectorized; ~20 lines) |
| Refraction `k` | exposed as argument, default `0.13` matching `device.rotation_calibration.terrestrial_refraction_deg` |
| Under-ground handling | DEM-primary altitude eliminates by construction, plus a 30 m ignore radius, plus the `LosResult.warnings` field on mismatch |
| MVP coverage | global (SRTM covers 60°N–56°S; one tile per observer) |
| Live-tracker re-check | off — terrain LoS is a preflight only |
| Deps | `rasterio`, `pyproj`, `geographiclib` as first-class `requirements.txt` entries |

## Deferred to Phase 2

Explicitly out of scope for this PR:

- USGS 3DEP 1/3 arcsec (10 m) for higher-precision CONUS coverage.
- LARIAC 0.25 m LA-county LiDAR.
- DSM variant for wooded / urban observers.
- Setup-wizard UX for `platform_height_agl` (rooftop / jetty observers).
  The argument already exists in the API; the UX isn't built yet.
- Per-tick LoS re-check in the live tracker (cheap but pointless for
  aircraft / satellites).
- Persistent `LosResult` cache at `~/.cache/seestar_alp/los_cache.json`.
- Multi-tile stitching when a path crosses a 1° boundary.

## Testing note

Unit tests in `tests/test_terrain_los.py` never hit the network — every
`check_los` / `dem_lookup_elevation` call in tests is constructed with
a `dem_provider` that reads an in-test-scoped synthetic GeoTIFF. The
default provider (on-disk cache + OpenTopography fetch) is reachable
only when the caller omits `dem_provider`; the one negative test that
touches it asserts we get a `FileNotFoundError` on cache miss rather
than silently hitting the network.

The end-to-end "MdR → Hyperion is visible" regression uses a small
synthetic DEM with Hyperion-bluff and Baldwin-Hills bumps in the right
locations — no real SRTM tile is committed to the repo (licensing and
size). If Phase 2 adds real tile regression, put tiles under
`tests/fixtures/dem/` with a provenance note.
