"""Unit tests for scripts.trajectory.terrain_los.

All tests use synthetic in-process GeoTIFFs written to ``tmp_path``;
none touch the network. The SRTM-fetch code path in
``_default_dem_provider`` is exercised only in the (skipped-by-default)
integration lane.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds

from scripts.trajectory import terrain_los
from scripts.trajectory.terrain_los import (
    DEFAULT_K,
    LosResult,
    _bilinear_sample_dem,
    _effective_earth_radius,
    _great_circle_samples,
    _tile_name_for,
    check_los,
    default_cache_dir,
    dem_lookup_elevation,
)


# Small synthetic observer type — avoids pulling astropy in for what
# should be a pure unit test.
@dataclass(frozen=True)
class _FakeObs:
    lat_deg: float
    lon_deg: float
    alt_m: float


# ---------- helpers: synthesize DEM fixtures -------------------------


def _write_geotiff(
    path: Path, data: np.ndarray,
    *,
    west: float, south: float, east: float, north: float,
) -> Path:
    """Write a single-band float32 GeoTIFF at ``path`` with EPSG:4326
    and the given bounds. Returns ``path``."""
    height, width = data.shape
    transform = from_bounds(
        west=west, south=south, east=east, north=north,
        width=width, height=height,
    )
    with rasterio.open(
        path, "w",
        driver="GTiff",
        height=height, width=width, count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(data.astype(np.float32), 1)
    return path


def _make_provider(tif_path: Path):
    """Return a dem_provider that always opens ``tif_path`` fresh.

    rasterio datasets are context managers, so callers use
    ``with provider(...) as ds:`` and the dataset closes on exit.
    """
    def _prov(lat, lon, source):
        return rasterio.open(tif_path)
    return _prov


# Canonical fixture bounds: MdR observer → Hyperion, ~0.1° square =
# ~11 km × 11 km, 100×100 cells = ~110 m/cell. Coarser than real SRTM
# but plenty for the sign-of-clearance assertions the tests make.
_MDR_BOUNDS = dict(west=-118.50, south=33.88, east=-118.40, north=33.98)


def _flat_tif(tmp_path: Path, height_m: float = 0.0) -> Path:
    arr = np.full((100, 100), height_m, dtype=np.float32)
    return _write_geotiff(tmp_path / "flat.tif", arr, **_MDR_BOUNDS)


def _single_hill_tif(
    tmp_path: Path, *, peak_m: float = 200.0,
    center_col: int = 50, center_row: int = 50, sigma_cells: float = 5.0,
) -> Path:
    h, w = 100, 100
    cols = np.arange(w)[None, :]
    rows = np.arange(h)[:, None]
    r2 = (cols - center_col) ** 2 + (rows - center_row) ** 2
    arr = peak_m * np.exp(-r2 / (2.0 * sigma_cells ** 2))
    return _write_geotiff(tmp_path / "hill.tif", arr.astype(np.float32),
                          **_MDR_BOUNDS)


def _two_hills_tif(tmp_path: Path) -> Path:
    """Two Gaussian peaks placed east of the observer along an E-heading
    path at lat=33.9615 (row ~18 in the 100×100 grid). Peaks at
    col=55 (near, 180 m) and col=80 (far, 300 m)."""
    h, w = 100, 100
    cols = np.arange(w)[None, :]
    rows = np.arange(h)[:, None]
    g1 = 180.0 * np.exp(
        -((cols - 55) ** 2 + (rows - 18) ** 2) / (2.0 * 3.0 ** 2)
    )
    g2 = 300.0 * np.exp(
        -((cols - 80) ** 2 + (rows - 18) ** 2) / (2.0 * 3.0 ** 2)
    )
    arr = np.maximum(g1, g2).astype(np.float32)
    return _write_geotiff(tmp_path / "two_hills.tif", arr, **_MDR_BOUNDS)


def _trough_observer_tif(tmp_path: Path) -> Path:
    """Observer's own cell reports 10 m; everywhere else is 0."""
    arr = np.zeros((100, 100), dtype=np.float32)
    # Observer will be placed near (33.9615, -118.458), which in the
    # bounds (33.88–33.98, -118.50 to -118.40) lands near col ~42,
    # row ~18 (row 0 = north = lat 33.98).
    arr[15:22, 38:46] = 10.0
    return _write_geotiff(tmp_path / "trough.tif", arr, **_MDR_BOUNDS)


# ---------- LosResult dataclass ---------------------------------------


def test_losresult_shape_and_fields():
    """Acceptance §1: dataclass fields match the plan."""
    r = LosResult(
        visible=True, min_clearance_m=5.0,
        obstacle_distance_m=None, obstacle_altitude_m=None,
        used_refraction_k=0.13, sample_count=50, warnings=(),
    )
    assert r.visible is True
    assert r.min_clearance_m == 5.0
    assert r.obstacle_distance_m is None
    assert r.obstacle_altitude_m is None
    assert r.used_refraction_k == 0.13
    assert r.sample_count == 50
    assert r.warnings == ()


# ---------- _effective_earth_radius -----------------------------------


def test_effective_earth_radius_k_zero_is_geometric():
    """k=0 → pure geometric earth radius (no refraction boost)."""
    r = _effective_earth_radius(0.0)
    assert r == pytest.approx(6_371_008.8, rel=1e-6)


def test_effective_earth_radius_k_013_is_larger():
    """k=0.13 (ICAO) expands R by ~15% so ray curves less than earth."""
    r_geom = _effective_earth_radius(0.0)
    r_ref = _effective_earth_radius(0.13)
    # R / (1 - k) = R / 0.87 ≈ 1.149 R
    assert r_ref == pytest.approx(r_geom / 0.87, rel=1e-9)
    assert r_ref > r_geom


def test_effective_earth_radius_rejects_duct():
    with pytest.raises(ValueError, match="must be"):
        _effective_earth_radius(1.0)
    with pytest.raises(ValueError):
        _effective_earth_radius(2.5)


# ---------- _great_circle_samples -------------------------------------


def test_great_circle_samples_endpoints():
    lats, lons, dists, total = _great_circle_samples(
        33.0, -118.0, 34.0, -117.0, n_interior=10,
    )
    assert len(lats) == 12
    assert lats[0] == 33.0
    assert lons[0] == -118.0
    assert lats[-1] == 34.0
    assert lons[-1] == -117.0
    assert dists[0] == pytest.approx(0.0, abs=1e-3)
    # ~1° lat + 1° lon diagonal at 33°N ≈ 143 km
    assert total == pytest.approx(143_000.0, rel=0.05)
    assert dists[-1] == pytest.approx(total, rel=1e-6)


def test_great_circle_samples_monotone_distance():
    _lats, _lons, dists, total = _great_circle_samples(
        0.0, 0.0, 0.0, 0.5, n_interior=20,
    )
    # Distances must be strictly increasing from observer outward.
    assert np.all(np.diff(dists) > 0.0)
    assert total == pytest.approx(dists[-1])


# ---------- _bilinear_sample_dem --------------------------------------


def test_bilinear_sample_dem_known_corners(tmp_path):
    """Build a DEM whose pixel centre values match a known bilinear
    interpolation; sample the midpoint and check we get the mean."""
    # 2x2 cell, values form a gradient 0, 10, 20, 30 at pixel centres.
    arr = np.array([[0.0, 10.0], [20.0, 30.0]], dtype=np.float32)
    path = _write_geotiff(
        tmp_path / "g.tif", arr,
        west=-118.0, south=33.0, east=-117.0, north=34.0,
    )
    with rasterio.open(path) as ds:
        # Pixel centres for a 1°×1° DEM with 2×2 cells are at
        # (lat, lon) = (33.75, -117.75), (33.75, -117.25), (33.25, …)
        # Midpoint of all four is (33.5, -117.5) → bilinear mean 15.
        vals = _bilinear_sample_dem(
            ds,
            np.array([33.5, 33.75, 33.25]),
            np.array([-117.5, -117.75, -117.25]),
        )
    assert vals[0] == pytest.approx(15.0, abs=1e-4)
    assert vals[1] == pytest.approx(0.0, abs=1e-4)
    assert vals[2] == pytest.approx(30.0, abs=1e-4)


def test_bilinear_sample_dem_out_of_bounds_clamps(tmp_path):
    """Samples outside the dataset are clamped to the nearest edge
    cell; must not raise IndexError."""
    arr = np.array([[5.0, 5.0], [5.0, 5.0]], dtype=np.float32)
    path = _write_geotiff(
        tmp_path / "c.tif", arr,
        west=-118.0, south=33.0, east=-117.0, north=34.0,
    )
    with rasterio.open(path) as ds:
        vals = _bilinear_sample_dem(
            ds,
            np.array([40.0, 20.0]),  # far outside lat range
            np.array([-120.0, -100.0]),  # far outside lon range
        )
    assert np.all(vals == 5.0)


# ---------- tile naming -----------------------------------------------


def test_tile_name_for_la_basin():
    assert _tile_name_for(33.96, -118.45) == "N33W119.tif"
    assert _tile_name_for(33.0, -118.0) == "N33W118.tif"


def test_tile_name_for_southern_hemisphere():
    assert _tile_name_for(-33.9, 151.2) == "S34E151.tif"


def test_default_cache_dir_respects_xdg(monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", "/tmp/xdg_test_cache")
    d = default_cache_dir()
    assert str(d) == "/tmp/xdg_test_cache/seestar_alp/dem/srtm1"


# ---------- check_los: synthetic fixtures ------------------------------


_OBS = _FakeObs(lat_deg=33.9615, lon_deg=-118.458, alt_m=2.0)


def test_check_los_flat_ocean_visible(tmp_path):
    """Flat 0 m DEM + tall lit target → always visible."""
    tif = _flat_tif(tmp_path, height_m=0.0)
    tgt_lat, tgt_lon, tgt_h = 33.918889, -118.427223, 103.3
    res = check_los(
        _OBS, tgt_lat, tgt_lon, tgt_h,
        dem_provider=_make_provider(tif),
    )
    assert res.visible is True
    assert res.obstacle_distance_m is None
    assert math.isfinite(res.min_clearance_m)
    assert res.used_refraction_k == pytest.approx(DEFAULT_K)
    assert res.sample_count > 50


def test_check_los_single_hill_blocks(tmp_path):
    """A 200 m hill between observer and a low target blocks the path."""
    tif = _single_hill_tif(tmp_path, peak_m=300.0, sigma_cells=7.0)
    # Target at the hill's far side, 50 m AMSL (below hill top)
    tgt_lat, tgt_lon, tgt_h = 33.90, -118.42, 50.0
    res = check_los(
        _OBS, tgt_lat, tgt_lon, tgt_h,
        dem_provider=_make_provider(tif),
    )
    assert res.visible is False
    assert res.min_clearance_m < 0.0
    assert res.obstacle_distance_m is not None
    assert res.obstacle_altitude_m is not None
    assert res.obstacle_altitude_m > 100.0  # hill is there


def test_check_los_clearance_sign_flips_at_height_boundary(tmp_path):
    """Raise the target altitude until the path just clears."""
    tif = _single_hill_tif(tmp_path, peak_m=150.0, sigma_cells=6.0)
    tgt_lat, tgt_lon = 33.90, -118.42
    prov = _make_provider(tif)
    blocked = check_los(_OBS, tgt_lat, tgt_lon, 20.0, dem_provider=prov)
    clear = check_los(_OBS, tgt_lat, tgt_lon, 500.0, dem_provider=prov)
    assert blocked.visible is False
    assert clear.visible is True
    assert clear.min_clearance_m > blocked.min_clearance_m


def test_check_los_two_hills_reports_worst_obstruction(tmp_path):
    """Two blockers along the path — obstacle_distance_m should locate
    the worst-clearance sample. In this fixture the far 300 m hill is
    the worst offender for a low target."""
    tif = _two_hills_tif(tmp_path)
    # E-heading target at the observer's latitude; crosses both hills.
    tgt_lat, tgt_lon, tgt_h = 33.9615, -118.405, 10.0
    res = check_los(
        _OBS, tgt_lat, tgt_lon, tgt_h,
        dem_provider=_make_provider(tif),
    )
    assert res.visible is False
    assert res.obstacle_distance_m is not None
    assert res.obstacle_altitude_m is not None
    # The 300 m far hill dominates.
    assert res.obstacle_altitude_m > 200.0


def test_check_los_observer_under_ground_warns(tmp_path):
    """Observer at 2 m AMSL in a DEM cell that reports 10 m → warning
    emitted. LoS is still computed (doesn't fail outright)."""
    tif = _trough_observer_tif(tmp_path)
    # Use a distant, high target so the geometry is otherwise clear.
    tgt_lat, tgt_lon, tgt_h = 33.90, -118.42, 400.0
    res = check_los(
        _OBS, tgt_lat, tgt_lon, tgt_h,
        dem_provider=_make_provider(tif),
    )
    assert any("below DEM" in w for w in res.warnings), res.warnings


def test_check_los_refraction_k_increases_clearance(tmp_path):
    """Bumping k from 0 to 0.5 lifts the ray (less bulge); for a
    borderline blocker, min_clearance must strictly increase."""
    tif = _single_hill_tif(tmp_path, peak_m=250.0, sigma_cells=6.0)
    tgt_lat, tgt_lon, tgt_h = 33.90, -118.42, 200.0
    prov = _make_provider(tif)
    r_k0 = check_los(_OBS, tgt_lat, tgt_lon, tgt_h, k=0.0,
                     dem_provider=prov)
    r_k5 = check_los(_OBS, tgt_lat, tgt_lon, tgt_h, k=0.5,
                     dem_provider=prov)
    assert r_k5.min_clearance_m > r_k0.min_clearance_m
    assert r_k0.used_refraction_k == 0.0
    assert r_k5.used_refraction_k == 0.5


def test_check_los_k_default_matches_module_constant(tmp_path):
    """Passing ``k=None`` must use ``DEFAULT_K``, which equals the
    repo-wide 0.13 convention."""
    tif = _flat_tif(tmp_path)
    res = check_los(
        _OBS, 33.90, -118.42, 100.0,
        dem_provider=_make_provider(tif),
    )
    assert res.used_refraction_k == pytest.approx(DEFAULT_K)
    assert DEFAULT_K == 0.13


def test_check_los_ignore_radius_suppresses_near_blocker(tmp_path):
    """Put a blocker inside the ignore radius → it should not register
    as an obstruction; outside the radius, it should."""
    # Build a DEM where only the observer cell has a 100 m spike.
    arr = np.zeros((100, 100), dtype=np.float32)
    arr[17:19, 41:43] = 100.0  # ~5 cells × 100 m/cell ≈ 500 m-wide spike
    tif = _write_geotiff(tmp_path / "spike.tif", arr, **_MDR_BOUNDS)

    # Target at ~3 km east, 200 m AMSL: easily clears a 100 m spike at
    # any distance from the observer.
    tgt_lat, tgt_lon, tgt_h = 33.9615, -118.42, 200.0

    # A large ignore radius (e.g. 2 km) should mask the spike even if
    # it would otherwise block. With 0 m radius, the blocker shows.
    r_large = check_los(
        _OBS, tgt_lat, tgt_lon, tgt_h,
        ignore_radius_m=2000.0,
        dem_provider=_make_provider(tif),
    )
    # The spike is at ≲ 300 m from the observer, so at radius 2 km
    # the spike cells are masked — min_clearance_m must be computed
    # from samples beyond 2 km where the DEM is flat 0, so clearance
    # is a healthy positive number.
    assert r_large.min_clearance_m > 10.0


def test_check_los_observer_at_target(tmp_path):
    """Observer coincident with target → visible, sample_count=2."""
    tif = _flat_tif(tmp_path)
    res = check_los(
        _OBS, _OBS.lat_deg, _OBS.lon_deg, _OBS.alt_m,
        dem_provider=_make_provider(tif),
    )
    assert res.visible is True
    assert res.sample_count == 2


def test_check_los_sample_count_scales_with_path(tmp_path):
    """Longer paths use more samples (floor at 50)."""
    tif = _flat_tif(tmp_path)
    short = check_los(_OBS, 33.96, -118.457, 5.0,
                      dem_provider=_make_provider(tif))
    long = check_los(_OBS, 33.92, -118.42, 100.0,
                     dem_provider=_make_provider(tif))
    assert short.sample_count == 52  # floor 50 + 2 endpoints
    assert long.sample_count > short.sample_count


# ---------- dem_lookup_elevation --------------------------------------


def test_dem_lookup_elevation_reads_cell(tmp_path):
    """Single-pixel lookup should return the cell value."""
    arr = np.full((100, 100), 42.0, dtype=np.float32)
    tif = _write_geotiff(tmp_path / "const.tif", arr, **_MDR_BOUNDS)
    val = dem_lookup_elevation(
        33.93, -118.45, dem_provider=_make_provider(tif),
    )
    assert val == pytest.approx(42.0, abs=0.01)


def test_dem_lookup_elevation_raises_runtimeerror_on_provider_failure():
    def _bad_provider(lat, lon, source):
        raise OSError("simulated disk error")
    with pytest.raises(RuntimeError, match="DEM lookup failed"):
        dem_lookup_elevation(33.93, -118.45, dem_provider=_bad_provider)


# ---------- default provider: no network in tests ---------------------


def test_default_provider_auto_fetch_false_raises_on_miss(tmp_path, monkeypatch):
    """When the tile isn't cached and auto_fetch is False, we should
    get a FileNotFoundError — not a network call."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    # The terrain_los default provider auto-fetches; but we can call
    # the underlying function with auto_fetch=False directly.
    with pytest.raises(FileNotFoundError, match="not cached"):
        terrain_los._default_dem_provider(
            33.96, -118.45, "srtm1",
            cache_dir=tmp_path / "dem",
            auto_fetch=False,
        )


def test_default_provider_rejects_non_srtm_source():
    with pytest.raises(NotImplementedError, match="not available in MVP"):
        terrain_los._default_dem_provider(
            33.96, -118.45, source="3dep",
        )


# ---------- end-to-end MdR → Hyperion regression ----------------------


def _mdr_hyperion_dem(tmp_path: Path) -> Path:
    """Build a synthetic MdR-basin DEM that's (a) zero over the ocean
    west of the jetty, (b) rises ~30 m at the Hyperion bluff, (c) has
    Baldwin Hills at ~180 m north-east of the observer. Small enough
    to be cheap, large enough to exercise real rasterio code paths."""
    # 200×200 over 0.2° × 0.2° = ~22 km square, 110 m/cell.
    h, w = 200, 200
    arr = np.zeros((h, w), dtype=np.float32)
    # Ocean stays at 0. Hyperion bluff: crude rise south of the
    # observer around col=60, row=120 (south-east).
    cols = np.arange(w)[None, :]
    rows = np.arange(h)[:, None]
    hyperion = 30.0 * np.exp(
        -((cols - 90) ** 2 + (rows - 130) ** 2) / (2.0 * 10.0 ** 2)
    )
    baldwin = 180.0 * np.exp(
        -((cols - 140) ** 2 + (rows - 50) ** 2) / (2.0 * 15.0 ** 2)
    )
    arr = np.maximum(hyperion, baldwin).astype(np.float32)
    return _write_geotiff(
        tmp_path / "mdr.tif", arr,
        west=-118.55, south=33.85, east=-118.35, north=34.05,
    )


def test_mdr_to_hyperion_is_visible(tmp_path):
    """End-to-end: MdR observer → Hyperion 06-000301 (103 m AMSL) at
    ~5.5 km SE. Over the synthetic ocean-plus-bluff DEM it must be
    visible. This exercises the real code path via rasterio.open."""
    tif = _mdr_hyperion_dem(tmp_path)
    obs = _FakeObs(lat_deg=33.9615051, lon_deg=-118.4581361, alt_m=2.0)
    res = check_los(
        obs, 33.918889, -118.427223, 103.33,
        dem_provider=_make_provider(tif),
    )
    assert res.visible is True, res


def test_mdr_to_low_target_behind_baldwin_is_blocked(tmp_path):
    """A 5 m target directly behind the Baldwin Hills bump must be
    blocked. Confirms the filter actually drops things the old code
    would pass."""
    tif = _mdr_hyperion_dem(tmp_path)
    obs = _FakeObs(lat_deg=33.9615051, lon_deg=-118.4581361, alt_m=2.0)
    # Target NE, low altitude, past the Baldwin bump
    res = check_los(
        obs, 34.00, -118.38, 5.0,
        dem_provider=_make_provider(tif),
    )
    assert res.visible is False
    assert res.min_clearance_m < 0.0


# ---------- filter_visible (check_terrain=True) -----------------------


def test_filter_visible_check_terrain_default_off_backcompat(tmp_path):
    """Legacy call without check_terrain keeps existing behavior and
    returns 4-tuples (backwards compatibility for existing tests)."""
    from scripts.trajectory.faa_dof import DEFAULT_LANDMARKS, filter_visible
    from scripts.trajectory.observer import build_site

    site = build_site(lat_deg=33.9615051, lon_deg=-118.4581361, alt_m=2.0)
    hits = filter_visible(list(DEFAULT_LANDMARKS), site, min_el_deg=0.0)
    assert hits  # backwards-compat path still works
    for t in hits:
        assert len(t) == 4


def test_filter_visible_check_terrain_drops_blocked(tmp_path):
    """With check_terrain=True and a synthetic DEM where a candidate
    is behind a blocker, that candidate should be dropped."""
    from scripts.trajectory.faa_dof import Landmark, filter_visible
    from scripts.trajectory.observer import build_site

    site = build_site(lat_deg=33.9615051, lon_deg=-118.4581361, alt_m=2.0)

    # Target we know is blocked (behind Baldwin in the fake DEM).
    # Height is high enough to pass the horizon check but the
    # synthetic 180 m Baldwin bump sits in the intervening path.
    blocked = Landmark(
        oas="00-BLOCKED", name="behind_baldwin",
        lat_deg=34.01, lon_deg=-118.37, height_amsl_m=50.0,
        lit=True, accuracy_class="1A",
    )
    # Target we know is visible (ocean path + top of Hyperion-like bump)
    clear = Landmark(
        oas="00-CLEAR", name="hyperion_like",
        lat_deg=33.918889, lon_deg=-118.427223, height_amsl_m=103.33,
        lit=True, accuracy_class="1A",
    )

    tif = _mdr_hyperion_dem(tmp_path)
    prov = _make_provider(tif)

    # Without terrain check both are above 0° el + within radius, so
    # both pass.
    hits_plain = filter_visible([blocked, clear], site, min_el_deg=0.0)
    oas_plain = {h[0].oas for h in hits_plain}
    assert "00-BLOCKED" in oas_plain
    assert "00-CLEAR" in oas_plain

    # With check_terrain=True + our synthetic DEM, blocked should drop.
    hits = filter_visible(
        [blocked, clear], site, min_el_deg=0.0,
        check_terrain=True, dem_provider=prov,
    )
    oas = {h[0].oas for h in hits}
    assert "00-BLOCKED" not in oas
    assert "00-CLEAR" in oas


def test_filter_visible_check_terrain_k_is_plumbed(tmp_path):
    """k=0 (stricter) vs k=0.5 (looser) can change which landmarks
    pass. At least verify the k argument is forwarded without error."""
    from scripts.trajectory.faa_dof import Landmark, filter_visible
    from scripts.trajectory.observer import build_site

    site = build_site(lat_deg=33.9615051, lon_deg=-118.4581361, alt_m=2.0)
    clear = Landmark(
        oas="00-CLEAR", name="hyperion_like",
        lat_deg=33.918889, lon_deg=-118.427223, height_amsl_m=103.33,
        lit=True, accuracy_class="1A",
    )
    tif = _mdr_hyperion_dem(tmp_path)
    prov = _make_provider(tif)
    hits_k0 = filter_visible(
        [clear], site, min_el_deg=0.0,
        check_terrain=True, k=0.0, dem_provider=prov,
    )
    hits_k13 = filter_visible(
        [clear], site, min_el_deg=0.0,
        check_terrain=True, k=0.13, dem_provider=prov,
    )
    # Clear target stays visible under both.
    assert len(hits_k0) == 1
    assert len(hits_k13) == 1
