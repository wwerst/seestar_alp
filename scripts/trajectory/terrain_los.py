"""Terrain line-of-sight filtering against a DEM.

MVP scope per ``plans/terrain_los.md`` (design doc at
``~/.openclaw/workspace/terrain-los-plan.md`` §I). This module answers:

    "If I stand at the observer's lat/lon/alt and look at a target at
     (lat, lon, amsl), does the intervening terrain block the view,
     accounting for Earth curvature and terrestrial refraction?"

The calibration REPL's FAA-landmark selector uses ``check_los`` to drop
candidates behind intervening ridges (Baldwin Hills, Hyperion bluff, etc.)
before asking the user to pick two. ``dem_lookup_elevation`` feeds into
the altitude menu so the observer's own altitude comes from the same
DEM that drives the LoS — which eliminates the "observer below ground"
degenerate case by construction.

Primary DEM source: SRTM 1-arcsec (SRTMGL1 v3), public-domain, global
between 60°N and 56°S, ~30 m horizontal / ~5 m vertical RMSE, fetched
on-demand from OpenTopography into ``~/.cache/seestar_alp/dem/srtm1/``.
Phase 2 will add USGS 3DEP (10 m) and LARIAC (0.25 m) — the provider
callable makes that a swap at call site rather than a rewrite.

Unit tests must not hit the network; always pass a ``dem_provider``
that reads a synthetic in-repo fixture.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import rasterio
from pyproj import Geod
from rasterio.windows import Window


# Default terrestrial-refraction coefficient. Mirrors the default on
# ``device.rotation_calibration.terrestrial_refraction_deg`` — kept in
# sync by convention, not import, so this module has no hard dep on
# ``device.*``. k=0 is pure geometric; k=0.13 is ICAO over-land; k=0.25
# typical marine inversion.
DEFAULT_K = 0.13

# WGS-84 mean radius in metres (matches the constant in
# device.rotation_calibration).
_EARTH_R_M = 6_371_008.8

# Belt-and-suspenders for close-range DSM noise around the observer
# cell. 30 m covers the SRTM bilinear footprint without hiding genuine
# nearby terrain.
DEFAULT_IGNORE_RADIUS_M = 30.0

# Telescope eyepiece height above ground when the tripod sits on flat
# earth. Applied in ``_altitude_menu`` when the altitude comes from a
# DEM lookup; not used inside ``check_los`` itself.
DEFAULT_EYE_HEIGHT_AGL_M = 1.6

# OpenTopography global-DEM endpoint. Returns a GeoTIFF for the
# requested bbox. Registered API keys get ~50 GB/day; unauth ~1 GB/day.
_OPENTOPOGRAPHY_URL = (
    "https://portal.opentopography.org/API/globaldem"
)


@dataclass(frozen=True)
class LosResult:
    """Outcome of a single line-of-sight check.

    ``min_clearance_m`` is the **minimum** vertical headroom over all
    interior samples — negative means blocked, positive means clear.
    ``obstacle_distance_m`` / ``obstacle_altitude_m`` locate the worst
    offender; both are ``None`` when the path is unobstructed, so a
    consumer can test just on ``visible``.
    """
    visible: bool
    min_clearance_m: float
    obstacle_distance_m: float | None
    obstacle_altitude_m: float | None
    used_refraction_k: float
    sample_count: int
    warnings: tuple[str, ...] = ()


DemProvider = Callable[[float, float, str], "rasterio.DatasetReader"]


def default_cache_dir() -> Path:
    """``$XDG_CACHE_HOME/seestar_alp/dem/srtm1/`` — mirrors the pattern
    used by ``scripts.trajectory.faa_dof.default_cache_path``."""
    root = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(root) / "seestar_alp" / "dem" / "srtm1"


def _effective_earth_radius(k: float) -> float:
    """Refraction-corrected Earth radius: ``R / (1 - k)``.

    At ``k=0`` returns the geometric ``R``. At ``k=0.13`` (ICAO
    over-land) returns ~7.32 Mm. k≥1 is unphysical (duct regime) and
    rejected.
    """
    if k >= 1.0:
        raise ValueError(f"refraction coefficient k={k!r} must be < 1")
    return _EARTH_R_M / (1.0 - float(k))


def _great_circle_samples(
    obs_lat: float, obs_lon: float,
    tgt_lat: float, tgt_lon: float,
    n_interior: int,
    *, geod: Geod | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Sample ``n_interior + 2`` points along the WGS-84 geodesic from
    observer to target. Includes both endpoints.

    Returns ``(lats, lons, dists_from_obs_m, total_path_m)``.
    """
    if geod is None:
        geod = Geod(ellps="WGS84")
    n_interior = max(1, int(n_interior))
    interior = geod.npts(obs_lon, obs_lat, tgt_lon, tgt_lat, n_interior)
    lons = np.empty(n_interior + 2, dtype=np.float64)
    lats = np.empty(n_interior + 2, dtype=np.float64)
    lons[0] = obs_lon
    lats[0] = obs_lat
    for i, (lo, la) in enumerate(interior, start=1):
        lons[i] = lo
        lats[i] = la
    lons[-1] = tgt_lon
    lats[-1] = tgt_lat
    # Distance from observer to each sample along the WGS-84 ellipsoid.
    _fwd, _back, dists = geod.inv(
        np.full_like(lons, obs_lon), np.full_like(lats, obs_lat),
        lons, lats,
    )
    dists = np.asarray(dists, dtype=np.float64)
    return lats, lons, dists, float(dists[-1])


def _bilinear_sample_dem(
    dataset: rasterio.DatasetReader,
    lats: np.ndarray,
    lons: np.ndarray,
) -> np.ndarray:
    """Bilinearly sample the DEM at the given lat/lon points.

    Assumes the dataset is EPSG:4326 (lat/lon in degrees). Returns a
    float64 numpy array of metres AMSL. Out-of-bounds samples clamp to
    the nearest edge cell (the caller is expected to size the DEM to
    cover the intended path).
    """
    inv = ~dataset.transform
    # Affine inverse: (lon, lat) → (col, row) in pixel coordinates.
    # Rasterio/GIS convention: pixel (0, 0) covers the half-open
    # square from (west, north) to (west + cell_w, north - cell_h).
    # The pixel CENTRE is offset by 0.5 from that corner — so for
    # bilinear interp between *cell centres* we subtract 0.5 from the
    # raw affine output. Without this we'd always be one half-cell
    # off and e.g. the midpoint of a 2×2 DEM would sample the
    # bottom-right cell instead of averaging all four.
    cols_f, rows_f = inv * (np.asarray(lons, dtype=np.float64),
                             np.asarray(lats, dtype=np.float64))
    cols_f = cols_f - 0.5
    rows_f = rows_f - 0.5
    cols0 = np.floor(cols_f).astype(np.int64)
    rows0 = np.floor(rows_f).astype(np.int64)
    cols1 = cols0 + 1
    rows1 = rows0 + 1
    dx = cols_f - cols0
    dy = rows_f - rows0

    h = dataset.height
    w = dataset.width
    cols0c = np.clip(cols0, 0, w - 1)
    cols1c = np.clip(cols1, 0, w - 1)
    rows0c = np.clip(rows0, 0, h - 1)
    rows1c = np.clip(rows1, 0, h - 1)

    c_min = int(cols0c.min())
    c_max = int(cols1c.max())
    r_min = int(rows0c.min())
    r_max = int(rows1c.max())
    window = Window(c_min, r_min, c_max - c_min + 1, r_max - r_min + 1)
    # Read band 1 once; DEMs are single-band.
    data = dataset.read(1, window=window).astype(np.float64)

    a = data[rows0c - r_min, cols0c - c_min]
    b = data[rows0c - r_min, cols1c - c_min]
    c = data[rows1c - r_min, cols0c - c_min]
    d = data[rows1c - r_min, cols1c - c_min]
    return (
        a * (1.0 - dx) * (1.0 - dy)
        + b * dx * (1.0 - dy)
        + c * (1.0 - dx) * dy
        + d * dx * dy
    )


def _dem_resolution_m(dataset: rasterio.DatasetReader, mid_lat: float) -> float:
    """Approximate DEM resolution in metres at ``mid_lat``.

    For EPSG:4326 datasets, ``.res`` is in degrees; 1° of latitude ≈
    111,132 m; 1° of longitude scales with ``cos(lat)``. Returns the
    smaller axis so the LoS sample count isn't starved in either
    direction.
    """
    rx, ry = dataset.res
    m_per_deg_lat = 111_132.0
    m_per_deg_lon = 111_132.0 * math.cos(math.radians(mid_lat))
    lon_m = abs(float(rx)) * m_per_deg_lon
    lat_m = abs(float(ry)) * m_per_deg_lat
    return float(max(1.0, min(lon_m, lat_m)))


def _tile_name_for(lat_deg: float, lon_deg: float) -> str:
    """1°×1° tile name in SRTM convention (``N33W119.tif``)."""
    south = int(math.floor(lat_deg))
    west = int(math.floor(lon_deg))
    ns = "N" if south >= 0 else "S"
    ew = "E" if west >= 0 else "W"
    return f"{ns}{abs(south):02d}{ew}{abs(west):03d}.tif"


def _fetch_srtm_tile(
    lat_deg: float, lon_deg: float, cache_dir: Path,
    *, timeout_s: float = 60.0,
) -> Path:
    """Download a single 1°×1° SRTM tile from OpenTopography into
    ``cache_dir``. Requires ``OPENTOPOGRAPHY_API_KEY`` for a registered
    rate limit (unauth works but is stingy). Raises ``RuntimeError``
    on any failure so callers can fall back cleanly.
    """
    import requests  # local import to keep this a soft dep

    cache_dir.mkdir(parents=True, exist_ok=True)
    tile = _tile_name_for(lat_deg, lon_deg)
    path = cache_dir / tile
    south = int(math.floor(lat_deg))
    west = int(math.floor(lon_deg))
    params = {
        "demtype": "SRTMGL1",
        "south": south,
        "north": south + 1,
        "west": west,
        "east": west + 1,
        "outputFormat": "GTiff",
    }
    api_key = os.environ.get("OPENTOPOGRAPHY_API_KEY")
    if api_key:
        params["API_Key"] = api_key
    try:
        resp = requests.get(_OPENTOPOGRAPHY_URL, params=params,
                            timeout=timeout_s, stream=True)
        resp.raise_for_status()
        tmp = path.with_suffix(path.suffix + ".part")
        with tmp.open("wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 16):
                if chunk:
                    f.write(chunk)
        tmp.replace(path)
    except requests.RequestException as exc:
        raise RuntimeError(f"OpenTopography fetch failed: {exc}") from exc
    except OSError as exc:
        raise RuntimeError(f"could not write DEM tile: {exc}") from exc
    return path


def _default_dem_provider(
    lat_deg: float, lon_deg: float, source: str = "srtm1",
    *, cache_dir: Path | None = None, auto_fetch: bool = True,
) -> rasterio.DatasetReader:
    """Open the cached SRTM tile covering (lat, lon); fetch on miss if
    ``auto_fetch``. The returned ``DatasetReader`` is a context manager
    — callers should use ``with dem_provider(...) as ds:`` and let it
    close at scope end.

    Non-``srtm1`` sources raise ``NotImplementedError`` — a Phase 2
    hook for 3DEP / LARIAC.
    """
    if source != "srtm1":
        raise NotImplementedError(
            f"DEM source {source!r} is not available in MVP; "
            "use 'srtm1' or pass a custom dem_provider"
        )
    cache = cache_dir if cache_dir is not None else default_cache_dir()
    cache.mkdir(parents=True, exist_ok=True)
    tile_path = cache / _tile_name_for(lat_deg, lon_deg)
    if not tile_path.exists():
        if not auto_fetch:
            raise FileNotFoundError(
                f"DEM tile not cached at {tile_path}; "
                "pass auto_fetch=True or pre-populate the cache"
            )
        _fetch_srtm_tile(lat_deg, lon_deg, cache)
    return rasterio.open(tile_path)


def dem_lookup_elevation(
    lat_deg: float, lon_deg: float,
    *,
    dem_source: str = "srtm1",
    dem_provider: DemProvider | None = None,
) -> float:
    """Return ground elevation in metres AMSL at (lat, lon).

    Reads a single bilinear sample from the DEM tile covering the
    point. Raises ``RuntimeError`` on any IO/network failure so the
    altitude-menu caller can fall back to Open-Meteo / prior / manual.

    The return value is **ground** elevation. Callers that want the
    telescope eyepiece altitude should add ``platform_height_agl +
    eye_height_agl`` themselves.
    """
    if dem_provider is None:
        dem_provider = _default_dem_provider
    try:
        with dem_provider(lat_deg, lon_deg, dem_source) as dataset:
            val = _bilinear_sample_dem(
                dataset,
                np.array([lat_deg], dtype=np.float64),
                np.array([lon_deg], dtype=np.float64),
            )
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(f"DEM lookup failed: {exc}") from exc
    return float(val[0])


def check_los(
    observer,
    target_lat_deg: float,
    target_lon_deg: float,
    target_h_amsl_m: float,
    *,
    k: float | None = None,
    ignore_radius_m: float = DEFAULT_IGNORE_RADIUS_M,
    dem_source: str = "srtm1",
    dem_provider: DemProvider | None = None,
    min_samples: int = 50,
) -> LosResult:
    """Refraction-corrected DEM line-of-sight from ``observer`` to a
    target at ``(target_lat_deg, target_lon_deg, target_h_amsl_m)``.

    Parameters
    ----------
    observer
        An ``ObserverSite``-like object exposing ``lat_deg``,
        ``lon_deg``, ``alt_m`` (metres AMSL, **eye** altitude).
    target_lat_deg, target_lon_deg, target_h_amsl_m
        Target WGS-84 position; height in metres AMSL.
    k
        Terrestrial-refraction coefficient. ``None`` (default) uses
        ``DEFAULT_K`` (= 0.13, ICAO over-land; matches
        ``device.rotation_calibration.terrestrial_refraction_deg``).
        Accepts any float in ``[0, 1)``.
    ignore_radius_m
        Interior samples within this distance of the observer are
        **not** treated as blockers. Suppresses bilinear noise in the
        observer's own DEM cell and tolerates 1–2 m jetty/tripod edges
        SRTM can't resolve. Default 30 m.
    dem_source
        DEM provider key — only ``"srtm1"`` in MVP.
    dem_provider
        Optional ``(lat, lon, source) -> DatasetReader`` callable.
        Defaults to ``_default_dem_provider`` (on-disk tile cache +
        OpenTopography fetch). Tests should always pass a provider
        that reads an in-repo synthetic fixture.
    min_samples
        Floor on sample count along the path; default 50.

    Returns
    -------
    LosResult
        ``visible`` is ``True`` iff every interior sample has
        ``h_los >= DEM(sample)``. ``used_refraction_k`` echoes the
        applied coefficient so the caller can log it.
    """
    if k is None:
        k = DEFAULT_K
    if dem_provider is None:
        dem_provider = _default_dem_provider

    obs_lat = float(observer.lat_deg)
    obs_lon = float(observer.lon_deg)
    obs_h = float(observer.alt_m)
    warnings: list[str] = []

    with dem_provider(obs_lat, obs_lon, dem_source) as dataset:
        dem_res_m = _dem_resolution_m(dataset, obs_lat)
        geod = Geod(ellps="WGS84")
        _fwd, _back, approx_len = geod.inv(
            obs_lon, obs_lat, target_lon_deg, target_lat_deg,
        )
        approx_len = float(approx_len)
        if approx_len <= 0.0:
            return LosResult(
                visible=True,
                min_clearance_m=float("inf"),
                obstacle_distance_m=None,
                obstacle_altitude_m=None,
                used_refraction_k=float(k),
                sample_count=2,
                warnings=("observer and target coincide",),
            )
        n_interior = max(min_samples, int(approx_len / dem_res_m))
        lats, lons, dists, path_len = _great_circle_samples(
            obs_lat, obs_lon, target_lat_deg, target_lon_deg, n_interior,
            geod=geod,
        )
        dem_h = _bilinear_sample_dem(dataset, lats, lons)

        r_eff = _effective_earth_radius(k)
        frac = dists / path_len
        h_los = (
            obs_h + frac * (target_h_amsl_m - obs_h)
            - (dists * (path_len - dists)) / (2.0 * r_eff)
        )
        clearance = h_los - dem_h

        # Interior = exclude both endpoints (observer + target) and the
        # close-in ignore belt around the observer.
        mask = np.ones_like(dists, dtype=bool)
        mask[0] = False
        mask[-1] = False
        if ignore_radius_m > 0.0:
            mask &= dists > float(ignore_radius_m)

        # Observer-below-DEM diagnostic (the degenerate case §D.2 of
        # the design doc). Not an error — we still compute LoS; just
        # surface a warning so callers can display it.
        obs_dem = float(dem_h[0])
        if obs_h < obs_dem - 0.5:
            warnings.append(
                f"observer altitude {obs_h:.1f} m is below DEM "
                f"{obs_dem:.1f} m at observer lat/lon (diff "
                f"{obs_dem - obs_h:.1f} m); consider DEM lookup as "
                "the altitude source"
            )

        if not mask.any():
            # All samples were inside the ignore radius — path too
            # short to evaluate meaningfully. Treat as visible.
            return LosResult(
                visible=True,
                min_clearance_m=float("inf"),
                obstacle_distance_m=None,
                obstacle_altitude_m=None,
                used_refraction_k=float(k),
                sample_count=int(len(lats)),
                warnings=tuple(warnings),
            )

        interior_clearance = clearance[mask]
        interior_dists = dists[mask]
        interior_dem = dem_h[mask]
        min_idx = int(np.argmin(interior_clearance))
        min_clearance = float(interior_clearance[min_idx])
        visible = min_clearance >= 0.0

        obstacle_distance = float(interior_dists[min_idx]) if not visible else None
        obstacle_altitude = float(interior_dem[min_idx]) if not visible else None

        return LosResult(
            visible=visible,
            min_clearance_m=min_clearance,
            obstacle_distance_m=obstacle_distance,
            obstacle_altitude_m=obstacle_altitude,
            used_refraction_k=float(k),
            sample_count=int(len(lats)),
            warnings=tuple(warnings),
        )
