"""Sky grid for the visibility-mapping feature.

Provides a uniform tiling of the sky in alt/az coordinates that the
visibility mapper samples. Two implementations:

* :func:`make_altaz_band_grid` — the default. Equal-area-ish tiling of
  the dome above ``min_alt``: 5° altitude bands, with each band's
  azimuth resolution scaled by ``cos(alt_center)`` so cells are
  approximately equal area. Cells are stable in alt/az (no per-step
  reprojection from RA/Dec) which keeps the acquisition code simple.
* :func:`make_healpix_grid` — optional, used if ``astropy_healpix`` is
  importable. HEALPix is uniform over the sphere; we project cells to
  alt/az at the time the grid is built and again every five minutes via
  :meth:`SkyGrid.refresh_centers` if the caller passes time + location.

Both share the :class:`SkyGrid` interface so the mapper does not need
to care which is in use.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional


# ---------- data model ------------------------------------------------


@dataclass
class SkyCell:
    """One alt/az cell on the dome."""

    idx: int
    az_deg: float
    alt_deg: float
    # Equatorial coordinates (HEALPix grids only — None for alt/az grids).
    # Stored so the grid can re-project to alt/az as the sky rotates.
    ra_deg: Optional[float] = None
    dec_deg: Optional[float] = None
    # Per-cell flag: cells whose center falls below the visibility floor
    # are excluded from sampling but kept in the grid so that re-running
    # with a different floor doesn't renumber the array.
    below_floor: bool = False


@dataclass
class SkyGrid:
    """A tiling of the sky above the user's altitude floor.

    Centers are in degrees. Azimuth wraps at 360° (north = 0°, east = 90°).
    Altitude is in [0, 90].
    """

    cells: list[SkyCell]
    neighbors: list[list[int]]
    min_alt_deg: float
    kind: str  # "altaz_band" or "healpix"
    # HEALPix-only: stash the params so refresh_centers can recompute.
    nside: Optional[int] = None
    site_lat_deg: Optional[float] = None
    site_lon_deg: Optional[float] = None
    site_alt_m: Optional[float] = None
    last_refresh_t: float = field(default=0.0)

    def __len__(self) -> int:
        return len(self.cells)

    def active_indices(self) -> list[int]:
        """Indices of cells that are above the alt floor."""
        return [c.idx for c in self.cells if not c.below_floor]

    def nearest(self, az_deg: float, alt_deg: float) -> int:
        """Index of the cell whose center is nearest the given (az, alt).

        Uses great-circle distance on the unit sphere — robust to
        azimuth wrap. Linear search; the grid is small (<1000 cells)
        so a KD-tree would be overkill.
        """
        a = float(az_deg) % 360.0
        e = float(alt_deg)
        best_idx = -1
        best_d = math.inf
        for c in self.cells:
            d = great_circle_deg(a, e, c.az_deg, c.alt_deg)
            if d < best_d:
                best_d = d
                best_idx = c.idx
        return best_idx

    def refresh_centers(self, t_unix: float, *, force: bool = False) -> bool:
        """For HEALPix grids, recompute alt/az centers from RA/Dec.

        Returns ``True`` if a refresh actually ran. No-op for alt/az
        band grids (centers are time-invariant). Default refresh
        interval is 5 minutes; pass ``force=True`` to override.
        """
        if self.kind != "healpix":
            return False
        if (
            not force
            and self.last_refresh_t > 0.0
            and abs(t_unix - self.last_refresh_t) < 300.0
        ):
            return False
        if (
            self.site_lat_deg is None
            or self.site_lon_deg is None
            or self.site_alt_m is None
        ):
            return False
        try:
            from astropy import units as u
            from astropy.coordinates import AltAz, EarthLocation, SkyCoord
            from astropy.time import Time
        except Exception:
            return False

        loc = EarthLocation.from_geodetic(
            lon=self.site_lon_deg * u.deg,
            lat=self.site_lat_deg * u.deg,
            height=self.site_alt_m * u.m,
        )
        ras = [c.ra_deg for c in self.cells]
        decs = [c.dec_deg for c in self.cells]
        sky = SkyCoord(ra=ras * u.deg, dec=decs * u.deg, frame="icrs")
        altaz = sky.transform_to(
            AltAz(obstime=Time(t_unix, format="unix"), location=loc)
        )
        for i, c in enumerate(self.cells):
            c.az_deg = float(altaz.az.deg[i]) % 360.0
            c.alt_deg = float(altaz.alt.deg[i])
            c.below_floor = c.alt_deg < self.min_alt_deg
        self.last_refresh_t = float(t_unix)
        return True


# ---------- math helpers ---------------------------------------------


def great_circle_deg(az1: float, alt1: float, az2: float, alt2: float) -> float:
    """Great-circle separation on the celestial sphere in degrees.

    Inputs in degrees; ``az`` wraps at 360. Spherical-law-of-cosines
    form clamped against the rounding-error case where ``cos(d) > 1``.
    """
    a1 = math.radians(az1)
    a2 = math.radians(az2)
    e1 = math.radians(alt1)
    e2 = math.radians(alt2)
    cos_d = math.sin(e1) * math.sin(e2) + math.cos(e1) * math.cos(e2) * math.cos(
        a1 - a2
    )
    cos_d = max(-1.0, min(1.0, cos_d))
    return math.degrees(math.acos(cos_d))


# ---------- alt/az band grid -----------------------------------------


def make_altaz_band_grid(
    min_alt_deg: float = 0.0,
    band_width_deg: float = 5.0,
    az_density_at_zenith: int = 72,
) -> SkyGrid:
    """Build an equal-area-ish alt/az band grid.

    Bands of ``band_width_deg`` from 0° up to (but not including) 90°.
    Within each band the azimuth is split into
    ``round(az_density_at_zenith * cos(center_alt))`` cells, falling to
    a minimum of 3 near the pole so the grid stays connected.

    Each cell remembers its 4 (or near-4) neighbors: same-band ±1 az
    (with wrap), nearest-az cell in the band above, nearest-az cell in
    the band below.
    """
    if band_width_deg <= 0:
        raise ValueError("band_width_deg must be > 0")
    if az_density_at_zenith < 4:
        raise ValueError("az_density_at_zenith must be >= 4")
    n_bands = int(math.floor(90.0 / band_width_deg))
    cells: list[SkyCell] = []
    band_start_idx: list[int] = []
    band_count: list[int] = []

    for b in range(n_bands):
        alt_lo = b * band_width_deg
        alt_hi = min(90.0, (b + 1) * band_width_deg)
        center_alt = (alt_lo + alt_hi) / 2.0
        cos_e = math.cos(math.radians(center_alt))
        n_az = max(3, int(round(az_density_at_zenith * cos_e)))
        band_start_idx.append(len(cells))
        band_count.append(n_az)
        for j in range(n_az):
            az_c = (j + 0.5) * 360.0 / n_az
            below = center_alt < min_alt_deg
            cells.append(
                SkyCell(
                    idx=len(cells),
                    az_deg=az_c,
                    alt_deg=center_alt,
                    below_floor=below,
                )
            )

    # Neighbor lookup. Build per-cell list of up to 4 neighbors:
    # same band ±1 az (wrap), nearest-az in band above, nearest-az in
    # band below.
    neighbors: list[list[int]] = [[] for _ in cells]
    for b in range(n_bands):
        n_az = band_count[b]
        start = band_start_idx[b]
        for j in range(n_az):
            cur = start + j
            # Same-band wrap neighbors
            left = start + (j - 1) % n_az
            right = start + (j + 1) % n_az
            if left != cur:
                neighbors[cur].append(left)
            if right != cur and right != left:
                neighbors[cur].append(right)
            # Band above
            if b + 1 < n_bands:
                above_start = band_start_idx[b + 1]
                above_count = band_count[b + 1]
                target_az = cells[cur].az_deg
                # Nearest center in the upper band by azimuth (with wrap).
                k = int(round(target_az * above_count / 360.0)) % above_count
                neighbors[cur].append(above_start + k)
            # Band below
            if b - 1 >= 0:
                below_start = band_start_idx[b - 1]
                below_count = band_count[b - 1]
                target_az = cells[cur].az_deg
                k = int(round(target_az * below_count / 360.0)) % below_count
                neighbors[cur].append(below_start + k)
            # Dedupe and drop self.
            seen = []
            for n in neighbors[cur]:
                if n != cur and n not in seen:
                    seen.append(n)
            neighbors[cur] = seen

    # Make neighbors symmetric: if A lists B, B should list A.
    for a, ns in enumerate(neighbors):
        for n in ns:
            if a not in neighbors[n]:
                neighbors[n].append(a)

    return SkyGrid(
        cells=cells,
        neighbors=neighbors,
        min_alt_deg=float(min_alt_deg),
        kind="altaz_band",
    )


# ---------- HEALPix grid (optional) ----------------------------------


def make_healpix_grid(
    nside: int,
    *,
    min_alt_deg: float = 0.0,
    site_lat_deg: float,
    site_lon_deg: float,
    site_alt_m: float,
    t_unix: float,
) -> SkyGrid:
    """Build a HEALPix grid projected to alt/az at ``t_unix``.

    Requires ``astropy_healpix`` plus ``astropy``. Uses ring ordering
    so neighbor lookup via ``HEALPix.neighbours`` returns the 8
    neighbors of each cell (some may be -1 at the poles, which we
    drop).
    """
    from astropy import units as u
    from astropy.coordinates import AltAz, EarthLocation, SkyCoord
    from astropy.time import Time
    from astropy_healpix import HEALPix

    hp = HEALPix(nside=int(nside), order="ring")
    npix = hp.npix
    ras_q, decs_q = hp.healpix_to_lonlat(list(range(npix)))
    ras = ras_q.to(u.deg).value
    decs = decs_q.to(u.deg).value

    loc = EarthLocation.from_geodetic(
        lon=site_lon_deg * u.deg,
        lat=site_lat_deg * u.deg,
        height=site_alt_m * u.m,
    )
    sky = SkyCoord(ra=ras * u.deg, dec=decs * u.deg, frame="icrs")
    altaz = sky.transform_to(AltAz(obstime=Time(t_unix, format="unix"), location=loc))
    az_arr = altaz.az.deg
    alt_arr = altaz.alt.deg

    # Filter to cells with alt >= 0 (above horizon) for compactness.
    keep_mask = alt_arr >= 0.0
    keep_idx = [i for i in range(npix) if keep_mask[i]]
    old_to_new = {old: new for new, old in enumerate(keep_idx)}

    cells: list[SkyCell] = []
    for new_i, old_i in enumerate(keep_idx):
        below = alt_arr[old_i] < min_alt_deg
        cells.append(
            SkyCell(
                idx=new_i,
                az_deg=float(az_arr[old_i]) % 360.0,
                alt_deg=float(alt_arr[old_i]),
                ra_deg=float(ras[old_i]),
                dec_deg=float(decs[old_i]),
                below_floor=bool(below),
            )
        )

    # Neighbor lookup via HEALPix.neighbours (8 neighbors per cell).
    raw_neigh = hp.neighbours(keep_idx)
    # raw_neigh shape: (8, n_cells); -1 indicates no neighbor (poles)
    neighbors: list[list[int]] = [[] for _ in cells]
    for new_i, old_i in enumerate(keep_idx):
        for k in range(8):
            n_old = int(raw_neigh[k][new_i])
            if n_old < 0:
                continue
            if n_old in old_to_new:
                neighbors[new_i].append(old_to_new[n_old])

    return SkyGrid(
        cells=cells,
        neighbors=neighbors,
        min_alt_deg=float(min_alt_deg),
        kind="healpix",
        nside=int(nside),
        site_lat_deg=float(site_lat_deg),
        site_lon_deg=float(site_lon_deg),
        site_alt_m=float(site_alt_m),
        last_refresh_t=float(t_unix),
    )


# ---------- factory --------------------------------------------------


def make_default_grid(
    *,
    min_alt_deg: float = 10.0,
    prefer: str = "altaz",
    nside: int = 32,
    site_lat_deg: Optional[float] = None,
    site_lon_deg: Optional[float] = None,
    site_alt_m: Optional[float] = None,
    t_unix: Optional[float] = None,
) -> SkyGrid:
    """Build a sky grid, preferring HEALPix if requested and available.

    The alt/az band grid is the default because it has no extra deps
    and produces stable centers in the alt/az frame, which simplifies
    the visibility mapper. HEALPix is selected when ``prefer="healpix"``
    and ``astropy_healpix`` is importable; on any failure we fall back
    silently to alt/az.
    """
    if prefer == "healpix":
        if (
            site_lat_deg is None
            or site_lon_deg is None
            or site_alt_m is None
            or t_unix is None
        ):
            return make_altaz_band_grid(min_alt_deg=min_alt_deg)
        try:
            import astropy_healpix  # noqa: F401
        except Exception:
            return make_altaz_band_grid(min_alt_deg=min_alt_deg)
        try:
            return make_healpix_grid(
                nside=nside,
                min_alt_deg=min_alt_deg,
                site_lat_deg=site_lat_deg,
                site_lon_deg=site_lon_deg,
                site_alt_m=site_alt_m,
                t_unix=t_unix,
            )
        except Exception:
            return make_altaz_band_grid(min_alt_deg=min_alt_deg)
    return make_altaz_band_grid(min_alt_deg=min_alt_deg)
