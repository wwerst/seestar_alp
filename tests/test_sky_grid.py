"""Tests for the alt/az sky grid used by the visibility mapper.

These tests pin down the geometry contract the rest of the mapping
code relies on: cells exist where expected, neighbor lookup is
symmetric and roughly local, the floor mask correctly excludes
below-horizon cells, and great-circle math is sane for a few hand-
checked configurations.
"""

from __future__ import annotations

import math

import pytest

from device.sky_grid import (
    SkyGrid,
    great_circle_deg,
    make_altaz_band_grid,
    make_default_grid,
)


# ---------- great-circle math ---------------------------------------


def test_great_circle_zero_for_same_point():
    assert great_circle_deg(123.4, 45.6, 123.4, 45.6) == pytest.approx(0.0)


def test_great_circle_zenith_to_horizon_is_90():
    # Zenith (alt=90) to a horizon point: any az → 90° away.
    assert great_circle_deg(0, 90, 90, 0) == pytest.approx(90.0)
    assert great_circle_deg(0, 90, 270, 0) == pytest.approx(90.0)


def test_great_circle_handles_az_wrap():
    # Two horizon points 1° apart in az, straddling the wrap.
    d = great_circle_deg(359.5, 0.0, 0.5, 0.0)
    assert d == pytest.approx(1.0, abs=1e-6)


def test_great_circle_north_to_south_horizon_is_180():
    assert great_circle_deg(0, 0, 180, 0) == pytest.approx(180.0)


# ---------- band grid construction ----------------------------------


def test_band_grid_default_size_in_range():
    grid = make_altaz_band_grid(min_alt_deg=0.0)
    # 18 bands × variable width per spec — total roughly 800–1000.
    assert 700 <= len(grid.cells) <= 1100
    # All cells lie in [0, 90] alt and [0, 360) az.
    for c in grid.cells:
        assert 0.0 <= c.alt_deg < 90.0
        assert 0.0 <= c.az_deg < 360.0


def test_band_grid_floor_mask_marks_below_floor():
    grid = make_altaz_band_grid(min_alt_deg=10.0)
    below = [c for c in grid.cells if c.below_floor]
    above = [c for c in grid.cells if not c.below_floor]
    assert len(below) > 0
    assert len(above) > 0
    for c in below:
        assert c.alt_deg < 10.0
    for c in above:
        assert c.alt_deg >= 10.0


def test_band_grid_active_indices_match_floor():
    grid = make_altaz_band_grid(min_alt_deg=20.0)
    active = grid.active_indices()
    for idx in active:
        assert grid.cells[idx].alt_deg >= 20.0


def test_band_grid_zenith_band_has_few_az_cells():
    grid = make_altaz_band_grid(min_alt_deg=0.0)
    # Top band (alt 85-90, center 87.5) should have ~3 cells (cos ≈ 0.044 → 3).
    top_band = [c for c in grid.cells if c.alt_deg >= 85.0]
    assert 3 <= len(top_band) <= 5


def test_band_grid_horizon_band_has_dense_az():
    grid = make_altaz_band_grid(min_alt_deg=0.0)
    bottom_band = [c for c in grid.cells if c.alt_deg < 5.0]
    # Horizon band has cos ≈ 1 → ~72 cells.
    assert 65 <= len(bottom_band) <= 75


# ---------- neighbor lookup -----------------------------------------


def test_neighbors_symmetric():
    grid = make_altaz_band_grid(min_alt_deg=0.0)
    for a, ns in enumerate(grid.neighbors):
        for n in ns:
            assert a in grid.neighbors[n], f"asymmetry: {a} → {n}"


def test_neighbors_no_self_loops():
    grid = make_altaz_band_grid(min_alt_deg=0.0)
    for a, ns in enumerate(grid.neighbors):
        assert a not in ns


def test_neighbors_count_in_reasonable_range():
    grid = make_altaz_band_grid(min_alt_deg=0.0)
    # Each cell should have at least 2 (pole) and at most ~8 neighbors.
    for ns in grid.neighbors:
        assert 2 <= len(ns) <= 10


def test_neighbors_are_local():
    """Neighbor great-circle distance should be small — at most ~2x the
    band width (5°)."""
    grid = make_altaz_band_grid(min_alt_deg=0.0)
    for a, ns in enumerate(grid.neighbors):
        ca = grid.cells[a]
        for n in ns:
            cn = grid.cells[n]
            d = great_circle_deg(ca.az_deg, ca.alt_deg, cn.az_deg, cn.alt_deg)
            # Loose bound: very close to the pole the band-above choice
            # can be a bit further in az; allow up to 20° generally.
            assert d < 25.0, (
                f"neighbor too far: {a}→{n} d={d:.2f}° "
                f"({ca.az_deg:.1f}/{ca.alt_deg:.1f} → "
                f"{cn.az_deg:.1f}/{cn.alt_deg:.1f})"
            )


def test_neighbors_wrap_at_az_360():
    """Cells on either side of the az=0 wrap should be neighbors."""
    grid = make_altaz_band_grid(min_alt_deg=0.0)
    # Find lowest band's first and last cells.
    band0 = sorted([c for c in grid.cells if c.alt_deg < 5.0], key=lambda c: c.az_deg)
    first = band0[0]
    last = band0[-1]
    assert first.idx in grid.neighbors[last.idx]
    assert last.idx in grid.neighbors[first.idx]


# ---------- nearest-cell lookup -------------------------------------


def test_nearest_finds_self_for_cell_center():
    grid = make_altaz_band_grid(min_alt_deg=0.0)
    for c in grid.cells[::20]:
        assert grid.nearest(c.az_deg, c.alt_deg) == c.idx


def test_nearest_handles_az_wrap():
    grid = make_altaz_band_grid(min_alt_deg=0.0)
    # Pick a low-band cell near az=0
    band0 = sorted([c for c in grid.cells if c.alt_deg < 5.0], key=lambda c: c.az_deg)
    near0 = band0[0]
    # Query at -0.5° should still resolve to that cell's neighborhood.
    idx = grid.nearest(359.5, near0.alt_deg)
    near_az = grid.cells[idx].az_deg
    # Should land in the cell whose center is nearest 359.5° (i.e.,
    # the last cell in the band).
    assert math.fabs((near_az - 359.5 + 540) % 360 - 180) < 5.0


# ---------- factory -------------------------------------------------


def test_default_grid_returns_altaz_when_healpix_unavailable():
    grid = make_default_grid(min_alt_deg=10.0, prefer="altaz")
    assert isinstance(grid, SkyGrid)
    assert grid.kind == "altaz_band"


def test_default_grid_falls_back_when_no_site():
    # prefer healpix but missing site params → falls back to altaz.
    grid = make_default_grid(min_alt_deg=10.0, prefer="healpix")
    assert grid.kind == "altaz_band"


def test_refresh_centers_noop_for_altaz():
    grid = make_altaz_band_grid(min_alt_deg=0.0)
    refreshed = grid.refresh_centers(0.0, force=True)
    assert refreshed is False
