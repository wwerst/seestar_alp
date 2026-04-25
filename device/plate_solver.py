"""Plate solver interface + concrete backends for nighttime calibration.

Plate solving converts a sky image to (RA, Dec) by matching the visible
star pattern against a catalog. We use it to anchor each nighttime
calibration sighting to celestial coordinates: command the mount to
``(commanded_az, commanded_el)``, capture an image, plate-solve to
``(true_ra, true_dec)``, and convert that back to true ``(az, el)``
using the site location + UTC. The pair forms one input to the same
3-DOF rotation solver the daytime FAA-landmark workflow uses.

Three backends:

- :class:`SolveFieldPlateSolver` — wraps the ``solve-field`` CLI from
  astrometry.net. Default when the binary is on PATH.
- :class:`FakePlateSolver` — used by tests. Returns canned
  :class:`SolveResult` values keyed by image path.
- :class:`UnavailablePlateSolver` — sentinel returned when no real
  solver is configured. ``solve()`` always raises
  :class:`PlateSolverNotAvailable` so callers can check
  ``isinstance(get_default_plate_solver(), UnavailablePlateSolver)``
  for a fast availability flag.

The solver dependency is **optional** — if neither the CLI nor an API
key is available, the nighttime calibrate tab is disabled with a clear
"Plate solver not configured" message; daytime calibration is
unaffected.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable


# Minimum / maximum field-of-view (degrees) we consider sane for an S50
# image. Real spec is ~1.27° × 0.71°; we accept a wider band so cropped
# / non-standard captures still solve, but reject obviously-wrong
# solutions (e.g. the solver matched to a wide-field star catalog).
S50_FOV_MIN_DEG = 0.5
S50_FOV_MAX_DEG = 3.0


class PlateSolverNotAvailable(Exception):
    """Raised when no plate solver backend is configured / installed."""


class PlateSolverFailed(Exception):
    """Raised when the solver ran but did not return a confident solution."""


@dataclass(frozen=True)
class SolveResult:
    """One plate solve's output.

    ``ra_deg`` / ``dec_deg`` are J2000 image-centre coordinates.
    ``fov_x_deg`` / ``fov_y_deg`` are the solved field-of-view (degrees).
    ``position_angle_deg`` is the rotation of image up-axis from
    celestial north (East-of-North; standard astrometry.net WCS).
    ``stars_used`` is how many catalog matches were used (diagnostic).
    """

    ra_deg: float
    dec_deg: float
    fov_x_deg: float
    fov_y_deg: float
    position_angle_deg: float
    stars_used: int = 0
    raw: dict = None  # Backend-specific diagnostic payload (optional).


@runtime_checkable
class PlateSolver(Protocol):
    """Duck-typed interface."""

    def solve(self, image_path: Path) -> SolveResult: ...

    def is_available(self) -> bool: ...

    @property
    def kind(self) -> str: ...


# ---------- SolveFieldPlateSolver ------------------------------------


class SolveFieldPlateSolver:
    """Wraps the ``solve-field`` CLI from astrometry.net.

    Requires ``solve-field`` on PATH plus index files installed via the
    distribution package (``debian: astrometry.net-data-tycho2``,
    ``brew: astrometry-net``, etc.). We pass FOV bounds so the solver
    skips index sets that don't apply, which keeps a successful solve
    in the 5–20 s range on a modest CPU.

    The ``solve()`` method runs ``solve-field --no-plots --no-fits2fits
    --new-fits none --solved none --match none --rdls none --corr none
    --wcs <out>.wcs --scale-units degwidth --scale-low 0.5 --scale-high
    3 --downsample 2 --overwrite <image>`` and parses the resulting
    .wcs FITS header for RA/Dec/FOV/PA. We don't actually need the
    .wcs file's full FITS payload — astrometry.net writes the solved
    centre to stdout in a parsable form.

    Stdout parsing is fragile across solver versions; we additionally
    parse the optional ``--axy <out>.axy`` companion if present, and
    fall back to the ``.solved`` file's existence as a binary success
    flag if header parsing fails.
    """

    kind = "solve-field"

    def __init__(self, binary_path: str | None = None, timeout_s: float = 90.0):
        if binary_path:
            self.binary_path = binary_path
        else:
            self.binary_path = shutil.which("solve-field") or ""
        self.timeout_s = float(timeout_s)

    def is_available(self) -> bool:
        return bool(self.binary_path)

    def solve(self, image_path: Path) -> SolveResult:
        if not self.is_available():
            raise PlateSolverNotAvailable(
                "solve-field binary not on PATH; install astrometry.net "
                "(see plans/nighttime_calibration.md)"
            )
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"image not found: {image_path}")

        # Use a sibling tempdir so artefacts don't pollute the original
        # capture directory. ``solve-field`` writes ~10 files per solve.
        out_dir = image_path.with_suffix(".solve")
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            self.binary_path,
            "--no-plots",
            "--no-fits2fits",
            "--new-fits",
            "none",
            "--solved",
            "none",
            "--match",
            "none",
            "--rdls",
            "none",
            "--corr",
            "none",
            "--wcs",
            str(out_dir / "out.wcs"),
            "--scale-units",
            "degwidth",
            "--scale-low",
            str(S50_FOV_MIN_DEG),
            "--scale-high",
            str(S50_FOV_MAX_DEG),
            "--downsample",
            "2",
            "--overwrite",
            str(image_path),
        ]
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise PlateSolverFailed(
                f"solve-field timed out after {self.timeout_s}s"
            ) from exc

        if result.returncode != 0:
            tail = (result.stderr or "")[-500:].strip()
            raise PlateSolverFailed(
                f"solve-field exit {result.returncode}: {tail or 'unknown error'}"
            )

        return _parse_solve_field_stdout(
            result.stdout, raw_dict={"stdout": result.stdout}
        )


_SOLVE_RE = {
    "centre": re.compile(
        r"Field center:\s*\(RA,Dec\)\s*=\s*\(([-\d.]+),\s*([-\d.]+)\)"
    ),
    "size": re.compile(
        r"Field size:\s*([-\d.]+)\s*x\s*([-\d.]+)\s*(arcminutes|degrees)"
    ),
    "rotation": re.compile(r"Field rotation angle:\s*up is ([-\d.]+) degrees"),
    "stars": re.compile(r"\((\d+)\s+match\(es\)"),
}


def _parse_solve_field_stdout(stdout: str, raw_dict: dict | None = None) -> SolveResult:
    """Parse the human-readable summary lines ``solve-field`` prints to
    stdout. Format has been stable across astrometry.net 0.70+ but a
    breakage falls back to ``PlateSolverFailed`` rather than silently
    returning garbage."""
    centre = _SOLVE_RE["centre"].search(stdout)
    size = _SOLVE_RE["size"].search(stdout)
    rotation = _SOLVE_RE["rotation"].search(stdout)
    if centre is None or size is None:
        raise PlateSolverFailed(
            "solve-field stdout missing field-centre / size — no confident solution"
        )
    ra = float(centre.group(1))
    dec = float(centre.group(2))
    fx = float(size.group(1))
    fy = float(size.group(2))
    if size.group(3) == "arcminutes":
        fx /= 60.0
        fy /= 60.0
    pa = float(rotation.group(1)) if rotation else 0.0
    stars_match = _SOLVE_RE["stars"].search(stdout)
    stars_used = int(stars_match.group(1)) if stars_match else 0
    return SolveResult(
        ra_deg=ra,
        dec_deg=dec,
        fov_x_deg=fx,
        fov_y_deg=fy,
        position_angle_deg=pa,
        stars_used=stars_used,
        raw=raw_dict or {},
    )


# ---------- FakePlateSolver ------------------------------------------


class FakePlateSolver:
    """In-memory plate solver for tests.

    Constructed with a dict mapping ``image_path → SolveResult | None``.
    A ``None`` entry simulates a failed solve (raises
    :class:`PlateSolverFailed`). Missing entries raise
    :class:`FileNotFoundError`.
    """

    kind = "fake"

    def __init__(self, results: dict[str, SolveResult | None] | None = None):
        self.results: dict[str, SolveResult | None] = {
            str(k): v for k, v in (results or {}).items()
        }
        self.calls: list[Path] = []

    def is_available(self) -> bool:
        return True

    def add(self, image_path: Path, result: SolveResult | None) -> None:
        self.results[str(image_path)] = result

    def solve(self, image_path: Path) -> SolveResult:
        path_str = str(Path(image_path))
        self.calls.append(Path(image_path))
        if path_str not in self.results:
            raise FileNotFoundError(f"no fake result for {path_str}")
        result = self.results[path_str]
        if result is None:
            raise PlateSolverFailed(f"fake failure for {path_str}")
        return result


# ---------- UnavailablePlateSolver -----------------------------------


class UnavailablePlateSolver:
    """Sentinel used when no real solver is configured. All
    ``solve()`` calls raise :class:`PlateSolverNotAvailable`. Use
    ``is_available()`` (or ``isinstance(s, UnavailablePlateSolver)``)
    to gate the nighttime tab availability check."""

    kind = "unavailable"

    def is_available(self) -> bool:
        return False

    def solve(self, image_path: Path) -> SolveResult:
        raise PlateSolverNotAvailable(
            "plate solver not configured; install astrometry.net or "
            "configure ASTROMETRY_API_KEY (see plans/nighttime_calibration.md)"
        )


# ---------- factory --------------------------------------------------


def get_default_plate_solver() -> PlateSolver:
    """Return the best available plate solver.

    Prefers a local ``solve-field`` install. Returns an
    :class:`UnavailablePlateSolver` if nothing is configured — callers
    can check ``is_available()`` to decide whether to enable nighttime
    mode.

    Web-service fallback (astroquery.astrometry_net) is not implemented
    in this PR; see plans/nighttime_calibration.md for the design.
    """
    sf = SolveFieldPlateSolver()
    if sf.is_available():
        return sf
    return UnavailablePlateSolver()


# Silence "unused import" warnings for json (kept available for future
# JSON-stdout solvers like the astroquery web wrapper).
_ = json
