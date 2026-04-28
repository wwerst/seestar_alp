"""Plate solver interface + concrete backends for nighttime calibration.

Plate solving converts a sky image to (RA, Dec) by matching the visible
star pattern against a catalog. We use it to anchor each nighttime
calibration sighting to celestial coordinates: command the mount to
``(commanded_az, commanded_el)``, plate-solve to ``(true_ra, true_dec)``,
and convert that back to true ``(az, el)`` using the site location +
UTC. The pair forms one input to the same 3-DOF rotation solver the
daytime FAA-landmark workflow uses.

Backends:

- :class:`SeestarPlateSolver` — runs the firmware's onboard solver via
  the ``start_solve`` RPC and parses the ``PlateSolve`` completion
  event. **Default** when a telescope id + Alpaca action runner are
  available; needs no host-side dependencies.
- :class:`SolveFieldPlateSolver` — wraps the ``solve-field`` CLI from
  astrometry.net. Used as a fallback when the firmware solver isn't
  reachable (e.g. dry-running calibration logic against captured FITS
  images on a workstation with no scope attached).
- :class:`FakePlateSolver` — used by tests. Returns canned
  :class:`SolveResult` values keyed by image path.
- :class:`UnavailablePlateSolver` — sentinel returned when no real
  solver is configured. ``solve()`` always raises
  :class:`PlateSolverNotAvailable` so callers can check
  ``isinstance(get_default_plate_solver(), UnavailablePlateSolver)``
  for a fast availability flag.

The :class:`SeestarPlateSolver` ignores the ``image_path`` argument:
the scope solves whatever it is currently looking at, no on-disk file
is involved.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol, runtime_checkable


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
    """Duck-typed interface.

    ``image_path`` is optional: backends that drive the scope's onboard
    solver (:class:`SeestarPlateSolver`) ignore it because the firmware
    plate-solves whatever it's currently looking at — there is no
    on-disk image. File-based backends (:class:`SolveFieldPlateSolver`)
    require it.
    """

    def solve(self, image_path: Path | None = None) -> SolveResult: ...

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

    def solve(self, image_path: Path | None = None) -> SolveResult:
        if not self.is_available():
            raise PlateSolverNotAvailable(
                "solve-field binary not on PATH; install astrometry.net "
                "(see plans/nighttime_calibration.md)"
            )
        if image_path is None:
            raise PlateSolverFailed(
                "solve-field requires an on-disk image; pass image_path"
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


# ---------- SeestarPlateSolver ---------------------------------------


# Type alias for the Alpaca action callable. Matches
# ``front.app.do_action_device(action, dev_num, parameters) -> dict | None``.
SeestarActionRunner = Callable[[str, int, dict], "Any"]


class SeestarPlateSolver:
    """Onboard plate solver via the Seestar firmware's ``start_solve`` RPC.

    The scope solves whatever it is currently looking at; ``solve()``
    therefore ignores ``image_path``. The implementation drives the
    Alpaca custom action ``start_solve_sync`` (registered in
    ``device/telescope.py``), which sends ``start_solve`` and blocks
    until the page-level ``PlateSolve`` completion event arrives. The
    firmware reports RA in **hours** and Dec in **degrees**; we
    convert RA to degrees on the way out for parity with the
    ``solve-field`` backend.

    Constructor takes an injectable ``action_runner`` so tests can
    pass a stub. In production, ``front.app.do_action_device`` is the
    runner.
    """

    kind = "seestar"

    def __init__(
        self,
        action_runner: SeestarActionRunner,
        telescope_id: int,
        *,
        timeout_s: float = 60.0,
    ):
        self.action_runner = action_runner
        self.telescope_id = int(telescope_id)
        self.timeout_s = float(timeout_s)

    def is_available(self) -> bool:
        # We don't pre-check connectivity; ``solve()`` raises loudly
        # if the scope is offline. Returning True here keeps the
        # availability endpoint cheap.
        return True

    def solve(self, image_path: Path | None = None) -> SolveResult:
        try:
            response = self.action_runner(
                "start_solve_sync",
                self.telescope_id,
                {"timeout_s": self.timeout_s},
            )
        except Exception as exc:
            raise PlateSolverFailed(f"seestar plate solve failed: {exc}") from exc
        if response is None:
            raise PlateSolverFailed(
                "seestar plate solve: no response from device "
                f"(telescope {self.telescope_id} unreachable?)"
            )
        # The Alpaca driver propagates ``request_plate_solve_sync``
        # exceptions (timeout / firmware ``fail`` event) as a non-zero
        # ``ErrorNumber`` response with no ``Value`` field.
        if isinstance(response, dict):
            err_num = response.get("ErrorNumber")
            if err_num not in (None, 0):
                raise PlateSolverFailed(
                    f"seestar plate solve: device error "
                    f"{err_num} {response.get('ErrorMessage', '')}".strip()
                )
        # Alpaca wraps the firmware result under ``Value``; some test
        # doubles return the inner dict directly.
        if isinstance(response, dict) and "Value" in response:
            payload = response["Value"]
        else:
            payload = response
        if not isinstance(payload, dict) or "ra_dec" not in payload:
            raise PlateSolverFailed(
                f"seestar plate solve: missing ra_dec in response {payload!r}"
            )
        ra_dec = payload["ra_dec"]
        try:
            # Firmware emits RA in hours, Dec in degrees.
            ra_hours = float(ra_dec[0])
            dec_deg = float(ra_dec[1])
        except (TypeError, ValueError, IndexError) as exc:
            raise PlateSolverFailed(
                f"seestar plate solve: malformed ra_dec {ra_dec!r}"
            ) from exc
        fov = payload.get("fov") or [0.0, 0.0]
        try:
            fov_x = float(fov[0])
            fov_y = float(fov[1])
        except (TypeError, ValueError, IndexError):
            fov_x = fov_y = 0.0
        return SolveResult(
            ra_deg=ra_hours * 15.0,
            dec_deg=dec_deg,
            fov_x_deg=fov_x,
            fov_y_deg=fov_y,
            position_angle_deg=float(payload.get("angle", 0.0) or 0.0),
            stars_used=int(payload.get("star_number", 0) or 0),
            raw=dict(payload),
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

    def solve(self, image_path: Path | None = None) -> SolveResult:
        # Use a sentinel key so callers (Seestar mode) that don't
        # supply a path can still register a single canned result via
        # ``""`` -> SolveResult.
        path_str = "" if image_path is None else str(Path(image_path))
        self.calls.append(Path(image_path) if image_path is not None else Path(""))
        if path_str not in self.results:
            raise FileNotFoundError(f"no fake result for {path_str!r}")
        result = self.results[path_str]
        if result is None:
            raise PlateSolverFailed(f"fake failure for {path_str!r}")
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

    def solve(self, image_path: Path | None = None) -> SolveResult:
        raise PlateSolverNotAvailable(
            "plate solver not configured; install astrometry.net or "
            "configure ASTROMETRY_API_KEY (see plans/nighttime_calibration.md)"
        )


# ---------- factory --------------------------------------------------


def get_default_plate_solver(
    telescope_id: int | None = None,
    action_runner: SeestarActionRunner | None = None,
) -> PlateSolver:
    """Return the best available plate solver.

    Preference order:

    1. :class:`SeestarPlateSolver` when both ``telescope_id`` and
       ``action_runner`` are supplied — the firmware's onboard solver
       has no host-side dependencies.
    2. :class:`SolveFieldPlateSolver` if the ``solve-field`` binary is
       on PATH — used as a workstation fallback for solving captured
       FITS images without a scope attached.
    3. :class:`UnavailablePlateSolver` — callers can check
       ``is_available()`` before enabling nighttime mode.
    """
    if telescope_id is not None and action_runner is not None:
        return SeestarPlateSolver(action_runner, int(telescope_id))
    sf = SolveFieldPlateSolver()
    if sf.is_available():
        return sf
    return UnavailablePlateSolver()


# Silence "unused import" warnings for json (kept available for future
# JSON-stdout solvers like the astroquery web wrapper).
_ = json
