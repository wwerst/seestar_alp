"""Bayesian active sky-visibility mapper.

The mapper progressively builds a per-cell ``Beta(alpha, beta)``
posterior on the probability that the telescope can plate-solve at
each cell's center. It runs in a daemon thread and emits live updates
via a small in-process pub/sub so the UI can render the map as it
fills in.

Algorithm summary:

1. **Grid**: alt/az band tiling with ~950 cells (or HEALPix if
   available). Each cell has a Beta posterior, a per-altitude prior, a
   recent-failure timestamp, and a failure counter.
2. **Observation model**: SOLVED bumps ``alpha`` by 3 (high-confidence
   positive); NO_STARS / TIMEOUT bumps ``beta`` by 0.5 / 1.0 / 2.0
   depending on attempt index; SYSTEM_ERROR does nothing to the
   posterior but increments a consecutive-error counter.
3. **MRF smoothing**: after each observation, low-confidence cells
   (``alpha + beta < 4``) borrow strength from their neighbors via a
   weighted blend toward the neighbor mean. This is what produces a
   contiguous visible polygon instead of a speckled-per-cell map.
4. **Time decay**: every 30 min of wall-clock, all cells decay toward
   their prior at rate 0.9 (floored at the prior). Cells previously
   marked obstructed re-open as clouds drift.
5. **Acquisition**: closed-form expected-information-gain-per-second
   from the Beta-Bernoulli model, modulated by a frontier bonus that
   decays from 3.0 → 0.5 over the first 20 min of run time. The
   acquisition only scores a candidate set (top entropy + neighbors of
   visible cells), not all cells.
6. **Stopping**: user stop after 5 min, median-entropy convergence,
   max-runtime, 3 consecutive system errors, or 5 sun-safety
   refusals in 60 s.

The mapper is unaware of telescope/site/time machinery; the caller
passes ``slew_func`` and ``plate_solve_func`` callbacks. The Falcon
route handler in ``front/app.py`` constructs those by composing the
device's ``_slew_to_ra_dec``, ``request_plate_solve_sync``, and
astropy alt/az ↔ RA/Dec conversion.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from device.sky_grid import SkyGrid, great_circle_deg
from device.visibility_persistence import (
    load_snapshot,
    save_snapshot,
    visibility_map_path,
)


logger = logging.getLogger(__name__)


# ---------- constants -------------------------------------------------


SLEW_RATE_DEG_S = 3.0
SLEW_SETTLE_S = 1.5
T_OBSERVE_S = 13.0  # 5s capture + 8s solve budget
SOLVE_TIMEOUT_S = 30.0  # generous: device may queue or take a moment

DECAY_INTERVAL_S = 30 * 60.0
DECAY_FACTOR = 0.9
FAILURE_RECENT_WINDOW_S = 600.0  # 10 min hard exclusion
FRONTIER_DECAY_WINDOW_S = 20 * 60.0
FRONTIER_BONUS_INITIAL = 3.0
FRONTIER_BONUS_FINAL = 0.5
VISIBLE_THRESHOLD = 0.7  # E[p] > this for "visible" / frontier neighbor count
POLYGON_THRESHOLD = 0.6  # E[p] > this is in the visible polygon overlay
LOW_CONFIDENCE_TOTAL = 4.0  # alpha + beta < this → MRF-smoothed
MRF_BLEND_W = 0.3

CANDIDATE_TOP_K = 100
CANDIDATE_CAP = 200

# Persist throttle: rewriting ~1 KB×n_cells of JSON every observation
# adds tens of ms per loop on slow disks and dominates the loop time on
# CI runners. Save at most this often during a run; we always save once
# at start (so the file exists for the UI) and once at stop.
PERSIST_INTERVAL_S = 5.0

MIN_RUN_BEFORE_USER_STOP_S = 5 * 60.0
CONVERGENCE_ELAPSED_S = 30 * 60.0
CONVERGENCE_MEDIAN_ENTROPY_NATS = 0.08
DEFAULT_MAX_RUNTIME_MIN = 8 * 60  # 8 h
SUN_REFUSAL_WINDOW_S = 60.0
SUN_REFUSAL_LIMIT = 5
SYSTEM_ERROR_LIMIT = 3

# Per-altitude priors: (alpha, beta)
_PRIOR_HIGH = (2.0, 1.0)  # alt >= 60°
_PRIOR_MID = (1.0, 1.0)  # 20° <= alt < 60°
_PRIOR_LOW = (1.0, 2.0)  # alt < 20°


# ---------- enums + outcomes ----------------------------------------


class SolveOutcome(str, Enum):
    SOLVED = "solved"
    NO_STARS = "no_stars"
    TIMEOUT = "timeout"
    SYSTEM_ERROR = "system_error"


@dataclass
class StopReason:
    reason: str  # human-friendly
    # One of: "user", "force", "convergence", "max_runtime",
    # "system_errors", "sun_refusals", "internal_error".
    code: str


# ---------- math utilities ------------------------------------------


def bernoulli_entropy(p: float) -> float:
    """Bernoulli entropy in nats. Edge cases (p∈{0,1}) → 0."""
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -p * math.log(p) - (1.0 - p) * math.log(1.0 - p)


def beta_eig(alpha: float, beta: float) -> float:
    """Closed-form expected information gain (nats) of one more Bernoulli
    observation under a ``Beta(alpha, beta)`` posterior."""
    n = alpha + beta
    if n <= 0:
        return 0.0
    mu = alpha / n
    H_p = bernoulli_entropy(mu)
    H_solved = bernoulli_entropy((alpha + 1.0) / (n + 1.0))
    H_failed = bernoulli_entropy(alpha / (n + 1.0))
    return max(0.0, H_p - (mu * H_solved + (1.0 - mu) * H_failed))


def prior_for_alt(alt_deg: float) -> tuple[float, float]:
    if alt_deg >= 60.0:
        return _PRIOR_HIGH
    if alt_deg >= 20.0:
        return _PRIOR_MID
    return _PRIOR_LOW


# ---------- callback types ------------------------------------------


# slew_func(az_deg, alt_deg) → bool. ``True`` if the slew completed
# (or was commanded successfully); ``False`` signals a sun-safety
# refusal that the mapper counts toward its sun-refusal abort
# threshold. To signal a transient/system failure (network, RPC,
# astropy), raise an exception instead — the mapper increments its
# consecutive-system-errors counter on raise.
SlewFunc = Callable[[float, float], bool]
PlateSolveFunc = Callable[[float], SolveOutcome]
EmitFunc = Callable[[dict], None]


# ---------- options -------------------------------------------------


@dataclass
class VisibilityMapperOptions:
    min_alt_deg: float = 10.0
    max_runtime_min: float = DEFAULT_MAX_RUNTIME_MIN
    solve_timeout_s: float = SOLVE_TIMEOUT_S
    slew_rate_deg_s: float = SLEW_RATE_DEG_S
    slew_settle_s: float = SLEW_SETTLE_S
    t_observe_s: float = T_OBSERVE_S
    # Test hooks: shorten timing in tests.
    decay_interval_s: float = DECAY_INTERVAL_S
    convergence_elapsed_s: float = CONVERGENCE_ELAPSED_S
    min_run_before_user_stop_s: float = MIN_RUN_BEFORE_USER_STOP_S
    frontier_decay_window_s: float = FRONTIER_DECAY_WINDOW_S
    failure_recent_window_s: float = FAILURE_RECENT_WINDOW_S
    persist_interval_s: float = PERSIST_INTERVAL_S


# ---------- mapper --------------------------------------------------


class VisibilityMapper:
    """Live visibility-map service. One per telescope.

    Lifecycle:
    1. Constructor builds the grid, primes priors (or loads snapshot).
    2. ``start()`` spawns the worker thread.
    3. Worker selects candidate cells, scores them, slews + plate-
       solves the winner, updates the posterior, smooths, persists,
       emits events.
    4. ``stop()`` / ``force_stop()`` ends the run; the worker exits at
       its next loop iteration.
    """

    def __init__(
        self,
        telescope_id: int,
        grid: SkyGrid,
        slew_func: SlewFunc,
        plate_solve_func: PlateSolveFunc,
        *,
        options: Optional[VisibilityMapperOptions] = None,
        state_dir: Optional[Path] = None,
        time_func: Callable[[], float] = time.time,
    ):
        self.telescope_id = int(telescope_id)
        self.grid = grid
        self.slew_func = slew_func
        self.plate_solve_func = plate_solve_func
        self.options = options or VisibilityMapperOptions()
        self.state_dir = (
            Path(state_dir)
            if state_dir is not None
            else Path(__file__).resolve().parents[1] / "device_state"
        )
        self._time = time_func

        n = len(grid)
        self._n = n
        # Posterior arrays.
        self._alpha = np.zeros(n, dtype=np.float64)
        self._beta = np.zeros(n, dtype=np.float64)
        self._prior_alpha = np.zeros(n, dtype=np.float64)
        self._prior_beta = np.zeros(n, dtype=np.float64)
        # Bookkeeping.
        self._last_obs_at = np.full(n, -math.inf, dtype=np.float64)
        self._last_failure_at = np.full(n, -math.inf, dtype=np.float64)
        self._failure_count = np.zeros(n, dtype=np.int32)
        self._n_observations_at = np.zeros(n, dtype=np.int32)
        # Vectorized cell coordinates for fast scoring.
        self._az = np.array([c.az_deg for c in grid.cells], dtype=np.float64)
        self._alt = np.array([c.alt_deg for c in grid.cells], dtype=np.float64)
        self._below_floor = np.array([c.below_floor for c in grid.cells], dtype=bool)

        for i, cell in enumerate(grid.cells):
            pa, pb = prior_for_alt(cell.alt_deg)
            self._prior_alpha[i] = pa
            self._prior_beta[i] = pb
            self._alpha[i] = pa
            self._beta[i] = pb

        # State.
        self._lock = threading.RLock()
        self._stop_evt = threading.Event()
        self._force_stop = False
        self._user_stop_requested = False
        self._thread: Optional[threading.Thread] = None
        self._started_at: Optional[float] = None
        self._stopped_at: Optional[float] = None
        self._stop_reason: Optional[StopReason] = None
        self._last_decay_at: Optional[float] = None
        self._last_persist_at: Optional[float] = None
        self._cur_target_idx: Optional[int] = None
        self._last_pointing: tuple[float, float] = (0.0, 90.0)  # zenith default
        self._sun_refusals: deque[float] = deque(maxlen=SUN_REFUSAL_LIMIT * 2)
        self._consecutive_system_errors = 0
        self._n_observations = 0
        self._seq = 0  # bumped on every state change; used by SSE polling
        self._listeners: list[Callable[[dict], None]] = []
        self._errors: list[str] = []

        # Try to resume from a snapshot if present.
        self._load_snapshot_if_present()

    # ---------- snapshot --------------------------------------------

    def _load_snapshot_if_present(self) -> None:
        try:
            path = visibility_map_path(self.state_dir, self.telescope_id)
        except OSError:
            return
        snap = load_snapshot(path)
        if snap is None:
            return
        if snap.get("grid_kind") != self.grid.kind:
            # Schema mismatch — different grid implementation, drop.
            return
        cells = snap.get("cells")
        if not isinstance(cells, list) or len(cells) != self._n:
            return
        for i, cd in enumerate(cells):
            if not isinstance(cd, dict):
                continue
            try:
                self._alpha[i] = float(cd.get("alpha", self._prior_alpha[i]))
                self._beta[i] = float(cd.get("beta", self._prior_beta[i]))
                self._failure_count[i] = int(cd.get("failure_count", 0))
                # Reset timestamps so a resume doesn't immediately
                # re-trigger the recent-failure exclusion: the user
                # has had to come back, so the 10-min clock starts fresh.
                self._last_obs_at[i] = -math.inf
                self._last_failure_at[i] = -math.inf
            except (TypeError, ValueError):
                continue

    def _persist(self) -> None:
        """Force a snapshot write. Use ``_maybe_persist`` from the loop."""
        try:
            path = visibility_map_path(self.state_dir, self.telescope_id)
        except OSError as exc:
            logger.warning("cannot create state dir for visibility map: %s", exc)
            return
        with self._lock:
            elapsed = self._elapsed_s_locked()
            cells = [
                {
                    "idx": int(i),
                    "az_deg": float(self._az[i]),
                    "alt_deg": float(self._alt[i]),
                    "alpha": float(self._alpha[i]),
                    "beta": float(self._beta[i]),
                    "failure_count": int(self._failure_count[i]),
                    "n_obs": int(self._n_observations_at[i]),
                }
                for i in range(self._n)
            ]
            grid_kind = self.grid.kind
            min_alt = self.options.min_alt_deg
            started = self._started_at or 0.0
            n_obs = self._n_observations
        try:
            save_snapshot(
                path,
                telescope_id=self.telescope_id,
                grid_kind=grid_kind,
                min_alt_deg=min_alt,
                cells=cells,
                started_at=started,
                elapsed_s=elapsed,
                n_observations=n_obs,
            )
        except OSError as exc:
            logger.warning("visibility map snapshot failed: %s", exc)
        with self._lock:
            self._last_persist_at = self._time()

    def _maybe_persist(self, now: float) -> None:
        """Persist only if it has been long enough since the last save.

        Always writes the first time so the snapshot file exists for the
        UI within the first observation window.
        """
        with self._lock:
            last = self._last_persist_at
            interval = self.options.persist_interval_s
        if last is not None and (now - last) < interval:
            return
        self._persist()

    # ---------- public API ------------------------------------------

    def start(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                raise RuntimeError("visibility mapper is already running")
            self._stop_evt.clear()
            self._force_stop = False
            self._user_stop_requested = False
            self._started_at = self._time()
            self._stopped_at = None
            self._stop_reason = None
            self._last_decay_at = self._started_at
            self._consecutive_system_errors = 0
            self._sun_refusals.clear()
            self._errors.clear()
            self._seq += 1
            t = threading.Thread(
                target=self._run_loop,
                name=f"VisibilityMapper-{self.telescope_id}",
                daemon=True,
            )
            self._thread = t
            t.start()

    def request_stop(self, *, force: bool = False) -> tuple[bool, str]:
        """Request the mapper stop.

        Returns ``(accepted, reason)``. Non-force stops are rejected if
        the run has been alive for less than the user-stop minimum
        (default 5 min); force stops always succeed.
        """
        with self._lock:
            elapsed = self._elapsed_s_locked()
            if not force and elapsed < self.options.min_run_before_user_stop_s:
                return (
                    False,
                    f"stop disabled until {self.options.min_run_before_user_stop_s:.0f}s elapsed (currently {elapsed:.0f}s)",
                )
            self._user_stop_requested = True
            if force:
                self._force_stop = True
                self._stop_reason = StopReason("user force-stopped", "force")
            else:
                self._stop_reason = StopReason("user stopped", "user")
            self._stop_evt.set()
            self._seq += 1
        return (True, "stopping")

    def join(self, timeout: Optional[float] = None) -> None:
        t = self._thread
        if t is not None:
            t.join(timeout=timeout)

    def is_active(self) -> bool:
        with self._lock:
            return self._thread is not None and self._thread.is_alive()

    # ---------- status ----------------------------------------------

    def status(self) -> dict:
        with self._lock:
            return self._status_locked()

    def _status_locked(self) -> dict:
        elapsed = self._elapsed_s_locked()
        active = self._thread is not None and self._thread.is_alive()
        active_mask = ~self._below_floor
        active_ix = np.where(active_mask)[0]
        if len(active_ix) > 0:
            mu = self._alpha[active_ix] / (
                self._alpha[active_ix] + self._beta[active_ix]
            )
            ent = np.array([bernoulli_entropy(float(p)) for p in mu])
            n_observed = int(np.sum(self._n_observations_at[active_ix] > 0))
            n_visible = int(np.sum(mu > VISIBLE_THRESHOLD))
            median_ent = float(np.median(ent)) if len(ent) > 0 else 0.0
        else:
            n_observed = 0
            n_visible = 0
            median_ent = 0.0
        max_ent = math.log(2.0)  # max Bernoulli entropy in nats
        map_quality = (
            max(0.0, min(1.0, 1.0 - median_ent / max_ent)) if max_ent > 0 else 0.0
        )
        # Convergence ETA (very rough): if median entropy is decreasing
        # toward the threshold, project from start.
        stop_eta_s = None
        if (
            active
            and elapsed >= 60.0
            and median_ent > CONVERGENCE_MEDIAN_ENTROPY_NATS
            and median_ent < max_ent
        ):
            initial_ent = max_ent
            decay_per_s = max(1e-9, (initial_ent - median_ent) / max(1.0, elapsed))
            remaining = (median_ent - CONVERGENCE_MEDIAN_ENTROPY_NATS) / decay_per_s
            stop_eta_s = max(0.0, remaining)
        max_elapsed = self.options.max_runtime_min * 60.0
        remaining_runtime = max(0.0, max_elapsed - elapsed) if active else None
        return {
            "active": bool(active),
            "telescope_id": int(self.telescope_id),
            "elapsed_s": float(elapsed),
            "n_cells_total": int(self._n),
            "n_cells_active": int(len(active_ix)),
            "n_cells_observed": int(n_observed),
            "n_cells_visible": int(n_visible),
            "n_observations": int(self._n_observations),
            "median_entropy_nats": float(median_ent),
            "max_entropy_nats": float(max_ent),
            "map_quality": float(map_quality),
            "stop_eta_s": stop_eta_s,
            "remaining_runtime_s": remaining_runtime,
            "min_alt_deg": float(self.options.min_alt_deg),
            "max_runtime_min": float(self.options.max_runtime_min),
            "stop_eligible": bool(elapsed >= self.options.min_run_before_user_stop_s),
            "current_target_idx": self._cur_target_idx,
            "stop_reason": (
                {"reason": self._stop_reason.reason, "code": self._stop_reason.code}
                if self._stop_reason
                else None
            ),
            "errors": list(self._errors),
            "seq": int(self._seq),
            "grid_kind": self.grid.kind,
        }

    def cells_snapshot(self) -> list[dict]:
        """Return the current per-cell state, suitable for the UI canvas."""
        with self._lock:
            out: list[dict] = []
            for i in range(self._n):
                a = float(self._alpha[i])
                b = float(self._beta[i])
                tot = a + b
                ep = a / tot if tot > 0 else 0.5
                out.append(
                    {
                        "idx": int(i),
                        "az": float(self._az[i]),
                        "alt": float(self._alt[i]),
                        "alpha": a,
                        "beta": b,
                        "ep": float(ep),
                        "n_obs": int(self._n_observations_at[i]),
                        "fail": int(self._failure_count[i]),
                        "below_floor": bool(self._below_floor[i]),
                    }
                )
            return out

    def add_listener(self, fn: EmitFunc) -> Callable[[], None]:
        """Subscribe to per-observation events. Returns an unsubscribe
        callable."""
        with self._lock:
            self._listeners.append(fn)

        def _unsub() -> None:
            with self._lock:
                try:
                    self._listeners.remove(fn)
                except ValueError:
                    pass

        return _unsub

    def _emit(self, payload: dict) -> None:
        with self._lock:
            listeners = list(self._listeners)
        for fn in listeners:
            try:
                fn(payload)
            except Exception:  # noqa: BLE001
                logger.exception("visibility mapper listener raised")

    def _seq_get(self) -> int:
        with self._lock:
            return self._seq

    # ---------- internals: timing -----------------------------------

    def _elapsed_s_locked(self) -> float:
        if self._started_at is None:
            return 0.0
        end_at = self._stopped_at if self._stopped_at is not None else self._time()
        return max(0.0, end_at - self._started_at)

    # ---------- internals: candidate selection ----------------------

    def _select_candidates(self, now: float) -> list[int]:
        """Return up to CANDIDATE_CAP cell indices to score this step."""
        with self._lock:
            mask = ~self._below_floor & (self._failure_count < 3)
            # Recent-failure exclusion.
            since_fail = now - self._last_failure_at
            mask &= ~np.isfinite(self._last_failure_at) | (
                since_fail >= self.options.failure_recent_window_s
            )
            ix = np.where(mask)[0]
            if len(ix) == 0:
                return []

            # Top-K by entropy.
            mu = self._alpha[ix] / (self._alpha[ix] + self._beta[ix])
            ents = np.array([bernoulli_entropy(float(p)) for p in mu])
            k = min(CANDIDATE_TOP_K, len(ix))
            if k < len(ix):
                top_local = np.argpartition(-ents, k - 1)[:k]
            else:
                top_local = np.arange(len(ix))
            top = set(int(ix[i]) for i in top_local)

            # Frontier set: any cell with a neighbor whose E[p] > VISIBLE_THRESHOLD.
            # Compute on the candidates ix only — frontier cells must
            # also be on the active mask.
            visible_mu = self._alpha / (self._alpha + self._beta)
            visible_set = visible_mu > VISIBLE_THRESHOLD
            mask_set = set(int(i) for i in ix)
            frontier: set[int] = set()
            for i in ix:
                ii = int(i)
                for n in self.grid.neighbors[ii]:
                    if visible_set[n]:
                        frontier.add(ii)
                        break

            cands = top | frontier
            if len(cands) > CANDIDATE_CAP:
                # Trim to the highest-entropy CANDIDATE_CAP from the union.
                pairs = [(-bernoulli_entropy(float(visible_mu[c])), c) for c in cands]
                pairs.sort()
                cands = set(c for _, c in pairs[:CANDIDATE_CAP])
            return [c for c in cands if c in mask_set]

    # ---------- internals: scoring ---------------------------------

    def _frontier_bonus(self, elapsed: float) -> float:
        win = self.options.frontier_decay_window_s
        if win <= 0:
            return FRONTIER_BONUS_FINAL
        if elapsed >= win:
            return FRONTIER_BONUS_FINAL
        progress = elapsed / win
        return (
            FRONTIER_BONUS_INITIAL
            + (FRONTIER_BONUS_FINAL - FRONTIER_BONUS_INITIAL) * progress
        )

    def _score_candidate(
        self, idx: int, cur_az: float, cur_alt: float, frontier_bonus: float
    ) -> float:
        a = float(self._alpha[idx])
        b = float(self._beta[idx])
        eig = beta_eig(a, b)
        slew_deg = great_circle_deg(
            cur_az, cur_alt, float(self._az[idx]), float(self._alt[idx])
        )
        t_total = (
            slew_deg / max(1e-3, self.options.slew_rate_deg_s)
            + self.options.slew_settle_s
            + self.options.t_observe_s
        )
        base = eig / max(1e-3, t_total)
        # Frontier neighbor count (E[p] > 0.7).
        n_visible = 0
        for n in self.grid.neighbors[idx]:
            if (self._alpha[n] / (self._alpha[n] + self._beta[n])) > VISIBLE_THRESHOLD:
                n_visible += 1
        return base * (1.0 + frontier_bonus * n_visible)

    def _argmax_score(
        self, candidates: list[int], cur_az: float, cur_alt: float, elapsed: float
    ) -> Optional[int]:
        if not candidates:
            return None
        bonus = self._frontier_bonus(elapsed)
        best_idx = -1
        best_score = -math.inf
        for c in candidates:
            s = self._score_candidate(c, cur_az, cur_alt, bonus)
            if s > best_score:
                best_score = s
                best_idx = c
        return best_idx if best_idx >= 0 else None

    # ---------- internals: observation update ----------------------

    def _apply_observation(self, idx: int, outcome: SolveOutcome, now: float) -> None:
        with self._lock:
            self._n_observations_at[idx] += 1
            self._last_obs_at[idx] = now
            if outcome == SolveOutcome.SOLVED:
                self._alpha[idx] += 3.0
                self._failure_count[idx] = 0
            elif outcome in (SolveOutcome.NO_STARS, SolveOutcome.TIMEOUT):
                self._failure_count[idx] += 1
                fc = int(self._failure_count[idx])
                if fc == 1:
                    inc = 0.5
                elif fc == 2:
                    inc = 1.0
                elif fc == 3:
                    inc = 2.0
                else:
                    inc = 0.0  # ≥4 — should not happen given filter
                self._beta[idx] += inc
                self._last_failure_at[idx] = now
            # SYSTEM_ERROR: do not update posterior; counter handled in loop.
            self._n_observations += 1
            self._seq += 1

    # ---------- internals: MRF smoothing ---------------------------

    def _mrf_smooth_pass(self) -> None:
        """One pass: low-confidence cells (alpha+beta < 4) blend their
        Beta with the weighted neighbor mean."""
        with self._lock:
            # Snapshot the current means + weights so we don't see our
            # own update mid-iteration.
            tot = self._alpha + self._beta
            mu = np.where(tot > 0, self._alpha / np.where(tot > 0, tot, 1.0), 0.5)
            new_alpha = self._alpha.copy()
            new_beta = self._beta.copy()
            for i in range(self._n):
                if self._below_floor[i]:
                    continue
                if tot[i] >= LOW_CONFIDENCE_TOTAL:
                    continue
                ns = self.grid.neighbors[i]
                if not ns:
                    continue
                w_sum = 0.0
                num = 0.0
                for n in ns:
                    w = float(tot[n])
                    if w <= 0:
                        continue
                    w_sum += w
                    num += float(mu[n]) * w
                if w_sum <= 0:
                    continue
                neighbor_mean = num / w_sum
                own_mean = float(mu[i])
                new_mean = MRF_BLEND_W * neighbor_mean + (1.0 - MRF_BLEND_W) * own_mean
                new_mean = max(1e-6, min(1.0 - 1e-6, new_mean))
                t = float(tot[i])
                new_alpha[i] = t * new_mean
                new_beta[i] = t * (1.0 - new_mean)
            self._alpha = new_alpha
            self._beta = new_beta

    # ---------- internals: time decay -----------------------------

    def _maybe_apply_time_decay(self, now: float) -> None:
        with self._lock:
            if self._last_decay_at is None:
                self._last_decay_at = now
                return
            if now - self._last_decay_at < self.options.decay_interval_s:
                return
            # Apply decay; floor at the prior so we don't decay below
            # initial belief.
            self._alpha = np.maximum(self._alpha * DECAY_FACTOR, self._prior_alpha)
            self._beta = np.maximum(self._beta * DECAY_FACTOR, self._prior_beta)
            # Slowly forgive failure counts so cells that consistently
            # failed a few decay cycles ago can be re-tried as
            # conditions change.
            self._failure_count = np.maximum(0, self._failure_count - 1)
            self._last_decay_at = now
            self._seq += 1

    # ---------- internals: stopping --------------------------------

    def _check_stopping_conditions(self, now: float) -> Optional[StopReason]:
        with self._lock:
            elapsed = self._elapsed_s_locked()
            max_elapsed = self.options.max_runtime_min * 60.0
            if elapsed >= max_elapsed:
                return StopReason(
                    f"max runtime reached ({self.options.max_runtime_min:.0f} min)",
                    "max_runtime",
                )
            if self._consecutive_system_errors >= SYSTEM_ERROR_LIMIT:
                return StopReason(
                    f"{SYSTEM_ERROR_LIMIT} consecutive system errors", "system_errors"
                )
            # Sun refusals rate-limited.
            window_start = now - SUN_REFUSAL_WINDOW_S
            recent = sum(1 for t in self._sun_refusals if t >= window_start)
            if recent >= SUN_REFUSAL_LIMIT:
                return StopReason(
                    f"{SUN_REFUSAL_LIMIT} sun-safety refusals in {SUN_REFUSAL_WINDOW_S:.0f}s",
                    "sun_refusals",
                )
            # Convergence.
            if elapsed >= self.options.convergence_elapsed_s:
                active_ix = np.where(~self._below_floor)[0]
                if len(active_ix) > 0:
                    mu = self._alpha[active_ix] / (
                        self._alpha[active_ix] + self._beta[active_ix]
                    )
                    ents = np.array([bernoulli_entropy(float(p)) for p in mu])
                    if (
                        len(ents) > 0
                        and float(np.median(ents)) < CONVERGENCE_MEDIAN_ENTROPY_NATS
                    ):
                        return StopReason(
                            "median entropy converged below threshold", "convergence"
                        )
            return None

    # ---------- internals: main loop -------------------------------

    def _run_loop(self) -> None:
        try:
            # Write an initial snapshot so the file exists for the UI
            # before the first observation completes (and so short
            # tests still see a saved snapshot).
            self._persist()
            self._loop_body()
        except Exception:  # noqa: BLE001
            logger.exception("visibility mapper crashed")
            with self._lock:
                self._stop_reason = StopReason(
                    "internal error in mapper loop", "internal_error"
                )
                self._errors.append("internal error in mapper loop")
        finally:
            with self._lock:
                self._stopped_at = self._time()
                self._seq += 1
            # Always save the final state on stop, regardless of when
            # the throttled in-loop save last ran.
            self._persist()
            self._emit({"type": "complete", "status": self.status()})

    def _loop_body(self) -> None:
        while not self._stop_evt.is_set():
            now = self._time()
            self._maybe_apply_time_decay(now)
            stop = self._check_stopping_conditions(now)
            if stop:
                with self._lock:
                    self._stop_reason = stop
                self._stop_evt.set()
                break
            if self._user_stop_requested:
                # Already set to stop above; honor it once min-run is past.
                if (
                    self._force_stop
                    or self._elapsed_s_locked()
                    >= self.options.min_run_before_user_stop_s
                ):
                    self._stop_evt.set()
                    break

            cur_az, cur_alt = self._last_pointing
            elapsed = self._elapsed_s_locked()
            cands = self._select_candidates(now)
            if not cands:
                # Nothing to sample (all cells excluded). Wait briefly.
                if self._stop_evt.wait(timeout=15.0):
                    break
                continue
            target = self._argmax_score(cands, cur_az, cur_alt, elapsed)
            if target is None:
                if self._stop_evt.wait(timeout=15.0):
                    break
                continue
            with self._lock:
                self._cur_target_idx = int(target)
                self._seq += 1
            cell_az = float(self._az[target])
            cell_alt = float(self._alt[target])

            # Slew. False = refused (treated as sun-safety refusal).
            slew_ok = False
            try:
                slew_ok = bool(self.slew_func(cell_az, cell_alt))
            except Exception as exc:  # noqa: BLE001
                with self._lock:
                    self._errors.append(f"slew error: {exc}")
                    self._consecutive_system_errors += 1
                logger.warning("visibility map slew raised: %s", exc)
                self._emit({"type": "system_error", "stage": "slew", "error": str(exc)})
                if self._stop_evt.wait(timeout=2.0):
                    break
                continue
            if not slew_ok:
                with self._lock:
                    self._sun_refusals.append(now)
                self._emit(
                    {
                        "type": "slew_refused",
                        "az": cell_az,
                        "alt": cell_alt,
                        "idx": int(target),
                    }
                )
                if self._stop_evt.wait(timeout=2.0):
                    break
                continue

            self._last_pointing = (cell_az, cell_alt)

            # Plate solve. Convert exceptions / errors to outcomes.
            outcome = SolveOutcome.SYSTEM_ERROR
            try:
                outcome = self.plate_solve_func(self.options.solve_timeout_s)
            except Exception as exc:  # noqa: BLE001
                logger.warning("visibility map plate-solve raised: %s", exc)
                with self._lock:
                    self._errors.append(f"plate solve error: {exc}")

            if outcome == SolveOutcome.SYSTEM_ERROR:
                with self._lock:
                    self._consecutive_system_errors += 1
                self._emit(
                    {
                        "type": "system_error",
                        "stage": "plate_solve",
                        "idx": int(target),
                    }
                )
                # Don't update posterior on SYSTEM_ERROR.
                if self._stop_evt.wait(timeout=2.0):
                    break
                continue

            # Successful observation (SOLVED or graceful failure).
            with self._lock:
                self._consecutive_system_errors = 0
            self._apply_observation(target, outcome, now)
            self._mrf_smooth_pass()
            self._maybe_persist(now)
            self._emit(
                {
                    "type": "observation",
                    "idx": int(target),
                    "outcome": outcome.value,
                    "az": cell_az,
                    "alt": cell_alt,
                    "alpha": float(self._alpha[target]),
                    "beta": float(self._beta[target]),
                    "ep": float(
                        self._alpha[target] / (self._alpha[target] + self._beta[target])
                    ),
                    "elapsed_s": float(self._elapsed_s_locked()),
                }
            )


# ---------- manager (singleton-per-process, telescope-keyed) --------


class VisibilityMapManager:
    """Singleton-per-process registry, telescope-keyed.

    Mirrors the pattern used by ``NighttimeCalibrationManager``. A
    process can have at most one mapper per telescope at a time.
    """

    def __init__(self) -> None:
        self._mappers: dict[int, VisibilityMapper] = {}
        self._lock = threading.Lock()

    def get(self, telescope_id: int) -> Optional[VisibilityMapper]:
        with self._lock:
            return self._mappers.get(int(telescope_id))

    def is_running(self, telescope_id: int) -> bool:
        m = self.get(telescope_id)
        return m is not None and m.is_active()

    def start(self, mapper: VisibilityMapper) -> VisibilityMapper:
        tid = int(mapper.telescope_id)
        with self._lock:
            existing = self._mappers.get(tid)
            if existing is not None and existing.is_active():
                raise RuntimeError(
                    f"telescope {tid} already has a visibility-map run in progress"
                )
            self._mappers[tid] = mapper
        mapper.start()
        return mapper

    def stop(
        self, telescope_id: int, *, force: bool = False
    ) -> tuple[bool, str, Optional[dict]]:
        m = self.get(telescope_id)
        if m is None:
            return (False, "no mapper", None)
        accepted, msg = m.request_stop(force=force)
        return (accepted, msg, m.status())

    def status(self, telescope_id: int) -> Optional[dict]:
        m = self.get(telescope_id)
        return m.status() if m is not None else None

    def clear_inactive(self, telescope_id: int) -> None:
        """Drop a finished mapper so a subsequent ``start`` is fresh."""
        with self._lock:
            m = self._mappers.get(int(telescope_id))
            if m is not None and not m.is_active():
                del self._mappers[int(telescope_id)]


_MANAGER: Optional[VisibilityMapManager] = None
_MANAGER_LOCK = threading.Lock()


def get_visibility_manager() -> VisibilityMapManager:
    global _MANAGER
    with _MANAGER_LOCK:
        if _MANAGER is None:
            _MANAGER = VisibilityMapManager()
        return _MANAGER
