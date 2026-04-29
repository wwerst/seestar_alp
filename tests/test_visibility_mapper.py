"""Tests for the Bayesian active sky-visibility mapper.

These exercise the algorithm end-to-end with synthetic
slew/plate-solve callbacks, so the math contract (priors, observation
update, MRF smoothing, time decay, acquisition, candidate selection,
stopping conditions, persistence) is pinned down without having to
spin a real telescope or simulator.
"""

from __future__ import annotations

import math
import threading
import time

import pytest

from device.sky_grid import make_altaz_band_grid
from device.visibility_mapper import (
    SolveOutcome,
    VisibilityMapManager,
    VisibilityMapper,
    VisibilityMapperOptions,
    bernoulli_entropy,
    beta_eig,
    prior_for_alt,
)
from device.visibility_persistence import (
    load_snapshot,
    save_snapshot,
    visibility_map_path,
)


# ---------- math helpers --------------------------------------------


def test_bernoulli_entropy_zero_at_endpoints():
    assert bernoulli_entropy(0.0) == 0.0
    assert bernoulli_entropy(1.0) == 0.0
    assert bernoulli_entropy(-0.1) == 0.0
    assert bernoulli_entropy(1.5) == 0.0


def test_bernoulli_entropy_max_at_half():
    assert bernoulli_entropy(0.5) == pytest.approx(math.log(2.0))


def test_beta_eig_nonnegative():
    for a in (1.0, 2.0, 5.0):
        for b in (1.0, 2.0, 5.0):
            assert beta_eig(a, b) >= 0.0


def test_beta_eig_drops_with_confidence():
    # Tight posterior (a=10, b=1) is nearly determined → low EIG.
    eig_diffuse = beta_eig(1.0, 1.0)
    eig_tight = beta_eig(20.0, 1.0)
    assert eig_diffuse > eig_tight


def test_prior_for_alt_brackets():
    assert prior_for_alt(75.0) == (2.0, 1.0)
    assert prior_for_alt(45.0) == (1.0, 1.0)
    assert prior_for_alt(60.0) == (2.0, 1.0)
    assert prior_for_alt(20.0) == (1.0, 1.0)
    assert prior_for_alt(10.0) == (1.0, 2.0)


# ---------- fixtures: synthetic plate-solver -----------------------


class FakeSky:
    """Synthetic sky for tests.

    A list of (az_min, az_max, alt_min, alt_max) "obstruction" boxes.
    Outside boxes → SOLVED. Inside boxes → NO_STARS.
    Optional ``error_pattern``: list of indices (in call order) at
    which to inject SYSTEM_ERROR.
    """

    def __init__(
        self,
        obstructions: list[tuple[float, float, float, float]],
        *,
        error_at_calls: list[int] | None = None,
        delay_s: float = 0.0,
    ):
        self.obstructions = list(obstructions)
        self.error_at_calls = set(error_at_calls or [])
        self.delay_s = float(delay_s)
        self.calls: list[tuple[float, float]] = []
        self.cur_az = 0.0
        self.cur_alt = 90.0
        self.lock = threading.Lock()

    def slew(self, az: float, alt: float) -> bool:
        with self.lock:
            self.cur_az = float(az) % 360.0
            self.cur_alt = float(alt)
        return True

    def plate_solve(self, timeout_s: float) -> SolveOutcome:
        with self.lock:
            call_idx = len(self.calls)
            self.calls.append((self.cur_az, self.cur_alt))
        if self.delay_s:
            time.sleep(self.delay_s)
        if call_idx in self.error_at_calls:
            return SolveOutcome.SYSTEM_ERROR
        for az_min, az_max, alt_min, alt_max in self.obstructions:
            if self._in_box(self.cur_az, az_min, az_max) and (
                alt_min <= self.cur_alt <= alt_max
            ):
                return SolveOutcome.NO_STARS
        return SolveOutcome.SOLVED

    @staticmethod
    def _in_box(az: float, az_min: float, az_max: float) -> bool:
        if az_min <= az_max:
            return az_min <= az <= az_max
        # Wrap.
        return az >= az_min or az <= az_max


@pytest.fixture
def small_grid():
    """A small grid for fast tests — 5° bands, full 0–90 alt."""
    return make_altaz_band_grid(min_alt_deg=10.0)


@pytest.fixture
def fast_options():
    """Test-only options: tiny min-run window so stop is allowed."""
    return VisibilityMapperOptions(
        min_alt_deg=10.0,
        max_runtime_min=10.0,
        solve_timeout_s=1.0,
        slew_rate_deg_s=1000.0,  # effectively instant
        slew_settle_s=0.0,
        t_observe_s=0.001,
        decay_interval_s=600.0,
        convergence_elapsed_s=600.0,
        min_run_before_user_stop_s=0.0,
        frontier_decay_window_s=600.0,
        failure_recent_window_s=600.0,
    )


# ---------- observation model ---------------------------------------


def test_observation_solved_increments_alpha(small_grid, fast_options, tmp_path):
    sky = FakeSky([])
    mapper = VisibilityMapper(
        telescope_id=1,
        grid=small_grid,
        slew_func=sky.slew,
        plate_solve_func=sky.plate_solve,
        options=fast_options,
        state_dir=tmp_path,
    )
    # Find a mid-altitude cell.
    idx = next(
        i
        for i, c in enumerate(small_grid.cells)
        if not c.below_floor and 30.0 < c.alt_deg < 60.0
    )
    a0 = float(mapper._alpha[idx])
    mapper._apply_observation(idx, SolveOutcome.SOLVED, time.time())
    assert mapper._alpha[idx] == pytest.approx(a0 + 3.0)
    assert int(mapper._failure_count[idx]) == 0


def test_observation_failure_progression(small_grid, fast_options, tmp_path):
    sky = FakeSky([])
    mapper = VisibilityMapper(
        telescope_id=1,
        grid=small_grid,
        slew_func=sky.slew,
        plate_solve_func=sky.plate_solve,
        options=fast_options,
        state_dir=tmp_path,
    )
    idx = next(
        i
        for i, c in enumerate(small_grid.cells)
        if not c.below_floor and 30.0 < c.alt_deg < 60.0
    )
    b0 = float(mapper._beta[idx])
    now = time.time()
    mapper._apply_observation(idx, SolveOutcome.NO_STARS, now)
    assert mapper._beta[idx] == pytest.approx(b0 + 0.5)
    mapper._apply_observation(idx, SolveOutcome.NO_STARS, now + 700)
    assert mapper._beta[idx] == pytest.approx(b0 + 0.5 + 1.0)
    mapper._apply_observation(idx, SolveOutcome.NO_STARS, now + 2500)
    assert mapper._beta[idx] == pytest.approx(b0 + 0.5 + 1.0 + 2.0)
    # 4th attempt: posterior unchanged but failure_count keeps climbing.
    b3 = float(mapper._beta[idx])
    mapper._apply_observation(idx, SolveOutcome.NO_STARS, now + 5000)
    assert mapper._beta[idx] == pytest.approx(b3)
    assert int(mapper._failure_count[idx]) == 4


def test_observation_solved_resets_failure_count(small_grid, fast_options, tmp_path):
    sky = FakeSky([])
    mapper = VisibilityMapper(
        telescope_id=1,
        grid=small_grid,
        slew_func=sky.slew,
        plate_solve_func=sky.plate_solve,
        options=fast_options,
        state_dir=tmp_path,
    )
    idx = next(i for i, c in enumerate(small_grid.cells) if not c.below_floor)
    mapper._apply_observation(idx, SolveOutcome.NO_STARS, time.time())
    assert mapper._failure_count[idx] == 1
    mapper._apply_observation(idx, SolveOutcome.SOLVED, time.time())
    assert mapper._failure_count[idx] == 0


# ---------- MRF smoothing ------------------------------------------


def test_mrf_pulls_low_confidence_toward_neighbors(small_grid, fast_options, tmp_path):
    sky = FakeSky([])
    mapper = VisibilityMapper(
        telescope_id=1,
        grid=small_grid,
        slew_func=sky.slew,
        plate_solve_func=sky.plate_solve,
        options=fast_options,
        state_dir=tmp_path,
    )
    # Pick a cell with at least one neighbor in the active grid.
    for idx in range(len(small_grid.cells)):
        if small_grid.cells[idx].below_floor:
            continue
        ns = [
            n for n in small_grid.neighbors[idx] if not small_grid.cells[n].below_floor
        ]
        if ns:
            break
    target = idx
    neighbor_ix = ns
    # Make every neighbor strongly visible.
    for n in neighbor_ix:
        mapper._alpha[n] = 20.0
        mapper._beta[n] = 1.0
    # Target is uninformative (still at prior).
    a_before = float(mapper._alpha[target])
    b_before = float(mapper._beta[target])
    ep_before = a_before / (a_before + b_before)
    mapper._mrf_smooth_pass()
    a_after = float(mapper._alpha[target])
    b_after = float(mapper._beta[target])
    ep_after = a_after / (a_after + b_after)
    # E[p] should have moved toward 1 (visible neighbors).
    assert ep_after > ep_before
    # Confidence preserved (sum unchanged within fp tolerance).
    assert (a_after + b_after) == pytest.approx(a_before + b_before)


def test_mrf_does_not_modify_high_confidence_cell(small_grid, fast_options, tmp_path):
    sky = FakeSky([])
    mapper = VisibilityMapper(
        telescope_id=1,
        grid=small_grid,
        slew_func=sky.slew,
        plate_solve_func=sky.plate_solve,
        options=fast_options,
        state_dir=tmp_path,
    )
    target = next(i for i, c in enumerate(small_grid.cells) if not c.below_floor)
    # Make target high-confidence (alpha+beta >= 4).
    mapper._alpha[target] = 5.0
    mapper._beta[target] = 5.0
    # Make all neighbors visible.
    for n in small_grid.neighbors[target]:
        mapper._alpha[n] = 20.0
        mapper._beta[n] = 1.0
    a_before = float(mapper._alpha[target])
    b_before = float(mapper._beta[target])
    mapper._mrf_smooth_pass()
    assert mapper._alpha[target] == pytest.approx(a_before)
    assert mapper._beta[target] == pytest.approx(b_before)


# ---------- time decay ---------------------------------------------


def test_time_decay_floors_at_prior(small_grid, fast_options, tmp_path):
    opts = VisibilityMapperOptions(
        min_alt_deg=fast_options.min_alt_deg,
        max_runtime_min=fast_options.max_runtime_min,
        solve_timeout_s=fast_options.solve_timeout_s,
        slew_rate_deg_s=fast_options.slew_rate_deg_s,
        slew_settle_s=fast_options.slew_settle_s,
        t_observe_s=fast_options.t_observe_s,
        decay_interval_s=10.0,
        convergence_elapsed_s=fast_options.convergence_elapsed_s,
        min_run_before_user_stop_s=fast_options.min_run_before_user_stop_s,
        frontier_decay_window_s=fast_options.frontier_decay_window_s,
        failure_recent_window_s=fast_options.failure_recent_window_s,
    )
    sky = FakeSky([])
    mapper = VisibilityMapper(
        telescope_id=1,
        grid=small_grid,
        slew_func=sky.slew,
        plate_solve_func=sky.plate_solve,
        options=opts,
        state_dir=tmp_path,
    )
    target = next(i for i, c in enumerate(small_grid.cells) if not c.below_floor)
    pa = float(mapper._prior_alpha[target])
    pb = float(mapper._prior_beta[target])
    mapper._alpha[target] = 5.0
    mapper._beta[target] = 5.0
    t0 = 1000.0
    mapper._last_decay_at = t0
    # Apply many decays — should floor at the prior.
    mapper._maybe_apply_time_decay(t0 + 11.0)
    mapper._maybe_apply_time_decay(t0 + 50.0)
    mapper._maybe_apply_time_decay(t0 + 200.0)
    mapper._maybe_apply_time_decay(t0 + 5000.0)
    assert mapper._alpha[target] >= pa
    assert mapper._beta[target] >= pb
    # And high counts decayed substantially.
    assert mapper._alpha[target] <= 5.0
    assert mapper._beta[target] <= 5.0


def test_time_decay_reduces_failure_count(small_grid, fast_options, tmp_path):
    opts = VisibilityMapperOptions(
        min_alt_deg=fast_options.min_alt_deg,
        max_runtime_min=fast_options.max_runtime_min,
        solve_timeout_s=fast_options.solve_timeout_s,
        slew_rate_deg_s=fast_options.slew_rate_deg_s,
        slew_settle_s=fast_options.slew_settle_s,
        t_observe_s=fast_options.t_observe_s,
        decay_interval_s=1.0,
        convergence_elapsed_s=fast_options.convergence_elapsed_s,
        min_run_before_user_stop_s=fast_options.min_run_before_user_stop_s,
        frontier_decay_window_s=fast_options.frontier_decay_window_s,
        failure_recent_window_s=fast_options.failure_recent_window_s,
    )
    sky = FakeSky([])
    mapper = VisibilityMapper(
        telescope_id=1,
        grid=small_grid,
        slew_func=sky.slew,
        plate_solve_func=sky.plate_solve,
        options=opts,
        state_dir=tmp_path,
    )
    target = next(i for i, c in enumerate(small_grid.cells) if not c.below_floor)
    mapper._failure_count[target] = 3
    mapper._last_decay_at = 0.0
    mapper._maybe_apply_time_decay(2.0)
    assert mapper._failure_count[target] == 2
    mapper._maybe_apply_time_decay(4.0)
    assert mapper._failure_count[target] == 1
    mapper._maybe_apply_time_decay(6.0)
    assert mapper._failure_count[target] == 0
    mapper._maybe_apply_time_decay(8.0)
    # Already at zero — no underflow.
    assert mapper._failure_count[target] == 0


# ---------- acquisition --------------------------------------------


def test_score_is_higher_for_higher_entropy(small_grid, fast_options, tmp_path):
    sky = FakeSky([])
    mapper = VisibilityMapper(
        telescope_id=1,
        grid=small_grid,
        slew_func=sky.slew,
        plate_solve_func=sky.plate_solve,
        options=fast_options,
        state_dir=tmp_path,
    )
    # Find two active cells.
    actives = [
        i
        for i, c in enumerate(small_grid.cells)
        if not c.below_floor and 30 < c.alt_deg < 50
    ][:2]
    a, b = actives
    # Make `a` very confident (low EIG); `b` uncertain (high EIG).
    mapper._alpha[a] = 20.0
    mapper._beta[a] = 1.0
    mapper._alpha[b] = 1.0
    mapper._beta[b] = 1.0
    # No frontier neighbors, no slew bonus → score should reflect EIG.
    s_a = mapper._score_candidate(a, 0.0, 90.0, 0.0)
    s_b = mapper._score_candidate(b, 0.0, 90.0, 0.0)
    assert s_b > s_a


def test_frontier_bonus_decays(small_grid, fast_options, tmp_path):
    opts = VisibilityMapperOptions(
        **{**fast_options.__dict__, "frontier_decay_window_s": 100.0}
    )
    sky = FakeSky([])
    mapper = VisibilityMapper(
        telescope_id=1,
        grid=small_grid,
        slew_func=sky.slew,
        plate_solve_func=sky.plate_solve,
        options=opts,
        state_dir=tmp_path,
    )
    assert mapper._frontier_bonus(0.0) == pytest.approx(3.0)
    # Mid-decay
    mid = mapper._frontier_bonus(50.0)
    assert 0.5 < mid < 3.0
    # After decay window
    assert mapper._frontier_bonus(200.0) == pytest.approx(0.5)


def test_frontier_bonus_prefers_neighbors_of_visible(
    small_grid, fast_options, tmp_path
):
    sky = FakeSky([])
    mapper = VisibilityMapper(
        telescope_id=1,
        grid=small_grid,
        slew_func=sky.slew,
        plate_solve_func=sky.plate_solve,
        options=fast_options,
        state_dir=tmp_path,
    )
    # Find a cell with at least one neighbor (active).
    target = next(
        i
        for i, c in enumerate(small_grid.cells)
        if not c.below_floor
        and any(not small_grid.cells[n].below_floor for n in small_grid.neighbors[i])
    )
    # Score with no visible neighbors.
    s_no_frontier = mapper._score_candidate(target, 0.0, 90.0, frontier_bonus=3.0)
    # Make all neighbors visible.
    for n in small_grid.neighbors[target]:
        if small_grid.cells[n].below_floor:
            continue
        mapper._alpha[n] = 20.0
        mapper._beta[n] = 1.0
    s_with_frontier = mapper._score_candidate(target, 0.0, 90.0, frontier_bonus=3.0)
    assert s_with_frontier > s_no_frontier


# ---------- candidate selection ------------------------------------


def test_candidate_selection_excludes_recently_failed(
    small_grid, fast_options, tmp_path
):
    sky = FakeSky([])
    mapper = VisibilityMapper(
        telescope_id=1,
        grid=small_grid,
        slew_func=sky.slew,
        plate_solve_func=sky.plate_solve,
        options=fast_options,
        state_dir=tmp_path,
    )
    # Make all active cells low-entropy (confident) except the target,
    # so the target reliably appears in the top-K. Priors otherwise
    # tie at log(2) and argpartition picks an arbitrary subset.
    for i, c in enumerate(small_grid.cells):
        if c.below_floor:
            continue
        mapper._alpha[i] = 20.0
        mapper._beta[i] = 1.0
    target = next(
        i
        for i, c in enumerate(small_grid.cells)
        if not c.below_floor and 30 < c.alt_deg < 60
    )
    mapper._alpha[target] = 1.0
    mapper._beta[target] = 1.0
    now = 1000.0
    mapper._last_failure_at[target] = now
    mapper._failure_count[target] = 1
    cands = mapper._select_candidates(now + 100)
    assert target not in cands, "cell should be excluded inside failure window"
    cands_later = mapper._select_candidates(
        now + fast_options.failure_recent_window_s + 1
    )
    assert target in cands_later, "cell should be eligible after failure window"


def test_candidate_selection_excludes_permanently_obstructed(
    small_grid, fast_options, tmp_path
):
    sky = FakeSky([])
    mapper = VisibilityMapper(
        telescope_id=1,
        grid=small_grid,
        slew_func=sky.slew,
        plate_solve_func=sky.plate_solve,
        options=fast_options,
        state_dir=tmp_path,
    )
    target = next(i for i, c in enumerate(small_grid.cells) if not c.below_floor)
    mapper._failure_count[target] = 3
    cands = mapper._select_candidates(1e9)
    assert target not in cands


def test_candidate_selection_excludes_below_floor(small_grid, fast_options, tmp_path):
    sky = FakeSky([])
    mapper = VisibilityMapper(
        telescope_id=1,
        grid=small_grid,
        slew_func=sky.slew,
        plate_solve_func=sky.plate_solve,
        options=fast_options,
        state_dir=tmp_path,
    )
    below = next(i for i, c in enumerate(small_grid.cells) if c.below_floor)
    cands = mapper._select_candidates(1000.0)
    assert below not in cands


# ---------- end-to-end loop ---------------------------------------


def test_run_clear_sky_converges_visible(tmp_path):
    """All-clear sky: after a short run, lots of cells should be in
    the visible polygon (E[p] > 0.6) and there should be many
    observations on disk."""
    grid = make_altaz_band_grid(min_alt_deg=20.0)
    sky = FakeSky([])
    opts = VisibilityMapperOptions(
        min_alt_deg=20.0,
        max_runtime_min=1.0,
        solve_timeout_s=0.001,
        slew_rate_deg_s=10000.0,
        slew_settle_s=0.0,
        t_observe_s=0.0,
        decay_interval_s=100.0,
        convergence_elapsed_s=10.0,
        min_run_before_user_stop_s=0.0,
        frontier_decay_window_s=10.0,
        failure_recent_window_s=10.0,
    )
    mapper = VisibilityMapper(
        telescope_id=1,
        grid=grid,
        slew_func=sky.slew,
        plate_solve_func=sky.plate_solve,
        options=opts,
        state_dir=tmp_path,
    )
    mapper.start()
    # Let it run briefly then stop.
    time.sleep(0.6)
    mapper.request_stop(force=True)
    mapper.join(timeout=5.0)
    status = mapper.status()
    assert status["n_observations"] > 30
    # Lots of visible cells.
    assert status["n_cells_visible"] > 20
    # Snapshot exists.
    p = visibility_map_path(tmp_path, 1)
    assert p.exists()


def test_run_obstructed_quadrant_marks_failures(tmp_path):
    """A 90°-wide block from az 90 to 180 (alt 0–90) → cells in that
    range should accumulate beta. Outside should accumulate alpha."""
    grid = make_altaz_band_grid(min_alt_deg=20.0)
    sky = FakeSky([(90.0, 180.0, 0.0, 90.0)])
    opts = VisibilityMapperOptions(
        min_alt_deg=20.0,
        max_runtime_min=1.0,
        solve_timeout_s=0.001,
        slew_rate_deg_s=10000.0,
        slew_settle_s=0.0,
        t_observe_s=0.0,
        decay_interval_s=100.0,
        convergence_elapsed_s=100.0,
        min_run_before_user_stop_s=0.0,
        frontier_decay_window_s=10.0,
        failure_recent_window_s=10.0,
    )
    mapper = VisibilityMapper(
        telescope_id=1,
        grid=grid,
        slew_func=sky.slew,
        plate_solve_func=sky.plate_solve,
        options=opts,
        state_dir=tmp_path,
    )
    mapper.start()
    time.sleep(0.8)
    mapper.request_stop(force=True)
    mapper.join(timeout=5.0)
    # Find cells inside vs outside the obstruction.
    inside_eps = []
    outside_eps = []
    for c in grid.cells:
        if c.below_floor:
            continue
        ep = float(mapper._alpha[c.idx] / (mapper._alpha[c.idx] + mapper._beta[c.idx]))
        if 90 <= c.az_deg <= 180:
            inside_eps.append(ep)
        else:
            outside_eps.append(ep)
    # The obstructed quadrant should average lower E[p].
    if inside_eps and outside_eps:
        assert sum(inside_eps) / len(inside_eps) < sum(outside_eps) / len(outside_eps)


def test_run_aborts_after_consecutive_system_errors(tmp_path):
    grid = make_altaz_band_grid(min_alt_deg=20.0)
    sky = FakeSky([], error_at_calls=[0, 1, 2, 3, 4])
    opts = VisibilityMapperOptions(
        min_alt_deg=20.0,
        max_runtime_min=10.0,
        solve_timeout_s=0.001,
        slew_rate_deg_s=10000.0,
        slew_settle_s=0.0,
        t_observe_s=0.0,
        decay_interval_s=600.0,
        convergence_elapsed_s=600.0,
        min_run_before_user_stop_s=0.0,
        frontier_decay_window_s=10.0,
        failure_recent_window_s=10.0,
    )
    mapper = VisibilityMapper(
        telescope_id=1,
        grid=grid,
        slew_func=sky.slew,
        plate_solve_func=sky.plate_solve,
        options=opts,
        state_dir=tmp_path,
    )
    mapper.start()
    mapper.join(timeout=10.0)
    status = mapper.status()
    assert status["stop_reason"] is not None
    assert status["stop_reason"]["code"] == "system_errors"


def test_user_stop_rejected_before_min_run_window(tmp_path):
    grid = make_altaz_band_grid(min_alt_deg=20.0)
    sky = FakeSky([], delay_s=0.1)
    opts = VisibilityMapperOptions(
        min_alt_deg=20.0,
        max_runtime_min=10.0,
        solve_timeout_s=1.0,
        slew_rate_deg_s=10000.0,
        slew_settle_s=0.0,
        t_observe_s=0.001,
        decay_interval_s=600.0,
        convergence_elapsed_s=600.0,
        min_run_before_user_stop_s=60.0,  # 1 minute minimum
        frontier_decay_window_s=600.0,
        failure_recent_window_s=600.0,
    )
    mapper = VisibilityMapper(
        telescope_id=1,
        grid=grid,
        slew_func=sky.slew,
        plate_solve_func=sky.plate_solve,
        options=opts,
        state_dir=tmp_path,
    )
    mapper.start()
    time.sleep(0.05)
    accepted, msg = mapper.request_stop(force=False)
    assert accepted is False
    assert "stop disabled" in msg
    accepted_force, _ = mapper.request_stop(force=True)
    assert accepted_force is True
    mapper.join(timeout=5.0)


def test_listener_receives_observation_events(tmp_path):
    grid = make_altaz_band_grid(min_alt_deg=20.0)
    sky = FakeSky([])
    opts = VisibilityMapperOptions(
        min_alt_deg=20.0,
        max_runtime_min=1.0,
        solve_timeout_s=0.001,
        slew_rate_deg_s=10000.0,
        slew_settle_s=0.0,
        t_observe_s=0.0,
        decay_interval_s=100.0,
        convergence_elapsed_s=100.0,
        min_run_before_user_stop_s=0.0,
        frontier_decay_window_s=10.0,
        failure_recent_window_s=10.0,
    )
    mapper = VisibilityMapper(
        telescope_id=1,
        grid=grid,
        slew_func=sky.slew,
        plate_solve_func=sky.plate_solve,
        options=opts,
        state_dir=tmp_path,
    )
    received: list[dict] = []
    unsub = mapper.add_listener(received.append)
    mapper.start()
    time.sleep(0.3)
    mapper.request_stop(force=True)
    mapper.join(timeout=5.0)
    unsub()
    obs_events = [e for e in received if e.get("type") == "observation"]
    assert len(obs_events) > 5
    complete_events = [e for e in received if e.get("type") == "complete"]
    assert len(complete_events) >= 1


# ---------- persistence -------------------------------------------


def test_persistence_round_trip(tmp_path):
    p = tmp_path / "vm.json"
    cells = [{"idx": 0, "alpha": 5.0, "beta": 2.0}]
    save_snapshot(
        p,
        telescope_id=1,
        grid_kind="altaz_band",
        min_alt_deg=10.0,
        cells=cells,
        started_at=100.0,
        elapsed_s=50.0,
        n_observations=12,
    )
    loaded = load_snapshot(p)
    assert loaded is not None
    assert loaded["grid_kind"] == "altaz_band"
    assert loaded["cells"] == cells


def test_persistence_skips_schema_mismatch(tmp_path):
    p = tmp_path / "vm.json"
    p.write_text('{"schema_version": 99, "cells": []}')
    assert load_snapshot(p) is None


def test_persistence_skips_garbage(tmp_path):
    p = tmp_path / "vm.json"
    p.write_text("not json")
    assert load_snapshot(p) is None


def test_persistence_resume_picks_up_priors(tmp_path):
    grid = make_altaz_band_grid(min_alt_deg=20.0)
    # Save a fake snapshot for telescope 7 with one well-observed cell.
    target_idx = next(i for i, c in enumerate(grid.cells) if not c.below_floor)
    cells_payload = []
    for i, c in enumerate(grid.cells):
        if i == target_idx:
            cells_payload.append(
                {"idx": i, "alpha": 8.0, "beta": 2.0, "failure_count": 0}
            )
        else:
            cells_payload.append(
                {"idx": i, "alpha": 1.0, "beta": 1.0, "failure_count": 0}
            )
    p = visibility_map_path(tmp_path, 7)
    save_snapshot(
        p,
        telescope_id=7,
        grid_kind="altaz_band",
        min_alt_deg=20.0,
        cells=cells_payload,
        started_at=0.0,
        elapsed_s=600.0,
        n_observations=42,
    )
    sky = FakeSky([])
    mapper = VisibilityMapper(
        telescope_id=7,
        grid=grid,
        slew_func=sky.slew,
        plate_solve_func=sky.plate_solve,
        options=VisibilityMapperOptions(min_alt_deg=20.0),
        state_dir=tmp_path,
    )
    assert mapper._alpha[target_idx] == pytest.approx(8.0)
    assert mapper._beta[target_idx] == pytest.approx(2.0)


# ---------- manager -----------------------------------------------


def test_manager_refuses_concurrent_starts(tmp_path):
    mgr = VisibilityMapManager()
    grid = make_altaz_band_grid(min_alt_deg=20.0)
    sky = FakeSky([], delay_s=0.05)
    opts = VisibilityMapperOptions(
        min_alt_deg=20.0,
        max_runtime_min=1.0,
        solve_timeout_s=1.0,
        slew_rate_deg_s=10000.0,
        slew_settle_s=0.0,
        t_observe_s=0.001,
        min_run_before_user_stop_s=0.0,
    )
    m1 = VisibilityMapper(
        telescope_id=1,
        grid=grid,
        slew_func=sky.slew,
        plate_solve_func=sky.plate_solve,
        options=opts,
        state_dir=tmp_path,
    )
    mgr.start(m1)
    try:
        m2 = VisibilityMapper(
            telescope_id=1,
            grid=grid,
            slew_func=sky.slew,
            plate_solve_func=sky.plate_solve,
            options=opts,
            state_dir=tmp_path,
        )
        with pytest.raises(RuntimeError):
            mgr.start(m2)
    finally:
        m1.request_stop(force=True)
        m1.join(timeout=5.0)
