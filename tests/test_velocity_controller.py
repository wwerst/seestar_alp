"""Unit tests for device.velocity_controller — exception discipline (PR-D).

Covers the three review-report fixes:

- P1-4: ``_motor_stop_on_exit`` always issues motor-stop, even via
  exception, and bypasses the lockout-aware ``speed_move`` wrapper so a
  ``SunSafetyLocked``-mid-loop does not also block the cleanup.
- P1-7: ``unwrap_az_series`` rejects non-finite (NaN, Inf) samples.
- P1-8: ``set_tracking`` logs a WARNING when the underlying RPC fails
  (was: silently swallowed).
"""

from __future__ import annotations

import logging
import math

import pytest

from device.velocity_controller import (
    _motor_stop_on_exit,
    set_tracking,
    unwrap_az_series,
)


class _FakeCli:
    """Minimal mount-client double — records every method_sync call."""

    def __init__(self, *, raise_on=None):
        self.calls: list[tuple[str, object]] = []
        self._raise_on = raise_on

    def method_sync(self, method: str, params=None):
        self.calls.append((method, params))
        if self._raise_on is not None and method == self._raise_on:
            raise RuntimeError(f"simulated RPC failure on {method}")
        return {"result": "ok"}


# --- P1-4: _motor_stop_on_exit -------------------------------------------


def test_motor_stop_on_exit_runs_on_clean_exit():
    cli = _FakeCli()
    with _motor_stop_on_exit(cli):
        pass
    assert cli.calls == [
        ("scope_speed_move", {"speed": 0, "angle": 0, "dur_sec": 1}),
    ]


def test_motor_stop_on_exit_runs_on_exception_and_propagates():
    """Exception inside the context must (a) propagate and (b) leave
    the motor-stop call recorded."""
    cli = _FakeCli()
    with pytest.raises(RuntimeError, match="boom"):
        with _motor_stop_on_exit(cli):
            raise RuntimeError("boom")
    assert cli.calls == [
        ("scope_speed_move", {"speed": 0, "angle": 0, "dur_sec": 1}),
    ]


def test_motor_stop_on_exit_swallows_cleanup_failure():
    """If the motor-stop ITSELF fails, the cleanup must not raise — the
    original exception (if any) must propagate untouched, and a warning
    is logged."""
    cli = _FakeCli(raise_on="scope_speed_move")

    # Clean path: cleanup raises but is swallowed.
    with _motor_stop_on_exit(cli):
        pass
    # We expect the call to have been attempted.
    assert cli.calls == [
        ("scope_speed_move", {"speed": 0, "angle": 0, "dur_sec": 1}),
    ]


def test_motor_stop_on_exit_swallows_cleanup_failure_on_exception(caplog):
    """When BOTH the body raises AND the cleanup raises, the body's
    exception must propagate (the cleanup failure is logged, not
    re-raised)."""
    cli = _FakeCli(raise_on="scope_speed_move")
    with caplog.at_level(logging.WARNING, logger="device.velocity_controller"):
        with pytest.raises(RuntimeError, match="primary"):
            with _motor_stop_on_exit(cli):
                raise RuntimeError("primary")
    # The cleanup attempt was made.
    assert cli.calls == [
        ("scope_speed_move", {"speed": 0, "angle": 0, "dur_sec": 1}),
    ]
    # And a warning was logged about the cleanup failure.
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert any("motor-stop" in r.getMessage() for r in warnings)


def test_motor_stop_bypasses_sun_safety_lockout(monkeypatch):
    """The cleanup motor-stop must call cli.method_sync DIRECTLY, NOT
    via the lockout-aware speed_move wrapper. Otherwise a SunSafetyLocked
    raised mid-loop would also block the cleanup itself.

    Verify by setting the global lockout and checking that the cleanup
    still records the method_sync call (would raise SunSafetyLocked
    otherwise).
    """
    from device import sun_safety as ss

    # Force the locked-out predicate to True.
    monkeypatch.setattr(ss, "sun_safety_is_locked_out", lambda: True)

    cli = _FakeCli()
    with _motor_stop_on_exit(cli):
        pass
    # If the wrapper had been used, we would NOT see a recorded call
    # (SunSafetyLocked would be raised before method_sync). Direct path
    # bypasses that check.
    assert cli.calls == [
        ("scope_speed_move", {"speed": 0, "angle": 0, "dur_sec": 1}),
    ]


# --- H1a: skip the raw stop while a sun-safety jog is in progress ---------


def test_motor_stop_on_exit_skipped_during_sun_safety_jog(monkeypatch, caplog):
    """While the SunSafetyMonitor's emergency jog-away is executing, the
    context-exit raw stop must be SKIPPED — issuing it would land ~tens of
    ms after the jog and truncate it, stranding the mount inside the sun
    cone. The jog's firmware dur_sec bounds the motion, so skipping is safe.
    """
    from device import sun_safety as ss

    monkeypatch.setattr(ss, "sun_safety_jog_in_progress", lambda tid=None: True)

    cli = _FakeCli()
    with caplog.at_level(logging.WARNING, logger="device.velocity_controller"):
        with _motor_stop_on_exit(cli):
            pass
    assert cli.calls == [], "no stop may be issued while a jog is in progress"
    assert any(
        "jog in progress" in r.getMessage()
        for r in caplog.records
        if r.levelno == logging.WARNING
    )


def test_motor_stop_on_exit_skipped_during_jog_on_exception(monkeypatch):
    """The SunSafetyLocked that drives this exit path is raised while the
    jog is in progress: the stop is skipped, but the exception still
    propagates unchanged."""
    from device import sun_safety as ss

    monkeypatch.setattr(ss, "sun_safety_jog_in_progress", lambda tid=None: True)

    cli = _FakeCli()
    with pytest.raises(RuntimeError, match="sun locked"):
        with _motor_stop_on_exit(cli):
            raise RuntimeError("sun locked")
    assert cli.calls == []


def test_motor_stop_on_exit_issued_when_no_jog(monkeypatch):
    """With no jog in progress the stop is issued exactly as before."""
    from device import sun_safety as ss

    monkeypatch.setattr(ss, "sun_safety_jog_in_progress", lambda tid=None: False)

    cli = _FakeCli()
    with _motor_stop_on_exit(cli):
        pass
    assert cli.calls == [
        ("scope_speed_move", {"speed": 0, "angle": 0, "dur_sec": 1}),
    ]


# --- P1-7: unwrap_az_series rejects non-finite ---------------------------


def test_unwrap_az_series_rejects_nan():
    with pytest.raises(ValueError, match="non-finite sample at index 1"):
        unwrap_az_series([0.0, float("nan"), 10.0])


def test_unwrap_az_series_rejects_inf():
    with pytest.raises(ValueError, match="non-finite sample"):
        unwrap_az_series([float("inf"), 0.0, 10.0])


def test_unwrap_az_series_rejects_neg_inf():
    with pytest.raises(ValueError, match="non-finite sample at index 2"):
        unwrap_az_series([0.0, 1.0, -math.inf])


def test_unwrap_az_series_finite_inputs_unchanged():
    # Finite inputs still produce the canonical unwrapped series.
    out = unwrap_az_series([0.0, 10.0, -179.0, 175.0])
    assert out[0] == pytest.approx(0.0)
    assert out[1] == pytest.approx(10.0)
    # The 10° → -179° step is a +171° wrapped delta (10 → 181 → -179).
    assert out[2] == pytest.approx(10.0 + (-179.0 - 10.0 + 360.0))  # = 181.0
    # And -179° → 175° is a -6° wrapped delta.
    assert out[3] == pytest.approx(out[2] - 6.0)


def test_unwrap_az_series_empty_returns_empty():
    assert unwrap_az_series([]) == []


# --- P1-8: set_tracking logs warning on RPC failure ----------------------


def test_set_tracking_logs_warning_on_failure(caplog):
    cli = _FakeCli(raise_on="scope_set_track_state")
    with caplog.at_level(logging.WARNING, logger="device.velocity_controller"):
        # Must not raise — the call is advisory, but the failure must
        # be visible.
        set_tracking(cli, True)
    assert any(
        r.levelno == logging.WARNING and "set_tracking(True)" in r.getMessage()
        for r in caplog.records
    )


def test_set_tracking_silent_on_success():
    cli = _FakeCli()
    set_tracking(cli, False)
    assert cli.calls == [("scope_set_track_state", False)]


# --- Finding #4: measure_altaz_timed guards non-finite encoder readings ---


class _HorizCli:
    """Client that returns a scripted sequence of scope_get_horiz_coord
    payloads (cycling on the last one)."""

    def __init__(self, payloads):
        self._payloads = list(payloads)
        self.calls = 0

    def method_sync(self, method, params=None):
        assert method == "scope_get_horiz_coord"
        idx = min(self.calls, len(self._payloads) - 1)
        self.calls += 1
        return self._payloads[idx]


def test_measure_altaz_timed_finite_ok():
    from device.velocity_controller import measure_altaz_timed

    cli = _HorizCli([{"result": [12.0, 190.0], "Timestamp": 3.5}])
    alt, az, fw_t = measure_altaz_timed(cli, loc=None)
    assert alt == 12.0
    assert az == pytest.approx(-170.0)  # 190° wrapped into [-180, +180)
    assert fw_t == 3.5


def test_measure_altaz_timed_retries_then_raises_on_non_finite():
    from device.velocity_controller import measure_altaz_timed

    cli = _HorizCli([{"result": [float("nan"), 0.0]}])
    with pytest.raises(RuntimeError, match="unexpected payload"):
        measure_altaz_timed(cli, loc=None)
    assert cli.calls == 2  # one retry before raising


def test_measure_altaz_timed_recovers_on_retry():
    from device.velocity_controller import measure_altaz_timed

    cli = _HorizCli(
        [
            {"result": [float("inf"), 0.0]},  # first sample non-finite
            {"result": [10.0, 20.0], "Timestamp": 1.0},  # retry is good
        ]
    )
    alt, az, fw_t = measure_altaz_timed(cli, loc=None)
    assert alt == 10.0
    assert az == pytest.approx(20.0)
    assert fw_t == 1.0
    assert cli.calls == 2


# --- Finding #1: unwind_azimuth drives cumulative az to center -----------


def test_unwind_azimuth_drives_cumulative_to_center(monkeypatch):
    """From a wound-up cum=+300°, unwind must plan toward cumulative 0 —
    NOT the +360° that pick_cum_target's shorter-wrapped path would choose
    (which winds further toward the hard stop)."""
    from device import velocity_controller as vc
    from device.plant_limits import (
        AzimuthLimits,
        CumulativeAzTracker,
        pick_cum_target,
    )

    limits = AzimuthLimits(
        ccw_hard_stop_cum_deg=-450.0,
        cw_hard_stop_cum_deg=+450.0,
        padding_deg=15.0,
    )
    wrapped_now = vc.wrap_pm180(300.0)  # -60.0
    tracker = CumulativeAzTracker()
    tracker.reset(cum_az_deg=+300.0, wrapped_az_deg=wrapped_now)

    captured: dict = {}

    def fake_move_to_ff(cli, **kwargs):
        captured.update(kwargs)
        return 0.0, 0.0, {"final_residual_deg": 0.0}

    monkeypatch.setattr(vc, "move_to_ff", fake_move_to_ff)

    cli = _HorizCli([{"result": [5.0, wrapped_now], "Timestamp": 0.0}])
    vc.unwind_azimuth(
        cli, loc=None, az_tracker=tracker, az_limits=limits, threshold_deg=180.0
    )

    # The plan is forced to cumulative 0, bypassing pick_cum_target.
    assert captured["force_cum_az_target"] == 0.0
    # Elevation is held where it is (an az unwind must not move el).
    assert captured["target_el_deg"] == 5.0
    assert captured["cur_el_deg"] == 5.0
    # Sanity: the old wrapped-target path WOULD have wound further out.
    wrapped_target = vc.wrap_pm180(wrapped_now - tracker.cum_az_deg)
    buggy = pick_cum_target(tracker.cum_az_deg, wrapped_now, wrapped_target, limits)
    assert buggy == pytest.approx(360.0)  # toward the hard stop, not 0


def test_unwind_azimuth_noop_within_threshold(monkeypatch):
    # Already near center: no motion planned, informational stats returned.
    from device import velocity_controller as vc
    from device.plant_limits import AzimuthLimits, CumulativeAzTracker

    def fail_move_to_ff(*args, **kwargs):
        raise AssertionError("no motion expected when within threshold")

    monkeypatch.setattr(vc, "move_to_ff", fail_move_to_ff)

    limits = AzimuthLimits(
        ccw_hard_stop_cum_deg=-450.0, cw_hard_stop_cum_deg=+450.0, padding_deg=15.0
    )
    tracker = CumulativeAzTracker()
    tracker.reset(cum_az_deg=+30.0, wrapped_az_deg=30.0)

    _, _, stats = vc.unwind_azimuth(
        cli=None, loc=None, az_tracker=tracker, az_limits=limits, threshold_deg=180.0
    )
    assert stats["controller"] == "unwind_noop"
    assert stats["commands_issued"] == 0


# --- L2: scripts/tune_vc.py drops dead PD/predictor-era CLI knobs ----------


def _load_tune_vc():
    """Load scripts/tune_vc.py as a module (it lives outside any package).
    Registers it in sys.modules first so its @dataclass can resolve its own
    module namespace."""
    import importlib.util
    import os
    import sys

    if "tune_vc" in sys.modules:
        return sys.modules["tune_vc"]
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "scripts",
        "tune_vc.py",
    )
    spec = importlib.util.spec_from_file_location("tune_vc", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["tune_vc"] = mod
    spec.loader.exec_module(mod)
    return mod


def _parse_tune_vc_args(monkeypatch, argv):
    """Invoke tune_vc.main() with argv, stubbing out all device I/O, and
    return (rc, namespace) where namespace is the argparse Namespace that
    main() dispatched to the mode's run_* function."""
    import sys

    mod = _load_tune_vc()
    captured: dict = {}

    def _capture(cli, args):
        captured["args"] = args
        return 0

    monkeypatch.setattr(mod, "run_setpoints", _capture)
    monkeypatch.setattr(mod, "run_step_response", _capture)
    monkeypatch.setattr(mod, "run_diagonal", _capture)
    monkeypatch.setattr(mod, "_print_latency_summary", lambda cli: None)
    monkeypatch.setattr(mod, "InstrumentedAlpacaClient", lambda *a, **k: object())
    monkeypatch.setattr(mod.Config, "load_toml", lambda: None, raising=False)
    monkeypatch.setattr(mod.Config, "init_lat", 1.0, raising=False)
    monkeypatch.setattr(mod.Config, "init_long", 2.0, raising=False)
    monkeypatch.setattr(sys, "argv", ["tune_vc.py", *argv])
    rc = mod.main()
    return rc, captured.get("args")


@pytest.mark.parametrize(
    "dead_flag",
    [
        "--kd",
        "--tau",
        "--min-speed",
        "--fine-min-speed",
        "--use-predictor",
        "--no-predictor",
    ],
)
def test_tune_vc_rejects_dead_pd_era_flags(monkeypatch, dead_flag):
    """The PD/predictor-era knobs were removed because move_azimuth_to_ff
    ignores them; argparse must now reject them outright."""
    with pytest.raises(SystemExit):
        _parse_tune_vc_args(monkeypatch, [dead_flag, "0.5"])


def test_tune_vc_keeps_ff_knobs_and_drops_pd_attrs(monkeypatch):
    """The FF+FB knobs actually wired into move_azimuth_to_ff stay and parse
    through; the removed PD/predictor-era knobs leave no attribute behind on
    the parsed namespace (guards against silent re-introduction)."""
    rc, args = _parse_tune_vc_args(
        monkeypatch,
        [
            "--control",
            "feedforward",
            "--kp-pos",
            "0.7",
            "--v-corr-max",
            "1.5",
            "--max-rate",
            "4.0",
            "--a-max",
            "8.0",
            "--j-max",
            "30.0",
            "--profile",
            "scurve",
        ],
    )
    assert rc == 0
    # Retained knobs are present and parsed to the values move_azimuth_to_ff
    # consumes.
    assert args.kp_pos == 0.7
    assert args.v_corr_max == 1.5
    assert args.max_rate == 4.0
    assert args.a_max == 8.0
    # Dead PD/predictor-era knobs leave no namespace attribute.
    for attr in (
        "kp",
        "kd",
        "tau",
        "min_speed",
        "fine_min_speed",
        "fine_thresh_factor",
        "max_halvings",
        "use_predictor",
    ):
        assert not hasattr(args, attr), f"dead PD-era arg still present: {attr}"
