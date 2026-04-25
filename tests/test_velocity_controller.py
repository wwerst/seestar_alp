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
