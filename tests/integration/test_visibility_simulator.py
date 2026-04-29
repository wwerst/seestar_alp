"""End-to-end integration test for the sky-visibility mapper.

Exercises the full path: HTTP /start → mapper thread → slew callback
→ plate-solve callback → posterior update → /status / /cells. The
plate-solve and slew callbacks are wired to a small in-process
simulator that always returns SOLVED, so we just verify the loop
runs and produces a non-empty visible polygon.

Marked ``integration`` so it's only included in the integration lane.
"""

from __future__ import annotations

import time

import falcon
import pytest
from falcon import testing

import front.app as front_app
from device.visibility_mapper import get_visibility_manager


pytestmark = pytest.mark.integration


def _build_visibility_test_app():
    """Minimal Falcon app with just the visibility-map routes."""
    app = falcon.App()
    app.add_route(
        "/{telescope_id:int}/sky_visibility",
        front_app.SkyVisibilityResource(),
    )
    app.add_route(
        "/api/{telescope_id:int}/calibrate_visibility/start",
        front_app.VisibilityStartResource(),
    )
    app.add_route(
        "/api/{telescope_id:int}/calibrate_visibility/status",
        front_app.VisibilityStatusResource(),
    )
    app.add_route(
        "/api/{telescope_id:int}/calibrate_visibility/cells",
        front_app.VisibilityCellsResource(),
    )
    app.add_route(
        "/api/{telescope_id:int}/calibrate_visibility/stop",
        front_app.VisibilityStopResource(),
    )
    app.add_route(
        "/api/{telescope_id:int}/calibrate_visibility/force_stop",
        front_app.VisibilityForceStopResource(),
    )
    return app


@pytest.fixture
def visibility_app(monkeypatch, tmp_path):
    """Build the test app with the device layer mocked.

    `do_action_device` returns canned plate-solve results so the mapper
    sees SOLVED on every observation.  GPS returns a stable lat/lon so
    the alt/az ↔ RA/Dec conversion in the slew callback can run.
    """
    monkeypatch.setattr(front_app, "_VISIBILITY_STATE_DIR", tmp_path)

    class _FakeAlpacaClient:
        def __init__(self, *args, **kwargs):
            pass

    monkeypatch.setattr(front_app, "AlpacaClient", _FakeAlpacaClient, raising=False)
    monkeypatch.setattr(
        "scripts.trajectory.observer.fetch_telescope_lonlat",
        lambda _cli: (34.0, -118.0),
    )

    def _do_action_device(action, dev_num, parameters, is_schedule=False):
        # Slew: return a benign success.
        if action == "method_sync" and isinstance(parameters, dict):
            method = parameters.get("method")
            if method == "scope_goto":
                return {
                    "ErrorNumber": 0,
                    "ErrorMessage": "",
                    "Value": {"result": 0},
                }
            return {"ErrorNumber": 0, "ErrorMessage": "", "Value": {}}
        # Plate solve: simulate a solved result.
        if action == "start_solve_sync":
            return {
                "ErrorNumber": 0,
                "ErrorMessage": "",
                "Value": {
                    "ra_dec": [12.0, 30.0],
                    "fov": [1.27, 0.71],
                    "angle": 0.0,
                    "star_number": 800,
                },
            }
        return {"ErrorNumber": 0, "ErrorMessage": "", "Value": None}

    monkeypatch.setattr(front_app, "do_action_device", _do_action_device)
    monkeypatch.setattr(front_app, "check_api_state", lambda _tid: True)

    app = _build_visibility_test_app()
    client = testing.TestClient(app)
    yield client
    # Teardown: stop any spawned mappers.
    mgr = get_visibility_manager()
    for tid in list(mgr._mappers.keys()):
        m = mgr._mappers.get(tid)
        if m is not None and m.is_active():
            m.request_stop(force=True)
            m.join(timeout=10.0)
        mgr.clear_inactive(tid)


def _wait_for_observations(client, tid, *, min_obs=5, timeout=15.0):
    """Poll /status until n_observations >= min_obs or timeout."""
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        r = client.simulate_get(f"/api/{tid}/calibrate_visibility/status")
        last = r.json
        if last.get("n_observations", 0) >= min_obs:
            return last
        time.sleep(0.2)
    return last


def test_e2e_start_runs_observations_and_stops(visibility_app):
    tid = 7
    # Start the mapper.
    r = visibility_app.simulate_post(
        f"/api/{tid}/calibrate_visibility/start",
        json={"min_alt": 30.0, "max_runtime_min": 5.0},
    )
    assert r.status == falcon.HTTP_200, r.text
    body = r.json
    assert body["active"] is True
    assert body["telescope_id"] == tid
    # Let it run until at least 5 observations are recorded. With
    # SOLVED-on-every-call, the loop should make rapid progress.
    status = _wait_for_observations(visibility_app, tid, min_obs=5, timeout=20.0)
    assert status is not None
    assert status["n_observations"] >= 5, status
    # Force stop.
    r = visibility_app.simulate_post(f"/api/{tid}/calibrate_visibility/force_stop")
    assert r.status == falcon.HTTP_200
    # Wait for mapper to finalize.
    deadline = time.time() + 5.0
    while time.time() < deadline:
        s = visibility_app.simulate_get(f"/api/{tid}/calibrate_visibility/status").json
        if not s.get("active"):
            break
        time.sleep(0.1)
    final = visibility_app.simulate_get(f"/api/{tid}/calibrate_visibility/status").json
    assert final["active"] is False
    # We've been seeing SOLVED on every observation, so cells with
    # n_obs > 0 should have alpha well above the prior. Verify the
    # visible polygon is non-empty.
    cells_resp = visibility_app.simulate_get(f"/api/{tid}/calibrate_visibility/cells")
    assert cells_resp.status == falcon.HTTP_200
    cells = cells_resp.json["cells"]
    visible = [c for c in cells if not c["below_floor"] and c["ep"] > 0.6]
    assert len(visible) > 0, (
        f"no visible cells; observed n_observations={final['n_observations']}, "
        f"sample cells={cells[:5]}"
    )


def test_e2e_status_when_no_run(visibility_app):
    r = visibility_app.simulate_get("/api/8/calibrate_visibility/status")
    assert r.status == falcon.HTTP_200
    assert r.json == {"active": False}


def test_e2e_concurrent_start_returns_409(visibility_app):
    tid = 9
    r1 = visibility_app.simulate_post(
        f"/api/{tid}/calibrate_visibility/start",
        json={"min_alt": 20.0, "max_runtime_min": 5.0},
    )
    assert r1.status == falcon.HTTP_200
    r2 = visibility_app.simulate_post(
        f"/api/{tid}/calibrate_visibility/start",
        json={"min_alt": 20.0, "max_runtime_min": 5.0},
    )
    assert r2.status == falcon.HTTP_409
    visibility_app.simulate_post(f"/api/{tid}/calibrate_visibility/force_stop")
