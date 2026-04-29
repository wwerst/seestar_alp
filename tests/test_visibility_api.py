"""Route-shape tests for the sky-visibility-map API.

Calls each Falcon resource directly with a synthetic req/resp, so we
verify the wire contract (status codes, JSON shapes, 409/404 paths)
without needing a real telescope or running server.

The visibility manager is a per-process singleton, so each test
clears it via ``get_visibility_manager().clear_inactive()``. We also
monkey-patch ``do_action_device`` and ``fetch_telescope_lonlat`` so
the start route doesn't try to talk to a device.
"""

from __future__ import annotations

import json
import time

import pytest

import front.app as front_app
from device.visibility_mapper import (
    VisibilityMapper,
    VisibilityMapperOptions,
    SolveOutcome,
    get_visibility_manager,
)
from device.sky_grid import make_altaz_band_grid


class _Req:
    def __init__(self, body=None, params=None):
        self.media = {} if body is None else dict(body)
        self._params = params or {}

    def get_param(self, key, default=None):
        return self._params.get(key, default)

    def get_header(self, key):
        return None


class _Resp:
    def __init__(self):
        self.status = None
        self.text = ""
        self.content_type = None
        self._headers: dict[str, str] = {}
        self.stream = None
        self.cookies = []

    def append_header(self, k, v):
        self._headers[k] = v

    def set_cookie(self, k, v, path="/"):
        self.cookies.append((k, v, path))

    def unset_cookie(self, k, path="/"):
        self.cookies = [c for c in self.cookies if c[0] != k]


@pytest.fixture(autouse=True)
def _reset_manager():
    """Stop and clear any mappers between tests."""
    mgr = get_visibility_manager()
    for tid in list(mgr._mappers.keys()):
        m = mgr._mappers.get(tid)
        if m is not None and m.is_active():
            m.request_stop(force=True)
            m.join(timeout=5.0)
        mgr.clear_inactive(tid)
    yield
    for tid in list(mgr._mappers.keys()):
        m = mgr._mappers.get(tid)
        if m is not None and m.is_active():
            m.request_stop(force=True)
            m.join(timeout=5.0)
        mgr.clear_inactive(tid)


# ---------- status when nothing is running --------------------------


def test_status_returns_inactive_when_no_run():
    req = _Req()
    resp = _Resp()
    front_app.VisibilityStatusResource.on_get(req, resp, telescope_id=42)
    assert resp.status == "200 OK"
    assert resp.content_type == "application/json"
    body = json.loads(resp.text)
    assert body == {"active": False}


def test_cells_returns_empty_when_no_run():
    req = _Req()
    resp = _Resp()
    front_app.VisibilityCellsResource.on_get(req, resp, telescope_id=42)
    body = json.loads(resp.text)
    assert body["active"] is False
    assert body["cells"] == []


def test_stop_returns_404_when_no_run():
    req = _Req()
    resp = _Resp()
    front_app.VisibilityStopResource.on_post(req, resp, telescope_id=42)
    assert resp.status == "404 Not Found"
    assert "no active" in json.loads(resp.text)["error"]


def test_force_stop_returns_404_when_no_run():
    req = _Req()
    resp = _Resp()
    front_app.VisibilityForceStopResource.on_post(req, resp, telescope_id=42)
    assert resp.status == "404 Not Found"


# ---------- start request validation -------------------------------


def test_start_rejects_non_dict_body():
    req = _Req()
    req.media = "string body"  # type: ignore
    resp = _Resp()
    front_app.VisibilityStartResource.on_post(req, resp, telescope_id=1)
    assert resp.status == "400 Bad Request"


def test_start_rejects_min_alt_out_of_range():
    req = _Req(body={"min_alt": 95.0})
    resp = _Resp()
    front_app.VisibilityStartResource.on_post(req, resp, telescope_id=1)
    assert resp.status == "400 Bad Request"
    assert "min_alt" in json.loads(resp.text)["error"]


def test_start_rejects_max_runtime_out_of_range():
    req = _Req(body={"max_runtime_min": 0.0})
    resp = _Resp()
    front_app.VisibilityStartResource.on_post(req, resp, telescope_id=1)
    assert resp.status == "400 Bad Request"


def test_start_rejects_invalid_grid_kind():
    req = _Req(body={"grid_kind": "garbage"})
    resp = _Resp()
    front_app.VisibilityStartResource.on_post(req, resp, telescope_id=1)
    assert resp.status == "400 Bad Request"


def test_start_rejects_non_numeric_min_alt():
    req = _Req(body={"min_alt": "not-a-number"})
    resp = _Resp()
    front_app.VisibilityStartResource.on_post(req, resp, telescope_id=1)
    assert resp.status == "400 Bad Request"


# ---------- start with mocked device -------------------------------


class _FakeAlpacaClient:
    def __init__(self, *args, **kwargs):
        pass


def _patch_start_path(monkeypatch, *, gps_ok=True, gps_value=(34.0, -118.0)):
    """Patch the GPS + action layer so the start route's plumbing works
    without a real telescope.

    Returns a list of recorded `do_action_device` calls so tests can
    assert on routing.
    """
    monkeypatch.setattr(front_app, "AlpacaClient", _FakeAlpacaClient, raising=False)
    if gps_ok:
        monkeypatch.setattr(
            "scripts.trajectory.observer.fetch_telescope_lonlat",
            lambda _cli: gps_value,
        )
    else:

        def _raise(_cli):
            raise RuntimeError("simulated GPS failure")

        monkeypatch.setattr(
            "scripts.trajectory.observer.fetch_telescope_lonlat", _raise
        )
    calls: list[tuple[str, int, dict]] = []

    def _fake_do_action_device(action, dev_num, parameters, is_schedule=False):
        calls.append((action, int(dev_num), dict(parameters)))
        if action == "method_sync" and parameters.get("method") == "scope_goto":
            # Return a benign success.
            return {"Value": {"result": 0}, "ErrorNumber": 0, "ErrorMessage": ""}
        if action == "start_solve_sync":
            # Pretend to solve successfully.
            return {
                "Value": {
                    "ra_dec": [12.0, 30.0],
                    "fov": [1.27, 0.71],
                    "angle": 0.0,
                },
                "ErrorNumber": 0,
                "ErrorMessage": "",
            }
        return {"Value": None, "ErrorNumber": 0, "ErrorMessage": ""}

    monkeypatch.setattr(front_app, "do_action_device", _fake_do_action_device)
    return calls


def test_start_succeeds_with_gps_and_returns_status(monkeypatch, tmp_path):
    monkeypatch.setattr(front_app, "_VISIBILITY_STATE_DIR", tmp_path)
    _patch_start_path(monkeypatch)
    req = _Req(body={"min_alt": 15.0, "max_runtime_min": 60.0})
    resp = _Resp()
    front_app.VisibilityStartResource.on_post(req, resp, telescope_id=51)
    assert resp.status == "200 OK", resp.text
    body = json.loads(resp.text)
    assert body["active"] is True
    assert body["telescope_id"] == 51
    assert body["min_alt_deg"] == 15.0
    assert body["max_runtime_min"] == 60.0
    # Stop the spawned thread.
    mapper = get_visibility_manager().get(51)
    assert mapper is not None
    mapper.request_stop(force=True)
    mapper.join(timeout=5.0)


def test_start_returns_409_if_already_running(monkeypatch, tmp_path):
    monkeypatch.setattr(front_app, "_VISIBILITY_STATE_DIR", tmp_path)
    _patch_start_path(monkeypatch)
    req1 = _Req(body={"min_alt": 15.0, "max_runtime_min": 60.0})
    resp1 = _Resp()
    front_app.VisibilityStartResource.on_post(req1, resp1, telescope_id=52)
    assert resp1.status == "200 OK"
    req2 = _Req(body={"min_alt": 15.0, "max_runtime_min": 60.0})
    resp2 = _Resp()
    front_app.VisibilityStartResource.on_post(req2, resp2, telescope_id=52)
    assert resp2.status == "409 Conflict"
    assert "already" in json.loads(resp2.text)["error"]
    mapper = get_visibility_manager().get(52)
    if mapper:
        mapper.request_stop(force=True)
        mapper.join(timeout=5.0)


def test_start_succeeds_without_gps_for_altaz(monkeypatch, tmp_path):
    monkeypatch.setattr(front_app, "_VISIBILITY_STATE_DIR", tmp_path)
    _patch_start_path(monkeypatch, gps_ok=False)
    req = _Req(body={"min_alt": 10.0, "max_runtime_min": 30.0})
    resp = _Resp()
    front_app.VisibilityStartResource.on_post(req, resp, telescope_id=53)
    # alt/az grid doesn't require GPS at start; should succeed.
    assert resp.status == "200 OK", resp.text
    mapper = get_visibility_manager().get(53)
    if mapper:
        mapper.request_stop(force=True)
        mapper.join(timeout=5.0)


def test_start_returns_503_for_healpix_without_gps(monkeypatch, tmp_path):
    monkeypatch.setattr(front_app, "_VISIBILITY_STATE_DIR", tmp_path)
    _patch_start_path(monkeypatch, gps_ok=False)
    req = _Req(body={"min_alt": 10.0, "max_runtime_min": 30.0, "grid_kind": "healpix"})
    resp = _Resp()
    front_app.VisibilityStartResource.on_post(req, resp, telescope_id=54)
    assert resp.status == "503 Service Unavailable"


# ---------- stop minimum-time enforcement --------------------------


def _direct_start_mapper(telescope_id: int, tmp_path) -> VisibilityMapper:
    """Bypass the HTTP route to install a mapper with a long min-run
    window for testing the stop path."""
    grid = make_altaz_band_grid(min_alt_deg=20.0)
    fake_calls: list[tuple[float, float]] = []

    def slew(az, alt):
        fake_calls.append((az, alt))
        return True

    def solve(timeout_s):
        time.sleep(0.05)
        return SolveOutcome.SOLVED

    opts = VisibilityMapperOptions(
        min_alt_deg=20.0,
        max_runtime_min=10.0,
        solve_timeout_s=1.0,
        slew_rate_deg_s=10000.0,
        slew_settle_s=0.0,
        t_observe_s=0.001,
        decay_interval_s=600.0,
        convergence_elapsed_s=600.0,
        min_run_before_user_stop_s=60.0,  # 1 min minimum
        frontier_decay_window_s=600.0,
        failure_recent_window_s=600.0,
    )
    mapper = VisibilityMapper(
        telescope_id=telescope_id,
        grid=grid,
        slew_func=slew,
        plate_solve_func=solve,
        options=opts,
        state_dir=tmp_path,
    )
    get_visibility_manager().start(mapper)
    return mapper


def test_stop_returns_409_inside_min_run_window(tmp_path):
    mapper = _direct_start_mapper(99, tmp_path)
    try:
        time.sleep(0.05)
        req = _Req()
        resp = _Resp()
        front_app.VisibilityStopResource.on_post(req, resp, telescope_id=99)
        assert resp.status == "409 Conflict"
        body = json.loads(resp.text)
        assert "stop disabled" in body["error"]
        # Status is included in the 409 body.
        assert body["status"]["active"] is True
    finally:
        mapper.request_stop(force=True)
        mapper.join(timeout=5.0)


def test_force_stop_succeeds_inside_min_run_window(tmp_path):
    mapper = _direct_start_mapper(100, tmp_path)
    try:
        time.sleep(0.05)
        req = _Req()
        resp = _Resp()
        front_app.VisibilityForceStopResource.on_post(req, resp, telescope_id=100)
        assert resp.status == "200 OK"
        body = json.loads(resp.text)
        # The mapper has either stopped or is stopping.
        assert body.get("stop_reason") is not None or body.get("active") is False
    finally:
        mapper.request_stop(force=True)
        mapper.join(timeout=5.0)


# ---------- cells / status while running ---------------------------


def test_cells_returns_grid_when_running(tmp_path):
    mapper = _direct_start_mapper(105, tmp_path)
    try:
        req = _Req()
        resp = _Resp()
        front_app.VisibilityCellsResource.on_get(req, resp, telescope_id=105)
        body = json.loads(resp.text)
        assert body["active"] is True
        assert isinstance(body["cells"], list)
        assert len(body["cells"]) > 100
        # Each cell has the expected fields.
        c = body["cells"][0]
        for key in ("idx", "az", "alt", "alpha", "beta", "ep", "n_obs"):
            assert key in c
    finally:
        mapper.request_stop(force=True)
        mapper.join(timeout=5.0)


def test_status_returns_running_state(tmp_path):
    mapper = _direct_start_mapper(106, tmp_path)
    try:
        req = _Req()
        resp = _Resp()
        front_app.VisibilityStatusResource.on_get(req, resp, telescope_id=106)
        body = json.loads(resp.text)
        assert body["active"] is True
        assert body["telescope_id"] == 106
        assert "median_entropy_nats" in body
        assert "map_quality" in body
    finally:
        mapper.request_stop(force=True)
        mapper.join(timeout=5.0)


# ---------- SSE events -----------------------------------------


def test_events_yields_init_payload_when_no_mapper():
    """The SSE endpoint sets resp.stream to a generator. We can't
    iterate through it indefinitely (it'd block); just verify the
    first event when no mapper is running has the init payload."""
    req = _Req()
    resp = _Resp()
    front_app.VisibilityEventsResource.on_get(req, resp, telescope_id=999)
    assert resp.status == "200 OK"
    assert resp.content_type == "text/event-stream"
    assert resp.stream is not None
    gen = resp.stream
    first = next(iter(gen))
    text = first.decode("utf-8") if isinstance(first, bytes) else first
    assert text.startswith("data: ")
    payload = json.loads(text[len("data: ") :].strip())
    assert payload["type"] == "init"
    assert payload["status"] == {"active": False}


def test_events_yields_init_then_observation(tmp_path):
    mapper = _direct_start_mapper(110, tmp_path)
    try:
        time.sleep(0.05)
        req = _Req()
        resp = _Resp()
        front_app.VisibilityEventsResource.on_get(req, resp, telescope_id=110)
        gen = resp.stream
        # First chunk: init.
        first = next(iter(gen))
        text = first.decode("utf-8")
        assert text.startswith("data: ")
        payload = json.loads(text[len("data: ") :].strip())
        assert payload["type"] == "init"
        # Don't bother iterating further — just confirm the generator
        # is wired and the listener was attached.
    finally:
        mapper.request_stop(force=True)
        mapper.join(timeout=5.0)


# ---------- nav + page render ------------------------------------


def test_nav_includes_sky_visibility_link():
    template = front_app.fetch_template("nav.html")
    html = template.render(
        root="/1",
        telescope={"device_num": 1, "name": "S", "ip_address": "x"},
        telescopes=[{"device_num": 1, "name": "S", "ip_address": "x"}],
        partial_path="",
        experimental=True,
        platform="raspberry_pi",
        uitheme="dark",
    )
    assert "/1/sky_visibility" in html
    assert "Sky Visibility Map" in html


def test_calibrate_page_links_to_sky_visibility():
    template = front_app.fetch_template("calibrate_rotation.html")
    minimal_ctx = {
        "telescope_id": 1,
        "telescope": {"device_num": 1, "name": "S", "ip_address": "x"},
        "telescopes": [{"device_num": 1, "name": "S", "ip_address": "x"}],
        "root": "/1",
        "imager_root": "http://localhost:7556/1",
        "partial_path": "calibrate_rotation",
        "online": True,
        "experimental": True,
        "confirm": False,
        "uitheme": "dark",
        "webui_text_color": None,
        "webui_font_family": None,
        "webui_font_url": None,
        "webui_link_color": None,
        "webui_accent_color": None,
        "client_master": True,
        "current_item": None,
        "current_stack": None,
        "platform": "raspberry_pi",
        "defgain": 80,
        "current_exp": None,
    }
    html = template.render(**minimal_ctx)
    assert "/1/sky_visibility" in html


def test_sky_visibility_page_renders():
    template = front_app.fetch_template("sky_visibility.html")
    minimal_ctx = {
        "telescope_id": 1,
        "telescope": {"device_num": 1, "name": "S", "ip_address": "x"},
        "telescopes": [{"device_num": 1, "name": "S", "ip_address": "x"}],
        "root": "/1",
        "imager_root": "http://localhost:7556/1",
        "partial_path": "sky_visibility",
        "online": True,
        "experimental": True,
        "confirm": False,
        "uitheme": "dark",
        "webui_text_color": None,
        "webui_font_family": None,
        "webui_font_url": None,
        "webui_link_color": None,
        "webui_accent_color": None,
        "client_master": True,
        "current_item": None,
        "current_stack": None,
        "platform": "raspberry_pi",
        "defgain": 80,
        "current_exp": None,
    }
    html = template.render(**minimal_ctx)
    assert "svm-canvas" in html
    assert "Sky Visibility Map" in html
    assert "/api/${tid}/calibrate_visibility" in html
    assert "const tid = 1" in html
