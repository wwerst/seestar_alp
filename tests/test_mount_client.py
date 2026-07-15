"""Tests for device.mount_client — the in-process mount seam.

Three concerns are covered:

1. Envelope equivalence — the same RPC issued through the real localhost
   HTTP path (AlpacaClient payload construction → the actual
   ``telescope.action`` server handler → ``.get("Value")`` unwrap) and
   through :class:`InProcessMount` against the same fake Seestar yields
   byte-identical caller-visible results. This empirically verifies the
   scout's claim that the HTTP round-trip is a pure identity transform, so
   the in-process shortcut is behaviour-identical.
2. Factory behaviour — :func:`get_mount_client` returns an in-process client
   only when the telescope is registered AND connected, and falls back to
   the HTTP client (unregistered, not-connected, force_http param, or the
   ``[mount_client] force_http`` config escape hatch).
3. Thread safety — concurrent in-process ``method_sync`` calls funnel
   through the device's ``_send_lock`` and stay correlated (no interleave,
   no cross-talk).
"""

from __future__ import annotations

import copy
import json
import logging
import threading
import time

import pytest

from device import exceptions as device_exceptions
from device import telescope
from device.alpaca_client import AlpacaClient
from device.mount_client import HttpMount, InProcessMount, get_mount_client
from device.shr import set_shr_logger

TID = 3


# --------------------------------------------------------------------------
# Fakes
# --------------------------------------------------------------------------


class _NullLogger:
    def info(self, *a, **k):
        return None

    debug = warn = warning = error = info


class _CannedSeestar:
    """Fake Seestar whose ``send_message_param_sync`` records the payload it
    received and returns a fixed canned reply (independent of the input)."""

    def __init__(self, reply, connected: bool = True):
        self.is_connected = connected
        self.logger = _NullLogger()
        self.event_callbacks = []
        self.calls = []
        self._reply = reply

    def send_message_param_sync(self, data):
        # Snapshot the payload so later mutation can't perturb the record.
        self.calls.append(copy.deepcopy(data))
        return self._reply


class _RaisingSeestar:
    """Fake Seestar whose send raises — models the firmware/socket error
    the Alpaca handler swallows to a value-less MethodResponse."""

    def __init__(self, connected: bool = True):
        self.is_connected = connected
        self.logger = _NullLogger()
        self.event_callbacks = []

    def send_message_param_sync(self, data):
        raise RuntimeError("boom")


class _LockingSeestar:
    """Fake Seestar that models ``send_message_param``'s locked send region.

    The critical section is guarded by an ``_send_lock`` (RLock, as the real
    device uses) and tracks peak concurrency. Each reply echoes the caller's
    ``seq`` so cross-talk between concurrent callers is detectable.
    """

    def __init__(self, connected: bool = True):
        self.is_connected = connected
        self.logger = _NullLogger()
        self.event_callbacks = []
        self._send_lock = threading.RLock()
        self._in_critical = 0
        self.max_concurrency = 0
        self.calls = 0

    def send_message_param_sync(self, data):
        seq = data["params"]["seq"]
        with self._send_lock:
            self._in_critical += 1
            self.max_concurrency = max(self.max_concurrency, self._in_critical)
            # Widen the window so an unserialized send would be caught.
            time.sleep(0.0005)
            self._in_critical -= 1
            self.calls += 1
        return {"result": seq, "method": data["method"]}


class _Req:
    """Minimal falcon-Request stand-in for the PUT action handler, mirroring
    tests/test_telescope_action.py."""

    method = "PUT"
    remote_addr = "127.0.0.1"
    path = "/api/v1/telescope/%d/action" % TID
    query_string = ""

    def __init__(self, action_name, params):
        self._media = {
            "Action": action_name,
            "Parameters": json.dumps(params),
            "ClientID": "1",
            "ClientTransactionID": "2",
        }
        self.content_length = len(json.dumps(self._media))

    @property
    def media(self):
        return self._media

    def get_media(self):
        return self._media


class _Resp:
    def __init__(self):
        self.text = ""


@pytest.fixture(autouse=True)
def _isolate_registry():
    """Snapshot/restore the process-global device registry so these tests
    (which register fakes) never leak into or inherit from other tests, and
    wire up the loggers the real server path expects."""
    set_shr_logger(logging.getLogger("test-mount-client"))
    device_exceptions.logger = _NullLogger()
    saved = dict(telescope.seestar_dev)
    telescope.seestar_dev.clear()
    try:
        yield
    finally:
        telescope.seestar_dev.clear()
        telescope.seestar_dev.update(saved)


def _http_value(dev, method, params):
    """Drive the RPC through the real HTTP server path against ``dev`` and
    return the caller-visible value, exactly as AlpacaClient.method_sync
    would: build the same conditional payload, hand it to the actual
    telescope.action handler, then unwrap ``Value``."""
    telescope.seestar_dev.clear()
    telescope.seestar_dev[TID] = dev
    payload = {"method": method}
    if params is not None:
        payload["params"] = params
    req = _Req("method_sync", payload)
    resp = _Resp()
    telescope.action().on_put(req, resp, devnum=TID)
    return json.loads(resp.text).get("Value")


# --------------------------------------------------------------------------
# Envelope equivalence
# --------------------------------------------------------------------------


def test_result_envelope_identical_no_params():
    canned = {
        "jsonrpc": "2.0",
        "Timestamp": "9507.243",
        "method": "scope_get_horiz_coord",
        "result": [45.0, -12.5],
        "code": 0,
        "id": 83,
    }
    dev_ip = _CannedSeestar(canned)
    ip_val = InProcessMount(dev_ip, TID).method_sync("scope_get_horiz_coord")

    dev_http = _CannedSeestar(canned)
    http_val = _http_value(dev_http, "scope_get_horiz_coord", None)

    assert ip_val == http_val == canned
    # Top-level result + Timestamp survive on both paths (measure_altaz_timed
    # relies on both).
    assert ip_val["result"] == [45.0, -12.5]
    assert ip_val.get("Timestamp") == "9507.243"
    # Identical payload reached send_message_param_sync; no "params" key when
    # params is None (the load-bearing conditional).
    assert dev_ip.calls == dev_http.calls == [{"method": "scope_get_horiz_coord"}]
    # In-process returns the live reply reference (the whole point — no copy).
    assert ip_val is canned


def test_result_envelope_identical_with_params():
    canned = {"result": None, "Timestamp": "1.0", "method": "scope_speed_move"}
    params = {"speed": 100, "angle": 90, "dur_sec": 5}

    dev_ip = _CannedSeestar(canned)
    ip_val = InProcessMount(dev_ip, TID).method_sync("scope_speed_move", params)

    dev_http = _CannedSeestar(canned)
    http_val = _http_value(dev_http, "scope_speed_move", params)

    assert ip_val == http_val == canned
    assert (
        dev_ip.calls
        == dev_http.calls
        == [{"method": "scope_speed_move", "params": params}]
    )


def test_payload_matches_alpaca_client_exactly():
    """The dict InProcessMount hands to send_message_param_sync is exactly the
    dict AlpacaClient.method_sync would PUT — including the None/bool params
    edge cases."""
    cases = [
        ("scope_get_horiz_coord", None),
        ("scope_speed_move", {"speed": 1, "angle": 0, "dur_sec": 1}),
        ("scope_set_track_state", True),
        ("get_device_state", {"keys": ["mount"]}),
    ]
    for method, params in cases:
        captured = []
        ac = AlpacaClient("127.0.0.1", 5555, TID)
        ac.action = lambda name, parameters, timeout=30.0, _c=captured: (
            _c.append(parameters),
            {"Value": None},
        )[1]
        ac.method_sync(method, params)

        dev = _CannedSeestar({"result": None})
        InProcessMount(dev, TID).method_sync(method, params)

        assert dev.calls[0] == captured[0], (method, params)


def test_send_exception_maps_to_none_both_paths():
    ip_val = InProcessMount(_RaisingSeestar(), TID).method_sync("scope_get_horiz_coord")
    http_val = _http_value(_RaisingSeestar(), "scope_get_horiz_coord", None)
    assert ip_val is None
    assert http_val is None


def test_timeout_dict_passes_through_both_paths():
    # send_message_param_sync returns a mutated dict (non-raising) on timeout.
    timeout_reply = {
        "method": "scope_get_horiz_coord",
        "result": "Error: Exceeded allotted wait time for result",
    }
    dev_ip = _CannedSeestar(timeout_reply)
    ip_val = InProcessMount(dev_ip, TID).method_sync("scope_get_horiz_coord")

    dev_http = _CannedSeestar(timeout_reply)
    http_val = _http_value(dev_http, "scope_get_horiz_coord", None)

    assert ip_val == http_val == timeout_reply
    assert ip_val is not None  # a dict, NOT swallowed to None


def test_device_attr_is_int_telescope_id():
    """`.device` MUST be the telescope-id int (never the Seestar object) —
    sun_safety_jog_in_progress(getattr(cli, "device", None)) compares it
    against the monitor's jog telescope-id int. Parity with AlpacaClient."""
    m = InProcessMount(_CannedSeestar({"result": None}), "7")
    assert m.device == 7 and isinstance(m.device, int)
    assert AlpacaClient("h", 5555, "7").device == 7


# --------------------------------------------------------------------------
# Factory
# --------------------------------------------------------------------------


def test_factory_returns_inprocess_when_registered_and_connected():
    dev = _CannedSeestar({"result": None}, connected=True)
    telescope.seestar_dev[TID] = dev
    client = get_mount_client(TID)
    assert isinstance(client, InProcessMount)
    assert client.device == TID
    client.method_sync("scope_get_horiz_coord")
    assert dev.calls == [{"method": "scope_get_horiz_coord"}]


def test_factory_falls_back_to_http_when_unregistered():
    # Registry cleared by the fixture.
    client = get_mount_client(TID, port=1234)
    assert isinstance(client, HttpMount)
    assert not isinstance(client, InProcessMount)
    assert client.device == TID
    assert ":1234/" in client.base


def test_factory_http_defaults_port_to_config():
    from device.config import Config

    client = get_mount_client(TID)  # port omitted
    assert isinstance(client, HttpMount)
    assert f":{int(Config.port)}/" in client.base


def test_factory_falls_back_to_http_when_registered_but_not_connected():
    telescope.seestar_dev[TID] = _CannedSeestar({"result": None}, connected=False)
    client = get_mount_client(TID)
    assert isinstance(client, HttpMount)
    assert not isinstance(client, InProcessMount)


def test_factory_force_http_param_bypasses_inprocess():
    telescope.seestar_dev[TID] = _CannedSeestar({"result": None}, connected=True)
    client = get_mount_client(TID, force_http=True)
    assert isinstance(client, HttpMount)
    assert not isinstance(client, InProcessMount)


def test_factory_force_http_config_escape_hatch(monkeypatch):
    telescope.seestar_dev[TID] = _CannedSeestar({"result": None}, connected=True)
    from device.config import Config

    orig = Config.get_toml

    def fake_get_toml(sect, item, default):
        if sect == "mount_client" and item == "force_http":
            return True
        return orig(sect, item, default)

    monkeypatch.setattr(Config, "get_toml", fake_get_toml)
    client = get_mount_client(TID)
    assert isinstance(client, HttpMount)
    assert not isinstance(client, InProcessMount)


# --------------------------------------------------------------------------
# Thread safety
# --------------------------------------------------------------------------


def test_concurrent_method_sync_serialized_by_send_lock():
    dev = _LockingSeestar()
    m = InProcessMount(dev, TID)
    errors: list[tuple[int, int]] = []

    def worker(base: int) -> None:
        for i in range(50):
            seq = base * 1000 + i
            resp = m.method_sync("scope_speed_move", {"seq": seq})
            if resp["result"] != seq:
                errors.append((seq, resp["result"]))

    threads = [threading.Thread(target=worker, args=(b,)) for b in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # No cross-talk: every caller got its own correlated reply back.
    assert errors == []
    # The device _send_lock serialized every send — no two sends overlapped.
    assert dev.max_concurrency == 1
    assert dev.calls == 8 * 50


def test_in_process_mount_returns_none_after_mid_session_disconnect():
    """The HTTP action layer refuses a not-connected device on EVERY
    request; a mid-session disconnect must yield the same immediate None
    from InProcessMount instead of pushing into a dead socket."""

    class _FlappingSeestar:
        is_connected = True

        def __init__(self):
            self.calls = []

        def send_message_param_sync(self, payload):
            self.calls.append(payload)
            return {"result": [10.0, 20.0]}

    dev = _FlappingSeestar()
    m = InProcessMount(dev, telescope_id=1)
    assert m.method_sync("scope_get_horiz_coord") == {"result": [10.0, 20.0]}

    dev.is_connected = False
    assert m.method_sync("scope_get_horiz_coord") is None
    assert len(dev.calls) == 1, "disconnected device must not be called"

    dev.is_connected = True
    assert m.method_sync("scope_get_horiz_coord") == {"result": [10.0, 20.0]}
