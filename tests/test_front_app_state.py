import json
import os
import re
import shutil
import threading
from types import SimpleNamespace

import pytest
import falcon
import front.app as front_app
from device.config import Config
from falcon import testing


@pytest.fixture(autouse=True)
def _clear_auth_cache():
    front_app._auth_needs_cache.clear()
    yield
    front_app._auth_needs_cache.clear()


@pytest.fixture(autouse=True)
def _isolate_emulator_flag(monkeypatch):
    # Isolate from the developer's local config.toml: if a [[seestars]] entry
    # has is_emulator = true (e.g. for local emulator testing), tests below
    # that don't care about it must not silently start skipping their
    # check_needs_auth path. Tests exercising is_emulator_telescope or the
    # is_emulator short-circuit explicitly override get_telescope again.
    monkeypatch.setattr(front_app, "get_telescope", lambda _tid: {"device_num": _tid})


class DummyReq:
    def __init__(self, host="localhost:5432", scheme="http"):
        self.host = host
        self.scheme = scheme
        self.relative_uri = "/1/live"


class DummyResp:
    def __init__(self):
        self.cookies = []

    def set_cookie(self, key, value, path="/"):
        self.cookies.append((key, value, path))

    def unset_cookie(self, key, path="/"):
        self.cookies = [c for c in self.cookies if c[0] != key]


class DummyHTMXReq(DummyReq):
    def __init__(
        self,
        host="localhost:5432",
        scheme="http",
        relative_uri="/1/",
        params=None,
        headers=None,
    ):
        super().__init__(host=host, scheme=scheme)
        self.relative_uri = relative_uri
        self._params = params or {}
        self._headers = headers or {}

    def get_param(self, key, default=None):
        return self._params.get(key, default)

    def get_header(self, key):
        return self._headers.get(key)

    def get_cookie_values(self, _key):
        return []


def _render_nav(partial_path):
    template = front_app.fetch_template("nav.html")
    telescopes = [
        {"device_num": 1, "name": "Seestar Alpha", "ip_address": "192.168.11.124"},
    ]
    context = {
        "root": "/1",
        "telescope": telescopes[0],
        "telescopes": telescopes,
        "partial_path": partial_path,
        "experimental": True,
        "platform": "raspberry_pi",
        "uitheme": "dark",
    }
    return template.render(**context)


def _minimal_context(partial_path, online=True):
    telescope = {
        "device_num": 1,
        "name": "Seestar Alpha",
        "ip_address": "192.168.11.124",
    }
    return {
        "telescope": telescope,
        "telescopes": [telescope],
        "root": "/1",
        "partial_path": partial_path,
        "online": online,
        "imager_root": "http://localhost:7556/1",
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
        "needs_auth_warning": False,
    }


def test_flash_and_get_messages_roundtrip():
    front_app.messages.clear()
    resp = DummyResp()
    front_app.flash(resp, "hello")

    assert resp.cookies == [("flash_cookie", "hello", "/")]
    assert front_app.get_messages() == ["hello"]
    assert front_app.get_messages() == []


def test_nav_shows_federation_option_on_home():
    html = _render_nav("")
    assert "Seestar Federation" in html
    assert 'href="/0/"' in html


def test_nav_shows_federation_option_for_supported_pages():
    partial_paths = [
        "command",
        "guestmode",
        "live",
        "planning",
        "config",
        "stats",
        "platform-rpi",
        "support",
    ]
    for path in partial_paths:
        html = _render_nav(path)
        assert "Seestar Federation" in html
        assert f'href="/0/{path}"' in html


def test_get_root_and_imager_root(monkeypatch):
    monkeypatch.setattr(
        Config,
        "seestars",
        [
            {"device_num": 1, "name": "A", "ip_address": "a.local"},
            {"device_num": 2, "name": "B", "ip_address": "b.local"},
        ],
    )
    monkeypatch.setattr(Config, "imgport", 7556)
    req = DummyReq(host="myhost:1234")

    assert front_app.get_root(0) == "/0"
    assert front_app.get_root(2) == "/2"
    assert front_app.get_imager_root(2, req) == "http://myhost:7556/2"


def test_get_imager_root_strips_incoming_port_and_preserves_scheme(monkeypatch):
    monkeypatch.setattr(
        Config,
        "seestars",
        [
            {"device_num": 2, "name": "B", "ip_address": "b.local"},
        ],
    )
    monkeypatch.setattr(Config, "imgport", 7556)
    req = DummyReq(host="securehost.example:8443", scheme="https")

    assert front_app.get_imager_root(2, req) == "https://securehost.example:7556/2"


def test_process_queue_dispatches_actions(monkeypatch):
    calls = []
    front_app.queue.clear()
    front_app.queue[1] = [
        {"Parameters": json.dumps({"action": "wait_for", "params": {"timer_sec": 5}})},
        {"Parameters": json.dumps({"action": "noop", "params": None})},
    ]
    monkeypatch.setattr(front_app, "check_api_state", lambda telescope_id: True)
    monkeypatch.setattr(
        front_app,
        "do_schedule_action_device",
        lambda action, params, telescope_id: (
            calls.append((action, params, telescope_id)) or {"ok": True}
        ),
    )

    front_app.process_queue(DummyResp(), 1)
    assert calls == [("wait_for", {"timer_sec": 5}, 1), ("noop", None, 1)]


def test_process_queue_offline_flashes_error(monkeypatch):
    monkeypatch.setattr(front_app, "check_api_state", lambda telescope_id: False)
    resp = DummyResp()
    front_app.process_queue(resp, 1)
    msgs = front_app.get_messages()
    assert any("API is Offline" in msg for msg in msgs)


def test_get_nearest_csc_uses_result_cache(monkeypatch):
    monkeypatch.setattr(Config, "init_lat", 42.0)
    monkeypatch.setattr(Config, "init_long", -71.0)
    front_app._nearest_csc_cache.clear()

    calls = {"count": 0}

    def fake_get_csc_sites_data():
        calls["count"] += 1
        return {
            "42": {
                "-71": [
                    {"id": "TEST", "lat": 42.0, "lng": -71.0},
                ]
            }
        }

    monkeypatch.setattr(front_app, "get_csc_sites_data", fake_get_csc_sites_data)

    first = front_app.get_nearest_csc()
    second = front_app.get_nearest_csc()

    assert first["status_msg"] == "SUCCESS"
    assert second["status_msg"] == "SUCCESS"
    assert first["href"] == "https://www.cleardarksky.com/c/TESTkey.html"
    assert calls["count"] == 1


def test_config_json_resource_contract(monkeypatch):
    monkeypatch.setattr(
        Config,
        "seestars",
        [
            {"device_num": 1, "name": "Seestar Alpha", "ip_address": "192.168.1.50"},
            {"device_num": 2, "name": "Seestar Beta", "ip_address": "192.168.1.51"},
        ],
    )
    monkeypatch.setattr(Config, "port", 5555)
    monkeypatch.setattr(Config, "imgport", 7556)

    resp = falcon.Response()
    front_app.ConfigJsonResource.on_get(DummyReq(), resp)

    assert resp.status == falcon.HTTP_200
    assert resp.content_type == "application/json"
    data = json.loads(resp.text)
    assert data["server"] == {"alpaca_port": 5555, "imaging_port": 7556}
    assert data["devices"] == [
        {"device_num": 1, "name": "Seestar Alpha", "ip_address": "192.168.1.50"},
        {"device_num": 2, "name": "Seestar Beta", "ip_address": "192.168.1.51"},
    ]


def test_get_planning_cards_uses_file_mtime_cache(monkeypatch, tmp_path):
    planning_file = tmp_path / "planning.json"
    planning_file.write_text(
        json.dumps(
            [
                {
                    "card_name": "twilight_times",
                    "planning_page_enable": True,
                    "planning_page_collapsed": False,
                }
            ]
        )
    )

    original_json_load = front_app.json.load
    calls = {"count": 0}

    def counting_json_load(fp):
        calls["count"] += 1
        return original_json_load(fp)

    monkeypatch.setattr(front_app.os.path, "dirname", lambda _: str(tmp_path))
    monkeypatch.setattr(front_app.json, "load", counting_json_load)
    front_app._planning_cards_cache = None
    front_app._planning_cards_cache_mtime = None

    first = front_app.get_planning_cards()
    second = front_app.get_planning_cards()

    assert first[0]["card_name"] == "twilight_times"
    assert second[0]["card_name"] == "twilight_times"
    assert calls["count"] == 1


def test_update_planning_card_state_invalidates_cache(monkeypatch, tmp_path):
    planning_file = tmp_path / "planning.json"
    planning_file.write_text(
        json.dumps(
            [
                {
                    "card_name": "twilight_times",
                    "planning_page_enable": True,
                    "planning_page_collapsed": False,
                }
            ]
        )
    )

    monkeypatch.setattr(front_app.os.path, "dirname", lambda _: str(tmp_path))
    front_app._planning_cards_cache = None
    front_app._planning_cards_cache_mtime = None

    cards = front_app.get_planning_cards()
    assert cards[0]["planning_page_enable"] is True
    assert front_app._planning_cards_cache is not None

    front_app.update_planning_card_state(
        "twilight_times", "planning_page_enable", False
    )

    assert front_app._planning_cards_cache is None
    updated_cards = front_app.get_planning_cards()
    assert updated_cards[0]["planning_page_enable"] is False


def test_get_planning_cards_merges_missing_cards_from_example(monkeypatch, tmp_path):
    planning_file = tmp_path / "planning.json"
    planning_file.write_text(
        json.dumps(
            [
                {
                    "card_name": "twilight_times",
                    "card_friendly_name": "Twilight Times",
                    "template": "partials/twilight_times.html",
                    "planning_page_enable": False,
                    "planning_page_collapsed": True,
                }
            ]
        )
    )
    example_file = tmp_path / "planning.json.example"
    example_file.write_text(
        json.dumps(
            [
                {
                    "card_name": "twilight_times",
                    "card_friendly_name": "Twilight Times",
                    "template": "partials/twilight_times.html",
                    "planning_page_enable": True,
                    "planning_page_collapsed": False,
                },
                {
                    "card_name": "framed_mosaic",
                    "card_friendly_name": "Framed Mosaic",
                    "template": "partials/framed_mosaic_planning.html",
                    "planning_page_enable": True,
                    "planning_page_collapsed": False,
                },
            ]
        )
    )

    monkeypatch.setattr(front_app.os.path, "dirname", lambda _: str(tmp_path))
    front_app._planning_cards_cache = None
    front_app._planning_cards_cache_mtime = None

    cards = front_app.get_planning_cards()

    names = {card["card_name"] for card in cards}
    assert names == {"twilight_times", "framed_mosaic"}

    # the user's existing settings for a card they already have must be preserved
    twilight = next(c for c in cards if c["card_name"] == "twilight_times")
    assert twilight["planning_page_enable"] is False
    assert twilight["planning_page_collapsed"] is True

    # the merge is persisted so a second load doesn't need the cache
    front_app._planning_cards_cache = None
    front_app._planning_cards_cache_mtime = None
    persisted = json.loads(planning_file.read_text())
    assert {c["card_name"] for c in persisted} == {"twilight_times", "framed_mosaic"}


def test_planning_page_includes_framed_mosaic_card(monkeypatch, tmp_path):
    example_source = os.path.join(
        os.path.dirname(front_app.__file__), "planning.json.example"
    )
    shutil.copyfile(example_source, tmp_path / "planning.json")

    monkeypatch.setattr(front_app.os.path, "dirname", lambda _: str(tmp_path))
    front_app._planning_cards_cache = None
    front_app._planning_cards_cache_mtime = None

    cards = front_app.get_planning_cards()
    names = {card["card_name"] for card in cards}
    assert "framed_mosaic" in names

    framed_mosaic_card = next(c for c in cards if c["card_name"] == "framed_mosaic")
    assert framed_mosaic_card["template"] == "partials/framed_mosaic_planning.html"


def test_get_csc_sites_data_uses_in_memory_cache(monkeypatch, tmp_path):
    csc_file = tmp_path / "csc_sites.json"
    csc_file.write_text(json.dumps({"42": {"-71": [{"id": "A"}]}}))

    original_json_load = front_app.json.load
    calls = {"count": 0}

    def counting_json_load(fp):
        calls["count"] += 1
        return original_json_load(fp)

    monkeypatch.setattr(front_app.os.path, "dirname", lambda _: str(tmp_path))
    monkeypatch.setattr(front_app.json, "load", counting_json_load)
    front_app._csc_sites_cache = None

    first = front_app.get_csc_sites_data()
    second = front_app.get_csc_sites_data()

    assert first == second
    assert calls["count"] == 1


def test_get_device_settings_uses_fallback_keys_for_newer_firmware(monkeypatch):
    monkeypatch.setattr(front_app, "get_client_master", lambda _tid: True)
    monkeypatch.setattr(front_app, "get_firmware_ver_int", lambda _tid: 2670)
    monkeypatch.setattr(front_app, "get_device_model", lambda _tid: "Seestar S50")

    def fake_method_sync(method, telescope_id=1, **kwargs):
        if method == "get_setting":
            return {
                "stack_dither": {"pix": 10, "interval": 2, "enable": True},
                "exp_ms": {"stack_l": 10000, "continuous": 500},
                "auto_3ppa_calib": True,
                "frame_calib": True,
                "focal_pos": 1500,
                "heater_enable": False,
                "auto_power_off": False,
                "stack_lenhance": False,
                "dark_mode": False,
                "stack_cont_capt": True,
                "stack": {"drizzle2x": False},
                "plan": {"target_af": True},
                "viewplan_go_home": True,
                "expert_mode": False,
            }
        if method == "get_stack_setting":
            return {}
        raise AssertionError(f"Unexpected method call: {method}")

    monkeypatch.setattr(front_app, "method_sync", fake_method_sync)

    settings = front_app.get_device_settings(1)

    assert settings["stack_cont_capt"] is True
    assert settings["plan_target_af"] is True
    assert settings["viewplan_gohome"] is True


def test_settings_post_tries_fallback_variants_for_firmware_specific_keys(monkeypatch):
    class DummyReq:
        def __init__(self):
            self.media = {
                "stack_lenhance": "false",
                "stack_dither_pix": "10",
                "stack_dither_interval": "2",
                "stack_dither_enable": "true",
                "exp_ms_stack_l": "10000",
                "exp_ms_continuous": "500",
                "focal_pos": "1500",
                "auto_power_off": "false",
                "auto_3ppa_calib": "true",
                "frame_calib": "true",
                "plan_target_af": "true",
                "viewplan_gohome": "true",
                "expert_mode": "false",
                "save_discrete_frame": "false",
                "save_discrete_ok_frame": "true",
                "light_duration_min": "10",
                "stack_capt_type": "stack",
                "stack_capt_num": "1",
                "stack_brightness": "0",
                "stack_contrast": "0",
                "stack_saturation": "0",
                "stack_dbe_enable": "false",
                "heater_enable": "false",
                "dark_mode": "false",
                "stack_cont_capt": "true",
                "stack_drizzle2x": "false",
            }

    captured = {"output": None, "calls": []}

    def fake_do_action_device(action, dev_num, parameters, is_schedule=False):
        captured["calls"].append((action, parameters))
        if action == "method_async":
            return {"ErrorNumber": 0, "Value": {"code": 0}}

        method = parameters.get("method")
        params = parameters.get("params", {})
        if method != "set_setting":
            return {"ErrorNumber": 0, "Value": {"code": 0}}

        if params == {"stack": {"cont_capt": True}}:
            return {"ErrorNumber": 0, "Value": {"code": -1, "error": "unsupported"}}
        if params == {"stack_cont_capt": True}:
            return {"ErrorNumber": 0, "Value": {"code": 0}}

        if params == {"plan_target_af": True}:
            return {"ErrorNumber": 0, "Value": {"code": -1, "error": "unsupported"}}
        if params == {"plan": {"target_af": True}}:
            return {"ErrorNumber": 0, "Value": {"code": 0}}

        if params == {"viewplan_gohome": True}:
            return {"ErrorNumber": 0, "Value": {"code": -1, "error": "unsupported"}}
        if params == {"viewplan_go_home": True}:
            return {"ErrorNumber": 0, "Value": {"code": 0}}

        return {"ErrorNumber": 0, "Value": {"code": 0}}

    monkeypatch.setattr(front_app, "get_firmware_ver_int", lambda _tid: 2670)
    monkeypatch.setattr(front_app, "get_device_model", lambda _tid: "Seestar S50")
    monkeypatch.setattr(front_app, "do_action_device", fake_do_action_device)
    monkeypatch.setattr(
        front_app.SettingsResource,
        "render_settings",
        staticmethod(
            lambda _req, _resp, _tid, output: captured.__setitem__("output", output)
        ),
    )

    front_app.SettingsResource().on_post(DummyReq(), object(), 1)

    assert captured["output"] == "Successfully Updated Settings."
    assert (
        "method_sync",
        {"method": "set_setting", "params": {"stack_cont_capt": True}},
    ) in captured["calls"]
    assert (
        "method_sync",
        {"method": "set_setting", "params": {"plan": {"target_af": True}}},
    ) in captured["calls"]
    assert (
        "method_sync",
        {"method": "set_setting", "params": {"viewplan_go_home": True}},
    ) in captured["calls"]


def test_get_device_settings_loads_discrete_save_flags_from_get_stack_setting(
    monkeypatch,
):
    monkeypatch.setattr(front_app, "get_client_master", lambda _tid: True)
    monkeypatch.setattr(front_app, "get_firmware_ver_int", lambda _tid: 2500)
    monkeypatch.setattr(front_app, "get_device_model", lambda _tid: "Seestar S50")

    def fake_method_sync(method, telescope_id=1, **kwargs):
        if method == "get_setting":
            return {
                "stack_dither": {"pix": 10, "interval": 2, "enable": True},
                "exp_ms": {"stack_l": 10000, "continuous": 500},
                "auto_3ppa_calib": True,
                "frame_calib": True,
                "focal_pos": 1500,
                "heater_enable": False,
                "auto_power_off": False,
                "stack_lenhance": False,
                "dark_mode": False,
                "stack_cont_capt": True,
                "stack": {"drizzle2x": False},
            }
        if method == "get_stack_setting":
            return {
                "save_discrete_frame": True,
                "save_discrete_ok_frame": False,
            }
        raise AssertionError(f"Unexpected method call: {method}")

    monkeypatch.setattr(front_app, "method_sync", fake_method_sync)

    settings = front_app.get_device_settings(1)

    assert settings["save_discrete_frame"] is True
    assert settings["save_discrete_ok_frame"] is False


def test_method_sync_handles_wrapped_single_device_value(monkeypatch):
    def fake_do_action_device(action, dev_num, parameters, is_schedule=False):
        assert action == "method_sync"
        return {
            "ServerTransactionID": 1,
            "ClientTransactionID": 999,
            "Value": {
                "1": {
                    "jsonrpc": "2.0",
                    "method": "get_stack_setting",
                    "result": {
                        "save_discrete_frame": True,
                        "save_discrete_ok_frame": True,
                        "light_duration_min": -1,
                    },
                    "code": 0,
                    "id": 27207,
                }
            },
            "ErrorNumber": 0,
            "ErrorMessage": "",
        }

    monkeypatch.setattr(front_app, "do_action_device", fake_do_action_device)

    result = front_app.method_sync("get_stack_setting", telescope_id=1)

    assert result["save_discrete_frame"] is True
    assert result["save_discrete_ok_frame"] is True
    assert result["light_duration_min"] == -1


def test_settings_post_saves_discrete_flags_under_stack_payload(monkeypatch):
    class DummyReq:
        def __init__(self):
            self.media = {
                "stack_lenhance": "false",
                "stack_dither_pix": "10",
                "stack_dither_interval": "2",
                "stack_dither_enable": "true",
                "exp_ms_stack_l": "10000",
                "exp_ms_continuous": "500",
                "focal_pos": "1500",
                "auto_power_off": "false",
                "auto_3ppa_calib": "true",
                "frame_calib": "true",
                "plan_target_af": "false",
                "viewplan_gohome": "false",
                "expert_mode": "false",
                "save_discrete_frame": "true",
                "save_discrete_ok_frame": "false",
                "light_duration_min": "10",
                "stack_capt_type": "stack",
                "stack_capt_num": "2",
                "stack_brightness": "1.1",
                "stack_contrast": "2.2",
                "stack_saturation": "3.3",
                "stack_dbe_enable": "true",
                "heater_enable": "false",
                "dark_mode": "false",
                "stack_cont_capt": "false",
                "stack_drizzle2x": "false",
            }

    captured = {"stack_payload": None}

    def fake_do_action_device(action, dev_num, parameters, is_schedule=False):
        method = parameters.get("method")
        params = parameters.get("params", {})
        if (
            action == "method_sync"
            and method == "set_setting"
            and isinstance(params, dict)
            and "stack" in params
        ):
            captured["stack_payload"] = params["stack"]
        return {"ErrorNumber": 0, "Value": {"code": 0}}

    monkeypatch.setattr(front_app, "get_firmware_ver_int", lambda _tid: 2670)
    monkeypatch.setattr(front_app, "get_device_model", lambda _tid: "Seestar S50")
    monkeypatch.setattr(front_app, "do_action_device", fake_do_action_device)
    monkeypatch.setattr(
        front_app.SettingsResource,
        "render_settings",
        staticmethod(lambda _req, _resp, _tid, _output: None),
    )

    front_app.SettingsResource().on_post(DummyReq(), object(), 1)

    assert captured["stack_payload"] is not None
    assert captured["stack_payload"]["save_discrete_frame"] is True
    assert captured["stack_payload"]["save_discrete_ok_frame"] is False


def test_settings_post_falls_back_to_set_stack_setting_for_discrete_flags(monkeypatch):
    class DummyReq:
        def __init__(self):
            self.media = {
                "stack_lenhance": "false",
                "stack_dither_pix": "10",
                "stack_dither_interval": "2",
                "stack_dither_enable": "true",
                "exp_ms_stack_l": "10000",
                "exp_ms_continuous": "500",
                "focal_pos": "1500",
                "auto_power_off": "false",
                "auto_3ppa_calib": "true",
                "frame_calib": "true",
                "plan_target_af": "false",
                "viewplan_gohome": "false",
                "expert_mode": "false",
                "save_discrete_frame": "true",
                "save_discrete_ok_frame": "false",
                "light_duration_min": "15",
                "stack_capt_type": "stack",
                "stack_capt_num": "2",
                "stack_brightness": "0",
                "stack_contrast": "0",
                "stack_saturation": "0",
                "stack_dbe_enable": "false",
                "heater_enable": "false",
                "dark_mode": "false",
                "stack_cont_capt": "false",
                "stack_drizzle2x": "false",
            }

    captured = {"output": None, "stack_fallback_called": False}

    def fake_do_action_device(action, dev_num, parameters, is_schedule=False):
        if action == "method_async":
            return {"ErrorNumber": 0, "Value": {"code": 0}}

        method = parameters.get("method")
        params = parameters.get("params", {})
        if method == "set_setting" and params == {"stack": {"cont_capt": False}}:
            return {"ErrorNumber": 0, "Value": {"code": 0}}
        if method == "set_setting" and isinstance(params, dict) and "stack" in params:
            # Simulate firmware that rejects stack payload via set_setting.
            return {"ErrorNumber": 0, "Value": {"code": -1, "error": "unsupported"}}
        if method == "set_stack_setting":
            captured["stack_fallback_called"] = True
            return {"ErrorNumber": 0, "Value": {"code": 0}}
        return {"ErrorNumber": 0, "Value": {"code": 0}}

    monkeypatch.setattr(front_app, "get_firmware_ver_int", lambda _tid: 2670)
    monkeypatch.setattr(front_app, "get_device_model", lambda _tid: "Seestar S50")
    monkeypatch.setattr(front_app, "do_action_device", fake_do_action_device)
    monkeypatch.setattr(
        front_app.SettingsResource,
        "render_settings",
        staticmethod(
            lambda _req, _resp, _tid, output: captured.__setitem__("output", output)
        ),
    )

    front_app.SettingsResource().on_post(DummyReq(), object(), 1)

    assert captured["stack_fallback_called"] is True
    assert captured["output"] == "Successfully Updated Settings."


def test_settings_post_older_firmware_uses_stack_setting_methods(monkeypatch):
    class DummyReq:
        def __init__(self):
            self.media = {
                "stack_lenhance": "false",
                "stack_dither_pix": "10",
                "stack_dither_interval": "2",
                "stack_dither_enable": "true",
                "exp_ms_stack_l": "10000",
                "exp_ms_continuous": "500",
                "focal_pos": "1500",
                "auto_power_off": "false",
                "auto_3ppa_calib": "true",
                "frame_calib": "true",
                "save_discrete_frame": "true",
                "save_discrete_ok_frame": "false",
                "light_duration_min": "20",
                "stack_capt_type": "stack",
                "stack_capt_num": "3",
                "stack_brightness": "0",
                "stack_contrast": "0",
                "stack_saturation": "0",
                "stack_dbe_enable": "false",
                "heater_enable": "false",
                "dark_mode": "false",
                "stack_cont_capt": "false",
                "stack_drizzle2x": "false",
            }

    captured = {"stack_method_calls": []}

    def fake_do_action_device(action, dev_num, parameters, is_schedule=False):
        method = parameters.get("method")
        params = parameters.get("params", {})
        if action == "method_async":
            return {"ErrorNumber": 0, "Value": {"code": 0}}
        if method in {"set_stack_setting", "set_stack_settings"}:
            captured["stack_method_calls"].append((method, params))
            return {"ErrorNumber": 0, "Value": {"code": 0}}
        return {"ErrorNumber": 0, "Value": {"code": 0}}

    monkeypatch.setattr(front_app, "get_firmware_ver_int", lambda _tid: 2500)
    monkeypatch.setattr(front_app, "get_device_model", lambda _tid: "Seestar S50")
    monkeypatch.setattr(front_app, "do_action_device", fake_do_action_device)
    monkeypatch.setattr(
        front_app.SettingsResource,
        "render_settings",
        staticmethod(lambda _req, _resp, _tid, _output: None),
    )

    front_app.SettingsResource().on_post(DummyReq(), object(), 1)

    assert captured["stack_method_calls"]
    method_name, payload = captured["stack_method_calls"][0]
    assert method_name in {"set_stack_setting", "set_stack_settings"}
    assert payload["save_discrete_frame"] is True
    assert payload["save_discrete_ok_frame"] is False


def test_settings_post_missing_light_duration_min_does_not_raise(monkeypatch):
    class DummyReq:
        def __init__(self):
            self.media = {
                "stack_lenhance": "false",
                "stack_dither_pix": "10",
                "stack_dither_interval": "2",
                "stack_dither_enable": "true",
                "exp_ms_stack_l": "10000",
                "exp_ms_continuous": "500",
                "focal_pos": "1500",
                "auto_power_off": "false",
                "auto_3ppa_calib": "true",
                "frame_calib": "true",
                "save_discrete_frame": "true",
                "save_discrete_ok_frame": "false",
                "heater_enable": "false",
                "dark_mode": "false",
                "stack_cont_capt": "false",
                "stack_drizzle2x": "false",
            }

    captured = {"stack_payload": None}

    def fake_do_action_device(action, dev_num, parameters, is_schedule=False):
        method = parameters.get("method")
        params = parameters.get("params", {})
        if method in {"set_stack_setting", "set_stack_settings"}:
            captured["stack_payload"] = params
            return {"ErrorNumber": 0, "Value": {"code": 0}}
        return {"ErrorNumber": 0, "Value": {"code": 0}}

    monkeypatch.setattr(front_app, "get_firmware_ver_int", lambda _tid: 2500)
    monkeypatch.setattr(front_app, "get_device_model", lambda _tid: "Seestar S50")
    monkeypatch.setattr(front_app, "do_action_device", fake_do_action_device)
    monkeypatch.setattr(
        front_app.SettingsResource,
        "render_settings",
        staticmethod(lambda _req, _resp, _tid, _output: None),
    )

    front_app.SettingsResource().on_post(DummyReq(), object(), 1)

    assert captured["stack_payload"] is not None
    assert captured["stack_payload"]["light_duration_min"] == -1


@pytest.mark.parametrize(
    "timed_out_call", ["settings", "dark_mode", "drizzle", "star_trails"]
)
def test_settings_post_does_not_crash_when_a_write_times_out(
    monkeypatch, timed_out_call
):
    # Issue #728: a device-side write timeout (do_action_device returns None,
    # matching a real requests.exceptions.Timeout in front/app.py's
    # do_action_device) must not crash on_post with
    # `TypeError: 'NoneType' object is not subscriptable` when the result is
    # later subscripted as e.g. settings_output["ErrorNumber"].
    model = "Seestar S30 Pro" if timed_out_call == "star_trails" else "Seestar S50"

    class DummyReq:
        def __init__(self):
            self.media = {
                "stack_lenhance": "false",
                "stack_dither_pix": "10",
                "stack_dither_interval": "2",
                "stack_dither_enable": "true",
                "exp_ms_stack_l": "10000",
                "exp_ms_continuous": "500",
                "focal_pos": "1500",
                "auto_power_off": "false",
                "auto_3ppa_calib": "true",
                "frame_calib": "true",
                "save_discrete_frame": "true",
                "save_discrete_ok_frame": "false",
                "heater_enable": "false",
                "dark_mode": "false",
                "stack_cont_capt": "false",
                "stack_drizzle2x": "false",
                "stack_star_trails": "false",
            }

    def fake_do_action_device(action, dev_num, parameters, is_schedule=False):
        method = parameters.get("method")
        params = parameters.get("params", {})
        if timed_out_call == "settings" and "stack_lenhance" in params:
            return None
        if timed_out_call == "dark_mode" and "dark_mode" in params:
            return None
        if timed_out_call == "drizzle" and "drizzle2x" in params.get("stack", {}):
            return None
        if timed_out_call == "star_trails" and "star_trails" in params.get("stack", {}):
            return None
        assert method  # sanity: every other call still gets a normal response
        return {"ErrorNumber": 0, "Value": {"code": 0}}

    captured = {}
    monkeypatch.setattr(front_app, "get_firmware_ver_int", lambda _tid: 2500)
    monkeypatch.setattr(front_app, "get_device_model", lambda _tid: model)
    monkeypatch.setattr(front_app, "do_action_device", fake_do_action_device)
    monkeypatch.setattr(
        front_app.SettingsResource,
        "render_settings",
        staticmethod(lambda _req, _resp, _tid, output: captured.update(output=output)),
    )

    front_app.SettingsResource().on_post(DummyReq(), object(), 1)

    assert captured["output"] == "Error Updating Settings."


def test_settings_post_auto_lenhance_sent_on_fw_2775(monkeypatch):
    class DummyReq:
        def __init__(self):
            self.media = {
                "stack_lenhance": "false",
                "auto_lenhance": "true",
                "stack_dither_pix": "10",
                "stack_dither_interval": "2",
                "stack_dither_enable": "true",
                "exp_ms_stack_l": "10000",
                "exp_ms_continuous": "500",
                "focal_pos": "1500",
                "auto_power_off": "false",
                "auto_3ppa_calib": "true",
                "frame_calib": "true",
                "plan_target_af": "false",
                "viewplan_gohome": "false",
                "expert_mode": "false",
                "heater_enable": "false",
                "dark_mode": "false",
                "stack_cont_capt": "false",
                "stack_drizzle2x": "false",
            }

    captured = {"main_set_setting_params": None}

    def fake_do_action_device(action, dev_num, parameters, is_schedule=False):
        if action == "method_async":
            return {"ErrorNumber": 0, "Value": {"code": 0}}
        method = parameters.get("method")
        params = parameters.get("params", {})
        if (
            action == "method_sync"
            and method == "set_setting"
            and "auto_lenhance" in params
        ):
            captured["main_set_setting_params"] = params
        return {"ErrorNumber": 0, "Value": {"code": 0}}

    monkeypatch.setattr(front_app, "get_firmware_ver_int", lambda _tid: 2775)
    monkeypatch.setattr(front_app, "get_device_model", lambda _tid: "Seestar S50")
    monkeypatch.setattr(front_app, "do_action_device", fake_do_action_device)
    monkeypatch.setattr(
        front_app.SettingsResource,
        "render_settings",
        staticmethod(lambda _req, _resp, _tid, _output: None),
    )

    front_app.SettingsResource().on_post(DummyReq(), object(), 1)

    assert captured["main_set_setting_params"] is not None
    assert captured["main_set_setting_params"]["auto_lenhance"] is True


def test_settings_post_auto_lenhance_not_sent_on_older_fw(monkeypatch):
    class DummyReq:
        def __init__(self):
            self.media = {
                "stack_lenhance": "false",
                "stack_dither_pix": "10",
                "stack_dither_interval": "2",
                "stack_dither_enable": "true",
                "exp_ms_stack_l": "10000",
                "exp_ms_continuous": "500",
                "focal_pos": "1500",
                "auto_power_off": "false",
                "auto_3ppa_calib": "true",
                "frame_calib": "true",
                "plan_target_af": "false",
                "viewplan_gohome": "false",
                "expert_mode": "false",
                "heater_enable": "false",
                "dark_mode": "false",
                "stack_cont_capt": "false",
                "stack_drizzle2x": "false",
            }

    captured = {"saw_auto_lenhance": False}

    def fake_do_action_device(action, dev_num, parameters, is_schedule=False):
        if action == "method_async":
            return {"ErrorNumber": 0, "Value": {"code": 0}}
        params = parameters.get("params", {})
        if "auto_lenhance" in params:
            captured["saw_auto_lenhance"] = True
        return {"ErrorNumber": 0, "Value": {"code": 0}}

    monkeypatch.setattr(front_app, "get_firmware_ver_int", lambda _tid: 2670)
    monkeypatch.setattr(front_app, "get_device_model", lambda _tid: "Seestar S50")
    monkeypatch.setattr(front_app, "do_action_device", fake_do_action_device)
    monkeypatch.setattr(
        front_app.SettingsResource,
        "render_settings",
        staticmethod(lambda _req, _resp, _tid, _output: None),
    )

    front_app.SettingsResource().on_post(DummyReq(), object(), 1)

    assert captured["saw_auto_lenhance"] is False


def test_auto_lenhance_defaults_to_false_when_absent_from_firmware(monkeypatch):
    # Firmware may not include auto_lenhance in get_setting response even on 7.75+.
    # Defaulting to False ensures the setting renders as a boolean toggle, not "None".
    monkeypatch.setattr(front_app, "check_api_state", lambda _tid: True)
    monkeypatch.setattr(front_app, "get_device_model", lambda _tid: "Seestar S50")
    monkeypatch.setattr(
        front_app,
        "do_action_device",
        lambda action, tid, params, **kw: {
            "Value": {
                "result": {
                    "exp_ms": {"stack_l": 10000, "continuous": 500},
                    "stack_dither": {"pix": 10, "interval": 2, "enable": True},
                    "stack_lenhance": False,
                    # auto_lenhance intentionally absent
                }
            }
        },
    )
    settings = front_app.get_device_settings(1)
    assert settings["auto_lenhance"] is False


def test_home_content_endpoint_returns_non_empty_html(monkeypatch):
    monkeypatch.setattr(
        front_app,
        "get_telescopes_state",
        lambda: [
            {
                "device_num": 1,
                "name": "Seestar Alpha",
                "ip_address": "192.168.11.124",
                "stats": {"View State": "Idle", "Wi-Fi Signal": "-62 dBm"},
            }
        ],
    )
    monkeypatch.setattr(
        front_app,
        "get_context",
        lambda _tid, _req: _minimal_context("home-content", online=True),
    )
    req = DummyHTMXReq(relative_uri="/1/home-content")
    resp = DummyResp()

    front_app.HomeContentResource.on_get(req, resp, telescope_id=1)

    assert "Welcome to the Simple Seestar" in resp.text
    assert "Seestar Alpha" in resp.text


def test_stats_content_endpoint_returns_non_empty_html(monkeypatch):
    front_app.StatsContentResource._last_render_by_key.clear()
    monkeypatch.setattr(
        front_app,
        "get_device_state",
        lambda _tid: {"View State": "Idle", "Wi-Fi Signal": "-62 dBm"},
    )
    monkeypatch.setattr(
        front_app,
        "get_context",
        lambda _tid, _req: _minimal_context("stats-content", online=True),
    )
    req = DummyHTMXReq(relative_uri="/1/stats-content")
    resp = DummyResp()

    front_app.StatsContentResource.on_get(req, resp, telescope_id=1)

    assert "Wi-Fi Signal" in resp.text


def test_guestmode_content_endpoint_handles_sparse_state(monkeypatch):
    front_app.GuestModeContentResource._last_render_by_key.clear()
    monkeypatch.setattr(
        front_app,
        "get_context",
        lambda _tid, _req: _minimal_context("guestmode-content", online=True),
    )
    monkeypatch.setattr(
        front_app,
        "get_guestmode_state",
        lambda _tid: {"guest_mode": False},
    )
    req = DummyHTMXReq(relative_uri="/1/guestmode-content")
    resp = DummyResp()

    front_app.GuestModeContentResource.on_get(req, resp, telescope_id=1)

    assert "Guest mode is unavailable" in resp.text


def test_eventstatus_endpoint_handles_empty_event_result(monkeypatch):
    front_app.EventStatus._last_render_by_key.clear()
    monkeypatch.setattr(
        front_app,
        "get_context",
        lambda _tid, _req: _minimal_context("eventstatus", online=True),
    )
    monkeypatch.setattr(
        front_app,
        "do_action_device",
        lambda *_args, **_kwargs: {"Value": {"result": {}}},
    )
    req = DummyHTMXReq(
        relative_uri="/1/eventstatus",
        params={"action": "goto"},
        headers={
            "User-Agent": "pytest-agent",
            "HX-Current-URL": "http://localhost/1/goto",
        },
    )
    resp = DummyResp()

    front_app.EventStatus.on_get(req, resp, telescope_id=1)

    assert "Current Status of Devices" in resp.text
    assert "No results available." in resp.text


def test_eventstatus_framed_mosaic_uses_same_eventlist_as_mosaic(monkeypatch):
    monkeypatch.setattr(
        front_app,
        "get_context",
        lambda _tid, _req: _minimal_context("eventstatus", online=True),
    )
    monkeypatch.setattr(
        front_app,
        "do_action_device",
        lambda *_args, **_kwargs: {"Value": {"result": {"x": {"state": "idle"}}}},
    )

    def render(action):
        front_app.EventStatus._last_render_by_key.clear()
        req = DummyHTMXReq(
            relative_uri="/1/eventstatus",
            params={"action": action},
            headers={
                "User-Agent": "pytest-agent",
                "HX-Current-URL": f"http://localhost/1/{action}",
            },
        )
        resp = DummyResp()
        front_app.EventStatus.on_get(req, resp, telescope_id=1)
        return resp.text

    mosaic_html = render("mosaic")
    framed_mosaic_html = render("framed_mosaic")

    for expected in ("DarkLibrary", "Stack"):
        assert expected in mosaic_html
        assert expected in framed_mosaic_html


@pytest.mark.parametrize(
    "stack_from_get_setting,stack_from_get_stack_setting,expected_discrete",
    [
        (
            {"save_discrete_frame": False, "save_discrete_ok_frame": True},
            {},
            (False, True),
        ),
        (
            {},
            {"save_discrete_frame": True, "save_discrete_ok_frame": False},
            (True, False),
        ),
    ],
)
def test_get_device_settings_discrete_flags_compat_matrix(
    monkeypatch,
    stack_from_get_setting,
    stack_from_get_stack_setting,
    expected_discrete,
):
    monkeypatch.setattr(front_app, "get_client_master", lambda _tid: True)
    monkeypatch.setattr(front_app, "get_firmware_ver_int", lambda _tid: 2670)
    monkeypatch.setattr(front_app, "get_device_model", lambda _tid: "Seestar S50")

    def fake_method_sync(method, telescope_id=1, **kwargs):
        if method == "get_setting":
            return {
                "stack_dither": {"pix": 10, "interval": 2, "enable": True},
                "exp_ms": {"stack_l": 10000, "continuous": 500},
                "auto_3ppa_calib": True,
                "frame_calib": True,
                "focal_pos": 1500,
                "heater_enable": False,
                "auto_power_off": False,
                "stack_lenhance": False,
                "dark_mode": False,
                "stack_cont_capt": True,
                "stack": {"drizzle2x": False, **stack_from_get_setting},
                "plan": {"target_af": True},
                "viewplan_go_home": True,
                "expert_mode": False,
            }
        if method == "get_stack_setting":
            return stack_from_get_stack_setting
        raise AssertionError(f"Unexpected method call: {method}")

    monkeypatch.setattr(front_app, "method_sync", fake_method_sync)

    settings = front_app.get_device_settings(1)

    assert settings["save_discrete_frame"] is expected_discrete[0]
    assert settings["save_discrete_ok_frame"] is expected_discrete[1]


@pytest.mark.parametrize(
    "template_name,context",
    [
        (
            "partials/home_content.html",
            {
                "telescopes": [
                    {
                        "device_num": 1,
                        "name": "Seestar Alpha",
                        "ip_address": "192.168.11.124",
                        "stats": {},
                    }
                ],
                "version": "test",
            },
        ),
        (
            "partials/guestmode_content.html",
            {
                "online": True,
                "state": {"guest_mode": False},
                "action": "/1/guestmode",
                "version": "test",
            },
        ),
        (
            "eventstatus.html",
            {"results": [], "events": [], "now": "now"},
        ),
    ],
)
def test_sparse_template_contexts_render_without_error(template_name, context):
    template = front_app.fetch_template(template_name)
    html = template.render(**context)
    assert isinstance(html, str)
    assert len(html) > 0


# ---------- Calibrate-motion routes -----------------------------------
#
# These tests exercise the resource handlers directly without spinning up
# a real falcon server. The motion session is patched so the route logic
# is exercised but no streaming-controller thread spawns. End-to-end
# behaviour is verified by tests/test_calibrate_motion.py.


class _DummyJSONReq:
    """Minimal fake of a falcon request with JSON ``media`` and a few
    convenience accessors the calibrate handlers don't use but that
    other handlers in front_app might pick up."""

    def __init__(self, body=None, params=None):
        self.media = {} if body is None else dict(body)
        self._params = params or {}

    def get_param(self, key, default=None):
        return self._params.get(key, default)

    def get_header(self, key):
        return None


class _DummyJSONResp(DummyResp):
    def __init__(self):
        super().__init__()
        self.status = None
        self.text = ""
        self.content_type = None


class _StubMotion:
    """Stand-in for CalibrateMotionSession that records calls without
    spawning a streaming-controller thread."""

    def __init__(self, alive=True):
        self._alive = alive
        self.calls = []
        self.target = (0.0, 0.0)
        self.jog = (0.0, 0.0)

    def is_alive(self):
        return self._alive

    def stop(self, timeout=5.0):
        self._alive = False

    def set_target(self, az, el):
        self.target = (az, el)
        self.calls.append(("set_target", az, el))

    def nudge_target(self, daz, del_):
        self.target = (self.target[0] + daz, self.target[1] + del_)
        self.calls.append(("nudge_target", daz, del_))

    def set_jog(self, az_degs, el_degs):
        self.jog = (az_degs, el_degs)
        self.calls.append(("set_jog", az_degs, el_degs))

    def freeze_jog(self):
        self.jog = (0.0, 0.0)
        self.calls.append(("freeze_jog",))

    def status(self):
        from device.calibrate_motion import MotionStatus

        return MotionStatus(
            active=self._alive,
            phase="track",
            elapsed_s=1.0,
            exit_reason=None if self._alive else "stop_signal",
            target_az_deg=self.target[0],
            target_el_deg=self.target[1],
            cur_cum_az_deg=self.target[0],
            cur_el_deg=self.target[1],
            err_az_deg=0.0,
            err_el_deg=0.0,
            jog_az_degs=self.jog[0],
            jog_el_degs=self.jog[1],
            is_settled=True,
            tick=10,
        )


class _StubMgr:
    """In-process registry that stores stubs by telescope id without
    starting any threads — start() just registers the supplied session."""

    def __init__(self):
        self.sessions = {}
        self.cross_check_should_refuse = False

    def get(self, tid):
        return self.sessions.get(int(tid))

    def is_running(self, tid):
        s = self.get(tid)
        return s is not None and s.is_alive()

    def start(self, session):
        if self.cross_check_should_refuse:
            raise RuntimeError("cross-manager refusal")
        existing = self.sessions.get(int(session.telescope_id))
        if existing is not None and existing.is_alive():
            raise RuntimeError("already in calibrate-motion mode")
        self.sessions[int(session.telescope_id)] = session
        # Mark the session "alive" — for the stub we leave that to the
        # stub's own state. The real session.start() spawns a thread.

    def stop(self, tid):
        s = self.sessions.get(int(tid))
        if s is None:
            return None
        s.stop()
        return s.status()

    def status(self, tid):
        s = self.get(tid)
        return s.status() if s is not None else None


def _patch_motion_manager(monkeypatch):
    """Replace the singleton getter with a fresh stub registry. Returns
    the stub so tests can inspect calls."""
    import device.calibrate_motion as cm

    mgr = _StubMgr()
    monkeypatch.setattr(cm, "get_calibrate_motion_manager", lambda: mgr)
    return mgr


def test_calibrate_motion_start_returns_status(monkeypatch):
    mgr = _patch_motion_manager(monkeypatch)
    # Replace CalibrateMotionSession's start() so we don't spawn a thread.
    import device.calibrate_motion as cm

    original_session_cls = cm.CalibrateMotionSession

    class _NoThreadSession(original_session_cls):
        def start(self):
            self._t_start = 1.0
            self._phase = "track"

        def is_alive(self):
            return True

        def stop(self, timeout=5.0):
            pass

    monkeypatch.setattr(cm, "CalibrateMotionSession", _NoThreadSession)
    # Also patch the symbol where front_app imports it — it imports
    # lazily inside on_post via ``from device.calibrate_motion import``,
    # so the module-level patch is enough.

    req = _DummyJSONReq(body={})
    resp = _DummyJSONResp()
    front_app.CalibrateMotionStartResource.on_post(req, resp, telescope_id=11)
    assert resp.status is not None
    assert "200" in str(resp.status)
    payload = json.loads(resp.text)
    assert payload["active"] is True
    # Manager registry should now hold the session.
    assert mgr.get(11) is not None


def test_calibrate_motion_start_idempotent_returns_existing(monkeypatch):
    mgr = _patch_motion_manager(monkeypatch)
    stub = _StubMotion()
    stub.telescope_id = 12
    mgr.sessions[12] = stub
    req = _DummyJSONReq(body={})
    resp = _DummyJSONResp()
    front_app.CalibrateMotionStartResource.on_post(req, resp, telescope_id=12)
    assert "200" in str(resp.status)
    payload = json.loads(resp.text)
    assert payload["active"] is True


def test_calibrate_motion_nudge_clamps_magnitude(monkeypatch):
    mgr = _patch_motion_manager(monkeypatch)
    stub = _StubMotion()
    stub.telescope_id = 13
    mgr.sessions[13] = stub
    # 100° is far above the bound; the handler should clamp to 5°.
    req = _DummyJSONReq(body={"daz": 100.0, "del": -100.0})
    resp = _DummyJSONResp()
    front_app.CalibrateMotionNudgeResource.on_post(req, resp, telescope_id=13)
    assert "200" in str(resp.status)
    kind, daz, del_ = stub.calls[-1]
    assert kind == "nudge_target"
    assert daz == 5.0  # _MOTION_NUDGE_BOUND_DEG
    assert del_ == -5.0


def test_calibrate_motion_nudge_rejects_nan(monkeypatch):
    mgr = _patch_motion_manager(monkeypatch)
    stub = _StubMotion()
    stub.telescope_id = 14
    mgr.sessions[14] = stub
    req = _DummyJSONReq(body={"daz": float("nan"), "del": 0.0})
    resp = _DummyJSONResp()
    front_app.CalibrateMotionNudgeResource.on_post(req, resp, telescope_id=14)
    assert "400" in str(resp.status)
    payload = json.loads(resp.text)
    assert "invalid" in payload["error"].lower()
    # Stub must not have received a nudge call.
    assert stub.calls == []


def test_calibrate_motion_nudge_404_when_no_session(monkeypatch):
    _patch_motion_manager(monkeypatch)
    req = _DummyJSONReq(body={"daz": 0.001, "del": 0.001})
    resp = _DummyJSONResp()
    front_app.CalibrateMotionNudgeResource.on_post(req, resp, telescope_id=15)
    assert "404" in str(resp.status)


def test_calibrate_motion_jog_clamps_rate(monkeypatch):
    mgr = _patch_motion_manager(monkeypatch)
    stub = _StubMotion()
    stub.telescope_id = 16
    mgr.sessions[16] = stub
    req = _DummyJSONReq(body={"az_degs": 99.0, "el_degs": -99.0})
    resp = _DummyJSONResp()
    front_app.CalibrateMotionJogResource.on_post(req, resp, telescope_id=16)
    assert "200" in str(resp.status)
    kind, az, el = stub.calls[-1]
    assert kind == "set_jog"
    assert az == 5.0  # _MOTION_JOG_BOUND_DEGS
    assert el == -5.0


def test_calibrate_motion_freeze_calls_session(monkeypatch):
    mgr = _patch_motion_manager(monkeypatch)
    stub = _StubMotion()
    stub.telescope_id = 17
    mgr.sessions[17] = stub
    req = _DummyJSONReq()
    resp = _DummyJSONResp()
    front_app.CalibrateMotionFreezeResource.on_post(req, resp, telescope_id=17)
    assert "200" in str(resp.status)
    assert ("freeze_jog",) in stub.calls


def test_calibrate_motion_set_target_validates_body(monkeypatch):
    mgr = _patch_motion_manager(monkeypatch)
    stub = _StubMotion()
    stub.telescope_id = 18
    mgr.sessions[18] = stub
    # Missing fields → 400.
    req = _DummyJSONReq(body={"az": 1.0})
    resp = _DummyJSONResp()
    front_app.CalibrateMotionSetTargetResource.on_post(req, resp, telescope_id=18)
    assert "400" in str(resp.status)
    # Valid body → 200.
    req2 = _DummyJSONReq(body={"az": 5.5, "el": 35.0})
    resp2 = _DummyJSONResp()
    front_app.CalibrateMotionSetTargetResource.on_post(req2, resp2, telescope_id=18)
    assert "200" in str(resp2.status)
    assert stub.target == (5.5, 35.0)


def test_calibrate_motion_set_target_rejects_nonfinite(monkeypatch):
    mgr = _patch_motion_manager(monkeypatch)
    stub = _StubMotion()
    stub.telescope_id = 28
    mgr.sessions[28] = stub
    initial_target = stub.target
    for body in (
        {"az": float("nan"), "el": 0.0},
        {"az": 0.0, "el": float("inf")},
    ):
        req = _DummyJSONReq(body=body)
        resp = _DummyJSONResp()
        front_app.CalibrateMotionSetTargetResource.on_post(req, resp, telescope_id=28)
        assert "400" in str(resp.status)
        payload = json.loads(resp.text)
        assert "finite" in payload["error"].lower()
    # Non-finite inputs must not have reached the session.
    assert stub.target == initial_target
    assert all(c[0] != "set_target" for c in stub.calls)


def test_calibrate_motion_start_rejects_nonfinite_initial(monkeypatch):
    _patch_motion_manager(monkeypatch)
    for body in (
        {"initial_az_deg": float("nan"), "initial_el_deg": 0.0},
        {"initial_az_deg": 0.0, "initial_el_deg": float("inf")},
    ):
        req = _DummyJSONReq(body=body)
        resp = _DummyJSONResp()
        front_app.CalibrateMotionStartResource.on_post(req, resp, telescope_id=29)
        assert "400" in str(resp.status)
        payload = json.loads(resp.text)
        assert "finite" in payload["error"].lower()


def test_calibrate_motion_state_returns_inactive_when_no_session(monkeypatch):
    _patch_motion_manager(monkeypatch)
    req = _DummyJSONReq()
    resp = _DummyJSONResp()
    front_app.CalibrateMotionStateResource.on_get(req, resp, telescope_id=19)
    assert "200" in str(resp.status)
    payload = json.loads(resp.text)
    assert payload == {"active": False}


def test_calibrate_motion_stop_when_no_session(monkeypatch):
    _patch_motion_manager(monkeypatch)
    req = _DummyJSONReq()
    resp = _DummyJSONResp()
    front_app.CalibrateMotionStopResource.on_post(req, resp, telescope_id=20)
    # The manager returns None when no session existed; the response
    # body should encode an inactive payload (handler returns 200).
    assert "200" in str(resp.status)
    payload = json.loads(resp.text)
    assert payload == {"active": False}


def test_calibrate_motion_start_409_on_cross_manager_refusal(monkeypatch):
    mgr = _patch_motion_manager(monkeypatch)
    mgr.cross_check_should_refuse = True
    import device.calibrate_motion as cm

    original_session_cls = cm.CalibrateMotionSession

    class _NoThreadSession(original_session_cls):
        def start(self):
            self._phase = "track"

        def is_alive(self):
            return False

    monkeypatch.setattr(cm, "CalibrateMotionSession", _NoThreadSession)
    req = _DummyJSONReq(body={})
    resp = _DummyJSONResp()
    front_app.CalibrateMotionStartResource.on_post(req, resp, telescope_id=21)
    assert "409" in str(resp.status)
    payload = json.loads(resp.text)
    assert "cross-manager" in payload["error"]


# ---------- calibrate_rotation increment values -----------------------


def test_calibrate_rotation_step_buttons_use_one_tenth_increments():
    """Step-size buttons on the calibrate page should expose the
    sub-degree increments {0.005, 0.02, 0.1} (one-tenth of the prior
    {0.05, 0.2, 1.0} set). Default-active button should be the
    middle value (0.02).
    """
    template = front_app.fetch_template("calibrate_rotation.html")
    html = template.render(
        telescope_id=1,
        **_minimal_context("calibrate_rotation", online=True),
    )
    # New increments present (with explicit data-step attribute so the
    # JS picks them up via parseFloat).
    assert 'data-step="0.005"' in html
    assert 'data-step="0.02"' in html
    assert 'data-step="0.1"' in html
    # Old coarse increments should be gone (the visible labels too —
    # otherwise the user sees "±0.2°" and is surprised by 1/10 motion).
    assert 'data-step="0.05"' not in html
    assert 'data-step="0.2"' not in html
    assert 'data-step="1.0"' not in html
    # Visible labels match.
    assert "±0.005°" in html
    assert "±0.02°" in html
    assert "±0.1°" in html
    # The default-active button is the middle (0.02°) so a fresh
    # session lands on a sensible mid-range step. Match the button
    # tag without depending on attribute order so unrelated edits
    # (id, aria-*, etc.) don't break this test.
    button_match = re.search(r"<button\b[^>]*\bdata-step=\"0\.02\"[^>]*>", html)
    assert button_match is not None, "Button with data-step=0.02 not found"
    button_tag = button_match.group(0)
    assert re.search(r'\bclass="[^"]*\bactive\b[^"]*"', button_tag), (
        f"data-step=0.02 button is not active: {button_tag}"
    )


# ---------- CalibrateRotationResource ---------------------------------


def test_calibrate_rotation_page_kicks_off_scenery_view_for_streaming(monkeypatch):
    """Visiting /{id}/calibrate_rotation must idempotently kick the
    firmware into scenery view mode so the MJPEG stream produces frames
    before the user clicks 'Start calibration' — but only after a
    get_view_state read confirms the mount isn't already in an active
    (e.g. star-imaging) view.

    Without this, the live-camera <img id="cal-vid"> sits on a Loading
    frame until the calibration session itself starts (which is when
    `ensure_scenery_mode` is called inside `_connect_mount`)."""
    import time as _time

    calls = []
    kick_event = threading.Event()

    def fake_do_action_device(action, dev_num, parameters, is_schedule=False):
        calls.append((action, dev_num, parameters))
        method = parameters.get("method") if isinstance(parameters, dict) else None
        if action == "method_sync" and method == "get_view_state":
            # Idle mount → no active view → the guard should kick scenery.
            return {"Value": {"result": {}}}
        if action == "method_async" and method == "iscope_start_view":
            kick_event.set()
        return {"Value": "ok"}

    monkeypatch.setattr(front_app, "do_action_device", fake_do_action_device)
    monkeypatch.setattr(
        front_app,
        "get_context",
        lambda _tid, _req: _minimal_context("calibrate_rotation", online=True),
    )
    req = DummyHTMXReq(relative_uri="/1/calibrate_rotation")
    resp = DummyResp()

    front_app.CalibrateRotationResource.on_get(req, resp, telescope_id=1)

    # The kick is dispatched in a daemon thread; wait briefly for it.
    assert kick_event.wait(timeout=2.0), (
        f"scenery iscope_start_view kick never landed within 2s; calls={calls}"
    )
    # Allow a tick for any duplicate kicks that might land after the
    # first — we want to assert exactly one.
    _time.sleep(0.05)

    # A get_view_state guard read must precede the kick.
    assert any(
        c[0] == "method_sync" and c[2].get("method") == "get_view_state" for c in calls
    ), f"expected a get_view_state guard read, got: {calls}"

    # The scenery-view kick must have been issued exactly once.
    matching = [
        c
        for c in calls
        if c[0] == "method_async"
        and c[2].get("method") == "iscope_start_view"
        and c[2].get("params", {}).get("mode") == "scenery"
    ]
    assert len(matching) == 1, (
        f"expected exactly one scenery iscope_start_view kick, got: {calls}"
    )
    assert matching[0][1] == 1

    # Page should still render the calibration UI.
    assert "Calibrate rotation" in resp.text
    assert 'id="cal-vid"' in resp.text


def test_calibrate_rotation_page_skips_scenery_kick_when_star_working(monkeypatch):
    """The scenery kick must be guarded: when get_view_state reports the
    mount is already in an active view (state=="working" — e.g. a star-mode
    imaging session), the page must NOT issue iscope_start_view scenery,
    which would tear that session down."""
    import time as _time

    calls = []
    guard_event = threading.Event()

    def fake_do_action_device(action, dev_num, parameters, is_schedule=False):
        calls.append((action, dev_num, parameters))
        method = parameters.get("method") if isinstance(parameters, dict) else None
        if action == "method_sync" and method == "get_view_state":
            guard_event.set()
            return {"Value": {"result": {"View": {"state": "working"}}}}
        return {"Value": "ok"}

    monkeypatch.setattr(front_app, "do_action_device", fake_do_action_device)
    monkeypatch.setattr(
        front_app,
        "get_context",
        lambda _tid, _req: _minimal_context("calibrate_rotation", online=True),
    )
    req = DummyHTMXReq(relative_uri="/1/calibrate_rotation")
    resp = DummyResp()

    front_app.CalibrateRotationResource.on_get(req, resp, telescope_id=1)

    # Wait for the guard read to land in the daemon thread, then give any
    # (erroneous) kick a chance to fire before asserting it did not.
    assert guard_event.wait(timeout=2.0), (
        f"guard get_view_state read never landed within 2s; calls={calls}"
    )
    _time.sleep(0.1)

    scenery_kicks = [
        c
        for c in calls
        if c[0] == "method_async" and c[2].get("method") == "iscope_start_view"
    ]
    assert scenery_kicks == [], (
        f"scenery kick must be suppressed while mount is working; calls={calls}"
    )
    assert "Calibrate rotation" in resp.text


# ---------- LiveTrackerResource scenery-video kick --------------------


def test_live_tracker_page_kicks_off_scenery_view_for_streaming(monkeypatch):
    """Visiting /{id}/live_tracker must (guarded + star-safe) kick the
    firmware into scenery view so the <img id="lt-vid"> streams frames on
    page load, matching the calibrate page."""
    import time as _time

    calls = []
    kick_event = threading.Event()

    def fake_do_action_device(action, dev_num, parameters, is_schedule=False):
        calls.append((action, dev_num, parameters))
        method = parameters.get("method") if isinstance(parameters, dict) else None
        if action == "method_sync" and method == "get_view_state":
            return {"Value": {"result": {}}}
        if action == "method_async" and method == "iscope_start_view":
            kick_event.set()
        return {"Value": "ok"}

    monkeypatch.setattr(front_app, "do_action_device", fake_do_action_device)
    monkeypatch.setattr(
        front_app,
        "get_context",
        lambda _tid, _req: _minimal_context("live_tracker", online=True),
    )
    req = DummyHTMXReq(relative_uri="/1/live_tracker")
    resp = DummyResp()

    front_app.LiveTrackerResource.on_get(req, resp, telescope_id=1)

    assert kick_event.wait(timeout=2.0), (
        f"scenery iscope_start_view kick never landed within 2s; calls={calls}"
    )
    _time.sleep(0.05)
    matching = [
        c
        for c in calls
        if c[0] == "method_async"
        and c[2].get("method") == "iscope_start_view"
        and c[2].get("params", {}).get("mode") == "scenery"
    ]
    assert len(matching) == 1, (
        f"expected exactly one scenery iscope_start_view kick, got: {calls}"
    )
    assert "Live Tracker" in resp.text
    assert 'id="lt-vid"' in resp.text


def test_live_tracker_page_no_scenery_kick_when_offline(monkeypatch):
    """The scenery kick is gated on the cached online state — an offline
    device must not spawn the kick (and thus never blocks on the request
    timeout)."""
    import time as _time

    calls = []

    def fake_do_action_device(action, dev_num, parameters, is_schedule=False):
        calls.append((action, dev_num, parameters))
        return {"Value": "ok"}

    monkeypatch.setattr(front_app, "do_action_device", fake_do_action_device)
    monkeypatch.setattr(
        front_app,
        "get_context",
        lambda _tid, _req: _minimal_context("live_tracker", online=False),
    )
    req = DummyHTMXReq(relative_uri="/1/live_tracker")
    resp = DummyResp()

    front_app.LiveTrackerResource.on_get(req, resp, telescope_id=1)
    _time.sleep(0.1)
    assert calls == [], f"offline page must not issue any device RPC; calls={calls}"
    assert "Live Tracker" in resp.text


# ---------- Live network stats (video throughput + RPC RTT) -----------


def test_live_stats_resource_returns_video_and_rpc(monkeypatch):
    class FakeImager:
        def video_throughput(self):
            return {
                "window_s": 10.0,
                "bytes_per_s": 1234.0,
                "frames_per_s": 5.0,
                "bytes_total": 100,
                "frames_total": 2,
            }

    class FakeDev:
        def rpc_rtt_stats(self):
            return {
                "count": 3,
                "sample_count": 3,
                "p50_ms": 21.0,
                "p95_ms": 88.0,
                "last_ms": 19.0,
            }

    fake_tel = SimpleNamespace(
        get_seestar_imager=lambda _tid: FakeImager(),
        get_seestar_device=lambda _tid: FakeDev(),
    )
    monkeypatch.setattr(front_app, "telescope", fake_tel)

    resp = _DummyJSONResp()
    front_app.LiveStatsResource.on_get(DummyReq(), resp, telescope_id=7)

    assert resp.status == front_app.falcon.HTTP_200
    assert resp.content_type == "application/json"
    payload = json.loads(resp.text)
    assert payload["telescope_id"] == 7
    assert payload["video"]["bytes_per_s"] == 1234.0
    assert payload["video"]["frames_per_s"] == 5.0
    assert payload["rpc"]["p50_ms"] == 21.0
    assert payload["rpc"]["p95_ms"] == 88.0


def test_live_stats_resource_null_sections_when_device_missing(monkeypatch):
    def boom(_tid):
        raise KeyError(_tid)

    fake_tel = SimpleNamespace(get_seestar_imager=boom, get_seestar_device=boom)
    monkeypatch.setattr(front_app, "telescope", fake_tel)

    resp = _DummyJSONResp()
    front_app.LiveStatsResource.on_get(DummyReq(), resp, telescope_id=1)

    # Missing/not-yet-started device must not 500 — stay 200 with null sections
    # so the readout renders placeholders.
    assert resp.status == front_app.falcon.HTTP_200
    payload = json.loads(resp.text)
    assert payload["telescope_id"] == 1
    assert payload["video"] is None
    assert payload["rpc"] is None


def test_net_stats_partial_render_contract():
    template = front_app.fetch_template("partials/net_stats.html")
    html = template.render(telescope_id=4)
    # Card + KPI readout slots present.
    assert 'id="ns-card-4"' in html
    for key in ("video_rate", "video_fps", "rpc_p50", "rpc_p95"):
        assert f'data-ns="{key}"' in html
    # Polls the per-telescope netstats endpoint at a bounded, slow cadence
    # (no HTMX swap, so no destructive empty-swap risk).
    assert "netstats" in html
    assert "POLL_MS = 3000" in html
    assert "setInterval(poll, POLL_MS)" in html


def test_live_tracker_page_embeds_net_stats_readout():
    template = front_app.fetch_template("live_tracker.html")
    html = template.render(
        telescope_id=1,
        **_minimal_context("live_tracker", online=True),
    )
    # The compact network-stats readout (partials/net_stats.html) is embedded
    # on the page and wired to the per-telescope netstats endpoint.
    assert 'id="ns-card-1"' in html
    assert 'data-ns="video_rate"' in html
    assert 'data-ns="rpc_p95"' in html
    assert "netstats" in html
    assert "POLL_MS = 3000" in html


# ---------- LiveVideoKickResource (guarded scenery primer) ------------


def test_live_video_kick_starts_scenery_when_idle(monkeypatch):
    """GET /{id}/live/video_kick refreshes view_state and, when the mount
    is idle, issues the scenery kick, returning 204."""
    calls = []

    def fake_method_sync(method, telescope_id=1, **kwargs):
        calls.append(("method_sync", method, telescope_id, kwargs))
        if method == "get_view_state":
            return {}  # idle → no View → not working
        return None

    def fake_do_action_device(action, dev_num, parameters, is_schedule=False):
        calls.append((action, dev_num, parameters))
        return {"Value": "ok"}

    monkeypatch.setattr(front_app, "check_api_state", lambda _tid: True)
    monkeypatch.setattr(front_app, "method_sync", fake_method_sync)
    monkeypatch.setattr(front_app, "do_action_device", fake_do_action_device)

    req = _DummyJSONReq()
    resp = _DummyJSONResp()
    front_app.LiveVideoKickResource().on_get(req, resp, telescope_id=1)

    assert "204" in str(resp.status)
    kicks = [
        c
        for c in calls
        if c[0] == "method_async"
        and c[2].get("method") == "iscope_start_view"
        and c[2].get("params", {}).get("mode") == "scenery"
    ]
    assert len(kicks) == 1, f"expected one scenery kick; calls={calls}"


def test_live_video_kick_skips_scenery_when_working(monkeypatch):
    """When the mount already reports state=="working", the endpoint must
    refresh view_state but issue no scenery kick (star-safe / idempotent)."""
    calls = []

    def fake_method_sync(method, telescope_id=1, **kwargs):
        calls.append(("method_sync", method, telescope_id, kwargs))
        if method == "get_view_state":
            return {"View": {"state": "working", "stage": "RTSP", "mode": "star"}}
        return None

    def fake_do_action_device(action, dev_num, parameters, is_schedule=False):
        calls.append((action, dev_num, parameters))
        return {"Value": "ok"}

    monkeypatch.setattr(front_app, "check_api_state", lambda _tid: True)
    monkeypatch.setattr(front_app, "method_sync", fake_method_sync)
    monkeypatch.setattr(front_app, "do_action_device", fake_do_action_device)

    req = _DummyJSONReq()
    resp = _DummyJSONResp()
    front_app.LiveVideoKickResource().on_get(req, resp, telescope_id=1)

    assert "204" in str(resp.status)
    # get_view_state was read (refresh), but no scenery start was issued.
    assert any(c[0] == "method_sync" and c[1] == "get_view_state" for c in calls)
    assert not any(c[0] == "method_async" for c in calls), (
        f"working mount must not be re-kicked; calls={calls}"
    )


def test_live_video_kick_offline_is_noop_204(monkeypatch):
    """Offline device: the endpoint returns 204 without touching the mount
    (no get_view_state, no scenery kick)."""
    calls = []

    def fake_method_sync(method, telescope_id=1, **kwargs):
        calls.append(("method_sync", method))
        return {}

    def fake_do_action_device(action, dev_num, parameters, is_schedule=False):
        calls.append((action, dev_num, parameters))
        return {"Value": "ok"}

    monkeypatch.setattr(front_app, "check_api_state", lambda _tid: False)
    monkeypatch.setattr(front_app, "method_sync", fake_method_sync)
    monkeypatch.setattr(front_app, "do_action_device", fake_do_action_device)

    req = _DummyJSONReq()
    resp = _DummyJSONResp()
    front_app.LiveVideoKickResource().on_get(req, resp, telescope_id=1)

    assert "204" in str(resp.status)
    assert calls == [], f"offline kick must be a no-op; calls={calls}"


def test_ensure_scenery_video_guard_matrix(monkeypatch):
    """Unit contract for the shared guarded primer: kick only when the
    fresh get_view_state read shows the mount is not already 'working'."""
    seen = []

    def fake_do_action_device(action, dev_num, parameters, is_schedule=False):
        seen.append((action, parameters))
        return {"Value": "ok"}

    monkeypatch.setattr(front_app, "do_action_device", fake_do_action_device)

    # Idle → kick.
    monkeypatch.setattr(front_app, "method_sync", lambda *a, **k: {})
    seen.clear()
    assert front_app._ensure_scenery_video(1) is True
    assert any(
        a == "method_async" and p.get("params", {}).get("mode") == "scenery"
        for a, p in seen
    )

    # Working → no kick.
    monkeypatch.setattr(
        front_app, "method_sync", lambda *a, **k: {"View": {"state": "working"}}
    )
    seen.clear()
    assert front_app._ensure_scenery_video(1) is False
    assert seen == []

    # result == 0 (well-formed success with no View) → idle → kick.
    monkeypatch.setattr(
        front_app,
        "method_sync",
        lambda *a, **k: {"command": "get_view_state", "status": "success", "result": 0},
    )
    seen.clear()
    assert front_app._ensure_scenery_video(1) is True

    # A failed read must NOT kick: the view state is unknown, and an online
    # mount that is star-imaging must not be switched to scenery because one
    # read failed. Covers "Offline", None, the device-layer timeout string
    # and a firmware error reply.
    failed_reads = [
        "Offline",
        None,
        "Error: Exceeded allotted wait time for result",
        {"command": "get_view_state", "status": "error", "result": "fail"},
    ]
    for bad in failed_reads:
        monkeypatch.setattr(front_app, "method_sync", lambda *a, _b=bad, **k: _b)
        seen.clear()
        assert front_app._ensure_scenery_video(1) is False, bad
        assert seen == [], bad

    # Federation (0) is never kicked and never read.
    reads = []
    monkeypatch.setattr(front_app, "method_sync", lambda *a, **k: reads.append(a) or {})
    seen.clear()
    assert front_app._ensure_scenery_video(0) is False
    assert seen == []
    assert reads == []


def test_live_video_kick_error_reply_does_not_kick(monkeypatch):
    """An online device whose get_view_state returns a firmware error reply
    must not get a scenery kick from the 2 s video_kick poll."""
    calls = []

    def fake_do_action_device(action, dev_num, parameters, is_schedule=False):
        calls.append((action, dev_num, parameters))
        if parameters.get("method") == "get_view_state":
            return {"Value": {"error": "busy", "code": 1}}
        return {"Value": "ok"}

    monkeypatch.setattr(front_app, "check_api_state", lambda _tid: True)
    monkeypatch.setattr(front_app, "do_action_device", fake_do_action_device)

    req = _DummyJSONReq()
    resp = _DummyJSONResp()
    front_app.LiveVideoKickResource().on_get(req, resp, telescope_id=1)

    assert "204" in str(resp.status)
    assert not any(c[0] == "method_async" for c in calls), calls


def test_live_video_kick_federation_is_noop_204(monkeypatch):
    """telescope_id 0 (federation) must not fan a scenery kick out to every
    scope; the endpoint stays a 204 no-op."""
    calls = []

    def fake_do_action_device(action, dev_num, parameters, is_schedule=False):
        calls.append((action, dev_num, parameters))
        return {"Value": {"result": {}}}

    monkeypatch.setattr(front_app, "check_api_state", lambda _tid: True)
    monkeypatch.setattr(front_app, "do_action_device", fake_do_action_device)

    req = _DummyJSONReq()
    resp = _DummyJSONResp()
    front_app.LiveVideoKickResource().on_get(req, resp, telescope_id=0)

    assert "204" in str(resp.status)
    assert calls == []


# ---------- fork routes refuse the federation (telescope_id 0) ----------


def test_reject_federation_target_helper():
    resp = _DummyJSONResp()
    assert front_app._reject_federation_target(resp, 1) is False
    assert resp.status is None
    for bad in (0, -1, "x", None):
        resp = _DummyJSONResp()
        assert front_app._reject_federation_target(resp, bad) is True
        assert "400" in str(resp.status)
        assert json.loads(resp.text) == {"error": "select a single telescope"}


def test_live_tracker_track_rejects_federation(monkeypatch):
    started = []
    _patch_live_tracker_manager(monkeypatch, started.append)
    calls = []
    monkeypatch.setattr(
        front_app, "do_action_device", lambda *a, **k: calls.append(a) or {}
    )
    req = _DummyJSONReq(body={"kind": "file", "id": "sat-123"})
    resp = _DummyJSONResp()
    front_app.LiveTrackerTrackResource.on_post(req, resp, telescope_id=0)
    assert "400" in str(resp.status)
    assert json.loads(resp.text)["error"] == "select a single telescope"
    assert started == []
    assert calls == []


def test_calibration_start_rejects_federation(monkeypatch):
    calls = []
    monkeypatch.setattr(
        front_app, "do_action_device", lambda *a, **k: calls.append(a) or {}
    )
    req = _DummyJSONReq(body={"targets": []})
    resp = _DummyJSONResp()
    front_app.CalibrationStartResource.on_post(req, resp, telescope_id=0)
    assert "400" in str(resp.status)
    assert json.loads(resp.text)["error"] == "select a single telescope"
    assert calls == []


def test_fork_nav_links_hidden_for_federation():
    """The four fork pages are single-telescope only: their nav links are
    rendered for /1 and hidden for the federation (/0)."""
    template = front_app.fetch_template("nav.html")
    base = {
        "telescopes": [
            {"device_num": 1, "name": "Seestar Alpha", "ip_address": "10.0.0.2"}
        ],
        "client_master": True,
        "experimental": False,
        "confirm": False,
        "uitheme": "dark",
        "online": True,
        "segment": "",
        "partial_path": "",
        "imager_root": "",
    }
    fork_paths = (
        "velocity_controller",
        "live_tracker",
        "calibrate_rotation",
        "sky_visibility",
    )
    html1 = template.render(
        root="/1",
        telescope={"device_num": 1, "name": "Seestar Alpha"},
        telescope_id=1,
        **base,
    )
    for path in fork_paths:
        assert f"/1/{path}" in html1
    html0 = template.render(
        root="/0",
        telescope={"device_num": 0, "name": "Federation"},
        telescope_id=0,
        **base,
    )
    for path in fork_paths:
        assert f"/0/{path}" not in html0


# ---------- live-view joystick (/position) sun-safety gate ----------


class _FakeJoystickDev:
    def __init__(self, connected=True, ok=True):
        self.is_connected = connected
        self.ra = 1.0
        self.dec = 2.0
        self._ok = ok
        self.moves = []

    def move_scope(self, in_angle, in_speed, in_dur=3):
        self.moves.append((in_angle, in_speed, in_dur))
        return self._ok


def _patch_joystick(monkeypatch, dev, locked=False, jogging=False):
    import device.sun_safety as ss

    calls = []
    jog_queries = []

    def fake_do_action_device(action, dev_num, parameters, is_schedule=False):
        calls.append((action, dev_num, parameters))
        return {"Value": {"result": 0}}

    def fake_jog(telescope_id=None):
        jog_queries.append(telescope_id)
        return jogging

    monkeypatch.setattr(front_app, "do_action_device", fake_do_action_device)
    monkeypatch.setattr(front_app, "method_sync", lambda *a, **k: None)
    monkeypatch.setattr(front_app.telescope, "seestar_dev", {1: dev})
    monkeypatch.setattr(ss, "sun_safety_is_locked_out", lambda: locked)
    monkeypatch.setattr(ss, "sun_safety_jog_in_progress", fake_jog)
    return calls, jog_queries


def _speed_moves(calls):
    return [
        c
        for c in calls
        if c[0] == "method_sync" and c[2].get("method") == "scope_speed_move"
    ]


def test_position_refuses_motion_during_sun_lockout(monkeypatch):
    dev = _FakeJoystickDev()
    calls, _ = _patch_joystick(monkeypatch, dev, locked=True)
    req = _DummyJSONReq(body={"distance": 50, "angle": 90, "force": 1})
    resp = _DummyJSONResp()
    front_app.TelescopePositionResource().on_post(req, resp, telescope_id=1)
    assert "200" in str(resp.status)
    assert dev.moves == []
    assert _speed_moves(calls) == []
    assert "lockout" in resp.text


def test_position_stop_suppressed_during_sun_jog(monkeypatch):
    dev = _FakeJoystickDev()
    calls, jog_queries = _patch_joystick(monkeypatch, dev, jogging=True)
    req = _DummyJSONReq(body={"distance": 0, "angle": 0, "force": 0})
    resp = _DummyJSONResp()
    front_app.TelescopePositionResource().on_post(req, resp, telescope_id=1)
    assert _speed_moves(calls) == []
    assert jog_queries == [1]


def test_position_stop_sent_when_no_jog(monkeypatch):
    dev = _FakeJoystickDev()
    calls, _ = _patch_joystick(monkeypatch, dev)
    req = _DummyJSONReq(body={"distance": 0, "angle": 0, "force": 0})
    resp = _DummyJSONResp()
    front_app.TelescopePositionResource().on_post(req, resp, telescope_id=1)
    stops = _speed_moves(calls)
    assert len(stops) == 1
    assert stops[0][2]["params"]["speed"] == 0


def test_position_motion_routes_through_device_move_scope(monkeypatch):
    dev = _FakeJoystickDev()
    calls, _ = _patch_joystick(monkeypatch, dev)
    req = _DummyJSONReq(body={"distance": 50, "angle": 90.7, "force": 1})
    resp = _DummyJSONResp()
    front_app.TelescopePositionResource().on_post(req, resp, telescope_id=1)
    assert dev.moves == [(90, min(50 * 14.4 * 1, 1440.0), 3)]
    assert _speed_moves(calls) == []


def test_position_reports_device_guard_refusal(monkeypatch):
    dev = _FakeJoystickDev(ok=False)
    calls, _ = _patch_joystick(monkeypatch, dev)
    req = _DummyJSONReq(body={"distance": 50, "angle": 90, "force": 1})
    resp = _DummyJSONResp()
    front_app.TelescopePositionResource().on_post(req, resp, telescope_id=1)
    assert len(dev.moves) == 1
    assert _speed_moves(calls) == []
    assert "refused" in resp.text


def test_position_falls_back_to_method_sync_without_in_process_device(monkeypatch):
    """A device that is not connected in-process keeps the original raw
    method_sync path (compatibility fallback)."""
    dev = _FakeJoystickDev(connected=False)
    calls, _ = _patch_joystick(monkeypatch, dev)
    req = _DummyJSONReq(body={"distance": 50, "angle": 90, "force": 1})
    resp = _DummyJSONResp()
    front_app.TelescopePositionResource().on_post(req, resp, telescope_id=1)
    assert dev.moves == []
    moves = _speed_moves(calls)
    assert len(moves) == 1
    assert moves[0][2]["params"]["angle"] == 90


def test_position_device_commands_run_under_the_telescope_lock(monkeypatch):
    """The page sends the release (stop) without waiting for an in-flight
    move, so the move (with its sun pre-flight RPC) and the stop both run
    under one per-telescope lock: the stop then reaches the mount after the
    move instead of before it."""
    held = []

    class _Dev(_FakeJoystickDev):
        def move_scope(self, in_angle, in_speed, in_dur=3):
            held.append(front_app._joystick_lock_for(1).locked())
            return super().move_scope(in_angle, in_speed, in_dur)

    def fake_do_action_device(action, dev_num, parameters, is_schedule=False):
        held.append(front_app._joystick_lock_for(1).locked())
        return {"Value": {"result": 0}}

    dev = _Dev()
    _patch_joystick(monkeypatch, dev)
    monkeypatch.setattr(front_app, "do_action_device", fake_do_action_device)
    move = _DummyJSONReq(body={"distance": 50, "angle": 90, "force": 1})
    resp = _DummyJSONResp()
    front_app.TelescopePositionResource().on_post(move, resp, telescope_id=1)
    stop = _DummyJSONReq(body={"distance": 0, "angle": 0, "force": 0})
    resp = _DummyJSONResp()
    front_app.TelescopePositionResource().on_post(stop, resp, telescope_id=1)
    assert len(dev.moves) == 1
    assert held == [True, True]
    assert not front_app._joystick_lock_for(1).locked()


def test_calibrate_rotation_page_renders_when_streaming_kick_fails(monkeypatch):
    """If the firmware is unreachable, the streaming-kick must not
    block the page render — the user can still drive the calibration
    UI's REST endpoints and the prior/targets calls surface the
    backend-offline state separately."""

    def boom(*_args, **_kwargs):
        raise RuntimeError("firmware offline")

    monkeypatch.setattr(front_app, "do_action_device", boom)
    monkeypatch.setattr(
        front_app,
        "get_context",
        lambda _tid, _req: _minimal_context("calibrate_rotation", online=True),
    )
    req = DummyHTMXReq(relative_uri="/1/calibrate_rotation")
    resp = DummyResp()

    front_app.CalibrateRotationResource.on_get(req, resp, telescope_id=1)
    assert "Calibrate rotation" in resp.text


def test_calibrate_rotation_targets_default_first_is_hyperion(monkeypatch):
    """CalibrationTargetsResource must return Hyperion as the first
    default landmark so the page UI's pre-selection (state.targets[0])
    lands on Hyperion. Regression for the previous behavior, where
    `filter_visible`'s height-based ranking pushed the taller LA
    broadcast tower ahead of Hyperion."""
    from device.config import Config
    from scripts.trajectory.faa_dof import HYPERION_06_000301

    monkeypatch.setattr(Config, "port", 5555)

    class FakeAlpaca:
        def __init__(self, *_args, **_kwargs):
            pass

    monkeypatch.setattr(
        "device.alpaca_client.AlpacaClient",
        FakeAlpaca,
    )
    # Dockweiler Beach — both default landmarks visible from here.
    monkeypatch.setattr(
        "scripts.trajectory.observer.fetch_telescope_lonlat",
        lambda _cli: (33.9615051, -118.4581361),
    )

    class _Req:
        params = {"altitude_m": "2.0"}

    class _Resp:
        status = None
        content_type = None
        text = None

    req = _Req()
    resp = _Resp()
    front_app.CalibrationTargetsResource.on_get(req, resp, telescope_id=1)
    payload = json.loads(resp.text)
    defaults = payload["defaults"]
    assert len(defaults) >= 1
    assert defaults[0]["oas"] == HYPERION_06_000301.oas
    assert "Hypername" not in defaults[0]["name"]  # smoke
    assert "Hyperion" in defaults[0]["name"]


# ---------- Unified /calibration/start --------------------------------


class _StartResp:
    """Minimal ``resp`` stub for falcon-style resource tests. The
    resources we test write ``status`` / ``content_type`` / ``text``
    directly, so this is enough."""

    status = None
    content_type = None
    text = None


class _StartReq:
    """Minimal ``req`` stub. Falcon resources access ``req.media`` for
    JSON; we precompute it here."""

    def __init__(self, media):
        self.media = media


def _patch_unified_start_env(monkeypatch, tmp_path):
    """Common setup for ``CalibrationStartResource.on_post`` tests:
    fake telescope GPS, a tmpdir-rooted calibration JSON, and a
    minimal Alpaca stub.

    Stops any previously-running calibration session so each test
    starts from a clean slate.
    """
    from device.config import Config

    monkeypatch.setattr(Config, "port", 5555)

    class FakeAlpaca:
        def __init__(self, *_a, **_k):
            pass

    monkeypatch.setattr("device.alpaca_client.AlpacaClient", FakeAlpaca)
    # GPS at Dockweiler so DEFAULT_LANDMARKS hit visibility.
    monkeypatch.setattr(
        "scripts.trajectory.observer.fetch_telescope_lonlat",
        lambda _cli: (33.9615051, -118.4581361),
    )
    # Redirect the calibration JSON path so we don't write into
    # device/mount_calibration.json on the dev machine.
    out = tmp_path / "cal.json"
    monkeypatch.setattr(front_app, "_CALIBRATION_JSON_PATH", out)
    # Disable the worker thread's actual mount driving — use dry_run
    # in the body instead. The session still spins up but doesn't
    # talk to the (fake) Alpaca client.
    # Neutralise sun-safety so the celestial slew doesn't fail when
    # the wall-clock places Vega near the sun.
    from device import sun_safety as ss

    monkeypatch.setattr(ss, "is_sun_safe", lambda *a, **kw: (True, ""))

    # Reset any leftover calibration manager from a previous test so
    # the `409` path doesn't trip when we call start() back-to-back.
    from device.rotation_calibration import get_calibration_manager

    mgr = get_calibration_manager()
    for tid in list(mgr._sessions.keys()):
        try:
            mgr.stop(tid)
        except Exception:
            pass
        mgr._sessions.pop(tid, None)
    return out


def test_calibration_start_with_unified_targets_payload(monkeypatch, tmp_path):
    """POST /start with a tagged-union ``targets`` array starts a
    session with kind-aware specs and the status surfaces them."""
    _patch_unified_start_env(monkeypatch, tmp_path)
    body = {
        "altitude_m": 2.0,
        "dry_run": True,
        "targets": [
            {"kind": "faa", "oas": "06-000301"},
            {
                "kind": "celestial",
                "name": "Vega",
                "ra_hours": 18.6157,
                "dec_deg": 38.7837,
                "vmag": 0.03,
                "bayer": "alpha Lyr",
            },
        ],
    }
    req = _StartReq(media=body)
    resp = _StartResp()
    front_app.CalibrationStartResource.on_post(req, resp, telescope_id=1)
    assert resp.status == "200 OK"
    payload = json.loads(resp.text)
    assert payload["active"] is True
    assert payload["n_targets"] == 2
    kinds = [t["kind"] for t in (payload.get("targets") or [])]
    assert kinds == ["faa", "celestial"]


def test_calibration_start_legacy_target_oas_payload_still_works(monkeypatch, tmp_path):
    """The existing ``target_oas`` shape must keep working — the
    resource translates it into FAA-kind specs."""
    _patch_unified_start_env(monkeypatch, tmp_path)
    body = {
        "altitude_m": 2.0,
        "dry_run": True,
        "target_oas": ["06-000301", "06-000177"],
    }
    req = _StartReq(media=body)
    resp = _StartResp()
    front_app.CalibrationStartResource.on_post(req, resp, telescope_id=1)
    assert resp.status == "200 OK"
    payload = json.loads(resp.text)
    assert payload["n_targets"] == 2
    assert payload["targets"][0]["kind"] == "faa"


def test_calibration_start_rejects_both_targets_and_target_oas(monkeypatch, tmp_path):
    _patch_unified_start_env(monkeypatch, tmp_path)
    body = {
        "altitude_m": 2.0,
        "dry_run": True,
        "targets": [{"kind": "faa", "oas": "06-000301"}],
        "target_oas": ["06-000301"],
    }
    req = _StartReq(media=body)
    resp = _StartResp()
    front_app.CalibrationStartResource.on_post(req, resp, telescope_id=1)
    assert resp.status == "400 Bad Request"
    assert "not both" in json.loads(resp.text)["error"]


def test_calibration_start_rejects_empty_targets_list(monkeypatch, tmp_path):
    _patch_unified_start_env(monkeypatch, tmp_path)
    body = {"altitude_m": 2.0, "dry_run": True, "targets": []}
    req = _StartReq(media=body)
    resp = _StartResp()
    front_app.CalibrationStartResource.on_post(req, resp, telescope_id=1)
    assert resp.status == "400 Bad Request"
    assert "at least 2" in json.loads(resp.text)["error"]


def test_calibration_start_rejects_single_target_list(monkeypatch, tmp_path):
    """Sessions with fewer than 2 sightings can never commit
    (`_on_commit` requires ≥2). Reject the request up-front instead of
    starting a session that's stuck."""
    _patch_unified_start_env(monkeypatch, tmp_path)
    body = {
        "altitude_m": 2.0,
        "dry_run": True,
        "targets": [{"kind": "faa", "oas": "06-000301"}],
    }
    req = _StartReq(media=body)
    resp = _StartResp()
    front_app.CalibrationStartResource.on_post(req, resp, telescope_id=1)
    assert resp.status == "400 Bad Request"
    assert "at least 2" in json.loads(resp.text)["error"]


def test_calibration_start_rejects_unknown_kind(monkeypatch, tmp_path):
    _patch_unified_start_env(monkeypatch, tmp_path)
    body = {
        "altitude_m": 2.0,
        "dry_run": True,
        "targets": [
            {"kind": "faa", "oas": "06-000301"},
            {"kind": "ad-hoc", "label": "?"},
        ],
    }
    req = _StartReq(media=body)
    resp = _StartResp()
    front_app.CalibrationStartResource.on_post(req, resp, telescope_id=1)
    assert resp.status == "400 Bad Request"
    assert "unknown kind" in json.loads(resp.text)["error"]


def test_calibration_start_rejects_celestial_missing_radec(monkeypatch, tmp_path):
    _patch_unified_start_env(monkeypatch, tmp_path)
    body = {
        "altitude_m": 2.0,
        "dry_run": True,
        "targets": [
            {"kind": "faa", "oas": "06-000301"},
            {"kind": "celestial", "name": "Vega"},  # missing ra/dec
        ],
    }
    req = _StartReq(media=body)
    resp = _StartResp()
    front_app.CalibrationStartResource.on_post(req, resp, telescope_id=1)
    assert resp.status == "400 Bad Request"
    assert "celestial" in json.loads(resp.text)["error"]


def test_calibration_start_rejects_celestial_non_numeric_vmag(monkeypatch, tmp_path):
    """A non-numeric ``vmag`` must produce a 400, not a 500 from a
    bare ``float()`` call."""
    _patch_unified_start_env(monkeypatch, tmp_path)
    body = {
        "altitude_m": 2.0,
        "dry_run": True,
        "targets": [
            {"kind": "faa", "oas": "06-000301"},
            {
                "kind": "celestial",
                "name": "Vega",
                "ra_hours": 18.6157,
                "dec_deg": 38.7837,
                "vmag": "bright",
            },
        ],
    }
    req = _StartReq(media=body)
    resp = _StartResp()
    front_app.CalibrationStartResource.on_post(req, resp, telescope_id=1)
    assert resp.status == "400 Bad Request"
    assert "vmag" in json.loads(resp.text)["error"]


def test_calibration_start_rejects_unknown_faa_oas(monkeypatch, tmp_path):
    _patch_unified_start_env(monkeypatch, tmp_path)
    body = {
        "altitude_m": 2.0,
        "dry_run": True,
        "targets": [
            {"kind": "faa", "oas": "06-000301"},
            {"kind": "faa", "oas": "ZZ-999999"},
        ],
    }
    req = _StartReq(media=body)
    resp = _StartResp()
    front_app.CalibrationStartResource.on_post(req, resp, telescope_id=1)
    assert resp.status == "400 Bad Request"
    assert "ZZ-999999" in json.loads(resp.text)["error"]


def test_calibration_start_platesolve_without_solver_returns_503(monkeypatch, tmp_path):
    """If the targets array has a platesolve entry but no plate solver
    is configured, the start request must 503 rather than crash on
    first sight."""
    _patch_unified_start_env(monkeypatch, tmp_path)
    # Force ``get_default_plate_solver`` to return UnavailablePlateSolver.
    from device.plate_solver import UnavailablePlateSolver

    monkeypatch.setattr(
        "device.plate_solver.get_default_plate_solver",
        lambda: UnavailablePlateSolver(),
    )
    body = {
        "altitude_m": 2.0,
        "dry_run": True,
        "targets": [
            {"kind": "faa", "oas": "06-000301"},
            {"kind": "platesolve", "label": "free aim 1"},
        ],
    }
    req = _StartReq(media=body)
    resp = _StartResp()
    front_app.CalibrationStartResource.on_post(req, resp, telescope_id=1)
    assert resp.status == "503 Service Unavailable"
    assert "plate solver" in json.loads(resp.text)["error"]


# ---------- Celestial targets endpoint --------------------------------


def test_celestial_targets_returns_visible_pool(monkeypatch):
    """``/calibration/celestial_targets`` returns the visible bright-
    star + planet pool with az/el populated."""
    from device.config import Config

    monkeypatch.setattr(Config, "port", 5555)
    monkeypatch.setattr("device.alpaca_client.AlpacaClient", lambda *a, **k: None)
    monkeypatch.setattr(
        "scripts.trajectory.observer.fetch_telescope_lonlat",
        lambda _cli: (34.0522, -118.2437),
    )

    class _Req:
        params = {"altitude_m": "30.0"}

    class _Resp:
        status = None
        content_type = None
        text = None

    req = _Req()
    resp = _Resp()
    front_app.CalibrationCelestialTargetsResource.on_get(req, resp, telescope_id=1)
    assert resp.status == "200 OK"
    payload = json.loads(resp.text)
    assert "targets" in payload
    assert "observer" in payload
    assert "when_utc" in payload
    # All entries have az/el and a name.
    for t in payload["targets"]:
        assert "name" in t
        assert "az_deg" in t
        assert "el_deg" in t


def test_celestial_targets_partial_pointing_400(monkeypatch):
    """current_az_deg without current_el_deg (or vice versa) is
    ambiguous — server must 400."""
    from device.config import Config

    monkeypatch.setattr(Config, "port", 5555)
    monkeypatch.setattr("device.alpaca_client.AlpacaClient", lambda *a, **k: None)
    monkeypatch.setattr(
        "scripts.trajectory.observer.fetch_telescope_lonlat",
        lambda _cli: (34.0522, -118.2437),
    )

    class _Req:
        params = {"altitude_m": "30.0", "current_az_deg": "100.0"}

    class _Resp:
        status = None
        content_type = None
        text = None

    req = _Req()
    resp = _Resp()
    front_app.CalibrationCelestialTargetsResource.on_get(req, resp, telescope_id=1)
    assert resp.status == "400 Bad Request"


def test_calibrate_nighttime_availability_seestar_online(monkeypatch):
    """When the Seestar onboard solver is selected and the API is
    reachable, the resource must report the solver as available with
    ``solver_path='onboard'`` (so the UI label is meaningful, not the
    misleading ``(... on PATH)`` text from an empty path)."""
    import device.plate_solver as ps

    class _FakeSeestarSolver:
        kind = "seestar"

        def is_available(self):
            return True

    monkeypatch.setattr(
        ps,
        "get_default_plate_solver",
        lambda telescope_id, action_runner: _FakeSeestarSolver(),
    )
    monkeypatch.setattr(front_app, "check_api_state", lambda _tid: True)

    class _Req:
        pass

    class _Resp:
        status = None
        content_type = None
        text = None

    resp = _Resp()
    front_app.CalibrateNighttimeAvailabilityResource.on_get(
        _Req(), resp, telescope_id=1
    )
    payload = json.loads(resp.text)
    assert payload["solver_available"] is True
    assert payload["solver_kind"] == "seestar"
    assert payload["solver_path"] == "onboard"


def test_calibrate_nighttime_availability_seestar_offline(monkeypatch):
    """When the scope is offline, ``check_api_state`` returns False and
    the resource must report unavailable with no solver_path so the UI
    doesn't show a green / "on PATH" indicator."""
    import device.plate_solver as ps

    class _FakeSeestarSolver:
        kind = "seestar"

        def is_available(self):
            return True

    monkeypatch.setattr(
        ps,
        "get_default_plate_solver",
        lambda telescope_id, action_runner: _FakeSeestarSolver(),
    )
    monkeypatch.setattr(front_app, "check_api_state", lambda _tid: False)

    class _Req:
        pass

    class _Resp:
        status = None
        content_type = None
        text = None

    resp = _Resp()
    front_app.CalibrateNighttimeAvailabilityResource.on_get(
        _Req(), resp, telescope_id=1
    )
    payload = json.loads(resp.text)
    assert payload["solver_available"] is False
    assert payload["solver_kind"] == "unavailable"
    assert payload["solver_path"] == ""


def test_calibrate_nighttime_availability_solve_field_branch(monkeypatch):
    """The non-Seestar (workstation) path keeps using
    ``is_available()`` + ``binary_path`` — verify it isn't broken by the
    new Seestar branch."""
    import device.plate_solver as ps

    class _FakeSolveField:
        kind = "solve-field"
        binary_path = "/usr/local/bin/solve-field"

        def is_available(self):
            return True

    monkeypatch.setattr(
        ps,
        "get_default_plate_solver",
        lambda telescope_id, action_runner: _FakeSolveField(),
    )

    class _Req:
        pass

    class _Resp:
        status = None
        content_type = None
        text = None

    resp = _Resp()
    front_app.CalibrateNighttimeAvailabilityResource.on_get(
        _Req(), resp, telescope_id=1
    )
    payload = json.loads(resp.text)
    assert payload["solver_available"] is True
    assert payload["solver_kind"] == "solve-field"
    assert payload["solver_path"] == "/usr/local/bin/solve-field"


# ---------- vendored Chart.js (offline field-Pi) ----------------------


def _render_chart_page(template_name):
    return front_app.fetch_template(template_name).render(
        telescope_id=1,
        **_minimal_context(template_name.replace(".html", ""), online=True),
    )


def test_velocity_controller_uses_vendored_chart_not_cdn():
    """The velocity-controller page must load Chart.js from the vendored
    /public asset (like the other public scripts) rather than jsdelivr, so
    the page JS doesn't abort offline on a field Pi."""
    html = _render_chart_page("velocity_controller.html")
    assert "/public/chart.umd.min.js" in html
    assert "cdn.jsdelivr.net" not in html
    # Chart construction is guarded so a missing/failed Chart still leaves the
    # polling/status loop alive.
    assert "typeof Chart === 'undefined'" in html


def test_live_tracker_uses_vendored_chart_not_cdn():
    """Same vendored-asset contract for the live-tracker page, plus a guard so
    status polling / start / stop / sliders survive a missing Chart."""
    html = _render_chart_page("live_tracker.html")
    assert "/public/chart.umd.min.js" in html
    assert "cdn.jsdelivr.net" not in html
    assert "typeof Chart !== 'undefined'" in html
    # updateChart() no-ops when the chart couldn't be constructed.
    assert "if (!chart) return;" in html


# ---------- threaded front WSGI server --------------------------------


def test_front_server_class_is_threaded():
    """The front app must be served by a ThreadingMixIn WSGI server so a
    long-lived SSE stream doesn't freeze every other request."""
    from socketserver import ThreadingMixIn
    from wsgiref.simple_server import WSGIServer

    assert issubclass(front_app.ThreadingWSGIServer, ThreadingMixIn)
    assert issubclass(front_app.ThreadingWSGIServer, WSGIServer)
    assert front_app.ThreadingWSGIServer.daemon_threads is True


def test_front_main_start_wires_threading_server(monkeypatch):
    """FrontMain.start must hand ThreadingWSGIServer to make_server."""
    captured = {}

    class _FakeHttpd:
        def serve_forever(self):
            raise KeyboardInterrupt

        def shutdown(self):
            captured["shutdown"] = True

    def _fake_make_server(host, port, app, server_class=None, handler_class=None):
        captured["server_class"] = server_class
        captured["handler_class"] = handler_class
        return _FakeHttpd()

    monkeypatch.setattr(front_app, "make_server", _fake_make_server)
    monkeypatch.setattr(front_app, "get_listening_ip", lambda: "127.0.0.1")
    monkeypatch.setattr(front_app, "check_internet_connection", lambda: False)

    front_app.FrontMain().start()

    assert captured["server_class"] is front_app.ThreadingWSGIServer
    assert captured["handler_class"] is front_app.LoggingWSGIRequestHandler
    assert captured.get("shutdown") is True


# ---------- LiveTrackerTrackResource pre-check mapping ----------------


def _patch_live_tracker_manager(monkeypatch, start_impl):
    """Stub device.live_tracker so LiveTrackerTrackResource.on_post reaches
    get_manager().start() without any real catalog / mount / thread work."""
    import device.live_tracker as lt

    class _Catalog:
        def make_provider(self, kind, target_id, mount_frame):
            return object()

        def list_cached(self):
            return []

        def list_live(self):
            return []

    class _Manager:
        def start(self, session):
            start_impl(session)

    monkeypatch.setattr(lt, "get_catalog", lambda: _Catalog())
    monkeypatch.setattr(lt, "load_session_mount_frame", lambda: object())
    monkeypatch.setattr(lt, "get_manager", lambda: _Manager())
    monkeypatch.setattr(lt, "LiveTrackSession", lambda **kw: object())
    return lt


def test_live_tracker_track_precheck_failed_returns_400(monkeypatch):
    """An infeasible trajectory raises device.live_tracker.PreCheckFailed;
    the route must surface it as HTTP 400 with the reason, not a 500."""
    import device.live_tracker as lt

    def _raise(session):
        raise lt.PreCheckFailed(
            "telescope 1 track pre-check failed: cable-wrap hard stop"
        )

    _patch_live_tracker_manager(monkeypatch, _raise)
    req = _DummyJSONReq(body={"kind": "file", "id": "sat-123"})
    resp = _DummyJSONResp()
    front_app.LiveTrackerTrackResource.on_post(req, resp, telescope_id=1)
    assert "400" in str(resp.status)
    payload = json.loads(resp.text)
    assert "pre-check failed" in payload["error"]


def test_live_tracker_track_runtime_error_still_returns_409(monkeypatch):
    """Existing RuntimeError handling (already-tracking) must keep its 409."""

    def _raise(session):
        raise RuntimeError("telescope 1 already tracking; stop first")

    _patch_live_tracker_manager(monkeypatch, _raise)
    req = _DummyJSONReq(body={"kind": "file", "id": "sat-123"})
    resp = _DummyJSONResp()
    front_app.LiveTrackerTrackResource.on_post(req, resp, telescope_id=1)
    assert "409" in str(resp.status)
    payload = json.loads(resp.text)
    assert "already tracking" in payload["error"]


def test_live_tracker_track_precheck_import_fallback_returns_400(monkeypatch):
    """If an older device module lacks PreCheckFailed, the defensive import
    falls back to ValueError and a ValueError from start() still maps to 400."""
    import device.live_tracker as lt

    monkeypatch.delattr(lt, "PreCheckFailed", raising=False)

    def _raise(session):
        raise ValueError("trajectory infeasible")

    _patch_live_tracker_manager(monkeypatch, _raise)
    req = _DummyJSONReq(body={"kind": "file", "id": "sat-123"})
    resp = _DummyJSONResp()
    front_app.LiveTrackerTrackResource.on_post(req, resp, telescope_id=1)
    assert "400" in str(resp.status)
    payload = json.loads(resp.text)
    assert "infeasible" in payload["error"]


# ---------- calibrate_rotation error-body surfacing -------------------


def test_calibrate_rotation_surfaces_server_error_body():
    """The celestial-fetch error handler must await the JSON body before
    reading .error (the old `(await r.json().catch(...).error)` read the
    field off the promise, so server error bodies never surfaced)."""
    html = front_app.fetch_template("calibrate_rotation.html").render(
        telescope_id=1,
        **_minimal_context("calibrate_rotation", online=True),
    )
    assert "const j = await r.json().catch(() => ({}));" in html
    assert "(j.error || r.statusText)" in html
    # The buggy member-access-on-promise form must be gone.
    assert ".catch(()=>({})).error" not in html


# ---------- sky_visibility SSE / polling coordination -----------------


def test_sky_visibility_polling_is_sse_fallback_only():
    """SSE and the 2s/5s polling loops must not run simultaneously: polling
    starts only when SSE errors, stops when SSE (re)connects, and all loops
    clear when the run goes idle."""
    html = front_app.fetch_template("sky_visibility.html").render(
        telescope_id=1,
        **_minimal_context("sky_visibility", online=True),
    )
    # Central router keyed off run state.
    assert "function syncTransports" in html
    # SSE connect drops the fallback polling loops.
    assert "sse.onopen" in html
    # SSE error starts polling only while a run is active.
    assert "if (status.active) startPolling();" in html
    # Boot no longer unconditionally starts polling alongside SSE.
    assert "fetchCells().then(fetchStatus);" in html


def test_sky_visibility_slow_status_poll_survives_healthy_sse():
    """A low-rate status-only poll must keep running while SSE is healthy so
    elapsed/ETA/quality/Stop-countdown don't freeze (the mapper emits no
    periodic status event). It is the deliberate exception to onopen's
    stopPolling(); the fast fallback loops stay off while SSE is up."""
    html = front_app.fetch_template("sky_visibility.html").render(
        telescope_id=1,
        **_minimal_context("sky_visibility", online=True),
    )
    # Dedicated slow-poll lifecycle, driven off the status endpoint only.
    assert "function startSlowStatusPoll" in html
    assert "function stopSlowStatusPoll" in html
    assert "slowStatusPolling = setInterval(fetchStatus, 8000);" in html
    # Started whenever a run is active; torn down when it goes idle.
    assert "startSlowStatusPoll();" in html
    assert "stopSlowStatusPoll();" in html
    # Crucially, sse.onopen drops the fast fallback loops but must NOT stop the
    # slow status poll (that is what keeps the status panel live under SSE).
    onopen = re.search(r"sse\.onopen = \(\) => \{(.*?)\};", html, re.S)
    assert onopen is not None
    assert "stopPolling();" in onopen.group(1)
    assert "stopSlowStatusPoll" not in onopen.group(1)


def test_sky_visibility_surfaces_error_and_refusal_events():
    """onmessage must handle system_error and slew_refused (previously dropped)
    so a sun-safety refusal is surfaced, appending to a client-side error
    buffer that survives status refreshes and triggering an immediate fetch."""
    html = front_app.fetch_template("sky_visibility.html").render(
        telescope_id=1,
        **_minimal_context("sky_visibility", online=True),
    )
    # Both event types are now handled in onmessage.
    assert "data.type === 'system_error'" in html
    assert "data.type === 'slew_refused'" in html
    # Persistent client-side error buffer + combined render.
    assert "let clientErrors" in html
    assert "function pushClientError" in html
    assert "function renderErrors" in html
    # renderErrors combines the mapper's status.errors with the client buffer
    # so a slew_refused (never in status.errors) stays visible.
    assert "base.concat(clientErrors)" in html
    # Each handler surfaces the event and pulls fresh status immediately.
    sys_block = re.search(r"data\.type === 'system_error'.*?return;", html, re.S)
    assert sys_block is not None
    assert "pushClientError" in sys_block.group(0)
    assert "fetchStatus();" in sys_block.group(0)
    refuse_block = re.search(r"data\.type === 'slew_refused'.*?return;", html, re.S)
    assert refuse_block is not None
    assert "pushClientError" in refuse_block.group(0)
    assert "fetchStatus();" in refuse_block.group(0)


# ---------- live_tracker slider vs status-poll race -------------------


def test_live_tracker_status_poll_skips_busy_sliders():
    """The 1 Hz status poll must not overwrite an offset slider the user is
    mid-edit (dragging/focused) or that has a debounced POST pending."""
    html = front_app.fetch_template("live_tracker.html").render(
        telescope_id=1,
        **_minimal_context("live_tracker", online=True),
    )
    assert "function offsetsEditing" in html
    assert "function sliderBusy" in html
    # Status poll guards its slider sync.
    assert "if (s.offsets && !offsetsEditing())" in html
    # Per-slider reconciliation also skips a busy slider.
    assert "if (sliderBusy(sid)) return;" in html
    # Debounced-post flag is cleared when the timer fires so the poll can
    # resume once the user stops editing.
    assert "pendingPost = null;" in html


# ---------- fork video pages: MJPEG stream reconnect ------------------


def test_live_tracker_reconnects_dead_video_stream():
    """The live-tracker <img id="lt-vid"> must self-heal: after /vid closes
    (idle mount → Loading/Idle then generator returns), it re-opens the
    stream with a cache-busted src, throttled, watching img.complete so a
    healthy (still-loading) stream is never interrupted."""
    html = front_app.fetch_template("live_tracker.html").render(
        telescope_id=1,
        **_minimal_context("live_tracker", online=True),
    )
    # Reconnect targets the live-view img and cache-busts so the browser
    # opens a NEW request instead of reusing the closed one.
    assert "document.getElementById('lt-vid')" in html
    assert "vid.src = base + '?t=' + now;" in html
    # Hard errors reconnect.
    assert "vid.addEventListener('error', reconnectVideo);" in html
    # Clean-close watchdog: img.complete is true only when the stream isn't
    # actively loading, so a healthy stream is left alone.
    assert "if (completeStreak >= 2) reconnectVideo();" in html
    # Throttled so the imaging server's worker pool isn't churned.
    assert "const RECONNECT_MS = 3000;" in html
    assert "if (now - lastReconnect < RECONNECT_MS) return;" in html


def test_calibrate_rotation_reconnects_dead_video_stream():
    """Same self-healing contract for the calibrate page's <img id="cal-vid">."""
    html = front_app.fetch_template("calibrate_rotation.html").render(
        telescope_id=1,
        **_minimal_context("calibrate_rotation", online=True),
    )
    assert "const vid = els.vid;" in html
    assert "vid.src = base + '?t=' + now;" in html
    assert "vid.addEventListener('error', reconnectVideo);" in html
    assert "if (completeStreak >= 2) reconnectVideo();" in html
    assert "const RECONNECT_MS = 3000;" in html


def test_calibrate_rotation_polls_video_kick_to_refresh_view_state():
    """The calibrate page has no other get_view_state driver, so it must
    poll /live/video_kick to refresh dev.view_state (and idempotently keep
    scenery alive) — with hx-swap="none" so there's no destructive empty
    swap, and a modest cadence (no 0.5s-style hammering on the shared
    wifi uplink)."""
    html = front_app.fetch_template("calibrate_rotation.html").render(
        telescope_id=1,
        **_minimal_context("calibrate_rotation", online=True),
    )
    driver = re.search(r'<div id="cal-video-kick"[^>]*>', html)
    assert driver is not None, "hidden video_kick driver div missing"
    tag = driver.group(0)
    assert 'hx-get="/1/live/video_kick"' in tag
    assert 'hx-swap="none"' in tag
    assert 'hx-trigger="every 2s"' in tag
    # Must not adopt the 0.5s /live/status cadence on the calibration page.
    assert 'hx-trigger="every 0.5s"' not in html


# ---------- Stream-quality knobs (browser-hop only) -------------------
#
# Two verified-controllable knobs shape the MJPEG stream the imaging server
# sends to the browser: max serve FPS and JPEG encode quality. They are
# BROWSER-HOP only (they do not change what the telescope pushes over its
# wifi uplink), so both the UI labels and the config docs must say so. These
# tests cover the SeestarImaging clamp/serve helpers, the front endpoint, and
# the live_tracker render contract.


def test_clamp_max_stream_fps_bounds():
    from device.seestar_imaging import SeestarImaging as SI

    assert SI._clamp_max_stream_fps(0) == 0.0
    assert SI._clamp_max_stream_fps(-4) == 0.0  # non-positive -> uncapped
    assert SI._clamp_max_stream_fps(10) == 10.0
    assert SI._clamp_max_stream_fps(999) == SI.MAX_STREAM_FPS_CAP  # clamped down
    assert SI._clamp_max_stream_fps(float("inf")) == SI.MAX_STREAM_FPS_CAP
    assert SI._clamp_max_stream_fps(float("nan")) == 0.0  # NaN -> uncapped
    # Non-numeric input raises so the endpoint can reject it with a 400.
    with pytest.raises((TypeError, ValueError)):
        SI._clamp_max_stream_fps("abc")


def test_clamp_jpeg_quality_bounds():
    from device.seestar_imaging import SeestarImaging as SI

    assert SI._clamp_jpeg_quality(95) == 95
    assert SI._clamp_jpeg_quality(5000) == SI.JPEG_QUALITY_MAX
    assert SI._clamp_jpeg_quality(1) == SI.JPEG_QUALITY_MIN  # clamped up
    assert SI._clamp_jpeg_quality(72.6) == 73  # rounded
    with pytest.raises((TypeError, ValueError)):
        SI._clamp_jpeg_quality("nope")


def test_default_stream_quality_falls_back_to_uncapped_q95(monkeypatch):
    from device.config import Config
    from device.seestar_imaging import SeestarImaging

    # get_toml returns the passed-in default => no [stream_quality] section.
    monkeypatch.setattr(Config, "get_toml", lambda s, i, d: d)
    assert SeestarImaging.default_stream_quality() == {
        "max_stream_fps": 0.0,
        "jpeg_quality": 95,
    }


def test_default_stream_quality_reads_and_clamps_config(monkeypatch):
    from device.config import Config
    from device.seestar_imaging import SeestarImaging

    vals = {
        ("stream_quality", "max_stream_fps"): 12,
        ("stream_quality", "jpeg_quality"): 300,  # out of range -> clamped
    }
    monkeypatch.setattr(Config, "get_toml", lambda s, i, d: vals.get((s, i), d))
    out = SeestarImaging.default_stream_quality()
    assert out["max_stream_fps"] == 12.0
    assert out["jpeg_quality"] == 100


def test_default_stream_quality_tolerates_garbage_config(monkeypatch):
    from device.config import Config
    from device.seestar_imaging import SeestarImaging

    # A bad config value must never block imager start-up.
    monkeypatch.setattr(Config, "get_toml", lambda s, i, d: "not-a-number")
    assert SeestarImaging.default_stream_quality() == {
        "max_stream_fps": 0.0,
        "jpeg_quality": 95,
    }


def _bare_imager():
    """A SeestarImaging with no socket/thread — bypasses __init__ so the
    pure knob helpers can be exercised without a live device."""
    from device.seestar_imaging import SeestarImaging

    obj = SeestarImaging.__new__(SeestarImaging)
    obj.max_stream_fps = 0.0
    obj.jpeg_quality = 95
    return obj


def test_set_stream_quality_partial_update_and_clamp():
    obj = _bare_imager()
    # Update only fps; jpeg left untouched.
    assert obj.set_stream_quality(max_stream_fps=15) == {
        "max_stream_fps": 15.0,
        "jpeg_quality": 95,
    }
    # Update only jpeg (clamped); fps retained.
    assert obj.set_stream_quality(jpeg_quality=5000) == {
        "max_stream_fps": 15.0,
        "jpeg_quality": 100,
    }
    assert obj.stream_quality() == {"max_stream_fps": 15.0, "jpeg_quality": 100}


def test_serve_delay_honors_fps_cap():
    obj = _bare_imager()
    # Uncapped -> base delay unchanged (historical behavior).
    obj.max_stream_fps = 0.0
    assert obj._serve_delay(0.001) == 0.001
    assert obj._serve_delay(0.1) == 0.1
    # 10 fps -> 0.1 s floor raises the fast streaming cadence.
    obj.max_stream_fps = 10.0
    assert obj._serve_delay(0.001) == pytest.approx(0.1)
    # 2 fps -> 0.5 s floor exceeds even the preview base.
    obj.max_stream_fps = 2.0
    assert obj._serve_delay(0.1) == pytest.approx(0.5)
    # A cap looser than the base delay never speeds the stream up.
    obj.max_stream_fps = 100.0  # 1/100 = 0.01 < 0.1 preview base
    assert obj._serve_delay(0.1) == 0.1


def test_build_frame_bytes_applies_jpeg_quality():
    """Lowering jpeg_quality must actually shrink the served frame — proves
    the encode quality param is wired into the hot path, not just stored."""
    import numpy as np

    obj = _bare_imager()
    obj.BOUNDARY = b"\r\n--frame\r\n"
    obj.raw_img_size = [None, None]
    # Noise so JPEG quality has a real effect on encoded size.
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, size=(200, 200, 3), dtype=np.uint8)
    obj.jpeg_quality = 100
    big = obj.build_frame_bytes(img, 200, 200)
    obj.jpeg_quality = 10
    small = obj.build_frame_bytes(img, 200, 200)
    assert len(small) < len(big)


def _boom_imager(_tid):
    raise KeyError(_tid)


def test_stream_quality_get_returns_imager_values(monkeypatch):
    class FakeImager:
        def stream_quality(self):
            return {"max_stream_fps": 12.0, "jpeg_quality": 70}

    monkeypatch.setattr(
        front_app,
        "telescope",
        SimpleNamespace(get_seestar_imager=lambda _tid: FakeImager()),
    )
    resp = _DummyJSONResp()
    front_app.LiveStreamQualityResource.on_get(DummyReq(), resp, telescope_id=3)

    assert resp.status == front_app.falcon.HTTP_200
    assert resp.content_type == "application/json"
    payload = json.loads(resp.text)
    assert payload["telescope_id"] == 3
    assert payload["max_stream_fps"] == 12.0
    assert payload["jpeg_quality"] == 70


def test_stream_quality_get_falls_back_to_config_defaults(monkeypatch):
    from device.config import Config

    monkeypatch.setattr(
        front_app, "telescope", SimpleNamespace(get_seestar_imager=_boom_imager)
    )
    monkeypatch.setattr(Config, "get_toml", lambda s, i, d: d)
    resp = _DummyJSONResp()
    front_app.LiveStreamQualityResource.on_get(DummyReq(), resp, telescope_id=1)

    payload = json.loads(resp.text)
    assert payload["max_stream_fps"] == 0.0
    assert payload["jpeg_quality"] == 95


def test_stream_quality_post_applies_and_persists(monkeypatch):
    applied = {}

    class FakeImager:
        def __init__(self):
            self.v = {"max_stream_fps": 0.0, "jpeg_quality": 95}

        def stream_quality(self):
            return dict(self.v)

        def set_stream_quality(self, max_stream_fps=None, jpeg_quality=None):
            if max_stream_fps is not None:
                self.v["max_stream_fps"] = float(max_stream_fps)
            if jpeg_quality is not None:
                self.v["jpeg_quality"] = int(jpeg_quality)
            applied.update(self.v)
            return dict(self.v)

    imager = FakeImager()
    monkeypatch.setattr(
        front_app,
        "telescope",
        SimpleNamespace(get_seestar_imager=lambda _tid: imager),
    )
    persisted = {}
    monkeypatch.setattr(
        front_app, "_persist_stream_quality", lambda vals: persisted.update(vals)
    )

    req = _DummyJSONReq(body={"max_stream_fps": 8, "jpeg_quality": 60})
    resp = _DummyJSONResp()
    front_app.LiveStreamQualityResource.on_post(req, resp, telescope_id=1)

    assert resp.status == front_app.falcon.HTTP_200
    payload = json.loads(resp.text)
    assert payload["max_stream_fps"] == 8.0
    assert payload["jpeg_quality"] == 60
    # Applied live to the running imager (immediate effect on the serve loop).
    assert applied == {"max_stream_fps": 8.0, "jpeg_quality": 60}
    # Persisted for restart survival.
    assert persisted == {"max_stream_fps": 8.0, "jpeg_quality": 60}


def test_stream_quality_post_clamps_out_of_range(monkeypatch):
    from device.config import Config
    from device.seestar_imaging import SeestarImaging as SI

    monkeypatch.setattr(
        front_app, "telescope", SimpleNamespace(get_seestar_imager=_boom_imager)
    )
    monkeypatch.setattr(Config, "get_toml", lambda s, i, d: d)
    monkeypatch.setattr(front_app, "_persist_stream_quality", lambda vals: None)

    req = _DummyJSONReq(body={"max_stream_fps": 999, "jpeg_quality": 5000})
    resp = _DummyJSONResp()
    front_app.LiveStreamQualityResource.on_post(req, resp, telescope_id=1)

    payload = json.loads(resp.text)
    assert payload["max_stream_fps"] == SI.MAX_STREAM_FPS_CAP
    assert payload["jpeg_quality"] == 100


def test_stream_quality_post_rejects_non_dict_body(monkeypatch):
    class _ListMediaReq:
        media = [1, 2, 3]

        def get_header(self, _k):
            return None

    resp = _DummyJSONResp()
    front_app.LiveStreamQualityResource.on_post(_ListMediaReq(), resp, telescope_id=1)
    assert "400" in str(resp.status)


def test_stream_quality_post_rejects_invalid_value(monkeypatch):
    monkeypatch.setattr(
        front_app, "telescope", SimpleNamespace(get_seestar_imager=_boom_imager)
    )
    # Non-numeric jpeg_quality must 400, not 500 or silently coerce.
    req = _DummyJSONReq(body={"jpeg_quality": "abc"})
    resp = _DummyJSONResp()
    front_app.LiveStreamQualityResource.on_post(req, resp, telescope_id=1)
    assert "400" in str(resp.status)


def test_persist_stream_quality_writes_config_section(monkeypatch):
    from device.config import Config

    saved = {"called": False}
    monkeypatch.setattr(
        Config, "save_toml", lambda *a, **k: saved.__setitem__("called", True)
    )
    had_section = "stream_quality" in Config._dict
    try:
        front_app._persist_stream_quality({"max_stream_fps": 14.0, "jpeg_quality": 66})
        assert saved["called"] is True
        # Real set_toml round-trips through the tomlkit dict.
        assert Config.get_toml("stream_quality", "max_stream_fps", None) == 14.0
        assert Config.get_toml("stream_quality", "jpeg_quality", None) == 66
    finally:
        # Don't leak the section into the shared Config singleton.
        if not had_section and "stream_quality" in Config._dict:
            del Config._dict["stream_quality"]


def test_live_tracker_has_stream_quality_controls():
    """The live_tracker page exposes the two browser-hop stream knobs near the
    video stats, POSTing to /api/<id>/stream_quality, with an honest label and
    user-triggered (not polled) apply."""
    html = front_app.fetch_template("live_tracker.html").render(
        telescope_id=1,
        **_minimal_context("live_tracker", online=True),
    )
    # Card + both sliders present.
    assert 'id="lt-stream-quality"' in html
    assert 'id="lt-sq-fps"' in html
    assert 'id="lt-sq-jpeg"' in html
    # Talks to the stream_quality endpoint.
    assert "/api/${telescopeId}/stream_quality" in html
    # Honest label: shared across all viewers, persistent, and does not
    # relieve the telescope's wifi uplink.
    assert "every" in html and "viewer" in html
    assert "persists across restarts" in html
    assert "saturated uplink" in html
    assert "stop live video" in html
    # Applied on slider release (change), not via a background poll — keeps
    # with AGENTS.md's no-unbounded-polling rule.
    assert "fps.addEventListener('change', apply)" in html
    assert "jpeg.addEventListener('change', apply)" in html


def test_fork_templates_honor_root_path_prefix():
    """Fork pages build their asset and API URLs under {{ root_path }}, so
    they keep working behind a sub-path reverse proxy; with no prefix the
    URLs are unchanged."""
    ctx = _minimal_context("live_tracker", online=True)
    prefixed = front_app.fetch_template("live_tracker.html").render(
        telescope_id=1, root_path="/ssc", **ctx
    )
    assert 'src="/ssc/public/chart.umd.min.js"' in prefixed
    assert "`/ssc/api/${telescopeId}/live_tracker`" in prefixed
    assert "`/ssc/api/${telescopeId}/stream_quality`" in prefixed
    assert "fetch('/ssc/api/' + tid + '/netstats'" in prefixed
    plain = front_app.fetch_template("live_tracker.html").render(telescope_id=1, **ctx)
    assert 'src="/public/chart.umd.min.js"' in plain
    assert "`/api/${telescopeId}/live_tracker`" in plain
    banner = front_app.fetch_template("partials/sun_safety_banner.html").render(
        root_path="/ssc"
    )
    assert 'hx-get="/ssc/api/sun_safety/status"' in banner
    # The dismiss call is prefixed at run time from the page's root-path data
    # attribute (like main.js's sscRootPath), not by Jinja: the banner is on
    # every page and upstream's byte-level root-path test only lets the
    # prefix change URL attributes.
    assert "document.documentElement.dataset.rootPath" in banner
    assert 'fetch(rootPath + "/api/sun_safety/dismiss"' in banner
    assert 'fetch("/api/sun_safety/dismiss"' not in banner


def test_live_tracker_target_rows_do_not_parse_feed_text_as_html():
    """Target names come from the third-party ADS-B feed: the list rows are
    built with textContent, never innerHTML."""
    html = front_app.fetch_template("live_tracker.html").render(
        telescope_id=1,
        **_minimal_context("live_tracker", online=True),
    )
    assert "row.innerHTML" not in html
    assert "nameSpan.textContent = String(t.display_name || id);" in html
    assert "infoSpan.textContent = right;" in html


def test_stream_quality_post_malformed_json_returns_400():
    """A present-but-unparseable body must surface as 400, not masquerade
    as a successful no-op (PR #25 review)."""

    class _BadMediaReq:
        content_length = 12

        @property
        def media(self):
            raise ValueError("bad json")

    resp = _DummyJSONResp()
    front_app.LiveStreamQualityResource.on_post(_BadMediaReq(), resp, telescope_id=1)
    assert resp.status == front_app.falcon.HTTP_400
    assert "not valid JSON" in resp.text


def test_stream_quality_post_empty_body_is_noop_echo(monkeypatch):
    """An omitted/empty body still means 'no changes': 200 + current values."""

    class FakeImager:
        def stream_quality(self):
            return {"max_stream_fps": 8.0, "jpeg_quality": 60}

        def set_stream_quality(self, **kw):
            raise AssertionError("no-op POST must not set anything")

    monkeypatch.setattr(
        front_app,
        "telescope",
        SimpleNamespace(get_seestar_imager=lambda _tid: FakeImager()),
    )
    monkeypatch.setattr(front_app, "_persist_stream_quality", lambda *a, **kw: None)

    class _EmptyReq:
        content_length = 0

    resp = _DummyJSONResp()
    front_app.LiveStreamQualityResource.on_post(_EmptyReq(), resp, telescope_id=1)
    assert resp.status == front_app.falcon.HTTP_200
    payload = json.loads(resp.text)
    assert payload["max_stream_fps"] == 8.0
    assert payload["jpeg_quality"] == 60


# --- check_needs_auth ---


def test_is_emulator_telescope_true_when_flagged(monkeypatch):
    monkeypatch.setattr(
        front_app, "get_telescope", lambda _tid: {"device_num": 1, "is_emulator": True}
    )
    assert front_app.is_emulator_telescope(1) is True


def test_is_emulator_telescope_false_when_absent(monkeypatch):
    monkeypatch.setattr(front_app, "get_telescope", lambda _tid: {"device_num": 1})
    assert front_app.is_emulator_telescope(1) is False


def test_is_emulator_telescope_false_when_no_such_telescope(monkeypatch):
    monkeypatch.setattr(front_app, "get_telescope", lambda _tid: None)
    assert front_app.is_emulator_telescope(1) is False


def test_check_needs_auth_false_for_emulator_telescope_without_network_call(
    monkeypatch,
):
    # The emulator's license signature check can never pass (no real device's
    # signing key is shipped in the repo), so is_emulator must short-circuit
    # before any pi_is_verified round trip -- not just return False after one.
    monkeypatch.setattr(front_app, "is_emulator_telescope", lambda _tid: True)

    def fail_if_called(_tid):
        raise AssertionError("check_api_state must not be called for emulator devices")

    monkeypatch.setattr(front_app, "check_api_state", fail_if_called)
    assert front_app.check_needs_auth(1) is False


def test_check_needs_auth_false_when_offline(monkeypatch):
    monkeypatch.setattr(front_app, "check_api_state", lambda _tid: False)
    assert front_app.check_needs_auth(1) is False


def test_check_needs_auth_false_for_old_firmware(monkeypatch):
    # Old firmware has no pi_is_verified → method_sync returns error dict → no warning
    monkeypatch.setattr(front_app, "check_api_state", lambda _tid: True)
    monkeypatch.setattr(
        front_app,
        "method_sync",
        lambda method, telescope_id=1, **kw: {
            "command": "pi_is_verified",
            "status": "error",
            "result": {"code": -32601, "message": "Method not found"},
        },
    )
    assert front_app.check_needs_auth(1) is False


def test_check_needs_auth_true_when_unverified(monkeypatch):
    # method_sync wraps result:False as {"status":"success","result":False}
    # because False==0 in Python triggers the == 0 branch in method_sync.
    monkeypatch.setattr(front_app, "check_api_state", lambda _tid: True)
    monkeypatch.setattr(
        front_app,
        "method_sync",
        lambda method, telescope_id=1, **kw: {
            "command": "pi_is_verified",
            "status": "success",
            "result": False,
        },
    )
    assert front_app.check_needs_auth(1) is True


def test_check_needs_auth_false_when_verified(monkeypatch):
    # method_sync returns True directly when firmware returns result:True
    monkeypatch.setattr(front_app, "check_api_state", lambda _tid: True)
    monkeypatch.setattr(
        front_app,
        "method_sync",
        lambda method, telescope_id=1, **kw: True,
    )
    assert front_app.check_needs_auth(1) is False


def test_check_needs_auth_false_on_exception(monkeypatch):
    monkeypatch.setattr(front_app, "check_api_state", lambda _tid: True)

    def raise_error(method, telescope_id=1, **kw):
        raise RuntimeError("connection failed")

    monkeypatch.setattr(front_app, "method_sync", raise_error)
    assert front_app.check_needs_auth(1) is False


def test_check_needs_auth_true_when_method_sync_returns_none(monkeypatch):
    # Firmware 7.32+ without auth silently ignores all commands.
    # The front-layer HTTP request times out → do_action_device returns None →
    # method_sync returns None.  With no prior confirmed-auth cache, this looks
    # like the auth gap (ALPACA says connected but firmware won't talk to us).
    front_app._auth_needs_cache.pop(99, None)
    monkeypatch.setattr(front_app, "check_api_state", lambda _tid: True)
    monkeypatch.setattr(
        front_app, "method_sync", lambda method, telescope_id=1, **kw: None
    )
    assert front_app.check_needs_auth(99) is True


def test_check_needs_auth_true_when_device_layer_times_out_faster_than_http(
    monkeypatch,
):
    # Once seestar_device.py's auth-gap fail-fast cap kicks in, the device
    # layer resolves its OWN timeout (~1.5s) faster than this request's HTTP
    # client timeout (Config.timeout, 5s) -- so the HTTP call succeeds and
    # method_sync unwraps the device's "Error: Exceeded allotted wait time for
    # result" string instead of the request itself timing out to None. That
    # string must be treated the same as the None case above, not silently
    # fall through to "no warning needed".
    front_app._auth_needs_cache.pop(98, None)
    monkeypatch.setattr(front_app, "check_api_state", lambda _tid: True)
    monkeypatch.setattr(
        front_app,
        "method_sync",
        lambda method, telescope_id=1, **kw: (
            "Error: Exceeded allotted wait time for result"
        ),
    )
    assert front_app.check_needs_auth(98) is True


def test_check_needs_auth_timeout_preserved_false_when_previously_authenticated(
    monkeypatch,
):
    # Device was confirmed authenticated (cached False), then pi_is_verified times
    # out — device is busy during imaging.  The warning must NOT reappear.
    front_app._auth_needs_cache[88] = (False, 0.0)  # expired entry, was authenticated
    monkeypatch.setattr(front_app, "check_api_state", lambda _tid: True)
    monkeypatch.setattr(
        front_app, "method_sync", lambda method, telescope_id=1, **kw: None
    )
    try:
        assert front_app.check_needs_auth(88) is False
    finally:
        front_app._auth_needs_cache.pop(88, None)


def test_check_needs_auth_timeout_stays_true_when_previously_needed_auth(monkeypatch):
    # Device previously needed auth (cached True), then times out again.
    # Warning must stay visible — don't suppress it.
    front_app._auth_needs_cache[77] = (True, 0.0)  # expired entry, was needing auth
    monkeypatch.setattr(front_app, "check_api_state", lambda _tid: True)
    monkeypatch.setattr(
        front_app, "method_sync", lambda method, telescope_id=1, **kw: None
    )
    try:
        assert front_app.check_needs_auth(77) is True
    finally:
        front_app._auth_needs_cache.pop(77, None)


def test_check_needs_auth_false_when_method_sync_returns_offline(monkeypatch):
    # "Offline" string means the device layer explicitly says the device is offline.
    monkeypatch.setattr(front_app, "check_api_state", lambda _tid: True)
    monkeypatch.setattr(
        front_app, "method_sync", lambda method, telescope_id=1, **kw: "Offline"
    )
    assert front_app.check_needs_auth(1) is False


def test_check_needs_auth_false_when_firmware_returns_bool_true(monkeypatch):
    # Firmware 7.32+ returns result: True (boolean) → method_sync returns True directly
    monkeypatch.setattr(front_app, "check_api_state", lambda _tid: True)
    monkeypatch.setattr(
        front_app, "method_sync", lambda method, telescope_id=1, **kw: True
    )
    assert front_app.check_needs_auth(1) is False


def test_check_needs_auth_true_when_firmware_returns_bool_false(monkeypatch):
    # Bare False boolean (also valid — treated as not-verified)
    monkeypatch.setattr(front_app, "check_api_state", lambda _tid: True)
    monkeypatch.setattr(
        front_app, "method_sync", lambda method, telescope_id=1, **kw: False
    )
    assert front_app.check_needs_auth(1) is True


def _auth_status_app():
    app = falcon.App()
    app.add_route("/{telescope_id:int}/auth-status", front_app.AuthStatusResource())
    return app


def test_auth_status_returns_204_no_swap_when_no_warning(monkeypatch):
    # When no auth warning is needed, endpoint should return 204 + HX-Reswap: none
    # so HTMX skips the DOM update entirely (no flash).
    monkeypatch.setattr(front_app, "check_needs_auth", lambda _tid: False)
    client = testing.TestClient(_auth_status_app())
    resp = client.simulate_get("/1/auth-status")
    assert resp.status_code == 204
    assert resp.headers.get("hx-reswap") == "none"


def test_auth_status_returns_warning_html_when_needed(monkeypatch):
    monkeypatch.setattr(front_app, "check_needs_auth", lambda _tid: True)
    client = testing.TestClient(_auth_status_app())
    resp = client.simulate_get("/1/auth-status")
    assert resp.status_code == 200
    assert "Authentication required" in resp.text
    assert resp.headers.get("hx-reswap") is None


def test_auth_warning_rendered_in_base_template(monkeypatch):
    context = _minimal_context("stats")
    context["needs_auth_warning"] = True
    context["flashed_messages"] = []
    context["messages"] = []
    context["version"] = "test"
    context["now"] = "now"
    template = front_app.fetch_template("base.html")
    html = template.render(**context)
    assert "Authentication required" in html
    assert "seestar-proxy" in html
    assert "interop_pem" in html


def test_auth_warning_absent_when_not_needed(monkeypatch):
    context = _minimal_context("stats")
    context["needs_auth_warning"] = False
    context["flashed_messages"] = []
    context["messages"] = []
    context["version"] = "test"
    context["now"] = "now"
    template = front_app.fetch_template("base.html")
    html = template.render(**context)
    assert "Authentication required" not in html


def test_form_text_stays_legible_with_default_theme():
    """Regression guard: Bootstrap's .form-text/.text-muted use
    --bs-secondary-color, which used to only get a theme-aware value inside
    `{% if webui_text_color %}`. With the default (unconfigured) theme that
    left it at Bootstrap's stock dark-gray, unreadable against this app's
    dark background -- surfaced by the new, longer mosaic help text added
    for issues #37/#680. --bs-secondary-color must resolve to a visible
    color even when no custom webui_text_color is set."""
    context = _minimal_context("stats")
    context["flashed_messages"] = []
    context["messages"] = []
    context["version"] = "test"
    context["now"] = "now"
    template = front_app.fetch_template("base.html")
    html = template.render(**context)
    assert "--bs-secondary-color: var(--ui-muted);" in html
    assert "--ui-muted: rgba(231, 237, 247, 0.75);" in html


# ---------------------------------------------------------------------------
# Wide angle camera settings – GET (get_device_settings)
# ---------------------------------------------------------------------------


def _s30_get_setting_response():
    return {
        "stack_dither": {"pix": 10, "interval": 2, "enable": True},
        "exp_ms": {"stack_l": 10000, "continuous": 500},
        "auto_3ppa_calib": True,
        "frame_calib": True,
        "focal_pos": 1500,
        "heater_enable": False,
        "auto_power_off": False,
        "stack_lenhance": False,
        "dark_mode": False,
        "stack_cont_capt": False,
        "stack": {"drizzle2x": False},
        "wide_cam": True,
        "wide_4k": False,
        "wide_focal_pos": 1500,
    }


def test_get_device_settings_wide_cam_shown_for_s30_with_experimental(monkeypatch):
    monkeypatch.setattr(front_app, "get_client_master", lambda _tid: True)
    monkeypatch.setattr(front_app, "get_firmware_ver_int", lambda _tid: 2775)
    monkeypatch.setattr(front_app, "get_device_model", lambda _tid: "Seestar S30")
    monkeypatch.setattr(Config, "experimental", True)

    def fake_method_sync(method, telescope_id=1, **kwargs):
        if method == "get_setting":
            return _s30_get_setting_response()
        if method == "get_stack_setting":
            return {"wide_denoise": True}
        raise AssertionError(f"Unexpected: {method}")

    monkeypatch.setattr(front_app, "method_sync", fake_method_sync)

    settings = front_app.get_device_settings(1)

    assert settings["wide_cam"] is True
    assert settings["wide_4k"] is False
    assert settings["wide_focal_pos"] == 1500
    assert settings["wide_denoise"] is True


def test_get_device_settings_wide_cam_shown_for_s30_pro_with_experimental(monkeypatch):
    monkeypatch.setattr(front_app, "get_client_master", lambda _tid: True)
    monkeypatch.setattr(front_app, "get_firmware_ver_int", lambda _tid: 2775)
    monkeypatch.setattr(front_app, "get_device_model", lambda _tid: "Seestar S30 Pro")
    monkeypatch.setattr(Config, "experimental", True)

    def fake_method_sync(method, telescope_id=1, **kwargs):
        if method == "get_setting":
            return _s30_get_setting_response()
        if method == "get_stack_setting":
            return {}
        raise AssertionError(f"Unexpected: {method}")

    monkeypatch.setattr(front_app, "method_sync", fake_method_sync)

    settings = front_app.get_device_settings(1)

    assert "wide_cam" in settings
    assert "wide_4k" in settings


def test_get_device_settings_wide_cam_absent_for_s50(monkeypatch):
    monkeypatch.setattr(front_app, "get_client_master", lambda _tid: True)
    monkeypatch.setattr(front_app, "get_firmware_ver_int", lambda _tid: 2775)
    monkeypatch.setattr(front_app, "get_device_model", lambda _tid: "Seestar S50")
    monkeypatch.setattr(Config, "experimental", True)

    def fake_method_sync(method, telescope_id=1, **kwargs):
        if method == "get_setting":
            return _s30_get_setting_response()
        if method == "get_stack_setting":
            return {}
        raise AssertionError(f"Unexpected: {method}")

    monkeypatch.setattr(front_app, "method_sync", fake_method_sync)

    settings = front_app.get_device_settings(1)

    assert "wide_cam" not in settings
    assert "wide_4k" not in settings
    assert "wide_denoise" not in settings
    assert "wide_focal_pos" not in settings


def test_get_device_settings_wide_cam_absent_when_experimental_off(monkeypatch):
    monkeypatch.setattr(front_app, "get_client_master", lambda _tid: True)
    monkeypatch.setattr(front_app, "get_firmware_ver_int", lambda _tid: 2775)
    monkeypatch.setattr(front_app, "get_device_model", lambda _tid: "Seestar S30")
    monkeypatch.setattr(Config, "experimental", False)

    def fake_method_sync(method, telescope_id=1, **kwargs):
        if method == "get_setting":
            return _s30_get_setting_response()
        if method == "get_stack_setting":
            return {}
        raise AssertionError(f"Unexpected: {method}")

    monkeypatch.setattr(front_app, "method_sync", fake_method_sync)

    settings = front_app.get_device_settings(1)

    assert "wide_cam" not in settings
    assert "wide_4k" not in settings


def test_get_device_settings_wide_cam_omits_none_fields(monkeypatch):
    """Fields the firmware doesn't return should be absent, not present as None."""
    monkeypatch.setattr(front_app, "get_client_master", lambda _tid: True)
    monkeypatch.setattr(front_app, "get_firmware_ver_int", lambda _tid: 2775)
    monkeypatch.setattr(front_app, "get_device_model", lambda _tid: "Seestar S30")
    monkeypatch.setattr(Config, "experimental", True)

    def fake_method_sync(method, telescope_id=1, **kwargs):
        if method == "get_setting":
            # wide_focal_pos and wide_4k absent from firmware response
            resp = _s30_get_setting_response()
            del resp["wide_focal_pos"]
            resp["wide_4k"] = None
            return resp
        if method == "get_stack_setting":
            return {}
        raise AssertionError(f"Unexpected: {method}")

    monkeypatch.setattr(front_app, "method_sync", fake_method_sync)

    settings = front_app.get_device_settings(1)

    assert settings["wide_cam"] is True
    assert "wide_focal_pos" not in settings
    assert "wide_4k" not in settings


def test_get_device_settings_wide_denoise_sourced_from_stack_settings(monkeypatch):
    """wide_denoise is returned by get_stack_setting, not get_setting."""
    monkeypatch.setattr(front_app, "get_client_master", lambda _tid: True)
    monkeypatch.setattr(front_app, "get_firmware_ver_int", lambda _tid: 2775)
    monkeypatch.setattr(front_app, "get_device_model", lambda _tid: "Seestar S30")
    monkeypatch.setattr(Config, "experimental", True)

    def fake_method_sync(method, telescope_id=1, **kwargs):
        if method == "get_setting":
            resp = _s30_get_setting_response()
            # wide_denoise NOT in get_setting response
            return resp
        if method == "get_stack_setting":
            return {"wide_denoise": True, "save_discrete_frame": False}
        raise AssertionError(f"Unexpected: {method}")

    monkeypatch.setattr(front_app, "method_sync", fake_method_sync)

    settings = front_app.get_device_settings(1)

    assert settings["wide_denoise"] is True


# ---------------------------------------------------------------------------
# Wide angle camera settings – POST (SettingsResource.on_post)
# ---------------------------------------------------------------------------


def _base_settings_form():
    """Minimal valid settings POST payload (no wide cam fields)."""
    return {
        "stack_lenhance": "false",
        "stack_dither_pix": "10",
        "stack_dither_interval": "2",
        "stack_dither_enable": "true",
        "exp_ms_stack_l": "10000",
        "exp_ms_continuous": "500",
        "focal_pos": "1500",
        "auto_power_off": "false",
        "auto_3ppa_calib": "true",
        "frame_calib": "true",
        "plan_target_af": "false",
        "viewplan_gohome": "false",
        "expert_mode": "false",
        "heater_enable": "false",
        "dark_mode": "false",
        "stack_cont_capt": "false",
        "stack_drizzle2x": "false",
    }


def _run_settings_post(monkeypatch, model, form_extra=None, fw=2775):
    """Helper: POST to SettingsResource and return (output, captured_calls)."""

    class FormReq:
        def __init__(self):
            self.media = {**_base_settings_form(), **(form_extra or {})}

    captured = {"output": None, "calls": []}

    def fake_do_action_device(action, dev_num, parameters, is_schedule=False):
        captured["calls"].append((action, parameters))
        return {"ErrorNumber": 0, "Value": {"code": 0}}

    monkeypatch.setattr(front_app, "get_firmware_ver_int", lambda _tid: fw)
    monkeypatch.setattr(front_app, "get_device_model", lambda _tid: model)
    monkeypatch.setattr(front_app, "do_action_device", fake_do_action_device)
    monkeypatch.setattr(
        front_app.SettingsResource,
        "render_settings",
        staticmethod(
            lambda _req, _resp, _tid, output: captured.__setitem__("output", output)
        ),
    )

    front_app.SettingsResource().on_post(FormReq(), object(), 1)
    return captured["output"], captured["calls"]


def test_settings_post_wide_cam_fields_sent_for_s30(monkeypatch):
    output, calls = _run_settings_post(
        monkeypatch,
        model="Seestar S30",
        form_extra={
            "wide_cam": "true",
            "wide_4k": "false",
            "wide_denoise": "true",
            "wide_focal_pos": "1600",
        },
    )

    assert output == "Successfully Updated Settings."
    # Gather the flat set_setting params that were sent
    merged = {}
    stack_members = {}
    for action, params in calls:
        if (
            action in ("method_sync", "method_async")
            and params.get("method") == "set_setting"
        ):
            p = params.get("params", {})
            if isinstance(p, dict):
                merged.update(p)
                stack_members.update(p.get("stack", {}))

    assert merged.get("wide_cam") is True
    assert merged.get("wide_4k") is False
    # A stack member; top-level wide_denoise gets 109 (seestar-spec, READ).
    assert "wide_denoise" not in merged
    assert stack_members.get("wide_denoise") is True
    assert merged.get("wide_focal_pos") == 1600


def test_settings_post_wide_cam_not_sent_for_s50(monkeypatch):
    output, calls = _run_settings_post(
        monkeypatch,
        model="Seestar S50",
        form_extra={
            "wide_cam": "true",
            "wide_4k": "true",
            "wide_denoise": "true",
            "wide_focal_pos": "1600",
        },
    )

    assert output == "Successfully Updated Settings."
    merged = {}
    for action, params in calls:
        if (
            action in ("method_sync", "method_async")
            and params.get("method") == "set_setting"
        ):
            p = params.get("params", {})
            if isinstance(p, dict):
                merged.update(p)

    assert "wide_cam" not in merged
    assert "wide_4k" not in merged
    assert "wide_denoise" not in merged
    assert "wide_focal_pos" not in merged


def test_settings_post_wide_cam_absent_fields_not_sent(monkeypatch):
    """S30 form without wide cam fields should not error and should not send them."""
    output, calls = _run_settings_post(
        monkeypatch,
        model="Seestar S30",
        form_extra={},  # no wide cam keys
    )

    assert output == "Successfully Updated Settings."
    merged = {}
    for action, params in calls:
        if (
            action in ("method_sync", "method_async")
            and params.get("method") == "set_setting"
        ):
            p = params.get("params", {})
            if isinstance(p, dict):
                merged.update(p)

    assert "wide_cam" not in merged
    assert "wide_focal_pos" not in merged


def test_settings_post_wide_focal_pos_bad_value_coerced_to_zero(monkeypatch):
    """Non-numeric wide_focal_pos is safely coerced to 0."""
    output, calls = _run_settings_post(
        monkeypatch,
        model="Seestar S30",
        form_extra={"wide_focal_pos": "not_a_number"},
    )

    assert output == "Successfully Updated Settings."
    merged = {}
    for action, params in calls:
        if (
            action in ("method_sync", "method_async")
            and params.get("method") == "set_setting"
        ):
            p = params.get("params", {})
            if isinstance(p, dict):
                merged.update(p)

    assert merged.get("wide_focal_pos") == 0


# ---------------------------------------------------------------------------
# Autofocus-enabled toggle (auto_af) -- present since firmware 2732 (v3.1.2);
# firmware 2846 (v3.3.0) additionally gates the new sensor-temp-drift
# auto-refocus check on it (confirmed against the real decompiled firmware
# and the live QEMU emulator, not just the key's presence).
# ---------------------------------------------------------------------------


def test_get_device_settings_shows_auto_af(monkeypatch):
    monkeypatch.setattr(front_app, "get_client_master", lambda _tid: True)
    monkeypatch.setattr(front_app, "get_firmware_ver_int", lambda _tid: 2732)
    monkeypatch.setattr(front_app, "get_device_model", lambda _tid: "Seestar S50")

    def fake_method_sync(method, telescope_id=1, **kwargs):
        if method == "get_setting":
            return {
                "stack_dither": {"pix": 10, "interval": 2, "enable": True},
                "exp_ms": {"stack_l": 10000, "continuous": 500},
                "auto_3ppa_calib": True,
                "frame_calib": True,
                "focal_pos": 1500,
                "heater_enable": False,
                "auto_power_off": False,
                "stack_lenhance": False,
                "dark_mode": False,
                "stack_cont_capt": False,
                "stack": {"drizzle2x": False},
                "auto_af": True,
            }
        if method == "get_stack_setting":
            return {}
        raise AssertionError(f"Unexpected: {method}")

    monkeypatch.setattr(front_app, "method_sync", fake_method_sync)

    settings = front_app.get_device_settings(1)

    assert settings["auto_af"] is True


def test_get_device_settings_omits_auto_af_when_absent(monkeypatch):
    """Firmware may not return auto_af; don't show it as None."""
    monkeypatch.setattr(front_app, "get_client_master", lambda _tid: True)
    monkeypatch.setattr(front_app, "get_firmware_ver_int", lambda _tid: 2732)
    monkeypatch.setattr(front_app, "get_device_model", lambda _tid: "Seestar S50")

    def fake_method_sync(method, telescope_id=1, **kwargs):
        if method == "get_setting":
            return {
                "stack_dither": {"pix": 10, "interval": 2, "enable": True},
                "exp_ms": {"stack_l": 10000, "continuous": 500},
                "auto_3ppa_calib": True,
                "frame_calib": True,
                "focal_pos": 1500,
                "heater_enable": False,
                "auto_power_off": False,
                "stack_lenhance": False,
                "dark_mode": False,
                "stack_cont_capt": False,
                "stack": {"drizzle2x": False},
                # auto_af intentionally absent
            }
        if method == "get_stack_setting":
            return {}
        raise AssertionError(f"Unexpected: {method}")

    monkeypatch.setattr(front_app, "method_sync", fake_method_sync)

    settings = front_app.get_device_settings(1)

    assert "auto_af" not in settings


def test_settings_post_auto_af_sent(monkeypatch):
    output, calls = _run_settings_post(
        monkeypatch,
        model="Seestar S50",
        fw=2846,
        form_extra={"auto_af": "true"},
    )

    assert output == "Successfully Updated Settings."
    merged = {}
    for action, params in calls:
        if (
            action in ("method_sync", "method_async")
            and params.get("method") == "set_setting"
        ):
            p = params.get("params", {})
            if isinstance(p, dict):
                merged.update(p)

    assert merged.get("auto_af") is True


def test_settings_post_auto_af_absent_not_sent(monkeypatch):
    """Form without auto_af should not error and should not send it."""
    output, calls = _run_settings_post(
        monkeypatch,
        model="Seestar S50",
        fw=2846,
        form_extra={},
    )

    assert output == "Successfully Updated Settings."
    merged = {}
    for action, params in calls:
        if (
            action in ("method_sync", "method_async")
            and params.get("method") == "set_setting"
        ):
            p = params.get("params", {})
            if isinstance(p, dict):
                merged.update(p)

    assert "auto_af" not in merged


# ---------------------------------------------------------------------------
# On-device ASCOM Alpaca server toggle (alpaca_enable) -- new in firmware
# v3.3.1 (build 9.16). This is ZWO's own Alpaca REST server running on the
# device itself, separate from seestar_alp's Alpaca API layer. Feature-detect
# by payload presence, like auto_af above, since we don't have a confirmed
# firmware_ver_int threshold for this firmware family's release.
# ---------------------------------------------------------------------------


def test_get_device_settings_shows_alpaca_enable(monkeypatch):
    monkeypatch.setattr(front_app, "get_client_master", lambda _tid: True)
    monkeypatch.setattr(front_app, "get_firmware_ver_int", lambda _tid: 2846)
    monkeypatch.setattr(front_app, "get_device_model", lambda _tid: "Seestar S50")

    def fake_method_sync(method, telescope_id=1, **kwargs):
        if method == "get_setting":
            return {
                "stack_dither": {"pix": 10, "interval": 2, "enable": True},
                "exp_ms": {"stack_l": 10000, "continuous": 500},
                "auto_3ppa_calib": True,
                "frame_calib": True,
                "focal_pos": 1500,
                "heater_enable": False,
                "auto_power_off": False,
                "stack_lenhance": False,
                "dark_mode": False,
                "stack_cont_capt": False,
                "stack": {"drizzle2x": False},
                "alpaca_enable": True,
            }
        if method == "get_stack_setting":
            return {}
        raise AssertionError(f"Unexpected: {method}")

    monkeypatch.setattr(front_app, "method_sync", fake_method_sync)

    settings = front_app.get_device_settings(1)

    assert settings["alpaca_enable"] is True


def test_get_device_settings_omits_alpaca_enable_when_absent(monkeypatch):
    """Firmware may not return alpaca_enable; don't show it as None."""
    monkeypatch.setattr(front_app, "get_client_master", lambda _tid: True)
    monkeypatch.setattr(front_app, "get_firmware_ver_int", lambda _tid: 2846)
    monkeypatch.setattr(front_app, "get_device_model", lambda _tid: "Seestar S50")

    def fake_method_sync(method, telescope_id=1, **kwargs):
        if method == "get_setting":
            return {
                "stack_dither": {"pix": 10, "interval": 2, "enable": True},
                "exp_ms": {"stack_l": 10000, "continuous": 500},
                "auto_3ppa_calib": True,
                "frame_calib": True,
                "focal_pos": 1500,
                "heater_enable": False,
                "auto_power_off": False,
                "stack_lenhance": False,
                "dark_mode": False,
                "stack_cont_capt": False,
                "stack": {"drizzle2x": False},
                # alpaca_enable intentionally absent
            }
        if method == "get_stack_setting":
            return {}
        raise AssertionError(f"Unexpected: {method}")

    monkeypatch.setattr(front_app, "method_sync", fake_method_sync)

    settings = front_app.get_device_settings(1)

    assert "alpaca_enable" not in settings


def test_settings_post_alpaca_enable_sent(monkeypatch):
    output, calls = _run_settings_post(
        monkeypatch,
        model="Seestar S50",
        fw=2846,
        form_extra={"alpaca_enable": "true"},
    )

    assert output == "Successfully Updated Settings."
    merged = {}
    for action, params in calls:
        if (
            action in ("method_sync", "method_async")
            and params.get("method") == "set_setting"
        ):
            p = params.get("params", {})
            if isinstance(p, dict):
                merged.update(p)

    assert merged.get("alpaca_enable") is True


def test_settings_post_alpaca_enable_absent_not_sent(monkeypatch):
    """Form without alpaca_enable should not error and should not send it."""
    output, calls = _run_settings_post(
        monkeypatch,
        model="Seestar S50",
        fw=2846,
        form_extra={},
    )

    assert output == "Successfully Updated Settings."
    merged = {}
    for action, params in calls:
        if (
            action in ("method_sync", "method_async")
            and params.get("method") == "set_setting"
        ):
            p = params.get("params", {})
            if isinstance(p, dict):
                merged.update(p)

    assert "alpaca_enable" not in merged


# ---------------------------------------------------------------------------
# Stack type – form extraction (do_create_image / do_create_mosaic)
# ---------------------------------------------------------------------------


def _make_image_form(extra=None):
    base = {
        "targetName": "Test Target",
        "ra": "10.5",
        "dec": "-5.0",
        "panelTime": "3600",
        "gain": "80",
        "num_tries": "1",
        "retry_wait_s": "300",
    }
    if extra:
        base.update(extra)
    return base


def _make_mosaic_form(extra=None):
    base = {
        "targetName": "Test Target",
        "ra": "10.5",
        "raPanels": "1",
        "dec": "-5.0",
        "decPanels": "1",
        "panelOverlap": "10",
        "panelSelect": "",
        "panelTime": "3600",
        "gain": "80",
        "num_tries": "1",
        "retry_wait_s": "300",
    }
    if extra:
        base.update(extra)
    return base


def _make_framed_mosaic_form(extra=None):
    base = {
        "targetName": "Test Target",
        "ra": "10.5",
        "dec": "-5.0",
        "mosaicScale": "1.5",
        "mosaicAngle": "45",
        "panelTime": "3600",
        "gain": "80",
        "num_tries": "1",
        "retry_wait_s": "300",
    }
    if extra:
        base.update(extra)
    return base


class _FormReq:
    def __init__(self, data):
        self.media = data


@pytest.mark.parametrize(
    "stack_type_input,expected",
    [
        (None, "DeepSky"),
        ("DeepSky", "DeepSky"),
        ("SolarSystem", "SolarSystem"),
        ("MilkyWay", "MilkyWay"),
        ("invalid_type", "DeepSky"),
        ("", "DeepSky"),
        ("solarsystem", "DeepSky"),  # case-sensitive; wrong case falls back
    ],
)
def test_do_create_image_stack_type_extraction(monkeypatch, stack_type_input, expected):
    captured = {}

    def fake_do_action_device(action, dev_num, params, is_schedule=False):
        captured["params"] = params
        return {"ErrorNumber": 0, "Value": {}}

    monkeypatch.setattr(front_app, "do_action_device", fake_do_action_device)

    form = _make_image_form(
        {"stackType": stack_type_input} if stack_type_input is not None else {}
    )
    req = _FormReq(form)
    resp = DummyResp()

    values, errors = front_app.do_create_image(req, resp, False, 1)

    assert not errors
    assert values["stack_type"] == expected


@pytest.mark.parametrize(
    "stack_type_input,expected",
    [
        (None, "DeepSky"),
        ("SolarSystem", "SolarSystem"),
        ("MilkyWay", "MilkyWay"),
        ("bogus", "DeepSky"),
    ],
)
def test_do_create_mosaic_stack_type_extraction(
    monkeypatch, stack_type_input, expected
):
    captured = {}

    def fake_do_action_device(action, dev_num, params, is_schedule=False):
        captured["params"] = params
        return {"ErrorNumber": 0, "Value": {}}

    monkeypatch.setattr(front_app, "do_action_device", fake_do_action_device)

    form = _make_mosaic_form(
        {"stackType": stack_type_input} if stack_type_input is not None else {}
    )
    req = _FormReq(form)
    resp = DummyResp()

    values, errors = front_app.do_create_mosaic(req, resp, False, 1)

    assert not errors
    assert values["stack_type"] == expected


def test_do_create_image_stack_type_included_in_start_mosaic_params(monkeypatch):
    """stack_type must reach the start_mosaic payload sent to the device layer."""
    captured = {}

    def fake_do_action_device(action, dev_num, params, is_schedule=False):
        if action == "start_mosaic":
            captured["start_mosaic_params"] = params
        return {"ErrorNumber": 0, "Value": {}}

    monkeypatch.setattr(front_app, "do_action_device", fake_do_action_device)

    form = _make_image_form({"stackType": "SolarSystem"})
    req = _FormReq(form)
    resp = DummyResp()

    front_app.do_create_image(req, resp, False, 1)

    assert captured.get("start_mosaic_params", {}).get("stack_type") == "SolarSystem"


def test_do_create_framed_mosaic_builds_expected_params(monkeypatch):
    captured = {}

    def fake_do_action_device(action, dev_num, params, is_schedule=False):
        captured["action"] = action
        captured["params"] = params
        return {"ErrorNumber": 0, "Value": {}}

    monkeypatch.setattr(front_app, "do_action_device", fake_do_action_device)

    form = _make_framed_mosaic_form()
    req = _FormReq(form)
    resp = DummyResp()

    values, errors = front_app.do_create_framed_mosaic(req, resp, False, 1)

    assert not errors
    assert captured["action"] == "start_framed_mosaic"
    assert values["mosaic_scale"] == 1.5
    assert values["mosaic_angle"] == 45.0
    assert values["target_name"] == "Test Target"
    assert values["panel_time_sec"] == 3600
    assert values["gain"] == 80


def test_do_create_framed_mosaic_rejects_scale_out_of_range(monkeypatch):
    called = {"device": False}

    def fake_do_action_device(action, dev_num, params, is_schedule=False):
        called["device"] = True
        return {"ErrorNumber": 0, "Value": {}}

    monkeypatch.setattr(front_app, "do_action_device", fake_do_action_device)

    form = _make_framed_mosaic_form({"mosaicScale": "4.5"})
    req = _FormReq(form)
    resp = DummyResp()

    values, errors = front_app.do_create_framed_mosaic(req, resp, False, 1)

    assert "mosaic_scale" in errors
    assert called["device"] is False


def test_do_create_framed_mosaic_rejects_angle_out_of_range(monkeypatch):
    called = {"device": False}

    def fake_do_action_device(action, dev_num, params, is_schedule=False):
        called["device"] = True
        return {"ErrorNumber": 0, "Value": {}}

    monkeypatch.setattr(front_app, "do_action_device", fake_do_action_device)

    form = _make_framed_mosaic_form({"mosaicAngle": "120"})
    req = _FormReq(form)
    resp = DummyResp()

    values, errors = front_app.do_create_framed_mosaic(req, resp, False, 1)

    assert "mosaic_angle" in errors
    assert called["device"] is False


def test_do_create_framed_mosaic_invalid_ra_does_not_call_device(monkeypatch):
    called = {"device": False}

    def fake_do_action_device(action, dev_num, params, is_schedule=False):
        called["device"] = True
        return {"ErrorNumber": 0, "Value": {}}

    monkeypatch.setattr(front_app, "do_action_device", fake_do_action_device)

    form = _make_framed_mosaic_form({"ra": "not_valid"})
    req = _FormReq(form)
    resp = DummyResp()

    values, errors = front_app.do_create_framed_mosaic(req, resp, False, 1)

    assert "ra" in errors
    assert called["device"] is False


def test_do_create_framed_mosaic_schedule_append(monkeypatch):
    captured = {}

    def fake_do_action_device(action, dev_num, params, is_schedule=False):
        captured["action"] = action
        captured["params"] = params
        return {"ErrorNumber": 0, "Value": {}}

    monkeypatch.setattr(front_app, "do_action_device", fake_do_action_device)

    form = _make_framed_mosaic_form({"action": "append"})
    req = _FormReq(form)
    resp = DummyResp()

    front_app.do_create_framed_mosaic(req, resp, True, 1)

    assert captured["action"] == "add_schedule_item"
    assert captured["params"]["action"] == "start_framed_mosaic"
    assert captured["params"]["params"]["mosaic_scale"] == 1.5


def test_framed_mosaic_resource_renders_form(monkeypatch):
    monkeypatch.setattr(
        front_app,
        "get_context",
        lambda telescope_id, req: {
            "online": True,
            "client_master": True,
            "root": "",
            "defgain": 80,
            "telescope": {"device_num": telescope_id},
        },
    )
    monkeypatch.setattr(
        front_app,
        "do_action_device",
        lambda action, dev_num, params, is_schedule=False: {
            "Value": {"state": "stopped"}
        },
    )

    app = falcon.App()
    app.add_route("/{telescope_id:int}/framed_mosaic", front_app.FramedMosaicResource())
    client = testing.TestClient(app)

    resp = client.simulate_get("/1/framed_mosaic")

    assert resp.status_code == 200
    assert "Framed Mosaic" in resp.text
    assert 'id="mosaicScale"' in resp.text
    assert 'id="mosaicAngle"' in resp.text


def test_mosaic_resource_renders_order_and_notation_guidance(monkeypatch):
    """Regression guard for the mosaic clarity cleanup (issues #37/#680):
    the create form must explain panel order, panel-select notation, and
    show a computed total-time readout, without touching Framed Mosaic."""
    monkeypatch.setattr(
        front_app,
        "get_context",
        lambda telescope_id, req: {
            "online": True,
            "client_master": True,
            "root": "",
            "defgain": 80,
            "telescope": {"device_num": telescope_id},
        },
    )
    monkeypatch.setattr(
        front_app,
        "do_action_device",
        lambda action, dev_num, params, is_schedule=False: {
            "Value": {"state": "stopped"}
        },
    )

    app = falcon.App()
    app.add_route("/{telescope_id:int}/mosaic", front_app.MosaicResource())
    client = testing.TestClient(app)

    resp = client.simulate_get("/1/mosaic")

    assert resp.status_code == 200
    assert "Panel Acquisition Time (per panel)" in resp.text
    assert 'id="mosaicTotalTime"' in resp.text
    assert "does not alternate direction row to row" in resp.text
    assert "RA panel number followed by its Dec panel number" in resp.text


def test_mosaic_panel_count_uses_grid_when_no_selection():
    assert front_app.mosaic_panel_count({"ra_num": 3, "dec_num": 2}) == 6


def test_mosaic_panel_count_uses_selected_panels_when_present():
    assert (
        front_app.mosaic_panel_count(
            {"ra_num": 3, "dec_num": 2, "selected_panels": "11;21"}
        )
        == 2
    )


def test_do_create_image_invalid_ra_does_not_propagate_stack_type_to_device(
    monkeypatch,
):
    """Coordinate validation errors should abort before the device call."""
    called = {"device": False}

    def fake_do_action_device(action, dev_num, params, is_schedule=False):
        called["device"] = True
        return {"ErrorNumber": 0, "Value": {}}

    monkeypatch.setattr(front_app, "do_action_device", fake_do_action_device)

    form = _make_image_form({"ra": "not_valid", "stackType": "SolarSystem"})
    req = _FormReq(form)
    resp = DummyResp()

    values, errors = front_app.do_create_image(req, resp, False, 1)

    assert "ra" in errors
    assert not called["device"]


# ---------------------------------------------------------------------------
# Settings page template – wide cam fields rendered / hidden
# ---------------------------------------------------------------------------


def test_settings_template_renders_wide_cam_fields_for_s30(monkeypatch):
    """Wide cam rows appear in the settings page when model is S30 and experimental."""
    context = _minimal_context("settings")
    context["experimental"] = True
    context["settings"] = {
        "wide_cam": True,
        "wide_4k": False,
        "wide_denoise": True,
        "wide_focal_pos": 1500,
        "focal_pos": 1500,
        "heater_enable": False,
        "auto_power_off": False,
        "stack_lenhance": False,
        "dark_mode": False,
    }
    context["settings_friendly_names"] = {k: k for k in context["settings"]}
    context["settings_helper_text"] = {k: "" for k in context["settings"]}
    context["output"] = None
    context["action"] = "/1/settings"
    context["firmware_ver_int"] = 2775
    context["version"] = "test"

    template = front_app.fetch_template("settings.html")
    html = template.render(**context)

    assert "wide_cam" in html
    assert "wide_4k" in html
    assert "wide_denoise" in html
    assert "wide_focal_pos" in html


def test_settings_template_groups_wide_fields_under_imaging(monkeypatch):
    """wide_* settings should land in the Imaging group, not General."""
    context = _minimal_context("settings")
    context["experimental"] = True
    context["settings"] = {
        "wide_cam": True,
        "wide_4k": False,
    }
    context["settings_friendly_names"] = {
        "wide_cam": "Wide Angle Camera",
        "wide_4k": "Wide 4K",
    }
    context["settings_helper_text"] = {"wide_cam": "", "wide_4k": ""}
    context["output"] = None
    context["action"] = "/1/settings"
    context["firmware_ver_int"] = 2775
    context["version"] = "test"

    template = front_app.fetch_template("settings.html")
    html = template.render(**context)

    imaging_pos = html.find("Imaging")
    general_pos = html.find("General")
    wide_cam_pos = html.find("wide_cam")

    # wide_cam should appear after the Imaging header and before (or without) General
    assert imaging_pos != -1
    assert wide_cam_pos > imaging_pos
    if general_pos != -1:
        assert wide_cam_pos < general_pos or general_pos < imaging_pos


# ---------------------------------------------------------------------------
# LiveWideCamResource – wide cam toggle in live view
# ---------------------------------------------------------------------------


def _live_wide_cam_app():
    app = falcon.App()
    app.add_route("/{telescope_id:int}/live/wide-cam", front_app.LiveWideCamResource())
    return app


def test_live_wide_cam_get_returns_204_for_non_s30(monkeypatch):
    monkeypatch.setattr(front_app.Config, "experimental", True)
    monkeypatch.setattr(front_app, "get_device_model", lambda _tid: "Seestar S50")
    client = testing.TestClient(_live_wide_cam_app())
    resp = client.simulate_get("/1/live/wide-cam")
    assert resp.status_code == 204
    assert resp.headers.get("hx-reswap") == "none"


def test_live_wide_cam_get_returns_204_when_experimental_off(monkeypatch):
    monkeypatch.setattr(front_app.Config, "experimental", False)
    monkeypatch.setattr(front_app, "get_device_model", lambda _tid: "Seestar S30")
    client = testing.TestClient(_live_wide_cam_app())
    resp = client.simulate_get("/1/live/wide-cam")
    assert resp.status_code == 204


def test_live_wide_cam_get_returns_toggle_for_s30(monkeypatch):
    monkeypatch.setattr(front_app.Config, "experimental", True)
    monkeypatch.setattr(front_app, "get_device_model", lambda _tid: "Seestar S30")
    monkeypatch.setattr(
        front_app,
        "do_action_device",
        lambda *a, **kw: {"Value": {"result": {"wide_cam": False}}},
    )
    client = testing.TestClient(_live_wide_cam_app())
    resp = client.simulate_get("/1/live/wide-cam")
    assert resp.status_code == 200
    assert "wide-cam-toggle" in resp.text
    assert "Wide Camera" in resp.text


def test_live_wide_cam_get_shows_checked_when_enabled(monkeypatch):
    monkeypatch.setattr(front_app.Config, "experimental", True)
    monkeypatch.setattr(front_app, "get_device_model", lambda _tid: "Seestar S30 Pro")
    monkeypatch.setattr(
        front_app,
        "do_action_device",
        lambda *a, **kw: {"Value": {"result": {"wide_cam": True}}},
    )
    client = testing.TestClient(_live_wide_cam_app())
    resp = client.simulate_get("/1/live/wide-cam")
    assert resp.status_code == 200
    # The checkbox attribute "checked" appears as a standalone HTML attribute
    # when enabled; hx-vals always contains "checked.toString()" regardless.
    assert resp.text.count("checked") >= 2


def test_live_wide_cam_post_sets_true(monkeypatch):
    monkeypatch.setattr(front_app.Config, "experimental", True)
    monkeypatch.setattr(front_app, "get_device_model", lambda _tid: "Seestar S30")
    calls = []
    monkeypatch.setattr(
        front_app,
        "do_action_device",
        lambda action, tid, params, **kw: calls.append(params) or {},
    )
    client = testing.TestClient(_live_wide_cam_app())
    resp = client.simulate_post("/1/live/wide-cam", json={"wide_cam": "true"})
    assert resp.status_code == 200
    assert any(
        p.get("params", {}).get("wide_cam") is True
        for p in calls
        if isinstance(p, dict) and "params" in p
    )
    assert resp.text.count("checked") >= 2  # attribute + hx-vals JS


def test_live_wide_cam_post_sets_false(monkeypatch):
    monkeypatch.setattr(front_app.Config, "experimental", True)
    monkeypatch.setattr(front_app, "get_device_model", lambda _tid: "Seestar S30")
    calls = []
    monkeypatch.setattr(
        front_app,
        "do_action_device",
        lambda action, tid, params, **kw: calls.append(params) or {},
    )
    client = testing.TestClient(_live_wide_cam_app())
    resp = client.simulate_post("/1/live/wide-cam", json={"wide_cam": "false"})
    assert resp.status_code == 200
    assert any(
        p.get("params", {}).get("wide_cam") is False
        for p in calls
        if isinstance(p, dict) and "params" in p
    )
    assert resp.text.count("checked") == 1  # only the hx-vals JS, not the attribute


def test_live_wide_cam_post_returns_204_when_experimental_off(monkeypatch):
    monkeypatch.setattr(front_app.Config, "experimental", False)
    calls = []
    monkeypatch.setattr(
        front_app,
        "do_action_device",
        lambda *a, **kw: calls.append(a),
    )
    client = testing.TestClient(_live_wide_cam_app())
    resp = client.simulate_post("/1/live/wide-cam", json={"wide_cam": "true"})
    assert resp.status_code == 204
    assert resp.headers.get("hx-reswap") == "none"
    assert calls == []


def test_live_wide_cam_post_returns_204_for_non_s30(monkeypatch):
    monkeypatch.setattr(front_app.Config, "experimental", True)
    monkeypatch.setattr(front_app, "get_device_model", lambda _tid: "Seestar S50")
    calls = []
    monkeypatch.setattr(
        front_app,
        "do_action_device",
        lambda *a, **kw: calls.append(a),
    )
    client = testing.TestClient(_live_wide_cam_app())
    resp = client.simulate_post("/1/live/wide-cam", json={"wide_cam": "true"})
    assert resp.status_code == 204
    assert resp.headers.get("hx-reswap") == "none"
    assert calls == []


def test_force_stop_goto_endpoint_returns_json(monkeypatch):
    seen = {}

    def _do_action(action, tid, params):
        seen["action"] = action
        seen["tid"] = tid
        return {"ErrorNumber": 0, "Value": {"ok": True, "stop_slew_result": {}}}

    monkeypatch.setattr(front_app, "do_action_device", _do_action)
    req = DummyHTMXReq(relative_uri="/api/1/force_stop_goto")
    resp = DummyResp()

    front_app.ForceStopGotoResource.on_post(req, resp, telescope_id=1)

    assert seen == {"action": "force_stop_goto", "tid": 1}
    assert resp.status == falcon.HTTP_200
    assert resp.content_type == "application/json"
    body = json.loads(resp.text)
    assert body["Value"]["ok"] is True


def test_force_stop_goto_endpoint_json_when_device_offline(monkeypatch):
    monkeypatch.setattr(front_app, "do_action_device", lambda *_a, **_k: None)
    req = DummyHTMXReq(relative_uri="/api/1/force_stop_goto")
    resp = DummyResp()

    front_app.ForceStopGotoResource.on_post(req, resp, telescope_id=1)

    body = json.loads(resp.text)
    assert body == {"ok": False, "reason": "no response"}


def test_force_stop_goto_endpoint_htmx_fragment(monkeypatch):
    monkeypatch.setattr(
        front_app,
        "do_action_device",
        lambda *_a, **_k: {"ErrorNumber": 0, "Value": {"ok": True}},
    )
    req = DummyHTMXReq(
        relative_uri="/api/1/force_stop_goto", headers={"HX-Request": "true"}
    )
    resp = DummyResp()

    front_app.ForceStopGotoResource.on_post(req, resp, telescope_id=1)

    assert resp.content_type == "text/html"
    assert "Cleared. Mount free." in resp.text


def test_force_stop_goto_endpoint_htmx_fragment_offline(monkeypatch):
    monkeypatch.setattr(front_app, "do_action_device", lambda *_a, **_k: None)
    req = DummyHTMXReq(
        relative_uri="/api/1/force_stop_goto", headers={"HX-Request": "true"}
    )
    resp = DummyResp()

    front_app.ForceStopGotoResource.on_post(req, resp, telescope_id=1)

    assert resp.content_type == "text/html"
    assert "No response from device." in resp.text


def test_goto_target_template_has_htmx_force_stop_button():
    template = front_app.fetch_template("goto_target.html")
    html = template.render(
        telescope={"device_num": 1},
        values={},
        errors={},
        action="/1/goto",
    )
    assert 'id="forceStopGotoBtn"' in html
    assert 'hx-post="/api/1/force_stop_goto"' in html
    assert 'hx-target="#forceStopGotoStatus"' in html
    assert 'id="forceStopGotoStatus"' in html
    # The button must not fall back to hand-rolled fetch() wiring.
    assert "fetch(" not in html


def test_schedule_list_renders_framed_mosaic_item():
    template = front_app.env.get_template("partials/schedule_list.html")
    html = template.render(
        schedule={
            "list": [
                {
                    "schedule_item_id": "fm1",
                    "action": "start_framed_mosaic",
                    "params": {
                        "target_name": "M31",
                        "ra": 0.7,
                        "dec": 41.27,
                        "mosaic_scale": 1.5,
                        "mosaic_angle": 45.0,
                        "panel_time_sec": 3600,
                        "gain": 80,
                        "num_tries": 1,
                        "retry_wait_s": 300,
                    },
                }
            ],
            "is_stacking": False,
        },
        current_item={},
    )

    assert "M31" in html
    assert "Scale: 1.5x" in html
    assert "Angle: 45.0&deg;" in html


def test_schedule_list_shows_total_mosaic_time_and_labeled_current_panel():
    """Regression guard for the mosaic clarity cleanup (issues #37/#680):
    the schedule list must show a computed grand total next to the
    per-panel time, and label the running panel by RA/Dec instead of an
    ambiguous concatenated digit pair."""
    template = front_app.env.get_template("partials/schedule_list.html")
    html = template.render(
        schedule={
            "list": [
                {
                    "schedule_item_id": "x1",
                    "action": "start_mosaic",
                    "params": {
                        "target_name": "M31",
                        "ra": 0.7,
                        "dec": 41.27,
                        "ra_num": 2,
                        "dec_num": 2,
                        "panel_overlap_percent": 10,
                        "panel_time_sec": 3600,
                        "gain": 80,
                        "selected_panels": "",
                        "num_tries": 1,
                        "retry_wait_s": 300,
                    },
                }
            ],
            "is_stacking": True,
        },
        current_item={
            "schedule_item_id": "x1",
            "item_total_time_s": 14400,
            "item_remaining_time_s": 10800,
            "cur_ra_panel_num": 2,
            "cur_dec_panel_num": 1,
        },
    )

    assert "01:00:00 ea." in html
    assert "Total: 04:00:00" in html
    assert "Current Panel: 21" in html
    assert "RA 2 / Dec 1" in html


def _row_column_count(html):
    """Count top-level Bootstrap grid columns (col/col-2/col-auto) in a
    rendered schedule_list.html row, to catch column-count drift between
    action-type branches that share the same table header."""
    return len(re.findall(r'class="col(?:-\d+|-auto)?[ "]', html))


@pytest.mark.parametrize("is_current", [True, False])
def test_schedule_list_framed_mosaic_column_count_matches_mosaic(is_current):
    """Regression guard: a framed_mosaic row must have the same column count
    as a start_mosaic row, since both render under schedule_base.html's one
    shared 10-column header. A mismatch silently shifts every column after
    the divergence point (this exact bug shipped once already)."""
    template = front_app.env.get_template("partials/schedule_list.html")
    current_item = {"schedule_item_id": "x1"} if is_current else {}

    mosaic_html = template.render(
        schedule={
            "list": [
                {
                    "schedule_item_id": "x1",
                    "action": "start_mosaic",
                    "params": {
                        "target_name": "M31",
                        "ra": 0.7,
                        "dec": 41.27,
                        "ra_num": 1,
                        "dec_num": 1,
                        "panel_overlap_percent": 10,
                        "panel_time_sec": 3600,
                        "gain": 80,
                        "selected_panels": "",
                        "num_tries": 1,
                        "retry_wait_s": 300,
                    },
                }
            ],
            "is_stacking": False,
        },
        current_item=current_item,
    )
    framed_html = template.render(
        schedule={
            "list": [
                {
                    "schedule_item_id": "x1",
                    "action": "start_framed_mosaic",
                    "params": {
                        "target_name": "M31",
                        "ra": 0.7,
                        "dec": 41.27,
                        "mosaic_scale": 1.5,
                        "mosaic_angle": 45.0,
                        "panel_time_sec": 3600,
                        "gain": 80,
                        "num_tries": 1,
                        "retry_wait_s": 300,
                    },
                }
            ],
            "is_stacking": False,
        },
        current_item=current_item,
    )

    assert _row_column_count(framed_html) == _row_column_count(mosaic_html)


@pytest.mark.parametrize(
    "model,wide,key",
    [
        ("Seestar S50", False, "S50"),
        ("Seestar S50 Pro", True, "S50 Pro"),
        ("S50Pro", True, "S50 Pro"),
        ("S50P", True, "S50 Pro"),
        ("Seestar S50P", True, "S50 Pro"),
        ("Seestar S30P", True, "S30 Pro"),
        ("Seestar S50N", False, "S50"),
        ("Seestar S30", True, "S30"),
        ("Seestar S30 Pro", True, "S30 Pro"),
        ("Seestar S30 Plus", True, "S30"),
        (None, False, None),
        (0, False, None),
    ],
)
def test_model_helpers_wide_camera_and_planner_key(model, wide, key):
    assert front_app.has_wide_camera(model) is wide
    assert front_app.planner_model_key(model) == key


def test_astro_mosaic_template_offers_s50_pro_and_preselects_model():
    from jinja2 import Environment, FileSystemLoader

    env = Environment(loader=FileSystemLoader("front/templates"))
    html = env.get_template("partials/astro_mosaic.html").render(
        planner_model="S50 Pro",
        utc_offset=0,
        config_lat=0,
        config_long=0,
        telescope={"device_num": 1},
        root="",
    )
    assert '<option value="S50 Pro" selected>' in html
    assert "Wide" not in html.split('id="seestar"')[1].split("</select>")[0]


def _render_astro_mosaic(**context):
    from jinja2 import Environment, FileSystemLoader

    env = Environment(loader=FileSystemLoader("front/templates"))
    base = {"planner_model": "S50", "telescope": {"device_num": 1}, "root": ""}
    base.update(context)
    return env.get_template("partials/astro_mosaic.html").render(**base)


def test_astro_mosaic_template_uses_configured_location():
    html = _render_astro_mosaic(utc_offset=1.0, config_lat=44.06, config_long=0.65)
    assert "location_lat: 44.06," in html
    assert "location_lng: 0.65," in html
    assert "32.92" not in html
    assert "-96.72" not in html
    # The chart's time zone is the site's, from the server, not a fixed UTC-5
    # nor the browser's.
    assert "timezoneOffset: 1.0," in html
    assert "getTimezoneOffset" not in html
    assert 'id="astro-mosaic-location-unset"' not in html


def test_astro_mosaic_template_handles_half_hour_site_offset():
    html = _render_astro_mosaic(utc_offset=5.5, config_lat=19.08, config_long=72.88)
    assert "timezoneOffset: 5.5," in html


def test_astro_mosaic_template_without_location_stays_valid_js():
    # Rendered without the planner's variables, the parameters must still be
    # JavaScript literals, not "location_lat: ," which breaks the card script.
    html = _render_astro_mosaic()
    assert "location_lat: 0," in html
    assert "location_lng: 0," in html
    assert "timezoneOffset: 0," in html


def test_astro_mosaic_template_warns_when_location_unset():
    html = _render_astro_mosaic(utc_offset=0, config_lat=0.0, config_long=0.0)
    assert 'id="astro-mosaic-location-unset"' in html


def test_astro_mosaic_template_versions_the_engine_script():
    # A cached engine of an older release must not be run with this template.
    html = _render_astro_mosaic(version="v3.3.1+17")
    assert 'src="/public/AstroMosaicEngine.js?v=v3.3.1%2B17"' in html


def test_site_utc_offset_follows_configured_timezone(monkeypatch):
    from datetime import datetime

    import pytz

    monkeypatch.setattr(Config, "init_timezone", "Europe/Paris", raising=False)
    winter = datetime(2026, 1, 15, 12, tzinfo=pytz.utc)
    summer = datetime(2026, 7, 15, 12, tzinfo=pytz.utc)
    assert front_app.site_utc_offset_hours(winter) == 1.0
    assert front_app.site_utc_offset_hours(summer) == 2.0

    monkeypatch.setattr(Config, "init_timezone", "Asia/Kolkata", raising=False)
    assert front_app.site_utc_offset_hours(winter) == 5.5


def test_site_utc_offset_falls_back_to_server_zone(monkeypatch):
    from datetime import datetime

    import pytz

    monkeypatch.setattr(
        front_app.tzlocal, "get_localzone", lambda: pytz.timezone("America/Chicago")
    )
    winter = datetime(2026, 1, 15, 12, tzinfo=pytz.utc)
    monkeypatch.setattr(Config, "init_timezone", "", raising=False)
    assert front_app.site_utc_offset_hours(winter) == -6.0
    # An unknown name is ignored rather than breaking the planning page.
    monkeypatch.setattr(Config, "init_timezone", "Mars/Olympus_Mons", raising=False)
    assert front_app.site_utc_offset_hours(winter) == -6.0


def _render_partial(name, **context):
    from jinja2 import Environment, FileSystemLoader

    env = Environment(loader=FileSystemLoader("front/templates"))
    base = {
        "planner_model": "S50",
        "utc_offset": 0,
        "config_lat": 44.06,
        "config_long": 0.65,
        "telescope": {"device_num": 1},
        "root": "",
    }
    return env.get_template(name).render(**{**base, **context})


def test_astro_mosaic_template_flat_horizon_without_profile():
    html = _render_partial("partials/astro_mosaic.html", horizon_limits=None)
    assert "horizonHard: [30]," in html


def test_astro_mosaic_template_uses_horizon_profile():
    limits = [12.5] * 71 + [40]
    html = _render_partial("partials/astro_mosaic.html", horizon_limits=limits)
    line = re.search(r"horizonHard: (\[[^\]]*\]),", html)
    assert json.loads(line.group(1)) == limits


def test_horizon_card_draws_the_profile():
    curve = [(azimuth, 10) for azimuth in range(361)]
    html = _render_partial(
        "partials/horizon.html",
        horizon_curve=curve,
        horizon_point_count=4,
        horizon_file="/etc/seestar/site.hrz",
        horizon_error=None,
    )
    assert 'id="horizon-profile"' in html
    assert "0,80 1,80" in html  # altitude 10 drawn 80 units down a 0-90 scale
    assert "/etc/seestar/site.hrz" in html
    assert "4 points" in html


def test_horizon_card_reports_an_unusable_file():
    html = _render_partial(
        "partials/horizon.html",
        horizon_curve=None,
        horizon_error="site.hrz, line 2: expected 'azimuth altitude'",
    )
    assert "Horizon file not used" in html
    assert "line 2" in html
    assert 'id="horizon-profile"' not in html


def test_horizon_card_explains_how_to_set_a_profile():
    html = _render_partial("partials/horizon.html", horizon_curve=None)
    assert "No horizon profile" in html
    assert "Horizon file" in html


def test_cached_context_is_not_rewritten_by_a_request_prefix(monkeypatch):
    # Isolate from a local config.toml [webui_settings] root_path.
    monkeypatch.setattr(Config, "root_path", "")
    # get_context() returns one cached dict per telescope for ~1s, shared by
    # every request. Each request must get its own prefix applied to a copy,
    # so a prefix from one proxy never leaks into another request's HTML.
    cached = {"root": "/1", "action": "/1/live", "imager_root": "/1/imaging"}
    snapshot = dict(cached)
    behind_a = DummyHTMXReq(headers={"X-Forwarded-Prefix": "/a"})
    behind_b = DummyHTMXReq(headers={"X-Forwarded-Prefix": "/b"})
    plain = DummyHTMXReq()

    out_a = front_app._template_context(behind_a, cached)
    out_b = front_app._template_context(behind_b, cached)
    out_plain = front_app._template_context(plain, cached)

    assert cached == snapshot
    assert out_a["root"] == "/a/1" and out_a["action"] == "/a/1/live"
    assert out_b["root"] == "/b/1" and out_b["action"] == "/b/1/live"
    assert out_plain["root"] == "/1" and out_plain["action"] == "/1/live"
    assert out_a["root_path"] == "/a" and out_plain["root_path"] == ""


def test_nav_links_to_the_horizon_page():
    html = _render_nav("planning")
    assert 'href="/1/horizon">Horizon (MMhorizon)</a>' in html


def test_horizon_card_links_to_the_panorama_page():
    for curve in (None, [(azimuth, 10) for azimuth in range(361)]):
        html = _render_partial("partials/horizon.html", horizon_curve=curve, root="/1")
        assert 'href="/1/horizon">Make the horizon from a 360&deg; panorama</a>' in html


@pytest.mark.parametrize(
    "horizon_file, horizon_error, shown",
    [
        ("", None, "flat horizon at 30"),
        ("/etc/seestar/site.hrz", None, "/etc/seestar/site.hrz &middot; 4 points"),
        ("/etc/seestar/gone.hrz", "No such file or directory", "Horizon file not used"),
    ],
)
def test_horizon_page_current_card_is_always_replaceable(
    horizon_file, horizon_error, shown
):
    # The page replaces #hz-current after a save, whatever it showed before,
    # a warning about an unusable file included.
    html = front_app.fetch_template("horizon.html").render(
        **_minimal_context("horizon"),
        config_lat=44.06,
        config_long=0.65,
        horizon_file=horizon_file,
        horizon_error=horizon_error,
        horizon_point_count=4,
    )
    current = re.search(
        r'<div class="card-body" id="hz-current">(.*?)</div>', html, re.S
    )
    assert current is not None
    assert shown in current.group(0)
    assert 'id="hz-page" data-lat="44.06" data-lon="0.65"' in html


def test_horizon_page_script_follows_the_root_path():
    html = front_app.fetch_template("horizon.html").render(
        **_minimal_context("horizon"),
        root_path="/ingress",
        config_lat=44.06,
        config_long=0.65,
        horizon_file="",
        horizon_error=None,
        horizon_point_count=0,
    )
    assert 'src="/ingress/public/horizon_panorama.js"' in html
    assert 'data-root-path="/ingress"' in html


def _record_device_actions(monkeypatch, reply=None):
    """Capture (action, telescope_id, parameters) for every device call."""
    calls = []

    def fake(action, telescope_id, parameters, is_schedule=False):
        calls.append((action, telescope_id, parameters))
        return reply

    monkeypatch.setattr(front_app, "do_action_device", fake)
    return calls


def test_get_remote_state_command_queries_the_selected_telescope(monkeypatch):
    calls = _record_device_actions(monkeypatch)
    req = type("Req", (), {"media": {"command": "get_remote_state"}})()

    front_app.do_command(req, None, 2)

    assert calls == [("method_sync", 2, {"method": "get_device_state"})]


@pytest.mark.parametrize(
    "helper, keys",
    [
        # An unfiltered get_device_state opens and closes the focuser, runs
        # shell queries and can wait ~30 s on the mount (seestar-spec, READ);
        # each helper asks only for the sections it reads.
        (front_app.get_firmware_ver_int, ["device"]),
        (front_app.get_device_model, ["device"]),
        (front_app.get_guestmode_state, ["device", "setting", "client"]),
        (
            front_app.get_device_state,
            [
                "device",
                "setting",
                "focuser",
                "storage",
                "balance_sensor",
                "compass_sensor",
                "mount",
                "pi_status",
                "client",
            ],
        ),
    ],
)
def test_device_state_helpers_request_only_the_sections_they_read(
    monkeypatch, helper, keys
):
    monkeypatch.setattr(front_app, "check_api_state", lambda _tid: True)
    calls = _record_device_actions(monkeypatch)

    helper(3)

    state_requests = [
        params
        for action, tid, params in calls
        if params.get("method") == "get_device_state"
    ]
    assert state_requests == [{"method": "get_device_state", "params": {"keys": keys}}]
    assert all(tid == 3 for _a, tid, _p in calls)


def test_balance_sensor_endpoint_requests_only_balance_sensor(monkeypatch):
    reply = {
        "ErrorNumber": 0,
        "Value": {"result": {"balance_sensor": {"data": {"x": 0.5, "y": -0.25}}}},
    }
    calls = _record_device_actions(monkeypatch, reply=reply)
    req = type("Req", (), {"get_header": lambda self, _n: "http://h/2/live"})()
    resp = type("Resp", (), {})()

    front_app.GetBalanceSensorResource.on_get(req, resp)

    assert calls == [
        (
            "method_sync",
            2,
            {"method": "get_device_state", "params": {"keys": ["balance_sensor"]}},
        )
    ]
    assert json.loads(resp.text) == {"x": 0.5, "y": -0.25}


def _sent_stack_dither(calls):
    for action, params in calls:
        if params.get("method") == "set_setting":
            dither = (params.get("params") or {}).get("stack_dither")
            if dither is not None:
                return dither
    return None


@pytest.mark.parametrize(
    "pix, interval, expected",
    [
        ("5", "3", {"pix": 5, "interval": 3}),
        # Blank or out-of-range values are left out: the firmware ignores a
        # missing member but answers 204 to an out-of-range one, which also
        # stops the rest of that set_setting (seestar-spec, READ).
        ("NaN", "", {}),
        ("201", "0", {}),
        ("-1", "101", {}),
    ],
)
def test_settings_post_sends_only_valid_dither_values(
    monkeypatch, pix, interval, expected
):
    _output, calls = _run_settings_post(
        monkeypatch,
        model="Seestar S50",
        form_extra={
            "stack_dither_pix": pix,
            "stack_dither_interval": interval,
            "stack_dither_enable": "true",
        },
    )

    assert _sent_stack_dither(calls) == {"enable": True, **expected}
