import json
import re
import threading

import pytest
import front.app as front_app
from device.config import Config


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
    before the user clicks 'Start calibration'.

    Without this, the live-camera <img id="cal-vid"> sits on a Loading
    frame until the calibration session itself starts (which is when
    `ensure_scenery_mode` is called inside `_connect_mount`)."""
    import time as _time

    calls = []
    call_event = threading.Event()

    def fake_do_action_device(action, dev_num, parameters, is_schedule=False):
        calls.append((action, dev_num, parameters))
        call_event.set()
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
    assert call_event.wait(timeout=2.0), (
        f"scenery iscope_start_view kick never landed within 2s; calls={calls}"
    )
    # Allow a tick for any duplicate kicks that might land after the
    # first — we want to assert exactly one.
    _time.sleep(0.05)

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


def test_calibration_start_legacy_target_oas_payload_still_works(
    monkeypatch, tmp_path
):
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


def test_calibration_start_rejects_both_targets_and_target_oas(
    monkeypatch, tmp_path
):
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
    assert "non-empty" in json.loads(resp.text)["error"]


def test_calibration_start_rejects_unknown_kind(monkeypatch, tmp_path):
    _patch_unified_start_env(monkeypatch, tmp_path)
    body = {
        "altitude_m": 2.0,
        "dry_run": True,
        "targets": [{"kind": "ad-hoc", "label": "?"}],
    }
    req = _StartReq(media=body)
    resp = _StartResp()
    front_app.CalibrationStartResource.on_post(req, resp, telescope_id=1)
    assert resp.status == "400 Bad Request"
    assert "unknown kind" in json.loads(resp.text)["error"]


def test_calibration_start_rejects_celestial_missing_radec(
    monkeypatch, tmp_path
):
    _patch_unified_start_env(monkeypatch, tmp_path)
    body = {
        "altitude_m": 2.0,
        "dry_run": True,
        "targets": [
            {"kind": "celestial", "name": "Vega"},  # missing ra/dec
        ],
    }
    req = _StartReq(media=body)
    resp = _StartResp()
    front_app.CalibrationStartResource.on_post(req, resp, telescope_id=1)
    assert resp.status == "400 Bad Request"
    assert "celestial" in json.loads(resp.text)["error"]


def test_calibration_start_rejects_unknown_faa_oas(monkeypatch, tmp_path):
    _patch_unified_start_env(monkeypatch, tmp_path)
    body = {
        "altitude_m": 2.0,
        "dry_run": True,
        "targets": [{"kind": "faa", "oas": "ZZ-999999"}],
    }
    req = _StartReq(media=body)
    resp = _StartResp()
    front_app.CalibrationStartResource.on_post(req, resp, telescope_id=1)
    assert resp.status == "400 Bad Request"
    assert "ZZ-999999" in json.loads(resp.text)["error"]


def test_calibration_start_platesolve_without_solver_returns_503(
    monkeypatch, tmp_path
):
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
    monkeypatch.setattr(
        "device.alpaca_client.AlpacaClient", lambda *a, **k: None
    )
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
    front_app.CalibrationCelestialTargetsResource.on_get(
        req, resp, telescope_id=1
    )
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
    monkeypatch.setattr(
        "device.alpaca_client.AlpacaClient", lambda *a, **k: None
    )
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
    front_app.CalibrationCelestialTargetsResource.on_get(
        req, resp, telescope_id=1
    )
    assert resp.status == "400 Bad Request"
