"""Every Seestar request written as a literal in the source matches seestar-spec.

The integration lane checks the requests that actually reach the simulator,
but only on paths the tests exercise, and never the CLI or MCP clients.
This scan covers every literal request in the code: a dict with a constant
"method" (and optional "params"), and helper calls that take the method
name first (method_sync, json_message, ...). Values computed at runtime are
replaced by UNKNOWN, so the method name and the literal structure (keys,
nesting, constant values) are checked.
"""

import ast
from pathlib import Path

import pytest

from tests.seestar_spec import UNKNOWN, validate_request

ROOT = Path(__file__).resolve().parents[1]
SCANNED = ["device", "front", "lib", "cli/ssalp_api_client", "mcp/seestar_mcp"]

# Helpers whose first positional argument is a firmware method name, and
# where their params come from: a "params" keyword, or a positional index.
# method_sync/method_async accept both: front/app.py passes params=...; the
# fork's AlpacaClient / MountClient callers (velocity controller, live
# tracker, calibration) pass method_sync(method, params) positionally.
METHOD_HELPERS = {
    "method_sync": {"keyword": "params", "positional": 1},
    "method_async": {"keyword": "params", "positional": 1},
    "json_message": {},
    "_auth_rpc_sync": {"positional": 1},
}
# The CLI client's method_sync(method, params) takes params positionally.
CLI_METHOD_SYNC = {"positional": 1}


def _literal(node):
    """Convert an AST expression to a Python value, UNKNOWN where dynamic."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, (ast.List, ast.Tuple)):
        if any(isinstance(e, ast.Starred) for e in node.elts):
            return UNKNOWN
        return [_literal(e) for e in node.elts]
    if isinstance(node, ast.Dict):
        if any(k is None for k in node.keys):  # {**spread}
            return UNKNOWN
        out = {}
        for k, v in zip(node.keys, node.values):
            if not (isinstance(k, ast.Constant) and isinstance(k.value, str)):
                return UNKNOWN
            out[k.value] = _literal(v)
        return out
    if (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, ast.USub)
        and isinstance(node.operand, ast.Constant)
    ):
        return -node.operand.value
    return UNKNOWN


def _requests_in(path):
    rel = path.relative_to(ROOT).as_posix()
    # utf-8 explicitly: the platform default (cp1252 on Windows) cannot decode
    # the non-ASCII comments in some scanned files.
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            value = _literal(node)
            if (
                isinstance(value, dict)
                and isinstance(value.get("method"), str)
                # Scheduler items and Alpaca actions use "action", not this.
                and set(value) <= {"method", "params", "id", "jsonrpc"}
            ):
                request = {"method": value["method"]}
                if "params" in value:
                    request["params"] = value["params"]
                yield f"{rel}:{node.lineno}", request
        elif isinstance(node, ast.Call):
            func = node.func
            name = (
                func.attr
                if isinstance(func, ast.Attribute)
                else getattr(func, "id", "")
            )
            how = METHOD_HELPERS.get(name)
            if name == "method_sync" and rel.startswith("cli/"):
                how = CLI_METHOD_SYNC
            if how is None or not node.args:
                continue
            first = node.args[0]
            if not (isinstance(first, ast.Constant) and isinstance(first.value, str)):
                continue
            request = {"method": first.value}
            if "keyword" in how:
                for kw in node.keywords:
                    if kw.arg == how["keyword"]:
                        request["params"] = _literal(kw.value)
            # Positional params only on a client method call (cli.method_sync(
            # method, params)); front/app.py's bare method_sync(method,
            # telescope_id, **kwargs) has the telescope id in that slot.
            positional_ok = name not in ("method_sync", "method_async") or (
                isinstance(func, ast.Attribute)
            )
            if (
                "params" not in request
                and positional_ok
                and "positional" in how
                and len(node.args) > how["positional"]
            ):
                request["params"] = _literal(node.args[how["positional"]])
            yield f"{rel}:{node.lineno}", request


def _all_requests():
    found = []
    for top in SCANNED:
        for path in sorted((ROOT / top).rglob("*.py")):
            if "tests" in path.relative_to(ROOT).parts:
                continue
            found.extend(_requests_in(path))
    return found


ALL_REQUESTS = _all_requests()

# Fork-only methods that are not in seestar-spec, exempted only at the source
# files known to send them (files, not line numbers, which drift). This is
# deliberately narrower than tests.seestar_spec.LEGACY_COMPAT_METHODS, which
# exempts a method everywhere, including the integration lane's run-time check.
KNOWN_OFF_SPEC = {
    # Sent by the unified calibration capture thunk
    # (front/app.py _calibration_capture_image_fn), used only when a plate
    # solver is configured. Unverified on firmware 9.75 (it may not exist on
    # any firmware); the spec has no equivalent single-frame capture method.
    "iscope_capture_image": frozenset({"front/app.py"}),
}


def _is_known_off_spec(where, request_):
    path = where.rsplit(":", 1)[0]
    return path in KNOWN_OFF_SPEC.get(request_["method"], ())


def test_scan_finds_requests():
    # Guards against the scanner silently matching nothing after a refactor.
    methods = {r["method"] for _, r in ALL_REQUESTS}
    assert {"scope_goto", "iscope_start_view", "set_setting"} <= methods
    assert len(ALL_REQUESTS) > 150


@pytest.mark.parametrize("method", sorted(KNOWN_OFF_SPEC))
def test_known_off_spec_exemption_still_needed(method):
    # Keeps KNOWN_OFF_SPEC from rotting: drop an entry once the method joins
    # seestar-spec, or once a listed file stops sending it.
    assert validate_request({"method": method}) == [
        f"method {method!r} is not in seestar-spec"
    ]
    senders = {w.rsplit(":", 1)[0] for w, r in ALL_REQUESTS if r["method"] == method}
    assert KNOWN_OFF_SPEC[method] <= senders, (
        f"{method} is no longer sent from "
        f"{sorted(KNOWN_OFF_SPEC[method] - senders)}; remove the exemption"
    )


@pytest.mark.parametrize(
    "where,request_", ALL_REQUESTS, ids=[w for w, _ in ALL_REQUESTS]
)
def test_literal_request_matches_spec(where, request_):
    if _is_known_off_spec(where, request_):
        pytest.skip(f"known off-spec fork method {request_['method']!r}")
    assert validate_request(request_) == []
