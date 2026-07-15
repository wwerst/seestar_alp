"""Mount client acquisition seam.

Every mover / streaming / session in this package talks to the mount
through a single structural protocol — ``method_sync(method, params=None)``
(see :class:`device.velocity_controller.MountClient`). Two concrete
transports satisfy it:

* :class:`~device.alpaca_client.AlpacaClient` (aliased here as
  :data:`HttpMount`) — the localhost Alpaca HTTP round-trip. Used by CLI
  tools, the unit-test suite, and any process where the live Seestar object
  is not registered.
* :class:`InProcessMount` — a thin wrapper that calls the in-process
  ``Seestar.send_message_param_sync`` directly, skipping the HTTP hop
  entirely. This is the whole point of the seam: the servo tick loops pay a
  localhost request/response per tick today (hundreds of ms under a saturated
  uplink), and the firmware round-trip floor is ~10-30 ms.

``get_mount_client`` picks the in-process transport when the target telescope
is registered *and* connected in this process, and falls back to HTTP
otherwise. The choice is behaviour-identical: the HTTP path is a pure
identity transform on the RPC payload (the server wraps the raw firmware
reply under ``"Value"`` and ``AlpacaClient.method_sync`` unwraps it), so
callers see byte-for-byte the same dict shapes on either transport. See
``tests/test_mount_client.py`` for the empirical both-paths equivalence check.
"""

from __future__ import annotations

from typing import Any

from device.alpaca_client import AlpacaClient

# Public alias for the HTTP transport. `get_mount_client` returns one of
# these when the in-process Seestar object is not available.
HttpMount = AlpacaClient

__all__ = ["InProcessMount", "HttpMount", "get_mount_client"]


class InProcessMount:
    """In-process :class:`MountClient` that calls the live Seestar directly.

    Behaviour-identical drop-in for ``AlpacaClient.method_sync``. The
    localhost Alpaca round-trip is a pure identity transform on the RPC
    payload:

    1. ``AlpacaClient.method_sync`` builds ``{"method": ...}`` (adding
       ``"params"`` only when params is not None) and PUTs it.
    2. The server hands that dict straight to
       ``Seestar.send_message_param_sync`` and wraps the raw firmware reply
       under ``"Value"`` via ``MethodResponse``.
    3. ``AlpacaClient.method_sync`` unwraps it with ``resp.get("Value")``.

    So ``AlpacaClient.method_sync`` already returns exactly what
    ``send_message_param_sync`` returns. This class short-circuits steps 1-3
    and calls ``send_message_param_sync`` itself, returning the identical
    dict (top-level ``result`` and ``Timestamp`` are preserved on both
    paths, as ``measure_altaz_timed`` relies on).

    Known, deliberate divergences from the HTTP path (payload-identical, but
    side effects differ): the Alpaca action layer's per-RPC INFO logging and
    its ``action`` event callbacks do NOT fire for in-process calls. Nothing
    in the servo/session stack consumes those for ``method_sync`` traffic;
    revisit if an event consumer ever starts depending on them.
    """

    def __init__(self, seestar: Any, telescope_id: int) -> None:
        self._seestar = seestar
        # MUST be the telescope-id int, NOT the Seestar object. The FF-loop
        # motor-stop-on-exit consults
        # ``sun_safety_jog_in_progress(getattr(cli, "device", None))``, which
        # compares this value against the sun monitor's jog telescope-id int.
        # ``AlpacaClient.device`` is likewise ``int(device)``. Exposing the
        # Seestar object here would silently break the
        # motor-stop-suppression-during-sun-jog logic.
        self.device = int(telescope_id)

    def method_sync(self, method: str, params: Any = None) -> Any:
        # Re-check connectivity on EVERY call, not just at acquisition: the
        # HTTP action handler refuses a not-connected device per request
        # (DevNotConnectedException -> no "Value" -> AlpacaClient.method_sync
        # returns None), so a mid-session disconnect must yield the same
        # immediate None here instead of pushing into a dead socket.
        if not getattr(self._seestar, "is_connected", False):
            return None
        # Mirror ``AlpacaClient.method_sync`` exactly: attach ``"params"``
        # only when it is not None. The presence/absence of the key is
        # load-bearing — ``Seestar.transform_message_for_verify`` injects
        # ``["verify"]`` differently when the key is absent, so building
        # ``{"params": None}`` unconditionally would diverge from the HTTP
        # path on older firmware.
        payload: dict[str, Any] = {"method": method}
        if params is not None:
            payload["params"] = params
        try:
            return self._seestar.send_message_param_sync(payload)
        except Exception:
            # The Alpaca action handler swallows any send-side exception into
            # a DevDriverException MethodResponse with no ``"Value"``, so
            # ``AlpacaClient.method_sync`` returns None over HTTP (Alpaca
            # errors are HTTP-200, so ``raise_for_status`` doesn't fire).
            # Mirror the swallow-to-None so callers see the identical
            # contract on both transports — every hot caller
            # (measure_altaz_timed retry, wait_for_mount_idle except,
            # make_scope_altaz_reader except) already treats None/garbage
            # defensively.
            return None


def _lookup_connected_device(telescope_id: int) -> Any:
    """Return the live, connected in-process Seestar for ``telescope_id``,
    or None when it is not registered/connected in this process.

    Gating on ``is_connected`` keeps the not-connected error path
    byte-identical to today: the HTTP action handler itself refuses a
    not-connected device (``DevNotConnectedException`` → no ``"Value"`` →
    ``method_sync`` returns None), so we only take the in-process shortcut
    when the object is genuinely live and let HTTP handle everything else.
    """
    try:
        from device import telescope
    except Exception:
        return None
    try:
        dev = telescope.get_seestar_device(int(telescope_id))
    except (KeyError, TypeError, ValueError):
        # get_seestar_device indexes a plain dict; an unregistered device
        # (CLI tools, the unit-test suite, another process) raises KeyError.
        return None
    if dev is not None and getattr(dev, "is_connected", False):
        return dev
    return None


def get_mount_client(
    telescope_id: int,
    *,
    host: str = "127.0.0.1",
    port: int | None = None,
    force_http: bool | None = None,
) -> Any:
    """Return a MountClient for ``telescope_id``.

    Prefers :class:`InProcessMount` when the telescope is registered and
    connected in this process; otherwise returns an :data:`HttpMount`
    (AlpacaClient) that reaches the running app over localhost. Both expose
    the same ``method_sync`` surface and an int ``.device``, so no
    downstream mover/streaming code changes with the transport.

    ``host``/``port`` are only consulted for the HTTP fallback; ``port``
    defaults to ``Config.port`` when None. Pass ``force_http=True`` to skip
    the in-process shortcut (e.g. a dry-run or a remote scope); when left
    None the ``[mount_client] force_http`` config escape hatch is honoured.
    """
    from device.config import Config

    if force_http is None:
        # Read the toml directly so no new Config attribute is required.
        force_http = bool(Config.get_toml("mount_client", "force_http", False))

    if not force_http:
        dev = _lookup_connected_device(telescope_id)
        if dev is not None:
            return InProcessMount(dev, int(telescope_id))

    resolved_port = int(port) if port is not None else int(Config.port)
    return HttpMount(host, resolved_port, int(telescope_id))
