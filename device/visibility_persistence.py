"""Snapshot/load helpers for the visibility mapper.

Each telescope writes its current map to
``device_state/visibility_map_{tid}.json`` after every observation.
A subsequent :class:`VisibilityMapper` start reuses these as priors:
the mapper picks up where it left off rather than re-observing every
cell.

The format is intentionally a single-file JSON so it is easy to read
in tests and easy to delete if the user wants a clean slate.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

from device._atomic_json import write_atomic_json


SCHEMA_VERSION = 1


def visibility_map_path(state_dir: Path, telescope_id: int) -> Path:
    """Return the snapshot path for a telescope.

    The directory is created lazily; tests can pass a tmp_path here.
    """
    state_dir = Path(state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    return state_dir / f"visibility_map_{int(telescope_id)}.json"


def save_snapshot(
    path: Path,
    *,
    telescope_id: int,
    grid_kind: str,
    min_alt_deg: float,
    cells: list[dict],
    started_at: float,
    elapsed_s: float,
    n_observations: int,
) -> None:
    """Write the snapshot atomically.

    Atomic-via-rename so a crash mid-write doesn't leave a truncated
    JSON that the next start would refuse to load.
    """
    payload = {
        "schema_version": SCHEMA_VERSION,
        "telescope_id": int(telescope_id),
        "grid_kind": grid_kind,
        "min_alt_deg": float(min_alt_deg),
        "saved_at": time.time(),
        "started_at": float(started_at),
        "elapsed_s": float(elapsed_s),
        "n_observations": int(n_observations),
        "cells": cells,
    }
    write_atomic_json(path, payload, indent=None)


def load_snapshot(path: Path) -> Optional[dict]:
    """Load a saved snapshot, returning ``None`` if missing or unreadable.

    Schema mismatch (older or future version) returns ``None`` so the
    mapper falls back to fresh priors instead of mis-interpreting the
    payload.
    """
    p = Path(path)
    if not p.exists():
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    if data.get("schema_version") != SCHEMA_VERSION:
        return None
    if not isinstance(data.get("cells"), list):
        return None
    return data
