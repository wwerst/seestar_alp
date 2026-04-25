"""Atomic JSON write helper.

Used by callers that persist user-visible state which the next process
run will read unconditionally (calibration, azimuth limits, cumulative
az tracker). The naive ``open(path, "w") + json.dump`` path is non-atomic:
SIGKILL or power-loss between the open and the close leaves a truncated
or empty file on disk, blocking the next-session loader.

The pattern here is the standard tmp + fsync + ``os.replace`` dance:
write to a sibling tmp file, fsync the file content + the descriptor,
close, then ``os.replace`` onto the destination. ``os.replace`` is
atomic on POSIX (rename(2)) and atomic-on-same-filesystem on Windows
since Python 3.3.

This module has no project dependencies so it can be imported from any
``device.*`` module without import-cycle risk.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def write_atomic_json(
    path: str | os.PathLike[str],
    obj: Any,
    *,
    indent: int | None = 2,
    cls: type[json.JSONEncoder] | None = None,
) -> None:
    """Atomically serialize ``obj`` as JSON to ``path``.

    Creates ``path.parent`` if missing. Writes to ``<path>.tmp`` first,
    fsyncs the data, then renames into place. On crash mid-write the
    destination retains its previous contents (or is absent if there
    was no previous version) — never truncated or partial.

    ``indent`` and ``cls`` mirror the equivalent ``json.dump`` kwargs.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    # Ensure no stale tmp lingers from a prior crashed run.
    try:
        tmp.unlink()
    except FileNotFoundError:
        pass
    with tmp.open("w", encoding="utf-8") as f:
        if cls is not None:
            json.dump(obj, f, indent=indent, cls=cls)
        else:
            json.dump(obj, f, indent=indent)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, p)
