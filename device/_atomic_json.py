"""Atomic JSON write helper.

Used by callers that persist user-visible state which the next process
run will read unconditionally (calibration, azimuth limits, cumulative
az tracker). The naive ``open(path, "w") + json.dump`` path is non-atomic:
SIGKILL or power-loss between the open and the close leaves a truncated
or empty file on disk, blocking the next-session loader.

The pattern here is the standard tmp + fsync + ``os.replace`` dance:
write to a *unique* tmp file in the destination's directory, fsync the
file content + the descriptor, close, then ``os.replace`` onto the
destination. ``os.replace`` is atomic on POSIX (rename(2)) and
atomic-on-same-filesystem on Windows since Python 3.3.

The tmp name is unique per write (``tempfile.mkstemp`` in the target's
directory) rather than a fixed ``<path>.tmp`` sibling. A fixed sibling
lets two concurrent writers of the same destination clobber each other's
tmp — one truncates the other mid-write, or ``os.replace`` races on a
half-written file. A unique tmp per writer means each writer's rename is
independent and atomic; last writer wins with a fully-formed file.

This module has no project dependencies so it can be imported from any
``device.*`` module without import-cycle risk.
"""

from __future__ import annotations

import json
import os
import tempfile
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

    Creates ``path.parent`` if missing. Writes to a unique temp file in
    that directory first, fsyncs the data, then renames into place. On
    crash mid-write the destination retains its previous contents (or is
    absent if there was no previous version) — never truncated or partial.
    Concurrent writers of the same destination each use a distinct temp,
    so they never clobber each other's in-flight write.

    ``indent`` and ``cls`` mirror the equivalent ``json.dump`` kwargs.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    # Unique temp in the target directory (same filesystem → rename stays
    # atomic). mkstemp creates the file with an exclusive fd, so two
    # concurrent writers never share a temp path.
    fd, tmp_name = tempfile.mkstemp(
        prefix=p.name + ".", suffix=".tmp", dir=str(p.parent)
    )
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            if cls is not None:
                json.dump(obj, f, indent=indent, cls=cls)
            else:
                json.dump(obj, f, indent=indent)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, p)
    except BaseException:
        # Any failure (serialization error, fsync/replace error): drop our
        # unique temp so a crashed write doesn't leak files, and leave the
        # destination untouched.
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
