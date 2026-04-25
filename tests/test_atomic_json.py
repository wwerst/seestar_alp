"""Unit tests for device._atomic_json.write_atomic_json.

Used by every JSON persistence site that the next process run reads
unconditionally (rotation_calibration.write_calibration,
plant_limits.AzimuthLimits.save, plant_limits.CumulativeAzTracker.save).
"""

from __future__ import annotations

import json

import pytest

from device._atomic_json import write_atomic_json


def test_atomic_write_creates_file_with_payload(tmp_path):
    p = tmp_path / "out.json"
    write_atomic_json(p, {"a": 1, "b": "two"})
    assert json.loads(p.read_text()) == {"a": 1, "b": "two"}


def test_atomic_write_creates_parent_dirs(tmp_path):
    p = tmp_path / "nested" / "dir" / "out.json"
    write_atomic_json(p, [1, 2, 3])
    assert p.exists()
    assert json.loads(p.read_text()) == [1, 2, 3]


def test_atomic_write_overwrites_existing_file(tmp_path):
    p = tmp_path / "out.json"
    write_atomic_json(p, {"v": 1})
    write_atomic_json(p, {"v": 2})
    assert json.loads(p.read_text()) == {"v": 2}


def test_atomic_write_preserves_destination_on_serialization_error(
    tmp_path, monkeypatch
):
    """Crash mid-json.dump → destination keeps prior contents."""
    p = tmp_path / "out.json"
    write_atomic_json(p, {"original": True})

    import device._atomic_json as aj

    def boom(*_a, **_kw):
        raise RuntimeError("bad encoder")

    monkeypatch.setattr(aj.json, "dump", boom)
    with pytest.raises(RuntimeError, match="bad encoder"):
        write_atomic_json(p, {"replacement": True})

    monkeypatch.undo()
    # Destination still has the original payload — the partial write was
    # to a tmp file that never got renamed.
    assert json.loads(p.read_text()) == {"original": True}


def test_atomic_write_is_absent_when_no_prior_file_and_crash(tmp_path, monkeypatch):
    """If there is no prior file and the write crashes, the destination
    is absent (NOT a partial file)."""
    p = tmp_path / "out.json"
    assert not p.exists()

    import device._atomic_json as aj

    def boom(*_a, **_kw):
        raise RuntimeError("crash")

    monkeypatch.setattr(aj.json, "dump", boom)
    with pytest.raises(RuntimeError, match="crash"):
        write_atomic_json(p, {"x": 1})

    monkeypatch.undo()
    assert not p.exists()


def test_atomic_write_with_custom_encoder(tmp_path):
    """`cls=` kwarg is plumbed through to json.dump."""
    import collections

    class _DequeEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, collections.deque):
                return list(obj)
            return super().default(obj)

    p = tmp_path / "out.json"
    write_atomic_json(p, {"q": collections.deque([1, 2, 3])}, cls=_DequeEncoder)
    assert json.loads(p.read_text()) == {"q": [1, 2, 3]}


def test_atomic_write_cleans_stale_tmp_from_prior_crash(tmp_path):
    """A leftover ``.tmp`` from a prior crashed run is removed before
    the new write — otherwise the new ``open(..., 'w')`` would silently
    overwrite a tmp from an unrelated context."""
    p = tmp_path / "out.json"
    stale = p.with_suffix(p.suffix + ".tmp")
    stale.write_text('{"stale": true}')
    assert stale.exists()
    write_atomic_json(p, {"fresh": True})
    assert json.loads(p.read_text()) == {"fresh": True}
    # Tmp was cleaned up after rename.
    assert not stale.exists()
