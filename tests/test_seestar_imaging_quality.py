"""Tests for the pure stream-quality helpers on SeestarImaging.

Regression: ``_clamp_jpeg_quality(float("inf"))`` used to raise
``OverflowError`` (from ``round(inf)``), which neither the stream-quality
POST route nor ``default_stream_quality`` catches. Non-finite input must
raise ``ValueError`` so the route answers 400 and the config path falls
back to the default.
"""

from __future__ import annotations

import pytest

from device.config import Config
from device.seestar_imaging import SeestarImaging


@pytest.mark.parametrize("bad", [float("inf"), float("-inf"), float("nan")])
def test_clamp_jpeg_quality_rejects_non_finite_with_value_error(bad):
    with pytest.raises(ValueError):
        SeestarImaging._clamp_jpeg_quality(bad)


def test_clamp_jpeg_quality_rejects_infinity_string_with_value_error():
    # json.loads accepts the bare token Infinity, and float("Infinity")
    # parses, so a string form must be rejected the same way.
    with pytest.raises(ValueError):
        SeestarImaging._clamp_jpeg_quality("Infinity")


def test_clamp_jpeg_quality_finite_values_unchanged():
    assert SeestarImaging._clamp_jpeg_quality(95) == 95
    assert SeestarImaging._clamp_jpeg_quality(72.6) == 73
    assert SeestarImaging._clamp_jpeg_quality(5000) == SeestarImaging.JPEG_QUALITY_MAX
    assert SeestarImaging._clamp_jpeg_quality(1) == SeestarImaging.JPEG_QUALITY_MIN


def test_default_stream_quality_falls_back_on_infinite_jpeg_quality(monkeypatch):
    vals = {("stream_quality", "jpeg_quality"): float("inf")}
    monkeypatch.setattr(Config, "get_toml", lambda s, i, d: vals.get((s, i), d))
    out = SeestarImaging.default_stream_quality()
    assert out["jpeg_quality"] == SeestarImaging.JPEG_QUALITY_DEFAULT
