import math

import pytest

from device.geometry import angular_separation_deg, unwrap_az_series, wrap_pm180


class TestWrapPm180:
    def test_identity_inside_range(self):
        assert wrap_pm180(0.0) == 0.0
        assert wrap_pm180(179.9) == pytest.approx(179.9)
        assert wrap_pm180(-179.9) == pytest.approx(-179.9)

    def test_boundary_convention_half_open(self):
        # [-180, +180): -180 inclusive, +180 wraps to -180.
        assert wrap_pm180(-180.0) == -180.0
        assert wrap_pm180(180.0) == -180.0
        assert wrap_pm180(540.0) == -180.0

    def test_multiple_turns(self):
        assert wrap_pm180(360.0 + 10.0) == pytest.approx(10.0)
        assert wrap_pm180(-720.0 - 10.0) == pytest.approx(-10.0)


class TestUnwrapAzSeries:
    def test_empty_and_single(self):
        assert unwrap_az_series([]) == []
        assert unwrap_az_series([12.0]) == [12.0]

    def test_short_path_across_wrap(self):
        # 170 -> -170 is +20 of true motion, not -340.
        out = unwrap_az_series([170.0, -170.0])
        assert out == pytest.approx([170.0, 190.0])

    def test_accumulates_multiple_wraps(self):
        samples = [0.0, 120.0, -120.0, 0.0]  # steady +120 steps
        assert unwrap_az_series(samples) == pytest.approx([0.0, 120.0, 240.0, 360.0])

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_rejects_non_finite(self, bad):
        with pytest.raises(ValueError, match="non-finite"):
            unwrap_az_series([0.0, bad, 10.0])


class TestAngularSeparation:
    def test_zero_for_same_direction(self):
        assert angular_separation_deg(45.0, 30.0, 45.0, 30.0) == pytest.approx(0.0)

    def test_pure_azimuth_at_horizon(self):
        assert angular_separation_deg(0.0, 0.0, 90.0, 0.0) == pytest.approx(90.0)

    def test_pole_to_horizon(self):
        assert angular_separation_deg(0.0, 90.0, 123.0, 0.0) == pytest.approx(90.0)

    def test_antipodal(self):
        assert angular_separation_deg(0.0, 0.0, 180.0, 0.0) == pytest.approx(180.0)

    def test_matches_sun_safety_reference(self):
        # Same spherical law of cosines as the sun_safety original.
        sep = angular_separation_deg(10.0, 20.0, 40.0, 25.0)
        a = (math.radians(10.0), math.radians(20.0))
        b = (math.radians(40.0), math.radians(25.0))
        cos_sep = math.sin(a[1]) * math.sin(b[1]) + math.cos(a[1]) * math.cos(
            b[1]
        ) * math.cos(a[0] - b[0])
        assert sep == pytest.approx(math.degrees(math.acos(cos_sep)))
