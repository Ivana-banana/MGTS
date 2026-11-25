import numpy as np
import pytest

from mgts_nav.clock_windows import diagnose, interpolate


def test_gap_is_not_interpolated_but_endpoints_are_valid():
    t = np.array([0., .1, 1., 1.1])
    out = interpolate(t, t, np.array([.1, .6, 1.]))
    np.testing.assert_allclose(out, [.1, np.nan, 1.], equal_nan=True)


def test_constant_motion_and_missing_records_are_distinct():
    t = np.arange(201)/20
    mask = t < 5
    output = diagnose((t, t*0), (t[mask], t[mask]*0))
    assert [row["status"] for row in output["windows"]] == ["low_excitation", "insufficient_coverage"]
    assert not output["jumps_s"]


def test_non_monotonic_clock_requires_split():
    t = np.array([0., 1., .9])
    with pytest.raises(ValueError, match="reset"):
        diagnose((t,t), (t,t))
