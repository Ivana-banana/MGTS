import numpy as np
import pytest

from mgts_nav.deskew import deskew_points


def test_static_wall_under_combined_translation_and_rotation():
    # The robot crosses +/-pi during this lidar sweep. Work in a rotated world
    # frame so the anchor yaw and translation are non-zero.
    times = np.arange(201) * 0.01
    odom = np.column_stack((times, 1 + 0.9 * times, -2 + 0.4 * times, 2.2 + times))
    scan_t = 1.0
    offsets = np.linspace(-0.15, 0.12, 151)
    t = scan_t + offsets
    x_wall = -3.0
    y_wall = np.linspace(-3, 1.0, len(t))
    dx, dy = x_wall - (1 + 0.9 * t), y_wall - (-2 + 0.4 * t)
    c, s = np.cos(2.2 + t), np.sin(2.2 + t)
    height = np.linspace(0.2, 0.8, len(t))
    raw = np.column_stack((c * dx + s * dy, -s * dx + c * dy, height))
    fixed = deskew_points(raw, offsets, scan_t, odom, 0.025)
    ca, sa = np.cos(3.2), np.sin(3.2)

    def wall_rmse(points):
        x_world = 1.9 + ca * points[:, 0] - sa * points[:, 1]
        return np.sqrt(np.mean((x_world - x_wall)**2))

    before, after = wall_rmse(raw), wall_rmse(fixed)
    print(f"Static wall RMSE: {before:.6f} -> {after:.3e} m")
    assert before > 0.08
    assert after < 1e-10
    assert np.allclose(-1.6 + sa * fixed[:, 0] + ca * fixed[:, 1], y_wall)
    np.testing.assert_array_equal(fixed[:, 2], height)

    outside = odom.copy()
    outside[(times < 0.80) | (times > 1.20), 1:] += 10_000
    np.testing.assert_allclose(deskew_points(raw, offsets, scan_t, outside, 0.025), fixed)


@pytest.mark.parametrize("offset, reference, reason", [
    (-0.2, 0.1, "cover"), (0.25, 0.8, "cover"),
    (-0.8, 1.1, "cover"), (0.6, 0.3, "gap"), (0, 0.5, "gap"),
])
def test_no_extrapolation_or_interpolation_across_gaps(offset, reference, reason):
    odom = np.zeros((4, 4))
    odom[:, 0] = [0.0, 0.3, 0.9, 1.0]
    with pytest.raises(ValueError, match=reason):
        deskew_points([[1, 0, 0]], [offset], reference, odom, 0.31)


def test_duplicate_time_and_wrong_offset_count_are_rejected():
    cloud = np.zeros((2, 3))
    with pytest.raises(ValueError, match="strictly increasing"):
        deskew_points(cloud, [0, 0], 0, np.zeros((2, 4)), 0.2)
    with pytest.raises(ValueError, match="offset"):
        deskew_points(cloud, [0], 0, np.zeros((1, 4)), 0.2)
