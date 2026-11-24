import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from mgts_nav.leveling import GRAVITY, build_leveler


def moving_case():
    times = np.arange(301) * 0.01
    axis = np.array([0.6, -0.5, 0.25])
    start = Rotation.from_euler("xyz", [0.18, 0.25, 1.1])
    attitudes = start * Rotation.from_rotvec(np.maximum(times - 0.8, 0)[:, None] * axis)
    acc = attitudes.inv().apply([0, 0, GRAVITY])
    gyro = np.tile([0.015, -0.005, 0.01], (len(times), 1))
    gyro[times > 0.8] += axis
    return times, attitudes, acc, gyro


def test_tilt_recovers_horizontal_plane_and_ignores_acceleration_spike():
    times, truth, acc, gyro = moving_case()
    acc[(times > 1.0) & (times < 1.35)] += [6.0, -4.5, 9.0]
    solver = build_leveler(times, gyro, acc, 0.5, 0.1)
    horizontal_in_base = truth.inv().apply([0.0, 2.0, 0.0])
    before = np.sqrt(np.mean(horizontal_in_base[:, 2]**2))
    base_to_level = Rotation.concatenate([solver.at(t) for t in times])
    corrected = base_to_level.apply(horizontal_in_base)
    after = np.sqrt(np.mean(corrected[:, 2]**2))
    print(f"Horizontal target height RMSE {before:.5f} -> {after:.5f} m")
    assert before > 0.2
    assert after < 0.006
    assert np.abs(base_to_level.as_euler("xyz")[:, 2]).max() < 1e-10
    for t in (0.0, 1.205, 2.995):
        pose = truth[round(t / 0.01)]
        vertical = solver.at(t).apply(pose.inv().apply([0.0, 0.0, 1.0]))
        assert np.allclose(vertical, [0.0, 0.0, 1.0], atol=0.009)


def test_gap_has_no_orientation_until_a_new_static_reference():
    times, _, acc, gyro = moving_case()
    keep = (times < 1.0) | (times > 1.5)
    times, acc, gyro = times[keep], acc[keep], gyro[keep]
    gyro[times >= 2.1] = [0.015, -0.005, 0.01]
    acc[times >= 2.1] = [0, 0, GRAVITY]
    solver = build_leveler(times, gyro, acc, 0.4, 0.1)
    for t in (-0.01, 1.2, 1.7, 3.01):
        with pytest.raises(ValueError):
            solver.at(t)
    assert np.allclose(solver.at(2.9).as_matrix(), np.eye(3))


def test_initial_vector_must_include_gravity_and_cover_interval():
    times = np.arange(20) * 0.01
    flat = np.zeros((len(times), 3))
    with pytest.raises(ValueError, match="stationary"):
        build_leveler(times, flat, flat, 0.1, 0.1)
    with pytest.raises(ValueError, match="coverage"):
        build_leveler(times, flat, flat, 0.5, 0.1)
