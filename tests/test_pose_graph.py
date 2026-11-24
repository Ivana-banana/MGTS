import numpy as np
import pytest

from mgts_nav.pose_graph import adjust


def test_closed_loop_distributes_drift_without_moving_anchor():
    t = np.linspace(0, 1, 120)
    truth = np.column_stack([3*np.sin(2*np.pi*t), 3*(1-np.cos(2*np.pi*t)), 2*np.pi*t])
    drifted = truth + t[:, None] * [0.22, -0.07, 0.03]
    corrected = adjust(drifted, [0, 0, 0])
    np.testing.assert_allclose(corrected[0], drifted[0])
    assert np.linalg.norm(corrected[-1, :2]) < 0.02
    assert np.linalg.norm(corrected[:, :2] - truth[:, :2]) < np.linalg.norm(drifted[:, :2] - truth[:, :2])


def test_stationary_graph_is_rejected():
    with pytest.raises(ValueError, match='No observable motion'):
        adjust(np.zeros((8, 3)), [0, 0, 0])
