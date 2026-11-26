import numpy as np
from scipy.spatial.transform import Rotation

from tools.go2_show3d import header_clock, model_joints, nearest_index, sample_poses, wall_boxes


def test_motor_order_maps_unitree_legs_to_model_legs():
    # LowState: FR, FL, RR, RL; model: FL, FR, RL, RR. Encode leg*10 + joint.
    motor = np.array([10 * leg + j for leg in range(4) for j in range(3)])
    assert model_joints(motor).tolist() == [10, 11, 12, 0, 1, 2, 30, 31, 32, 20, 21, 22]


def test_header_clock_removes_reference_delivery_offset():
    ref_header = np.array([0.0, 1.0, 2.0])
    ref_delivery = ref_header + np.array([80.8, 80.9, 81.0])
    # 81.3 lies between deliveries 80.8 and 81.9; the offset is interpolated on the delivery axis.
    offset = 80.8 + 0.1 * (81.3 - 80.8) / (81.9 - 80.8)
    assert np.allclose(header_clock(np.array([81.3]), ref_delivery, ref_header), [81.3 - offset])


def test_sample_poses_interpolates_and_clamps():
    t = np.array([0.0, 1.0])
    pos = np.array([[0.0, 0.0, 0.3], [1.0, 2.0, 0.3]])
    quat = Rotation.from_euler('z', [0.0, np.pi / 2]).as_quat()
    xyz, rot, gap = sample_poses(np.array([0.5, 2.0]), t, pos, quat)
    assert np.allclose(xyz, [[0.5, 1.0, 0.3], [1.0, 2.0, 0.3]])
    assert np.isclose(rot[0].as_euler('xyz')[2], np.pi / 4)
    assert np.allclose(gap, [0.5, 0.0])


def test_nearest_index_picks_closest_sample():
    assert nearest_index(np.array([0.0, 1.0, 2.0]), np.array([0.4, 0.6, 5.0])).tolist() == [0, 1, 2]


def test_wall_boxes_keep_door_gap_and_merge_runs():
    v = 0.12
    cells = [(x, 0, z) for x in list(range(0, 5)) + list(range(9, 14)) for z in range(0, 8)]
    cells += [(20, 5, z) for z in range(0, 8)]  # isolated column without neighbours: dropped
    points = (np.array(cells) + 0.5) * v
    boxes = wall_boxes(points, floor_z=0.0, cut_z=0.7)
    assert len(boxes) == 2
    spans = sorted((round(b[0] / v), round(b[1] / v)) for b in boxes)
    assert spans == [(0, 5), (9, 14)]  # the 4-cell door between them stays open
    assert all(b[4] == 0.0 and b[5] == 0.7 for b in boxes)
