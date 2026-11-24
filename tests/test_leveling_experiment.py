import numpy as np
from scipy.spatial.transform import Rotation

from mgts_nav.bag import load_config, scans
from mgts_nav.demo import PROFILE, START_NS, WALLS, generate, pose_at
from mgts_nav.leveling import GRAVITY
from mgts_nav.map_eval import segment_distances
from tools.evaluate_leveling import IMU_MOUNT, LIDAR_MOUNT, imu_ideal, make_bag, render


def test_body_rates_gravity_and_lidar_extrinsics():
    t = 2.25
    amp = np.deg2rad(15.)
    u = t-1.
    roll = amp*np.sin(2*np.pi*.8*u)
    pitch = .7*amp*np.sin(2*np.pi*.6*u+.45)
    rd = 2*np.pi*.8*amp*np.cos(2*np.pi*.8*u)
    pd = 2*np.pi*.6*.7*amp*np.cos(2*np.pi*.6*u+.45)
    yd = 2*np.pi/18.
    expected = [rd-yd*np.sin(pitch),
                pd*np.cos(roll)+yd*np.sin(roll)*np.cos(pitch),
                -pd*np.sin(roll)+yd*np.cos(roll)*np.cos(pitch)]
    omega, force = imu_ideal("mapping", t, 15)
    np.testing.assert_allclose(IMU_MOUNT.apply(omega), expected, atol=1e-6)
    pose = pose_at("mapping", t)
    # Independent composition around fixed axes, from sensor to world.
    axes = Rotation.from_rotvec([0, 0, pose[2]]) * Rotation.from_rotvec([0, pitch, 0]) * Rotation.from_rotvec([roll, 0, 0])
    acceleration = [-2.4*yd**2*np.sin(yd*u), 1.4*yd**2*np.cos(yd*u), GRAVITY]
    np.testing.assert_allclose(axes.apply(IMU_MOUNT.apply(force)), acceleration, atol=1e-4)
    cfg = load_config(PROFILE)
    xyz = render("mapping", t, 15, cfg, np.random.default_rng(42))
    observed = axes.apply(LIDAR_MOUNT.apply(xyz)+cfg["base_from_lidar_xyz_m"]) + [*pose[:2], 0.]
    wall = segment_distances(observed[:, :2], WALLS)
    surface = np.minimum(wall, np.minimum(abs(observed[:, 2]+.45), abs(observed[:, 2]-2.4)))
    assert np.sqrt(np.mean(surface**2)) < .02


def test_recorded_gap_disables_geometry_until_new_rest(tmp_path):
    source = tmp_path/"source"
    generate(source, seed=41)
    cfg = dict(load_config(PROFILE), gravity_level=True, base_from_lidar_xyzw=LIDAR_MOUNT.as_quat(),
               base_from_imu_xyzw=IMU_MOUNT.as_quat())
    for name, end in (("full", None), ("cut", 8.5)):
        path = tmp_path/name
        make_bag(source/"inspection", path, "inspection", 15, cfg, 42, gap=True, end=end)
        rows = list(scans(path, cfg))
        t = np.array([s.t for s in rows])-START_NS["inspection"]/1e9
        valid = np.array([len(s.xy) > 0 for s in rows])
        assert valid[t < 2.9].all()
        assert not valid[(t >= 3.) & (t < 8.5)].any()
        assert all(row.empty_reason == "leveling_unavailable" for row, ok in zip(rows, valid) if not ok)
        assert bool(valid[-1]) == (end is None)
        assert all(len(s.xy) for s in scans(path, dict(cfg, gravity_level=False)))
