"""Generate two deterministic ROS 2 bags in the same simulated office.

Run from the checkout with: python -m mgts_nav.demo OUT
"""
import argparse
import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from rosbags.rosbag2 import StoragePlugin, Writer
from rosbags.typesys import Stores, get_typestore
from scipy.spatial.transform import Rotation


PROFILE = Path(__file__).resolve().parents[1] / "configs" / "example.json"
DURATION = {"mapping": 20.0, "inspection": 10.0}
START_NS = {"mapping": 100_000_000_000, "inspection": 250_000_000_000}
TOPIC_TYPES = {
    "lidar": "sensor_msgs/msg/PointCloud2",
    "imu": "sensor_msgs/msg/Imu",
    "odom": "nav_msgs/msg/Odometry",
}
STORAGE = {"sqlite3": StoragePlugin.SQLITE3, "mcap": StoragePlugin.MCAP}


def pose_at(run, elapsed, layout="office", duration=None):
    """Analytic truth in the mapping start frame, with stationary ends."""
    duration = DURATION[run] if duration is None else duration
    fraction = np.clip((elapsed - 1.0) / (duration - 2.0), 0.0, 1.0)
    if layout == "corridor":
        a = 2 * np.pi * fraction
        if run == "mapping":
            return np.array([4.3*np.sin(a), .45*(1-np.cos(a)), a])
        return np.array([2.1 - 4.7*fraction, .3 + .28*np.sin(a), 2.65 + .22*np.sin(a)])
    if run == "mapping":
        a = 2 * np.pi * fraction
        return np.array([2.4 * np.sin(a), 1.4 * (1 - np.cos(a)), a])
    a = np.pi * fraction
    return np.array([-1.0 + 3.0 * fraction, 0.55 + 0.75 * np.sin(a),
                     0.48 * np.cos(a)])


def rectangle(x0, y0, x1, y1):
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]


def scene_segments():
    shapes = [
        rectangle(-4.2, -2.0, 5.2, 4.8),
        rectangle(2.8, 2.6, 3.7, 3.8),
        rectangle(-3.3, -1.3, -2.8, -0.5),
        rectangle(0.7, 3.3, 1.3, 4.3),
        [[4.5, 0.1], [3.8, -0.6], [3.8, -1.4]],
    ]
    return np.array([(a, b) for shape in shapes for a, b in zip(shape, shape[1:])])


WALLS = scene_segments()
ANGLES = np.linspace(-np.pi, np.pi, 240, endpoint=False)
RAYS = np.column_stack((np.cos(ANGLES), np.sin(ANGLES)))


def rotation2(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s], [s, c]])


def cross(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def lidar_points(pose, elapsed, profile, rng, walls=WALLS, dropout=0.10):
    """Occluded first returns from static walls and a moving cylindrical target."""
    rotation = rotation2(pose[2])
    offset = np.asarray(profile["base_from_lidar_xyz_m"])
    origin = pose[:2] + rotation @ offset[:2]
    directions = RAYS @ rotation.T
    edge = walls[:, 1] - walls[:, 0]
    relative = walls[:, 0] - origin
    denom = cross(directions[:, None], edge)
    valid = np.abs(denom) > 1e-10
    distance = np.divide(cross(relative, edge), denom, where=valid,
                         out=np.zeros_like(denom))
    along = np.divide(cross(relative, directions[:, None]), denom, where=valid,
                      out=np.zeros_like(denom))
    hits = np.where(valid & (distance > 0) & (along >= 0) & (along <= 1),
                    distance, np.inf).min(axis=1)

    center = np.array([1.3 + 0.8 * np.sin(elapsed * 0.7), -0.9]) - origin
    along_ray = directions @ center
    disc = 0.2**2 - (center @ center - along_ray**2)
    near = along_ray - np.sqrt(np.maximum(0, disc))
    hits = np.minimum(hits, np.where((disc >= 0) & (near > 0), near, np.inf))
    keep = np.isfinite(hits) & (rng.random(len(hits)) > dropout)
    distance = hits[keep] + rng.normal(0.0, 0.012, keep.sum())
    base_relative = np.column_stack((distance[:, None] * RAYS[keep],
                                     np.full(keep.sum(), 0.65 - offset[2])))
    sensor = Rotation.from_quat(profile["base_from_lidar_xyzw"]).inv()
    return sensor.apply(base_relative).astype("<f4")


def record(directory, run, profile, store, seed, layout="office", storage="sqlite3", *, duration=None,
           camera_stress=False):
    duration = DURATION[run] if duration is None else duration
    rng = np.random.default_rng(seed)
    types = store.types
    vector = types["geometry_msgs/msg/Vector3"]
    quaternion = types["geometry_msgs/msg/Quaternion"]
    field = types["sensor_msgs/msg/PointField"]
    fields = [field(name, 4 * idx, 7, 1) for idx, name in enumerate("xyz")]
    zero_cov = np.zeros(36)
    start = pose_at(run, 0.0, layout, duration)
    walls = WALLS
    if layout == "corridor":
        # A long narrow hall with asymmetric recesses; a side door swings between runs.
        border = [(-6,-1.3),(-1.6,-1.3),(-1.6,-2.5),(.1,-2.5),(.1,-1.3),
                  (5.8,-1.3),(5.8,2.1),(3.4,2.1),(3.4,3.8),(1.2,3.8),
                  (1.2,2.1),(-6,2.1),(-6,-1.3)]
        door_tip = (-.5,-2.35) if run == "mapping" else (-.25,-1.4)
        walls = np.array([*zip(border, border[1:]), [(-1.6,-1.3),door_tip]])
    inverse_imu = Rotation.from_quat(profile["base_from_imu_xyzw"]).inv()
    # Distinct header clocks/sample phases share the same epoch.
    schedule = []
    end_ns = int(duration * 1e9)
    for key, period, first in [("imu", 20_000_000, 0), ("odom", 50_000_000, 0),
                               ("lidar", 100_000_000, 35_000_000)]:
        schedule.extend((t, key) for t in range(first, end_ns + 1, period))
    with Writer(directory / run, version=8, storage_plugin=STORAGE[storage]) as bag, (
        directory / f"{run}_truth.csv"
    ).open("w", newline="", encoding="utf-8") as truth:
        rows = csv.writer(truth)
        rows.writerow(["t", "x", "y", "yaw"])
        connections = {key: bag.add_connection(profile[key], typ, typestore=store)
                       for key, typ in TOPIC_TYPES.items()}
        for dt_ns, key in sorted(schedule):
            elapsed = dt_ns / 1e9
            ns = START_NS[run] + dt_ns
            t = types["builtin_interfaces/msg/Time"](*divmod(ns, 1_000_000_000))
            header = types["std_msgs/msg/Header"](t, profile[f"{key}_frame"])
            pose = pose_at(run, elapsed, layout, duration)
            if key == "lidar":
                xyz = lidar_points(pose, elapsed, profile, rng, walls, 0.35 if layout == "corridor" else 0.10)
                if camera_stress and 4 <= elapsed < 6:
                    xyz = xyz[:8]
                message = types[TOPIC_TYPES[key]](
                    header, 1, len(xyz), fields, False, 12, 12 * len(xyz),
                    xyz.view(np.uint8).ravel(), True)
                rows.writerow([f"{ns / 1e9:.9f}", *pose])
            elif key == "imu":
                rate = (pose_at(run, elapsed + 0.001, layout, duration)[2]
                        - pose_at(run, elapsed - 0.001, layout, duration)[2]) / 0.002
                gyro = inverse_imu.apply([0.0, 0.0, rate + 0.015])
                gyro += rng.normal(0.0, 0.002, 3)
                if camera_stress and 4.8 <= elapsed < 6:
                    continue
                gravity = inverse_imu.apply([0.0, 0.0, 9.81])
                message = types[TOPIC_TYPES[key]](
                    header, quaternion(0, 0, 0, 1), np.r_[-1.0, np.zeros(8)],
                    vector(*gyro), np.eye(3).ravel() * 0.002**2,
                    vector(*gravity), np.zeros(9))
            else:
                # Each run has its own odom frame, with scale and heading drift.
                active = np.clip(elapsed - 1.0, 0, duration - 2.0)
                local = rotation2(-start[2]) @ (pose[:2] - start[:2])
                local = 1.045 * (rotation2(0.004 * active) @ local)
                yaw = pose[2] - start[2] + 0.007 * active
                orient = quaternion(0, 0, np.sin(yaw / 2), np.cos(yaw / 2))
                position = types["geometry_msgs/msg/Point"](*local, 0.0)
                est = types["geometry_msgs/msg/Pose"](position, orient)
                twist = types["geometry_msgs/msg/Twist"](vector(0, 0, 0), vector(0, 0, 0))
                message = types[TOPIC_TYPES[key]](
                    header, profile["base_frame"],
                    types["geometry_msgs/msg/PoseWithCovariance"](est, zero_cov),
                    types["geometry_msgs/msg/TwistWithCovariance"](twist, zero_cov))
            bag.write(connections[key], ns + 2_000_000,
                      store.serialize_cdr(message, TOPIC_TYPES[key]))


def generate(output, layout="office", seed=17, storage="sqlite3", camera_experiment=False):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"Choose a fresh demo directory: {output}")
    if camera_experiment and layout != "office":
        raise ValueError("Camera landmarks and bounds are defined for the office layout")
    profile = json.loads(PROFILE.read_text(encoding="utf-8"))
    store = get_typestore(Stores.ROS2_HUMBLE)
    with TemporaryDirectory(dir=output.parent) as temp:
        staging = Path(temp) / "dataset"
        staging.mkdir()
        for i, run in enumerate(DURATION):
            record(staging, run, profile, store, seed+i, layout, storage,
                   camera_stress=camera_experiment and run == "inspection")
        scenario = dict(layout=layout, seed=seed, storage=storage)
        if camera_experiment:
            from .camera_demo import write_camera_pairs
            write_camera_pairs(staging / 'inspection-pairs.jsonl', seed + 2, pose_at, WALLS, START_NS['inspection'])
            scenario['camera_experiment'] = True
        (staging / "initial_pose.json").write_text(
            json.dumps(pose_at("inspection", 0, layout).tolist()) + "\n", encoding="utf-8")
        (staging / "scenario.json").write_text(json.dumps(scenario), encoding="utf-8")
        staging.rename(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path, help="New directory for a synthetic data set")
    parser.add_argument("--layout", choices=["office", "corridor"], default="office")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--storage", choices=STORAGE, default="sqlite3", help="ROS 2 storage format")
    parser.add_argument("--camera-experiment", action="store_true",
                        help="Generate camera feature pairs and a weak-lidar inspection in the office")
    args = parser.parse_args()
    generate(args.out, args.layout, args.seed, args.storage, args.camera_experiment)
