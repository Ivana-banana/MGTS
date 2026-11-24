import csv
import json
import subprocess
import sys

import numpy as np
from PIL import Image
from rosbags.typesys import Stores, get_typestore
import yaml

from mgts_nav import demo
from mgts_nav.bag import Scan
from mgts_nav.obstacles import detect


def run(*args):
    subprocess.run([sys.executable, "-m", *map(str, args)], check=True)


def test_paired_inspections(tmp_path, monkeypatch):
    # Surveyed rectangular room; use the same physical sensor pose for both records.
    pixels = np.full((81, 101), 254, dtype=np.uint8)
    pixels[[0, -1], :] = pixels[:, [0, -1]] = 0
    Image.fromarray(pixels).save(tmp_path / "room.pgm")
    map_path = tmp_path / "room.yaml"
    map_path.write_text(yaml.safe_dump(dict(image="room.pgm", resolution=.1, origin=[-5.05, -4.05, 0],
                                            occupied_thresh=.65, free_thresh=.196)))
    profile = json.loads(demo.PROFILE.read_text())
    pose = np.array([-.7, .35, .43])
    sensor = pose[:2] + demo.rotation2(pose[2]) @ profile["base_from_lidar_xyz_m"][:2]
    rays = demo.RAYS @ demo.rotation2(pose[2]).T
    edges = np.where(rays > 0, [5., 4.], [-5., -4.])
    walls = ((edges-sensor) / rays).min(axis=1)
    target = np.array([1.2, .8])
    projection = rays @ (target-sensor)
    discriminant = .30**2 - (np.linalg.norm(target-sensor)**2 - projection**2)
    hit = (discriminant >= 0) & (projection > 0)
    store = get_typestore(Stores.ROS2_HUMBLE)
    for added in (False, True):
        directory = tmp_path / ("changed" if added else "control")
        directory.mkdir()

        def returns(p, elapsed, cfg, rng, walls_arg, dropout):
            distance = walls.copy()
            if added and elapsed >= 1.0:
                distance[hit] = np.minimum(distance[hit], projection[hit] - np.sqrt(discriminant[hit]))
            distance += rng.normal(0., .008, len(distance))
            return np.column_stack((distance[:, None] * demo.RAYS, np.full(len(distance), .35))).astype("<f4")

        monkeypatch.setattr(demo, "lidar_points", returns)
        demo.record(directory, "inspection", profile, store, 38, duration=3.,
                    trajectory=lambda *args: pose)
        bag = directory / "inspection"
        localized = directory / "located"
        run("mgts_nav", "localize", bag, "--config", demo.PROFILE, "--map", map_path,
            "--initial", *pose, "--out", localized)
        trajectory = localized / "trajectory.csv"
        output = directory / "report.json"
        def report(name, traj, dest):
            run("mgts_nav.obstacles", bag, "--config", demo.PROFILE, "--map", map_path,
                "--trajectory", traj, "--out", dest)
            data = json.loads(dest.read_text())
            print(name, json.dumps(data))
            return data

        data = report(directory.name, trajectory, output)
        assert data["matched_pose_scans"] == data["scans"] == 30
        if not added:
            assert data["events"] == []
            continue
        event, = data["events"]
        assert event["observations"] == 20
        assert abs(event["confirmed_s"] - 251.035 - .2) < 1e-6
        assert np.linalg.norm(np.array([event["x_m"], event["y_m"]]) - target) < .35
        with trajectory.open(newline="") as stream:
            reader = csv.DictReader(stream)
            rows = list(reader)
        masked = directory / "masked.csv"
        with masked.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=reader.fieldnames)
            writer.writeheader()
            for row in rows:
                if float(row["t"]) >= 251.:
                    row["status"] = "prediction_only"
                writer.writerow(row)
        masked_report = report("unconfirmed", masked, directory / "masked.json")
        assert masked_report["matched_pose_scans"] == 10 and masked_report["events"] == []


def test_unknown_region_and_missing_poses():
    pixels = np.full((40, 40), 205, dtype=np.uint8)
    meta = dict(origin=[0., 0., 0.], resolution=.1, free_thresh=.196)
    xy = np.array([[1., 1.], [1.12, 1.], [1.24, 1.], [1.36, 1.]])
    stream = [Scan(i*.1, xy, np.zeros(2), None, None) for i in range(5)]
    poses = {s.t: np.zeros(3) for s in stream}
    assert detect(stream, poses, pixels, meta)["events"] == []
    pixels[:] = 254
    assert len(detect(stream, poses, pixels, meta)["events"]) == 1
    del poses[stream[2].t]
    del poses[stream[3].t]
    del poses[stream[4].t]
    assert detect(stream, poses, pixels, meta)["events"] == []
