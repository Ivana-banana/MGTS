"""Run full solves with and without camera observations on synthetic and KITTI bags."""
import argparse
import io
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image
from rosbags.typesys import Stores, get_typestore
from scipy.spatial.transform import Rotation

from mgts_nav.demo import generate
from mgts_nav.image_pairs import features, tracks
from mgts_nav.metrics import evaluate, read_csv
from mgts_nav.solve import solve
from tools.heap1_sample import write_bag
from tools.kitti_repeat import PREFIX, calibration, load_scan, timestamps


def pairs(path, records):
    with path.open("x", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, allow_nan=False) + "\n")


def synthetic(data):
    generate(data, camera_experiment=True)
    records = [json.loads(s) for s in (data/"inspection-pairs.jsonl").read_text().splitlines()]
    result = {"observed": data/"inspection-pairs.jsonl"}
    for mode in ("parallax", "mismatched", "clock", "no_depth"):
        modified = []
        for record in records:
            row = record.copy()
            if mode == "parallax":
                row["camera_motion_bound_m"] = .3
            elif mode == "no_depth":
                row.pop("min_scene_range_m")
            elif mode == "clock":
                row.update(t0=row["t0"]+12, t1=row["t1"]+12)
            else:
                rows = row["tracks"]
                row["tracks"] = [a[:3]+b[3:] for a, b in zip(rows, rows[1:]+rows[:1])]
            modified.append(row)
        result[mode] = data/f"{mode}.jsonl"
        pairs(result[mode], modified)
    return Path("configs/example.json"), read_csv(data/"inspection_truth.csv"), result


def calib_values(archive, file):
    lines = archive.read("2011_09_30/"+file).decode().splitlines()
    return dict(line.split(":", 1) for line in lines if ":" in line)


def field(data, archives, scores):
    data.mkdir()
    store = get_typestore(Stores.ROS2_HUMBLE)
    first = json.loads((scores/"scores.json").read_text())["inspection_first"]
    profile = dict(lidar="/kitti/cloud", lidar_frame="imu_projected", base_frame="imu",
                   imu=None, odom=None, base_from_lidar_xyzw=[0, 0, 0, 1],
                   height_band_m=[-.1, .1], stationary_s=0, max_sync_gap_s=.35)
    with ZipFile(archives/"drive.zip") as drive, ZipFile(archives/"calib.zip") as calib:
        transform = calibration(calib)
        origin = -transform[1] @ transform[0]
        profile["base_from_lidar_xyz_m"] = [*origin[:2], 0.]
        times = timestamps(drive, "velodyne_points")
        for run, window in (("mapping", range(100)), ("inspection", range(first, first+40))):
            points = []
            for i in window:
                scan = load_scan(drive, i, times[i], *transform)
                points.append(np.column_stack((scan.xy-scan.origin, np.zeros(len(scan.xy)))).astype("<f4"))
            write_bag(data/run, store, profile["lidar"], profile["lidar_frame"], times[window],
                      points, list(map(len, points)))
        camera = calib_values(calib, "calib_cam_to_cam.txt")
        k = np.fromstring(camera["P_rect_00"], sep=" ").reshape(3, 4)[:, :3].tolist()
        rect = np.fromstring(camera["R_rect_00"], sep=" ").reshape(3, 3)
        extrinsic = calib_values(calib, "calib_velo_to_cam.txt")
        mount = rect @ np.fromstring(extrinsic["R"], sep=" ").reshape(3, 3) @ transform[0]
        quaternion = Rotation.from_matrix(mount.T).as_quat().tolist()
        camera_times = timestamps(drive, "image_00")
        records = []
        previous = None
        for i in range(first-1, first+41):
            frame = drive.read(PREFIX+f"image_00/data/{i:010d}.png")
            current = features(np.array(Image.open(io.BytesIO(frame)), dtype=float))
            if previous is not None:
                records.append(dict(t0=camera_times[i-1], t1=camera_times[i], K=k,
                                    base_from_camera_xyzw=quaternion, tracks=tracks(previous, current)))
            previous = current
        # The monocular tracks lack a certified lower distance bound.
        pairs(data/"real.jsonl", records)
    config = data/"profile.json"
    config.write_text(json.dumps(profile), encoding="utf-8")
    return config, read_csv(scores/"inspection-reference.csv"), {"observed": data/"real.jsonl"}


def trial(output, name, dll):
    data = output/f"{name}-data"
    if name == "office":
        config, truth, variants = synthetic(data)
    else:
        config, truth, variants = field(data, Path("out/kitti161"), Path("out/kitti161-run"))
    initial = list(truth[0][1:])
    result = {}
    for mode, source in {"off": None, **variants}.items():
        dest = output/f"{name}-{mode}"
        solve(data/"mapping", data/"inspection", config, dest, initial=initial,
              inspection_visual_pairs=source, grid_dll=dll, resolution=.12 if name == "kitti" else .08)
        route = read_csv(dest/"inspection/trajectory.csv")
        np.testing.assert_allclose([p.t for p in route], [p.t for p in truth], atol=1e-6, rtol=0)
        summary = json.loads((dest/"inspection/summary.json").read_text())
        result[mode] = dict(metrics=evaluate(route, truth), status=summary["status"],
                            visual=summary.get("visual_yaw"))
        (output/f"{name}-scores.json").write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--dll", type=Path)
    parser.add_argument("--scene", choices=["office", "kitti", "both"], default="both")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for name in (("office", "kitti") if args.scene == "both" else (args.scene,)):
        trial(args.output, name, args.dll)
