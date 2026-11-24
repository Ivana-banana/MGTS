"""Paired office experiment with missing wall echoes and a person who pauses."""
import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from rosbags.rosbag2 import Writer
from rosbags.typesys import Stores, get_typestore
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from .bag import cloud_xyz, load_config, reader, scans, stamp
from .demo import PROFILE, START_NS, WALLS, generate, pose_at
from .grid import Grid, read_map
from .map_eval import segment_distances
from .metrics import Pose, evaluate, read_csv
from .slam import rotation, track, world

RESOLUTION = 0.08
TOLERANCE = 0.12
PANEL = np.array([[-1.4, -2.0], [1.4, -2.0]])
STOP = np.array([-1.6, -1.32])
RADIUS = 0.22


def person_center(t):
    # Walk east, stand still for 12 seconds, then walk west out of the stop area.
    delta = -0.3 * (max(4.0 - t, 0.0) + max(t - 16.0, 0.0))
    return STOP + [delta, 0.0]


def modify_hits(origin, hits, panel_loss, center, uniform):
    rays = hits - origin
    distance = np.linalg.norm(rays, axis=1)
    directions = rays / distance[:, None]
    person = np.zeros(len(hits), dtype=bool)
    if center is not None:
        offset = center - origin
        proj = directions @ offset
        disc = RADIUS**2 - (offset @ offset - proj**2)
        near = proj - np.sqrt(np.maximum(disc, 0))
        person = (disc >= 0) & (near > 0) & (near < distance)
        hits = hits.copy()
        hits[person] = origin + near[person, None] * directions[person]
    panel = segment_distances(hits, [PANEL]) <= 0.045
    lost = panel_loss & panel & ~person & (uniform < 0.995)
    return hits, ~lost


def variant(source, target, run, cfg, seed, panel_loss, person):
    store = get_typestore(Stores.ROS2_HUMBLE)
    sensor_rot = Rotation.from_quat(cfg["base_from_lidar_xyzw"])
    offset = np.array(cfg["base_from_lidar_xyz_m"])
    rng = np.random.default_rng(seed)
    with reader(source) as bag, Writer(target, version=8) as writer:
        outputs = {c.id: writer.add_connection(c.topic, c.msgtype, typestore=store)
                   for c in bag.connections}
        for c, received, raw in bag.messages():
            if c.topic == cfg["lidar"]:
                m = bag.deserialize(raw, c.msgtype)
                elapsed = stamp(m) - START_NS[run] / 1e9
                pose = pose_at(run, elapsed)
                base = sensor_rot.apply(cloud_xyz(m)) + offset
                hits = world(base[:, :2], pose)
                center = person_center(elapsed) if person else None
                hits, keep = modify_hits(world(offset[:2], pose), hits, panel_loss, center,
                                         rng.random(len(hits)))
                base[:, :2] = (hits - pose[:2]) @ rotation(pose[2])
                xyz = sensor_rot.inv().apply(base[keep] - offset).astype("<f4")
                m.width = len(xyz)
                m.row_step = 12 * m.width
                m.data = xyz.view(np.uint8).ravel()
                raw = store.serialize_cdr(m, c.msgtype)
            writer.write(outputs[c.id], received, raw)


def false_stop_cells(occupied):
    near_stop = np.linalg.norm(occupied - STOP, axis=1) <= RADIUS + RESOLUTION / np.sqrt(2)
    off_wall = segment_distances(occupied, WALLS) > TOLERANCE
    return int((near_stop & off_wall).sum())


def score(mapping, inspection, cfg, initial, truth, directory):
    directory.mkdir()
    grid = Grid(RESOLUTION)
    at_stop = directory/"during-stop"
    count_during = None
    for row in track(scans(mapping, cfg), np.zeros(3), grid=grid):
        if count_during is None and row.t >= START_NS["mapping"] / 1e9 + 15.9:
            at_stop.mkdir()
            grid.save(at_stop)
            count_during = false_stop_cells(read_map(at_stop/"map.yaml")[2])
    grid.save(directory)
    occupied = read_map(directory/"map.yaml")[2]
    # One sample per 4 cm, centred within each segment of the panel.
    fractions = (np.arange(70) + 0.5) / 70
    panel_samples = PANEL[0] + fractions[:, None] * (PANEL[1] - PANEL[0])
    missed = cKDTree(occupied).query(panel_samples)[0] > TOLERANCE
    estimates = list(track(scans(inspection, cfg), initial, map_points=occupied))
    error = evaluate([Pose(e.t, *e.pose) for e in estimates], truth)
    return dict(panel_missed_samples=int(missed.sum()), panel_samples=len(missed),
                false_stop_cells=false_stop_cells(occupied), false_stop_cells_during=count_during,
                inspection_matched=sum(e.status == "matched" for e in estimates),
                inspection_scans=len(estimates), inspection_ate_m=error["ate_xy_rmse_m"])


def experiment(root):
    cfg = load_config(PROFILE)
    results = {}
    for seed in (41, 73):
        directory = root/str(seed)
        directory.mkdir()
        control = directory/"control"
        generate(control, seed=seed)
        truth = read_csv(control/"inspection_truth.csv")
        initial = np.array(json.loads((control/"initial_pose.json").read_text(encoding="utf-8")))
        inspection_glass = directory/"inspection-glass"
        variant(control/"inspection", inspection_glass, "inspection", cfg, seed+1, True, False)
        for name, glass, person in (("control", False, False), ("glass", True, False),
                                    ("person", False, True), ("both", True, True)):
            mapping = control/"mapping"
            if glass or person:
                mapping = directory/("mapping-"+name)
                variant(control/"mapping", mapping, "mapping", cfg, seed, glass, person)
            inspection = inspection_glass if glass else control/"inspection"
            results[f"{seed}/{name}"] = score(mapping, inspection, cfg, initial, truth,
                                               directory/("map-"+name))
    (root/"summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    output = parser.parse_args().out
    if output.exists():
        parser.error("Choose a new directory")
    output.parent.mkdir(exist_ok=True, parents=True)
    with TemporaryDirectory(dir=output.parent) as tmp:
        stage = Path(tmp)/"result"
        stage.mkdir()
        results = experiment(stage)
        stage.rename(output)
    print(json.dumps(results, indent=2))
