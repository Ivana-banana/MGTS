import json
import os
import shutil
import subprocess
import sys

import yaml

from mgts_nav.solve_manifest import REQUIRED


def split_bag(path):
    path.mkdir(parents=True)
    names = ["part0.db3", "part1.db3"]
    for name in names:
        (path / name).write_bytes(name.encode())
    (path / "metadata.yaml").write_text(yaml.safe_dump({"rosbag2_bagfile_information":
                                                       {"relative_file_paths": names}}))


def test_manifest_cli_relocation_and_mutation(tmp_path):
    root = tmp_path / "source"
    mapping = root / "survey"
    split_bag(mapping)
    (root / "inspection.bag").write_bytes(b"single ROS1 input")
    solution = root / "solution"
    for name in REQUIRED:
        path = solution / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")
    (solution / "map/map.yaml").write_text("image: floor.pgm\nresolution: 0.1\n")
    (solution / "map/floor.pgm").write_bytes(b"PGM content")
    (solution / "profiles/inspection-pairs.jsonl").write_text('{"t0":0,"t1":0.1}\n')
    split_bag(solution / "rviz")
    # Only the test data is relocated. The project still executes from the MGTS checkout.
    def invoke(action, bundle, *args):
        return subprocess.run([sys.executable, "-m", "mgts_nav.solve_manifest", action,
                               bundle / "survey", bundle / "inspection.bag", bundle / "solution",
                               bundle / "manifest.json", *args], env=dict(os.environ, PATH="", GIT_DIR="unavailable"),
                              text=True, capture_output=True)

    created = invoke("create", root, "--solve-revision", "example-release")
    assert created.returncode == 0, created.stderr
    raw = (root / "manifest.json").read_text()
    assert str(tmp_path) not in raw
    report = json.loads(raw)
    assert report["solve_revision"] == "example-release"
    assert {"mapping/part1.db3", "inspection/recording", "solution/rviz/part1.db3",
            "solution/profiles/inspection.json"} <= report["sha256"].keys()
    assert invoke("create", root).returncode != 0
    moved = tmp_path / "moved"
    shutil.copytree(root, moved)
    verified = invoke("verify", moved)
    assert verified.returncode == 0, verified.stderr
    for name, label in (("survey/part1.db3", "mapping/part1.db3"),
                        ("solution/profiles/inspection-pairs.jsonl", "solution/profiles/inspection-pairs.jsonl"),
                        ("solution/inspection/trajectory.csv", "solution/inspection/trajectory.csv")):
        path = moved / name
        old = path.read_bytes()
        path.write_bytes(b"changed")
        failure = invoke("verify", moved)
        assert failure.returncode != 0 and label in failure.stderr
        path.write_bytes(old)
