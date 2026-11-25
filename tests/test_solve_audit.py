import csv
from io import BytesIO
import json
import subprocess
import sys

import numpy as np
from PIL import Image
import pytest

from mgts_nav.demo import PROFILE, generate
from mgts_nav.solve import solve


@pytest.mark.parametrize("dense", [True, False])
def test_solve_product_audit(tmp_path, dense):
    data = tmp_path / "data"
    generate(data)
    mapping, inspection = data / "mapping", data / "inspection"
    solution = tmp_path / "solution"
    cfg = json.loads(PROFILE.read_bytes())
    if not dense:
        cfg["odom"] = None
    profile = tmp_path / "inspection.json"
    profile.write_text(json.dumps(cfg))
    solve(mapping, inspection, PROFILE, solution, inspection_profile=profile,
          initial=json.loads((data / "initial_pose.json").read_bytes()))

    def check(error=None, swapped=False):
        bags = [inspection, mapping] if swapped else [mapping, inspection]
        result = subprocess.run([sys.executable, "-m", "mgts_nav.solve_audit", *bags, solution],
                                capture_output=True, text=True)
        if error is None:
            assert result.returncode == 0, result.stderr
            return json.loads(result.stdout)
        assert result.returncode == 2 and error in result.stderr, result.stderr

    report = check()
    assert report["mapping_scans"] == 200 and report["inspection_scans"] == 100
    assert report["exported_route"] == ("continuous.csv" if dense else "trajectory.csv")
    if not dense:
        return
    check("source scan 1 timestamp", swapped=True)
    path = solution / "inspection/trajectory.csv"
    original = path.read_bytes()
    with path.open(newline="") as stream:
        rows = list(csv.reader(stream))
    def write(table):
        with path.open("w", newline="") as stream:
            csv.writer(stream).writerows(table)
    write(rows[:-1])
    check("scan count")
    rows[3][1] = str(float(rows[3][1]) + 1.)
    write(rows)
    check("anchors")
    path.write_bytes(original)
    pgm = solution / "map/map.pgm"
    with Image.open(BytesIO(pgm.read_bytes())) as img:
        pixels = np.array(img)
    pixels[0, 0] = 0 if pixels[0, 0] != 0 else 254
    Image.fromarray(pixels).save(pgm)
    check("occupancy differs from PGM")
