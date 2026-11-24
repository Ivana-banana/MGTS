import sys

import numpy as np
from PIL import Image
import pytest
import yaml

from mgts_nav import __main__ as cli
from mgts_nav.demo import PROFILE, generate
from mgts_nav.grid import read_map


@pytest.fixture(scope="module")
def demo(tmp_path_factory):
    directory = tmp_path_factory.mktemp("map-geometry")
    generate(directory / "data")
    pixels = np.full((6, 8), 254, dtype=np.uint8)
    pixels[[0, -1]] = 0
    Image.fromarray(pixels).save(directory / "room.pgm")
    meta = dict(image="room.pgm", resolution=.25, origin=[-2., 4., 0.],
                free_thresh=.196, occupied_thresh=.65)
    return directory, meta


@pytest.mark.parametrize("changes,field", [
    ({"origin": []}, "origin"), ({"origin": [0., 0.]}, "origin"),
    ({"origin": [0., float("nan"), 0.]}, "origin"), ({"origin": [0., 0., 1.]}, "origin"),
    ({"resolution": 0.}, "resolution"), ({"resolution": float("inf")}, "resolution"),
    ({"resolution": True}, "resolution"), ({"resolution": 1e308}, "resolution"),
    ({"occupied_thresh": -1}, "occupied_thresh"), ({"free_thresh": 2}, "free_thresh"),
    ({"occupied_thresh": .196}, "free_thresh"), ({"free_thresh": "0.1"}, "free_thresh"),
    ({"negate": 1}, "negate"), ({"image": None}, "image"),
    ("origin: [", "YAML"), ("[]", "YAML mapping"), ({}, "resolution"),
])
def test_localize_rejects_bad_map(demo, tmp_path, monkeypatch, capsys, changes, field):
    directory, meta = demo
    path = tmp_path / "input.yaml"
    if isinstance(changes, dict):
        updated = dict(meta, image=str(directory/"room.pgm"))
        updated.update(changes)
        if not changes:
            del updated["resolution"]
        path.write_text(yaml.safe_dump(updated))
    else:
        path.write_text(changes)
    out = tmp_path / "result"
    monkeypatch.setattr(sys, "argv", ["mgts_nav", "localize", str(directory/"data/inspection"),
                                      "--map", str(path), "--initial", "0", "0", "0",
                                      "--config", str(PROFILE), "--out", str(out)])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    error = capsys.readouterr().err
    assert exc.value.code == 2 and field in error and str(path) in error
    assert "Traceback" not in error and not out.exists()


def test_loaded_points_keep_metric_centers(demo):
    directory, meta = demo
    path = directory / "valid.yaml"
    path.write_text(yaml.safe_dump(meta))
    _, _, points = read_map(path)
    expected = [(x*.25-1.875, y) for y in (5.375, 4.125) for x in range(8)]
    np.testing.assert_array_equal(points, expected)
