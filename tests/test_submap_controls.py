"""Reproduce the twelve predeclared multi-map decisions in issue 95."""
from dataclasses import replace
from itertools import combinations
from itertools import islice
import json
from pathlib import Path

import numpy as np
import pytest

from mgts_nav.bag import load_config, scans
from mgts_nav.delayed_start import estimate_start
from mgts_nav.demo import PROFILE, generate
from mgts_nav.grid import Grid, read_map
from mgts_nav.metrics import read_csv, wrap
from mgts_nav.slam import track, world


@pytest.fixture(scope="module")
def surveys(tmp_path_factory):
    config = load_config(PROFILE)
    result = {}
    for layout, seed in (("office", 17), ("office", 29), ("corridor", 41), ("corridor", 73)):
        root = tmp_path_factory.mktemp(f"{layout}-{seed}")
        data = root / "data"
        generate(data, layout=layout, seed=seed)
        survey = list(scans(data / "mapping", config))
        trajectory = list(track(survey, np.zeros(3), grid=Grid(.08)))
        sections = []
        for first in range(0, len(survey), 40):
            grid = Grid(.08)
            for scan, row in zip(survey[first:first+40], trajectory[first:first+40], strict=True):
                assert scan.t == row.t
                if row.status in ("origin", "matched"):
                    grid.observe(world(scan.origin, row.pose), world(scan.xy, row.pose))
            directory = root / str(first)
            directory.mkdir()
            try:
                grid.save(directory)
            except ValueError as exc:
                sections.append((first, None, str(exc)))
            else:
                pixels, meta, _ = read_map(directory / "map.yaml")
                sections.append((first, (pixels, meta), None))
        inspection = list(scans(data / "inspection", dict(config, odom=None)))
        result[layout, seed] = sections, inspection, data / "inspection_truth.csv"
    return result


@pytest.mark.parametrize("layout,seed,foreign", [
    ("office", 17, False), ("office", 29, False), ("corridor", 41, False),
    ("corridor", 73, False), ("office", 17, True), ("office", 29, True),
])
@pytest.mark.parametrize("radius", [None, 3.])
@pytest.mark.parametrize("local_method", ["pairwise", "keyframes"])
def test_blind_sections(surveys, layout, seed, foreign, radius, local_method):
    sections, scans_in, truth_path = surveys[layout, seed]
    if foreign:
        sections = surveys["corridor", 41][0]
    index = 0 if layout == "office" else 30
    window = scans_in[index:index+51]
    assert len(window) == 51 and all(scan.odom is None for scan in window)
    if radius is not None:
        window = [replace(s, xy=s.xy[np.linalg.norm(s.xy - s.origin, axis=1) <= radius]) for s in window]
    accepted, reasons = {}, {}
    for first, raster, error in sections:
        if raster is None:
            reasons[first] = error
            continue
        try:
            pose = estimate_start(window, *raster, {}, local_method=local_method)
        except ValueError as exc:
            reasons[first] = str(exc)
        else:
            assert np.all(np.isfinite(pose))
            accepted[first] = pose.tolist()
    outcome, xy, yaw = "refused", None, None
    if len(accepted) == 1:
        outcome = "false"
        if not foreign:
            ref = read_csv(truth_path)[index]
            assert abs(ref.t - window[0].t) < 1e-8
            pose = np.asarray(next(iter(accepted.values())))
            xy = float(np.linalg.norm(pose[:2] - ref[1:3]))
            yaw = float(np.rad2deg(abs(wrap(pose[2] - ref.yaw))))
            if xy <= .25 and yaw <= 10:
                outcome = "correct"
    pairs = list(combinations(np.asarray(list(accepted.values())), 2))
    spread_m = max((np.linalg.norm(a[:2] - b[:2]) for a, b in pairs), default=None)
    spread_deg = max((np.rad2deg(abs(wrap(a[2] - b[2]))) for a, b in pairs), default=None)
    print(json.dumps(dict(layout=layout, seed=seed, foreign=foreign, radius=radius, method=local_method,
                          spread_m=spread_m, spread_deg=spread_deg,
                          accepted=accepted, reasons=reasons, outcome=outcome, xy_m=xy, yaw_deg=yaw)))
    assert set(accepted) | set(reasons) == {0, 40, 80, 120, 160}
    assert all(raster is not None for _, raster, _ in sections)
    assert outcome != "false"
    if foreign:
        assert outcome == "refused"


def test_heap1_keyframes_on_saved_sections():
    data = Path("out/heap1-long-80")
    maps = Path("out/issue94")
    if not (data / "inspection").is_dir() or not (maps / "report.json").is_file():
        pytest.skip("Requires the field recording and section maps from issue 94")
    cfg = load_config(data / "profile.json")
    window = list(islice(scans(data / "inspection", cfg), 51))
    assert len(window) == 51 and all(s.odom is None for s in window)
    starts, reasons = {}, {}
    for first in range(0, 300, 40):
        pixels, meta, _ = read_map(maps / f"section-{first}" / "map.yaml")
        try:
            pose = estimate_start(window, pixels, meta, {}, local_method="keyframes")
        except ValueError as exc:
            reasons[first] = str(exc)
        else:
            starts[first] = pose.tolist()
    report = dict(starts=starts, reasons=reasons)
    if len(starts) == 1:
        ref = read_csv(data / "inspection-reference.csv")[0]
        assert abs(ref.t - window[0].t) < 1e-6
        pose = np.asarray(next(iter(starts.values())))
        report["xy_m"] = np.linalg.norm(pose[:2] - ref[1:3])
        report["yaw_deg"] = np.rad2deg(abs(wrap(pose[2] - ref.yaw)))
    print(json.dumps(report))
    assert len(reasons) + len(starts) == 8
