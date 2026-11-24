"""Repeatable visibility experiment for global starts in the synthetic corridor."""
from collections import Counter
from dataclasses import replace
import json

import numpy as np
import pytest

from mgts_nav.bag import load_config, scans
from mgts_nav.demo import PROFILE, generate
from mgts_nav.grid import Grid, read_map
from mgts_nav.initialization import estimate_initial
from mgts_nav.metrics import read_csv, wrap
from mgts_nav.multiscan import estimate_multiscan
from mgts_nav.slam import track, world


def restrict(scan, reference, view):
    offset = scan.xy - scan.origin
    valid = np.ones(len(offset), dtype=bool)
    if view == "walls":
        # Simulator-side mask only. Keep the two straight side-wall surfaces.
        xy = world(scan.xy, np.array(reference[1:]))
        valid = ((np.minimum(abs(xy[:, 1]+1.3), abs(xy[:, 1]-2.1)) <= .04)
                 & (xy[:, 0] > -5.9) & (xy[:, 0] < 5.7))
    if view in ("short", "sector"):
        valid &= np.linalg.norm(offset, axis=1) <= 3
    if view == "sector":
        valid &= abs(np.arctan2(offset[:, 1], offset[:, 0])) <= np.deg2rad(40)
    return replace(scan, xy=scan.xy[valid])


def classify(start, reference):
    xy = float(np.linalg.norm(start[:2] - reference[1:3]))
    yaw = abs(wrap(start[2] - reference.yaw))
    category = "correct" if xy <= .25 and yaw <= np.deg2rad(10) else "false_accept"
    return dict(result=category, error_m=round(xy, 3), error_deg=round(np.rad2deg(yaw), 2),
                returned_pose=np.round(start, 3).tolist())


@pytest.mark.parametrize("seed", [41, 73])
def test_ambiguity(tmp_path, seed):
    demo = tmp_path / "demo"
    generate(demo, layout="corridor", seed=seed)
    config = load_config(PROFILE)
    grid = Grid(.08)
    list(track(scans(demo / "mapping", config), np.zeros(3), grid=grid))
    grid.save(tmp_path)
    pixels, meta, _ = read_map(tmp_path / "map.yaml")
    measured = list(scans(demo / "inspection", config))
    truth = read_csv(demo / "inspection_truth.csv")
    assert [s.t for s in measured] == pytest.approx([p.t for p in truth], abs=1e-8, rel=0)
    counts = {}
    for view in ("full", "walls", "short", "sector"):
        stream = [restrict(s, p, view) for s, p in zip(measured, truth, strict=True)]
        counts[view] = {"single": Counter(), "multi": Counter()}
        for index in (0, 30, 60):
            part = stream[index:index+31]
            for name in ("single", "multi"):
                try:
                    if name == "single":
                        pose = estimate_initial(part[0].xy, pixels, meta)
                    else:
                        pose = estimate_multiscan(part, pixels, meta)
                    result = classify(pose, truth[index])
                except ValueError as exc:
                    assert any(word in str(exc).lower() for word in
                               ("ambiguous", "too few", "supported", "non-empty")), str(exc)
                    result = dict(result="refused", reason=str(exc))
                counts[view][name][result["result"]] += 1
                print(json.dumps(dict(seed=seed, view=view, index=index, method=name,
                                      endpoints=len(part[0].xy), **result)))
    assert counts["full"]["single"]["correct"] >= 1
    assert sum(counts["walls"]["single"].values()) == 3
    print(json.dumps({"seed": seed, "counts": counts}))
