import csv
import json
import subprocess
import sys

import numpy as np
import pytest

from mgts_nav.bag import load_config, scans
from mgts_nav.continuous import read_anchors, reconstruct
from mgts_nav.demo import PROFILE, generate
from mgts_nav.grid import Grid, read_map
from mgts_nav.metrics import Pose, evaluate, read_csv
from mgts_nav.slam import track
from mgts_nav.stress import fault_bag


def test_prediction_uses_body_frame_and_short_heading_interpolation():
    odom = np.array([[0, 0, 0, np.radians(179)], [1, -1, 0, np.radians(-179)],
                     [2, -2, 0, np.radians(-177)]])
    anchor = np.array([[.5, 5, 8, np.pi / 2]])
    result, info = reconstruct(anchor, odom, 1.1)
    assert [r.t for r in result] == [.5, 1., 2.]
    assert result[1].status == 'prediction_only'
    assert (result[1].x, result[1].y, result[1].yaw) == pytest.approx((5, 8.5, np.radians(91)))
    assert info['skipped_odom_samples'] == 1


def test_odom_outage_needs_a_new_matched_anchor():
    odom = np.array([[t, t, 0, 0] for t in (0, 1, 5, 6, 7, 8)], dtype=float)
    anchors = np.array([[0, 0, 0, 0], [3, 10, 0, 0], [7, 12, 0, 0.]])
    result, info = reconstruct(anchors, odom, 1.1)
    assert [(r.t, r.x, r.status) for r in result] == [
        (0, 0, 'matched'), (1, 1, 'prediction_only'), (3, 10, 'matched'),
        (7, 12, 'matched'), (8, 13, 'prediction_only')]
    assert info['odom_gaps_s'] == [[1, 5]]
    assert info['max_prediction_age_s'] == 1


def test_predictions_never_become_anchors(tmp_path):
    p = tmp_path / 'scan_trajectory.csv'
    p.write_text('t,x,y,yaw,status\n1,0,0,0,matched\n2,99,8,2,prediction_only\n')
    np.testing.assert_array_equal(read_anchors(p), [[1, 0, 0, 0]])
    rows, _ = reconstruct(read_anchors(p), np.empty((0, 4)), .3)
    assert len(rows) == 1 and rows[0].status == 'matched'


def test_no_messages_dropout_via_bag_and_cli(tmp_path):
    data = tmp_path / 'demo'
    generate(data)
    cfg = load_config(PROFILE)
    grid = Grid(.08)
    list(track(scans(data / 'mapping', cfg), np.zeros(3), grid=grid))
    grid.save(tmp_path)
    points = read_map(tmp_path / 'map.yaml')[2]
    initial = json.loads((data / 'initial_pose.json').read_text())
    altered = tmp_path / 'no_messages'
    assert fault_bag(data / 'inspection', altered, 'lidar_no_messages', cfg) == 20
    truth = read_csv(data / 'inspection_truth.csv')
    for source in (data / 'inspection', altered):
        anchors = list(track(scans(source, cfg), initial, map_points=points))
        scan_csv = tmp_path / f'{source.name}.csv'
        output = tmp_path / f'{source.name}-continuous.csv'
        with scan_csv.open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['t', 'x', 'y', 'yaw', 'status'])
            writer.writerows([r.t, *r.pose, r.status] for r in anchors)
        cmd = [sys.executable, '-m', 'mgts_nav.continuous', str(source), str(scan_csv),
               str(output), '--config', str(PROFILE)]
        info = json.loads(subprocess.run(cmd, check=True, capture_output=True, text=True).stdout)
        native = evaluate([Pose(r.t, *r.pose) for r in anchors], truth)
        dense = evaluate(read_csv(output), truth)
        assert dense['coverage'] == 1.0 and info['prediction_samples'] > 0
        assert not info['odom_gaps_s']
        changed = source == altered
        assert native['coverage'] == (0.8 if changed else 1.0)
        assert 0 < dense['ate_xy_rmse_m'] < .15
        assert (2. < info['max_prediction_age_s'] < 2.1) if changed else info['max_prediction_age_s'] < .1
        print(source.name, 'native ATE', native['ate_xy_rmse_m'], 'continuous ATE',
              dense['ate_xy_rmse_m'], 'max prediction age', info['max_prediction_age_s'])
