import csv
import json

import pytest

from mgts_nav.bag import load_config
from mgts_nav.demo import PROFILE, generate
from mgts_nav.solve import solve
from mgts_nav.solve_manifest import fingerprints
from mgts_nav.stress import fault_bag


def test_solve_zones_excludes_prediction_intervals(tmp_path):
    data = tmp_path / 'recordings'
    generate(data)
    inspection = tmp_path / 'outage'
    fault_bag(data/'inspection', inspection, 'lidar_no_returns', load_config(PROFILE))
    zones = tmp_path / 'zones.json'
    floor = [[-20, -20], [20, -20], [20, 20], [-20, 20]]
    zones.write_text(json.dumps({'frame': 'map', 'zones': {'floor': floor}}), encoding='utf-8')
    output = tmp_path / 'solution'
    initial = json.loads((data/'initial_pose.json').read_text())
    solve(data/'mapping', inspection, PROFILE, output, initial=initial, zones=zones, zone_max_gap=.25)
    with (output/'inspection/trajectory.csv').open(newline='') as source:
        rows = list(csv.DictReader(source))
    with (output/'inspection/zone-visits.csv').open(newline='') as source:
        entries = list(csv.DictReader(source))
    assert len(entries) == 2 and any(row['status'] == 'prediction_only' for row in rows)
    for event in entries:
        span = [r for r in rows if float(event['first_inside_t']) <= float(r['t']) <= float(event['last_inside_t'])]
        assert len(span) == int(event['samples']) and all(r['status'] == 'matched' for r in span)
        assert event['entry_observed'] == event['exit_observed'] == '0'
    assert (output/'profiles/zones.json').read_bytes() == zones.read_bytes()
    hashed = fingerprints(data/'mapping', inspection, output)
    assert {'solution/profiles/zones.json', 'solution/inspection/zone-visits.csv'} <= hashed.keys()


def test_incomplete_zone_options_fail_before_io(tmp_path):
    with pytest.raises(ValueError, match='together'):
        solve('absent', 'absent', 'absent', tmp_path/'result', zones='absent')
