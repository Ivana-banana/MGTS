"""Check the real CLI, output frames, map export, and withheld inspection truth."""
import json

from mgts_nav.rehearse import run


def test_rehearsal(tmp_path):
    out = tmp_path / 'run'
    metrics = run(out)
    assert metrics['inspection']['coverage'] == 1
    assert metrics['inspection']['ate_xy_rmse_m'] < metrics['baseline']['ate_xy_rmse_m']
    assert metrics['mapping']['ate_xy_rmse_m'] < 0.12
    assert (out / 'map' / 'map.pgm').stat().st_size > 100
    assert json.loads((out / 'inspection' / 'summary.json').read_text())['status']['matched'] >= 95
