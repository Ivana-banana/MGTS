"""Check the real CLI, output frames, map export, and withheld inspection truth."""
import json

from PIL import Image

from mgts_nav.rehearse import run


def test_rehearsal(tmp_path, capsys):
    out = tmp_path / 'run'
    metrics = run(out)
    assert metrics['inspection']['coverage'] == 1
    assert metrics['inspection']['ate_xy_rmse_m'] < metrics['baseline']['ate_xy_rmse_m']
    assert metrics['mapping']['ate_xy_rmse_m'] < 0.12
    assert (out / 'map' / 'map.pgm').stat().st_size > 100
    assert json.loads((out / 'inspection' / 'summary.json').read_text())['status']['matched'] >= 95
    # Pictures must be readable without zooming in.
    for picture in ('map/route.png', 'inspection/route.png', 'comparison.png'):
        with Image.open(out / picture) as image:
            assert max(image.size) >= 900, picture
    report = capsys.readouterr().out
    assert f"ATE {100 * metrics['inspection']['ate_xy_rmse_m']:.1f} см".replace('.', ',') in report
    assert str(out / 'comparison.png') in report and str(out / 'metrics.json') in report
    assert '{' not in report
