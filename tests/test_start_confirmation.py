from dataclasses import replace
import sys

import numpy as np
import pytest

from mgts_nav import __main__ as cli
from mgts_nav.bag import Scan, load_config, scans
from mgts_nav.demo import PROFILE, generate
from mgts_nav.grid import Grid, read_map
from mgts_nav.initialization import estimate_initial
from mgts_nav.metrics import read_csv, wrap
from mgts_nav.slam import track
from mgts_nav.start_confirmation import confirm_start


@pytest.mark.parametrize("seed", [41, 73])
def test_full_views_and_reproduced_false_accept(tmp_path, seed, monkeypatch, capsys):
    root = tmp_path / "demo"
    generate(root, layout="corridor", seed=seed)
    cfg = load_config(PROFILE)
    grid = Grid(.08)
    list(track(scans(root / "mapping", cfg), np.zeros(3), grid=grid))
    grid.save(tmp_path)
    pixels, meta, _ = read_map(tmp_path / "map.yaml")
    stream = list(scans(root / "inspection", cfg))
    truth = read_csv(root / "inspection_truth.csv")
    for index in (0, 30, 60):
        first = stream[index]
        candidate = estimate_initial(first.xy, pixels, meta)
        ref = truth[index]
        assert np.linalg.norm(candidate[:2] - ref[1:3]) <= .25
        assert abs(wrap(candidate[2] - ref.yaw)) <= np.deg2rad(10)
        accepted = confirm_start(candidate, first, iter(stream[index+1:]), pixels, meta)
        np.testing.assert_array_equal(accepted, candidate)
    if seed == 73:
        limited = [replace(s, xy=s.xy[np.linalg.norm(s.xy-s.origin, axis=1) <= 3])
                   for s in stream[30:61]]
        first = limited[0]
        candidate = estimate_initial(first.xy, pixels, meta)
        ref = truth[30]
        assert np.linalg.norm(candidate[:2] - ref[1:3]) > 5
        assert abs(wrap(candidate[2] - ref.yaw)) > np.deg2rad(170)
        with pytest.raises(ValueError, match="Ambiguous|disagree") as reason:
            confirm_start(candidate, first, limited[1:], pixels, meta)
        # Exercise the same limited observations through the localization entry point.
        output = tmp_path / "limited-result"
        monkeypatch.setattr(cli, "scans", lambda bag, config: iter(limited))
        monkeypatch.setattr(sys, "argv", ["mgts_nav", "localize", str(root / "inspection"),
                            "--map", str(tmp_path / "map.yaml"), "--config", str(PROFILE),
                            "--out", str(output), "--confirm-start"])
        with pytest.raises(SystemExit) as failure:
            cli.main()
        assert failure.value.code == 2
        assert f"Could not confirm automatic start: {reason.value}" in capsys.readouterr().err
        assert not output.exists()


def test_rejects_reuse_of_first_scan_or_missing_window():
    first = Scan(10., np.empty((0, 2)), np.zeros(2), np.zeros(3), None)
    with pytest.raises(ValueError, match="strictly later"):
        confirm_start(np.zeros(3), first, [first], None, None)
    with pytest.raises(ValueError, match="Insufficient"):
        confirm_start(np.zeros(3), first, [replace(first, t=13.01)], None, None)
