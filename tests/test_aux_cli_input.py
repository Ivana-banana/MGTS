import subprocess
import sys

import pytest

from mgts_nav.demo import PROFILE


@pytest.mark.parametrize("module,filename,content,expected", [
    ("continuous", "short.csv", "t,x,y,yaw,status\n1,0,0,0,matched\n2,0,0\n", ":3: invalid number"),
    ("loop", "missing.csv", None, "missing.csv"),
    ("loop", "words.csv", "t,x,y,yaw\na,b,c,d\n", ":2: invalid number"),
    ("map_eval", "missing.yaml", None, "missing.yaml"),
    ("map_eval", "broken.yaml", "image: map.pgm\nresolution: [\n", "invalid map YAML"),
    ("metrics", "excel.csv", "t;x;y;yaw\n0;1;2;3\n", "columns are separated by ';'"),
])
def test_invalid_auxiliary_cli_inputs(tmp_path, module, filename, content, expected):
    source = tmp_path / filename
    if content is not None:
        source.write_text(content, encoding="utf-8")
    dense = tmp_path / "dense.csv"
    args = {
        "continuous": [tmp_path, source, dense, "--config", PROFILE],
        "loop": [tmp_path, source, "--config", PROFILE],
        "map_eval": [source],
        "metrics": [source, source],
    }[module]
    run = subprocess.run([sys.executable, "-m", f"mgts_nav.{module}", *map(str, args)],
                         capture_output=True, text=True)
    assert run.returncode == 2, run.stderr
    assert len(run.stderr.strip().splitlines()) == 1
    assert filename in run.stderr and expected in run.stderr
    assert not dense.exists()
