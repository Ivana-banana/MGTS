import subprocess
import sys

import pytest

from mgts_nav.demo import PROFILE, generate


@pytest.fixture(scope="module")
def inputs(tmp_path_factory):
    root = tmp_path_factory.mktemp("cli-input")
    generate(root / "demo")
    subprocess.run([sys.executable, "-m", "mgts_nav", "map", str(root/"demo/mapping"),
                    "--config", str(PROFILE), "--out", str(root/"map")], check=True, capture_output=True)
    for name, content in {"other.txt": "not a recording", "wrong.yaml": "some text",
                          "invalid.json": "{", "object.json": '{"x":0}', "short.json": "[1,2]"}.items():
        (root/name).write_text(content)
    return root


CASES = {
    "inspect-text": "mgts_nav inspect other.txt",
    "map-text": "mgts_nav map other.txt --config PROFILE --out new",
    "localize-text": "mgts_nav localize other.txt --config PROFILE --map map/map.yaml --initial 0 0 0 --out new",
    "baseline-text": "mgts_nav baseline other.txt --config PROFILE --initial 0 0 0 --out new",
    "calibration-text": "mgts_nav.calibration other.txt --base a --lidar b --imu c",
    "profile-text": "mgts_nav.profile other.txt",
    "observability-text": "mgts_nav.observability --bag other.txt --config PROFILE",
    "metadata": "mgts_nav map demo/mapping/metadata.yaml --config PROFILE --out new",
    "profile-json": "mgts_nav map demo/mapping --config invalid.json --out new",
    "solve-profile-json": "mgts_nav.solve demo/mapping demo/inspection --config PROFILE --inspection-config invalid.json --out new",
}
LOCALIZE = "mgts_nav localize demo/inspection --config PROFILE"
for name, path in (("pgm", "map/map.pgm"), ("directory", "map"), ("other-yaml", "wrong.yaml")):
    CASES[name] = f"{LOCALIZE} --map {path} --initial 0 0 0 --out new"
for name in ("object", "short", "invalid"):
    CASES["initial-"+name] = f"{LOCALIZE} --map map/map.yaml --initial-file {name}.json --out new"
for name, args in (("two-values", "-1 0.55"), ("comma", "1,5 0 0")):
    CASES[name] = f"{LOCALIZE} --map map/map.yaml --initial {args} --out new"
CASES["existing"] = f"{LOCALIZE} --map map/map.yaml --initial 0 0 0 --out map"


@pytest.mark.parametrize("name", list(CASES))
def test_invalid_cli_input(inputs, name):
    tokens = CASES[name].split()
    args = []
    for i, token in enumerate(tokens):
        if token == "PROFILE":
            token = str(PROFILE)
        elif i >= (2 if tokens[0] == "mgts_nav" else 1):
            if token in {"map", "new"} or "/" in token or token.endswith((".txt", ".yaml", ".json")):
                token = str(inputs / token)
        args.append(token)
    run = subprocess.run([sys.executable, "-m", *args], capture_output=True, text=True)
    print(name, run.returncode, run.stderr.strip().splitlines()[-1])
    assert run.returncode == 2 and "Traceback" not in run.stderr, run.stderr
    assert not (inputs/"new").exists()
    if name.endswith("-text"):
        assert "other.txt" in run.stderr and "recording" in run.stderr
    elif name in ("profile-json", "solve-profile-json"):
        assert str(inputs/"invalid.json") in run.stderr and "invalid JSON" in run.stderr
    elif name == "metadata":
        assert "recording directory" in run.stderr
    elif name in ("pgm", "directory"):
        assert str(inputs/"map/map.yaml") in run.stderr
    elif name == "existing":
        assert len(run.stderr.strip().splitlines()) == 1
