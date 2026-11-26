import json

import pytest

from tools.final_run_index import COMMANDS, ROLES, build_index, main


def test_split_bag_and_download_verification(tmp_path):
    root = tmp_path
    (root / "run.json").write_text(json.dumps(dict(commit="b" * 40, commands=dict.fromkeys(COMMANDS, "test"))))
    for role, name in ROLES.items():
        target = root / name
        target.parent.mkdir(exist_ok=True)
        target.write_text(role)
    bag = root / "bag"
    bag.mkdir()
    (bag / "metadata.yaml").write_text("rosbag2_bagfile_information: {relative_file_paths: [a.mcap, b.mcap]}")
    for name in ("a.mcap", "b.mcap"):
        (bag / name).write_bytes(b"volume")
    assert main(["build", str(root)]) == main(["verify", str(root)]) == 0
    (bag / "b.mcap").write_bytes(b"tamper")
    assert main(["verify", str(root)]) == 1
    (bag / "b.mcap").unlink()
    with pytest.raises(FileNotFoundError):
        build_index(root, partial=True)


def test_partial_is_explicit(tmp_path):
    (tmp_path / "run.json").write_text(json.dumps(dict(commit="c" * 40, commands={})))
    index = build_index(tmp_path, partial=True)
    assert not index["complete"]
    assert {"command:mapping", "rviz_top", "output_bag"} <= set(index["missing"])
    with pytest.raises(ValueError, match="incomplete"):
        build_index(tmp_path)
