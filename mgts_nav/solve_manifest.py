"""Fingerprint the inputs and products of a completed solve run."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path, PureWindowsPath
import sys

import yaml


REQUIRED = (
    "profiles/mapping.json", "profiles/inspection.json", "map/map.yaml",
    "map/summary.json", "map/trajectory.csv", "inspection/summary.json", "inspection/trajectory.csv",
)
OPTIONAL = (
    "profiles/mapping-pairs.jsonl", "profiles/inspection-pairs.jsonl",
    "profiles/zones.json", "inspection/zone-visits.csv",
    "map/route.png", "map/loop.json", "inspection/route.png", "inspection/continuous.csv",
    "inspection/continuous-summary.json", "baseline/trajectory.csv", "baseline-status.txt",
)


def member(root, name):
    if not isinstance(name, str) or not name or PureWindowsPath(name).anchor or ".." in PureWindowsPath(name).parts:
        raise ValueError(f"Invalid relative file name: {name!r}")
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"File outside directory: {name}")
    return path


def bag_files(path):
    if path.is_file():
        return [path]
    metadata = path / "metadata.yaml"
    try:
        info = yaml.safe_load(metadata.read_bytes())["rosbag2_bagfile_information"]
        names = info["relative_file_paths"]
        if not isinstance(names, list) or not names:
            raise ValueError("empty volume list")
        volumes = [member(path, name) for name in names]
        if len(set(volumes)) != len(volumes):
            raise ValueError("duplicate volume")
    except (KeyError, TypeError, ValueError, yaml.YAMLError) as exc:
        raise ValueError(f"{metadata}: invalid bag metadata ({exc})") from exc
    return [metadata, *volumes]


def digest(path):
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def fingerprints(mapping, inspection, solution):
    result = {}
    for role, bag in (("mapping", mapping), ("inspection", inspection)):
        for path in bag_files(bag):
            name = path.relative_to(bag).as_posix() if bag.is_dir() else "recording"
            result[f"{role}/{name}"] = digest(path)
    files = [solution / name for name in REQUIRED]
    files.extend(solution / name for name in OPTIONAL if (solution / name).exists())
    map_yaml = solution / "map/map.yaml"
    try:
        image = yaml.safe_load(map_yaml.read_bytes())["image"]
        files.append(member(map_yaml.parent, image))
    except (KeyError, TypeError, ValueError, yaml.YAMLError) as exc:
        raise ValueError(f"{map_yaml}: invalid image reference") from exc
    files.extend(bag_files(solution / "rviz"))
    for path in files:
        result["solution/" + path.relative_to(solution).as_posix()] = digest(path)
    return result


def metadata(solution):
    summary = json.loads((solution / "inspection/summary.json").read_bytes())
    map_info = yaml.safe_load((solution / "map/map.yaml").read_bytes())
    return dict(map_resolution_m=map_info["resolution"], initialization=summary.get("initialization"),
                initial_pose=summary.get("initial_pose"))


def warn_shared_volumes(hashes):
    groups = defaultdict(lambda: defaultdict(list))
    for key, checksum in hashes.items():
        role, _, name = key.partition("/")
        if role in ("mapping", "inspection") and name != "metadata.yaml":
            groups[checksum][role].append(key)
    for files in groups.values():
        if "mapping" in files and "inspection" in files:
            print("WARNING: Byte-identical recording volumes: "
                  + ", ".join(files["mapping"] + files["inspection"]), file=sys.stderr)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    for action in ("create", "verify"):
        command = commands.add_parser(action)
        for arg in ("mapping", "inspection", "solution", "manifest"):
            command.add_argument(arg, type=Path)
        if action == "create":
            command.add_argument("--solve-revision", help="Known version of the code used for this solve")
    args = parser.parse_args(argv)
    mapping, inspection, solution = (p.resolve() for p in (args.mapping, args.inspection, args.solution))
    try:
        actual = fingerprints(mapping, inspection, solution)
        if args.action == "create":
            data = dict(format=1, solve_revision=args.solve_revision, parameters=metadata(solution), sha256=actual)
            payload = json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            with args.manifest.open("x", encoding="utf-8") as stream:
                stream.write(payload + "\n")
        else:
            data = json.loads(args.manifest.read_bytes())
            if not isinstance(data, dict) or data.get("format") != 1 or not isinstance(data.get("sha256"), dict):
                raise ValueError("Invalid solve manifest format")
            expected = data["sha256"]
            changed = sorted(key for key in expected.keys() | actual.keys() if expected.get(key) != actual.get(key))
            if changed:
                raise ValueError("Changed file(s): " + ", ".join(changed))
            if data.get("parameters") != metadata(solution):
                raise ValueError("Manifest parameters disagree with solve products")
            print(f"Verified {len(actual)} files")
        warn_shared_volumes(actual)
    except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError) as exc:
        parser.exit(2, f"{exc}\n")


if __name__ == "__main__":
    main()
