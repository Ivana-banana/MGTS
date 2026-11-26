"""Build or verify the file index of a staged Go2 submission."""
import argparse
import json
import re
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile

from mgts_nav.solve_manifest import bag_files, digest, member


ROLES = {
    "mapping_input": "inputs/bag_for_mapping.zip",
    "inspection_input": "inputs/bag_for_localization.zip",
    "configuration": "profile.json",
    "map_3d": "map.npz",
    "latency": "latency.csv",
    "rviz_top": "video/top.mp4",
    "rviz_third_person": "video/third-person.mp4",
}
COMMANDS = ("mapping", "localization", "playback")


def build_index(root, partial=False):
    root = root.resolve()
    run_path = member(root, "run.json")
    run = json.loads(run_path.read_text(encoding="utf-8-sig"))
    if not isinstance(run, dict) or not re.fullmatch(r"[a-f0-9]{40}", str(run.get("commit"))):
        raise ValueError("run.json needs the full solution commit")
    commands = run.get("commands")
    if not isinstance(commands, dict) or set(commands) - set(COMMANDS):
        raise ValueError("commands must be an object with mapping, localization and playback keys")
    missing = []
    for key in COMMANDS:
        if key not in commands:
            missing.append(f"command:{key}")
        elif not isinstance(commands[key], str) or not commands[key].strip():
            raise ValueError(f"empty or invalid command: {key}")

    entries = []
    def record(role, path):
        relative = path.relative_to(root).as_posix()
        path = member(root, relative)
        size = path.stat().st_size
        if not path.is_file() or size == 0:
            raise ValueError(f"expected a nonempty file: {relative}")
        entries.append(dict(role=role, path=relative, bytes=size, sha256=digest(path)))

    record("provenance", run_path)
    for role, name in ROLES.items():
        path = member(root, name)
        if path.exists():
            record(role, path)
        else:
            missing.append(role)
    bag = member(root, "bag")
    if bag.exists():
        if not bag.is_dir():
            raise ValueError("bag must be a rosbag2 directory")
        files = bag_files(bag)
        for i, path in enumerate(files):
            record("bag_metadata" if i == 0 else "bag_volume", path)
    else:
        missing.append("output_bag")
    if missing and not partial:
        raise ValueError("incomplete submission: " + ", ".join(missing))
    return dict(schema=1, commit=run["commit"], commands=commands,
                complete=not missing, missing=missing, files=entries)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["build", "verify"])
    parser.add_argument("root", type=Path)
    parser.add_argument("--allow-partial", action="store_true", help="accept a rehearsal with explicitly missing roles")
    args = parser.parse_args(argv)
    try:
        index = build_index(args.root, args.allow_partial)
        target = args.root / "index.json"
        if args.action == "verify":
            if json.loads(target.read_text(encoding="utf-8-sig")) != index:
                raise ValueError("index differs from the downloaded files or run metadata")
        else:
            with NamedTemporaryFile(mode="w", encoding="utf-8", dir=args.root, delete=False) as output:
                temporary = Path(output.name)
                json.dump(index, output, ensure_ascii=False, indent=2)
                output.write("\n")
            try:
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
        print(f"{args.action}: {len(index['files'])} files, complete={index['complete']}")
        return 0
    except (OSError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
