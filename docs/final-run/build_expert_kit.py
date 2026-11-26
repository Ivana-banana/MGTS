"""Assemble the private Go2 expert kit from the checked release files."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path(__file__).resolve().parents[2]
RUN = "a5ad606167c82d7d462a3de9dc2fae889315cf06"
VIEW = "c935bb13f1f8348426d3ea974417b8164c100184"


def git_bytes(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT)


def assemble(bundle, output):
    index = json.loads((ROOT / "docs/final-run/index.json").read_text(encoding="utf-8"))
    if not index["complete"] or index["commit"] != RUN:
        raise ValueError("Unexpected final run index")
    files = {}
    for entry in index["files"]:
        if entry["role"] in {"mapping_input", "inspection_input"}:
            continue
        data = (bundle / entry["path"]).read_bytes()
        if hashlib.sha256(data).hexdigest() != entry["sha256"]:
            raise ValueError(f"Digest mismatch: {entry['path']}")
        files[entry["path"]] = data
    with tempfile.TemporaryDirectory() as temp:
        source = Path(temp) / "source.zip"
        git_bytes("archive", "--format=zip", "-o", str(source), RUN, "--",
                  "mgts_nav", "tools", "configs", "tests", "requirements.txt",
                  "requirements-win-amd64-py310.lock", "pytest.ini", "Dockerfile", ".dockerignore")
        files["source.zip"] = source.read_bytes()
    for path in ("configs/go2-result.rviz", "configs/go2-result-qos.yaml",
                 "configs/go2-video-top.rviz", "configs/go2-video-third-person.rviz",
                 "tools/go2_replay_view.py"):
        files["viewer/" + path] = git_bytes("show", f"{VIEW}:{path}")
    docs = ROOT / "docs"
    for name in ("go2-windows-ros.md", "go2-independent-check.md", "go2-video.md"):
        files["docs/" + name] = (docs / name).read_bytes()
    files["final-go2.pdf"] = (docs / "final-go2.pdf").read_bytes()
    files["README.md"] = (docs / "go2-expert-kit.md").read_bytes()
    files["official-run-index.json"] = (docs / "final-run/index.json").read_bytes()
    hashes = {name: hashlib.sha256(data).hexdigest() for name, data in sorted(files.items())}
    files["sha256.json"] = json.dumps(hashes, indent=2).encode("ascii")
    output.parent.mkdir(exist_ok=True, parents=True)
    with ZipFile(output, "x", ZIP_DEFLATED) as archive:
        for name, data in sorted(files.items()):
            archive.writestr(name, data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path, help="directory with the verified outputs of #200")
    parser.add_argument("output", type=Path, help="new ZIP path")
    args = parser.parse_args()
    assemble(args.bundle, args.output)
