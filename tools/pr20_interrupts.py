"""Replay PR 20's Ctrl+C cases on Windows, each in a private hidden console."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def interrupt(pid):
    # This helper attaches only to the private console created below. Group 0
    # addresses the whole console, including a solve/rehearse mapping worker.
    import ctypes
    kernel = ctypes.windll.kernel32
    kernel.FreeConsole()
    if not kernel.AttachConsole(pid):
        raise ctypes.WinError()
    if not kernel.SetConsoleCtrlHandler(None, True) or not kernel.GenerateConsoleCtrlEvent(0, 0):
        raise ctypes.WinError()
    time.sleep(.2)


def run_case(root, name, args, marker):
    directory = root / name
    directory.mkdir()
    target = directory / "product"
    startup = subprocess.STARTUPINFO()
    startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startup.wShowWindow = subprocess.SW_HIDE
    with (directory / "stderr.txt").open("wb") as err, (directory / "stdout.txt").open("wb") as out:
        with subprocess.Popen([sys.executable, "-m", *args, str(target)], stdout=out, stderr=err,
                              creationflags=subprocess.CREATE_NEW_CONSOLE, startupinfo=startup) as child:
            deadline = time.monotonic() + 60
            while not any(directory.glob(marker)):
                if child.poll() is not None or time.monotonic() >= deadline:
                    child.kill()
                    raise RuntimeError(f"{name}: CLI did not reach stage {marker}")
                time.sleep(.02)
            sender = "import sys; from tools.pr20_interrupts import interrupt; interrupt(int(sys.argv[1]))"
            subprocess.run([sys.executable, "-c", sender, str(child.pid)],
                           check=True, creationflags=subprocess.CREATE_NO_WINDOW)
            code = child.wait(timeout=45)
    diagnostic = (directory / "stderr.txt").read_text(encoding="utf-8", errors="replace")
    return dict(case=name, exit=code, traceback="Traceback (" in diagnostic,
                output_exists=target.exists(), leftover_dirs=[p.name for p in directory.iterdir() if p.is_dir()])


if __name__ == "__main__":
    if sys.platform != "win32":
        sys.exit("Requires Windows console control events")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path, help="Generated demo directory")
    parser.add_argument("output", type=Path, help="New directory for captured diagnostics")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    mapping, inspection = str(args.data / "mapping"), str(args.data / "inspection")
    cfg = ["--config", "configs/example.json"]
    cases = [
        ("map", ["mgts_nav", "map", mapping, *cfg, "--out"], "tmp*/result"),
        ("solve", ["mgts_nav.solve", mapping, inspection, *cfg, "--out"], "tmp*/solution/tmp*/result"),
        ("rehearse", ["mgts_nav.rehearse"], "tmp*/run/tmp*/result"),
    ]
    failed = []
    for case in cases:
        result = run_case(args.output, *case)
        print(json.dumps(result), flush=True)
        if result["exit"] != 130 or result["traceback"] or result["output_exists"] or result["leftover_dirs"]:
            failed.append(result["case"])
    if failed:
        sys.exit(f"Unexpected interrupt outcome for {', '.join(failed)}")
