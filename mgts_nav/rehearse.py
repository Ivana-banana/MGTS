"""Run both synthetic bags, compare localization to raw odometry, save evidence."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

from .demo import generate
from .metrics import evaluate, read_csv


def run(output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f'Choose a new output directory: {output}')
    output.parent.mkdir(parents=True, exist_ok=True)
    profile = Path(__file__).resolve().parents[1] / 'configs' / 'example.json'
    with TemporaryDirectory(dir=output.parent) as tmp:
        stage = Path(tmp) / 'run'
        stage.mkdir()
        data = stage / 'data'
        generate(data)
        def nav(*args):
            subprocess.run([sys.executable, '-m', 'mgts_nav', *map(str, args)], check=True)
        nav('map', data / 'mapping', '--config', profile, '--out', stage / 'map')
        options = [data / 'inspection', '--config', profile, '--initial-file', data / 'initial_pose.json']
        nav('baseline', *options, '--out', stage / 'baseline')
        nav('localize', *options, '--map', stage / 'map' / 'map.yaml', '--out', stage / 'inspection')
        truth = read_csv(data / 'inspection_truth.csv')
        metrics = {name: evaluate(read_csv(stage / name / 'trajectory.csv'), truth)
                   for name in ('baseline', 'inspection')}
        metrics['mapping'] = evaluate(read_csv(stage / 'map' / 'trajectory.csv'),
                                      read_csv(data / 'mapping_truth.csv'))
        report = json.dumps(metrics, indent=2, allow_nan=False)
        (stage / 'metrics.json').write_text(report, encoding='utf-8')
        stage.rename(output)
    print(report)
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    try:
        run(parser.parse_args().out)
    except (OSError, ValueError) as error:
        parser.exit(2, f'{error}\n')
    except KeyboardInterrupt:
        parser.exit(130, 'Interrupted; no output directory was created\n')
