"""Build or verify the MGTS runtime ZIP from an immutable Git revision."""
import argparse
import ast
import hashlib
import io
from pathlib import Path
import subprocess
import os
from tempfile import NamedTemporaryFile
import zipfile


ROOT = Path(__file__).resolve().parents[1]
MODULES = (
    '__init__', '__main__', 'bag', 'benchmark', 'calibration', 'camera_demo', 'camera_info', 'clock_experiment',
    'coarse_start', 'continuous', 'delayed_start', 'demo', 'deskew', 'glass_experiment', 'grid',
    'image_pairs', 'initialization', 'interior_loop', 'leveling', 'loop', 'map_cache', 'map_eval', 'metrics',
    'motion_disagreement', 'multiscan', 'native_grid', 'observability', 'obstacles', 'plan_alignment',
    'pose_graph', 'profile', 'quality', 'drift_warning', 'recovery', 'rehearse', 'ros_export', 'self_check', 'slam', 'solve',
    'solve_audit', 'solve_manifest', 'start_confirmation', 'start_trial', 'stress', 'sync', 'vision',
    'visual_fusion', 'zones', 'event_uncertainty',
)
TESTS = (
    'ambiguity', 'aux_cli_input', 'bag', 'calibration', 'camera_demo', 'cli_input_audit',
    'coarse_corridor', 'coarse_start_budget', 'continuous', 'corridor', 'delayed_start_controls', 'demo',
    'custom_lidar', 'deskew', 'deskew_experiment', 'glass_experiment', 'gyro_deskew', 'heap1_warning_score', 'image_pairs', 'imu_availability',
    'initialization', 'intake', 'intel_lab', 'intel_laser_clock', 'intel_longitudinal',
    'intel_order_odom', 'intel_scan_pair', 'interior_loop', 'joint_outage', 'laser_scan', 'leveling',
    'leveling_experiment', 'loop', 'map_cache', 'map_eval', 'map_yaml', 'mapping_rejections', 'metrics',
    'motion_disagreement', 'multiscan', 'native_grid', 'observability', 'obstacles', 'organized_solve',
    'plan_alignment', 'plot_quality', 'plot_zones', 'point_time_semantics', 'pose_graph',
    'prepared_window', 'profile', 'quality', 'recording_order', 'recovery', 'rehearsal', 'ros1_pipeline',
    'ros_export', 'route_preview', 'scan_pair_overlap', 'scan_rejections', 'self_check', 'solve', 'solve_audit',
    'solve_manifest', 'solve_zones', 'split_bags', 'start_confirmation', 'start_search_cli', 'stress',
    'submap_controls', 'sync', 'sync_imu', 'turn_recovery', 'unicode_volume', 'velocity_prior', 'vision',
    'tum_depth_motion', 'tum_sparse_depth', 'visit_agreement', 'visual_fusion', 'zones', 'event_uncertainty',
)
FILES = tuple(sorted((
    '.dockerignore', 'Dockerfile', 'README.md', 'requirements.txt', 'requirements-win-amd64-py310.lock', 'pytest.ini',
    'configs/demo-zones.json', 'configs/example.json', 'configs/inspection.rviz', 'docs/ambiguity.md',
    'configs/custom_lidar/Packet.msg', 'configs/custom_lidar/Return.msg', 'tools/custom_lidar.py',
    'docs/drift-warning-trial.md', 'docs/independent-icp.md', 'docs/kitti-repeat.md',
    'docs/jury-field-example.html', 'tools/jury_report.py',
    'docs/drift-recovery-trial.md', 'tools/drift_recovery_trial.py',
    'docs/event-uncertainty.md', 'tools/event_uncertainty_trial.py',
    'docs/gait-slip.md', 'tools/gait_slip_trial.py',
    'docs/heldout-floorplan.md', 'docs/heldout-floorplan.png', 'tools/heldout_floorplan.py',
    'docs/internal-graph-trial.md', 'tools/internal_graph_trial.py',
    'docs/velocity-transfer.md', 'tools/velocity_transfer.py',
    'docs/visual-solve-transfer.md', 'tools/visual_solve_transfer.py',
    'docs/ray-start-trial.md', 'tools/ray_start_trial.py',
    'docs/stable-occupancy-trial.md', 'tools/stable_occupancy_trial.py',
    'tools/drift_warning_trial.py', 'tools/independent_icp.py', 'tools/kitti_repeat.py', 'tools/kitti_warning_trial.py',
    'docs/angular-recovery-controls.md', 'docs/camera-diagnosis.csv', 'docs/camera-diagnosis.md',
    'docs/camera-experiment.md', 'docs/clock-offset.md', 'docs/coarse-corridor.md',
    'docs/coarse-start.md', 'docs/corridor.md', 'docs/delayed-start-controls.md', 'docs/delayed-start.md',
    'docs/demo-results.md', 'docs/demo-results.png', 'docs/demo-zone-map.png',
    'docs/demo-zone-visits.csv', 'docs/deskew-experiment.md', 'docs/false-zone-visits.md',
    'docs/glass-person.md', 'docs/glass-person.png', 'docs/heap1-clock-lag.md', 'docs/heap1-drift.png',
    'docs/heap1-factorial.md', 'docs/heap1-field.png', 'docs/heap1-gyro-dropout.md',
    'docs/heap1-inspection-deskew.md', 'docs/heap1-long-mapping.md', 'docs/heap1-map-anchors.md',
    'docs/heap1-point-times.md', 'docs/heap1-pr81-self-check.md', 'docs/heap1-surface.md',
    'docs/heap1-warning.md', 'docs/heap1-turn-recovery.md', 'docs/heap1-velocity-prior.md', 'docs/image-pairs.md',
    'docs/intake-rehearsal.md', 'docs/intake.md', 'docs/intel-lab.md', 'docs/intel-laser-clock.md',
    'docs/intel-laserscan.md',
    'docs/intel-scan-divergence.md',
    'docs/intel-voxel-ablation.md',
    'docs/intel-longitudinal.md', 'docs/intel-longitudinal.png', 'docs/intel-order-odom.md',
    'docs/intel-relative-motion.md', 'docs/intel-scan-pair.md', 'docs/interior-loop.md',
    'docs/joint-outage.md', 'docs/keyframe-start.md', 'docs/leveling-experiment.md',
    'docs/lidar-extrinsics.md', 'docs/micrometre-sensitivity.md', 'docs/motion-disagreement.md', 'docs/native-grid-memory.md',
    'docs/native-grid.md', 'docs/obstacles.md', 'docs/office-plan.png', 'docs/package.md',
    'docs/plan-alignment.md', 'docs/plan-landmarks.json', 'docs/plan-route.png', 'docs/point-time.md',
    'docs/pr20-cli-audit.md', 'docs/public-data.md', 'docs/qualification.txt', 'docs/quality-clean.png',
    'docs/quality-gap.png', 'docs/report.md', 'docs/ros1-validation.md', 'docs/runtime.md',
    'docs/scan-pair-overlap.md', 'docs/scan-rejections.md', 'docs/self-check.md', 'docs/sensors.md', 'docs/solve-audit.md',
    'docs/solve-manifest.md', 'docs/start-confirmation.md', 'docs/submap-search-profile.md',
    'docs/submap-start-controls.md', 'docs/visit-agreement.md', 'docs/zones.md',
    'docs/tum-depth-motion.md', 'docs/tum-visual-yaw.md',
    'tools/angular_recovery_controls.py', 'tools/benchmark_grid_memory.py', 'tools/benchmark_length.py',
    'tools/benchmark_native_grid.py', 'tools/diagnose_camera.py', 'tools/evaluate_camera.py',
    'tools/delayed_after_gap.py', 'tools/evaluate_deskew.py', 'tools/evaluate_false_visits.py', 'tools/evaluate_joint_outage.py',
    'tools/evaluate_leveling.py', 'tools/evaluate_lidar_extrinsics.py', 'tools/evaluate_micrometre.py',
    'tools/evaluate_motion_disagreement.py', 'tools/evaluate_visit_agreement.py',
    'tools/evaluate_zone_visits.py', 'tools/heap1_clock_lag.py', 'tools/heap1_drift.py',
    'tools/heap1_factorial.py', 'tools/heap1_gyro_dropout.py', 'tools/heap1_holdout.py',
    'tools/heap1_inspection_deskew.py', 'tools/heap1_micrometre.py', 'tools/heap1_long_mapping.py', 'tools/heap1_map_anchors.py',
    'tools/heap1_point_times.py', 'tools/heap1_revisit.py', 'tools/heap1_sample.py',
    'tools/heap1_stationary_start.py', 'tools/heap1_submap_holdout.py', 'tools/heap1_submap_start.py',
    'tools/heap1_warning_inputs.py', 'tools/heap1_warning_score.py',
    'tools/heap1_surface_trial.py', 'tools/heap1_timed_bag.py', 'tools/heap1_turn_recovery.py',
    'tools/heap1_velocity_prior.py', 'tools/intel_lab.py', 'tools/intel_laser_clock.py',
    'tools/intel_laserscan.py',
    'tools/intel_scan_divergence.py',
    'tools/intel_voxel_ablation.py',
    'tools/intel_longitudinal.py', 'tools/intel_order_odom.py', 'tools/intel_scan_pair.py',
    'tools/__init__.py', 'tools/laser_scan.py', 'tools/repeat_solve.py',
    'tools/native_grid.cpp', 'tools/package_solution.py', 'tools/plot_heap1_drift.py',
    'tools/plot_heap1_revisit.py', 'tools/plot_quality.py', 'tools/plot_zones.py',
    'tools/obstacle_pose_controls.py', 'tools/post_gap_controls.py', 'tools/pr20_interrupts.py', 'tools/profile_submap_search.py', 'tools/scan_pair_overlap.py',
    'tools/tum_depth_motion.py', 'tools/tum_sparse_depth.py', 'tools/tum_speed_bounds.py', 'tools/tum_visual_yaw.py',
    *(f'mgts_nav/{name}.py' for name in MODULES),
    *(f'tests/test_{name}.py' for name in TESTS),
)))


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT)


def revision_files(commit):
    """Read the selected packager's declarative inventory without executing revision code."""
    tree = ast.parse(git('show', commit + ':tools/package_solution.py'))
    declarations = {node.targets[0].id: node.value for node in tree.body
                    if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)}
    expr = declarations['FILES']
    if not (isinstance(expr, ast.Call) and isinstance(expr.func, ast.Name) and expr.func.id == 'tuple'
            and len(expr.args) == 1 and not expr.keywords):
        raise ValueError('Unsupported FILES declaration in ' + commit)
    order = expr.args[0]
    if not (isinstance(order, ast.Call) and isinstance(order.func, ast.Name) and order.func.id == 'sorted'
            and len(order.args) == 1 and not order.keywords and isinstance(order.args[0], ast.Tuple)):
        raise ValueError('Expected sorted literal inventory in ' + commit)
    paths = []
    expansions = {ast.dump(ast.parse(f"(f'{prefix}{{name}}.py' for name in {group})", mode='eval').body):
                  (group, prefix) for group, prefix in [('MODULES', 'mgts_nav/'), ('TESTS', 'tests/test_')]}
    for item in order.args[0].elts:
        if isinstance(item, ast.Constant) and isinstance(item.value, str):
            paths.append(item.value)
        elif isinstance(item, ast.Starred) and ast.dump(item.value) in expansions:
            group, prefix = expansions[ast.dump(item.value)]
            names = ast.literal_eval(declarations[group])
            if not isinstance(names, tuple) or not all(isinstance(name, str) for name in names):
                raise ValueError('Expected literal module names in ' + group)
            paths.extend(prefix + name + '.py' for name in names)
        else:
            raise ValueError('Unsupported inventory entry in ' + commit)
    return tuple(sorted(paths))


def build(revision):
    commit = git('rev-parse', '--verify', '--end-of-options', revision + '^{commit}').decode().strip()
    files = revision_files(commit)
    entries = {}
    for record in git('ls-tree', '-r', '-z', commit, '--', *files).split(b'\0'):
        if record:
            header, name = record.split(b'\t', 1)
            mode, kind, oid = header.split()
            if mode not in (b'100644', b'100755') or kind != b'blob':
                raise ValueError(f'Expected a regular file: {name!r}')
            entries[name.decode()] = oid
    # Request immutable blobs directly so checkout filters cannot convert line endings.
    request = b''.join(entries[path] + b'\n' for path in files)
    blobs = io.BytesIO(subprocess.check_output(['git', 'cat-file', '--batch'], input=request, cwd=ROOT))
    output = io.BytesIO()
    with zipfile.ZipFile(output, 'w') as archive:
        for path in files:
            oid, kind, size = blobs.readline().split()
            if oid != entries[path] or kind != b'blob':
                raise ValueError(f'Unexpected Git object for {path}')
            content = blobs.read(int(size))
            if blobs.read(1) != b'\n':
                raise ValueError(f'Incomplete Git response for {path}')
            # Fixed ZIP metadata and stored bytes give the same digest across hosts.
            info = zipfile.ZipInfo(path, date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, content)
    return commit, output.getvalue()


def verify(path, expected_digest, files=FILES):
    with path.open('rb') as stream:
        actual = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            actual.update(block)
        if actual.hexdigest() != expected_digest:
            raise ValueError('SHA-256 differs from the selected revision')
        with zipfile.ZipFile(stream) as archive:
            if tuple(archive.namelist()) != files or archive.testzip() is not None:
                raise ValueError('ZIP membership or CRC mismatch')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    parser.add_argument('--revision', default='HEAD')
    parser.add_argument('--verify', action='store_true', help='Check an existing ZIP')
    args = parser.parse_args()
    try:
        commit, payload = build(args.revision)
        with zipfile.ZipFile(io.BytesIO(payload)) as expected:
            files = tuple(expected.namelist())
        digest = hashlib.sha256(payload).hexdigest()
        if not args.verify:
            args.archive.parent.mkdir(parents=True, exist_ok=True)
            with NamedTemporaryFile(dir=args.archive.parent, delete=False) as stage:
                staged_path = Path(stage.name)
            try:
                staged_path.write_bytes(payload)
                verify(staged_path, digest, files)
                # Atomic publication on NTFS. An existing destination is preserved.
                os.link(staged_path, args.archive)
            finally:
                staged_path.unlink(missing_ok=True)
        verify(args.archive, digest, files)
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError, zipfile.BadZipFile) as exc:
        parser.exit(2, f'{exc}\n')
    print(f'{commit}  {digest}  {args.archive}')


if __name__ == '__main__':
    main()
