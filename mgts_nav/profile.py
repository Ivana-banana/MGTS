"""Draft a sensor profile from recorded topics, sample headers and static TF."""
import argparse
import json
from itertools import islice
from pathlib import Path

from rosbags.serde.errors import SerdeError
from rosbags.typesys.base import TypesysError

from .bag import reader
from .calibration import extrinsics, read_edges


TYPES = {'lidar': 'sensor_msgs/msg/PointCloud2', 'imu': 'sensor_msgs/msg/Imu',
         'odom': 'nav_msgs/msg/Odometry'}


def draft_profile(path, *, lidar=None, imu=None, odom=None, base=None, tf_topic='/tf_static',
                  verify_frames=False):
    selected = dict(lidar=lidar, imu=imu, odom=odom)
    config = dict(base_frame=base, base_from_lidar_xyz_m=None, base_from_lidar_xyzw=None,
                  base_from_imu_xyzw=None, stationary_s=None, max_sync_gap_s=None,
                  height_band_m=None)
    questions, candidates, observations = [], {}, {}
    with reader(path) as bag:
        inventory = sorted({(c.topic, c.msgtype) for c in bag.connections})
        for key, msgtype in TYPES.items():
            available = sorted({topic for topic, typ in inventory if typ == msgtype})
            candidates[key] = available
            topic = selected[key]
            if topic is not None and topic not in available:
                raise ValueError(f'No {msgtype} on {topic}; candidates: {available}')
            if topic is None and len(available) == 1:
                topic = available[0]
            config[key], config[key + '_frame'] = topic, None
            if topic is None:
                if available:
                    questions.append(f'Which {key} stream is mounted for navigation? Select --{key} from {available}.')
                else:
                    questions.append(f'Where is {key} ({msgtype})? Supply its schema or confirm its absence.')
                continue
            if any(name == topic and typ != msgtype for name, typ in inventory):
                raise ValueError(f'Conflicting message types on {topic}')
            frames, children = set(), set()
            checked = 0
            complete = True
            for conn in (c for c in bag.connections if c.topic == topic):
                messages = bag.messages(connections=[conn])
                for _, _, raw in messages if verify_frames else islice(messages, 1):
                    try:
                        msg = bag.deserialize(raw, conn.msgtype)
                    except (KeyError, ValueError, TypeError, TypesysError, SerdeError) as exc:
                        questions.append(f'Can the producer supply message definitions for {topic}? {exc}')
                        complete = False
                        break
                    checked += 1
                    frames.add(msg.header.frame_id)
                    if key == 'odom':
                        children.add(msg.child_frame_id)
            observations[topic] = dict(checked_messages=checked, header_frame_ids=sorted(frames))
            if complete and len(frames) == 1 and '' not in frames:
                config[key + '_frame'] = frames.pop()
            else:
                questions.append(f'Which stable header.frame_id is used on {topic}? Observed {sorted(frames)}.')
            if key == 'odom':
                observations[topic]['child_frame_ids'] = sorted(children)
                if complete and len(children) == 1 and '' not in children:
                    child = children.pop()
                    if base and base != child:
                        raise ValueError(f'--base {base} conflicts with odometry child_frame_id {child}')
                    config['base_frame'] = child
                else:
                    config['base_frame'] = None
                    questions.append(f'Which body frame is represented by child_frame_id on {topic}? '
                                     f'Observed {sorted(children)}.')

    base = config['base_frame']
    if not base:
        questions.append('Which TF frame represents the robot body? Supply --base.')
    try:
        edges = read_edges(path, tf_topic)
    except (ValueError, KeyError, TypesysError, SerdeError) as exc:
        edges = {}
        questions.append(f'Where is the verified static calibration? {exc}')
    for key in ('lidar', 'imu'):
        frame = config[key + '_frame']
        if not base or not frame:
            continue
        try:
            transform = extrinsics(edges, base, frame, frame)
        except ValueError as exc:
            questions.append(f'What is the measured transform {base} <- {frame}? {exc}')
            continue
        config['base_from_' + key + '_xyzw'] = transform['base_from_' + key + '_xyzw']
        if key == 'lidar':
            config['base_from_lidar_xyz_m'] = transform['base_from_lidar_xyz_m']
    questions.extend([
        'Was the robot stationary at the beginning, and for how long? Use stationary_s=0 to skip bias estimation.',
        'What time gaps are acceptable for interpolation, and what height band is informative in the body frame?',
        'Are clocks aligned? If point timing is supplied, what is its field, unit and signed reference to header.stamp?',
    ])
    return dict(status='draft', profile=config, candidates=candidates, observed_frames=observations,
                frame_check='all_messages' if verify_frames else 'first_per_connection',
                observed_topics=[dict(topic=t, type=m) for t, m in inventory], questions=questions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bag', type=Path)
    for role in TYPES:
        parser.add_argument('--' + role)
    parser.add_argument('--base')
    parser.add_argument('--tf-topic', default='/tf_static')
    parser.add_argument('--verify-frames', action='store_true',
                        help='Read all selected messages to verify header and odometry child frames')
    parser.add_argument('--out', type=Path, help='Create a new draft JSON')
    args = parser.parse_args()
    try:
        report = draft_profile(args.bag, lidar=args.lidar, imu=args.imu, odom=args.odom,
                               base=args.base, tf_topic=args.tf_topic, verify_frames=args.verify_frames)
        serialized = json.dumps(report, ensure_ascii=args.out is None, indent=2)
        if args.out:
            with args.out.open('x', encoding='utf-8') as file:
                file.write(serialized + '\n')
        else:
            print(serialized)
    except (OSError, ValueError) as exc:
        parser.exit(2, f'{exc}\n')


if __name__ == '__main__':
    main()
