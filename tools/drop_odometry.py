"""Copy a delivery-time window to MCAP, optionally dropping only odometry.

All intervals are absolute integer nanoseconds, start inclusive, stop exclusive.
CDR payloads (including headers), delivery timestamps and reader order are retained.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import tempfile

from rosbags.rosbag2 import StoragePlugin, Writer

from mgts_nav.bag import reader


def copy_window(source, target, start_ns, stop_ns, drop_start_ns=None,
                drop_stop_ns=None, odom_topic="/utlidar/robot_odom"):
    source, target = Path(source), Path(target)
    if target.exists():
        raise FileExistsError(target)
    if start_ns >= stop_ns:
        raise ValueError("Window must be nonempty")
    dropping = drop_start_ns is not None or drop_stop_ns is not None
    if dropping and (drop_start_ns is None or drop_stop_ns is None
                     or not start_ns <= drop_start_ns < drop_stop_ns <= stop_ns):
        raise ValueError("Supply both drop bounds inside the window")
    counts, removed = Counter(), Counter()
    target.parent.mkdir(parents=True, exist_ok=True)
    with reader(source) as bag, tempfile.TemporaryDirectory(dir=target.parent) as temp:
        if start_ns < bag.start_time or stop_ns > bag.end_time:
            raise ValueError("Window must be within the recording")
        if dropping and not any(c.topic == odom_topic for c in bag.connections):
            raise ValueError(f"No odometry topic {odom_topic}")
        staged = Path(temp) / target.name
        with Writer(staged, version=8, storage_plugin=StoragePlugin.MCAP) as writer:
            connections = {
                c.id: writer.add_connection(
                    c.topic, c.msgtype, msgdef=c.msgdef.data or None,
                    rihs01=c.digest or bag.typestore.hash_rihs01(c.msgtype),
                    typestore=bag.typestore, serialization_format=c.ext.serialization_format,
                    offered_qos_profiles=c.ext.offered_qos_profiles)
                for c in bag.connections
            }
            for c, t, raw in bag.messages(start=start_ns, stop=stop_ns):
                if dropping and c.topic == odom_topic and drop_start_ns <= t < drop_stop_ns:
                    removed[c.topic] += 1
                    continue
                writer.write(connections[c.id], t, raw)
                counts[c.topic] += 1
        staged.rename(target)
    return dict(start_ns=start_ns, stop_exclusive_ns=stop_ns,
                drop_start_ns=drop_start_ns, drop_stop_exclusive_ns=drop_stop_ns,
                counts=dict(counts), removed=dict(removed))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)
    parser.add_argument("--start-ns", type=int, required=True)
    parser.add_argument("--stop-ns", type=int, required=True)
    parser.add_argument("--drop-start-ns", type=int)
    parser.add_argument("--drop-stop-ns", type=int)
    parser.add_argument("--odom-topic", default="/utlidar/robot_odom")
    print(json.dumps(copy_window(**vars(parser.parse_args()))))
