"""Copy a ROS 2 recording prefix by delivery time, preserving all topics and CDR bytes."""
import argparse
from collections import Counter
import json
from pathlib import Path
import tempfile

from rosbags.rosbag2 import StoragePlugin, Writer

from mgts_nav.bag import reader


def copy_prefix(source, target, duration_ns):
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    with reader(source) as bag, tempfile.TemporaryDirectory(dir=target.parent) as temp:
        if not 0 < duration_ns < bag.duration:
            raise ValueError("Duration must select a nonempty proper prefix of the bag")
        cutoff = bag.start_time + duration_ns
        staged = Path(temp) / "prefix"
        with Writer(staged, version=8, storage_plugin=StoragePlugin.MCAP) as writer:
            connections = {
                c.id: writer.add_connection(
                    c.topic, c.msgtype, msgdef=c.msgdef.data or None,
                    rihs01=c.digest or bag.typestore.hash_rihs01(c.msgtype),
                    typestore=bag.typestore, serialization_format=c.ext.serialization_format,
                    offered_qos_profiles=c.ext.offered_qos_profiles)
                for c in bag.connections
            }
            for c, t, raw in bag.messages(stop=cutoff):
                writer.write(connections[c.id], t, raw)
                counts[c.topic] += 1
        # The bag metadata names the volume after the writer's directory.
        staged.rename(target)
    return dict(stop_exclusive_ns=cutoff, counts=counts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)
    parser.add_argument("--duration-ns", type=int, required=True)
    print(json.dumps(copy_prefix(**vars(parser.parse_args()))))
