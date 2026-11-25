"""Restore native HEAP-1 signed per-point times to a verified XYZ bag."""
import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from rosbags.rosbag2 import Writer
from rosbags.typesys import Stores, get_typestore

from mgts_nav.bag import reader
from tools.heap1_point_times import load_arrays


def export(archive, data, recording, output):
    first, stop = (150, 450) if recording == "mapping" else (2560, 2620)
    arrays = load_arrays(archive, first, stop)
    store = get_typestore(Stores.ROS2_HUMBLE)
    field = store.types["sensor_msgs/msg/PointField"]
    dtype = np.dtype([("xyz", "<f4", (3,)), ("time", "<f8")])
    count = 0
    with reader(data / recording) as bag, Writer(output, version=8) as writer:
        outputs = {c.id: writer.add_connection(c.topic, c.msgtype, typestore=store) for c in bag.connections}
        for c, t, raw in bag.messages():
            if c.topic == "/velodyne_points":
                msg = bag.deserialize(raw, c.msgtype)
                n = int(arrays["valid"][count, 0])
                xyz = arrays["points"][count, :n]
                if (msg.width != n or msg.data.tobytes() != xyz.tobytes() or
                        t != round(arrays["timestamp"][count]*1e9) or
                        t != msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec):
                    raise ValueError(f"Unpaired cloud at frame {first+count}")
                pts = np.empty(n, dtype)
                pts["xyz"], pts["time"] = xyz, arrays["time"][count, :n]
                msg.fields.append(field("time", 12, 8, 1))
                msg.point_step, msg.row_step, msg.data = dtype.itemsize, pts.nbytes, pts.view(np.uint8)
                raw = store.serialize_cdr(msg, c.msgtype)
                count += 1
            writer.write(outputs[c.id], t, raw)
    if count != stop-first:
        raise ValueError("Unexpected HEAP-1 scan count")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("data", type=Path)
    parser.add_argument("recording", choices=("mapping", "inspection"))
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists")
    with TemporaryDirectory(dir=args.output.parent) as tmp:
        stage = Path(tmp) / args.output.name
        export(args.archive, args.data, args.recording, stage)
        stage.rename(args.output)
