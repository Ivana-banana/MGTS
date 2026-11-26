"""Replay the official inspection in native ROS 2 and capture the actual published poses."""
import argparse
import csv
from pathlib import Path
import shutil
import subprocess
from time import monotonic

import rclpy
import rosbag2_py
from geometry_msgs.msg import PoseStamped
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
from rclpy.serialization import serialize_message
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import PointCloud2

from mgts_nav.ros_localize3d import Localizer


def ns(stamp):
    return stamp.sec * 10**9 + stamp.nanosec


class Capture(Node):
    def __init__(self, output, arrivals):
        super().__init__("mgts_prefix_capture")
        self.clock_ns = 0
        self.arrivals = csv.writer(arrivals)
        self.arrivals.writerow(["header_ns", "last_replay_clock_ns"])
        self.bag = rosbag2_py.SequentialWriter()
        self.bag.open(rosbag2_py.StorageOptions(uri=str(output), storage_id="mcap"),
                      rosbag2_py.ConverterOptions("", ""))
        self.create_subscription(Clock, "/clock", self.clock_callback, qos_profile_sensor_data)
        for topic, cls, type_name, qos in (
            ("/mgts/map3d", PointCloud2, "sensor_msgs/msg/PointCloud2",
             QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)),
            ("/mgts/pose3d", PoseStamped, "geometry_msgs/msg/PoseStamped", QoSProfile(depth=100)),
        ):
            self.bag.create_topic(rosbag2_py.TopicMetadata(
                name=topic, type=type_name, serialization_format="cdr"))
            self.create_subscription(cls, topic, self.on_map if cls is PointCloud2 else self.on_pose, qos)
        self.maps = []
        self.poses = 0
        self.last_stamp = None

    def clock_callback(self, msg):
        self.clock_ns = ns(msg.clock)

    def on_map(self, msg):
        raw = serialize_message(msg)
        if self.last_stamp is None:
            self.maps.append(raw)
        else:
            self.bag.write("/mgts/map3d", raw, self.last_stamp)

    def on_pose(self, msg):
        self.last_stamp = ns(msg.header.stamp)
        for raw in self.maps:
            self.bag.write("/mgts/map3d", raw, self.last_stamp)
        self.maps.clear()
        self.bag.write("/mgts/pose3d", serialize_message(msg), self.last_stamp)
        self.arrivals.writerow([self.last_stamp, self.clock_ns])
        self.poses += 1


def run(inspect_bag, map_file, output):
    output.mkdir(parents=True)
    rclpy.init()
    with (output / "timings.csv").open("x", newline="") as timings, (
            output / "arrivals.csv").open("x", newline="") as arrivals, (
            output / "player.log").open("x") as log:
        node = Localizer(map_file, timings)
        capture = Capture(output / "result", arrivals)
        executor = SingleThreadedExecutor()
        executor.add_node(node)
        executor.add_node(capture)
        command = [shutil.which("ros2"), "bag", "play", str(inspect_bag), "--rate", "1",
                   "--delay", "2", "--clock", "100", "--disable-keyboard-controls",
                   "--topics", "/utlidar/cloud_base", "/utlidar/robot_odom"]
        with subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT) as player:
            try:
                while player.poll() is None:
                    executor.spin_once(timeout_sec=.05)
                until = monotonic() + 1
                while monotonic() < until:
                    executor.spin_once(timeout_sec=.05)
                if player.returncode or not capture.poses:
                    raise RuntimeError(f"No successful replay; see {output}/player.log")
            finally:
                if player.poll() is None:
                    player.terminate()
                for scan in node.pending:
                    node.log_scan(scan, "unprocessed")
                executor.shutdown()
                del capture.bag
                capture.destroy_node()
                node.destroy_node()
                rclpy.shutdown()
    print(f"{output}: {capture.poses} published poses")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inspect_bag", type=Path)
    parser.add_argument("map_file", type=Path)
    parser.add_argument("output", type=Path)
    run(**vars(parser.parse_args()))
