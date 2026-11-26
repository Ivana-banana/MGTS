"""Observe scan and pose callback entry times in one independent ROS 2 process."""
import argparse
import csv
from pathlib import Path
import subprocess
import sys
from time import monotonic_ns

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2


def observe(inspection, map_file, output):
    output.mkdir(parents=True)
    records = []
    rclpy.init()
    node = Node('go2_latency_observer')

    def listener(topic):
        def receive(msg):
            received = monotonic_ns()
            stamp = msg.header.stamp
            records.append((topic, stamp.sec, stamp.nanosec, received))
        return receive

    node.create_subscription(PointCloud2, '/utlidar/cloud_base', listener('scan'),
                             QoSProfile(depth=100, reliability=ReliabilityPolicy.BEST_EFFORT))
    node.create_subscription(PoseStamped, '/mgts/pose3d', listener('pose'), QoSProfile(depth=100))
    try:
        if node.count_publishers('/utlidar/cloud_base') or node.count_publishers('/mgts/pose3d'):
            raise RuntimeError('Choose an unused ROS_DOMAIN_ID')
        cmd = [sys.executable, '-m', 'tools.go2_live_replay', str(inspection), str(map_file), str(output/'replay')]
        with subprocess.Popen(cmd) as child:
            try:
                while child.poll() is None:
                    rclpy.spin_once(node, timeout_sec=.1)
                if child.returncode:
                    raise RuntimeError(f'Replay exited with code {child.returncode}')
                end = monotonic_ns() + 2_000_000_000
                while monotonic_ns() < end:
                    rclpy.spin_once(node, timeout_sec=.1)
            finally:
                if child.poll() is None:
                    child.terminate()
    finally:
        node.destroy_node()
        rclpy.shutdown()
        with (output/'observations.csv').open('x', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(('topic', 'sec', 'nanosec', 'callback_monotonic_ns'))
            writer.writerows(records)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('inspection', 'map_file', 'output'):
        parser.add_argument(name, type=Path)
    observe(**vars(parser.parse_args()))
