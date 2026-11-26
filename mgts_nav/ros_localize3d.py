"""Publish the fixed 3D map and localize on a ROS 2 Humble bag replay."""

import argparse
import csv
from collections import deque
from pathlib import Path
from time import perf_counter

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
from rclpy.utilities import remove_ros_args
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py.point_cloud2 import create_cloud_xyz32, read_points
from std_msgs.msg import Header
from tf2_ros import TransformBroadcaster

from .localize3d import (CorrectionFilter, Motion, SurfaceMap, alignment_argument,
                         alignment_pose, odometry_sample, scan_points)


def stamp(message):
    time = message.header.stamp
    return time.sec + time.nanosec * 1e-9


class Localizer(Node):
    def __init__(self, map_file, timings_file, map_from_odom=None):
        super().__init__('mgts_localize3d')
        self.alignment = alignment_pose(map_from_odom)
        with np.load(map_file) as data:
            xyz = data['xyz']
        self.surface = SurfaceMap(xyz)
        self.filter = CorrectionFilter()
        self.map = self.create_publisher(
            PointCloud2, '/mgts/map3d',
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.map_message = create_cloud_xyz32(Header(frame_id='map'), xyz.astype(np.float32))
        self.map.publish(self.map_message)
        self.pose = self.create_publisher(PoseStamped, '/mgts/pose3d', 10)
        self.tf = TransformBroadcaster(self)
        self.odometry = deque(maxlen=120)
        self.pending = deque()
        self.timings = timings_file
        self.csv = csv.writer(timings_file)
        self.csv.writerow(['header_s', 'status', 'latency_ms', 'pending'])
        self.create_subscription(Odometry, '/utlidar/robot_odom', self.on_odom, 100)
        self.create_subscription(PointCloud2, '/utlidar/cloud_base', self.on_cloud,
                                 qos_profile_sensor_data)

    def on_odom(self, message):
        self.odometry.append((stamp(message), *odometry_sample(message)))
        if self.pending:
            self.process()

    def on_cloud(self, cloud):
        started = perf_counter()
        values = read_points(cloud, field_names=['x', 'y', 'z', 'time'])
        xyz, offsets = scan_points(np.column_stack((values['x'], values['y'], values['z'])),
                                   values['time'])
        self.pending.append((stamp(cloud), xyz, offsets, started, cloud.header.stamp))
        if len(self.pending) > 8:
            self.log_scan(self.pending.popleft(), 'queue_full')
        self.process()

    def process(self):
        while self.pending and len(self.odometry) >= 2:
            scan = self.pending[0]
            t, xyz, offsets, _, header_stamp = scan
            if not len(xyz):
                self.log_scan(self.pending.popleft(), 'empty_scan')
                continue
            end = t + max(0., float(offsets.max()))
            if self.odometry[-1][0] < end:
                return
            self.pending.popleft()
            motion = Motion(*zip(*self.odometry))
            if not motion.covers(t + min(0., float(offsets.min())), end):
                self.log_scan(scan, 'missing_odometry')
                continue
            points = motion.deskew(t, xyz, offsets)
            selected = np.arange(len(points)) % 4 != 0
            prior = motion.at(t)
            if self.alignment is not None:
                prior = self.alignment.compose(prior)
            fit, accepted = self.surface.align(points[selected], prior)
            self.publish_pose(self.filter.update(t, prior, fit, accepted), header_stamp)
            self.log_scan(scan, 'matched' if accepted else 'prediction_only')

    def publish_pose(self, fit, header_stamp):
        message = PoseStamped()
        message.header.frame_id = 'map'
        message.header.stamp = header_stamp
        if self.map_message is not None:
            self.map.publish(self.map_message)
            self.map_message = None
        p, q = message.pose.position, message.pose.orientation
        p.x, p.y, p.z = map(float, fit.xyz)
        q.x, q.y, q.z, q.w = map(float, fit.rotation.as_quat())
        self.pose.publish(message)
        transform = TransformStamped(header=message.header, child_frame_id='base_link')
        transform.transform.translation.x = p.x
        transform.transform.translation.y = p.y
        transform.transform.translation.z = p.z
        transform.transform.rotation = q
        self.tf.sendTransform(transform)

    def log_scan(self, scan, status):
        self.csv.writerow([scan[0], status, (perf_counter() - scan[3]) * 1000, len(self.pending)])
        self.timings.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('map', type=Path)
    parser.add_argument('--timings', type=Path, required=True)
    alignment_argument(parser)
    args = parser.parse_args(remove_ros_args()[1:])
    rclpy.init()
    with args.timings.open('x', newline='', encoding='utf-8') as timings:
        node = Localizer(args.map, timings, args.map_from_odom)
        try:
            rclpy.spin(node)
        except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
            pass
        finally:
            for scan in node.pending:
                node.log_scan(scan, 'unprocessed')
            node.destroy_node()
            rclpy.try_shutdown()


if __name__ == '__main__':
    main()
