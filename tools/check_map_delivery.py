"""Check map retention and moving poses on an active ROS 2 result graph."""
import argparse
import json
from time import monotonic

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=float, default=30)
    args = parser.parse_args()
    rclpy.init()
    node = rclpy.create_node("mgts_check_map_delivery")
    cloud_points = 0
    poses = {}

    def on_cloud(msg):
        nonlocal cloud_points
        if msg.header.frame_id == "map" and msg.data:
            cloud_points = msg.height * msg.width

    def on_pose(msg):
        if msg.header.frame_id == "map":
            t = msg.header.stamp
            p = msg.pose.position
            poses[(t.sec, t.nanosec)] = (p.x, p.y, p.z)

    topics = (("/mgts/map3d", PointCloud2, on_cloud, DurabilityPolicy.TRANSIENT_LOCAL, 1),
              ("/mgts/pose3d", PoseStamped, on_pose, DurabilityPolicy.VOLATILE, 10))
    try:
        for topic, kind, callback, durability, depth in topics:
            qos = QoSProfile(depth=depth, durability=durability, reliability=ReliabilityPolicy.RELIABLE)
            node.create_subscription(kind, topic, callback, qos)
        deadline = monotonic() + args.timeout
        moving = False
        while rclpy.ok() and monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
            moving = len(set(poses.values())) >= 2
            if moving and cloud_points:
                break
        publishers = {}
        subscribers = {}
        for topic, *_ in topics:
            publishers[topic] = [
                dict(node=e.node_name, reliability=e.qos_profile.reliability.name,
                     durability=e.qos_profile.durability.name, depth=e.qos_profile.depth)
                for e in node.get_publishers_info_by_topic(topic)]
            subscribers[topic] = [
                dict(node=e.node_name, reliability=e.qos_profile.reliability.name,
                     durability=e.qos_profile.durability.name, depth=e.qos_profile.depth)
                for e in node.get_subscriptions_info_by_topic(topic)]
        print(json.dumps(dict(map_points=cloud_points, distinct_stamps=len(poses),
                              moving=moving, publishers=publishers, subscribers=subscribers), ensure_ascii=False))
        return 0 if cloud_points and moving else 1
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())

