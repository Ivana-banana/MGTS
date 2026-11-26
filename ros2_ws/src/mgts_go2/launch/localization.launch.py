"""Localization on a prebuilt map, optionally with bag playback, RViz and recording of the results.

ros2 launch mgts_go2 localization.launch.py map_dir:=/ros2_ws/mgts_map \
    bag:=/ros2_ws/bag_for_localization/bag_for_localization view:=top record:=/ros2_ws/results/run1
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, EmitEvent, ExecuteProcess, RegisterEventHandler,
                            TimerAction)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node

RESULT_TOPICS = ['/mgts/pose', '/mgts/odometry', '/mgts/path', '/mgts/odom_path', '/mgts/map',
                 '/mgts/scan_map', '/mgts/dynamic_points', '/mgts/status', '/mgts/timing', '/tf', '/tf_static']


def generate_launch_description():
    share = get_package_share_directory('mgts_go2')
    cfg = LaunchConfiguration
    args = [
        DeclareLaunchArgument('map_dir', default_value='/ros2_ws/mgts_map'),
        DeclareLaunchArgument('bag', default_value='', description='bag to play; empty = use live topics'),
        DeclareLaunchArgument('rate', default_value='1.0'),
        DeclareLaunchArgument('start_delay', default_value='4.0', description='s before playback (map loading)'),
        DeclareLaunchArgument('rviz', default_value='true'),
        DeclareLaunchArgument('view', default_value='top', description='top | third'),
        DeclareLaunchArgument('record', default_value='', description='output bag directory; empty = no recording'),
        DeclareLaunchArgument('init_mode', default_value='odom'),
        DeclareLaunchArgument('timing_csv', default_value=''),
        DeclareLaunchArgument('params', default_value=os.path.join(share, 'config', 'localizer.yaml')),
    ]
    playing = PythonExpression(["'", cfg('bag'), "' != ''"])
    use_sim = PythonExpression(["'", cfg('bag'), "' != ''"])

    node = Node(
        package='mgts_go2', executable='localizer_node', name='mgts_localizer', output='screen',
        parameters=[cfg('params'), {'map_dir': cfg('map_dir'), 'init_mode': cfg('init_mode'),
                                    'timing_csv': cfg('timing_csv'), 'use_sim_time': use_sim}])
    play = ExecuteProcess(
        cmd=['ros2', 'bag', 'play', cfg('bag'), '--clock', '100', '--rate', cfg('rate'), '--delay', '2.0',
             '--topics', '/utlidar/cloud_base', '/utlidar/robot_odom', '/utlidar/imu'],
        output='screen', condition=IfCondition(playing))
    rviz = Node(
        package='rviz2', executable='rviz2', name='rviz2', output='log',
        arguments=['-d', PythonExpression(["'", os.path.join(share, 'rviz', 'mgts_'), "' + '", cfg('view'),
                                           "' + '.rviz'"])],
        parameters=[{'use_sim_time': use_sim}], condition=IfCondition(cfg('rviz')))
    record = ExecuteProcess(
        cmd=['ros2', 'bag', 'record', '-o', cfg('record'), '--storage', 'mcap', *RESULT_TOPICS],
        output='screen', condition=IfCondition(PythonExpression(["'", cfg('record'), "' != ''"])))
    stop_after_bag = RegisterEventHandler(OnProcessExit(
        target_action=play, on_exit=[TimerAction(period=2.0, actions=[EmitEvent(event=Shutdown(reason='bag finished'))])]))
    return LaunchDescription(args + [node, rviz, record,
                                     TimerAction(period=cfg('start_delay'), actions=[play]), stop_after_bag])
