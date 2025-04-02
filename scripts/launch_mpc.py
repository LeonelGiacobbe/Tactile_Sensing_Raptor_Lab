import os
from launch import LaunchDescription
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource

def generate_launch_description():
    kinova_sim_bringup = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([os.path.join(
            get_package_share_directory('kortex_bringup'), 'launch'),
            '/gen3.launch.py'])
        )
    
    return LaunchDescription([
        kinova_sim_bringup,
        Node(
            package='gsrobotics',
            namespace='tactile_sensing',
            executable='show3d_ros2',
            name='pcd_depth_publisher'
        ),
    ])