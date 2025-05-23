import os
from launch import LaunchDescription
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():

    robot_ip_1 = DeclareLaunchArgument(
        'robot_ip_1',
        default_value='192.168.1.10',
        description='Robot IP address'
    )

    robot_ip_2 = DeclareLaunchArgument(
        'robot_ip_2',
        default_value='192.168.1.12',
        description='Robot IP address'
    )

    kinova_sim_bringup_1 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(
                get_package_share_directory('kortex_bringup'),
                'launch',
                'gen3.launch.py'
            )
        ]),
        launch_arguments={
            'robot_ip': LaunchConfiguration('robot_ip_1')
        }.items()
    )

    kinova_sim_bringup_2 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(
                get_package_share_directory('kortex_bringup'),
                'launch',
                'gen3.launch.py'
            )
        ]),
        launch_arguments={
            'robot_ip': LaunchConfiguration('robot_ip_2')
        }.items()
    )

    depth_publisher_node = Node(
            package='tactile_sensing',
            executable='show3d_ros2',
            name='pcd_depth_publisher',
            emulate_tty=True
        )
    
    mpc_controller_node = Node(
            package='tactile_sensing',
            executable='model_based_mpc',
            name='mpc_controller',
            emulate_tty=True
        )
    
    return LaunchDescription([
        robot_ip_1,
        robot_ip_2,  
        kinova_sim_bringup_1,
        kinova_sim_bringup_2,
        depth_publisher_node,
        mpc_controller_node
    ])