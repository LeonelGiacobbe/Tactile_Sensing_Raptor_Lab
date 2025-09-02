import os
from launch import LaunchDescription
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():

    robot_ip = DeclareLaunchArgument(
        'robot_ip',
        default_value='192.168.1.10',
        description='Robot IP address'
    )
    gripper_type = DeclareLaunchArgument(
        'gripper_type',
        default_value='robotiq_2f_85',
        description='Robot gripper type'
    )

    kinova_sim_bringup = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(
                get_package_share_directory('kortex_bringup'),
                'launch',
                'gen3.launch.py'
            )
        ]),
        launch_arguments={
            'robot_ip': LaunchConfiguration('robot_ip'),
            'gripper': LaunchConfiguration('gripper_type')
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
            executable='model_based_pd',
            name='pd_controller',
            emulate_tty=True
        )
    
    return LaunchDescription([
        robot_ip,   
        gripper_type,
        kinova_sim_bringup,
        depth_publisher_node,
        mpc_controller_node
    ])