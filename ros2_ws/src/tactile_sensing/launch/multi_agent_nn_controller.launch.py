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

    kinova_sim_bringup = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(
                get_package_share_directory('kortex_bringup'),
                'launch',
                'gen3.launch.py'
            )
        ]),
        launch_arguments={
            'robot_ip': LaunchConfiguration('robot_ip')
        }.items()
    )

    image_publisher_node = Node(
            package='tactile_sensing',
            executable='show_tac_image',
            name='show_tactile_image',
            emulate_tty=True
        )
    
    nn_controller_node = Node(
            package='tactile_sensing',
            executable='multi_agent_nn',
            name='multi_agent_nn_controller',
            emulate_tty=True
        )
    
    return LaunchDescription([
        robot_ip,   
        kinova_sim_bringup,
        image_publisher_node,
        nn_controller_node
    ])