import os
from launch import LaunchDescription
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.actions import DeclareLaunchArgument, GroupAction
from launch_ros.actions import PushRosNamespace
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
            'robot_ip': LaunchConfiguration('robot_ip_1'),
            'namespace': 'arm_1',
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
            'robot_ip': LaunchConfiguration('robot_ip_2'),
            'namespace': 'arm_2',
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
        robot_ip_1, 
        robot_ip_2,  
        kinova_sim_bringup_1,
        #kinova_sim_bringup_2,
        # image_publisher_node,
        # nn_controller_node
    ])