# Copyright (c) 2021 PickNik, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Denis Stogl

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, ThisLaunchFileDir
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
    # Declare arguments
    declared_arguments = []
    declared_arguments.append(
        DeclareLaunchArgument(
            "robot_type",
            default_value="gen3",
            description="Type/series of robot.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "robot_ip_1",
            default_value="192.168.1.10",
            description="IP address by which the robot can be reached.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "robot_ip_2",
            default_value="192.168.1.12",
            description="IP address by which the robot can be reached.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument("dof", default_value="7", description="DoF of robot.")
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "use_fake_hardware",
            default_value="false",
            description="Start robot with fake hardware mirroring command to its states.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "fake_sensor_commands",
            default_value="false",
            description="Enable fake command interfaces for sensors used for simple simulations. \
            Used only if 'use_fake_hardware' parameter is true.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "robot_controller",
            default_value="joint_trajectory_controller",
            description="Robot controller to start.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "controllers_file",
            default_value="ros2_controllers.yaml",
            description="Robot controller to start.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "gripper",
            default_value="",
            description="Name of the gripper attached to the arm",
            choices=["", "robotiq_2f_85", "robotiq_2f_140"],
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "gripper_joint_name",
            default_value="robotiq_85_left_knuckle_joint",
            description="Name of the gripper attached to the arm",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "use_internal_bus_gripper_comm",
            default_value="true",
            description="Use internal bus for gripper communication?",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "gripper_max_velocity",
            default_value="100.0",
            description="Max velocity for gripper commands",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "gripper_max_force",
            default_value="100.0",
            description="Max force for gripper commands",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument("launch_rviz", default_value="true", description="Launch RViz?")
    )
    declared_arguments.append(
        DeclareLaunchArgument("prefix_1", default_value="arm_1_", description="prefix to differentiate arms")
    )
    declared_arguments.append(
        DeclareLaunchArgument("prefix_2", default_value="arm_2_", description="prefix to differentiate arms")
    )

    # Initialize Arguments
    robot_type = LaunchConfiguration("robot_type")
    robot_ip_1 = LaunchConfiguration("robot_ip_1")
    robot_ip_2 = LaunchConfiguration("robot_ip_2")
    dof = LaunchConfiguration("dof")
    use_fake_hardware = LaunchConfiguration("use_fake_hardware")
    fake_sensor_commands = LaunchConfiguration("fake_sensor_commands")
    robot_controller = LaunchConfiguration("robot_controller")
    gripper = LaunchConfiguration("gripper")
    use_internal_bus_gripper_comm = LaunchConfiguration("use_internal_bus_gripper_comm")
    gripper_max_velocity = LaunchConfiguration("gripper_max_velocity")
    gripper_max_force = LaunchConfiguration("gripper_max_force")
    gripper_joint_name = LaunchConfiguration("gripper_joint_name")
    launch_rviz = LaunchConfiguration("launch_rviz")
    controllers_file = LaunchConfiguration("controllers_file")
    prefix_1 = LaunchConfiguration("prefix_1")
    prefix_2 = LaunchConfiguration("prefix_2")

    arm_control_1 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(
                get_package_share_directory('kortex_bringup'),
                'launch',
                'kortex_control.launch.py'
            )
        ]),
        launch_arguments={
            "robot_type": robot_type,
            "robot_ip": robot_ip_1,
            "dof": dof,
            "use_fake_hardware": use_fake_hardware,
            "fake_sensor_commands": fake_sensor_commands,
            "robot_controller": robot_controller,
            "gripper": gripper,
            "use_internal_bus_gripper_comm": use_internal_bus_gripper_comm,
            "gripper_max_velocity": gripper_max_velocity,
            "gripper_max_force": gripper_max_force,
            "gripper_joint_name": gripper_joint_name,
            "launch_rviz": launch_rviz,
            "controllers_file": controllers_file,
            "description_file": "gen3.xacro",
            "prefix": prefix_1,
        }.items(),
    )

    arm_control_2 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(
                get_package_share_directory('kortex_bringup'),
                'launch',
                'kortex_control.launch.py'
            )
        ]),
        launch_arguments={
            "robot_type": robot_type,
            "robot_ip": robot_ip_2,
            "dof": dof,
            "use_fake_hardware": use_fake_hardware,
            "fake_sensor_commands": fake_sensor_commands,
            "robot_controller": robot_controller,
            "gripper": gripper,
            "use_internal_bus_gripper_comm": use_internal_bus_gripper_comm,
            "gripper_max_velocity": gripper_max_velocity,
            "gripper_max_force": gripper_max_force,
            "gripper_joint_name": gripper_joint_name,
            "launch_rviz": launch_rviz,
            "controllers_file": controllers_file,
            "description_file": "gen3.xacro",
            "prefix": prefix_2,
        }.items(),
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
    
    return LaunchDescription(
    declared_arguments +  
    [                    
        arm_control_1,
        arm_control_2,
        # image_publisher_node,
        # nn_controller_node
    ]
)