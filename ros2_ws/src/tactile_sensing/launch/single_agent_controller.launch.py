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
from launch.substitutions import PathJoinSubstitution, LaunchConfiguration
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    
    two_arm_bringup = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
                PathJoinSubstitution([
                    FindPackageShare('kortex_bringup'),
                    'launch',
                    'two_arms.launch.py'
                ])
            ])
    )

    image_publisher_node = Node(
            package='tactile_sensing',
            executable='show_tac_image',
            name='show_tactile_image',
            emulate_tty=True
        )
    
    nn_controller_node = Node(
            package='tactile_sensing',
            executable='single_agent_nn',
            name='single_agent_nn_controller',
            emulate_tty=True
        )
    
    return LaunchDescription([
        two_arm_bringup,
        image_publisher_node,
        nn_controller_node
    ])