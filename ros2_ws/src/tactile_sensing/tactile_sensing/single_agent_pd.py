import numpy as np
import cv2
from scipy import sparse
import time
import sys, tty, termios
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, UInt16
from rclpy.action import ActionClient
from control_msgs.action import GripperCommand
from sensor_msgs.msg import JointState, Image
import osqp
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.wait_for_message import wait_for_message
from concurrent.futures import ThreadPoolExecutor
import threading

CONVERSION_RATE = 0.005714

def gripper_posi_to_mm(gripper_posi):
    opening = 0.8 - gripper_posi
    return opening / CONVERSION_RATE

def mm_to_gripper_posi(millimeters):
    opening = 140 - millimeters
    return opening * CONVERSION_RATE

class ModelBasedPDNode(Node):
    def __init__(self):
        super().__init__('model_based_pd_node')

        self.gripper_posi_ = 0.0
        self.gripper_ini_flag_ = False
        self.contact_area_ini_flag = False
        self.dis_sum_ = 0
        self.contact_area_ = 0
        self.last_contact_area_ = 0

        # Action client for gripper control
        self._action_client = ActionClient(self, GripperCommand, '/robotiq_gripper_controller/gripper_cmd')
        self._action_client.wait_for_server()  # Wait once here

        gs_qos_profile = QoSProfile(depth=5, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.VOLATILE)

        # Subscribe to contact area
        self.create_subscription(UInt16, '/gs_contact_area', self.contact_area_cb, gs_qos_profile)

        # Subscribe to gripper joint state
        posi_qos_profile = QoSProfile(depth=1000, reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE)
        self.create_subscription(JointState, '/joint_states', self.joint_state_cb, posi_qos_profile)

        self.frequency = 8  # Hz
        self.q_d = 2
        self.c_ref = 1000
        self.k_p = 1 / 40000
        self.k_d = 1 / 6000

        self.gripper_cmd = GripperCommand.Goal()

        # Timer for periodic control
        self.create_timer(1.0 / self.frequency, self.run)

    def joint_state_cb(self, msg: JointState):
        self.gripper_ini_flag_ = True
        if 'robotiq_85_left_knuckle_joint' in msg.name:
            index = msg.name.index('robotiq_85_left_knuckle_joint')
            self.gripper_posi_ = msg.position[index]

    def contact_area_cb(self, msg):
        self.contact_area_ = msg.data
        self.get_logger().info(f"Received contact area value: {self.contact_area_}")
        self.contact_area_ini_flag = True

    def run(self):
        if not self.gripper_ini_flag_:
            self.get_logger().info("Waiting for gripper initialization...")
            return

        # Compute new position
        new_gripper_posi = gripper_posi_to_mm(self.gripper_posi_) + \
                           (self.contact_area_ - (self.c_ref + (self.q_d * self.dis_sum_))) * self.k_p + \
                           (self.contact_area_ - self.last_contact_area_) * self.k_d

        # Convert and clamp position
        pos = max(0.0, min(0.8, mm_to_gripper_posi(new_gripper_posi)))
        self.gripper_cmd.command.position = pos
        self._send_goal(self.gripper_cmd)

        self.last_contact_area_ = self.contact_area_
        self.get_logger().info(f"Target gripper posi: {pos:.4f}")

    def _send_goal(self, goal):
        send_goal_future = self._action_client.send_goal_async(goal)
        send_goal_future.add_done_callback(self._goal_response_callback)

    def _goal_response_callback(self, future):
        goal_handle = future.result()
        if goal_handle.accepted:
            self.get_logger().info('Goal accepted :)')
            goal_handle.get_result_async().add_done_callback(self._get_result_callback)
        else:
            self.get_logger().info('Goal rejected :(')

    def _get_result_callback(self, future):
        result = future.result().result
        self.get_logger().info(f'Result: {result}')

def main(args=None):
    rclpy.init(args=args)
    node = ModelBasedPDNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
