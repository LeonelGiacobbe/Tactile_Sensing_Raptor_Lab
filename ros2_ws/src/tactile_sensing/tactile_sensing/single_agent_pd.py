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

class modelBasedPDNode(Node):
    def __init__(self):
        super().__init__('model_based_pd_node')
        
        self.gripper_posi_ = 0.0
        self.gripper_ini_flag_ = False
        self.contact_area_ini_flag = False
        self.dis_sum_ = 0
        self.contact_area_ = 0
        self.processing_executor = ThreadPoolExecutor(max_workers=1)
        self.contact_area_lock = threading.Lock()

        # Replace publisher with ActionClient
        self._action_client = ActionClient(self, GripperCommand, '/robotiq_gripper_controller/gripper_cmd')
        self.contact_group = ReentrantCallbackGroup()
        gs_qos_profile = QoSProfile(
            depth=5,  # Last 5 messages kept
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST
        )

        # Receives white pixel count from tactile sensor
        self.contact_area_sub = self.create_subscription(
            UInt16, 
            '/gs_contact_area', 
            self.contact_area_cb, gs_qos_profile, 
            callback_group=self.contact_group
        )
        
        # Subscribe to the JointState topic to get gripper position
        posi_qos_profile = QoSProfile(
            depth=1000,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE  # Change to TRANSIENT_LOCAL
        )
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_cb,
            posi_qos_profile,
        )

        if sys.stdin.isatty():
            self.old_attr = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
        else:
            self.old_attr = None

        self.frequency = 60
        self.init_posi_mm = 70
        self.q_d = 2
        self.c_ref = 1000
        self.k_p = 1/40000
        self.k_d = 1/6000

        self.del_t = 1/self.frequency
        self.gripper_cmd = GripperCommand()

        # Timer to call the run method periodically
        self.timer = self.create_timer(1.0 / self.frequency, self.run)
# Receives position of gripper: 0.0 -> completely open. 0.8 -> completely closed
    def joint_state_cb(self, msg: JointState): 
        # Flag to allow run method to go out of inf loop
        self.gripper_ini_flag_ = True
        
        self.get_logger().info(f"Received JointState message with joints: {msg.name}")
        if 'robotiq_85_left_knuckle_joint' in msg.name:
            index = msg.name.index('robotiq_85_left_knuckle_joint')
            gripper_position = msg.position[index]
            self.gripper_posi_ = gripper_position
            self.get_logger().info(f"Current gripper position: {gripper_position:.4f}")
        else:
            self.get_logger().warn("Gripper joint not found in JointState message")

    # Not currently used (original author said it can be set to zero?) but here just in case
    def dis_sum_cb(self, msg):
        self.dis_sum_ = msg.data
        
    # Offloaded calculation to show3d publisher
    def contact_area_cb(self, msg):
        self.contact_area_ = msg.data
        self.get_logger().info(f"Received contact area msg with value {msg.data}")
        self.contact_area_ini_flag = True

    def run(self):
        try:
            while not self.gripper_ini_flag_:
                print("waiting for initializing the gripper")
        
            last_contact_area_ = 0
            while rclpy.ok():
                new_gripper_posi = gripper_posi_to_mm(self.gripper_posi_) + (self.contact_area_ - (self.c_ref+(self.q_d*self.dis_sum_)))*self.k_p + (self.contact_area_ - last_contact_area_)*self.k_d
                self.gripper_cmd.command.position = mm_to_gripper_posi(new_gripper_posi)
                last_contact_area_ = self.contact_area_
                self._send_goal(self.gripper_cmd)
                self.rate.sleep()
            print("RCLPY not OK. Exiting... ")
        except KeyboardInterrupt:
            print("Interrupted by keyboard")

    def _send_goal(self, goal):
        self._action_client.wait_for_server()
        self._send_goal_future = self._action_client.send_goal_async(goal)
        self._send_goal_future.add_done_callback(self._goal_response_callback)

    def _goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected :(')
            return
        self.get_logger().info('Goal accepted :)')
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self._get_result_callback)

    def _get_result_callback(self, future):
        result = future.result().result
        self.get_logger().info(f'Result: {result}')

def main(args=None):
    rclpy.init(args=args)
    node = modelBasedPDNode()
    executor = MultiThreadedExecutor(num_threads=10)
    executor.add_node(node)
    executor.spin()

if __name__ == '__main__':
    main()