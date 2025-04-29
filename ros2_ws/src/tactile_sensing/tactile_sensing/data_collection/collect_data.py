import rclpy, time
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from functools import partial

from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
from kortex_api.autogen.messages import Base_pb2, BaseCyclic_pb2, Common_pb2

from sensor_msgs.msg import Image, JointState
from cv_bridge import CvBridge

import sys, os, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import utilities
from movement_functions import *


class KinovaDataCollector(Node):
    def __init__(self):
        super().__init__('kinova_data_collector')

        #class variables
        self.bridge = CvBridge()
        self.contact_group = ReentrantCallbackGroup()
        self.gripper_posi_1 = 0.0
        self.gripper_posi_2 = 0.0
        self.gripper_ini_flag_1 = False
        self.gripper_ini_flag_2 = False

        # Action server to operate gripper width
        self._action_client_1 = ActionClient(self, GripperCommand, '/robotiq_gripper_controller/gripper_cmd')
        self._action_client_2 = ActionClient(self, GripperCommand, '/robotiq_gripper_controller/gripper_cmd') # adjust to new namespace eventually

        # QoS profiles
        self.gs_qos_profile = QoSProfile(
            depth=5,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST
        )

        self.posi_qos_profile = QoSProfile(
            depth=1000,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE  # Change to TRANSIENT_LOCAL
        )

        # Gelsight subscribers (don't know if we'll need both. Talk to Dr. Sun)
        self.gs_sub_1 = self.create_subscription(
            Image,
            '/bot_1/gsmini_rawimg_0',
            self.capture_raw_image,
            self.gs_qos_profile,
            callback_group=self.contact_group
        )

        self.gs_sub_2 = self.create_subscription(
            Image,
            '/bot_2/gsmini_rawimg_0',
            self.capture_raw_image,
            self.gs_qos_profile,
            callback_group=self.contact_group
        )

        # Subscribers for gripper posi
        self.gripper_posi_sub_1 = self.create_subscription(
            JointState,
            '/bot_1/joint_states',
            partial(self.gripper_posi_cb, bot_name="bot_1"),
            posi_qos_profile,
        )

        self.gripper_posi_sub_2 = self.create_subscription(
            JointState,
            '/bot_2/joint_states',
            partial(self.gripper_posi_cb, bot_name="bot_2"),
            posi_qos_profile,
        )

        self.get_logger().info("Parsing connection arguments...")
        args1 = utilities.parseConnectionArguments1()
        args2 = utilities.parseConnectionArguments2()

        self.get_logger().info("Creating TCP connections to both arms...")
        self.router1 = utilities.DeviceConnection.createTcpConnection(args1)
        self.router2 = utilities.DeviceConnection.createTcpConnection(args2)

        self.base1 = BaseClient(self.router1)
        self.base_cyclic1 = BaseCyclicClient(self.router1)
        self.base2 = BaseClient(self.router2)
        self.base_cyclic2 = BaseCyclicClient(self.router2)

        self.reset_arms()

    def gripper_posi_cb(self, bot_name, msg):
        try:
            self.get_logger().debug(f"Received joints: {msg.name}")  # Debug joint names
            
            if 'robotiq_85_left_knuckle_joint' in msg.name:
                index = msg.name.index('robotiq_85_left_knuckle_joint')
                gripper_position = msg.position[index]
                if bot_name == "bot_1":
                    self.gripper_posi_1 = gripper_position
                else if bot_name == "bot_2":
                    self.gripper_posi_2 = gripper_position
                else:
                    self.get_logger().warn("Namespace passed to gripper posi callback is not valid")
                # self.get_logger().info(f"Current gripper position: {gripper_position:.4f}")
            else:
                self.get_logger().warn("Gripper joint not found in JointState message")
        except Exception as e:
            self.get_logger().error(f"joint_state_cb error: {str(e)}", throttle_duration_sec=5)

    def reset_arms(self):
        self.get_logger().info("Sending both arms to home position...")
        home_pos(self.base2, self.base_cyclic2)
        home_pos(self.base1, self.base_cyclic1)
        self.get_logger().info("Both arms are in home position!")

    def capture_raw_image(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            return cv_image
        except CvBridgeError as e:
            self.get_logger().error(f"CV Bridge Error: {e}")
            return None

    def calculate_external_force(self, follower_arm, delta_x, delta_y):
        return follower_arm.calculate_force(delta_x, delta_y)

    def relax_gripper():
        gripper_cmd = GripperCommand.Goal()
        # Paper limits movement speed to 4.5 mm/s
        # No clear way of doing that in kinova, so we will have to use manual timing delay

        opening_delta = 0.04 # might need tuning

        start = time.time()
        while( grasping_force > external force of impedance controller) {
            now = time.time()
            if (now - start >  0.05) {
                target_posi = self.gripper_posi_2 + opening_delta # treating arm 2 as follower
                start = now
            }
        }
    def simulate_impedance_controller(self):
        """
            The idea of this is to simulate an impedance controller by
            carrying out inverse movement between the arms. For example, 
            if leader moves left and forwards 10cm, follower moves right
            and back 10cm

            Need to find a way to ensure feasability and no collissions in paths
            before executing them.
        """

        delta_x = random.uniform(-0.035, 0.035)  # 35mm range
        delta_y = random.uniform(-0.021, 0.021)  # 21mm range
        
        # This probably executes sequentally so might need to use threading 
        # To do it at the same time
        move_arm(self.base1, self.base_cyclic1, delta_x, delta_y, 0, 0, 0, 0)
        move_arm(self.base2, self.base_cyclic2, delta_x, delta_y, 0, 0, 0, 0)
    

    def run_trial(self)


def main(args=None):
    rclpy.init(args=args)
    node = KinovaDataCollector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.router1.__exit__(None, None, None)
        node.router2.__exit__(None, None, None)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

"""

Possible impedance controller: https://github.com/empriselab/gen3_compliant_controllers
"""
