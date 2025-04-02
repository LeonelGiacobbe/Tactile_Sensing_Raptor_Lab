import numpy as np
import cv2
from scipy import sparse
import time
import sys, tty, termios
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from rclpy.action import ActionClient
from control_msgs.action import GripperCommand
from sensor_msgs.msg import JointState, Image
import osqp
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from rclpy.executors import MultiThreadedExecutor
from cv_bridge import CvBridge

def vstack_help(vec, n):
    combo = vec.reshape(vec.size, 1)
    single = vec.reshape(vec.size, 1)
    for i in range(n - 1):
        combo = np.vstack((combo, single))
    return combo

def zeros_hstack_help(vec, n, size_row, size_col):
    combo = vec
    single = sparse.csc_matrix((size_row, size_col), dtype=np.int8)
    for i in range(n - 1):
        combo = sparse.hstack((combo, single))
    return combo

def zeros_hstack_help_inverse(vec, n, size_row, size_col):
    end = vec
    single = sparse.csc_matrix((size_row, size_col), dtype=np.int8)
    combo = single
    for i in range(n - 2):
        combo = sparse.hstack((combo, single))
    combo = sparse.hstack((combo, end))
    return combo

def getCS_(C, S_):
    C_ = sparse.block_diag([sparse.kron(sparse.eye(15), C)])  # CHANGE => N to 15
    return C_ * S_

def getCT_(C, T_):
    C_ = sparse.block_diag([sparse.kron(sparse.eye(15), C)])  # CHANGE => N to 15
    return C_ * T_

def b_CT_x0(b_, CT_, x0):
    return b_ - CT_ * x0

class ModelBasedMPCNode(Node):
    def __init__(self):
        super().__init__('model_based_mpc_node')
        
        self.gripper_posi_ = 0.0
        self.gripper_ini_flag_ = False
        self.contact_area_ini_flag = False
        self.dis_sum_ = 0
        self.contact_area_ = 0

        # Replace publisher with ActionClient
        self._action_client = ActionClient(self, GripperCommand, '/robotiq_gripper_controller/gripper_cmd')

        # Receives tactile image from gelsight (why in format float32?) encoding is 8UC3
        gs_qos_profile = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE  # Change to TRANSIENT_LOCAL
        )
        self.contact_area_sub = self.create_subscription(
            Image, '/gs_depth', self.contact_area_cb, gs_qos_profile)
        
        self.cv_bridge = CvBridge()

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

        self.old_attr = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())

        # Parameters initialization
        self.frequency = 10
        self.init_posi = 0.0
        self.lower_pos_lim = 0.0 # for wsg grippers, original values
        self.upper_pos_lim = 110 # for wsg grippers, original values
        self.new_min = 0.0
        self.new_max = 0.7 # robotiq gripper can do up to 0.8 but that causes mounts to collide
        self.N = 15  # horizon
        self.q_c = 36
        self.q_v = 1
        self.q_d = 2
        self.q_a = 2
        self.p = 5
        self.c_ref = 1000
        self.k_c = 36000
        self.acc_max = 30
        self.vel_max = 10
        self.dim = 4

        self.del_t = 1 / self.frequency
        self.gripper_cmd = GripperCommand.Goal()
        self.gripper_cmd.command.position = float(self.init_posi)
        self.gripper_cmd.command.max_effort = 100.0  # Set max effort
        self.rate = self.create_rate(self.frequency)

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
        
    # Gets data from gelsight sensor. Need to figure out if this is the correct data format
    def contact_area_cb(self, msg):
        try: 
            self.contact_area_ini_flag = True
            depth_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding="32FC1")

            # Normalize to avoid errors in thresholding
            depth_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            
            # Convert image to binary
            _, binary_image = cv2.threshold(depth_normalized, 240, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            white_pixels = np.count_nonzero(binary_image)
    
            # Total pixels in the image
            total_pixels = binary_image.size
            self.get_logger().info(f"Received image with encoding: {msg.encoding}")
            
            # Calculate percentage (scaled to [0, 1])
            self.contact_area_ = white_pixels / total_pixels

            print("Current contact area value: ", self.contact_area_)
        except Exception as e:
            self.get_logger().error(f"Failed to process image: {e}")
        
    
    def run(self):
        try:
            # Wait until gripper posi callback is called once
            while not self.gripper_ini_flag_:
                self.get_logger().info('Wait for initializing the gripper.')
                time.sleep(0.1)

            while not self.contact_area_ini_flag:
                self.get_logger().info('Wait for initializing the contact area sub.')
                time.sleep(0.1)

            # Wait for user to press 'l'
            while (sys.stdin.read(1) != 'l'):
                self.get_logger().info('Wait for starting! Press l to start')
                time.sleep(0.1)

            # state and control Initialization
            x_state = np.array([0., 0., 0., 0.])
            u0 = np.array([[0.]])

            # reference to track
            r = np.array([self.c_ref, 0, 0, 0])
            r_ = vstack_help(r, self.N)

            # model
            Ad = sparse.csc_matrix([
                [1, 0, 0, self.k_c * self.del_t],
                [0, 1, 0, 0],
                [0, 0, 1, -self.del_t],
                [0, 0, 0, 1]
            ])

            Bd = sparse.csc_matrix([
                [0],
                [0],
                [-0.5 * self.del_t * self.del_t],
                [self.del_t]
            ])

            # weights
            Q = sparse.csc_matrix([
                [self.q_c, self.q_c * self.q_d, 0, 0],
                [self.q_c * self.q_d, self.q_c * (self.q_d ** 2), 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, self.q_v]
            ])
            R = self.q_a * sparse.eye(1)
            QN = self.p * Q
            Q_ = sparse.block_diag([sparse.kron(sparse.eye(self.N - 1), Q), QN], format='csc')
            R_ = sparse.block_diag([sparse.kron(sparse.eye(self.N), R)], format='csc')

            # T initialization
            T_ = Ad
            temp = Ad
            for i in range(self.N - 1):
                temp = temp.dot(Ad)
                T_ = sparse.vstack([T_, temp])

            I = sparse.eye(self.dim)
            row_single = zeros_hstack_help(I, self.N, self.dim, self.dim)
            AN_ = row_single
            for i in range(self.N - 1):
                AN = I
                row_single = I
                for j in range(i + 1):
                    AN = Ad.dot(AN)
                    row_single = sparse.hstack([AN, row_single])
                row_single = zeros_hstack_help(row_single, self.N - i - 1, self.dim, self.dim)
                AN_ = sparse.vstack([AN_, row_single])

            Bd_ = sparse.block_diag([sparse.kron(sparse.eye(self.N), Bd)])
            S_ = AN_ * Bd_

            # vel and acc constraints
            max_con_b = (np.array([self.vel_max])).reshape(1, 1)
            min_con_b = (np.array([-self.vel_max])).reshape(1, 1)
            u_max = self.acc_max * np.ones(1 * self.N)
            u_max = u_max.reshape(1 * self.N, 1)

            max_con_b_ = vstack_help(max_con_b, self.N)
            min_con_b_ = vstack_help(min_con_b, self.N)

            # vel select matrix
            C_con = sparse.csc_matrix([
                [0, 0, 0, 1]
            ])

            C_con_T_ = getCT_(C_con, T_)

            # real-time vel bounds
            max_con_b_update = b_CT_x0(max_con_b_, C_con_T_, x_state.reshape(self.dim, 1))
            min_con_b_update = b_CT_x0(min_con_b_, C_con_T_, x_state.reshape(self.dim, 1))

            u_ = np.vstack([u_max, max_con_b_update])
            l_ = np.vstack([u_max * -1, min_con_b_update])

            # select matrix for cost function
            L = sparse.eye(self.dim)
            L_ = sparse.block_diag([sparse.kron(sparse.eye(self.N), L)], format='csc')

            # QP setup
            P_ = 2 * (R_ + (S_.T) * (L_.T) * Q_ * L_ * S_)
            q_ = 2 * (x_state.reshape(1, self.dim) * (T_.T) * (L_.T) - r_.T) * Q_ * L_ * S_
            A_ = sparse.vstack([sparse.block_diag([sparse.eye(1 * self.N)], format='csc'), getCS_(C_con, S_)])

            prob = osqp.OSQP()
            prob.setup(P_, q_.T, A_, l_, u_, warm_start=True, max_iter=8000)
            initial_flag = False

            while rclpy.ok():
                if x_state[2] == 0.:
                    # state initialization
                    print("Gripper position before movement 1: ", self.gripper_posi_)
                    x_state = np.array([self.contact_area_, 0, self.gripper_posi_, x_state[3]]) # change -self.dis_sum_ to 0
                else:
                    # tactile state update
                    # contact area, dis sum, p, v
                    print("Gripper position before movement 2: ", self.gripper_posi_)
                    x_state = np.array([self.contact_area_, 0, x_state[2], x_state[3]]) # change -self.dis_sum_ to 0

                # constraints update
                max_con_b_update = b_CT_x0(max_con_b_, C_con_T_, x_state.reshape(self.dim, 1))
                min_con_b_update = b_CT_x0(min_con_b_, C_con_T_, x_state.reshape(self.dim, 1))
                u_ = np.vstack([u_max, max_con_b_update])
                l_ = np.vstack([u_max * -1, min_con_b_update])

                # QP update
                q_ = 2 * (x_state.reshape(1, self.dim) * (T_.T) * (L_.T) - r_.T) * Q_ * L_ * S_
                prob.update(q=q_.T, l=l_, u=u_)
                res = prob.solve()
                ctrl = res.x[0:1].copy()

                if ctrl[0] is not None:
                    # p, v update
                    x_state = Ad.dot(x_state) + Bd.dot(ctrl)
                    normalized_x_state = self.new_min + ((x_state[2] - self.lower_pos_lim) / (self.upper_pos_lim - self.lower_pos_lim)) * (self.new_max - self.new_min)
                    print("x_state_2 ", abs(normalized_x_state))
                    print("Current tactile value: ", self.contact_area_)
                    self.gripper_cmd.command.position = self.gripper_posi_ + abs(normalized_x_state)

                # Send goal to the action server
                self._send_goal(self.gripper_cmd)
                self.rate.sleep()
            print("RCLPY not OK. Exiting... ")

        except KeyboardInterrupt:
            self.get_logger().info('Interrupted!')

    def _send_goal(self, goal):
        self.get_logger().info('Sending goal...')
        self._action_client.wait_for_server()
        print("Goal: ", goal)
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
    node = ModelBasedMPCNode()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    executor.spin()

if __name__ == '__main__':
    main()
