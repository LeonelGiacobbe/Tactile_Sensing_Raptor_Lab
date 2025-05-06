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

def vstack_help(vec, n):
    """
    Repeats a given vector vertically `n` times to create a stacked array.

    Parameters:
    vec (numpy.ndarray): A 1D numpy array to be vertically stacked.
    n (int): The number of times to repeat the vector vertically.

    Returns:
    numpy.ndarray: A 2D numpy array where the input vector is stacked vertically `n` times.
    """
    combo = vec.reshape(vec.size, 1)
    single = vec.reshape(vec.size, 1)
    for i in range(n - 1):
        combo = np.vstack((combo, single))
    return combo

def zeros_hstack_help(vec, n, size_row, size_col):
    """
    Horizontally stacks a given sparse matrix with n-1 additional zero matrices.

    Parameters:
        vec (scipy.sparse.csc_matrix): The initial sparse matrix to be horizontally stacked.
        n (int): The total number of matrices to be horizontally stacked, including `vec`.
        size_row (int): The number of rows in the zero matrices to be added.
        size_col (int): The number of columns in the zero matrices to be added.

    Returns:
        scipy.sparse.csc_matrix: A sparse matrix resulting from the horizontal stacking of `vec` 
        and `n-1` zero matrices.
    """
    combo = vec
    single = sparse.csc_matrix((size_row, size_col), dtype=np.int8)
    for i in range(n - 1):
        combo = sparse.hstack((combo, single))
    return combo

def zeros_hstack_help_inverse(vec, n, size_row, size_col):
    """
    Constructs a sparse matrix by horizontally stacking a given sparse vector `vec` 
    with `n-1` zero matrices of specified dimensions.

    Parameters:
        vec (scipy.sparse.csc_matrix): The sparse vector to be appended at the end of the stacked matrices.
        n (int): The total number of matrices to be horizontally stacked, including `vec`.
        size_row (int): The number of rows in each zero matrix.
        size_col (int): The number of columns in each zero matrix.

    Returns:
        scipy.sparse.csc_matrix: A sparse matrix resulting from horizontally stacking 
        `n-1` zero matrices with the given `vec` at the end.
    """
    end = vec
    single = sparse.csc_matrix((size_row, size_col), dtype=np.int8)
    combo = single
    for i in range(n - 2):
        combo = sparse.hstack((combo, single))
    combo = sparse.hstack((combo, end))
    return combo

def getCS_(C, S_):
    C_ = sparse.block_diag([sparse.kron(sparse.eye(10), C)])
    return C_ * S_

def getCT_(C, T_):
    C_ = sparse.block_diag([sparse.kron(sparse.eye(10), C)])  
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

        # Parameters initialization
        self.frequency = 2
        self.init_posi = 0.0
        self.lower_pos_lim = 0.0 # for wsg grippers, original values
        self.upper_pos_lim = 110 # for wsg grippers, original values
        self.new_min = 0.0
        self.new_max = 0.7 # robotiq gripper can do up to 0.8 but that causes mounts to collide
        self.N = 10  # horizon steps. Higher = more stable but more computation
        self.q_c = 36 # weight for contact tracking error. Higher = more aggressive maintenance
        self.q_v = 1 # velocity weight. Higher = smoother but slower movement
        self.q_d = 2 # displacement sum weight
        self.q_a = 2 # acceleration control weight. Higher = smoother but less responsive
        self.p = 5 # termainal cost weight
        self.c_ref = 3500 # amount of white pixels to ideally reach
        self.k_c = 50000 # stiffness coefficient. Higher = faster response to contact changes
        self.acc_max = 300 # max allowed acc
        self.vel_max = 500 # max allowed vel
        self.dim = 4 # state vector dimension

        self.del_t = 1 / self.frequency
        self.gripper_cmd = GripperCommand.Goal()
        self.gripper_cmd.command.position = float(self.init_posi)
        self.gripper_cmd.command.max_effort = 100.0  # Set max effort
        self.rate = self.create_rate(self.frequency)

        # Initialize all constant MPC components
        self._initialize_mpc_components()

        # Timer to call the run method periodically
        self.timer = self.create_timer(1.0 / self.frequency, self.run)

    def _initialize_mpc_components(self):
        """Initialize all constant MPC components that don't change during execution"""
        # reference to track
        self.r = np.array([self.c_ref, 0, 0, 0])
        self.r_ = vstack_help(self.r, self.N)

        # model
        self.Ad = sparse.csc_matrix([
            [1, 0, 0, self.k_c * self.del_t],
            [0, 1, 0, 0],
            [0, 0, 1, -self.del_t],
            [0, 0, 0, 1]
        ])

        self.Bd = sparse.csc_matrix([
            [0],
            [0],
            [-0.5 * self.del_t * self.del_t],
            [self.del_t]
        ])

        # weights
        self.Q = sparse.csc_matrix([
            [self.q_c, self.q_c * self.q_d, 0, 0],
            [self.q_c * self.q_d, self.q_c * (self.q_d ** 2), 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, self.q_v]
        ])
        self.R = self.q_a * sparse.eye(1)
        self.QN = self.p * self.Q
        self.Q_ = sparse.block_diag([sparse.kron(sparse.eye(self.N - 1), self.Q), self.QN], format='csc')
        self.R_ = sparse.block_diag([sparse.kron(sparse.eye(self.N), self.R)], format='csc')

        # T initialization
        self.T_ = self.Ad
        temp = self.Ad
        for i in range(self.N - 1):
            temp = temp.dot(self.Ad)
            self.T_ = sparse.vstack([self.T_, temp])

        I = sparse.eye(self.dim)
        row_single = zeros_hstack_help(I, self.N, self.dim, self.dim)
        self.AN_ = row_single
        for i in range(self.N - 1):
            AN = I
            row_single = I
            for j in range(i + 1):
                AN = self.Ad.dot(AN)
                row_single = sparse.hstack([AN, row_single])
            row_single = zeros_hstack_help(row_single, self.N - i - 1, self.dim, self.dim)
            self.AN_ = sparse.vstack([self.AN_, row_single])

        self.Bd_ = sparse.block_diag([sparse.kron(sparse.eye(self.N), self.Bd)])
        self.S_ = self.AN_ * self.Bd_

        # vel and acc constraints
        self.max_con_b = (np.array([self.vel_max])).reshape(1, 1)
        self.min_con_b = (np.array([-self.vel_max])).reshape(1, 1)
        self.u_max = self.acc_max * np.ones(1 * self.N)
        self.u_max = self.u_max.reshape(1 * self.N, 1)

        self.max_con_b_ = vstack_help(self.max_con_b, self.N)
        self.min_con_b_ = vstack_help(self.min_con_b, self.N)

        # vel select matrix
        self.C_con = sparse.csc_matrix([
            [0, 0, 0, 1]
        ])

        self.C_con_T_ = getCT_(self.C_con, self.T_)

        # select matrix cost function
        self.L = sparse.eye(self.dim)
        self.L_ = sparse.block_diag([sparse.kron(sparse.eye(self.N), self.L)], format='csc')

        # QP setup matrices
        self.P_ = 2 * (self.R_ + (self.S_.T) * (self.L_.T) * self.Q_ * self.L_ * self.S_)
        self.A_ = sparse.vstack([sparse.block_diag([sparse.eye(1 * self.N)], format='csc'), getCS_(self.C_con, self.S_)])

        # Initialize OSQP problem
        self.prob = osqp.OSQP()
        self.prob.setup(self.P_, np.zeros(self.P_.shape[0]), self.A_, 
                      np.zeros(self.A_.shape[0]), np.zeros(self.A_.shape[0]), 
                      warm_start=True, max_iter=4000)

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
            # Wait until gripper posi callback is called once
            wait_for_message(JointState, self, '/joint_states', time_to_wait=10.0)
            wait_for_message(UInt16, self, '/gs_contact_area', time_to_wait=10.0)

            # Initial state
            x_state = np.array([0., 0., 0., 0.])

            while rclpy.ok():
                if x_state[2] == 0.:
                    # state initialization
                    print("Gripper velocity: ", x_state[3])
                    with self.contact_area_lock:
                        x_state = np.array([self.contact_area_, 0, gripper_posi_to_mm(self.gripper_posi_), x_state[3]]) # change -self.dis_sum_ to 0
                else:
                    # tactile state update
                    # contact area, dis sum, p, v
                    print("Gripper velocity: ", x_state[3])
                    with self.contact_area_lock:
                        x_state = np.array([self.contact_area_, 0, gripper_posi_to_mm(x_state[2]), x_state[3]]) # change -self.dis_sum_ to 0

                # constraints update
                max_con_b_update = b_CT_x0(self.max_con_b_, self.C_con_T_, x_state.reshape(self.dim, 1))
                min_con_b_update = b_CT_x0(self.min_con_b_, self.C_con_T_, x_state.reshape(self.dim, 1))
                u_ = np.vstack([self.u_max, max_con_b_update])
                l_ = np.vstack([self.u_max * -1, min_con_b_update])

                # QP update
                q_ = 2 * (x_state.reshape(1, self.dim) * (self.T_.T) * (self.L_.T) - self.r_.T) * self.Q_ * self.L_ * self.S_
                self.prob.update(q=q_.T, l=l_, u=u_)
                res = self.prob.solve()
                ctrl = res.x[0:1].copy()

                self.gripper_cmd = GripperCommand.Goal()
                self.gripper_cmd.command.max_effort = 100.0

                if ctrl[0] is not None:
                    # p, v update
                    x_state = self.Ad.dot(x_state) + self.Bd.dot(ctrl)
                    print(f"x_state: {x_state}")
                    print("Current tactile value: ", self.contact_area_)
                    self.gripper_cmd.command.position = self.gripper_posi_ + mm_to_gripper_posi(x_state[2])
                    self._action_client.send_goal_async(self.gripper_cmd)
                
                # Send goal to the action server
                self._send_goal(self.gripper_cmd)
                self.rate.sleep()
            print("RCLPY not OK. Exiting... ")

        except KeyboardInterrupt:
            self.get_logger().info('Interrupted!')

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
    node = ModelBasedMPCNode()
    executor = MultiThreadedExecutor(num_threads=10)
    executor.add_node(node)
    executor.spin()

if __name__ == '__main__':
    main()