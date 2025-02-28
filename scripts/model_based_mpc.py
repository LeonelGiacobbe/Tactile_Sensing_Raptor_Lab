import numpy as np
from scipy import sparse
import time
import sys, tty, termios
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from rclpy.action import ActionClient
from control_msgs.action import GripperCommand
from sensor_msgs.msg import JointState
import osqp

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
        
        self.gripper_posi_ = 0
        self.gripper_ini_flag_ = True
        self.dis_sum_ = 0
        self.contact_area_ = 0

        # Replace publisher with ActionClient
        self._action_client = ActionClient(self, GripperCommand, '/robotiq_gripper_controller/gripper_cmd')

        # Receives tactile image from gelsight
        self.dis_sum_sub = self.create_subscription(
            Float32, '/gsmini_rawimg_0', self.dis_sum_cb, 10)

        self.old_attr = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())

        # Parameters initialization
        self.frequency = 60
        self.init_posi = 70
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

        self.run()

    def dis_sum_cb(self, msg):
        self.dis_sum_ = msg.data

    def run(self):
        try:
            while not self.gripper_ini_flag_:
                self.get_logger().info('Wait for initializing the gripper.')

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
                    # state initialization. Need way to get current position of gripper
                    x_state = np.array([self.contact_area_, -self.dis_sum_, self.gripper_cmd.command.position, x_state[3]])
                else:
                    # tactile state update
                    # contact area, dis sum, p, v
                    x_state = np.array([self.contact_area_, -self.dis_sum_, x_state[2], x_state[3]])

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
                    self.gripper_cmd.command.position = x_state[2]

                # Send goal to the action server
                self._send_goal(self.gripper_cmd)
                self.rate.sleep()
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_attr)
        except KeyboardInterrupt:
            self.get_logger().info('Interrupted!')

    def _send_goal(self, goal):
        self.get_logger().info('Sending goal...')
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
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()