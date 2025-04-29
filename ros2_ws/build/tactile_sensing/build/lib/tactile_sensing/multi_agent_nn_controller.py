import numpy as np
import cv2
from scipy import sparse
import sys, tty, termios
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, UInt16
from rclpy.action import ActionClient
from control_msgs.action import GripperCommand
from sensor_msgs.msg import JointState
from sensor_msgs.msg import Image as ROSImage
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.wait_for_message import wait_for_message
from concurrent.futures import ThreadPoolExecutor
import threading
import torch
from torchvision import transforms
from PIL import Image
from .changed_functions import ResCNNEncoder, MPClayer
from cv_bridge import CvBridge, CvBridgeError
import os, time
from ament_index_python.packages import get_package_share_directory
from torch.nn import functional as F

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
        self.gripper_vel_ = 0.0
        self.gripper_ini_flag_ = False
        self.contact_area_ini_flag = False
        self.dis_sum_ = 0
        self.contact_area_ = 0
        self.processing_executor = ThreadPoolExecutor(max_workers=1)
        self.contact_area_lock = threading.Lock()
        self.frequency = 2

        # Parameters initialization
        self.init_posi = 0.0
        self.lower_pos_lim = 0.0 # for wsg grippers, original values
        self.upper_pos_lim = 110 # for wsg grippers, original values
        self.new_min = 0.0
        self.new_max = 0.7 # robotiq gripper can do up to 0.8 but that causes mounts to collide
        self.N = 15  # horizon steps. Higher = more stable but more computation
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


        # Neural network stuff
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.get_logger().info(f"Using {self.device} in controller node")
        self.nn_encoder = ResCNNEncoder(outputDim=20).to(self.device)
        self.mpc_layer = MPClayer(nHidden=20, nStep=15).to(self.device)
        if self.device.type == 'cuda':
            self.stream = torch.cuda.Stream()
        self.nn_encoder.eval()
        self.mpc_layer.eval()
        # Warmup pass
        dummy_input = torch.randn(1, 3, 224, 224, device=self.device)
        _ = self.nn_encoder(dummy_input)
        if self.device.type == 'cuda':
            torch.cuda.synchronize()

        # Load weights
        package_dir = get_package_share_directory('tactile_sensing')
        model_path = os.path.join(package_dir, 'models', 'letac_mpc_model.pth')
        checkpoint = torch.load(model_path, map_location=torch.device(self.device))
        self.nn_encoder.load_state_dict(checkpoint['cnn_encoder_state_dict'])
        mpc_state_dict = checkpoint['mpc_layer_state_dict']
        model_dict = self.mpc_layer.state_dict()
        pretrained_dict = {k: v for k, v in mpc_state_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.mpc_layer.load_state_dict(model_dict, strict=False)
        self.get_logger().info("Loaded MPC weights (ignoring missing Qf_linear/Qv_linear)")

        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.bridge = CvBridge()

        # Removed all batching-related buffers
        self.current_image = None

        self.rate = self.create_rate(self.frequency)

        # Timer to call the run method periodically
        self.timer = self.create_timer(1.0 / self.frequency, self.run)

        gs_qos_profile = QoSProfile(
            depth=0,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST
        )   

        self._action_client = ActionClient(self, GripperCommand, '/robotiq_gripper_controller/gripper_cmd')
        self.contact_group = ReentrantCallbackGroup()
        # Receives image from tactile sensor
        self.contact_area_sub = self.create_subscription(
            ROSImage, 
            '/gsmini_rawimg_0', 
            self.contact_area_cb,
            gs_qos_profile, 
            callback_group=self.contact_group
        )
        
        # Subscribe to the JointState topic to get gripper position
        posi_qos_profile = QoSProfile(
            depth=0,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE  # Change to TRANSIENT_LOCAL
        )
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_cb,
            posi_qos_profile,
        )

    # Receives position of gripper: 0.0 -> completely open. 0.8 -> completely closed

    def process_joint_states(self, msg):
        try:
            self.get_logger().debug(f"Received joints: {msg.name}")  # Debug joint names
            
            if 'robotiq_85_left_knuckle_joint' in msg.name:
                index = msg.name.index('robotiq_85_left_knuckle_joint')
                gripper_position = msg.position[index]
                self.gripper_posi_ = gripper_position
                # self.get_logger().info(f"Current gripper position: {gripper_position:.4f}")
                
                gripper_vel = msg.velocity[index]
                self.gripper_vel_ = gripper_vel
                # self.get_logger().info(f"Current gripper vel: {gripper_vel:.4f}")  # Debug velocity
            else:
                self.get_logger().warn("Gripper joint not found in JointState message")
                
        except Exception as e:
            self.get_logger().error(f"joint_state_cb error: {str(e)}", throttle_duration_sec=5)


    def joint_state_cb(self, msg: JointState):
        # self.get_logger().debug("Entered joint_state_cb")  # Debug entry
        
        # Flag to allow run method to go out of inf loop
        self.gripper_ini_flag_ = True

        # Offload joint processing to background thread
        self.processing_executor.submit(self.process_joint_states, msg)
        
        
    # Not currently used (original author said it can be set to zero?) but here just in case
    def dis_sum_cb(self, msg):
        self.dis_sum_ = msg.data        
        
    def contact_area_cb(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')            
            # Convert to tensor and process single image
            tensor = torch.from_numpy(cv_image).permute(2, 0, 1).float().to(self.device) / 255.0
            mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(3, 1, 1)
            tensor = F.interpolate(tensor.unsqueeze(0), size=(224, 224)).squeeze(0)
            tensor = (tensor - mean) / std
            
            with self.contact_area_lock:
                self.current_image = tensor  # Store single image

        except Exception as e:
            self.get_logger().error(f'Tactile processing failed: {str(e)}')
        
    def run(self):
        if self.current_image is None:
            return  # Not enough data

        try:
            # Prepare batched tensors
            with torch.no_grad():
                image_tensor = self.current_image.unsqueeze(0)
                # 0.2s runtime approx before this
                gripper_p_batch = torch.tensor([self.gripper_posi_], dtype=torch.float32).to(self.device).unsqueeze(1)
                gripper_v_batch = torch.tensor([self.gripper_vel_], dtype=torch.float32).to(self.device).unsqueeze(1)
                # Not much more runtime before this
                tactile_embeddings = self.nn_encoder(image_tensor) # 0.16s spent here
                start_time = time.time()
                pos_sequences = self.mpc_layer(tactile_embeddings, gripper_p_batch, gripper_v_batch) # 0.6s here
                stop_time = time.time()

            # Take the action from the most recent image (last in batch)
            target_pos = pos_sequences[:, 0].item()
            self.get_logger().info(f"Position sequence: {pos_sequences}")

            
            # self.get_logger().info(f"Inference run time: {stop_time - start_time:.4f}s")
            
            # Send command
            normalized_target = (self.new_min + ((target_pos - self.lower_pos_lim) / (self.upper_pos_lim - self.lower_pos_lim)) * (self.new_max - self.new_min))
            self.get_logger().info(f"Current normalized target: {normalized_target}")
            self.goal = GripperCommand.Goal()
            self.goal.command.position = self.gripper_posi_ + normalized_target
            # self.goal.command.max_effort = 100.0
            self.get_logger().info(f"Sending goal: {self.goal.command.position:.4f}")
            self._send_goal(self.goal)
            self.get_logger().info(f"Current gripper posi: {self.gripper_posi_}")
            self.rate.sleep()


        except Exception as e:
            self.get_logger().error(f'Control loop failed: {str(e)}')

    def _send_goal(self, goal):
        if not self._action_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().error("Action server not available after waiting")
            return
        self.get_logger().info("Got life check from server")
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