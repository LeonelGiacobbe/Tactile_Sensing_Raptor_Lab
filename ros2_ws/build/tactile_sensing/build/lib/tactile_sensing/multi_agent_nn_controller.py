import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from control_msgs.action import GripperCommand
from sensor_msgs.msg import JointState
from sensor_msgs.msg import Image as ROSImage
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from concurrent.futures import ThreadPoolExecutor
import threading, queue
import torch
from torchvision import transforms
from PIL import Image
from .multi_agent_functions import ResCNNEncoder, MPClayer
from cv_bridge import CvBridge
import os, time
from ament_index_python.packages import get_package_share_directory

CONVERSION_RATE_140_MM = 0.00571429 # Kinova unit to mm
CONVERSION_RATE_85_MM = 0.00941176 # Kinova unit to mm

# Model architecture params
CNN_hidden1, CNN_hidden2 = 128, 128 
CNN_embed_dim = 20  
res_size = 224       
eps = 1e-4
nStep = 15
del_t = 1/25
dropout_p = 0.15

def gripper_posi_to_mm_140(gripper_posi):
    opening = 0.8 - gripper_posi
    return opening / CONVERSION_RATE_140_MM

def mm_to_gripper_posi_140(millimeters):
    opening = 140 - millimeters
    return opening * CONVERSION_RATE_140_MM

def gripper_posi_to_mm_85(gripper_posi):
    opening = 0.8 - gripper_posi
    return opening / CONVERSION_RATE_85_MM

def mm_to_gripper_posi_85(millimeters):
    opening = 85 - millimeters
    return opening * CONVERSION_RATE_85_MM


class ModelBasedMPCNode(Node):
    def __init__(self):
        super().__init__('model_based_mpc_node')
        
        # Model variables
        self.gripper_posi_1 = 0.0
        self.gripper_vel_1 = 0.0
        self.gripper_posi_2 = 0.0
        self.gripper_vel_2 = 0.0
        self.processing_executor = ThreadPoolExecutor(max_workers=1)
        self.contact_area_lock = threading.Lock()
        self.frequency = 10
        self.current_image_1 = None
        self.current_image_2 = None

        # Inference thread setup
        self.inference_queue = queue.Queue(maxsize=1)
        self.inference_lock = threading.Lock()
        self.inference_thread = threading.Thread(target=self._inference_worker, daemon=True)
        self.inference_thread.start()
        self.latest_results = None
        self.results_lock = threading.Lock()

        # Neural network stuff
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.get_logger().info(f"Using {self.device} in controller node")
        self.nn_encoder = ResCNNEncoder(hidden1=CNN_hidden1, hidden2=CNN_hidden2, dropP=dropout_p, outputDim=CNN_embed_dim).to(self.device)
        self.mpc_layer = MPClayer(nHidden = CNN_embed_dim, eps = eps, nStep = nStep, del_t = del_t).to(self.device)
        if self.device.type == 'cuda':
            self.stream = torch.cuda.Stream()
            torch.cuda.synchronize()
        self.nn_encoder.eval()
        self.mpc_layer.eval()

        # Warmup pass
        dummy_input = torch.randn(1, 3, 224, 224, device=self.device)
        dummy_mpc_var = torch.tensor(0).to(self.device)
        _ = self.nn_encoder(dummy_input)  
        _ = self.mpc_layer(_, _, dummy_mpc_var, dummy_mpc_var, dummy_mpc_var, dummy_mpc_var)          

        # Load weights
        package_dir = get_package_share_directory('tactile_sensing')
        model_path = os.path.join(package_dir, 'models', 'multi_agent_tactile_model.pth')
        checkpoint = torch.load(model_path, map_location=torch.device(self.device))
        self.nn_encoder.load_state_dict(checkpoint['cnn_encoder_state_dict'])
        self.mpc_layer.load_state_dict(checkpoint['mpc_layer_state_dict'])
        self.get_logger().info("Loaded MPC weights")

        # Image preprocessing (using the same transform as in training of the CNN encoder)
        self.transform = transforms.Compose([transforms.Resize([224, 224]),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0, 0, 0], std=[0.2, 0.2, 0.2])])
        
        # To convert GelSight images from ROS Image to CV Image
        self.bridge = CvBridge()

        self.rate = self.create_rate(self.frequency)

        # Timer to call the run method periodically
        self.timer = self.create_timer(1.0 / self.frequency, self.run)

        # QoS profiles for subscriptions
        gs_qos_profile = QoSProfile(
            depth=0,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST
        )   
        posi_qos_profile = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE 
        )

        # Using Kinova arm, and we send gripper posi commands through Action Server
        self._action_client_1 = ActionClient(self, GripperCommand, '/arm_1_/robotiq_gripper_controller/gripper_cmd')
        self._action_client_2 = ActionClient(self, GripperCommand, '/arm_2_/robotiq_gripper_controller/gripper_cmd')
        self.callback_group = ReentrantCallbackGroup()

        # Receives image from tactile sensor
        self.contact_area_sub_1 = self.create_subscription(
            ROSImage, 
            '/gsmini_rawimg_1', 
            self.contact_area_cb_1,
            gs_qos_profile, 
            callback_group=self.callback_group
        )

        self.contact_area_sub_2 = self.create_subscription(
            ROSImage, 
            '/gsmini_rawimg_2', 
            self.contact_area_cb_2,
            gs_qos_profile, 
            callback_group=self.callback_group
        )
        
        # Gripper position subscriptions
        self.arm_1_joint_state_sub = self.create_subscription(
            JointState,
            '/arm_1_/joint_states',
            self.arm_1_joint_state_cb,
            posi_qos_profile,
            callback_group=self.callback_group
        )

        self.arm_2_joint_state_sub = self.create_subscription(
            JointState,
            '/arm_2_/joint_states',
            self.arm_2_joint_state_cb,
            posi_qos_profile,
            callback_group=self.callback_group
        )

    # Receives position of gripper: 0.0 -> completely open. 0.8 -> completely closed        
    def arm_1_joint_state_cb(self, msg: JointState):
        # self.get_logger().debug("Entered joint_state_cb")  # Debug entry
        try:
            self.get_logger().debug(f"Received joints: {msg.name}")  # Debug joint names
            
            if 'arm_1_robotiq_85_left_knuckle_joint' in msg.name:
                index = msg.name.index('arm_1_robotiq_85_left_knuckle_joint')
                gripper_position = msg.position[index]
                if gripper_position < 0.0:
                    raise ValueError("Invalid gripper position, check publisher")

                self.gripper_posi_1 = gripper_position
                # self.get_logger().info(f"Current arm 1 gripper position: {gripper_position:.4f}")
                
                gripper_vel = msg.velocity[index]
                self.gripper_vel_1 = gripper_vel
                # self.get_logger().info(f"Current gripper vel: {gripper_vel:.4f}")  # Debug velocity
                
            else:
                self.get_logger().warn("Gripper joint not found in JointState message")
                
        except Exception as e:
            self.get_logger().error(f"joint_state_cb error: {str(e)}", throttle_duration_sec=5)

    def arm_2_joint_state_cb(self, msg: JointState):
        # self.get_logger().debug("Entered joint_state_cb")  # Debug entry
        try:
            self.get_logger().debug(f"Received joints: {msg.name}")  # Debug joint names
            
            if 'arm_2_robotiq_140_left_knuckle_joint' in msg.name:
                index = msg.name.index('arm_2_robotiq_140_left_knuckle_joint')
                gripper_position = msg.position[index]
                if gripper_position < 0.0:
                    raise ValueError("Invalid gripper position, check publisher")

                self.gripper_posi_2 = gripper_position
                # self.get_logger().info(f"Current arm 2 gripper position: {gripper_position:.4f}")
                
                gripper_vel = msg.velocity[index]

                self.gripper_vel_2 = gripper_vel
                # self.get_logger().info(f"Current gripper vel: {gripper_vel:.4f}")  # Debug velocity
                
            else:
                self.get_logger().warn("Gripper joint not found in JointState message")
                
        except Exception as e:
            self.get_logger().error(f"joint_state_cb error: {str(e)}", throttle_duration_sec=5)

        
    # Not currently used (original author said it can be set to zero?) but here just in case
    def dis_sum_cb(self, msg):
        self.dis_sum_ = msg.data        
        
    def contact_area_cb_1(self, msg):
        #start = time.time()
        try:
            # self.get_logger().info(f"Received image in gelsight topic 1")
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')  
            pil_image = Image.fromarray(cv_image).convert('RGB')        
            # Convert to tensor and process single image
            tensor = self.transform(pil_image).to(self.device)
            
            with self.contact_area_lock:
                self.current_image_1 = tensor  # Store single image
            #stop = time.time()
            # self.get_logger().info(f"Image processing runtime: {stop - start}")

        except Exception as e:
            self.get_logger().error(f'Tactile processing failed: {str(e)}')

    def contact_area_cb_2(self, msg):
        #start = time.time()
        try:
            # self.get_logger().info(f"Received image in gelsight topic 2")
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')   
            # cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)  
            pil_image = Image.fromarray(cv_image).convert('RGB')        
            # Convert to tensor and process single image
            tensor = self.transform(pil_image).to(self.device)
            
            with self.contact_area_lock:
                self.current_image_2 = tensor  # Store single image
            #stop = time.time()
            # self.get_logger().info(f"Image processing runtime: {stop - start}")

        except Exception as e:
            self.get_logger().error(f'Tactile processing failed: {str(e)}')
        
    def _inference_worker(self):
        while rclpy.ok():
            try:
                # Get data from queue with timeout to periodically check rclpy.ok()
                data = self.inference_queue.get(timeout=0.1)
                
                with self.stream, torch.no_grad():
                    # Unpack data
                    (image_tensor_1, image_tensor_2, 
                     gripper_posi_1, gripper_vel_1,
                     gripper_posi_2, gripper_vel_2) = data
                    
                    self.get_logger().info(f"p and v data types: {type(gripper_posi_1)}, {type(gripper_vel_1)}")
                    # Perform inference
                    start = time.time()
                    tactile_embeddings_1 = self.nn_encoder(image_tensor_1) 
                    tactile_embeddings_2 = self.nn_encoder(image_tensor_2)
                    pos_sequences_1, pos_sequences_2 = self.mpc_layer(
                        tactile_embeddings_1, tactile_embeddings_2, 
                        gripper_posi_1, gripper_vel_1, 
                        gripper_posi_2, gripper_vel_2
                    )
                    end = time.time()
                    self.get_logger().info(f"Total inference time: {end - start}")
                    # Store results
                    with self.results_lock:
                        self.latest_results = (pos_sequences_1, pos_sequences_2)
                        
            except queue.Empty:
                continue
            except Exception as e:
                self.get_logger().error(f'Inference worker failed: {str(e)}')

    def run(self):
        try:
            # Prepare data for inference
            with self.contact_area_lock:
                if self.current_image_1 is None or self.current_image_2 is None:
                    return
                    
                # Prepare tensors
                image_tensor_1 = self.current_image_1.unsqueeze(0).float()
                image_tensor_2 = self.current_image_2.unsqueeze(0).float()
                self.current_image_1 = None
                self.current_image_2 = None

            # Prepare other inputs
            gripper_posi_1 = torch.tensor([gripper_posi_to_mm_85(self.gripper_posi_1)]).to(self.device)
            gripper_vel_1 = torch.tensor(self.gripper_vel_1).to(self.device)
            gripper_posi_2 = torch.tensor([gripper_posi_to_mm_140(self.gripper_posi_2)]).to(self.device)
            gripper_vel_2 = torch.tensor(self.gripper_vel_2).to(self.device)

            # Put data in queue (non-blocking, replace if full)
            try:
                self.inference_queue.put_nowait((
                    image_tensor_1, image_tensor_2,
                    gripper_posi_1, gripper_vel_1,
                    gripper_posi_2, gripper_vel_2
                ))
            except queue.Full:
                pass  # Skip this cycle if previous inference still running

            # Check for and use latest results
            with self.results_lock:
                if self.latest_results is not None:
                    pos_sequences_1, pos_sequences_2 = self.latest_results
                    self.latest_results = None  # Clear after use
                    
                    # Take the first action in the horizon
                    target_pos_1 = pos_sequences_1[:, 0].item() # Now in mm
                    target_pos_1 = mm_to_gripper_posi_85(target_pos_1) # Now converted to kinova scale

                    target_pos_2 = pos_sequences_2[:, 0].item()
                    target_pos_2 = mm_to_gripper_posi_140(target_pos_2)
                    
                    self.get_logger().info(f"Target pos sequence 1: {pos_sequences_1}")
                    self.get_logger().info(f"Target pos sequence 2: {pos_sequences_2}")
                    
                    # Send command for arm 1
                    self.goal_1 = GripperCommand.Goal()
                    self.goal_1.command.position = target_pos_1
                    self.goal_1.command.max_effort = 100.0
                    self._send_goal_1(self.goal_1)

                    # Send command for arm 2
                    self.goal_2 = GripperCommand.Goal()
                    self.goal_2.command.position = target_pos_2
                    self.goal_2.command.max_effort = 100.0
                    self._send_goal_2(self.goal_2)

        except Exception as e:
            self.get_logger().error(f'Control loop failed: {str(e)}')

    def _send_goal_1(self, goal):
        if not self._action_client_1.wait_for_server(timeout_sec=2.0):
            self.get_logger().error("Action server not available after waiting")
            return
        # self.get_logger().info("Got life check from server")
        self._send_goal_future_1 = self._action_client_1.send_goal_async(goal)
        self._send_goal_future_1.add_done_callback(self.goal_response_callback_1)

    def _send_goal_2(self, goal):
        if not self._action_client_2.wait_for_server(timeout_sec=2.0):
            self.get_logger().error("Action server not available after waiting")
            return
        # self.get_logger().info("Got life check from server")
        self._send_goal_future_2 = self._action_client_2.send_goal_async(goal)
        self._send_goal_future_2.add_done_callback(self.goal_response_callback_2)

    def goal_response_callback_1(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Arm 1 Goal rejected :(')
            return
        # self.get_logger().info('Goal accepted :)')
        self.get_result_future_1 = goal_handle.get_result_async()
        self.get_result_future_1.add_done_callback(self.get_result_callback_1)

    def goal_response_callback_2(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Arm 2 Goal rejected :(')
            return
        # self.get_logger().info('Goal accepted :)')
        self.get_result_future_2 = goal_handle.get_result_async()
        self.get_result_future_2.add_done_callback(self.get_result_callback_2)

    def get_result_callback_1(self, future):
        result = future.result().result
        # self.get_logger().info(f'Result: {result}')

    def get_result_callback_2(self, future):
        result = future.result().result
        # self.get_logger().info(f'Result: {result}')

def main(args=None):
    rclpy.init(args=args)
    node = ModelBasedMPCNode()
    executor = MultiThreadedExecutor(num_threads=10)
    executor.add_node(node)
    executor.spin()

if __name__ == '__main__':
    main()