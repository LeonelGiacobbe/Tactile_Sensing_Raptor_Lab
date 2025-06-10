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
import threading
import torch
from torchvision import transforms
from PIL import Image
from .implemented_functions import ResCNNEncoder, MPClayer
from cv_bridge import CvBridge
import os, time
from ament_index_python.packages import get_package_share_directory

CONVERSION_RATE_140_MM = 0.00571429 # Kinova unit to mm
CONVERSION_RATE_85_MM = 0.00941176 # Kinova unit to mm

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
        
        self.gripper_posi_1 = 0.0
        self.gripper_vel_1 = 0.0
        self.gripper_posi_2 = 0.0
        self.gripper_vel_2 = 0.0
        self.processing_executor = ThreadPoolExecutor(max_workers=1)
        self.contact_area_lock = threading.Lock()
        self.frequency = 10

        # # Neural network stuff
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.get_logger().info(f"Using {self.device} in controller node")
        # self.nn_encoder = ResCNNEncoder().to(self.device)
        # self.mpc_layer = MPClayer().to(self.device)
        # if self.device.type == 'cuda':
        #     self.stream = torch.cuda.Stream()
        # self.nn_encoder.eval()
        # self.mpc_layer.eval()

        # # Warmup pass
        # dummy_input = torch.randn(1, 3, 224, 224, device=self.device)
        # _ = self.nn_encoder(dummy_input)
        # if self.device.type == 'cuda':
        #     torch.cuda.synchronize()

        # # Load weights
        # package_dir = get_package_share_directory('tactile_sensing')
        # model_path = os.path.join(package_dir, 'models', 'letac_mpc_model.pth')
        # checkpoint = torch.load(model_path, map_location=torch.device(self.device))
        # self.nn_encoder.load_state_dict(checkpoint['cnn_encoder_state_dict'])
        # self.mpc_layer.load_state_dict(checkpoint['mpc_layer_state_dict'])
        # self.get_logger().info("Loaded MPC weights")

        # Image preprocessing (using the same transform as in training of the CNN encoder)
        self.transform = transforms.Compose([transforms.Resize([224, 224]),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0, 0, 0], std=[0.2, 0.2, 0.2])])
        
        # To convert GelSight images from ROS Image to CV Image
        self.bridge = CvBridge()

        # Gelsight image vars
        self.current_image_1 = None
        self.current_image_2 = None

        self.rate = self.create_rate(self.frequency)

        # Timer to call the run method periodically
        self.timer = self.create_timer(1.0 / self.frequency, self.run)

        gs_qos_profile = QoSProfile(
            depth=0,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST
        )   

        # Using Kinova arm, and we send gripper posi commands through Action Server
        # WILL NEED TO DO THIS TWICE
        self._own_action_client = ActionClient(self, GripperCommand, '/robotiq_gripper_controller/gripper_cmd')
        # self._other_action_client = ...
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
        
        # Subscribe to the JointState topic to get gripper position
        posi_qos_profile = QoSProfile(
            depth=0,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE  # Change to TRANSIENT_LOCAL
        )
        
        # WILL NEED TO DO THIS TWICE
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_cb,
            posi_qos_profile,
            callback_group=self.callback_group
        )

    # Receives position of gripper: 0.0 -> completely open. 0.8 -> completely closed        
    def joint_state_cb(self, msg: JointState):
        # self.get_logger().debug("Entered joint_state_cb")  # Debug entry
        try:
            self.get_logger().debug(f"Received joints: {msg.name}")  # Debug joint names
            
            if 'arm_1_robotiq_85_left_knuckle_joint' in msg.name:
                index = msg.name.index('arm_1_robotiq_85_left_knuckle_joint')
                gripper_position = msg.position[index]

                self.gripper_posi_1 = gripper_position
                self.get_logger().info(f"Current arm 1 gripper position: {gripper_position:.4f}")
                
                gripper_vel = msg.velocity[index]
                self.gripper_vel_1 = gripper_vel
                # self.get_logger().info(f"Current gripper vel: {gripper_vel:.4f}")  # Debug velocity
            elif 'arm_2_robotiq_85_left_knuckle_joint' in msg.name:
                index = msg.name.index('arm_2_robotiq_85_left_knuckle_joint')
                gripper_position = msg.position[index]

                self.gripper_posi_2 = gripper_position
                self.get_logger().info(f"Current arm 2 gripper position: {gripper_position:.4f}")
                
                gripper_vel = msg.velocity[index]
                self.gripper_vel_2 = gripper_vel
                
            else:
                self.get_logger().warn("Gripper joint not found in JointState message")
                
        except Exception as e:
            self.get_logger().error(f"joint_state_cb error: {str(e)}", throttle_duration_sec=5)

        
    # Not currently used (original author said it can be set to zero?) but here just in case
    def dis_sum_cb(self, msg):
        self.dis_sum_ = msg.data        
        
    def contact_area_cb_1(self, msg):
        start = time.time()
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg)   
            # cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)  
            pil_image = Image.fromarray(cv_image)        
            # Convert to tensor and process single image
            tensor = self.transform(pil_image).to(self.device)
            
            with self.contact_area_lock:
                self.current_image_1 = tensor  # Store single image
            stop = time.time()
            # self.get_logger().info(f"Image processing runtime: {stop - start}")

        except Exception as e:
            self.get_logger().error(f'Tactile processing failed: {str(e)}')

    def contact_area_cb_2(self, msg):
        start = time.time()
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg)   
            # cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)  
            pil_image = Image.fromarray(cv_image)        
            # Convert to tensor and process single image
            tensor = self.transform(pil_image).to(self.device)
            
            with self.contact_area_lock:
                self.current_image_2 = tensor  # Store single image
            stop = time.time()
            # self.get_logger().info(f"Image processing runtime: {stop - start}")

        except Exception as e:
            self.get_logger().error(f'Tactile processing failed: {str(e)}')
        
    def run(self):
        if self.current_image_1 is None or self.current_image_2 is None:
            return  # Not enough data

        try:
            # Prepare tensors
            with torch.no_grad():
                own_image_tensor = self.current_image_1.unsqueeze(0)
                other_image_tensor = self.current_image_2.unsqueeze(0)
                # Kinova uses a custom scale (see gripper posi callback for details), here we convert to mm
                own_gripper_p = torch.tensor([gripper_posi_to_mm_140(self.gripper_posi_1)]).to(self.device)
                own_gripper_v = torch.tensor(self.gripper_vel_1).to(self.device)

                other_gripper_p = torch.tensor([gripper_posi_to_mm_85(self.gripper_posi_2)]).to(self.device)
                other_gripper_v = torch.tensor(self.gripper_vel_2).to(self.device)

                own_tactile_embeddings = self.nn_encoder(own_image_tensor) # 0.16s spent here
                other_tactile_embeddings = self.nn_encoder(other_image_tensor)
                
                pos_sequences = self.mpc_layer(own_tactile_embeddings, own_gripper_p, own_gripper_v) # 0.6s here
                # own_output, other_output = self.mpc_layer(own_embeddings, other_embeddings, own_gripper_p, own_gripper_v, other_gripper_p, other_gripper_v, )

            # Take the first action in the horizon
            own_target_pos = pos_sequences[:, 0].item() # Now in mm
            own_target_pos = mm_to_gripper_posi_140(own_target_pos) # Now converted to kinova scale

            # other_target_pos = other_output[:, 0].item()
            # other_target_pos = mm_to_gripper_posi_85(other_target_pos)
            
            self.get_logger().info(f"Target pos sequence: {pos_sequences}")
            
            # Send own command
            self.own_goal = GripperCommand.Goal()
            self.own_goal.command.position = own_target_pos
            self.own_goal.command.max_effort = 100.0
            self.get_logger().info(f"Sending own goal: {self.own_goal.command.position:.4f}")
            self._own_send_goal(self.own_goal)

            # Send own command
            # self.other_goal = GripperCommand.Goal()
            # self.other_goal.command.position = other_target_pos
            # self.other_goal.command.max_effort = 100.0
            # self.get_logger().info(f"Sending other goal: {self.other_goal.command.position:.4f}")
            # self._other_send_goal(self.other_goal)
            # # self.get_logger().info(f"Current gripper posi: {self.gripper_posi_}")
            # self.rate.sleep()


        except Exception as e:
            self.get_logger().error(f'Control loop failed: {str(e)}')

    def _own_send_goal(self, goal):
        if not self._own_action_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().error("Action server not available after waiting")
            return
        self.get_logger().info("Got life check from server")
        self._own_send_goal_future = self._own_action_client.send_goal_async(goal)
        self._own_send_goal_future.add_done_callback(self._own_goal_response_callback)

    def _other_send_goal(self, goal):
        if not self._own_action_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().error("Action server not available after waiting")
            return
        self.get_logger().info("Got life check from server")
        self._other_send_goal_future = self._other_action_client.send_goal_async(goal)
        self._other_send_goal_future.add_done_callback(self._other_goal_response_callback)

    def _own_goal_response_callback(self, future):
        own_goal_handle = future.result()
        if not own_goal_handle.accepted:
            self.get_logger().info('Own Goal rejected :(')
            return
        # self.get_logger().info('Goal accepted :)')
        self._own_get_result_future = own_goal_handle.get_result_async()
        self._own_get_result_future.add_done_callback(self._own_get_result_callback)

    def _other_goal_response_callback(self, future):
        other_goal_handle = future.result()
        if not other_goal_handle.accepted:
            self.get_logger().info('Other Goal rejected :(')
            return
        # self.get_logger().info('Goal accepted :)')
        self._other_get_result_future = other_goal_handle.get_result_async()
        self._other_get_result_future.add_done_callback(self._other_get_result_callback)

    def _own_get_result_callback(self, future):
        result = future.result().result
        # self.get_logger().info(f'Result: {result}')

    def _other_get_result_callback(self, future):
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