import sys, os, time
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
from kortex_api.autogen.messages import Base_pb2, BaseCyclic_pb2, Common_pb2
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import utilities
from movement_functions import *
from sensor_msgs.msg import Image
from rclpy.callback_groups import ReentrantCallbackGroup
from cv_bridge import CvBridge, CvBridgeError
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy



# Helper functions
def both_grippers_grasp_object(gripper_posi):
    # Close both grippers to a certain position
    pass

def is_slipping(gripper_width, external_force):
    # Determine slipping by threshold values based on gripper width and applied force
    # Maybe use effort field under /joint_states
    if gripper_width < threshold_width and external_force > threshold_force:
        return True
    return False

def label_slip(gripper_width, raw_image):
    # Label data points when slipping occurs
    data_label = 'slip'
    return label_data(gripper_width, raw_image, data_label)

def store_data(data):
    # Store or save the collected data for future training
    data_storage.save(data)

def reset_arms(base1, base_cyclic1, base2, base_cyclic2):
    # use send to home position from Kinova_Raptor driver
    print("Sending both arms to home position...")
    home_pos(base2, base_cyclic2)
    home_pos(base1, base_cyclic1)
    print("Both arms reached home position!")

    # Also open grippers all the way

def capture_raw_image():
    # Capture raw image from GelSight sensor mounted on follower gripper
    # Transform to cv image (easier to work with)
    cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

    return cv_image

def calculate_external_force(follower_arm, delta_x, delta_y):
    # Add effort fields of both arms
    return follower_arm.calculate_force(delta_x, delta_y)

def main():
    contact_group = ReentrantCallbackGroup()
    gs_qos_profile = QoSProfile(
            depth=0,  # Last 5 messages kept
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST
        )

    # Receives image from tactile sensor
    # contact_area_sub = create_subscription(
    #     Image, 
    #     '/gsmini_rawimg_0', 
    #     capture_raw_image,
    #     gs_qos_profile, 
    #     callback_group=contact_group
    # )
    
    # Parse arguments
    args1 = utilities.parseConnectionArguments1()
    args2 = utilities.parseConnectionArguments2()
    print("Parsed arguments")

    
    # Create connection to the device and get the router
    # Instead of using impedance, have both arms lined up. Follower arm will mirror behavior of leader arm
    # Set leader arm to follow specific trajectories
    # router1 = utilities.DeviceConnection.createTcpConnection(args1)
    # router2 = utilities.DeviceConnection.createTcpConnection(args2)
    print("Established router connection to both arms")

    with utilities.DeviceConnection.createTcpConnection(args1) as router1, \
        utilities.DeviceConnection.createTcpConnection(args2) as router2:

        # Create required services
        base1 = BaseClient(router1)
        base_cyclic1 = BaseCyclicClient(router1)

        base2 = BaseClient(router2)
        base_cyclic2 = BaseCyclicClient(router2)


        reset_arms(base1, base_cyclic1, base2, base_cyclic2)

        # # Data collection loop
        # for trial in range(NUM_TRIALS):
            
        #     # Step 1: Start grasp
        #     both_grippers_grasp_object()

        #     # Step 2: Randomly select incremental movement for leader arm
        #     delta_x = random_uniform(-35, 35)   # Random value within range for X
        #     delta_y = random_uniform(-21, 21)   # Random value within range for Y
            
        #     # Apply leader arm's incremental movement
        #     leader_arm.move(delta_x, delta_y)

        #     # Step 3: Follow leader arm's motion with follower arm
        #     follower_arm.follow_leader_motion()
            
        #     # Step 4: Monitor force applied on gripper and object
        #     external_force = calculate_external_force(follower_arm, delta_x, delta_y)
            
        #     # Step 5: Capture tactile feedback (raw images)
        #     while not slipping_occurred:
        #         # Record gripper width and raw tactile image at 60 Hz
        #         gripper_width = follower_gripper.get_width()
        #         raw_image = tactile_sensor.capture_raw_image()
                
        #         # Check for slipping condition
        #         if is_slipping(gripper_width, external_force):
        #             slipping_occurred = True
        #             slipping_time = get_current_time()
        #             label_slip(gripper_width, raw_image)  # Label all images with this slip event
        #             break

        #     # Step 6: Record data for the trial
        #     data = {
        #         'trial': trial,
        #         'gripper_width': gripper_width,
        #         'raw_image': raw_image,
        #         'external_force': external_force,
        #         'slip_event_time': slipping_time if slipping_occurred else None,
        #         'slip_label': pslip_value,  # For images where slip is detected
        #     }
        #     store_data(data)
            
        #     # Step 7: Relax gripper
        #     follower_gripper.relax_grip()

        #     # Step 8: Reset arms for next trial
        #     reset_arms(leader_arm, follower_arm)
            
        # End of data collection loop

if __name__ == "__main__":
    exit(main())