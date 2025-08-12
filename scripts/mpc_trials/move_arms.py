import sys
import os
import time
import threading
import numpy as np
from scipy.spatial.transform import Rotation as R

from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient


from kortex_api.autogen.messages import Base_pb2, BaseCyclic_pb2, Common_pb2

# Maximum allowed waiting time during actions (in seconds)
TIMEOUT_DURATION = 30

# Create closure to set an event after an END or an ABORT
def check_for_end_or_abort(e):
    """Return a closure checking for END or ABORT notifications

    Arguments:
    e -- event to signal when the action is completed
        (will be set when an END or ABORT occurs)
    """
    def check(notification, e = e):
        print("EVENT : " + \
              Base_pb2.ActionEvent.Name(notification.action_event))
        if notification.action_event == Base_pb2.ACTION_END \
        or notification.action_event == Base_pb2.ACTION_ABORT:
            e.set()
    return check
 
def example_move_to_home_position(base):
    # Make sure the arm is in Single Level Servoing mode
    base_servo_mode = Base_pb2.ServoingModeInformation()
    base_servo_mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
    base.SetServoingMode(base_servo_mode)
    
    # Move arm to ready position
    print("Moving the arm to a safe position")
    action_type = Base_pb2.RequestedActionType()
    action_type.action_type = Base_pb2.REACH_JOINT_ANGLES
    action_list = base.ReadAllActions(action_type)
    action_handle = None
    for action in action_list.action_list:
        if action.name == "Home":
            action_handle = action.handle

    if action_handle == None:
        print("Can't reach safe position. Exiting")
        return False

    e = threading.Event()
    notification_handle = base.OnNotificationActionTopic(
        check_for_end_or_abort(e),
        Base_pb2.NotificationOptions()
    )

    base.ExecuteActionFromReference(action_handle)
    finished = e.wait(TIMEOUT_DURATION)
    base.Unsubscribe(notification_handle)

    if finished:
        print("Safe position reached")
    else:
        print("Timeout on action notification wait")
    return finished
def populateCartesianCoordinate(waypointInformation):
    
    waypoint = Base_pb2.CartesianWaypoint()  
    waypoint.pose.x = waypointInformation[0]
    waypoint.pose.y = waypointInformation[1]
    waypoint.pose.z = waypointInformation[2]
    waypoint.blending_radius = waypointInformation[3]
    waypoint.pose.theta_x = waypointInformation[4]
    waypoint.pose.theta_y = waypointInformation[5]
    waypoint.pose.theta_z = waypointInformation[6] 
    waypoint.reference_frame = Base_pb2.CARTESIAN_REFERENCE_FRAME_TOOL
    
    return waypoint

def move(base, base_cyclic, waypoint):

    base_servo_mode = Base_pb2.ServoingModeInformation()
    base_servo_mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
    base.SetServoingMode(base_servo_mode)
    product = base.GetProductConfiguration()
    waypointsDefinition = waypoint
    if(   product.model == Base_pb2.ProductConfiguration__pb2.MODEL_ID_L53 
       or product.model == Base_pb2.ProductConfiguration__pb2.MODEL_ID_L31):
        pass
    else:
        print("Product is not compatible to run this example please contact support with KIN number bellow")
        print("Product KIN is : " + product.kin())

    waypoints = Base_pb2.WaypointList()
    
    waypoints.duration = 0.0
    waypoints.use_optimal_blending = False
    
    index = 0
    for waypointDefinition in waypointsDefinition:
        waypoint = waypoints.waypoints.add()
        waypoint.name = "waypoint_" + str(index)   
        waypoint.cartesian_waypoint.CopyFrom(populateCartesianCoordinate(waypointDefinition))
        index = index + 1 

    # Verify validity of waypoints
    result = base.ValidateWaypointList(waypoints)
    if(len(result.trajectory_error_report.trajectory_error_elements) == 0):
        e = threading.Event()
        notification_handle = base.OnNotificationActionTopic(check_for_end_or_abort(e),
                                                                Base_pb2.NotificationOptions())

        print("Moving cartesian trajectory...")
        
        base.ExecuteWaypointTrajectory(waypoints)

        print("Waiting for trajectory to finish ...")
        finished = e.wait(TIMEOUT_DURATION)
        base.Unsubscribe(notification_handle)

        if finished:
            print("Cartesian trajectory with no optimization completed ")
            e_opt = threading.Event()
            notification_handle_opt = base.OnNotificationActionTopic(check_for_end_or_abort(e_opt),
                                                                Base_pb2.NotificationOptions())

            waypoints.use_optimal_blending = True
            base.ExecuteWaypointTrajectory(waypoints)

            print("Waiting for trajectory to finish ...")
            finished_opt = e_opt.wait(TIMEOUT_DURATION)
            base.Unsubscribe(notification_handle_opt)

            if(finished_opt):
                print("Cartesian trajectory with optimization completed ")
            else:
                print("Timeout on action notification wait for optimized trajectory")

            return finished_opt
        else:
            print("Timeout on action notification wait for non-optimized trajectory")

        return finished
        
    else:
        print("Error found in trajectory") 
        result.trajectory_error_report.PrintDebugString();  

def get_target_state(filepath):
    target_posi = None
    target_rotation = None
    try:
        npz_keys = np.load(filepath, allow_pickle=True)
        print("Loaded npz file at path: ", filepath)

        print(f"NpzFile with keys: {list(npz_keys.keys())}")
        
        if "scores" in npz_keys.keys() and "pred_grasps_cam" in npz_keys.keys():
            # Get the value associated with the 'scores' key
            # This value is a list containing a dictionary
            scores_data_list = npz_keys["scores"]
            posi_pred_cam_list = npz_keys["pred_grasps_cam"]

            # Check if it's a list and if it contains a dictionary as expected
            if (isinstance(scores_data_list, np.ndarray) and scores_data_list.ndim == 0 and isinstance(scores_data_list.item(), dict)) and \
            (isinstance(posi_pred_cam_list, np.ndarray) and posi_pred_cam_list.ndim == 0 and isinstance(posi_pred_cam_list.item(), dict)):
                
                # Extract the dictionary from the 0-d array wrapper
                scores_dict = scores_data_list.item()
                posi_pred_cam_dict = posi_pred_cam_list.item()

                print("\nFinding max confidence value, its index, and the posi/rotation:")
                for key, score_array_value in scores_dict.items():
                    if isinstance(score_array_value, np.ndarray):
                        # Ensure it's at least 1-D for sorting
                        if score_array_value.ndim == 0:
                            score_array_value = np.array([score_array_value.item()])
                        current_max_confidence = np.max(score_array_value)
                        current_max_index = np.argmax(score_array_value)

                        print(f"For Key {key}:")
                        print(f"  Max confidence: {current_max_confidence}")
                        print(f"  Index of max confidence: {current_max_index}")

                        if key in posi_pred_cam_dict:
                            corresponding_grasp_array = posi_pred_cam_dict[key]
                            if isinstance(corresponding_grasp_array, np.ndarray) and corresponding_grasp_array.ndim >= 1:
                                grasp_info_at_max_confidence = corresponding_grasp_array[current_max_index]
                                print("  Position/rotation matrix at max confidence index (using tool as reference frame):")
                                print(grasp_info_at_max_confidence)

                                print("  Rotation after converting matrix to angles:")
                                rotation_obj = R.from_matrix(grasp_info_at_max_confidence[:3, :3])
                                euler_zyx_rad = rotation_obj.as_euler('zyx')
                                target_rotation = np.rad2deg(euler_zyx_rad)
                                print(target_rotation)

                                print("Position extracted from matrix:")
                                target_posi = [grasp_info_at_max_confidence[0,2], grasp_info_at_max_confidence[1,2], grasp_info_at_max_confidence[2,2]]
                                print(target_posi)

                    else:
                        print(f"Warning: Value for key {key} is not a NumPy array. Type: {type(score_array_value)}")

            else:
                print(f"The 'scores' key does not contain the expected list of dictionaries. Type: {type(scores_data_list)}, Dims: {getattr(scores_data_list, 'ndim', 'N/A')}")
                print(f"Content of 'scores' key: {scores_data_list}")

        else:
            print("\n'scores' key not found in the .npz file.")

        npz_keys.close()
        res = []
        res.extend(posi.item() for posi in target_posi)
        res.append(0.)
        res.extend(rot.item() for rot in target_rotation)
        return res


    except FileNotFoundError:
        print("Error: The file was not found. Please check the path.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        
def main():
    # Import the utilities helper module
    import argparse
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    import utilities

    # Parse arguments
    parser1 = argparse.ArgumentParser()
    parser2 = argparse.ArgumentParser()
    args1 = utilities.parseConnectionArguments(parser1)
    args2 = utilities.parseConnectionArguments(parser2)    
    # Create connection to the device and get the router
    with utilities.DeviceConnection.createTcpConnection(args1) as router1, utilities.DeviceConnection.createTcpConnection(args2) as router2:

        # Create required services
        base1 = BaseClient(router1)
        base_cyclic1 = BaseCyclicClient(router1)
        base2 = BaseClient(router2)
        base_cyclic2 = BaseCyclicClient(router2)
        

        # Example core
        success = True
        # Pretty sure waypoint definition is: poseX, poseY, poseZ, blendingRadius, kTheta_x, kTheta_y, kTheta_z
        success &= example_move_to_home_position(base1)
        success &= example_move_to_home_position(base2)
        
        arm_1_np_path = None
        arm_2_np_path = None
        arm_1_target = get_target_state(arm_1_np_path)
        arm_2_target = get_target_state(arm_2_np_path)
        success &= move(base1, base_cyclic1, arm_1_target)
        success &= move(base2, base_cyclic2, arm_2_target)
       
        return 0 if success else 1

if __name__ == "__main__":
    exit(main())