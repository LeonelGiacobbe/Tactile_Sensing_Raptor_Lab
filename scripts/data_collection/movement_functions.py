
import time
import threading

from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient

from kortex_api.autogen.messages import Base_pb2, BaseCyclic_pb2, Common_pb2


TIMEOUT_DURATION = 20 # seconds
TIMEOUT_GRIPPER = 0.3 # seconds
SPEED = 90.0 # degrees

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


def GripperCommand(base, base_cyclic, position = 0.0):
    # 0.0 = Fully open. 0.8 = fully closed


    # Create the GripperCommand we will send
    gripper_command = Base_pb2.GripperCommand()
    finger = gripper_command.gripper.finger.add()
    feedback = base_cyclic.RefreshFeedback()
    # Close the gripper with position increments
    print("Performing gripper test in position...")
    gripper_command.mode = Base_pb2.GRIPPER_POSITION
    finger.finger_identifier = 1
    finger.value = position
    print("Going to position {:0.2f}...".format(finger.value))
    base.SendGripperCommand(gripper_command)
    time.sleep(1)

    e = threading.Event() # Callback to tell us status of action topic, when it starts and stops moving
    notification_handle = base.OnNotificationActionTopic(
        check_for_end_or_abort(e),
        Base_pb2.NotificationOptions()
    )

    finished = e.wait(TIMEOUT_GRIPPER) # Sort of while loop to wait until the bot is done moving
    base.Unsubscribe(notification_handle) # After executing movement, notifications will not be useful anymore.

    if finished:
        print("Gripper movement completed")
    else:
        print("Timeout on action notification wait")
    return finished

def move_arm(base, base_cyclic, xMov, yMov, zMov, xTheta, yTheta, zTheta):
    print("Starting Arm Cartesian movement ...")
    action = Base_pb2.Action() # Create action object
    action.name = "Arm movement"
    action.application_data = ""

    feedback = base_cyclic.RefreshFeedback() # Provides current state of bot (pose_values, etc) at 1Khz
    cartesian_pose = action.reach_pose.target_pose # Specify tipe of action to execute. Reach target pose.
    
    # Cartesian coordinates
    cartesian_pose.x = feedback.base.tool_pose_x + xMov
    cartesian_pose.y = feedback.base.tool_pose_y + yMov
    cartesian_pose.z = feedback.base.tool_pose_z + zMov

    # Angular rotation
    cartesian_pose.theta_x = feedback.base.tool_pose_theta_x + xTheta
    cartesian_pose.theta_y = feedback.base.tool_pose_theta_y + yTheta
    cartesian_pose.theta_z = feedback.base.tool_pose_theta_z - zTheta

     # Callback to tell us status of action topic, when it starts and stops moving
    e = threading.Event()
    notification_handle = base.OnNotificationActionTopic(
        check_for_end_or_abort(e),
        Base_pb2.NotificationOptions()
    )

    print("Executing action")
    base.ExecuteAction(action)

    

    print("Waiting for movement to finish ...")
    finished = e.wait(TIMEOUT_DURATION) # Sort of while loop to wait until the bot is done moving
    base.Unsubscribe(notification_handle) # After executing movement, notifications will not be useful anymore.

    if finished:
        print("Cartesian movement completed")
    else:
        print("Timeout on action notification wait")
    return finished

def front_view(base, base_cyclic):
    print("Starting Cartesian action movement ... Front View")
    action = Base_pb2.Action() #Create action object 
    action.name = "Front View"
    action.application_data = ""

    feedback = base_cyclic.RefreshFeedback()
    cartesian_pose = action.reach_pose.target_pose
    cartesian_pose.x = 0  # (meters)
    cartesian_pose.y = 0 # (meters)
    cartesian_pose.z = 0   # (meters)
    cartesian_pose.theta_x = 0 # (degrees)
    cartesian_pose.theta_y = 0 # (degrees)
    cartesian_pose.theta_z = 0 # (degrees)

    e = threading.Event() # Callback to tell us status of action topic, when it starts and stops moving
    notification_handle = base.OnNotificationActionTopic(
        check_for_end_or_abort(e),
        Base_pb2.NotificationOptions()
    )

    print("Executing action")
    base.ExecuteAction(action)

    

    print("Waiting for movement to finish ...")
    finished = e.wait(TIMEOUT_DURATION) # Sort of while loop to wait until the bot is done moving
    base.Unsubscribe(notification_handle) # After executing movement, notifications will not be useful anymore.

    if finished:
        print("Cartesian movement completed")
    else:
        print("Timeout on action notification wait")
    return finished

def side_view(base, base_cyclic):
    print("Starting Cartesian action movement ... Side View")
    action = Base_pb2.Action() #Create action object 
    action.name = "Side View"
    action.application_data = ""

    feedback = base_cyclic.RefreshFeedback()
    cartesian_pose = action.reach_pose.target_pose
    cartesian_pose.x = 0  # (meters)
    cartesian_pose.y = 0 # (meters)
    cartesian_pose.z = 0 # (meters)
    cartesian_pose.theta_x = 0 # (degrees)
    cartesian_pose.theta_y = 0 # (degrees)
    cartesian_pose.theta_z = 0 # (degrees)

    e = threading.Event() # Callback to tell us status of action topic, when it starts and stops moving
    notification_handle = base.OnNotificationActionTopic(
        check_for_end_or_abort(e),
        Base_pb2.NotificationOptions()
    )

    print("Executing action")
    base.ExecuteAction(action)

    

    print("Waiting for movement to finish ...")
    finished = e.wait(TIMEOUT_DURATION) # Sort of while loop to wait until the bot is done moving
    base.Unsubscribe(notification_handle) # After executing movement, notifications will not be useful anymore.

    if finished:
        print("Cartesian movement completed")
    else:
        print("Timeout on action notification wait")
    return finished    

def start_pos(base, base_cyclic):
    print("Starting Cartesian action movement ...")
    action = Base_pb2.Action() # Create action object
    action.name = "Start Position"
    action.application_data = ""

    feedback = base_cyclic.RefreshFeedback() # Provides current state of bot (pose_values, etc) at 1Khz
    cartesian_pose = action.reach_pose.target_pose # Specify tipe of action to execute. Reach target pose.
    cartesian_pose.x = 0.45         # (meters)
    cartesian_pose.y = -0.07  # (meters)
    cartesian_pose.z = 0.252   # (meters)
    cartesian_pose.theta_x = 176.5 # (degrees)
    cartesian_pose.theta_y = -2.9 # (degrees)
    cartesian_pose.theta_z = 90.0 # (degrees)

    e = threading.Event() # Callback to tell us status of action topic, when it starts and stops moving
    notification_handle = base.OnNotificationActionTopic(
        check_for_end_or_abort(e),
        Base_pb2.NotificationOptions()
    )

    print("Executing action")
    base.ExecuteAction(action)

    

    print("Waiting for movement to finish ...")
    finished = e.wait(TIMEOUT_DURATION) # Sort of while loop to wait until the bot is done moving
    base.Unsubscribe(notification_handle) # After executing movement, notifications will not be useful anymore.

    if finished:
        print("Cartesian movement completed")
    else:
        print("Timeout on action notification wait")
    return finished

def place_block(base, base_cyclic, horizontal, turn, height, hollow):
    print("Moving block to tower ...")
    action = Base_pb2.Action() # Create action object
    action.name = "Block to tower"
    action.application_data = ""
    
    # Offset to modify position of arm after each turn
    if hollow:
        xLoc = 0.0824 * ((turn - 1) % 2)
    else:
        xLoc = 0.0412 * ((turn - 1) % 3)

    feedback = base_cyclic.RefreshFeedback() # Provides current state of bot (pose_values, etc) at 1Khz
    cartesian_pose = action.reach_pose.target_pose # Specify tipe of action to execute. Reach target pose.
    cartesian_pose.x = 0.425 + xLoc      # (meters)
    cartesian_pose.y = 0.19   # (meters)
    cartesian_pose.z = 0.15   # (meters) 
    cartesian_pose.theta_x = 180.0 # (degrees)
    cartesian_pose.theta_y = 0.0 # (degrees)
    cartesian_pose.theta_z = 90.0 # (degrees)

    # Rotate bot 90 degrees around z axis
    if not horizontal:
        cartesian_pose.theta_z = 180.0 # (degrees)
    
    e = threading.Event() # Callback to tell us status of action topic, when it starts and stops moving
    notification_handle = base.OnNotificationActionTopic(
        check_for_end_or_abort(e),
        Base_pb2.NotificationOptions()
    )

    print("Executing block placement")
    base.ExecuteAction(action)
    

    print("Waiting for movement to finish ...")
    finished = e.wait(TIMEOUT_DURATION) # Sort of while loop to wait until the bot is done moving
    base.Unsubscribe(notification_handle) # After executing movement, notifications will not be useful anymore.

    # If arm is not in dropping position, execute drop
    if feedback.base.tool_pose_z > 0.12:
        if horizontal:
            drop_horizontal(base, base_cyclic, height)
        else:
            drop_vertical(base, base_cyclic, turn, height, hollow)

    if finished:
        print("Block placement completed")
    else:
        print("Timeout on action notification wait")
    return finished

def v2_place_block(base, base_cyclic, horizontal, turn, height, hollow):
    print("Moving block to tower ...")
    action = Base_pb2.Action() # Create action object
    action.name = "Block to tower"
    action.application_data = ""
    
    # Offset to modify position of arm after each turn
    if hollow:
        xLoc = 0.0824 * ((turn - 1) % 2)
    else:
        xLoc = 0.0412 * ((turn - 1) % 3)

    feedback = base_cyclic.RefreshFeedback() # Provides current state of bot (pose_values, etc) at 1Khz
    cartesian_pose = action.reach_pose.target_pose # Specify tipe of action to execute. Reach target pose.
    cartesian_pose.x = 0.5074 - xLoc      # (meters)
    cartesian_pose.y = 0.19   # (meters)
    cartesian_pose.z = 0.15   # (meters) 
    cartesian_pose.theta_x = 180.0 # (degrees)
    cartesian_pose.theta_y = 0.0 # (degrees)
    cartesian_pose.theta_z = 90.0 # (degrees)

    # Rotate bot 90 degrees around z axis
    if not horizontal:
        cartesian_pose.theta_z = 180.0 # (degrees)
    
    e = threading.Event() # Callback to tell us status of action topic, when it starts and stops moving
    notification_handle = base.OnNotificationActionTopic(
        check_for_end_or_abort(e),
        Base_pb2.NotificationOptions()
    )

    print("Executing block placement")
    base.ExecuteAction(action)
    

    print("Waiting for movement to finish ...")
    finished = e.wait(TIMEOUT_DURATION) # Sort of while loop to wait until the bot is done moving
    base.Unsubscribe(notification_handle) # After executing movement, notifications will not be useful anymore.

    # If arm is not in dropping position, execute drop
    if feedback.base.tool_pose_z > 0.12:
        if horizontal:
            drop_horizontal(base, base_cyclic, height)
        else:
            v2_drop_vertical(base, base_cyclic, turn, height, hollow)

    if finished:
        print("Block placement completed")
    else:
        print("Timeout on action notification wait")
    return finished

def drop_horizontal(base, base_cyclic, height):
    print("Moving block to tower ...")
    action = Base_pb2.Action() # Create action object
    action.name = "Block to tower"
    action.application_data = ""

    feedback = base_cyclic.RefreshFeedback() # Provides current state of bot (pose_values, etc) at 1Khz
    cartesian_pose = action.reach_pose.target_pose # Specify tipe of action to execute. Reach target pose.
    cartesian_pose.x = feedback.base.tool_pose_x       # (meters)
    cartesian_pose.y = 0.19    # (meters)
    cartesian_pose.z = 0.095 + height # (meters) 
    cartesian_pose.theta_x = 180.0 # (degrees)
    cartesian_pose.theta_y = 0.0 # (degrees)
    cartesian_pose.theta_z = 90.0 # (degrees)
    
    
    

    e = threading.Event() # Callback to tell us status of action topic, when it starts and stops moving
    notification_handle = base.OnNotificationActionTopic(
        check_for_end_or_abort(e),
        Base_pb2.NotificationOptions()
    )

    print("Executing block placement")
    base.ExecuteAction(action)
    
    print("Waiting for movement to finish ...")
    finished = e.wait(TIMEOUT_DURATION) # Sort of while loop to wait until the bot is done moving
    base.Unsubscribe(notification_handle) # After executing movement, notifications will not be useful anymore.

    if finished:
        print("Block placement completed")
    else:
        print("Timeout on action notification wait")
    return finished

def twist_wrist(base, reverse=False):
    joint_speeds = Base_pb2.JointSpeeds()

    actuator_count = base.GetActuatorCount().count
    if not reverse:
        speeds = [0, 0, 0, 0, 0, 0, SPEED]
    else:
        speeds = [0, 0, 0, 0, 0, 0, -SPEED]
    i = 0
    for speed in speeds:
        joint_speed = joint_speeds.joint_speeds.add()
        joint_speed.joint_identifier = i 
        joint_speed.value = speed
        i = i + 1
    print ("Sending the joint speeds for 10 seconds...")
    base.SendJointSpeedsCommand(joint_speeds)
    time.sleep(2)
    base.Stop()

    return True

def drop_vertical(base, base_cyclic, turn, height, hollow):

    print("Moving block to tower ...")
    action = Base_pb2.Action() # Create action object
    action.name = "Block to tower"
    action.application_data = ""

    if hollow:
        yLoc = 0.0839 * ((turn - 1) % 2)
    else:
        yLoc = 0.0412 * ((turn - 1) % 3)

    feedback = base_cyclic.RefreshFeedback() # Provides current state of bot (pose_values, etc) at 1Khz
    cartesian_pose = action.reach_pose.target_pose # Specify tipe of action to execute. Reach target pose.
    cartesian_pose.x = 0.462      # (meters)
    cartesian_pose.y = 0.153  + yLoc   # (meters)
    cartesian_pose.z = 0.095 + height  # (meters) 
    cartesian_pose.theta_x = 180 # (degrees)
    cartesian_pose.theta_y = 0.0 # (degrees)
    cartesian_pose.theta_z = 180.0 # (degrees)

    e = threading.Event() # Callback to tell us status of action topic, when it starts and stops moving
    notification_handle = base.OnNotificationActionTopic(
        check_for_end_or_abort(e),
        Base_pb2.NotificationOptions()
    )

    print("Executing block placement")
    base.ExecuteAction(action)

    print("Waiting for movement to finish ...")
    finished = e.wait(TIMEOUT_DURATION) # Sort of while loop to wait until the bot is done moving
    base.Unsubscribe(notification_handle) # After executing movement, notifications will not be useful anymore.

    

    if finished:
        print("Block placement completed")
    else:
        print("Timeout on action notification wait")
    return finished

def v2_drop_vertical(base, base_cyclic, turn, height, hollow):

    print("Moving block to tower ...")
    action = Base_pb2.Action() # Create action object
    action.name = "Block to tower"
    action.application_data = ""

    if hollow:
        yLoc = 0.0839 * ((turn - 1) % 2)
    else:
        yLoc = 0.0412 * ((turn - 1) % 3)

    feedback = base_cyclic.RefreshFeedback() # Provides current state of bot (pose_values, etc) at 1Khz
    cartesian_pose = action.reach_pose.target_pose # Specify tipe of action to execute. Reach target pose.
    cartesian_pose.x = 0.462      # (meters)
    cartesian_pose.y = 0.237  - yLoc   # (meters)
    cartesian_pose.z = 0.095 + height  # (meters) 
    cartesian_pose.theta_x = 180 # (degrees)
    cartesian_pose.theta_y = 0.0 # (degrees)
    cartesian_pose.theta_z = 180.0 # (degrees)

    e = threading.Event() # Callback to tell us status of action topic, when it starts and stops moving
    notification_handle = base.OnNotificationActionTopic(
        check_for_end_or_abort(e),
        Base_pb2.NotificationOptions()
    )

    print("Executing block placement")
    base.ExecuteAction(action)

    print("Waiting for movement to finish ...")
    finished = e.wait(TIMEOUT_DURATION) # Sort of while loop to wait until the bot is done moving
    base.Unsubscribe(notification_handle) # After executing movement, notifications will not be useful anymore.

    

    if finished:
        print("Block placement completed")
    else:
        print("Timeout on action notification wait")
    return finished