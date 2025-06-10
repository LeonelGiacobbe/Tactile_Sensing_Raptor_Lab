import sys, os, threading
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
from kortex_api.autogen.messages import Base_pb2, BaseCyclic_pb2, Common_pb2

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

def move_start_pos1(base, base_cyclic):
    print("Starting Cartesian action movement ...")
    action = Base_pb2.Action() # Create action object
    action.name = "Start Position"
    action.application_data = ""

    feedback = base_cyclic.RefreshFeedback() # Provides current state of bot (pose_values, etc) at 1Khz
    cartesian_pose = action.reach_pose.target_pose # Specify tipe of action to execute. Reach target pose.
    cartesian_pose.x = 0.369         # (meters)
    cartesian_pose.y = -0.053 # (meters)
    cartesian_pose.z = 0.268   # (meters)
    cartesian_pose.theta_x = 0.0 # (degrees)
    cartesian_pose.theta_y = -180.0 # (degrees)
    cartesian_pose.theta_z = 0.0 # (degrees)

    e = threading.Event() # Callback to tell us status of action topic, when it starts and stops moving
    notification_handle = base.OnNotificationActionTopic(
        check_for_end_or_abort(e),
        Base_pb2.NotificationOptions()
    )

    print("Executing action")
    base.ExecuteAction(action)

    

    print("Waiting for movement to finish ...")
    finished = e.wait(200) # Sort of while loop to wait until the bot is done moving
    base.Unsubscribe(notification_handle) # After executing movement, notifications will not be useful anymore.

    if finished:
        print("Cartesian movement completed")
    else:
        print("Timeout on action notification wait")
    return finished

def move_start_pos2(base, base_cyclic):
    print("Starting Cartesian action movement ...")
    action = Base_pb2.Action() # Create action object
    action.name = "Start Position"
    action.application_data = ""

    feedback = base_cyclic.RefreshFeedback() # Provides current state of bot (pose_values, etc) at 1Khz
    cartesian_pose = action.reach_pose.target_pose # Specify tipe of action to execute. Reach target pose.
    cartesian_pose.x = 0.369         # (meters)
    cartesian_pose.y = 0.219 # (meters)
    cartesian_pose.z = 0.15   # (meters)
    cartesian_pose.theta_x = 0.0 # (degrees)
    cartesian_pose.theta_y = 180.0 # (degrees)
    cartesian_pose.theta_z = 180.0 # (degrees)

    e = threading.Event() # Callback to tell us status of action topic, when it starts and stops moving
    notification_handle = base.OnNotificationActionTopic(
        check_for_end_or_abort(e),
        Base_pb2.NotificationOptions()
    )

    print("Executing action")
    base.ExecuteAction(action)

    

    print("Waiting for movement to finish ...")
    finished = e.wait(200) # Sort of while loop to wait until the bot is done moving
    base.Unsubscribe(notification_handle) # After executing movement, notifications will not be useful anymore.

    if finished:
        print("Cartesian movement completed")
    else:
        print("Timeout on action notification wait")
    return finished

def main():
    success = True
    # Import the utilities helper module
    import argparse
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    import utilities

    # Parse arguments
    parser1 = argparse.ArgumentParser()
    parser2 = argparse.ArgumentParser()

    args1 = utilities.parseConnectionArguments1(parser1)
    args2 = utilities.parseConnectionArguments2(parser2)

    # Create connection to the device and get the router
    with utilities.DeviceConnection.createTcpConnection(args1) as router1, utilities.DeviceConnection.createTcpConnection(args2) as router2:
        base1 = BaseClient(router1)
        base_cyclic1 = BaseCyclicClient(router1)

        base2 = BaseClient(router2)
        base_cyclic2 = BaseCyclicClient(router2)

        success &= move_start_pos1(base1, base_cyclic1)
        #success &= move_start_pos2(base2, base_cyclic2)


if __name__ == "__main__":
    main()