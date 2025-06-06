import sys, os, time, cv2, glob, shutil, re, random, threading

from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
from kortex_api.autogen.messages import Base_pb2
from kortex_api.autogen.messages import BaseCyclic_pb2
from movement_functions import *
from Gripper_Command import GripperCommand
    

def move_arm_thread1(base, base_cyclic, x, y, z, theta_x, theta_y, theta_z):
    move_arm(base, base_cyclic, x, y, z, theta_x, theta_y, theta_z)

def move_arm_thread2(base, base_cyclic, x, y, z, theta_x, theta_y, theta_z):
    move_arm(base, base_cyclic, x, y, z, theta_x, theta_y, theta_z)

def gripper_thread1(gripper, pos):
    gripper.Goto(pos)

def gripper_thread2(gripper, pos):
    gripper.Goto(pos)

def main():

    # Import the utilities helper module
    import argparse
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    import utilities

    # Parse arguments
    parser1 = argparse.ArgumentParser()
    parser2 = argparse.ArgumentParser()
    # parser.add_argument("--proportional_gain", type=float, help="proportional gain used in control loop", default=0.5) # 0.006
    args1 = utilities.parseConnectionArguments1(parser1)
    args2 = utilities.parseConnectionArguments2(parser2)

    # Create connection to the device and get the router
    with utilities.DeviceConnection.createTcpConnection(args1) as router1, utilities.DeviceConnection.createTcpConnection(args2) as router2:

        print("Connected to arms")
        gripper1 = GripperCommand(router1, 0.5) # 2f-85 gripper
        gripper2 = GripperCommand(router2, 0.5) # 2f-140 gripper
        # To avoid using impedance controller, we can measure before-hand a gripper width 
        # That causes the object to fall barely. then we can stop the measuring there.

        base1 = BaseClient(router1)
        base2 = BaseClient(router2)
        base_cyclic1 = BaseCyclicClient(router1)
        base_cyclic2 = BaseCyclicClient(router2)

        iter = 0.9

        while (iter >= 0.0):
            gripper2.Goto(iter)
            speed = gripper2.GetGripperVel()
            print("Current gripper speed is ", speed)
            iter -= 0.01

        time.sleep(2)
        speed = gripper2.GetGripperVel()
        print("Current gripper speed is ", speed)

        


if __name__ == "__main__":
    main()

# 75.877 makes jenga block fall
# 47.36 for pink ball