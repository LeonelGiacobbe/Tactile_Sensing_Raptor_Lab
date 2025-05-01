#! /usr/bin/env python3

import sys
import os
import time
import threading
import numpy as np
from kortex_api.autogen.client_stubs.ActuatorConfigClientRpc import ActuatorConfigClient
from kortex_api.autogen.client_stubs.ActuatorCyclicClientRpc import ActuatorCyclicClient
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
from kortex_api.autogen.client_stubs.DeviceConfigClientRpc import DeviceConfigClient
from kortex_api.autogen.client_stubs.DeviceManagerClientRpc import DeviceManagerClient
from kortex_api.autogen.messages import Session_pb2, ActuatorConfig_pb2, Base_pb2, BaseCyclic_pb2, Common_pb2
from kortex_api.RouterClient import RouterClientSendOptions

class ImpedanceController:
    def __init__(self, base, base_cyclic, actuator_config, control_config, K, B, M):
        self.base = base
        self.base_cyclic = base_cyclic
        self.actuator_config = actuator_config
        self.control_config = control_config
        self.damping = B 
        self.stiffness = K 
        self.mass_matrix = M  

        # State initialization
        self.joint_count = 7
        self.current_position = np.zeros(self.joint_count)
        self.current_velocity = np.zeros(self.joint_count)

        # Initialize command structure
        for _ in range(self.joint_count): 
            self.base_command = BaseCyclic_pb2.Command()
            self.base_command.actuators.add()

        # Set control modes
        servoing_mode = Base_pb2.ServoingModeInformation()
        servoing_mode.servoing_mode = Base_pb2.LOW_LEVEL_SERVOING
        self.base.SetServoingMode(servoing_mode)

        # Set all actuators to torque mode initially
        control_mode = ActuatorConfig_pb2.ControlModeInformation()
        control_mode.control_mode = ActuatorConfig_pb2.ControlMode.Value('TORQUE')
        for actuator_id in range(1, 8):  # IDs 1-7
            self.actuator_config.SetControlMode(control_mode, actuator_id)

    def update_state(self):
        feedback = self.base_cyclic.RefreshFeedback() # Provides current state of bot (pose_values, etc) at 1Khz 
        self.current_position = np.array([feedback.actuators[i].position for i in range(self.joint_count)])
        self.current_velocity = np.array([feedback.actuators[i].velocity for i in range(self.joint_count)])

    def control(self, q_des, v_des, compensate):
        """
        Calculate torque command using impedance control law for the last 3 joints.
        q_des: Desired positions (shape: (3,))
        v_des: Desired velocities (shape: (3,))
        compensate: Gravity compensation or other forces (shape: (3,))
        """
        # Extract current velocity for the last 3 joints (5, 6, 7)
        current_velocity_last_3 = self.current_velocity[4:7]  # Only last 3 joints (5, 6, 7)

        # Ensure that the damping applies only to the last 3 joints
        damping_last_3 = self.damping[4:7]  # Get the damping for joints 5, 6, and 7

        # Compute the desired acceleration
        acc_des = damping_last_3 * (v_des - current_velocity_last_3) + \
                self.stiffness[4:7] * (q_des - self.current_position[4:7])  # Only last 3 joints (5, 6, 7)
        
        # Compute the torque for the last 3 joints
        tau = np.dot(self.mass_matrix[4:7, 4:7], acc_des.T).T + compensate
        return tau



    def send_torque_command(self, torques):
        for i in range(1, 7):
            self.base_command.actuators[i].torque_joint = torques[i]
        self.base_cyclic.Refresh(self.base_command)

class TorqueExample:
    def __init__(self, router, router_real_time):
        self.ACTION_TIMEOUT_DURATION = 20
        self.torque_amplification = 2.0  # Torque measure on last actuator is sent as a command to first actuator

        # Create required services
        device_manager = DeviceManagerClient(router)
        self.actuator_config = ActuatorConfigClient(router)
        self.base = BaseClient(router)
        self.base_cyclic = BaseCyclicClient(router_real_time)

        self.base_command = BaseCyclic_pb2.Command()
        self.base_feedback = BaseCyclic_pb2.Feedback()
        self.base_custom_data = BaseCyclic_pb2.CustomData()

        device_handles = device_manager.ReadAllDevices()
        self.actuator_count = self.base.GetActuatorCount().count
        for handle in device_handles.device_handle:
            if handle.device_type == Common_pb2.BIG_ACTUATOR or handle.device_type == Common_pb2.SMALL_ACTUATOR:
                self.base_command.actuators.add()
                self.base_feedback.actuators.add()

        # Change send option to reduce max timeout at 3ms
        self.sendOption = RouterClientSendOptions()
        self.sendOption.andForget = False
        self.sendOption.delay_ms = 0
        self.sendOption.timeout_ms = 3

        self.cyclic_t_end = 30  #Total duration of the thread in seconds.
        self.cyclic_thread = {}

        self.kill_the_thread = False
        self.already_stopped = False
        self.cyclic_running = False

    def check_for_end_or_abort(self, e):
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

    def MoveToHomePosition(self):
         # Make sure the arm is in Single Level Servoing mode
        base_servo_mode = Base_pb2.ServoingModeInformation()
        base_servo_mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
        self.base.SetServoingMode(base_servo_mode)
    
        # Move arm to ready position
        print("Moving the arm to a safe position")
        action_type = Base_pb2.RequestedActionType()
        action_type.action_type = Base_pb2.REACH_JOINT_ANGLES
        action_list = self.base.ReadAllActions(action_type)
        action_handle = None
        for action in action_list.action_list:
            if action.name == "Home":
                action_handle = action.handle

        if action_handle == None:
            print("Can't reach safe position. Exiting")
            return False

        e = threading.Event()
        notification_handle = self.base.OnNotificationActionTopic(
            self.check_for_end_or_abort(e),
            Base_pb2.NotificationOptions()
        )

        self.base.ExecuteActionFromReference(action_handle)

        print("Waiting for movement to finish ...")
        finished = e.wait(self.ACTION_TIMEOUT_DURATION)
        self.base.Unsubscribe(notification_handle)

        if finished:
            print("Cartesian movement completed")
        else:
            print("Timeout on action notification wait")
        return finished

        return True

    def InitCyclic(self, sampling_time_cyclic, t_end, print_stats):
        if self.cyclic_running:
            return True

        # Move to Home position first
        print("Before moving to home position")
        if not self.MoveToHomePosition():
            return False

        base_feedback = self.SendCallWithRetry(self.base_cyclic.RefreshFeedback, 3)
        if base_feedback:
            self.base_feedback = base_feedback
            for x in range(self.actuator_count):
                self.base_command.actuators[x].flags = 1  # servoing
                self.base_command.actuators[x].position = self.base_feedback.actuators[x].position

            # Set arm in LOW_LEVEL_SERVOING
            base_servo_mode = Base_pb2.ServoingModeInformation()
            base_servo_mode.servoing_mode = Base_pb2.LOW_LEVEL_SERVOING
            self.base.SetServoingMode(base_servo_mode)

            self.base_feedback = self.base_cyclic.Refresh(self.base_command, 0, self.sendOption)
            self.cyclic_t_end = t_end
            self.cyclic_thread = threading.Thread(target=self.RunCyclic, args=(sampling_time_cyclic, print_stats))
            self.cyclic_thread.daemon = True
            self.cyclic_thread.start()
            return True
        else:
            print("InitCyclic: failed to communicate")
            return False

    def RunCyclic(self, t_sample, print_stats):
        self.cyclic_running = True
        cyclic_count = 0  # Counts refresh
        stats_count = 0  # Counts stats prints
        failed_cyclic_count = 0  # Count communication timeouts
        t_now = time.time()
        t_cyclic = t_now  # cyclic time

        print("Running torque control example for {} seconds".format(self.cyclic_t_end))

        # Create the Impedance Controller instance (apply it only on last 3 joints)
        K = 100.0 * np.ones(7)
        B = 20.0 * np.ones(7)
        M = np.eye(7)  # Example mass matrix, adjust as needed
        impedance_controller = ImpedanceController(self.base, self.base_cyclic, self.actuator_config, None, K, B, M)

        while not self.kill_the_thread:
            t_now = time.time()

            if (t_now - t_cyclic) >= t_sample:
                t_cyclic = t_now

                # Send torque commands for the first 4 joints
                for i in range(4):  # joints 1 to 4 in torque control
                    self.base_command.actuators[i].torque_joint = self.base_feedback.actuators[i].torque

                # Impedance control for last 3 joints (5 to 7)
                q_des = np.array([self.base_feedback.actuators[i].position for i in range(4, 7)])  # desired position
                v_des = np.zeros(3)  # desired velocity is 0 (static)
                compensate = np.zeros(3)  # gravity compensation or other factors
                torques = impedance_controller.control(q_des, v_des, compensate)

                # Apply impedance torques for the last 3 joints
                for i in range(4, 7):
                    self.base_command.actuators[i].torque_joint = torques[i - 4]

                # Refresh command
                self.base_command.frame_id += 1
                if self.base_command.frame_id > 65535:
                    self.base_command.frame_id = 0
                for i in range(self.actuator_count):
                    self.base_command.actuators[i].command_id = self.base_command.frame_id

                try:
                    self.base_feedback = self.base_cyclic.Refresh(self.base_command, 0, self.sendOption)
                except:
                    failed_cyclic_count += 1
                cyclic_count += 1

            if self.cyclic_t_end != 0 and (t_now - time.time() > self.cyclic_t_end):
                print("Cyclic Finished")
                break
        self.cyclic_running = False
        return True

    def StopCyclic(self):
        print("Stopping the cyclic and putting the arm back in position mode...")
        if self.already_stopped:
            return
        if self.cyclic_running:
            self.kill_the_thread = True
            self.cyclic_thread.join()
        
        # Set first actuator back in position mode
        control_mode_message = ActuatorConfig_pb2.ControlModeInformation()
        control_mode_message.control_mode = ActuatorConfig_pb2.ControlMode.Value('POSITION')
        self.SendCallWithRetry(self.actuator_config.SetControlMode, 3, control_mode_message, device_id=1)

        base_servo_mode = Base_pb2.ServoingModeInformation()
        base_servo_mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
        self.base.SetServoingMode(base_servo_mode)

        self.already_stopped = True
        print('Clean Exit')

    @staticmethod
    def SendCallWithRetry(call, retry, *args):
        i = 0
        while i < retry:
            try:
                return call(*args)
            except:
                i += 1
        print("Failed to communicate")

def main():
    import argparse
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    import utilities

    parser = argparse.ArgumentParser()
    parser.add_argument("--cyclic_time", type=float, help="delay, in seconds, between cylic control call", default=0.001)
    parser.add_argument("--duration", type=int, help="example duration, in seconds (0 means infinite)", default=300)
    parser.add_argument("--print_stats", default=True, help="print stats in command line or not (0 to disable)", type=lambda x: (str(x).lower() not in ['false', '0', 'no']))
    args = utilities.parseConnectionArguments1(parser)

    # Create connection to the device and get the router
    with utilities.DeviceConnection.createTcpConnection(args) as router:

        with utilities.DeviceConnection.createUdpConnection(args) as router_real_time:
            print("Passed real-time post")

            example = TorqueExample(router, router_real_time)
            success = example.InitCyclic(args.cyclic_time, args.duration, args.print_stats)

            if success:
                while example.cyclic_running:
                    try:
                        time.sleep(0.5)
                    except KeyboardInterrupt:
                        break

                example.StopCyclic()
            return 0 if success else 1

if __name__ == "__main__":
    exit(main())


if __name__ == "__main__":
    exit(main())
