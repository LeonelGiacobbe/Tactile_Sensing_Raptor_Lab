import time, sys, os
import numpy as np
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
from kortex_api.autogen.messages import Base_pb2, BaseCyclic_pb2, ActuatorConfig_pb2
from kortex_api.autogen.client_stubs.ActuatorConfigClientRpc import ActuatorConfigClient
from movement_functions import home_pos
from kortex_api.RouterClient import RouterClientSendOptions

class ImpedanceController:
    def __init__(self, base, base_cyclic, actuator_config, K, B, M):
        self.base = base
        self.base_cyclic = base_cyclic
        self.base_command = BaseCyclic_pb2.Command()
        self.base_feedback = BaseCyclic_pb2.Feedback()
        self.actuator_config = actuator_config
        self.damping = B 
        self.stiffness = K 
        self.mass_matrix = M  

        # Change send option to reduce max timeout at 3ms
        self.sendOption = RouterClientSendOptions()
        self.sendOption.andForget = False
        self.sendOption.delay_ms = 0
        self.sendOption.timeout_ms = 3

        # State initialization
        self.joint_count = 7 # Comment to commit
        self.current_position = np.zeros(self.joint_count)
        self.current_velocity = np.zeros(self.joint_count)

        # Initialize command structure
        for _ in range(7):  # 7 actuators
            self.base_command.actuators.add()

        for x in range(self.actuator_count):
            self.base_command.actuators[x].flags = 1  # servoing
            self.base_command.actuators[x].position = self.base_feedback.actuators[x].position

        self.base_command.actuators[0].torque_joint = self.base_feedback.actuators[0].torque
        self.base_command.actuators[1].torque_joint = self.base_feedback.actuators[1].torque

        servoing_mode = Base_pb2.ServoingModeInformation()
        servoing_mode.servoing_mode = Base_pb2.LOW_LEVEL_SERVOING
        self.base.SetServoingMode(servoing_mode)

        # Send first frame
        self.base_feedback = self.base_cyclic.Refresh(self.base_command, 0, self.sendOption)

        # Set first and second actuator in torque mode now that the command is equal to measure
        control_mode_message = ActuatorConfig_pb2.ControlModeInformation()
        control_mode_message.control_mode = ActuatorConfig_pb2.ControlMode.Value('TORQUE')
        device_id_1 = 1  # first actuator as id = 1
        device_id_2 = 2
        self.SendCallWithRetry(self.actuator_config.SetControlMode, 3, control_mode_message, device_id_1)
        self.SendCallWithRetry(self.actuator_config.SetControlMode, 3, control_mode_message, device_id_2)

    def update_state(self):
        feedback = self.base_cyclic.RefreshFeedback() # Provides current state of bot (pose_values, etc) at 1Khz 

        self.current_position = np.array([feedback.actuators[i].position for i in range(self.joint_count)])
        self.current_velocity = np.array([feedback.actuators[i].velocity for i in range(self.joint_count)])

        # return np.array([feedback.actuators[i].torque_gravity for i in range(self.joint_count)])

    def control(self, q_des, v_des, compensate):
        """
        Calculate torque command using impedance control law
        """
        acc_des = self.damping * (v_des - self.current_velocity) + \
                 self.stiffness * (q_des - self.current_position)
        tau = np.dot(self.mass_matrix, acc_des.T).T + compensate
        return tau

    def send_torque_command(self, torques):
        for i in range(self.joint_count):
            self.base_command.actuators[i].position = self.current_position[i]
            self.base_command.actuators[i].flags = 1
            self.base_command.actuators[i].torque_joint = torques[i]

        # Set arm in LOW_LEVEL_SERVOING
        base_servo_mode = Base_pb2.ServoingModeInformation()
        base_servo_mode.servoing_mode = Base_pb2.LOW_LEVEL_SERVOING
        self.base.SetServoingMode(base_servo_mode)

        self.base_cyclic.Refresh(self.base_command)

        # Incrementing identifier ensure actuators can reject out of time frames
        self.base_command.frame_id += 1
        if self.base_command.frame_id > 65535:
            self.base_command.frame_id = 0
        for i in range(self.actuator_count):
            self.base_command.actuators[i].command_id = self.base_command.frame_id

        # Frame is sent
        self.base_feedback = self.base_cyclic.Refresh(self.base_command, 0, self.sendOption)

def main():
    # Import the utilities helper module
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    import utilities
    # Parse arguments
    args1 = utilities.parseConnectionArguments1()
    args2 = utilities.parseConnectionArguments2()
    print("Parsed arguments")

    # with utilities.DeviceConnection.createTcpConnection(args1) as router1, \
    #     utilities.DeviceConnection.createTcpConnection(args2) as router2:
    with utilities.DeviceConnection.createTcpConnection(args1) as router1:
        with utilities.DeviceConnection.createUdpConnection(args1) as router_real_time:
            print("Established router connection to both arms")

            # Create required services
            base1 = BaseClient(router1)
            base_cyclic1 = BaseCyclicClient(router_real_time)
            actuator_config1 = ActuatorConfigClient(router1)

            # base2 = BaseClient(router2)
            # base_cyclic2 = BaseCyclicClient(router2)

            # Initialize controller parameters
            K = 100.0 * np.ones(7)  # Stiffness
            B = 20.0 * np.ones(7)   # Damping
            # Replace with proper 7x7 matrix
            M = np.array([
                [ 8.34640616e-01, -1.67586904e-02, -3.33845338e-01, -1.67568732e-02, 8.27666657e-02, 4.45099485e-06, 7.27667533e-05],
                [-1.67586904e-02, 1.03098181e+00, -1.68736725e-02, 2.83497686e-01, 1.19122733e-03, 8.20031847e-02, 3.23246777e-03],
                [-3.33845338e-01, -1.68736725e-02, 2.04327238e-01, -1.70956956e-03, -4.46219323e-02, -1.70839121e-03, 6.93765398e-04],
                [-1.67568732e-02, 2.83497686e-01, -1.70956956e-03, 2.23746485e-01, 1.19148924e-03, 2.22576808e-02, 7.21967130e-04],
                [ 8.27666657e-02, 1.19122733e-03, -4.46219323e-02, 1.19148924e-03, 2.31551123e-02, 1.35710992e-06, 1.52237837e-05],
                [ 4.45099485e-06, 8.20031847e-02, -1.70839121e-03, 2.22576808e-02, 1.35710992e-06, 2.23006309e-02, 7.21973837e-04],
                [ 7.27667533e-05, 3.23246777e-03, 6.93765398e-04, 7.21967130e-04, 1.52237837e-05, 7.21973837e-04, 4.07463788e-03]
            ])


            home_pos(base1, base_cyclic1)

            # Assuming arm 1 is the leader (moved with move_arm) and arm 2 is the follower
            controller = ImpedanceController(base1, base_cyclic1, actuator_config1, K, B, M)

            # Main control loop
            DURATION = 300  # seconds
            start_time = time.time()
            
            controller.update_state()
            # start_pos[0] = 0.0
            # compensate = np.array([2.2, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0])
            # Generate desired trajectory
            desire_pos = controller.current_position
            desire_vel = np.zeros(7)

            compensate = np.array([1.36624422e-04, 1.85970991e+01, -3.22901227e-01, 1.39311793e+00, 2.23671438e-05, 1.39296359e+00, 5.85322326e-02])

            while time.time() - start_time < DURATION:
                # Update current state
                controller.update_state()
                
                # Compute torque command
                torques = controller.control(desire_pos, desire_vel, compensate)
                # torques[0] = 0.0            
                # Send command
                controller.send_torque_command(torques)
                
                # Control loop timing
                time.sleep(0.001)  # 1ms loop
        

if __name__ == "__main__":
    exit(main())

"""
https://github.com/CROBOT974/KinovaGen3-impedance-controller/blob/main/controller/impedance.py

"""