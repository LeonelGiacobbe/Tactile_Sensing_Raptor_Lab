import time, sys, os
import numpy as np
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
from kortex_api.autogen.messages import Base_pb2, BaseCyclic_pb2, ActuatorConfig_pb2, ControlConfig_pb2
from kortex_api.autogen.client_stubs.ActuatorConfigClientRpc import ActuatorConfigClient
from kortex_api.autogen.client_stubs.ControlConfigClientRpc import ControlConfigClient
from movement_functions import home_pos

def get_gravity_compensation(current_pos):
    return np.array([
        2.5 * np.cos(np.radians(current_pos[0])),  # Joint 1
        1.8 * np.cos(np.radians(current_pos[1])),  # Joint 2
        0.6 * np.cos(np.radians(current_pos[2])),  # Joint 3
        0.3 * np.cos(np.radians(current_pos[3])),  # Joint 4
        0.15 * np.cos(np.radians(current_pos[4])), # Joint 5
        0.05 * np.cos(np.radians(current_pos[5])), # Joint 6
        0.01 * np.cos(np.radians(current_pos[6]))  # Joint 7
    ])

class ImpedanceController:
    def __init__(self, base, base_cyclic, actuator_config, control_config, K, B, M):
        self.base = base
        self.base_cyclic = base_cyclic
        self.base_command = BaseCyclic_pb2.Command()
        self.actuator_config = actuator_config
        self.control_config = control_config
        self.damping = B 
        self.stiffness = K 
        self.mass_matrix = M  

        # State initialization
        self.joint_count = 7 # Comment to commit
        self.current_position = np.zeros(self.joint_count)
        self.current_velocity = np.zeros(self.joint_count)

        # Initialize command structure
        for _ in range(7):  # 7 actuators
            self.base_command.actuators.add()

        # Set control modes

        # Set all actuators to torque mode
        control_mode = ActuatorConfig_pb2.ControlModeInformation()
        control_mode.control_mode = ActuatorConfig_pb2.ControlMode.Value('TORQUE')
        for actuator_id in range(1, self.joint_count + 1):  # IDs 1-7
            self.actuator_config.SetControlMode(control_mode, actuator_id)
            # gravity_params = ControlConfig_pb2.GravityVector()
            # # gravity_params.gravity_vector.extend([0, 0, -9.81])  # Assuming Z-down
            # self.control_config.SetGravityVector(gravity_params, actuator_id)

        # Set arm in LOW_LEVEL_SERVOING
        base_servo_mode = Base_pb2.ServoingModeInformation()
        base_servo_mode.servoing_mode = Base_pb2.LOW_LEVEL_SERVOING
        self.base.SetServoingMode(base_servo_mode)


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
        tau = np.clip(tau, -5.0, 5.0)
        return tau

    def send_torque_command(self, torques):
        self.base_command.frame_id += 1
        if self.base_command.frame_id > 65535:
            self.base_command.frame_id = 0
        for i in range(self.joint_count):
            self.base_command.actuators[i].torque_joint = torques[i]
            self.base_command.actuators[i].flags = 1

        # Set arm in LOW_LEVEL_SERVOING
        base_servo_mode = Base_pb2.ServoingModeInformation()
        base_servo_mode.servoing_mode = Base_pb2.LOW_LEVEL_SERVOING
        self.base.SetServoingMode(base_servo_mode)
        
        self.base_cyclic.Refresh(self.base_command)


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
        print("Established router connection to both arms")

        # Create required services
        base1 = BaseClient(router1)
        base_cyclic1 = BaseCyclicClient(router1)
        actuator_config1 = ActuatorConfigClient(router1)
        control_config1 = ControlConfigClient(router1)

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
        controller = ImpedanceController(base1, base_cyclic1, actuator_config1, control_config1, K, B, M)

        # Main control loop
        DURATION = 300  # seconds
        start_time = time.time()
        
        controller.update_state()
        # start_pos[0] = 0.0
        # compensate = np.array([2.2, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0])
        # Generate desired trajectory
        desire_pos = controller.current_position
        desire_vel = np.zeros(7)

        while time.time() - start_time < DURATION:
            # Update current state
            controller.update_state()
            compensate = get_gravity_compensation(controller.current_position)
            
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