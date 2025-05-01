import time, sys, os
import numpy as np
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
from kortex_api.autogen.messages import Base_pb2, BaseCyclic_pb2

class ImpedanceController:
    def __init__(self, base, base_cyclic, K, B, M):
        self.base = base
        self.base_cyclic = base_cyclic
        self.base_command = BaseCyclic_pb2.Command()
        self.damping = B 
        self.stiffness = K 
        self.mass_matrix = M  

        # State initialization
        self.joint_count = 7
        self.current_position = np.zeros(self.joint_count)
        self.current_velocity = np.zeros(self.joint_count)


    def update_state(self, base, base_cyclic):
        feedback = self.base_cyclic.RefreshFeedback() # Provides current state of bot (pose_values, etc) at 1Khz 

        self.current_position = np.array([feedback.actuators[x].position for i in range(self.joint_count)])
        self.current_velocity = np.array([feedback.actuators[x].velocity for i in range(self.joint_count)])

        return np.array([feedback.actuators[i].torque_gravity for i in range(self.joint_count)])

    def control(self, q_des, v_des, compensate):
        """
        Calculate torque command using impedance control law
        """
        acc_des = self.damping * (v_des - self.current_velocity) + \
                 self.stiffness * (q_des - self.current_position)
        tau = np.dot(self.mass_mat, acc_des.T).T + compensate
        return tau

    def send_torque_command(self, torques):
        for i in range(self.joint_count):
            self.base_command.actuators[i].torque_joint = torques[i]

        self.base_cyclic.Refresh(self.base_command)


def main():
    # Import the utilities helper module
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    import utilities
    # Parse arguments
    args1 = utilities.parseConnectionArguments1()
    args2 = utilities.parseConnectionArguments2()
    print("Parsed arguments")

    with utilities.DeviceConnection.createTcpConnection(args1) as router1, \
        utilities.DeviceConnection.createTcpConnection(args2) as router2:
        print("Established router connection to both arms")

        # Create required services
        base1 = BaseClient(router1)
        base_cyclic1 = BaseCyclicClient(router1)

        base2 = BaseClient(router2)
        base_cyclic2 = BaseCyclicClient(router2)

        # Initialize controller parameters
        K = 100.0 * np.ones(7)  # Stiffness
        B = 20.0 * np.ones(7)   # Damping
        M = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]) # might need tuning

        # Assuming arm 1 is the leader (moved with move_arm) and arm 2 is the follower
        controller = ImpedanceController(base2, base_cyclic2, K, B, M)

        control_mode = Base_pb2.JointTorqueControl()
        base2.SetControlMode(control_mode)

        # Main control loop
        DURATION = 30  # seconds
        start_time = time.time()
        
        # Desired trajectory (simple sine wave for first joint)
        while time.time() - start_time < DURATION:
            # Update current state
            compensate = controller.update_state()
            
            # Generate desired trajectory
            t = time.time() - start_time
            desire_pos = controller.current_position.copy()
            desire_pos[controller.joint_count - 1] = 30 * np.sin(t)  # 30 degree amplitude sine wave
            desire_vel = np.zeros(7)
            desire_vel[0] = 30 * np.cos(t)  # Derivative of position
            
            # Compute torque command
            torques = controller.control(desire_pos, desire_vel, compensate)
            
            # Send command
            controller.send_torque_command(torques)
            
            # Control loop timing
            time.sleep(0.001)  # 1ms loop
        

if __name__ == "__main__":
    exit(main())

"""
https://github.com/CROBOT974/KinovaGen3-impedance-controller/blob/main/controller/impedance.py

"""
