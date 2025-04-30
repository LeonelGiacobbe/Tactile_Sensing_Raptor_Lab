import time
import numpy as np
from kortex_api.autogen.messages import Base_pb2, Torque_pb2
from kortex_api.autogen.client_stubs import BaseCyclicClient, TorqueCyclicClient
from kortex_api.RouterClient import RouterClient
from kortex_api.SessionManager import SessionManager

# Initialize session
def create_session():
    router = RouterClient("192.168.1.10", 10000)
    session_manager = SessionManager(router)
    session_manager.CreateSession("admin", "admin")
    return router, session_manager

# Impedance control loop
def impedance_control():
    router, session = create_session()
    base_client = BaseCyclicClient(router)
    torque_client = TorqueCyclicClient(router)

    # Desired joint positions and velocities
    q_d = np.array([0.0, 0.2, 0.0, 1.0, 0.0, 1.0, 0.0])
    dq_d = np.zeros(7)

    # Control gains
    Kp = np.diag([15.0]*7)
    Kd = np.diag([2.0]*7)

    try:
        print("Starting impedance control loop...")
        for _ in range(300):  # ~3 seconds at 10 Hz
            # Get current joint states
            feedback = base_client.GetMeasuredJointAngles()
            velocities = base_client.GetMeasuredJointVelocities()

            q = np.array([j.value for j in feedback.joint_angles])
            dq = np.array([j.value for j in velocities.joint_velocities])

            # Compute gravity compensation
            gravity_torques = torque_client.ComputeGravityCompensation()
            tau_grav = np.array([t.torque for t in gravity_torques.joint_torques])

            # Compute torque command
            pos_error = q_d - q
            vel_error = dq_d - dq
            tau_cmd = Kp.dot(pos_error) + Kd.dot(vel_error) + tau_grav

            # Send torque command
            torque_msg = Torque_pb2.SendJointTorqueCommand()
            for i in range(7):
                jt = torque_msg.joint_torques.add()
                jt.torque = tau_cmd[i]
            torque_client.SendJointTorqueCommand(torque_msg)

            time.sleep(0.01)

    finally:
        print("Exiting control loop.")
        session.CloseSession()
        router.Close()

if __name__ == "__main__":
    impedance_control()
