import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Wrench
import numpy as np

class ImpedanceControlNode(Node):
    def __init__(self):
        super().__init__('impedance_control_node')

        # Subscribe to JointState for joint positions/velocities/efforts
        self.create_subscription(JointState, '/m1/joint_states', self.joint_state_callback, 10)

        # Subscribe to Force/Torque sensor data (from the end-effector)
        self.create_subscription(Wrench, '/m1/ft_sensor/ft_data', self.ft_sensor_callback, 10)

        # Publisher for joint control (positions or torques)
        self.joint_pub = self.create_publisher(JointState, '/m1/joint_angles', 10)

        # Impedance parameters (stiffness and damping)
        self.K = 1000  # Stiffness
        self.B = 0.1   # Damping

        # Initialize other variables (position, velocity, etc.)
        self.current_position = np.zeros(7)
        self.current_velocity = np.zeros(7)
        self.current_force = np.zeros(3)

    def joint_state_callback(self, msg: JointState):
        self.current_position = np.array(msg.position)
        self.current_velocity = np.array(msg.velocity)

    def ft_sensor_callback(self, msg: Wrench):
        # Extract force and torque from the Wrench message
        force = np.array([msg.force.x, msg.force.y, msg.force.z])
        torque = np.array([msg.torque.x, msg.torque.y, msg.torque.z])

        # Implement Impedance Control logic here
        # F = K * (x - x_d) + B * (v - v_d)
        desired_position = np.array([2.15e-5, 0.262, -3.14, -2.269, -4.42, 0.959, 1.57])  # Example desired position (can be updated dynamically)
        desired_velocity = np.zeros(7)  # Example desired velocity

        force_control = self.K * (self.current_position - desired_position) + self.B * (self.current_velocity - desired_velocity)
        
        # Update the control signal (e.g., joint torques)
        joint_effort = force_control  # This should be mapped to joint torques or velocities

        # Publish the joint control commands
        joint_state_msg = JointState()
        joint_state_msg.effort = joint_effort.tolist()
        self.joint_pub.publish(joint_state_msg)

def main(args=None):
    rclpy.init(args=args)
    node = ImpedanceControlNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()


"""
Need to find way of getting to cartesian admittance mode with code

I am going to have two arms start opposite each other. This is the target position the follower wants to go back to. 
The grippers both firmly grasp an object
The end effector of the leader will move a random distance in x and y, 
causing the follower to follow. Then, the follower will gradually open the gripper, 
until the friction force is not enough to keep it in place and the follower goes back to its original position. 

Look at force control in kinova
In Force Control mode, a wrench command (force and torque) is sent to the tool. This allows you to
use the tool to apply a desired force and torque on an external object
"""