import time
import threading
import queue 
from threading import Lock

from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
from kortex_api.autogen.messages import Base_pb2
from kortex_api.autogen.messages import BaseCyclic_pb2

class GripperCommand: 
    
    MAX_SPEED_85_MM_PER_SEC = 150.0 
    MAX_SPEED_140_MM_PER_SEC = 250.0 

    def __init__(self, router, router_real_time, proportional_gain=2.0, gripper_type="85", gripper_index=0):
        self.proportional_gain = proportional_gain
        self.gripper_type = gripper_type
        self.gripper_index = gripper_index # To differentiate between multiple grippers if on same base
        # I don't think its possible to have more than one gripper commanded by one arm? Just following the kinova api for consistency
        # Set max speed based on type
        if self.gripper_type == "85":
            self.MAX_VEL_MM = self.MAX_SPEED_85_MM_PER_SEC
        elif self.gripper_type == "140":
            self.MAX_VEL_MM = self.MAX_SPEED_140_MM_PER_SEC
        else:
            raise ValueError("Unsupported gripper type. Must be '85' or '140'.")

        self.router = router
        self.router_real_time = router_real_time
        self.base = BaseClient(self.router)
        self.base_cyclic = BaseCyclicClient(self.router_real_time)

        self.base_command = BaseCyclic_pb2.Command()
        self.base_command.frame_id = 0
        self.base_command.interconnect.command_id.identifier = 0
        self.base_command.interconnect.gripper_command.command_id.identifier = 0
        self.motorcmd = self.base_command.interconnect.gripper_command.motor_cmd.add()

        # Get initial feedback to set current position
        base_feedback = self.base_cyclic.RefreshFeedback()
        # Ensure we are accessing the correct gripper's feedback if multiple exist
        self.motorcmd.position = base_feedback.interconnect.gripper_feedback.motor[self.gripper_index].position
        self.motorcmd.velocity = 0
        self.motorcmd.force = 100

        for actuator in base_feedback.actuators:
            self.actuator_command = self.base_command.actuators.add()
            self.actuator_command.position = actuator.position
            self.actuator_command.velocity = 0.0
            self.actuator_command.torque_joint = 0.0
            self.actuator_command.command_id = 0

        self.previous_servoing_mode = self.base.GetServoingMode()
        servoing_mode_info = Base_pb2.ServoingModeInformation()
        servoing_mode_info.servoing_mode = Base_pb2.LOW_LEVEL_SERVOING
        self.base.SetServoingMode(servoing_mode_info)

        # Threading and communication variables
        self._target_position_percentage = self.motorcmd.position # Target for the internal control loop
        self._stop_event = threading.Event() # Event to signal thread to stop
        self._feedback_queue = queue.Queue(maxsize=1) # For sending (pos_perc, vel_mm/s) to MPC
        self._command_queue = queue.Queue(maxsize=1) # For receiving target_position_percentage from MPC
        self._control_thread = None

        # Lock for accessing internal state if multiple parts of the gripper class need it
        # self._state_lock = Lock()

    def Cleanup(self):
        """Stops the control thread and restores servoing mode."""
        self._stop_event.set() # Signal the control thread to stop
        if self._control_thread and self._control_thread.is_alive():
            self._control_thread.join(timeout=1.0) # Wait for thread to finish, with a timeout
            if self._control_thread.is_alive():
                print(f"Warning: Gripper {self.gripper_index} control thread did not terminate gracefully.")
        self.base.SetServoingMode(self.previous_servoing_mode)
        print(f"Gripper {self.gripper_index} cleaned up.")

    def start_control_thread(self):
        """Starts the dedicated control loop for this gripper."""
        if not self._control_thread or not self._control_thread.is_alive():
            self._stop_event.clear() # Reset stop event for fresh start
            self._control_thread = threading.Thread(target=self._run_gripper_control_loop, daemon=True)
            self._control_thread.start()
            print(f"Gripper {self.gripper_index} control thread started.")
        else:
            print(f"Gripper {self.gripper_index} control thread already running.")

    def _run_gripper_control_loop(self):
        """
        Main loop for gripper control thread
        It continuously sends commands and gets feedback.
        """
        while not self._stop_event.is_set():
            try:
                # Check for new target position from MPC (non-blocking)
                try:
                    new_target = self._command_queue.get_nowait()
                    self._target_position_percentage = new_target
                except queue.Empty:
                    pass # No new command, continue with current target

                # Refresh feedback and send command
                base_feedback = self.base_cyclic.Refresh(self.base_command)
                current_position_percentage = base_feedback.interconnect.gripper_feedback.motor[self.gripper_index].position
                current_velocity_percentage = base_feedback.interconnect.gripper_feedback.motor[self.gripper_index].velocity

                # Calculate velocity command based on error to the internal target
                position_error = self._target_position_percentage - current_position_percentage
                # print("position error: ", position_error)
                
                # Apply proportional control
                if abs(position_error) < 0.03: # Tolerance for stopping
                    self.motorcmd.velocity = 0
                    self.motorcmd.position = self._target_position_percentage # Ensure it settles at target
                else:
                    self.motorcmd.velocity = self.proportional_gain * abs(position_error) 
                    if self.motorcmd.velocity > 100.0:
                        self.motorcmd.velocity = 100.0
                    self.motorcmd.position = self._target_position_percentage # Continuously command target

                # Prepare and send feedback to MPC (non-blocking)
                current_velocity_mm_s = (current_velocity_percentage / 100.0) * self.MAX_VEL_MM
                
                # Using current_position_percentage directly here as it's from latest refresh
                feedback_data = (current_position_percentage, current_velocity_mm_s)
                
                try:
                    # Clear old feedback if not read, then put new feedback
                    if not self._feedback_queue.empty():
                        self._feedback_queue.get_nowait()
                    self._feedback_queue.put_nowait(feedback_data)
                except queue.Full:
                    # Should not happen with maxsize=1 and get_nowait first, but good practice
                    pass 

                time.sleep(0.001) # Cyclic command frequency (e.g., 1000 Hz)

            except Exception as e:
                print(f"Error in gripper {self.gripper_index} control loop: {e}")
                self._stop_event.set() # Stop thread on error

    def set_target_position_percentage(self, target_position_percentage):
        """
        Sends a new target position (in percentage) to the gripper's control thread.
        This call is non-blocking.
        """
        # Clamp target to valid range (0-100%)
        target_position_percentage = max(0.0, min(100.0, target_position_percentage))
        
        try:
            # Clear old command if not processed, then put new command
            if not self._command_queue.empty():
                self._command_queue.get_nowait()
            self._command_queue.put_nowait(target_position_percentage)
            # print(f"Gripper {self.gripper_index} command queued: {target_position_percentage:.2f}%")
        except queue.Full:
            # This case means the queue has maxsize=1 and get_nowait failed, which is unexpected after empty check
            print(f"Warning: Command queue for gripper {self.gripper_index} is full. Command not sent.")
            pass 

    def get_latest_feedback(self):
        """
        Retrieves the latest position (percentage) and velocity (mm/s)
        from the gripper's control thread feedback queue. Non-blocking.
        Returns (position_percentage, velocity_mm_s) or None if no new feedback.
        """
        try:
            return self._feedback_queue.get_nowait()
        except queue.Empty:
            return None # No new feedback yet

    def is_target_position_reached(self, tolerance=0.1):
        """
        Checks if the gripper is close to its _currently commanded_ target percentage.
        This is useful for knowing if a specific movement is "done".
        """
        try:
            # Currently calls api, might be a good idea to have a "cached"
            # last state as a class attribute
            base_feedback = self.base_cyclic.RefreshFeedback()
            current_position = base_feedback.interconnect.gripper_feedback.motor[self.gripper_index].position
            return abs(self._target_position_percentage - current_position) < tolerance
        except Exception as e:
            print(f"Error checking target reached for gripper {self.gripper_index}: {e}")
            return False

    def GetGripperPosi(self):
        """
        Provides gripper position from a non-cyclic API call.
        """
        gripper_request = Base_pb2.GripperRequest()
        gripper_request.mode = Base_pb2.GRIPPER_POSITION
        gripper_measure = self.base.GetMeasuredGripperMovement(gripper_request)
        return gripper_measure.finger[self.gripper_index].value 