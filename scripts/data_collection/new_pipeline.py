import sys, os, time, cv2, gsdevice, glob, shutil, re

from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
from kortex_api.autogen.messages import Base_pb2
from kortex_api.autogen.messages import BaseCyclic_pb2

P_SLIP = 47.36

class GripperLowLevelExample:
    def __init__(self, router, router_real_time, proportional_gain = 2.0):
        """
            GripperLowLevelExample class constructor.

            Inputs:
                kortex_api.RouterClient router:            TCP router
                kortex_api.RouterClient router_real_time:  Real-time UDP router
                float proportional_gain: Proportional gain used in control loop (default value is 2.0)

            Outputs:
                None
            Notes:
                - Actuators and gripper initial position are retrieved to set initial positions
                - Actuator and gripper cyclic command objects are created in constructor. Their
                  references are used to update position and speed.
        """
        self.counter = 1

        self.proportional_gain = proportional_gain

        ###########################################################################################
        # UDP and TCP sessions are used in this example.
        # TCP is used to perform the change of servoing mode
        # UDP is used for cyclic commands.
        #
        # 2 sessions have to be created: 1 for TCP and 1 for UDP
        ###########################################################################################

        self.router = router
        self.router_real_time = router_real_time

        # Create base client using TCP router
        self.base = BaseClient(self.router)

        # Create base cyclic client using UDP router.
        self.base_cyclic = BaseCyclicClient(self.router_real_time)

        # Create base cyclic command object.
        self.base_command = BaseCyclic_pb2.Command()
        self.base_command.frame_id = 0
        self.base_command.interconnect.command_id.identifier = 0
        self.base_command.interconnect.gripper_command.command_id.identifier = 0

        # Add motor command to interconnect's cyclic
        self.motorcmd = self.base_command.interconnect.gripper_command.motor_cmd.add()

        # Set gripper's initial position velocity and force
        base_feedback = self.base_cyclic.RefreshFeedback()
        self.motorcmd.position = base_feedback.interconnect.gripper_feedback.motor[0].position
        self.motorcmd.velocity = 0
        self.motorcmd.force = 100

        for actuator in base_feedback.actuators:
            self.actuator_command = self.base_command.actuators.add()
            self.actuator_command.position = actuator.position
            self.actuator_command.velocity = 0.0
            self.actuator_command.torque_joint = 0.0
            self.actuator_command.command_id = 0
            print("Position = ", actuator.position)

        # Save servoing mode before changing it
        self.previous_servoing_mode = self.base.GetServoingMode()

        # Set base in low level servoing mode
        servoing_mode_info = Base_pb2.ServoingModeInformation()
        servoing_mode_info.servoing_mode = Base_pb2.LOW_LEVEL_SERVOING
        self.base.SetServoingMode(servoing_mode_info)

    def Cleanup(self):
        """
            Restore arm's servoing mode to the one that
            was effective before running the example.

            Inputs:
                None
            Outputs:
                None
            Notes:
                None

        """
        # Restore servoing mode to the one that was in use before running the example
        self.base.SetServoingMode(self.previous_servoing_mode)


    def Goto(self, target_position):
        """
            Position gripper to a requested target position using a simple
            proportional feedback loop which changes speed according to error
            between target position and current gripper position

            Inputs:
                float target_position: position (0% - 100%) to send gripper to.
            Outputs:
                Returns True if gripper was positionned successfully, returns False
                otherwise.
            Notes:
                - This function blocks until position is reached.
                - If target position exceeds 100.0, its value is changed to 100.0.
                - If target position is below 0.0, its value is set to 0.0.
        """
        if target_position > 100.0:
            target_position = 100.0
        if target_position < 0.0:
            target_position = 0.0

        while True:
            try:
                base_feedback = self.base_cyclic.Refresh(self.base_command)

                # Calculate speed according to position error (target position VS current position)
                position_error = target_position - base_feedback.interconnect.gripper_feedback.motor[0].position
                # print(f"Position error: {position_error}")
                
                # If positional error is small, stop gripper
                if abs(position_error) < 1.5:
                    position_error = 0
                    self.motorcmd.velocity = 0
                    self.base_cyclic.Refresh(self.base_command)
                    return True
                else:
                    self.motorcmd.velocity = self.proportional_gain * abs(position_error)
                    if self.motorcmd.velocity > 100.0:
                        self.motorcmd.velocity = 100.0
                    self.motorcmd.position = target_position

            except Exception as e:
                print("Error in refresh: " + str(e))
                return False
            time.sleep(0.001)
        return True
    
    def GetGripperPosi(self):
        base_feedback = self.base_cyclic.Refresh(self.base_command)
        return base_feedback.interconnect.gripper_feedback.motor[0].position
    
def get_next_trial_number(base_dir):
    # List all directories in the given base directory
    folders = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]
    
    # Regex to extract the trial number 
    trial_numbers = []
    for folder in folders:
        match = re.match(r"tr_(\d+)", folder)
        if match:
            trial_numbers.append(int(match.group(1)))
    
    # If there are no trials yet, start at 1
    if not trial_numbers:
        return 1
    
    # Return the next trial number (max + 1)
    return max(trial_numbers) + 1
    
def capture_image(gripper_posi_own, gripper_posi_other, counter, dev):
    f0 = dev.get_raw_image()
    if f0 is not None:
        cv2.imwrite(f'gpown_{gripper_posi_own}_gpother_{gripper_posi_other}_frame{counter}.jpg', f0)
        print("Image saved")
    else:
        print("Error: No image captured")


def main():
    # Counter for captured frame identification
    counter1 = 1
    counter2 = 1
    # Gelsight connection
    # To define multiple connections, edit gsdevice to accept dev_id as a constructor argument
    # That way we can instantiate multiple objects, according to /dev/videoX
    # dev_id will be the X in videoX
    dev1 = gsdevice.Camera("GelSight Mini", 4) # second arg should be X in videoX
    dev2 = gsdevice.Camera("Gelsight Mini", 2) # second arg should be X in videoX

    dev1.connect()
    dev2.connect()

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

        with utilities.DeviceConnection.createUdpConnection(args1) as router_real_time1, utilities.DeviceConnection.createUdpConnection(args2) as router_real_time2:

            gripper1 = GripperLowLevelExample(router1, router_real_time1, 0.5) # Fast gripper
            gripper2 = GripperLowLevelExample(router2, router_real_time2, 0.75) # Slow gripper
            # To avoid using impedance controller, we can measure before-hand a gripper width 
            # That causes the object to fall barely. then we can stop the measuring there.
            try:
                posi1 = gripper1.GetGripperPosi()
                posi2 = gripper2.GetGripperPosi()
                start1, start2 = posi1, posi2
                print(f"Current positions (1 and 2): {posi1}, {posi2}")

                # Below condition is basically "While object has not slipped"
                while posi2 > P_SLIP:
                    if posi1 > P_SLIP:
                        gripper1.Goto(posi1 - 2)
                        posi1 = gripper1.GetGripperPosi()
                        capture_image(posi1, posi2, counter1, dev1)
                        counter1 += 1
                    else:
                        print("Moving stationary gripper")
                        gripper1.Goto(start1 - 2)
                        posi1 = gripper1.GetGripperPosi()
                        gripper2.Goto(posi2 - 2)
                        posi2 = gripper2.GetGripperPosi()

                print("Finished trial, moving images to folder...")

                trial_cnt = get_next_trial_number(os.getcwd())
                subcnt = 1
                trial_dir_name = f"tr_{trial_cnt}_dp_{subcnt}_some_material_x_{1.1}_y_{3.3}_gp_{str(posi1)}"
                os.mkdir(trial_dir_name)
                images = glob.glob(os.path.join(os.getcwd(), '*.jpg'), recursive=True)
                for image in images:
                    dst_path = os.path.join(trial_dir_name, os.path.basename(image))
                    shutil.move(image, dst_path)

                print("Moved images to ", trial_dir_name)
                # Restore state
                gripper1.Cleanup()
                gripper2.Cleanup()

                print("Returned to original state")

                # frame = dev.get_raw_image()

            except Exception as e :
                gripper1.Cleanup()
                gripper2.Cleanup()
                print("Error! ", e)
                

if __name__ == "__main__":
    main()

# 75.877 makes jenga block fall
# 47.36 for pink ball