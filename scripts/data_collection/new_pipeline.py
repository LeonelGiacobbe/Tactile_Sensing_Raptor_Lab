import sys, os, time, cv2, gsdevice, glob, shutil, re, random, threading

from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
from kortex_api.autogen.messages import Base_pb2
from kortex_api.autogen.messages import BaseCyclic_pb2
from movement_functions import *

P_SLIP_1 = 26.00 # Must be a value in mm (not percentage of gripper opening)
P_SLIP_2 = 26.00 # Must be a value in mm (not percentage of gripper opening)

class GripperCommandExample:
    def __init__(self, router, proportional_gain = 2.0):

        self.proportional_gain = proportional_gain
        self.router = router

        # Create base client using TCP router
        self.base = BaseClient(self.router)

        # Create base_cyclic client using UDP router

        

    def Goto(self, percentage):
        gripper_command = Base_pb2.GripperCommand()
        finger = gripper_command.gripper.finger.add()

        # Close the gripper with position increments
        #print("Performing gripper test in position...")
        gripper_command.mode = Base_pb2.GRIPPER_POSITION
        position = percentage
        finger.finger_identifier = 1
        finger.value = position
        # print("Going to position {:0.2f}...".format(finger.value))
        self.base.SendGripperCommand(gripper_command)
        time.sleep(0.5)
    
    def GetGripperPosi(self):
        gripper_request = Base_pb2.GripperRequest()
        # Wait for reported position to be opened
        gripper_request.mode = Base_pb2.GRIPPER_POSITION
        gripper_measure = self.base.GetMeasuredGripperMovement(gripper_request)

        return gripper_measure.finger[0].value
    
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
        # print("Image saved")
    else:
        print("Error: No image captured")

def move_arm_thread(base, base_cyclic, x, y, z, theta_x, theta_y, theta_z):
    move_arm(base, base_cyclic, x, y, z, theta_x, theta_y, theta_z)

def main():
    # Counter for captured frame identification
    counter1 = 1
    counter2 = 1
    # Gelsight connection
    # To define multiple connections, edit gsdevice to accept dev_id as a constructor argument
    # That way we can instantiate multiple objects, according to /dev/videoX
    # dev_id will be the X in videoX
    # For 2f-140 gripper
    # dev2 = gsdevice.Camera("GelSight Mini", 4) # second arg should be X in videoX 
    # # For 2f-85 gripper
    # dev1 = gsdevice.Camera("Gelsight Mini", 2) # second arg should be X in videoX

    # dev1.connect()
    # dev2.connect()

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

        gripper1 = GripperCommandExample(router1, 0.5) # 2f-85 gripper
        gripper2 = GripperCommandExample(router2, 0.5) # 2f-140 gripper
        # To avoid using impedance controller, we can measure before-hand a gripper width 
        # That causes the object to fall barely. then we can stop the measuring there.

        base1 = BaseClient(router1)
        base2 = BaseClient(router2)
        base_cyclic1 = BaseCyclicClient(router1)
        base_cyclic2 = BaseCyclicClient(router2)
        try:
            subcnt = 1
            trial_cnt = get_next_trial_number(os.getcwd())
            for i in range(3):
                gripper1.Goto(0.75) # 0.711
                gripper2.Goto(0.83)
                posi2 = gripper2.GetGripperPosi()
                posi1 = gripper1.GetGripperPosi()
                start1, start2 = posi1, posi2
                print(f"Current positions in percentage (1 and 2): {posi1}, {posi2}")
                print(f"Current positions in mm (1 and 2): {85 - posi1 * 85}, {(140 - posi2 * 140)}")

                xmov = random.uniform(0, 0.035)
                ymov = random.uniform(0, 0.021)
                print("x and y mov: ", xmov * 100, ymov)
                # Create threads for each move_arm call
                thread1 = threading.Thread(target=move_arm_thread, args=(base1, base_cyclic1, 0, xmov, ymov, 0, 0, 0))
                thread2 = threading.Thread(target=move_arm_thread, args=(base2, base_cyclic2, 0, xmov, ymov, 0, 0, 0))
                
                # Start both threads
                thread1.start()
                thread2.start()
                
                # Wait for both threads to complete
                thread1.join()
                thread2.join()

                time.sleep(2)

                # Below condition is basically "While object has not slipped"
                while (140 - posi2 * 140) < P_SLIP_2: # values in mm
                    if (85 - posi1 * 85) < P_SLIP_1: # values in mm
                        gripper1.Goto(posi1 - .01)
                        posi1 = gripper1.GetGripperPosi()
                        mm_posi2 = (140 - posi2 * 140)
                        mm_posi1 = (85 - posi1 * 85) # current opening (percentage) times max opening (85 mm)
                        capture_image(mm_posi1, mm_posi2, counter1, dev1)
                        # print(f"gripper1 opening in mm: {85 - posi1 * 85}")
                        counter1 += 1
                    else:
                        print("Moving stationary gripper")
                        gripper2.Goto(posi2 - .01)
                        
                        # Only close gripper 1 if the trial has not ended:
                        if (140 - posi2 * 140) < P_SLIP_2:
                            gripper1.Goto(start1 - .01)
                        posi1 = gripper1.GetGripperPosi()
                        posi2 = gripper2.GetGripperPosi()
                        # print(f"gripper2 posi in mm: {(140 - posi2 * 140)}")

                print("Finished trial, moving images to folder...")
                
                trial_dir_name = f"tr_{trial_cnt}_dp_{subcnt}_wood_block_x_{xmov * 1000}_y_{ymov * 1000}_gpown_{str(85 - posi1 * 85)}_gpother_{str((140 - posi2 * 140))}"
                os.mkdir(trial_dir_name)
                images = glob.glob(os.path.join(os.getcwd(), '*.jpg'), recursive=True)
                for image in images:
                    dst_path = os.path.join(trial_dir_name, os.path.basename(image))
                    shutil.move(image, dst_path)

                print("Moved images to ", trial_dir_name)

                thread1 = threading.Thread(target=move_arm_thread, args=(base1, base_cyclic1, 0, 0, -ymov, 0, 0, 0))
                thread2 = threading.Thread(target=move_arm_thread, args=(base2, base_cyclic2, 0, 0, -ymov, 0, 0, 0))

                # Start both threads
                thread1.start()
                thread2.start()
                
                # Wait for both threads to complete
                thread1.join()
                thread2.join()
            
                print("Returned arms to original position")

                time.sleep(2)

                subcnt += 1

            print("Done with trial")

        except Exception as e :
            print("Error! ", e)
        

if __name__ == "__main__":
    main()

# 75.877 makes jenga block fall
# 47.36 for pink ball