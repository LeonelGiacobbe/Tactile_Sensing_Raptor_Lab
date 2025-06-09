import sys, os, time, cv2, gsdevice, glob, shutil, re, random, threading

from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
from kortex_api.autogen.messages import Base_pb2
from kortex_api.autogen.messages import BaseCyclic_pb2
from movement_functions import *

class GripperCommand:
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
    
def capture_image(gripper_posi, counter, dev, gripper_no):
    f0 = dev.get_raw_image()
    if f0 is not None:
        cv2.imwrite(f'{gripper_no}_gp_{gripper_posi}_frame{counter}.jpg', f0)
        # print("Image saved")
    else:
        print("Error: No image captured")

def move_arm_thread1(base, base_cyclic, x, y, z, theta_x, theta_y, theta_z):
    move_arm(base, base_cyclic, x, y, z, theta_x, theta_y, theta_z)

def move_arm_thread2(base, base_cyclic, x, y, z, theta_x, theta_y, theta_z):
    move_arm(base, base_cyclic, x, y, z, theta_x, theta_y, theta_z)

def gripper_thread1(gripper, pos):
    gripper.Goto(pos)

def gripper_thread2(gripper, pos):
    gripper.Goto(pos)

def main():
    # Counter for captured frame identification
    counter = 1
    # Gelsight connection
    # To define multiple connections, edit gsdevice to accept dev_id as a constructor argument
    # That way we can instantiate multiple objects, according to /dev/videoX
    # dev_id will be the X in videoX
    # For 2f-140 gripper
    dev2 = gsdevice.Camera("GelSight Mini", 4) # second arg should be X in videoX 
    # For 2f-85 gripper
    dev1 = gsdevice.Camera("Gelsight Mini", 2) # second arg should be X in videoX

    dev1.connect()
    dev2.connect()

    print("Connected to gelsights")

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
        try:
            subcnt = 1
            cwd = os.getcwd()
            full_dir = cwd + "/gel"
            trial_cnt = get_next_trial_number(full_dir)

            ymov = random.uniform(-0.035, 0.035) # 35 mm
            zmov = random.uniform(-0.021, 0.021) # 21 mm
            print("y and z mov: ", ymov * 100, zmov * 100)
            

            for i in range(3):
                # Open grippers for a consistent start
                gripper2.Goto(0.5)
                gripper1.Goto(0.18)

                # Since we're not using impedance, add small variability to P_SLIP
                rand_slip = random.uniform (0.0, 0.35)
                
                P_SLIP_1 = 35.8 + rand_slip # Must be a value in mm (not percentage of gripper opening)
                P_SLIP_2 = 35.8 + rand_slip # Must be a value in mm (not percentage of gripper opening)

                rand_goto = random.uniform(0.0,0.005)
                gripper1.Goto(0.645 + rand_goto)
                gripper2.Goto(0.78 + rand_goto)
                
                posi1 = gripper1.GetGripperPosi()
                posi2 = gripper2.GetGripperPosi()
                start1, start2 = posi1, posi2
                # print(f"Current positions in percentage (1 and 2): {posi1}, {posi2}")
                # print(f"Current positions in mm (1 and 2): {85 - posi1 * 85}, {(140 - posi2 * 140)}")

                thread_elevate_1 = threading.Thread(target=move_arm_thread1, args=(base1, base_cyclic1, 0, 0, 0.05, 0, 0, 0))
                thread_elevate_2 = threading.Thread(target=move_arm_thread2, args=(base2, base_cyclic2, 0, 0, 0.05, 0, 0, 0))

                thread_elevate_1.start(), thread_elevate_2.start()
                thread_elevate_1.join(), thread_elevate_2.join()
                
                # Create threads for each move_arm call
                thread_randmov_1 = threading.Thread(target=move_arm_thread1, args=(base1, base_cyclic1, 0, ymov, zmov, 0, 0, 0))
                thread_randmov_2 = threading.Thread(target=move_arm_thread2, args=(base2, base_cyclic2, 0, ymov, zmov, 0, 0, 0))
                
                thread_randmov_1.start(), thread_randmov_2.start()
                thread_randmov_1.join(), thread_randmov_2.join()

                time.sleep(1)

                # Below condition is basically "While object has not slipped"
                move_gr1 = True
                while (140 - posi2 * 140) < P_SLIP_2 and (85 - posi1 * 85) < P_SLIP_1:
                    mm_posi2 = (140 - posi2 * 140)
                    mm_posi1 = (85 - posi1 * 85)  # current opening (percentage) times max opening (85 mm)

                    for _ in range(3):
                        capture_image(mm_posi1, counter, dev1, 1)
                        capture_image(mm_posi2, counter, dev2, 2)
                        counter += 1
                    if move_gr1:
                        gripper1.Goto(posi1 - 0.001)
                        posi1 = gripper1.GetGripperPosi()
                        move_gr1 = False
                    else:
                        gripper2.Goto(posi2 - 0.001)
                        posi2 = gripper2.GetGripperPosi()
                        move_gr1 = True

                print("Finished trial, moving images to folder...")
                
                trial_dir_name = f"tr_{trial_cnt}_dp_{subcnt}_gel_y_{ymov * 1000}_z_{zmov * 1000}_gpown_{str(P_SLIP_1)}_gpother_{str((P_SLIP_2))}"
                os.mkdir(trial_dir_name)
                images = glob.glob(os.path.join(os.getcwd(), '*.jpg'), recursive=True)
                for image in images:
                    dst_path = os.path.join(trial_dir_name, os.path.basename(image))
                    shutil.move(image, dst_path)

                jpg_files = glob.glob(os.path.join(trial_dir_name, "[12]_*.jpg"))

                # Create a dictionary: key = frame number, value = [1_ file, 2_ file]
                frame_to_pair = {}
                for f in jpg_files:
                    frame_num_match = re.search(r"frame(\d+)\.jpg", os.path.basename(f))
                    if frame_num_match:
                        frame_num = frame_num_match.group(1)
                        if frame_num not in frame_to_pair:
                            frame_to_pair[frame_num] = []
                        frame_to_pair[frame_num].append(f)

                # Only keep pairs (both 1_ and 2_ exist)
                paired_frames = {k: v for k, v in frame_to_pair.items() if len(v) == 2}

                if len(paired_frames) > 25:
                    print(f"Removed {2 * (len(paired_frames) - 25)} images")

                    # Sort frames by gp_ value
                    def extract_gp_sort_key(files):
                        # Use the 1_ file for gp_ extraction if available, else 2_ file
                        primary_file = next((f for f in files if "1_" in f), files[0])
                        basename = os.path.basename(primary_file)
                        gp_match = re.search(r"gp_(.*?)_", basename)
                        gp_value = float(gp_match.group(1)) if gp_match else float("inf")
                        return gp_value

                    # Sort frame numbers by gp value (lowest gp kept)
                    sorted_frame_nums = sorted(paired_frames.keys(),
                                                key=lambda k: extract_gp_sort_key(paired_frames[k]),
                                                reverse=True)

                    # Determine frames to delete
                    frames_to_delete = sorted_frame_nums[25:]

                    # Delete the pairs
                    for frame_num in frames_to_delete:
                        for f in paired_frames[frame_num]:
                            os.remove(f)

                elif len(paired_frames) == 25:
                    print("Exactly 25 pairs found.")
                else:
                    print("WARNING!!!!!!!!!!!! Not enough complete pairs in trial!!!!!!!!!!")

                # After deletion, verify we have exactly 25 pairs
                remaining_1 = glob.glob(os.path.join(trial_dir_name, "1_*.jpg"))
                remaining_2 = glob.glob(os.path.join(trial_dir_name, "2_*.jpg"))
                if len(remaining_1) != 25 or len(remaining_2) != 25:
                    print("ERROR: Final count mismatch! Found:")
                    print(f"- 1_*.jpg: {len(remaining_1)} files")
                    print(f"- 2_*.jpg: {len(remaining_2)} files")

                trials = glob.glob(os.path.join(os.getcwd(), 'tr_*'), recursive=True)
                for trial in trials:
                    shutil.move(trial, full_dir)

                print("Moved images to ", trial_dir_name)

                thread_rev_randmov_1 = threading.Thread(target=move_arm_thread2, args=(base1, base_cyclic1, 0, -ymov, -zmov, 0, 0, 0))
                thread_rev_randmov_2 = threading.Thread(target=move_arm_thread2, args=(base2, base_cyclic2, 0, -ymov, -zmov, 0, 0, 0))

                thread_rev_randmov_1.start(), thread_rev_randmov_2.start()
                
                thread_rev_randmov_1.join(), thread_rev_randmov_2.join()

                thread_lower_1 = threading.Thread(target=move_arm_thread1, args=(base1, base_cyclic1, 0, 0, -0.05, 0, 0, 0))
                thread_lower_2 = threading.Thread(target=move_arm_thread2, args=(base2, base_cyclic2, 0, 0, -0.05, 0, 0, 0))

                thread_lower_1.start(), thread_lower_2.start()
                thread_lower_1.join(), thread_lower_2.join()
                
            
                print("Returned arms to original position")

                time.sleep(0.5)

                subcnt += 1

            print("Done with trial")

        except Exception as e :
            print("Error! ", e)
        

if __name__ == "__main__":
    main()

# 75.877 makes jenga block fall
# 47.36 for pink ball