import sys, os, time, cv2, gsdevice, glob, shutil, re, random, threading

from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
from kortex_api.autogen.messages import Base_pb2
from kortex_api.autogen.messages import BaseCyclic_pb2
from movement_functions import *
from Gripper_Command import GripperCommand
    
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
            full_dir = cwd + "/wood_block"
            trial_cnt = get_next_trial_number(full_dir)

            ymov = random.uniform(-0.035, 0.035) # 35 mm
            zmov = random.uniform(-0.021, 0.021) # 21 mm
            print("y and z mov: ", ymov * 100, zmov * 100)
            

            for i in range(4):
                # Open grippers for a consistent start
                gripper2.Goto(0.5)
                gripper1.Goto(0.18)

                # Since we're not using impedance, add small variability to P_SLIP
                rand_slip_1 = random.uniform(0.1, 0.35)
                rand_slip_2 = random.uniform (0.1, 0.35)
                
                P_SLIP_1 = 24.35 + rand_slip_1 # Must be a value in mm (not percentage of gripper opening)
                P_SLIP_2 = 24.35 + rand_slip_2 # Must be a value in mm (not percentage of gripper opening)

                rand_goto = random.uniform(0.0,0.005)
                gripper1.Goto(0.74 + rand_goto)
                gripper2.Goto(0.85 + rand_goto)
                
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
                if (i % 2 == 0):
                    while (140 - posi2 * 140) < P_SLIP_2: # values in mm
                        if (85 - posi1 * 85) < P_SLIP_1: # values in mm
                            mm_posi2 = (140 - posi2 * 140)
                            mm_posi1 = (85 - posi1 * 85) # current opening (percentage) times max opening (85 mm)
                            capture_image(mm_posi1, counter, dev1, 1)
                            capture_image(mm_posi2, counter, dev2, 2)
                            gripper1.Goto(posi1 - .003)
                            posi1 = gripper1.GetGripperPosi()
                            
                            # print(f"gripper1 opening in mm: {85 - posi1 * 85}")
                            counter += 1
                        else:
                            # print("Moving stationary gripper")
                            gripper2.Goto(posi2 - .003)
                            posi2 = gripper2.GetGripperPosi()
                            
                            # Only close gripper 1 if the trial has not ended:
                            if (140 - posi2 * 140) < P_SLIP_2:
                                gripper1.Goto(start1)
                            
                            
                            posi1 = gripper1.GetGripperPosi()
                            # print(f"gripper2 posi in mm: {(140 - posi2 * 140)}")
                else:
                    while (85 - posi1 * 85) < P_SLIP_1: # values in mm
                        if (140 - posi2 * 140) < P_SLIP_2: # values in mm
                            mm_posi2 = (140 - posi2 * 140)
                            mm_posi1 = (85 - posi1 * 85) # current opening (percentage) times max opening (85 mm)
                            capture_image(mm_posi1, counter, dev1, 1)
                            capture_image(mm_posi2, counter, dev2, 2)
                            gripper2.Goto(posi2 - .003)
                            posi2 = gripper2.GetGripperPosi()
                            
                            # print(f"gripper1 opening in mm: {85 - posi1 * 85}")
                            counter += 1
                        else:
                            # print("Moving stationary gripper")
                            gripper1.Goto(posi1 - .003)
                            posi1 = gripper1.GetGripperPosi()
                            
                            # Only close gripper 1 if the trial has not ended:
                            if (85 - posi1 * 85) < P_SLIP_1:
                                gripper2.Goto(start2)
                            
                            
                            posi2 = gripper2.GetGripperPosi()
                            # print(f"gripper2 posi in mm: {(140 - posi2 * 140)}")
                
                print("Finished trial, moving images to folder...")
                
                trial_dir_name = f"tr_{trial_cnt}_dp_{subcnt}_wood_block_y_{ymov * 1000}_z_{zmov * 1000}_gpown_{str(P_SLIP_1)}_gpother_{str((P_SLIP_2))}"
                os.mkdir(trial_dir_name)
                images = glob.glob(os.path.join(os.getcwd(), '*.jpg'), recursive=True)
                for image in images:
                    dst_path = os.path.join(trial_dir_name, os.path.basename(image))
                    shutil.move(image, dst_path)

                jpg_files = glob.glob(os.path.join(trial_dir_name, "1_*.jpg"))

                if len(jpg_files) > 25:
                    print(f"Removed {2 * (len(jpg_files) - 25)} images")
                    # Sort by gpown_ value extracted from filename
                    def extract_sort_key(filename):
                        basename = os.path.basename(filename)
                        gp_match = re.search(r"gp_(.*?)_", basename)
                        frame_match = re.search(r"frame(\d+)\.jpg", basename)
                        gp_value = float(gp_match.group(1)) if gp_match else float("inf")
                        frame_num = frame_match.group(1) if frame_match else None
                        return (gp_value, frame_num)

                    # Sort by lowest gpown value
                    jpg_files.sort(key=extract_sort_key, reverse=True)

                    # Get the files we want to delete (keep first 25)
                    files_to_delete = jpg_files[25:]
                    
                    # Delete each file and its pair
                    for file_to_delete in files_to_delete:
                        # Extract frame number
                        frame_num = re.search(r"frame(\d+)\.jpg", os.path.basename(file_to_delete)).group(1)
                        
                        # Build pair filename pattern
                        pair_pattern = os.path.join(trial_dir_name, f"2_*frame{frame_num}.jpg")
                        paired_files = glob.glob(pair_pattern)
                        
                        # Delete both files
                        os.remove(file_to_delete)
                        if paired_files:  # Ensure pair exists before deletion
                            os.remove(paired_files[0])
                elif len(jpg_files) == 25:
                    # Verify pairs exist for all 25
                    missing_pairs = 0
                    for f in jpg_files:
                        frame_num = re.search(r"frame(\d+)\.jpg", os.path.basename(f)).group(1)
                        pair_pattern = os.path.join(trial_dir_name, f"2_*frame{frame_num}.jpg")
                        if not glob.glob(pair_pattern):
                            missing_pairs += 1
                    if missing_pairs:
                        print(f"WARNING: {missing_pairs} pairs are incomplete!")
                else:
                    print("WARNING!!!!!!!!!!!! Not enough pictures in trial!!!!!!!!!!")

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

                thread_rev_randmov_1 = threading.Thread(target=move_arm_thread1, args=(base1, base_cyclic1, 0, -ymov, -zmov, 0, 0, 0))
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