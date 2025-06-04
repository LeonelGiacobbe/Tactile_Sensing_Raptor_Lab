import sys, os, time, cv2, gsdevice, glob, shutil, re, random, threading

from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
from kortex_api.autogen.messages import Base_pb2
from kortex_api.autogen.messages import BaseCyclic_pb2
from movement_functions import *

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


def main():
    # Counter for captured frame identification
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
    try:
        
        cwd = os.getcwd()
        full_dir = cwd + "/empty"
        

        for _ in range(1):
            trial_cnt = get_next_trial_number(full_dir)
            subcnt = 1
            counter = 1
            for i in range(3):
                for j in range(25):
                    capture_image(30, counter, dev1, 1)
                    capture_image(30, counter, dev2, 2)
                    counter += 1
                print("Finished trial, moving images to folder...")
                
                trial_dir_name = f"tr_{trial_cnt}_dp_{subcnt}_empty_y_0_z_0_gpown_30_gpother_30"
                os.mkdir(trial_dir_name)
                images = glob.glob(os.path.join(os.getcwd(), '*.jpg'), recursive=True)
                for image in images:
                    dst_path = os.path.join(trial_dir_name, os.path.basename(image))
                    shutil.move(image, dst_path)

                trials = glob.glob(os.path.join(os.getcwd(), 'tr_*'), recursive=True)
                for trial in trials:
                    shutil.move(trial, full_dir)

                print("Moved images to ", trial_dir_name)

                subcnt += 1

        print("Done with trial")

    except Exception as e :
        print("Error! ", e)
        

if __name__ == "__main__":
    main()

# 75.877 makes jenga block fall
# 47.36 for pink ball